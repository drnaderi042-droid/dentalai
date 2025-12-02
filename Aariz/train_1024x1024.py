"""
Training Script for Cephalometric Landmark Detection - 1024x1024
Optimized for RTX 3070 Ti with CPU RAM caching
"""
import os
# Disable albumentations update warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast as amp_autocast, GradScaler as amp_GradScaler
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from dataset import create_dataloaders, AarizDataset
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    save_checkpoint,
    load_checkpoint
)


class RAMCacheDataset:
    """
    Wrapper to cache dataset in CPU RAM
    Useful when you have large RAM (48GB) and want to speed up data loading
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}
        print(f"Caching {len(dataset)} samples in RAM...")
        for idx in tqdm(range(len(dataset)), desc="Loading to RAM"):
            self.cache[idx] = dataset[idx]
        print("RAM cache completed!")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.cache[idx]


class LimitedRAMCacheDataset:
    """
    Wrapper to cache only a limited number of samples in CPU RAM
    Uses LRU (Least Recently Used) eviction policy to manage cache size
    Useful when RAM is limited but you still want some caching benefit
    """
    def __init__(self, dataset, max_cache_size=100):
        """
        Args:
            dataset: The original dataset
            max_cache_size: Maximum number of samples to cache (default: 100)
        """
        self.dataset = dataset
        self.max_cache_size = max_cache_size
        self.cache = {}
        self.access_order = []  # Track access order for LRU eviction
        print(f"Limited RAM cache enabled: max {max_cache_size} samples")
        print(f"Dataset has {len(dataset)} samples, only frequently accessed ones will be cached")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Check if item is in cache
        if idx in self.cache:
            # Move to end (most recently used) - update access order
            if idx in self.access_order:
                self.access_order.remove(idx)
            self.access_order.append(idx)
            return self.cache[idx]
        
        # Item not in cache, load from dataset
        item = self.dataset[idx]
        
        # Only cache if max_cache_size > 0
        if self.max_cache_size > 0:
            # Add to cache if there's space
            if len(self.cache) < self.max_cache_size:
                self.cache[idx] = item
                self.access_order.append(idx)
            else:
                # Cache is full, evict least recently used
                if self.access_order:
                    lru_idx = self.access_order.pop(0)
                    if lru_idx in self.cache:
                        del self.cache[lru_idx]
                    self.cache[idx] = item
                    self.access_order.append(idx)
        
        return item


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        delta_y = (target - pred).abs()
        
        mask1 = delta_y < self.theta
        mask2 = ~mask1
        
        if mask1.sum() > 0:
            delta_y1 = delta_y[mask1]
            y1 = target[mask1]
            exp_term = torch.pow(delta_y1 / (self.epsilon + 1e-8), self.alpha - y1)
            exp_term = torch.clamp(exp_term, max=10.0)
            loss1 = self.omega * torch.log(1 + exp_term)
        else:
            loss1 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        if mask2.sum() > 0:
            delta_y2 = delta_y[mask2]
            y2 = target[mask2]
            theta_term = torch.pow(self.theta / (self.epsilon + 1e-8), self.alpha - y2)
            theta_term = torch.clamp(theta_term, max=10.0)
            A = (self.omega / (1 + theta_term + 1e-8)) * (self.alpha - y2) * \
                (torch.pow(self.theta / (self.epsilon + 1e-8), torch.clamp(self.alpha - y2 - 1, min=0.1, max=10))) / (self.epsilon + 1e-8)
            C = self.theta * A - self.omega * torch.log(1 + theta_term + 1e-8)
            loss2 = A * delta_y2 - C
        else:
            loss2 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        if isinstance(loss1, torch.Tensor) and loss1.numel() > 0:
            if isinstance(loss2, torch.Tensor) and loss2.numel() > 0:
                total_loss = (loss1.sum() + loss2.sum()) / (loss1.numel() + loss2.numel() + 1e-8)
            else:
                total_loss = loss1.mean()
        elif isinstance(loss2, torch.Tensor) and loss2.numel() > 0:
            total_loss = loss2.mean()
        else:
            total_loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        return total_loss


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing"""
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, T_0=10, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None, 
                use_mixed_precision=False, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        # Forward pass
        if use_mixed_precision:
            with amp_autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps  # Scale loss for accumulation
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item() * gradient_accumulation_steps, global_step)
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(model, val_loader, criterion, device, epoch, writer=None, use_mixed_precision=False):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    
    all_predictions = []
    all_targets = []
    all_pixel_sizes = []
    all_orig_sizes = []
    current_image_size = None
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Valid]')
        
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            landmarks = batch['landmarks'].cpu().numpy()
            pixel_sizes = batch['pixel_size'].cpu().numpy()
            orig_sizes = batch['orig_size'].cpu().numpy()
            
            if current_image_size is None:
                current_image_size = images.shape[2:]
            
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    
                    if outputs.shape[2:] != targets.shape[2:]:
                        outputs_resized = torch.nn.functional.interpolate(
                            outputs, size=targets.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    else:
                        outputs_resized = outputs
                    
                    loss = criterion(outputs_resized, targets)
            else:
                outputs = model(images)
                
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs_resized = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                else:
                    outputs_resized = outputs
                
                loss = criterion(outputs_resized, targets)
            
            total_loss += loss.item()
            
            heatmaps = torch.sigmoid(outputs_resized).cpu().numpy()
            h, w = heatmaps.shape[2:]
            
            for i in range(heatmaps.shape[0]):
                pred_coords = heatmap_to_coordinates(heatmaps[i], h, w)
                all_predictions.append(pred_coords)
                all_targets.append(landmarks[i])
                all_pixel_sizes.append(pixel_sizes[i])
                all_orig_sizes.append(orig_sizes[i])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_pixel_sizes = np.array(all_pixel_sizes)
    
    mre_pixels = calculate_mre(all_predictions, all_targets)
    
    mre_mm_list = []
    for i in range(len(all_predictions)):
        pixel_size = all_pixel_sizes[i]
        orig_size = all_orig_sizes[i]
        
        scale_x = orig_size[1] / current_image_size[1]
        scale_y = orig_size[0] / current_image_size[0]
        
        pred_scaled = all_predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = all_targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        radial_errors = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2, axis=1))
        mre_mm = np.mean(radial_errors * pixel_size)
        mre_mm_list.append(mre_mm)
    
    mre_mm = np.mean(mre_mm_list)
    
    sdr_2mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=2.0)
    sdr_2_5mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=2.5)
    sdr_3mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=3.0)
    sdr_4mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=4.0)
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MRE_pixels', mre_pixels, epoch)
        writer.add_scalar('Val/MRE_mm', mre_mm, epoch)
        writer.add_scalar('Val/SDR_2mm', sdr_2mm, epoch)
        writer.add_scalar('Val/SDR_2.5mm', sdr_2_5mm, epoch)
        writer.add_scalar('Val/SDR_3mm', sdr_3mm, epoch)
        writer.add_scalar('Val/SDR_4mm', sdr_4mm, epoch)
    
    metrics = {
        'loss': avg_loss,
        'mre_pixels': mre_pixels,
        'mre_mm': mre_mm,
        'sdr_2mm': sdr_2mm,
        'sdr_2.5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Cephalometric Landmark Detection Model - 1024x1024')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--model', type=str, default='hrnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'], 
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (reduced for 1024x1024)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=3, 
                       help='Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='adaptive_wing', choices=['adaptive_wing'],
                       help='Loss function to use')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[1024, 1024], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints_1024x1024', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_1024x1024', help='Directory for tensorboard logs')
    parser.add_argument('--annotation_type', type=str, default='Senior Orthodontists',
                       choices=['Senior Orthodontists', 'Junior Orthodontists'],
                       help='Annotation type to use')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--use_ram_cache', action='store_true',
                       help='Cache entire dataset in CPU RAM (requires large RAM, e.g., 48GB)')
    parser.add_argument('--limited_ram_cache', type=int, default=None,
                       help='Use limited RAM cache with specified max size (e.g., 100). Useful when RAM is limited. Set to 0 to disable caching.')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create datasets
    print("Loading dataset...")
    train_dataset = AarizDataset(
        dataset_folder_path=args.dataset_path,
        mode="TRAIN",
        image_size=tuple(args.image_size),
        use_heatmap=True,
        augmentation=True,
        annotation_type=args.annotation_type
    )
    
    val_dataset = AarizDataset(
        dataset_folder_path=args.dataset_path,
        mode="VALID",
        image_size=tuple(args.image_size),
        use_heatmap=True,
        augmentation=False,
        annotation_type=args.annotation_type
    )
    
    test_dataset = AarizDataset(
        dataset_folder_path=args.dataset_path,
        mode="TEST",
        image_size=tuple(args.image_size),
        use_heatmap=True,
        augmentation=False,
        annotation_type=args.annotation_type
    )
    
    # RAM cache if requested
    if args.use_ram_cache:
        print("\n" + "="*60)
        print("Full RAM Caching Enabled")
        print("="*60)
        print(f"Available RAM: 48GB")
        print(f"Caching train dataset ({len(train_dataset)} samples)...")
        train_dataset = RAMCacheDataset(train_dataset)
        print(f"Caching val dataset ({len(val_dataset)} samples)...")
        val_dataset = RAMCacheDataset(val_dataset)
        print("RAM cache completed!")
        print("="*60 + "\n")
    elif args.limited_ram_cache is not None:
        print("\n" + "="*60)
        print("Limited RAM Caching Enabled")
        print("="*60)
        print(f"Max cache size: {args.limited_ram_cache} samples")
        print(f"Train dataset: {len(train_dataset)} samples (only {args.limited_ram_cache} will be cached)")
        train_dataset = LimitedRAMCacheDataset(train_dataset, max_cache_size=args.limited_ram_cache)
        if args.limited_ram_cache > 0:
            print(f"Val dataset: {len(val_dataset)} samples (only {min(args.limited_ram_cache, len(val_dataset))} will be cached)")
            val_dataset = LimitedRAMCacheDataset(val_dataset, max_cache_size=min(args.limited_ram_cache, len(val_dataset)))
        print("="*60 + "\n")
    
    # Create dataloaders with conservative prefetch for 1024x1024
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=1,  # Conservative for 1024x1024 to avoid OOM
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=1
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=1
    )
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_landmarks=29)
    model = model.to(device)
    
    # Loss function
    criterion = AdaptiveWingLoss(omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1).to(device)
    print("Using Adaptive Wing Loss")
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Mixed precision training
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    scaler = None
    if use_mixed_precision:
        print("Using Mixed Precision Training (FP16)")
        scaler = amp_GradScaler('cuda')
    
    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        max_epochs=args.epochs
    )
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Resume from checkpoint
    start_epoch = 0
    best_mre = float('inf')
    
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_mre = checkpoint.get('best_mre', float('inf'))
            print(f"Resumed from epoch {start_epoch}, best MRE: {best_mre:.2f}mm")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size} (effective: {effective_batch_size})")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Mixed precision: {use_mixed_precision}")
    print(f"RAM cache: {args.use_ram_cache}")
    if args.limited_ram_cache is not None:
        print(f"Limited RAM cache: {args.limited_ram_cache} samples")
    print("="*80 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_mixed_precision, scaler, args.gradient_accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, use_mixed_precision)
        
        # Update learning rate
        scheduler.step(epoch)
        current_lr = scheduler.get_last_lr()[0]
        
        # Log to tensorboard
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f} (Adaptive Wing)")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MRE: {val_metrics['mre_mm']:.2f}mm")
        print(f"Val - SDR: 2mm={val_metrics['sdr_2mm']:.2f}%, "
              f"2.5mm={val_metrics['sdr_2.5mm']:.2f}%, "
              f"3mm={val_metrics['sdr_3mm']:.2f}%, "
              f"4mm={val_metrics['sdr_4mm']:.2f}%")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['mre_mm'] < best_mre
        if is_best:
            best_mre = val_metrics['mre_mm']
            print(f"*** New best MRE: {best_mre:.2f}mm ***")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mre': best_mre,
            'val_metrics': val_metrics,
            'args': vars(args)
        }
        
        save_checkpoint(checkpoint, args.save_dir, is_best=is_best, epoch=epoch)
        
        print("-" * 80)
    
    writer.close()
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best MRE: {best_mre:.2f}mm")
    print(f"Checkpoints saved in: {args.save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
