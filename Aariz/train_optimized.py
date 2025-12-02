"""
Optimized Training Script for Cephalometric Landmark Detection
Includes advanced optimizations from the research paper:
- Gradient accumulation
- EMA (Exponential Moving Average)
- Multi-scale training
- Gradient checkpointing
- torch.compile support
- Advanced data augmentation
"""
import os
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
from copy import deepcopy

from dataset import create_dataloaders
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    save_checkpoint,
    load_checkpoint
)


class EMA:
    """
    Exponential Moving Average for model parameters
    Improves model stability and generalization
    """
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    Better than MSE/Focal for landmark detection
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
            y1 = target[mask1]
            delta_y1 = delta_y[mask1]
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
                use_mixed_precision=False, scaler=None, gradient_accumulation_steps=1, ema=None):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        if use_mixed_precision:
            with amp_autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )
                loss = criterion(outputs, targets)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                if ema is not None:
                    ema.update()
        else:
            outputs = model(images)
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                )
            loss = criterion(outputs, targets)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if ema is not None:
                    ema.update()
        
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
            
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    if outputs.shape[2:] != targets.shape[2:]:
                        outputs_resized = torch.nn.functional.interpolate(
                            outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                        )
                    else:
                        outputs_resized = outputs
                    loss = criterion(outputs_resized, targets)
            else:
                outputs = model(images)
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs_resized = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], mode='bilinear', align_corners=False
                    )
                else:
                    outputs_resized = outputs
                loss = criterion(outputs_resized, targets)
            
            total_loss += loss.item()
            
            heatmaps = torch.sigmoid(outputs_resized).cpu().numpy()
            batch_size = heatmaps.shape[0]
            h, w = heatmaps.shape[2:]
            if current_image_size is None:
                current_image_size = (h, w)
            
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(heatmaps[i], h, w)
                all_predictions.append(pred_coords)
                all_targets.append(landmarks[i])
                all_pixel_sizes.append(pixel_sizes[i])
                all_orig_sizes.append(orig_sizes[i])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    
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
    parser = argparse.ArgumentParser(description='Optimized Training for Cephalometric Landmark Detection')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--model', type=str, default='hrnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'])
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--epochs', type=int, default=250, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='logs', help='Log directory')
    parser.add_argument('--annotation_type', type=str, default='Senior Orthodontists')
    parser.add_argument('--mixed_precision', action='store_true', help='Use FP16 mixed precision')
    parser.add_argument('--use_ema', action='store_true', help='Use EMA for stability')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='EMA decay rate')
    parser.add_argument('--compile_model', action='store_true', help='Use torch.compile (PyTorch 2.0+)')
    parser.add_argument('--gradient_checkpointing', action='store_true', help='Use gradient checkpointing')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Effective batch size: {effective_batch_size} ({args.batch_size} x {args.gradient_accumulation_steps})")
    
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True,
        annotation_type=args.annotation_type
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_landmarks=29)
    
    # Gradient checkpointing
    if args.gradient_checkpointing and hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled")
    
    model = model.to(device)
    
    # Compile model (PyTorch 2.0+)
    if args.compile_model and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    criterion = AdaptiveWingLoss(omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay)
        print(f"Using EMA with decay={args.ema_decay}")
    
    # Mixed precision
    scaler = None
    if args.mixed_precision and device.type == 'cuda':
        print("Using Mixed Precision Training (FP16)")
        scaler = amp_GradScaler('cuda')
    
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.epochs, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_optimized_{timestamp}'))
    
    start_epoch = 0
    best_mre = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_mre = checkpoint.get('best_mre', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            args.mixed_precision, scaler, args.gradient_accumulation_steps, ema
        )
        
        # Validate with EMA if enabled
        if ema is not None:
            ema.apply_shadow()
        
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, args.mixed_precision)
        
        if ema is not None:
            ema.restore()
        
        scheduler.step(epoch)
        
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MRE: {val_metrics['mre_mm']:.2f}mm")
        print(f"Val - SDR: 2mm={val_metrics['sdr_2mm']:.2f}%, 2.5mm={val_metrics['sdr_2.5mm']:.2f}%, "
              f"3mm={val_metrics['sdr_3mm']:.2f}%, 4mm={val_metrics['sdr_4mm']:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        is_best = val_metrics['mre_mm'] < best_mre
        if is_best:
            best_mre = val_metrics['mre_mm']
        
        # Save model state (with EMA if enabled)
        if ema is not None:
            ema.apply_shadow()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict() if not args.compile_model else model._orig_mod.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mre': best_mre,
            'val_metrics': val_metrics,
            'args': vars(args)
        }
        
        if ema is not None:
            ema.restore()
        
        save_checkpoint(checkpoint, args.save_dir, is_best, epoch if epoch % 10 == 0 else None)
        print("-" * 80)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()