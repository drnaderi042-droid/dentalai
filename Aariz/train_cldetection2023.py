"""
Training Script for CLdetection2023 Cephalometric Landmark Detection
Based on train2.py but adapted for CLdetection2023 dataset
"""
import os
# Disable albumentations update warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
# Suppress tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
import sys

# Suppress all warnings during import
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Try to import tensorboard, but make it optional
# Suppress all output during import to avoid repeated warnings in multiprocessing
TENSORBOARD_AVAILABLE = False
_original_stderr = sys.stderr

try:
    # Redirect stderr temporarily to suppress tensorflow warnings
    with open(os.devnull, 'w') as devnull:
        sys.stderr = devnull
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from torch.utils.tensorboard import SummaryWriter
        sys.stderr = _original_stderr
    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError, Exception):
    sys.stderr = _original_stderr
    # Create a dummy SummaryWriter class
    class SummaryWriter:
        def __init__(self, *args, **kwargs):
            pass
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
        def close(self):
            pass
from torch.amp import autocast as amp_autocast, GradScaler as amp_GradScaler
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from dataset_cldetection2023 import create_dataloaders
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    save_checkpoint,
    load_checkpoint
)

# CLdetection2023 has 38 landmarks, but we'll use all of them
# No need for DIFFICULT_LANDMARKS_ONLY filtering for now
LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]
# CLdetection2023 has 38 landmarks, extend the list
for i in range(29, 38):
    LANDMARK_SYMBOLS.append(f"L{i}")


class HeatmapLoss(nn.Module):
    """
    Loss function for heatmap prediction
    Combines MSE and focal loss for better training
    """
    def __init__(self, alpha=2.0, beta=4.0, focal_weight=0.5):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.focal_weight = focal_weight
        self.mse_loss = nn.MSELoss(reduction='mean')
    
    def focal_loss(self, pred, target):
        """
        Focal loss for handling class imbalance in heatmaps
        """
        pos_mask = (target > 0).float()
        neg_mask = (target == 0).float()
        
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred + 1e-8) * pos_mask
        neg_loss = -torch.pow(pred, self.alpha) * torch.pow(1 - target, self.beta) * torch.log(1 - pred + 1e-8) * neg_mask
        
        loss = pos_loss + neg_loss
        return loss.mean()
    
    def forward(self, pred, target):
        # Normalize heatmaps
        pred = torch.sigmoid(pred)
        
        # MSE loss
        mse = self.mse_loss(pred, target)
        
        # Focal loss
        focal = self.focal_loss(pred, target)
        
        # Combined loss
        total_loss = (1 - self.focal_weight) * mse + self.focal_weight * focal
        
        return total_loss, mse, focal


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing Loss for heatmap regression
    Better than MSE for landmark detection
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted heatmaps [B, C, H, W]
            target: Target heatmaps [B, C, H, W]
        """
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)
        
        # Calculate delta
        delta = torch.abs(pred - target)
        
        # Calculate A
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - target))) * (self.alpha - target) * torch.pow(self.theta / self.epsilon, self.alpha - target - 1) / self.epsilon
        
        # Calculate C
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - target)))
        
        # Calculate loss
        loss = torch.where(
            delta < self.theta,
            self.omega * torch.log(1 + torch.pow(delta / self.epsilon, self.alpha - target)),
            A * delta - C
        )
        
        return loss.mean()


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing with restarts
    """
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
            # Linear warmup
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            # Cosine annealing with restarts
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None, use_adaptive_wing=False, use_mixed_precision=False, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch with gradient accumulation"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    # فقط در ابتدای accumulation، gradient را صفر می‌کنیم
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        if use_mixed_precision:
            # Mixed precision training (FP16)
            with amp_autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                
                # Resize outputs to match target size if needed
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # Calculate loss
                if use_adaptive_wing:
                    loss = criterion(outputs, targets)
                    mse = torch.tensor(0.0)
                    focal = torch.tensor(0.0)
                else:
                    loss, mse, focal = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
            
            # Backward pass with scaler (gradient accumulation)
            scaler.scale(loss).backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Standard precision training
            outputs = model(images)
            
            # Resize outputs to match target size if needed
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Calculate loss
            if use_adaptive_wing:
                loss = criterion(outputs, targets)
                mse = torch.tensor(0.0)
                focal = torch.tensor(0.0)
            else:
                loss, mse, focal = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Accumulate losses
        total_loss += loss.item() * gradient_accumulation_steps
        if isinstance(mse, torch.Tensor):
            total_mse += mse.item()
        if isinstance(focal, torch.Tensor):
            total_focal += focal.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
        })
    
    # Average losses
    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader) if total_mse > 0 else 0.0
    avg_focal = total_focal / len(train_loader) if total_focal > 0 else 0.0
    
    return avg_loss, avg_mse, avg_focal


def validate(model, val_loader, criterion, device, epoch, writer=None, use_adaptive_wing=False, use_mixed_precision=False):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    
                    # Resize outputs to match target size if needed
                    if outputs.shape[2:] != targets.shape[2:]:
                        outputs = torch.nn.functional.interpolate(
                            outputs, size=targets.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    
                    if use_adaptive_wing:
                        loss = criterion(outputs, targets)
                    else:
                        loss, _, _ = criterion(outputs, targets)
            else:
                outputs = model(images)
                
                # Resize outputs to match target size if needed
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                if use_adaptive_wing:
                    loss = criterion(outputs, targets)
                else:
                    loss, _, _ = criterion(outputs, targets)
            
            total_loss += loss.item()
            
            # Convert to coordinates for metrics
            batch_size = outputs.shape[0]
            height, width = outputs.shape[2:]
            
            for b in range(batch_size):
                pred_heatmaps = outputs[b].cpu().numpy()
                target_heatmaps = targets[b].cpu().numpy()
                
                pred_coords = heatmap_to_coordinates(pred_heatmaps, height, width)
                target_coords = heatmap_to_coordinates(target_heatmaps, height, width)
                
                all_predictions.append(pred_coords)
                all_targets.append(target_coords)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    mre_mm = calculate_mre(all_predictions, all_targets)
    sdr_2mm = calculate_sdr(all_predictions, all_targets, threshold=2.0)
    sdr_2_5mm = calculate_sdr(all_predictions, all_targets, threshold=2.5)
    sdr_3mm = calculate_sdr(all_predictions, all_targets, threshold=3.0)
    sdr_4mm = calculate_sdr(all_predictions, all_targets, threshold=4.0)
    
    avg_loss = total_loss / len(val_loader)
    
    metrics = {
        'loss': avg_loss,
        'mre_mm': mre_mm,
        'sdr_2mm': sdr_2mm,
        'sdr_2.5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm
    }
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MRE', mre_mm, epoch)
        writer.add_scalar('Val/SDR_2mm', sdr_2mm, epoch)
        writer.add_scalar('Val/SDR_2.5mm', sdr_2_5mm, epoch)
        writer.add_scalar('Val/SDR_3mm', sdr_3mm, epoch)
        writer.add_scalar('Val/SDR_4mm', sdr_4mm, epoch)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train CLdetection2023 Cephalometric Landmark Detection Model')
    parser.add_argument('--dataset_path', type=str, default='CLdetection2023', help='Path to CLdetection2023 dataset folder')
    parser.add_argument('--model', type=str, default='hrnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'], 
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--loss', type=str, default='adaptive_wing', choices=['adaptive_wing', 'heatmap'],
                       help='Loss function to use')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[768, 768], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints_cldetection2023', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_cldetection2023', help='Directory for tensorboard logs')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16) for faster training')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                       help='Gradient accumulation steps (effective batch size = batch_size * gradient_accumulation_steps)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    prefetch_factor = min(4, max(2, args.num_workers // 2))
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True,
        prefetch_factor=prefetch_factor
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create model - CLdetection2023 has 38 landmarks
    print(f"Creating model: {args.model} with 38 landmarks")
    model = get_model(args.model, num_landmarks=38)
    model = model.to(device)
    
    # Loss function
    use_adaptive_wing = (args.loss == 'adaptive_wing')
    if use_adaptive_wing:
        print("Using Adaptive Wing Loss")
        criterion = AdaptiveWingLoss(omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1).to(device)
    else:
        print("Using Heatmap Loss (MSE + Focal)")
        criterion = HeatmapLoss(alpha=2.0, beta=4.0, focal_weight=0.5).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Mixed precision training
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    scaler = None
    if use_mixed_precision:
        print("Using Mixed Precision Training (FP16)")
        scaler = amp_GradScaler('cuda')
    
    # Learning rate scheduler
    print(f"Using WarmupCosineScheduler (warmup={args.warmup_epochs} epochs)")
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        max_epochs=args.epochs, 
        T_0=10, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Tensorboard writer
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_{timestamp}'))
        print("TensorBoard logging enabled.")
    else:
        writer = None
        # Print warning only once in main process
        print("Note: TensorBoard logging is disabled. Training will continue normally.")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mre = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_mre = checkpoint.get('best_mre', float('inf'))
        
        # Update LR if new LR is specified
        checkpoint_lr = optimizer.param_groups[0]['lr']
        if abs(checkpoint_lr - args.lr) > 1e-8:
            print(f"Updating LR from {checkpoint_lr:.6f} to {args.lr:.6f}")
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            scheduler = WarmupCosineScheduler(
                optimizer, 
                warmup_epochs=args.warmup_epochs, 
                max_epochs=args.epochs, 
                T_0=10, 
                T_mult=2, 
                eta_min=1e-6
            )
        
        # Set scheduler to correct epoch
        if start_epoch > 0:
            scheduler.current_epoch = start_epoch
            if start_epoch >= args.warmup_epochs:
                cosine_start_epoch = start_epoch - args.warmup_epochs
                for _ in range(cosine_start_epoch):
                    scheduler.cosine_scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = args.lr * (start_epoch + 1) / args.warmup_epochs
                for pg in optimizer.param_groups:
                    pg['lr'] = current_lr
        
        print(f"Resumed from epoch {start_epoch}")
        print(f"Current LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_mse, train_focal = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer, 
            use_adaptive_wing, use_mixed_precision, scaler, args.gradient_accumulation_steps
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, 
                              use_adaptive_wing, use_mixed_precision)
        
        # Update learning rate
        scheduler.step(epoch)
        
        # Log to tensorboard
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        if not use_adaptive_wing:
            writer.add_scalar('Train/EpochMSE', train_mse, epoch)
            writer.add_scalar('Train/EpochFocal', train_focal, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        if use_adaptive_wing:
            print(f"Train - Loss: {train_loss:.4f} (Adaptive Wing)")
        else:
            print(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, Focal: {train_focal:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, MRE: {val_metrics['mre_mm']:.2f}mm")
        print(f"Val - SDR: 2mm={val_metrics['sdr_2mm']:.2f}%, 2.5mm={val_metrics['sdr_2.5mm']:.2f}%, "
              f"3mm={val_metrics['sdr_3mm']:.2f}%, 4mm={val_metrics['sdr_4mm']:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['mre_mm'] < best_mre
        if is_best:
            best_mre = val_metrics['mre_mm']
            print(f"Saved best model (MRE: {best_mre:.2f}mm)")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mre': best_mre,
            'val_metrics': val_metrics,
            'args': vars(args)
        }
        
        save_checkpoint(checkpoint, args.save_dir, is_best, epoch)
        
        print("-" * 80)
    
    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()

