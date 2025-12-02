"""
Training Script for Cephalometric Landmark Detection
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

from dataset import create_dataloaders
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    save_checkpoint,
    load_checkpoint
)

# فقط لندمارک‌های مشکل‌دار برای training (بقیه ignore می‌شوند)
DIFFICULT_LANDMARKS_ONLY = {
    'UMT': True,   # Upper Molar Tip - بیشترین خطا (3.805 mm)
    'UPM': True,   # Upper Premolar (3.486 mm)
    'R': True,     # Ramus point (3.331 mm)
    'Ar': True,    # Articulare (2.645 mm)
    'Go': True,    # Gonion (2.618 mm)
    'LMT': True,   # Lower Molar Tip (2.545 mm)
    'LPM': True,   # Lower Premolar
    'Or': True,    # Orbitale (2.326 mm)
    'Co': True,    # Condylion (2.200 mm)
    'PNS': True,   # Posterior Nasal Spine (2.155 mm)
    'Po': True,    # Porion (2.042 mm)
    'ANS': True,   # Anterior Nasal Spine (1.822 mm)
    # بقیه لندمارک‌ها: False = ignore می‌شوند
}

LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

def calculate_difficult_landmarks_only_loss(outputs, targets, landmark_symbols, base_criterion, device):
    """
    محاسبه loss فقط برای لندمارک‌های مشکل‌دار
    لندمارک‌های سالم ignore می‌شوند (در loss محاسبه نمی‌شوند)
    
    Args:
        outputs: خروجی مدل [B, C, H, W]
        targets: targets [B, C, H, W]
        landmark_symbols: لیست نام لندمارک‌ها
        base_criterion: loss function پایه (AdaptiveWingLoss)
        device: device
    
    Returns:
        loss: loss فقط برای لندمارک‌های مشکل‌دار
    """
    batch_size = outputs.shape[0]
    num_landmarks = outputs.shape[1]
    
    total_loss = 0.0
    count = 0
    
    for i in range(num_landmarks):
        if i >= len(landmark_symbols):
            # اگر نام لندمارک مشخص نبود، skip می‌کنیم
            continue
        
        landmark_name = landmark_symbols[i]
        
        # فقط لندمارک‌های مشکل‌دار را محاسبه کن
        if not DIFFICULT_LANDMARKS_ONLY.get(landmark_name, False):
            continue  # لندمارک‌های سالم را ignore می‌کنیم
        
        # استخراج heatmap این لندمارک
        landmark_output = outputs[:, i:i+1, :, :]
        landmark_target = targets[:, i:i+1, :, :]
        
        # محاسبه loss برای این لندمارک
        landmark_loss = base_criterion(landmark_output, landmark_target)
        
        total_loss += landmark_loss
        count += 1
    
    # اگر هیچ لندمارک مشکل‌داری نبود، loss صفر برمی‌گردانیم
    if count == 0:
        # Fallback: حداقل یک loss برمی‌گردانیم (برای جلوگیری از error)
        # اما warning می‌دهیم
        print("[WARNING] No difficult landmarks found! Using all landmarks as fallback.")
        # Fallback به محاسبه loss برای همه
        return base_criterion(outputs, targets)
    
    # میانگین loss فقط برای لندمارک‌های مشکل‌دار
    average_loss = total_loss / count
    
    return average_loss


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
    Better than MSE/Focal for landmark detection
    Paper: Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression
    """
    def __init__(self, omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
    
    def forward(self, pred, target):
        # Normalize predictions
        pred = torch.sigmoid(pred)
        
        delta_y = (target - pred).abs()
        
        # Split into two regions
        mask1 = delta_y < self.theta
        mask2 = ~mask1
        
        # Region 1: small errors
        if mask1.sum() > 0:
            y1 = target[mask1]
            delta_y1 = delta_y[mask1]
            
            # Avoid numerical issues
            exp_term = torch.pow(delta_y1 / (self.epsilon + 1e-8), self.alpha - y1)
            exp_term = torch.clamp(exp_term, max=10.0)  # Prevent overflow
            
            loss1 = self.omega * torch.log(1 + exp_term)
        else:
            loss1 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Region 2: large errors
        if mask2.sum() > 0:
            delta_y2 = delta_y[mask2]
            y2 = target[mask2]
            
            # Compute A and C more safely
            theta_term = torch.pow(self.theta / (self.epsilon + 1e-8), self.alpha - y2)
            theta_term = torch.clamp(theta_term, max=10.0)
            
            A = (self.omega / (1 + theta_term + 1e-8)) * (self.alpha - y2) * \
                (torch.pow(self.theta / (self.epsilon + 1e-8), torch.clamp(self.alpha - y2 - 1, min=0.1, max=10))) / (self.epsilon + 1e-8)
            
            C = self.theta * A - self.omega * torch.log(1 + theta_term + 1e-8)
            
            loss2 = A * delta_y2 - C
        else:
            loss2 = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        
        # Combine losses
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
                    # فقط لندمارک‌های مشکل‌دار آموزش می‌بینند (بقیه ignore می‌شوند)
                    loss = calculate_difficult_landmarks_only_loss(
                        outputs, targets,
                        LANDMARK_SYMBOLS,
                        criterion,
                        device
                    )
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
            # Standard training (FP32)
            outputs = model(images)
            
            # Resize outputs to match target size if needed
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # Calculate loss
            if use_adaptive_wing:
                # فقط لندمارک‌های مشکل‌دار آموزش می‌بینند (بقیه ignore می‌شوند)
                loss = calculate_difficult_landmarks_only_loss(
                    outputs, targets,
                    LANDMARK_SYMBOLS,
                    criterion,
                    device
                )
                mse = torch.tensor(0.0)
                focal = torch.tensor(0.0)
            else:
                loss, mse, focal = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps
            
            # Backward pass (gradient accumulation)
            loss.backward()
            
            # Update weights only after accumulating gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Accumulate losses (scale back for logging)
        total_loss += loss.item() * gradient_accumulation_steps
        if isinstance(mse, torch.Tensor):
            total_mse += mse.item() if mse.numel() > 0 else 0.0
        else:
            total_mse += mse
        if isinstance(focal, torch.Tensor):
            total_focal += focal.item() if focal.numel() > 0 else 0.0
        else:
            total_focal += focal
        
        # Update progress bar
        if use_adaptive_wing:
            pbar.set_postfix({'loss': f'{loss.item() * gradient_accumulation_steps:.4f}'})
        else:
            pbar.set_postfix({
                'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                'mse': f'{mse.item() if isinstance(mse, torch.Tensor) else mse:.4f}',
                'focal': f'{focal.item() if isinstance(focal, torch.Tensor) else focal:.4f}'
            })
        
        # Log to tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item() * gradient_accumulation_steps, global_step)
    
    # Handle remaining gradients if batch size doesn't divide evenly
    # Note: با drop_last=True در DataLoader، این مشکل پیش نمی‌آید
    # اما برای اطمینان، چک می‌کنیم
    remaining = len(train_loader) % gradient_accumulation_steps
    if remaining != 0 and len(train_loader) > 0:
        # Update optimizer with remaining accumulated gradients
        if use_mixed_precision and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif not use_mixed_precision:
            optimizer.step()
            optimizer.zero_grad()
    
    # Calculate average loss (adjusted for gradient accumulation)
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / len(train_loader)
    avg_focal = total_focal / len(train_loader)
    
    return avg_loss, avg_mse, avg_focal


def validate(model, val_loader, criterion, device, epoch, writer=None, use_adaptive_wing=False, use_mixed_precision=False):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    
    all_predictions = []
    all_targets = []
    all_landmarks = []
    all_pixel_sizes = []
    all_orig_sizes = []
    current_image_size = None  # Will be set from first batch
    
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
            
            # Forward pass
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    
                    # Resize outputs to match target size if needed
                    if outputs.shape[2:] != targets.shape[2:]:
                        outputs_resized = torch.nn.functional.interpolate(
                            outputs, size=targets.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    else:
                        outputs_resized = outputs
                    
                    # Calculate loss
                    if use_adaptive_wing:
                        loss = criterion(outputs_resized, targets)
                        mse = torch.tensor(0.0)
                        focal = torch.tensor(0.0)
                    else:
                        loss, mse, focal = criterion(outputs_resized, targets)
            else:
                outputs = model(images)
                
                # Resize outputs to match target size if needed
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs_resized = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                else:
                    outputs_resized = outputs
                
                # Calculate loss
                if use_adaptive_wing:
                    loss = criterion(outputs_resized, targets)
                    mse = torch.tensor(0.0)
                    focal = torch.tensor(0.0)
                else:
                    loss, mse, focal = criterion(outputs_resized, targets)
            
            total_loss += loss.item()
            if isinstance(mse, torch.Tensor):
                total_mse += mse.item() if mse.numel() > 0 else 0.0
            else:
                total_mse += mse
            if isinstance(focal, torch.Tensor):
                total_focal += focal.item() if focal.numel() > 0 else 0.0
            else:
                total_focal += focal
            
            # Convert heatmaps to coordinates for evaluation
            heatmaps = torch.sigmoid(outputs_resized).cpu().numpy()
            batch_size = heatmaps.shape[0]
            h, w = heatmaps.shape[2:]
            
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(heatmaps[i], h, w)
                all_predictions.append(pred_coords)
                all_targets.append(landmarks[i])
                all_landmarks.append(landmarks[i])
                all_pixel_sizes.append(pixel_sizes[i])
                all_orig_sizes.append(orig_sizes[i])
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(val_loader)
    avg_mse = total_mse / len(val_loader)
    avg_focal = total_focal / len(val_loader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_pixel_sizes = np.array(all_pixel_sizes)
    
    # Calculate MRE in pixels
    mre_pixels = calculate_mre(all_predictions, all_targets)
    
    # Calculate MRE in millimeters
    # Scale predictions and targets back to original size
    mre_mm_list = []
    for i in range(len(all_predictions)):
        pixel_size = all_pixel_sizes[i]
        orig_size = all_orig_sizes[i]
        
        # Scale predictions and targets to original image size
        scale_x = orig_size[1] / current_image_size[1]  # orig_width / current_width
        scale_y = orig_size[0] / current_image_size[0]  # orig_height / current_height
        
        pred_scaled = all_predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = all_targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        # Calculate radial error in millimeters
        radial_errors = np.sqrt(np.sum((pred_scaled - target_scaled) ** 2, axis=1))
        mre_mm = np.mean(radial_errors * pixel_size)
        mre_mm_list.append(mre_mm)
    
    mre_mm = np.mean(mre_mm_list)
    
    # Calculate SDR at different thresholds
    sdr_2mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=2.0)
    sdr_2_5mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=2.5)
    sdr_3mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=3.0)
    sdr_4mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, all_orig_sizes, current_image_size, threshold_mm=4.0)
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MSE', avg_mse, epoch)
        writer.add_scalar('Val/Focal', avg_focal, epoch)
        writer.add_scalar('Val/MRE_pixels', mre_pixels, epoch)
        writer.add_scalar('Val/MRE_mm', mre_mm, epoch)
        writer.add_scalar('Val/SDR_2mm', sdr_2mm, epoch)
        writer.add_scalar('Val/SDR_2.5mm', sdr_2_5mm, epoch)
        writer.add_scalar('Val/SDR_3mm', sdr_3mm, epoch)
        writer.add_scalar('Val/SDR_4mm', sdr_4mm, epoch)
    
    metrics = {
        'loss': avg_loss,
        'mse': avg_mse,
        'focal': avg_focal,
        'mre_pixels': mre_pixels,
        'mre_mm': mre_mm,
        'sdr_2mm': sdr_2mm,
        'sdr_2.5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Cephalometric Landmark Detection Model')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'unet', 'hourglass', 'hrnet'], 
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--loss', type=str, default='adaptive_wing', choices=['adaptive_wing', 'heatmap'],
                       help='Loss function to use')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory for tensorboard logs')
    parser.add_argument('--annotation_type', type=str, default='Senior Orthodontists',
                       choices=['Senior Orthodontists', 'Junior Orthodontists'],
                       help='Annotation type to use')
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
    
    # Create dataloaders with increased prefetch for better GPU utilization
    print("Loading dataset...")
    # افزایش prefetch_factor برای استفاده بیشتر از GPU memory و buffering بیشتر
    prefetch_factor = min(4, max(2, args.num_workers // 2))  # بیشترین 4 برای prefetch
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True,
        annotation_type=args.annotation_type,
        prefetch_factor=prefetch_factor  # افزایش prefetch برای buffering بیشتر
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Effective batch size با gradient accumulation
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    print(f"Batch size: {args.batch_size}, Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {effective_batch_size}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_landmarks=29)
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_{timestamp}'))
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mre = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_mre = checkpoint.get('best_mre', float('inf'))
        
        # Update LR if new LR is specified (different from checkpoint)
        checkpoint_lr = optimizer.param_groups[0]['lr']
        if abs(checkpoint_lr - args.lr) > 1e-8:
            print(f"Updating LR from {checkpoint_lr:.6f} to {args.lr:.6f}")
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr
            # Recreate scheduler with new LR and settings
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
            # If we're past warmup, adjust cosine scheduler
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

