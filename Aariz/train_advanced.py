"""
Advanced Training Script for Cephalometric Landmark Detection
- HRNet-W48 backbone
- Input: 512x512, Heatmap: 128x128
- Combined Dark-Pose + Wing Loss
- Two-stage pipeline (Stage-1: HRNet-w48, Stage-2: 5 crop HRNet-w32)
- Multi-task: Landmark + CVM
- TTA for evaluation
"""
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast as amp_autocast, GradScaler
from tqdm import tqdm
from datetime import datetime
import torch.nn.functional as F

from models_advanced import HRNetW48MultiTask, HRNetW32CropRefiner
from advanced_losses import CombinedDarkWingLoss, CVMLoss
from dataset_advanced import create_advanced_dataloaders
from utils import load_checkpoint, save_checkpoint, heatmap_to_coordinates, calculate_mre, calculate_sdr


def train_stage1_epoch(model, train_loader, landmark_criterion, cvm_criterion, optimizer, 
                       device, epoch, writer=None, use_mixed_precision=False, scaler=None,
                       cvm_weight=0.3):
    """Train Stage-1 (HRNet-W48) for one epoch"""
    model.train()
    total_loss = 0.0
    total_landmark_loss = 0.0
    total_cvm_loss = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Stage-1 Train]')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        heatmaps = batch['heatmaps'].to(device, non_blocking=True)
        cvm_targets = batch['cvm'].to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if use_mixed_precision:
            with amp_autocast(device_type='cuda', dtype=torch.float16):
                heatmaps_pred, cvm_pred = model(images)
                
                # Resize heatmaps if needed
                if heatmaps_pred.shape[2:] != heatmaps.shape[2:]:
                    heatmaps_pred = F.interpolate(
                        heatmaps_pred, size=heatmaps.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                # Calculate losses
                landmark_loss = landmark_criterion(heatmaps_pred, heatmaps)
                cvm_loss = cvm_criterion(cvm_pred, cvm_targets)
                loss = landmark_loss + cvm_weight * cvm_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            heatmaps_pred, cvm_pred = model(images)
            
            if heatmaps_pred.shape[2:] != heatmaps.shape[2:]:
                heatmaps_pred = F.interpolate(
                    heatmaps_pred, size=heatmaps.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            landmark_loss = landmark_criterion(heatmaps_pred, heatmaps)
            cvm_loss = cvm_criterion(cvm_pred, cvm_targets)
            loss = landmark_loss + cvm_weight * cvm_loss
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_landmark_loss += landmark_loss.item()
        total_cvm_loss += cvm_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'lm': f'{landmark_loss.item():.4f}',
            'cvm': f'{cvm_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_landmark_loss = total_landmark_loss / len(train_loader)
    avg_cvm_loss = total_cvm_loss / len(train_loader)
    
    if writer:
        writer.add_scalar('Train/Stage1_Loss', avg_loss, epoch)
        writer.add_scalar('Train/Stage1_Landmark_Loss', avg_landmark_loss, epoch)
        writer.add_scalar('Train/Stage1_CVM_Loss', avg_cvm_loss, epoch)
    
    return avg_loss


def validate_stage1(model, val_loader, landmark_criterion, cvm_criterion, device, epoch, writer=None,
                    use_mixed_precision=False, cvm_weight=0.3):
    """Validate Stage-1"""
    model.eval()
    total_loss = 0.0
    total_landmark_loss = 0.0
    total_cvm_loss = 0.0
    
    all_predictions = []
    all_targets = []
    all_pixel_sizes = []
    all_orig_sizes = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f'Epoch {epoch} [Stage-1 Val]'):
            images = batch['image'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            landmarks = batch['landmarks'].cpu().numpy()
            cvm_targets = batch['cvm'].to(device)
            pixel_sizes = batch['pixel_size'].cpu().numpy()
            orig_sizes = batch['orig_size'].cpu().numpy()
            
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    heatmaps_pred, cvm_pred = model(images)
                    
                    if heatmaps_pred.shape[2:] != heatmaps.shape[2:]:
                        heatmaps_pred = F.interpolate(
                            heatmaps_pred, size=heatmaps.shape[2:], 
                            mode='bilinear', align_corners=False
                        )
                    
                    landmark_loss = landmark_criterion(heatmaps_pred, heatmaps)
                    cvm_loss = cvm_criterion(cvm_pred, cvm_targets)
                    loss = landmark_loss + cvm_weight * cvm_loss
            else:
                heatmaps_pred, cvm_pred = model(images)
                
                if heatmaps_pred.shape[2:] != heatmaps.shape[2:]:
                    heatmaps_pred = F.interpolate(
                        heatmaps_pred, size=heatmaps.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                
                landmark_loss = landmark_criterion(heatmaps_pred, heatmaps)
                cvm_loss = cvm_criterion(cvm_pred, cvm_targets)
                loss = landmark_loss + cvm_weight * cvm_loss
            
            total_loss += loss.item()
            total_landmark_loss += landmark_loss.item()
            total_cvm_loss += cvm_loss.item()
            
            # Convert heatmaps to coordinates
            heatmaps_np = torch.sigmoid(heatmaps_pred).cpu().numpy()
            h, w = heatmaps_np.shape[2:]
            
            for i in range(heatmaps_np.shape[0]):
                pred_coords = heatmap_to_coordinates(heatmaps_np[i], h, w)
                all_predictions.append(pred_coords)
                all_targets.append(landmarks[i])
                all_pixel_sizes.append(pixel_sizes[i])
                all_orig_sizes.append(orig_sizes[i])
    
    avg_loss = total_loss / len(val_loader)
    avg_landmark_loss = total_landmark_loss / len(val_loader)
    avg_cvm_loss = total_cvm_loss / len(val_loader)
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    
    # Scale to original image size
    mre_mm_list = []
    for i in range(len(all_predictions)):
        pixel_size = all_pixel_sizes[i]
        orig_size = all_orig_sizes[i]
        
        scale_x = orig_size[1] / w
        scale_y = orig_size[0] / h
        
        pred_scaled = all_predictions[i].copy()
        pred_scaled[:, 0] *= scale_x
        pred_scaled[:, 1] *= scale_y
        
        target_scaled = all_targets[i].copy()
        target_scaled[:, 0] *= scale_x
        target_scaled[:, 1] *= scale_y
        
        # Calculate MRE in mm
        valid_mask = (target_scaled >= 0).all(axis=1)
        if valid_mask.sum() > 0:
            errors = np.sqrt(np.sum((pred_scaled[valid_mask] - target_scaled[valid_mask])**2, axis=1))
            errors_mm = errors * pixel_size
            mre_mm_list.append(np.mean(errors_mm))
    
    mre_mm = np.mean(mre_mm_list) if mre_mm_list else float('inf')
    
    if writer:
        writer.add_scalar('Val/Stage1_Loss', avg_loss, epoch)
        writer.add_scalar('Val/Stage1_Landmark_Loss', avg_landmark_loss, epoch)
        writer.add_scalar('Val/Stage1_CVM_Loss', avg_cvm_loss, epoch)
        writer.add_scalar('Val/Stage1_MRE_mm', mre_mm, epoch)
    
    return avg_loss, mre_mm


def apply_tta(model, image, device, scales=[0.8, 1.0, 1.2], rotations=[-3, 0, 3], flips=[False, True]):
    """
    Test-Time Augmentation
    10 augmentations: flip + multi-scale + rotate
    """
    predictions = []
    
    for scale in scales:
        for rotation in rotations:
            for flip in flips:
                # Apply augmentation
                img_aug = image.clone()
                
                # Scale
                if scale != 1.0:
                    h, w = img_aug.shape[2:]
                    new_h, new_w = int(h * scale), int(w * scale)
                    img_aug = F.interpolate(img_aug, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_aug = F.interpolate(img_aug, size=(h, w), mode='bilinear', align_corners=False)
                
                # Rotate
                if rotation != 0:
                    img_aug = F.rotate(img_aug, rotation)
                
                # Flip
                if flip:
                    img_aug = torch.flip(img_aug, [3])
                
                # Predict
                with torch.no_grad():
                    heatmaps_pred, _ = model(img_aug.to(device))
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                
                # Reverse augmentation
                if flip:
                    heatmaps_pred = torch.flip(heatmaps_pred, [3])
                
                if rotation != 0:
                    heatmaps_pred = F.rotate(heatmaps_pred, -rotation)
                
                predictions.append(heatmaps_pred.cpu())
    
    # Average predictions
    predictions = torch.stack(predictions)
    prediction = predictions.mean(dim=0)
    
    return prediction


def main():
    parser = argparse.ArgumentParser(description='Advanced Training: HRNet-W48 + Multi-task + Two-stage')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size (RTX 3070 Ti: 6)')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size (H W)')
    parser.add_argument('--heatmap_size', type=int, default=128, help='Heatmap size')
    parser.add_argument('--num_workers', type=int, default=6, help='Number of workers (CPU i5 9400F: 6)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoints_advanced_rtx3070ti', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='logs_advanced_rtx3070ti', help='Log directory')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    parser.add_argument('--cvm_weight', type=float, default=0.3, help='CVM loss weight')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader, test_loader = create_advanced_dataloaders(
        dataset_folder_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        heatmap_size=args.heatmap_size,
        heatmap_sigma=3.0,
        annotation_type="Senior Orthodontists"
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")
    
    # Create Stage-1 model (HRNet-W48)
    print("Creating Stage-1 model: HRNet-W48")
    model_stage1 = HRNetW48MultiTask(
        num_landmarks=29,
        num_cvm_measurements=10,
        heatmap_size=args.heatmap_size
    ).to(device)
    
    # Loss functions
    landmark_criterion = CombinedDarkWingLoss().to(device)
    cvm_criterion = CVMLoss().to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model_stage1.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Mixed precision
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    scaler = None
    if use_mixed_precision:
        print("Using Mixed Precision Training (FP16)")
        scaler = GradScaler('cuda')
    
    # Learning rate scheduler
    from train import WarmupCosineScheduler
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=args.warmup_epochs, 
        max_epochs=args.epochs, 
        T_0=20, 
        T_mult=2, 
        eta_min=1e-6
    )
    
    # Tensorboard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, f'advanced_{timestamp}'))
    
    # Resume
    start_epoch = 0
    best_mre = float('inf')
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model_stage1, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_mre = checkpoint.get('best_mre', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_stage1_epoch(
            model_stage1, train_loader, landmark_criterion, cvm_criterion,
            optimizer, device, epoch, writer, use_mixed_precision, scaler, args.cvm_weight
        )
        
        # Validate
        val_loss, val_mre = validate_stage1(
            model_stage1, val_loader, landmark_criterion, cvm_criterion,
            device, epoch, writer, use_mixed_precision, args.cvm_weight
        )
        
        # Scheduler step
        scheduler.step(epoch)
        
        # Save checkpoint
        is_best = val_mre < best_mre
        if is_best:
            best_mre = val_mre
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_stage1.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_mre': best_mre,
            'val_mre': val_mre,
            'config': {
                'image_size': args.image_size,
                'heatmap_size': args.heatmap_size,
                'cvm_weight': args.cvm_weight
            }
        }
        
        save_checkpoint(checkpoint, is_best, args.save_dir)
        
        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val MRE={val_mre:.2f}mm, Best MRE={best_mre:.2f}mm")
    
    print("\n" + "="*80)
    print("Training Completed!")
    print(f"Best MRE: {best_mre:.2f}mm")
    print("="*80)
    
    writer.close()


if __name__ == "__main__":
    main()

