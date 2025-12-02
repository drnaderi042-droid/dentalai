"""
Fine-tuning Script for Cephalometric Landmark Detection
Fine-tunes a model trained on Aariz dataset (29 landmarks) with combined datasets
"""
import os
# Disable albumentations update warnings
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from dataset_combined import create_combined_dataloaders
from model import get_model
from utils import (
    heatmap_to_coordinates,
    calculate_mre,
    calculate_sdr,
    save_checkpoint,
    load_checkpoint
)
from train import AdaptiveWingLoss, HeatmapLoss, WarmupCosineScheduler


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer,
                use_adaptive_wing, use_mixed_precision, scaler, use_channels_last=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        # Use non_blocking transfer for better GPU utilization
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        # Convert to channels_last if enabled
        if use_channels_last:
            images = images.to(memory_format=torch.channels_last)
        
        optimizer.zero_grad()
        
        if use_mixed_precision:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                if use_adaptive_wing:
                    loss = criterion(outputs, targets)
                    mse = None
                    focal = None
                else:
                    loss, mse, focal = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            if use_adaptive_wing:
                loss = criterion(outputs, targets)
                mse = None
                focal = None
            else:
                loss, mse, focal = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        if mse is not None:
            total_mse += mse.item()
        if focal is not None:
            total_focal += focal.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{total_loss/num_batches:.4f}'
        })
        
        # Log to tensorboard
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            if mse is not None:
                writer.add_scalar('Train/BatchMSE', mse.item(), global_step)
            if focal is not None:
                writer.add_scalar('Train/BatchFocal', focal.item(), global_step)
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches if total_mse > 0 else None
    avg_focal = total_focal / num_batches if total_focal > 0 else None
    
    return avg_loss, avg_mse, avg_focal


def validate(model, val_loader, criterion, device, epoch, writer, 
            use_adaptive_wing, use_mixed_precision):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    num_batches = 0
    
    all_predictions = []
    all_targets = []
    all_pixel_sizes = []
    all_orig_sizes = []
    
    current_image_size = None
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch in pbar:
            # Use non_blocking transfer for better GPU utilization
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            landmarks = batch['landmarks'].cpu().numpy()
            pixel_sizes = batch['pixel_size'].cpu().numpy()
            orig_sizes = batch['orig_size'].cpu().numpy()
            
            if current_image_size is None:
                current_image_size = (images.shape[2], images.shape[3])
            
            if use_mixed_precision:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    if use_adaptive_wing:
                        loss = criterion(outputs, targets)
                        mse = None
                        focal = None
                    else:
                        loss, mse, focal = criterion(outputs, targets)
            else:
                outputs = model(images)
                if use_adaptive_wing:
                    loss = criterion(outputs, targets)
                    mse = None
                    focal = None
                else:
                    loss, mse, focal = criterion(outputs, targets)
            
            total_loss += loss.item()
            if mse is not None:
                total_mse += mse.item()
            if focal is not None:
                total_focal += focal.item()
            num_batches += 1
            
            # Convert heatmaps to coordinates
            outputs_np = outputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            
            batch_predictions = []
            batch_targets = []
            
            for i in range(outputs_np.shape[0]):
                pred_coords = heatmap_to_coordinates(
                    outputs_np[i], 
                    current_image_size[0], 
                    current_image_size[1]
                )
                target_coords = heatmap_to_coordinates(
                    targets_np[i],
                    current_image_size[0],
                    current_image_size[1]
                )
                
                batch_predictions.append(pred_coords)
                batch_targets.append(target_coords)
            
            all_predictions.extend(batch_predictions)
            all_targets.extend(batch_targets)
            all_pixel_sizes.extend(pixel_sizes)
            all_orig_sizes.extend(orig_sizes)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg': f'{total_loss/num_batches:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_mse = total_mse / num_batches if total_mse > 0 else None
    avg_focal = total_focal / num_batches if total_focal > 0 else None
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_pixel_sizes = np.array(all_pixel_sizes)
    all_orig_sizes = np.array(all_orig_sizes)
    
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
        valid_mask = (all_predictions[i] >= 0).all(axis=1) & (all_targets[i] >= 0).all(axis=1)
        valid_errors = radial_errors[valid_mask]
        
        if len(valid_errors) > 0:
            mre_mm = np.mean(valid_errors * pixel_size)
            mre_mm_list.append(mre_mm)
    
    mre_mm = np.mean(mre_mm_list) if len(mre_mm_list) > 0 else 0.0
    
    sdr_2mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, 
                             all_orig_sizes, current_image_size, threshold_mm=2.0)
    sdr_2_5mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, 
                              all_orig_sizes, current_image_size, threshold_mm=2.5)
    sdr_3mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, 
                            all_orig_sizes, current_image_size, threshold_mm=3.0)
    sdr_4mm = calculate_sdr(all_predictions, all_targets, all_pixel_sizes, 
                            all_orig_sizes, current_image_size, threshold_mm=4.0)
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        if avg_mse is not None:
            writer.add_scalar('Val/MSE', avg_mse, epoch)
        if avg_focal is not None:
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
    parser = argparse.ArgumentParser(description='Fine-tune Cephalometric Landmark Detection Model')
    parser.add_argument('--aariz_path', type=str, default='Aariz', 
                       help='Path to Aariz dataset')
    parser.add_argument('--dataset_3471833_path', type=str, default='../3471833',
                       help='Path to 3471833 dataset')
    parser.add_argument('--model', type=str, default='hrnet', 
                       choices=['resnet', 'unet', 'hourglass', 'hrnet'],
                       help='Model architecture')
    parser.add_argument('--resume', type=str, required=True,
                       help='Path to checkpoint to resume from (Aariz trained model)')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5, 
                       help='Learning rate (lower for fine-tuning)')
    parser.add_argument('--loss', type=str, default='adaptive_wing', 
                       choices=['adaptive_wing', 'heatmap'],
                       help='Loss function to use')
    parser.add_argument('--warmup_epochs', type=int, default=5, 
                       help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], 
                       help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=8, 
                       help='Number of data loading workers (increase for better GPU utilization)')
    parser.add_argument('--prefetch_factor', type=int, default=4,
                       help='Number of batches to prefetch (increase for better GPU utilization)')
    parser.add_argument('--device', type=str, default='cuda', 
                       help='Device (cuda/cpu)')
    parser.add_argument('--save_dir', type=str, default='checkpoints_finetuned',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs_finetuned',
                       help='Directory for tensorboard logs')
    parser.add_argument('--aariz_annotation_type', type=str, 
                       default='Senior Orthodontists',
                       choices=['Senior Orthodontists', 'Junior Orthodontists'],
                       help='Aariz annotation type to use')
    parser.add_argument('--dataset_3471833_annotation_type', type=str, 
                       default='400_senior',
                       choices=['400_senior', '400_junior'],
                       help='3471833 annotation type to use')
    parser.add_argument('--use_aariz_only', action='store_true',
                       help='Only use Aariz dataset (no 3471833)')
    parser.add_argument('--use_3471833_only', action='store_true',
                       help='Only use 3471833 dataset (no Aariz)')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--freeze_backbone', action='store_true',
                       help='Freeze backbone weights (only train head)')
    parser.add_argument('--use_compile', action='store_true',
                       help='Use torch.compile for faster training (PyTorch 2.0+)')
    parser.add_argument('--channels_last', action='store_true',
                       help='Use channels_last memory format for better GPU utilization')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders with optimized settings
    print("Loading combined datasets...")
    print(f"DataLoader optimization settings:")
    print(f"  num_workers: {args.num_workers}")
    print(f"  prefetch_factor: {args.prefetch_factor}")
    print(f"  pin_memory: True")
    
    train_loader, val_loader, test_loader = create_combined_dataloaders(
        aariz_path=args.aariz_path,
        dataset_3471833_path=args.dataset_3471833_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size),
        use_heatmap=True,
        aariz_annotation_type=args.aariz_annotation_type,
        dataset_3471833_annotation_type=args.dataset_3471833_annotation_type,
        use_aariz_only=args.use_aariz_only,
        use_3471833_only=args.use_3471833_only,
        prefetch_factor=args.prefetch_factor
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_model(args.model, num_landmarks=29)
    model = model.to(device)
    
    # Use channels_last memory format for better GPU utilization
    if args.channels_last:
        print("Using channels_last memory format for better GPU utilization")
        model = model.to(memory_format=torch.channels_last)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.resume}")
    checkpoint = load_checkpoint(args.resume, model, None)
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.use_compile:
        print("Compiling model with torch.compile (this may take a moment on first run)...")
        print("Note: If Triton is not installed, compilation will automatically fall back to eager mode")
        try:
            # Suppress errors and fall back to eager if compilation fails
            import torch._dynamo as dynamo_module
            dynamo_module.config.suppress_errors = True
            
            # Try with reduce-overhead mode first
            model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
        except Exception as e:
            print(f"Warning: torch.compile failed: {e}")
            print("Continuing without compilation (performance may be slightly slower)...")
            # Model remains uncompiled
    
    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone weights...")
        for name, param in model.named_parameters():
            if 'head' not in name.lower() and 'final' not in name.lower():
                param.requires_grad = False
        print("Backbone frozen. Only training head.")
    
    # Loss function
    use_adaptive_wing = (args.loss == 'adaptive_wing')
    if use_adaptive_wing:
        print("Using Adaptive Wing Loss")
        criterion = AdaptiveWingLoss(omega=14.0, theta=0.5, epsilon=2.0, alpha=2.1).to(device)
    else:
        print("Using Heatmap Loss (MSE + Focal)")
        criterion = HeatmapLoss(alpha=2.0, beta=4.0, focal_weight=0.5).to(device)
    
    # Optimizer (only train unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    
    # Mixed precision training
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    scaler = None
    if use_mixed_precision:
        print("Using Mixed Precision Training (FP16)")
        scaler = torch.amp.GradScaler('cuda')
    
    # Learning rate scheduler
    print(f"Using WarmupCosineScheduler (warmup={args.warmup_epochs} epochs)")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        T_0=10,
        T_mult=2,
        eta_min=1e-7
    )
    
    # Tensorboard writer
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_finetuned_{timestamp}'))
    
    # Training loop
    start_epoch = 0
    best_mre = checkpoint.get('best_mre', float('inf'))
    
    print("Starting fine-tuning...")
    print(f"Initial best MRE: {best_mre:.2f}mm")
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_mse, train_focal = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_adaptive_wing, use_mixed_precision, scaler, args.channels_last
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer,
                              use_adaptive_wing, use_mixed_precision)
        
        # Update learning rate
        scheduler.step(epoch)
        
        # Log to tensorboard
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        if train_mse is not None:
            writer.add_scalar('Train/EpochMSE', train_mse, epoch)
        if train_focal is not None:
            writer.add_scalar('Train/EpochFocal', train_focal, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        current_lr = scheduler.get_last_lr()[0]
        print(f"\nEpoch {epoch}/{args.epochs}")
        if use_adaptive_wing:
            print(f"Train - Loss: {train_loss:.4f} (Adaptive Wing)")
        else:
            print(f"Train - Loss: {train_loss:.4f}, MSE: {train_mse:.4f}, "
                  f"Focal: {train_focal:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, "
              f"MRE: {val_metrics['mre_mm']:.2f}mm")
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
            'args': vars(args),
            'source_checkpoint': args.resume
        }
        
        save_checkpoint(checkpoint, args.save_dir, is_best, epoch)
        
        print("-" * 80)
    
    writer.close()
    print("Fine-tuning completed!")
    print(f"Best MRE achieved: {best_mre:.2f}mm")


if __name__ == "__main__":
    main()

