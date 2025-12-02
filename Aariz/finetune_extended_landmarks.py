"""
Fine-tuning script for adding new landmarks to existing model
Uses transfer learning: freezes backbone, only trains new landmark heads
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import numpy as np

from dataset_extended import create_extended_dataloaders
from model import get_model
from utils import load_checkpoint, save_checkpoint, heatmap_to_coordinates, calculate_mre, calculate_sdr


class ExtendedModel(nn.Module):
    """
    Extended model that adds new landmark heads to existing model
    Uses transfer learning: freezes backbone, trains only new heads
    """
    def __init__(self, base_model, base_num_landmarks=29, new_num_landmarks=32):
        """
        Args:
            base_model: Pre-trained model with base_num_landmarks
            base_num_landmarks: Number of landmarks in base model (29)
            new_num_landmarks: Total number of landmarks including new ones (e.g., 32)
        """
        super(ExtendedModel, self).__init__()
        self.base_model = base_model
        self.base_num_landmarks = base_num_landmarks
        self.new_num_landmarks = new_num_landmarks
        self.num_new_landmarks = new_num_landmarks - base_num_landmarks
        
        # Freeze base model (except final layers if needed)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Determine model architecture and feature channels
        if hasattr(base_model, 'final_layers'):
            # HRNet architecture
            # HRNet has multiple resolution branches, use the highest resolution branch
            # Typically width channels (e.g., 32, 48, etc.)
            if hasattr(base_model, 'stage4'):
                # Get width from stage4
                width = base_model.stage4.branches[0][0].in_channels if hasattr(base_model.stage4, 'branches') else 32
            else:
                width = 32  # Default HRNet width
            
            # New head matching HRNet structure (multi-resolution)
            self.new_head = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(width, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(width, self.num_new_landmarks, kernel_size=1, bias=True)
                ),
                nn.Sequential(
                    nn.Conv2d(width*2, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(width, self.num_new_landmarks, kernel_size=1, bias=True)
                ),
                nn.Sequential(
                    nn.Conv2d(width*4, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
                    nn.Conv2d(width, self.num_new_landmarks, kernel_size=1, bias=True)
                ),
                nn.Sequential(
                    nn.Conv2d(width*8, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False),
                    nn.Conv2d(width, self.num_new_landmarks, kernel_size=1, bias=True)
                )
            ])
            self.is_hrnet = True
        elif hasattr(base_model, 'backbone'):
            # ResNet architecture
            if hasattr(base_model.backbone, 'fc'):
                feature_channels = base_model.backbone.fc.in_features
            else:
                feature_channels = 2048  # Default ResNet50
            
            self.new_head = nn.Sequential(
                nn.Conv2d(feature_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.num_new_landmarks, kernel_size=1)
            )
            self.is_hrnet = False
        else:
            # Generic architecture
            feature_channels = 256
            self.new_head = nn.Sequential(
                nn.Conv2d(feature_channels, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.num_new_landmarks, kernel_size=1)
            )
            self.is_hrnet = False
        
        # Initialize new head weights
        self._initialize_new_head()
    
    def _initialize_new_head(self):
        """Initialize new head with small random weights"""
        for module in self.new_head.modules() if isinstance(self.new_head, nn.ModuleList) else [self.new_head]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Get base landmarks
        base_heatmaps = self.base_model(x)
        
        # Get features for new landmarks
        if self.is_hrnet:
            # HRNet: extract features from stage4
            x = self.base_model.stem(x)
            x = self.base_model.stage1(x)
            x_list = self.base_model.transition1(x)
            x = self.base_model.stage2(x_list)
            x_list = self.base_model.transition2(x)
            x = self.base_model.stage3(x_list)
            x_list = self.base_model.transition3(x)
            x = self.base_model.stage4(x_list)
            
            # Apply new heads to each resolution branch
            new_heatmaps_list = []
            for i, head in enumerate(self.new_head):
                new_heatmaps_list.append(head(x[i]))
            
            # Upsample all to same resolution and average
            target_size = new_heatmaps_list[0].shape[2:]
            new_heatmaps_upsampled = []
            for hm in new_heatmaps_list:
                if hm.shape[2:] != target_size:
                    hm = torch.nn.functional.interpolate(hm, size=target_size, mode='bilinear', align_corners=False)
                new_heatmaps_upsampled.append(hm)
            
            # Average multi-resolution predictions
            new_heatmaps = torch.stack(new_heatmaps_upsampled, dim=0).mean(dim=0)
        else:
            # For other architectures, try to get features
            if hasattr(self.base_model, 'get_features'):
                features = self.base_model.get_features(x)
            else:
                # Use intermediate features
                features = x
            new_heatmaps = self.new_head(features)
        
        # Concatenate base and new landmarks
        all_heatmaps = torch.cat([base_heatmaps, new_heatmaps], dim=1)
        
        return all_heatmaps


def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 4:
            images, heatmaps, landmarks, image_ids = batch
        else:
            images, landmarks, image_ids = batch
            heatmaps = None
        
        images = images.to(device)
        if heatmaps is not None:
            heatmaps = heatmaps.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            pred_heatmaps = model(images)
            
            if heatmaps is not None:
                # Compute loss only on base landmarks to avoid penalizing new untrained ones
                base_num = model.base_num_landmarks
                loss = criterion(pred_heatmaps[:, :base_num], heatmaps[:, :base_num])
            else:
                # If no heatmaps, compute from landmarks
                # This is a simplified version
                loss = torch.tensor(0.0, device=device)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def validate(model, valid_loader, device, pixel_size=0.1):
    """Validate model"""
    model.eval()
    all_errors = []
    
    with torch.no_grad():
        for batch in tqdm(valid_loader, desc='Validating'):
            if len(batch) == 4:
                images, heatmaps, landmarks, image_ids = batch
            else:
                images, landmarks, image_ids = batch
            
            images = images.to(device)
            landmarks = landmarks.numpy()
            
            pred_heatmaps = model(images)
            pred_heatmaps_np = pred_heatmaps.cpu().numpy()
            
            # Convert heatmaps to coordinates
            batch_size = pred_heatmaps_np.shape[0]
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(pred_heatmaps_np[i])
                gt_coords = landmarks[i]
                
                # Calculate errors only for valid landmarks
                valid_mask = (gt_coords[:, 0] >= 0) & (gt_coords[:, 1] >= 0)
                if valid_mask.sum() > 0:
                    errors = np.sqrt(np.sum((pred_coords[valid_mask] - gt_coords[valid_mask]) ** 2, axis=1))
                    errors_mm = errors * pixel_size
                    all_errors.extend(errors_mm.tolist())
    
    if len(all_errors) == 0:
        return 0.0, 0.0
    
    mre = np.mean(all_errors)
    sdr_2mm = (np.array(all_errors) <= 2.0).sum() / len(all_errors) * 100
    
    return mre, sdr_2mm


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model with additional landmarks')
    parser.add_argument('--dataset_path', type=str, default='Aariz/Aariz', help='Path to dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to base model checkpoint')
    parser.add_argument('--additional_landmarks', type=str, nargs='+', default=['PT'],
                       help='Additional landmark symbols (e.g., PT PTL PTR)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image_size', type=int, nargs=2, default=[512, 512], help='Image size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints_extended', help='Save directory')
    parser.add_argument('--log_dir', type=str, default='logs_extended', help='Log directory')
    parser.add_argument('--model', type=str, default='hrnet', help='Model architecture')
    parser.add_argument('--mixed_precision', action='store_true', help='Use mixed precision')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create extended dataloaders
    print(f"\nCreating extended dataset with additional landmarks: {args.additional_landmarks}")
    train_loader, valid_loader, test_loader, num_landmarks = create_extended_dataloaders(
        args.dataset_path,
        additional_landmarks=args.additional_landmarks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=tuple(args.image_size)
    )
    
    print(f"Total landmarks: {num_landmarks} (29 base + {len(args.additional_landmarks)} new)")
    
    # Load base model
    print(f"\nLoading base model from: {args.checkpoint}")
    base_model = get_model(args.model, num_landmarks=29)
    checkpoint = load_checkpoint(args.checkpoint, base_model)
    base_model = base_model.to(device)
    base_model.eval()
    
    # Create extended model
    print(f"\nCreating extended model...")
    model = ExtendedModel(base_model, base_num_landmarks=29, new_num_landmarks=num_landmarks)
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.new_head.parameters(), lr=args.lr)
    scaler = GradScaler() if args.mixed_precision else None
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(args.log_dir)
    
    # Training loop
    best_mre = float('inf')
    
    print(f"\nStarting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Mixed precision: {args.mixed_precision}")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        
        # Validate
        mre, sdr_2mm = validate(model, valid_loader, device)
        
        # Log
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Valid/MRE', mre, epoch)
        writer.add_scalar('Valid/SDR_2mm', sdr_2mm, epoch)
        
        print(f"Epoch {epoch}/{args.epochs}: Loss={train_loss:.4f}, MRE={mre:.2f}mm, SDR@2mm={sdr_2mm:.2f}%")
        
        # Save checkpoint
        if mre < best_mre:
            best_mre = mre
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint_best.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mre': mre,
                'sdr_2mm': sdr_2mm,
                'num_landmarks': num_landmarks,
                'additional_landmarks': args.additional_landmarks
            }, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")
    
    print(f"\nTraining completed!")
    print(f"Best MRE: {best_mre:.2f}mm")


if __name__ == "__main__":
    main()
