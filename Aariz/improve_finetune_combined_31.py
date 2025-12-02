"""
بهبود fine-tuning مدل combined با راهکارهای پیشرفته
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel
from utils import heatmap_to_coordinates, calculate_mre
import torch.nn.functional as F


def generate_heatmap(coords, size, sigma=3.0):
    """Generate Gaussian heatmap from coordinates"""
    H, W = size
    x, y = coords
    
    x_px = int(x * W)
    y_px = int(y * H)
    
    x_px = max(0, min(W - 1, x_px))
    y_px = max(0, min(H - 1, y_px))
    
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    heatmap = np.exp(-((x_grid - x_px)**2 + (y_grid - y_px)**2) / (2 * sigma**2))
    
    return heatmap


class Combined31LandmarkDataset(Dataset):
    """Dataset for 31 landmarks with augmentation"""
    
    def __init__(self, annotations_file, images_dir, image_size=768, heatmap_size=768, augment=True):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment
        
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.samples = []
        for item in data:
            if 'annotations' in item and item['annotations']:
                annotation = item['annotations'][0]
                
                if 'result' in annotation:
                    p1 = None
                    p2 = None
                    
                    for result_item in annotation['result']:
                        if result_item.get('type') == 'keypointlabels':
                            value = result_item.get('value', {})
                            labels = value.get('keypointlabels', [])
                            
                            if 'p1' in labels:
                                p1 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                            elif 'p2' in labels:
                                p2 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                    
                    if p1 and p2:
                        image_url = item['data'].get('image', '')
                        image_filename = image_url.split('/')[-1]
                        
                        if '-' in image_filename:
                            parts = image_filename.split('-')
                            if len(parts[0]) == 8:
                                image_filename = '-'.join(parts[1:])
                        
                        image_path = self.images_dir / image_filename
                        if not image_path.exists():
                            alt_filename = image_url.split('/')[-1]
                            alt_path = self.images_dir / alt_filename
                            if alt_path.exists():
                                image_path = alt_path
                            else:
                                continue
                        
                        self.samples.append({
                            'image_path': str(image_path),
                            'p1': p1,
                            'p2': p2
                        })
        
        print(f"Loaded {len(self.samples)} samples with P1/P2 annotations")
        
        # Augmentation transforms
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        image = Image.open(sample['image_path']).convert('RGB')
        original_size = image.size
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Generate heatmaps for P1/P2 only
        heatmaps = np.zeros((31, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # P1 (landmark 29)
        heatmaps[29] = generate_heatmap((sample['p1']['x'], sample['p1']['y']), 
                                       (self.heatmap_size, self.heatmap_size), sigma=3.0)
        
        # P2 (landmark 30)
        heatmaps[30] = generate_heatmap((sample['p2']['x'], sample['p2']['y']), 
                                       (self.heatmap_size, self.heatmap_size), sigma=3.0)
        
        # Ground truth coordinates (normalized)
        coords = np.zeros((4,), dtype=np.float32)  # [p1_x, p1_y, p2_x, p2_y]
        coords[0] = sample['p1']['x']
        coords[1] = sample['p1']['y']
        coords[2] = sample['p2']['x']
        coords[3] = sample['p2']['y']
        
        return image_tensor, torch.from_numpy(heatmaps), torch.from_numpy(coords), sample['image_path']


class FocalHeatmapLoss(nn.Module):
    """Focal loss for heatmap prediction - focuses on hard examples"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred, target):
        mse = self.mse(pred, target)
        # Focal weight: (1 - error)^gamma
        focal_weight = (1 - mse) ** self.gamma
        loss = self.alpha * focal_weight * mse
        return loss.mean()


class ImprovedCombinedLoss(nn.Module):
    """Improved loss function with focal loss and higher coordinate weight"""
    def __init__(self, heatmap_weight=1.0, coord_weight=20.0, use_focal=True):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        
        if use_focal:
            self.heatmap_loss = FocalHeatmapLoss(alpha=0.25, gamma=2.0)
        else:
            self.heatmap_loss = nn.MSELoss()
        
        self.coord_loss = nn.L1Loss()
    
    def forward(self, pred_heatmaps, gt_heatmaps, pred_coords, gt_coords):
        hm_loss = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
        coord_loss = self.coord_loss(pred_coords, gt_coords)
        total_loss = self.heatmap_weight * hm_loss + self.coord_weight * coord_loss
        return total_loss, hm_loss, coord_loss


def extract_coordinates_from_heatmaps(heatmaps, image_size):
    """Extract coordinates from heatmaps using soft-argmax"""
    batch_size, num_landmarks, H, W = heatmaps.shape
    
    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
    
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
    
    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    
    coords = torch.stack([x_mean, y_mean], dim=-1)
    return coords.detach().cpu().numpy()


def calculate_pixel_error(pred_coords, gt_coords, image_size):
    """Calculate pixel error"""
    # pred_coords: (N, num_landmarks, 2)
    # gt_coords: (N, num_landmarks, 2)
    pred_px = pred_coords * image_size
    gt_px = gt_coords * image_size
    errors = np.sqrt(np.sum((pred_px - gt_px) ** 2, axis=2))  # (N, num_landmarks)
    return errors


def calculate_sdr_pixel(errors, thresholds=[5, 10, 20]):
    """Calculate SDR"""
    sdr_dict = {}
    total = len(errors)
    
    for threshold in thresholds:
        success = np.sum(errors <= threshold)
        sdr = (success / total) * 100.0
        sdr_dict[f'sdr_{threshold}px'] = sdr
    
    return sdr_dict


def improve_finetune_combined_model(
    model_path='checkpoint_best_768_combined_31_finetuned.pth',
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    output_path='checkpoint_best_768_combined_31_improved.pth',
    image_size=768,
    batch_size=8,
    num_epochs=50,
    backbone_lr=1e-5,  # Lower LR for backbone
    final_layers_lr=1e-4,  # Higher LR for final_layers
    device='cuda',
    train_split=0.8,
    use_mixed_precision=True,
    use_focal_loss=True,
    use_augmentation=True
):
    """
    بهبود fine-tuning با راهکارهای پیشرفته:
    1. Unfreeze backbone با learning rate پایین‌تر
    2. Focal loss برای heatmap
    3. Data augmentation
    4. Cosine annealing scheduler
    5. Higher coordinate loss weight
    """
    print("="*80)
    print("Improved Fine-tuning Combined 31-Landmark Model")
    print("="*80)
    
    # Load model
    print(f"\n[1/6] Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model = HRNetLandmarkModel(num_landmarks=31, width=32)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    
    # Unfreeze backbone with different learning rates
    print(f"\n[2/6] Setting up differential learning rates...")
    backbone_params = []
    final_layers_params = []
    
    for name, param in model.named_parameters():
        if name.startswith('final_layers'):
            param.requires_grad = True
            final_layers_params.append(param)
            print(f"  Training (LR={final_layers_lr}): {name}")
        else:
            param.requires_grad = True  # Unfreeze backbone
            backbone_params.append(param)
            print(f"  Training (LR={backbone_lr}): {name}")
    
    # Create dataset with augmentation
    print(f"\n[3/6] Creating dataset (augmentation: {use_augmentation})...")
    full_dataset = Combined31LandmarkDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        image_size=image_size,
        heatmap_size=image_size,
        augment=use_augmentation
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid samples found in {annotations_file}")
    
    # Split train/val
    import random
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    print(f"  Total samples: {len(full_dataset)}")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    
    # Loss and optimizer with differential learning rates
    print(f"\n[4/6] Setting up loss and optimizer...")
    
    criterion = ImprovedCombinedLoss(
        heatmap_weight=1.0, 
        coord_weight=20.0,  # Increased from 10.0
        use_focal=use_focal_loss
    )
    
    # Optimizer with different learning rates
    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': backbone_lr},
        {'params': final_layers_params, 'lr': final_layers_lr}
    ])
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )
    
    # Mixed precision
    scaler = None
    if use_mixed_precision and device == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        print(f"  Mixed precision: Enabled")
    
    print(f"  Backbone LR: {backbone_lr}")
    print(f"  Final layers LR: {final_layers_lr}")
    print(f"  Loss: Improved Combined (Focal Heatmap + Coordinate L1)")
    print(f"    - Heatmap weight: 1.0")
    print(f"    - Coordinate weight: 20.0")
    print(f"    - Focal loss: {use_focal_loss}")
    
    # Training loop
    print(f"\n[5/6] Starting improved training...")
    best_val_loss = float('inf')
    best_mre = float('inf')
    patience = 15
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_hm_loss = 0.0
        train_coord_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, heatmaps_gt, coords_gt, image_paths in train_bar:
            images = images.to(device, non_blocking=True)
            heatmaps_gt = heatmaps_gt.to(device, non_blocking=True)
            coords_gt = coords_gt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    heatmaps_pred = model(images)
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                    
                    p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                    p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                    
                    # Extract coordinates
                    batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                    y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                    x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                    y_coords = y_coords / (H - 1) if H > 1 else y_coords
                    x_coords = x_coords / (W - 1) if W > 1 else x_coords
                    heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                    x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    p1p2_coords_pred = torch.stack([x_mean, y_mean], dim=-1).view(batch_size, -1)
                    
                    loss, hm_loss, coord_loss = criterion(
                        p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                        p1p2_coords_pred, coords_gt
                    )
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                heatmaps_pred = model(images)
                heatmaps_pred = torch.sigmoid(heatmaps_pred)
                p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                
                batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                y_coords = y_coords / (H - 1) if H > 1 else y_coords
                x_coords = x_coords / (W - 1) if W > 1 else x_coords
                heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                p1p2_coords_pred = torch.stack([x_mean, y_mean], dim=-1).view(batch_size, -1)
                
                loss, hm_loss, coord_loss = criterion(
                    p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                    p1p2_coords_pred, coords_gt
                )
                
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_hm_loss += hm_loss.item()
            train_coord_loss += coord_loss.item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'hm': f'{hm_loss.item():.6f}',
                'coord': f'{coord_loss.item():.6f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_hm_loss = train_hm_loss / len(train_loader)
        avg_train_coord_loss = train_coord_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_hm_loss = 0.0
        val_coord_loss = 0.0
        
        all_pred_coords = []
        all_gt_coords = []
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, heatmaps_gt, coords_gt, image_paths in val_bar:
                images = images.to(device, non_blocking=True)
                heatmaps_gt = heatmaps_gt.to(device, non_blocking=True)
                coords_gt = coords_gt.to(device, non_blocking=True)
                
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        heatmaps_pred = model(images)
                        heatmaps_pred = torch.sigmoid(heatmaps_pred)
                        
                        p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                        p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                        
                        batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                        y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                        x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                        y_coords = y_coords / (H - 1) if H > 1 else y_coords
                        x_coords = x_coords / (W - 1) if W > 1 else x_coords
                        heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                        x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                        y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                        p1p2_coords_pred = torch.stack([x_mean, y_mean], dim=-1).view(batch_size, -1)
                        
                        loss, hm_loss, coord_loss = criterion(
                            p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                            p1p2_coords_pred, coords_gt
                        )
                else:
                    heatmaps_pred = model(images)
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                    p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                    p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                    
                    batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                    y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                    x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                    y_coords = y_coords / (H - 1) if H > 1 else y_coords
                    x_coords = x_coords / (W - 1) if W > 1 else x_coords
                    heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                    x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    p1p2_coords_pred = torch.stack([x_mean, y_mean], dim=-1).view(batch_size, -1)
                    
                    loss, hm_loss, coord_loss = criterion(
                        p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                        p1p2_coords_pred, coords_gt
                    )
                
                val_loss += loss.item()
                val_hm_loss += hm_loss.item()
                val_coord_loss += coord_loss.item()
                
                p1p2_coords_pred_np = p1p2_coords_pred.detach().cpu().numpy()
                all_pred_coords.append(p1p2_coords_pred_np)
                all_gt_coords.append(coords_gt.cpu().numpy())
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'hm': f'{hm_loss.item():.6f}',
                    'coord': f'{coord_loss.item():.6f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_hm_loss = val_hm_loss / len(val_loader)
        avg_val_coord_loss = val_coord_loss / len(val_loader)
        
        # Calculate metrics
        all_pred_coords = np.concatenate(all_pred_coords, axis=0)  # (N, 4)
        all_gt_coords = np.concatenate(all_gt_coords, axis=0)  # (N, 4)
        
        # Reshape to (N, 2, 2) for pixel error calculation
        pred_coords_2d = all_pred_coords.reshape(-1, 2, 2)  # (N, 2, 2)
        gt_coords_2d = all_gt_coords.reshape(-1, 2, 2)  # (N, 2, 2)
        
        pixel_errors = calculate_pixel_error(pred_coords_2d, gt_coords_2d, image_size)  # (N, 2)
        mre_p1 = np.mean(pixel_errors[:, 0])
        mre_p2 = np.mean(pixel_errors[:, 1])
        mre_avg = np.mean(pixel_errors)
        
        sdr_p1 = calculate_sdr_pixel(pixel_errors[:, 0:1])
        sdr_p2 = calculate_sdr_pixel(pixel_errors[:, 1:2])
        sdr_avg = calculate_sdr_pixel(pixel_errors)
        
        # Update learning rate
        scheduler.step()
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (HM: {avg_train_hm_loss:.6f}, Coord: {avg_train_coord_loss:.6f})")
        print(f"  Val Loss: {avg_val_loss:.6f} (HM: {avg_val_hm_loss:.6f}, Coord: {avg_val_coord_loss:.6f})")
        print(f"  LR - Backbone: {optimizer.param_groups[0]['lr']:.2e}, Final: {optimizer.param_groups[1]['lr']:.2e}")
        print(f"\n  Validation Metrics (P1/P2):")
        print(f"    MRE - P1: {mre_p1:.2f} px | P2: {mre_p2:.2f} px | Avg: {mre_avg:.2f} px")
        print(f"    SDR (5px)  - P1: {sdr_p1['sdr_5px']:.1f}% | P2: {sdr_p2['sdr_5px']:.1f}% | Avg: {sdr_avg['sdr_5px']:.1f}%")
        print(f"    SDR (10px) - P1: {sdr_p1['sdr_10px']:.1f}% | P2: {sdr_p2['sdr_10px']:.1f}% | Avg: {sdr_avg['sdr_10px']:.1f}%")
        print(f"    SDR (20px) - P1: {sdr_p1['sdr_20px']:.1f}% | P2: {sdr_p2['sdr_20px']:.1f}% | Avg: {sdr_avg['sdr_20px']:.1f}%")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_mre = mre_avg
            patience_counter = 0
            
            print(f"  [OK] New best validation loss: {best_val_loss:.6f} (MRE: {mre_avg:.2f} px)")
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch + 1,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss,
                'mre_p1': mre_p1,
                'mre_p2': mre_p2,
                'mre_avg': mre_avg,
                'sdr_p1': sdr_p1,
                'sdr_p2': sdr_p2,
                'sdr_avg': sdr_avg,
                'num_landmarks': 31,
                'width': 32,
                'image_size': image_size,
            }, output_path)
            print(f"  Model saved to {output_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break
    
    print(f"\n[6/6] Training completed!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best MRE: {best_mre:.2f} px")
    print(f"  Model saved to: {output_path}")
    
    return model, best_val_loss


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Improved fine-tuning for combined 31-landmark model')
    parser.add_argument('--model', type=str, default='checkpoint_best_768_combined_31_finetuned.pth')
    parser.add_argument('--annotations', type=str, default='annotations_p1_p2.json')
    parser.add_argument('--images', type=str, default='Aariz/train/Cephalograms')
    parser.add_argument('--output', type=str, default='checkpoint_best_768_combined_31_improved.pth')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--final_layers_lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--no_focal', dest='use_focal', action='store_false', default=True)
    parser.add_argument('--no_augment', dest='use_augment', action='store_false', default=True)
    
    args = parser.parse_args()
    
    try:
        model, best_loss = improve_finetune_combined_model(
            model_path=args.model,
            annotations_file=args.annotations,
            images_dir=args.images,
            output_path=args.output,
            num_epochs=args.epochs,
            backbone_lr=args.backbone_lr,
            final_layers_lr=args.final_layers_lr,
            batch_size=args.batch_size,
            use_focal_loss=args.use_focal,
            use_augmentation=args.use_augment
        )
        print("\n[OK] Improved fine-tuning completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

