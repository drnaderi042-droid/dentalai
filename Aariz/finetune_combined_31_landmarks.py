"""
Fine-tuning مدل ترکیبی 31 لندمارک
فقط لایه‌های final_layers را train می‌کند تا P1/P2 بهتر یاد بگیرند
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
import random

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel
from utils import heatmap_to_coordinates, calculate_mre
import torch.nn.functional as F


def generate_heatmap(coords, size, sigma=3.0):
    """
    Generate Gaussian heatmap from coordinates
    Args:
        coords: (x, y) normalized [0, 1]
        size: (H, W) of heatmap
        sigma: Gaussian sigma
    Returns:
        heatmap: (H, W) numpy array
    """
    H, W = size
    x, y = coords
    
    # Convert to pixel coordinates
    x_px = int(x * W)
    y_px = int(y * H)
    
    # Clamp to valid range
    x_px = max(0, min(W - 1, x_px))
    y_px = max(0, min(H - 1, y_px))
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Gaussian
    heatmap = np.exp(-((x_grid - x_px)**2 + (y_grid - y_px)**2) / (2 * sigma**2))
    
    return heatmap


class Combined31LandmarkDataset(Dataset):
    """Dataset for 31 landmarks (29 anatomical + 2 P1/P2)"""
    
    def __init__(self, annotations_file, images_dir, image_size=768, heatmap_size=768, augment=False):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.augment = augment
        
        # Load annotations
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
                                    'x': value.get('x', 0) / 100.0,  # Normalize to [0, 1]
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
                        
                        # Remove UUID prefix if exists
                        if '-' in image_filename:
                            parts = image_filename.split('-')
                            if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
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
        
        # Transforms - استفاده از normalization مشابه training
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Generate heatmaps for 31 landmarks
        # 29 لندمارک اول: صفر (چون annotation نداریم)
        # 2 لندمارک آخر: P1/P2
        heatmaps = np.zeros((31, self.heatmap_size, self.heatmap_size), dtype=np.float32)
        
        # P1 (landmark 29)
        p1_heatmap = generate_heatmap(
            (sample['p1']['x'], sample['p1']['y']),
            (self.heatmap_size, self.heatmap_size),
            sigma=3.0
        )
        heatmaps[29] = p1_heatmap
        
        # P2 (landmark 30)
        p2_heatmap = generate_heatmap(
            (sample['p2']['x'], sample['p2']['y']),
            (self.heatmap_size, self.heatmap_size),
            sigma=3.0
        )
        heatmaps[30] = p2_heatmap
        
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
        
        # Ground truth coordinates for P1/P2
        coords = torch.tensor([
            sample['p1']['x'], sample['p1']['y'],
            sample['p2']['x'], sample['p2']['y']
        ], dtype=torch.float32)
        
        return image, heatmaps, coords, sample['image_path']


def extract_coordinates_from_heatmaps(heatmaps, image_size):
    """
    استخراج مختصات از heatmap‌ها
    Args:
        heatmaps: (batch, num_landmarks, H, W) tensor
        image_size: اندازه تصویر (برای scale کردن) - استفاده نمی‌شود اما برای سازگاری نگه داشته شده
    Returns:
        coords: (batch, num_landmarks, 2) numpy array - مختصات normalized [0, 1]
    """
    batch_size, num_landmarks, H, W = heatmaps.shape
    
    # استفاده از soft-argmax برای استخراج مختصات
    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
    
    # Normalize to [0, 1]
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    # Weighted average
    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
    
    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    
    # Stack: (batch, num_landmarks, 2)
    coords = torch.stack([x_mean, y_mean], dim=-1)
    
    # Detach از computation graph قبل از تبدیل به numpy
    return coords.detach().cpu().numpy()


def calculate_pixel_error(pred_coords, gt_coords, image_size):
    """
    محاسبه خطای پیکسلی برای P1/P2
    Args:
        pred_coords: (batch, 2, 2) - مختصات پیش‌بینی شده [p1_x, p1_y, p2_x, p2_y] normalized
        gt_coords: (batch, 4) - مختصات ground truth [p1_x, p1_y, p2_x, p2_y] normalized
        image_size: اندازه تصویر
    Returns:
        errors: (batch, 2) - خطاهای پیکسلی برای P1 و P2
    """
    # Convert to pixel coordinates
    pred_px = pred_coords * image_size  # (batch, 2, 2)
    gt_px = gt_coords.reshape(-1, 2, 2) * image_size  # (batch, 2, 2)
    
    # Calculate radial error for each landmark
    errors = np.sqrt(np.sum((pred_px - gt_px) ** 2, axis=2))  # (batch, 2)
    
    return errors


def calculate_sdr_pixel(errors, thresholds=[5, 10, 20]):
    """
    محاسبه SDR در پیکسل
    Args:
        errors: (batch, num_landmarks) - خطاهای پیکسلی
        thresholds: لیست threshold ها در پیکسل
    Returns:
        sdr_dict: دیکشنری با SDR برای هر threshold
    """
    sdr_dict = {}
    total = errors.size
    
    for threshold in thresholds:
        success = np.sum(errors <= threshold)
        sdr = (success / total) * 100.0
        sdr_dict[f'sdr_{threshold}px'] = sdr
    
    return sdr_dict


def finetune_combined_model(
    model_path='checkpoint_best_768_combined_31.pth',
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    output_path='checkpoint_best_768_combined_31_finetuned.pth',
    image_size=768,
    batch_size=4,
    num_epochs=100,
    learning_rate=1e-4,
    device='cuda',
    train_split=0.8,
    freeze_backbone=True,
    use_mixed_precision=True,
    resume_checkpoint=None
):
    """
    Fine-tune مدل ترکیبی 31 لندمارک
    
    Args:
        model_path: مسیر مدل ترکیبی
        annotations_file: فایل annotations برای P1/P2
        images_dir: مسیر تصاویر
        output_path: مسیر ذخیره مدل fine-tuned
        image_size: اندازه تصویر
        batch_size: batch size
        num_epochs: تعداد epochs
        learning_rate: learning rate
        device: 'cuda' یا 'cpu'
        train_split: نسبت train/val
        freeze_backbone: اگر True باشد، فقط final_layers را train می‌کند
    """
    print("="*80)
    print("Fine-tuning Combined 31-Landmark Model")
    print("="*80)
    
    # بررسی resume checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    best_mre = float('inf')
    resume_from_checkpoint = False
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n[RESUME] Loading checkpoint from {resume_checkpoint}...")
        resume_ckpt = torch.load(resume_checkpoint, map_location='cpu', weights_only=False)
        
        # بارگذاری مدل
        print(f"[1/6] Loading model from resume checkpoint...")
        model = HRNetLandmarkModel(num_landmarks=31, width=32)
        model.load_state_dict(resume_ckpt['model_state_dict'], strict=False)
        model = model.to(device)
        
        # بارگذاری اطلاعات training
        start_epoch = resume_ckpt.get('epoch', 0)
        best_val_loss = resume_ckpt.get('val_loss', float('inf'))
        best_mre = resume_ckpt.get('mre_avg', float('inf'))
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Best validation loss so far: {best_val_loss:.6f}")
        print(f"  Best MRE so far: {best_mre:.2f} px")
        resume_from_checkpoint = True
    else:
        # بارگذاری مدل از ابتدا
        print(f"\n[1/6] Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        model = HRNetLandmarkModel(num_landmarks=31, width=32)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
    
    # Freeze backbone اگر لازم باشد
    if freeze_backbone:
        print(f"[2/6] Freezing backbone (only training final_layers)...")
        for name, param in model.named_parameters():
            if not name.startswith('final_layers'):
                param.requires_grad = False
            else:
                param.requires_grad = True
                print(f"  Training: {name}")
    else:
        print(f"[2/6] Training all layers...")
        for param in model.parameters():
            param.requires_grad = True
    
    # ایجاد dataset
    print(f"\n[3/6] Creating dataset...")
    full_dataset = Combined31LandmarkDataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        image_size=image_size,
        heatmap_size=image_size,
        augment=False
    )
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid samples found in {annotations_file}")
    
    # Split train/val
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    # استفاده از num_workers بیشتر و pin_memory برای سرعت بیشتر
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,  # افزایش از 0 به 4
        pin_memory=True,  # برای انتقال سریع‌تر به GPU
        persistent_workers=True  # نگه داشتن workers بین epochs
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
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {image_size}x{image_size}")
    
    # Loss و optimizer
    print(f"\n[4/6] Setting up loss and optimizer...")
    
    # استفاده از loss ترکیبی: Heatmap MSE + Coordinate L1
    # این باعث می‌شود مدل هم heatmap و هم مختصات را یاد بگیرد
    class CombinedHeatmapCoordLoss(nn.Module):
        def __init__(self, heatmap_weight=1.0, coord_weight=10.0):
            super().__init__()
            self.heatmap_weight = heatmap_weight
            self.coord_weight = coord_weight
            self.heatmap_loss = nn.MSELoss()
            self.coord_loss = nn.L1Loss()
        
        def forward(self, pred_heatmaps, gt_heatmaps, pred_coords, gt_coords):
            # Heatmap loss
            hm_loss = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
            
            # Coordinate loss (برای اطمینان از دقت مختصات)
            coord_loss = self.coord_loss(pred_coords, gt_coords)
            
            # Total loss
            total_loss = self.heatmap_weight * hm_loss + self.coord_weight * coord_loss
            
            return total_loss, hm_loss, coord_loss
    
    criterion = CombinedHeatmapCoordLoss(heatmap_weight=1.0, coord_weight=10.0)
    
    # فقط پارامترهای trainable را optimize کن
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # اگر resume می‌کنیم، optimizer و scheduler را هم لود کن
    if resume_from_checkpoint:
        if 'optimizer_state_dict' in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            print(f"  Optimizer state loaded")
        if 'scheduler_state_dict' in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
            print(f"  Scheduler state loaded")
        # Learning rate را از checkpoint بگیر (اگر وجود دارد)
        if 'learning_rate' in resume_ckpt:
            for param_group in optimizer.param_groups:
                param_group['lr'] = resume_ckpt['learning_rate']
            print(f"  Learning rate restored: {resume_ckpt['learning_rate']}")
    
    # Mixed precision training برای سرعت بیشتر
    scaler = None
    if use_mixed_precision and device == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
        # اگر resume می‌کنیم، scaler state را هم لود کن
        if resume_from_checkpoint and 'scaler_state_dict' in resume_ckpt:
            scaler.load_state_dict(resume_ckpt['scaler_state_dict'])
            print(f"  GradScaler state loaded")
        print(f"  Mixed precision training: Enabled (FP16)")
    else:
        print(f"  Mixed precision training: Disabled")
    
    print(f"  Learning rate: {learning_rate}")
    print(f"  Optimizer: Adam")
    print(f"  Loss: Combined (Heatmap MSE + Coordinate L1)")
    print(f"    - Heatmap weight: 1.0")
    print(f"    - Coordinate weight: 10.0")
    
    # Training loop
    print(f"\n[5/6] Starting training...")
    if resume_from_checkpoint:
        patience_counter = resume_ckpt.get('patience_counter', 0)
    else:
        patience_counter = 0
    patience = 10
    
    # مسیر checkpoint آخرین epoch
    last_checkpoint_path = output_path.replace('.pth', '_last.pth')
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_p1p2_loss = 0.0
        train_hm_loss = 0.0
        train_coord_loss = 0.0
        
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, heatmaps_gt, coords_gt, image_paths in train_bar:
            images = images.to(device, non_blocking=True)
            heatmaps_gt = heatmaps_gt.to(device, non_blocking=True)
            coords_gt = coords_gt.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # Forward pass با mixed precision
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    heatmaps_pred = model(images)
                    
                    # اعمال sigmoid به خروجی مدل (چون GT heatmap‌ها در بازه [0, 1] هستند)
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                    
                    # Loss فقط برای P1/P2 (landmarks 29 و 30)
                    p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]  # (batch, 2, H, W)
                    p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]  # (batch, 2, H, W)
                    
                    # استخراج مختصات برای coordinate loss (مستقیماً روی GPU)
                    # استفاده از soft-argmax برای استخراج مختصات
                    batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                    y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                    x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                    y_coords = y_coords / (H - 1) if H > 1 else y_coords
                    x_coords = x_coords / (W - 1) if W > 1 else x_coords
                    heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                    x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    p1p2_coords_pred_tensor = torch.stack([x_mean, y_mean], dim=-1)  # (batch, 2, 2)
                    p1p2_coords_pred_tensor = p1p2_coords_pred_tensor.view(-1, 4)  # (batch, 4)
                    
                    # Combined loss (heatmap + coordinate)
                    loss, hm_loss, coord_loss = criterion(
                        p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                        p1p2_coords_pred_tensor, coords_gt
                    )
                
                # Backward با mixed precision
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Forward pass بدون mixed precision
                heatmaps_pred = model(images)
                
                # اعمال sigmoid به خروجی مدل (چون GT heatmap‌ها در بازه [0, 1] هستند)
                heatmaps_pred = torch.sigmoid(heatmaps_pred)
                
                # Loss فقط برای P1/P2 (landmarks 29 و 30)
                p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]  # (batch, 2, H, W)
                p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]  # (batch, 2, H, W)
                
                # استخراج مختصات برای coordinate loss (مستقیماً روی GPU)
                # استفاده از soft-argmax برای استخراج مختصات
                batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                y_coords = y_coords / (H - 1) if H > 1 else y_coords
                x_coords = x_coords / (W - 1) if W > 1 else x_coords
                heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                p1p2_coords_pred_tensor = torch.stack([x_mean, y_mean], dim=-1)  # (batch, 2, 2)
                p1p2_coords_pred_tensor = p1p2_coords_pred_tensor.view(-1, 4)  # (batch, 4)
                
                # Combined loss (heatmap + coordinate)
                loss, hm_loss, coord_loss = criterion(
                    p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                    p1p2_coords_pred_tensor, coords_gt
                )
                
                # Backward
                loss.backward()
                optimizer.step()
            
            train_loss += loss.item()
            train_p1p2_loss += loss.item()
            train_hm_loss += hm_loss.item()
            train_coord_loss += coord_loss.item()
            
            train_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'hm': f'{hm_loss.item():.6f}',
                'coord': f'{coord_loss.item():.6f}'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_p1p2_loss = train_p1p2_loss / len(train_loader)
        avg_train_hm_loss = train_hm_loss / len(train_loader)
        avg_train_coord_loss = train_coord_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_p1p2_loss = 0.0
        val_hm_loss = 0.0
        val_coord_loss = 0.0
        
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, heatmaps_gt, coords_gt, image_paths in val_bar:
                images = images.to(device, non_blocking=True)
                heatmaps_gt = heatmaps_gt.to(device, non_blocking=True)
                coords_gt = coords_gt.to(device, non_blocking=True)
                
                # Validation با mixed precision (فقط برای سرعت)
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        heatmaps_pred = model(images)
                        
                        # اعمال sigmoid به خروجی مدل
                        heatmaps_pred = torch.sigmoid(heatmaps_pred)
                        
                        # Loss فقط برای P1/P2
                        p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                        p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                        
                        # استخراج مختصات برای coordinate loss (مستقیماً روی GPU)
                        batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                        y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                        x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                        y_coords = y_coords / (H - 1) if H > 1 else y_coords
                        x_coords = x_coords / (W - 1) if W > 1 else x_coords
                        heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                        x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                        y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                        p1p2_coords_pred_tensor = torch.stack([x_mean, y_mean], dim=-1)  # (batch, 2, 2)
                        p1p2_coords_pred_tensor = p1p2_coords_pred_tensor.view(-1, 4)  # (batch, 4)
                        
                        loss, hm_loss, coord_loss = criterion(
                            p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                            p1p2_coords_pred_tensor, coords_gt
                        )
                else:
                    heatmaps_pred = model(images)
                    
                    # اعمال sigmoid به خروجی مدل
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                    
                    # Loss فقط برای P1/P2
                    p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
                    p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
                    
                    # استخراج مختصات برای coordinate loss (مستقیماً روی GPU)
                    batch_size, num_landmarks, H, W = p1p2_heatmaps_pred.shape
                    y_coords = torch.arange(H, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, H, 1)
                    x_coords = torch.arange(W, dtype=p1p2_heatmaps_pred.dtype, device=device).view(1, 1, 1, W)
                    y_coords = y_coords / (H - 1) if H > 1 else y_coords
                    x_coords = x_coords / (W - 1) if W > 1 else x_coords
                    heatmaps_sum = p1p2_heatmaps_pred.sum(dim=(2, 3), keepdim=True) + 1e-8
                    x_mean = (p1p2_heatmaps_pred * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    y_mean = (p1p2_heatmaps_pred * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
                    p1p2_coords_pred_tensor = torch.stack([x_mean, y_mean], dim=-1)  # (batch, 2, 2)
                    p1p2_coords_pred_tensor = p1p2_coords_pred_tensor.view(-1, 4)  # (batch, 4)
                    
                    loss, hm_loss, coord_loss = criterion(
                        p1p2_heatmaps_pred, p1p2_heatmaps_gt,
                        p1p2_coords_pred_tensor, coords_gt
                    )
                
                val_loss += loss.item()
                val_p1p2_loss += loss.item()
                val_hm_loss += hm_loss.item()
                val_coord_loss += coord_loss.item()
                
                val_bar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'hm': f'{hm_loss.item():.6f}',
                    'coord': f'{coord_loss.item():.6f}'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_p1p2_loss = val_p1p2_loss / len(val_loader)
        avg_val_hm_loss = val_hm_loss / len(val_loader)
        avg_val_coord_loss = val_coord_loss / len(val_loader)
        
        # محاسبه MRE و SDR برای validation
        print(f"\n  Calculating MRE and SDR on validation set...")
        model.eval()
        all_pred_coords = []
        all_gt_coords = []
        
        with torch.no_grad():
            for images, heatmaps_gt, coords_gt, image_paths in val_loader:
                images = images.to(device, non_blocking=True)
                
                # پیش‌بینی
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        heatmaps_pred = model(images)
                        heatmaps_pred = torch.sigmoid(heatmaps_pred)
                else:
                    heatmaps_pred = model(images)
                    heatmaps_pred = torch.sigmoid(heatmaps_pred)
                
                # استخراج مختصات P1/P2
                p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]  # (batch, 2, H, W)
                p1p2_coords_pred = extract_coordinates_from_heatmaps(p1p2_heatmaps_pred, image_size)
                
                # ذخیره مختصات
                all_pred_coords.append(p1p2_coords_pred)
                all_gt_coords.append(coords_gt.numpy())
        
        # ترکیب همه batch ها
        all_pred_coords = np.concatenate(all_pred_coords, axis=0)  # (N, 2, 2)
        all_gt_coords = np.concatenate(all_gt_coords, axis=0)  # (N, 4)
        
        # محاسبه خطاهای پیکسلی
        pixel_errors = calculate_pixel_error(all_pred_coords, all_gt_coords, image_size)  # (N, 2)
        
        # محاسبه MRE (میانگین خطای پیکسلی)
        mre_p1 = np.mean(pixel_errors[:, 0])
        mre_p2 = np.mean(pixel_errors[:, 1])
        mre_avg = np.mean(pixel_errors)
        
        # محاسبه SDR
        sdr_p1 = calculate_sdr_pixel(pixel_errors[:, 0:1], thresholds=[5, 10, 20])
        sdr_p2 = calculate_sdr_pixel(pixel_errors[:, 1:2], thresholds=[5, 10, 20])
        sdr_avg = calculate_sdr_pixel(pixel_errors, thresholds=[5, 10, 20])
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (HM: {avg_train_hm_loss:.6f}, Coord: {avg_train_coord_loss:.6f})")
        print(f"  Val Loss: {avg_val_loss:.6f} (HM: {avg_val_hm_loss:.6f}, Coord: {avg_val_coord_loss:.6f})")
        print(f"\n  Validation Metrics (P1/P2):")
        print(f"    MRE - P1: {mre_p1:.2f} px | P2: {mre_p2:.2f} px | Avg: {mre_avg:.2f} px")
        print(f"    SDR (5px)  - P1: {sdr_p1['sdr_5px']:.1f}% | P2: {sdr_p2['sdr_5px']:.1f}% | Avg: {sdr_avg['sdr_5px']:.1f}%")
        print(f"    SDR (10px) - P1: {sdr_p1['sdr_10px']:.1f}% | P2: {sdr_p2['sdr_10px']:.1f}% | Avg: {sdr_avg['sdr_10px']:.1f}%")
        print(f"    SDR (20px) - P1: {sdr_p1['sdr_20px']:.1f}% | P2: {sdr_p2['sdr_20px']:.1f}% | Avg: {sdr_avg['sdr_20px']:.1f}%")
        
        # Save best model (بر اساس validation loss یا MRE)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_mre = mre_avg
            patience_counter = 0
            
            print(f"  [OK] New best validation loss: {best_val_loss:.6f} (MRE: {mre_avg:.2f} px)")
            
            # Save best checkpoint
            checkpoint_data = {
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
                'learning_rate': optimizer.param_groups[0]['lr'],
                'patience_counter': patience_counter,
            }
            if scaler is not None:
                checkpoint_data['scaler_state_dict'] = scaler.state_dict()
            
            torch.save(checkpoint_data, output_path)
            print(f"  Model saved to {output_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping (patience={patience})")
                break
        
        # ذخیره checkpoint آخرین epoch (برای resume)
        last_checkpoint_data = {
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
            'learning_rate': optimizer.param_groups[0]['lr'],
            'patience_counter': patience_counter,
            'best_val_loss': best_val_loss,
            'best_mre': best_mre,
        }
        if scaler is not None:
            last_checkpoint_data['scaler_state_dict'] = scaler.state_dict()
        
        torch.save(last_checkpoint_data, last_checkpoint_path)
        if (epoch + 1) % 5 == 0:  # هر 5 epoch یکبار پیام بده
            print(f"  Last checkpoint saved to {last_checkpoint_path} (epoch {epoch + 1})")
    
    print(f"\n[6/6] Training completed!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Best MRE: {best_mre:.2f} px")
    print(f"  Model saved to: {output_path}")
    
    return model, best_val_loss


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune combined 31-landmark model')
    parser.add_argument('--model', type=str, default='checkpoint_best_768_combined_31.pth',
                        help='Path to combined model')
    parser.add_argument('--annotations', type=str, default='annotations_p1_p2.json',
                        help='Path to annotations JSON file')
    parser.add_argument('--images', type=str, default='Aariz/train/Cephalograms',
                        help='Path to images directory')
    parser.add_argument('--output', type=str, default='checkpoint_best_768_combined_31_finetuned.pth',
                        help='Output path for fine-tuned model')
    parser.add_argument('--image_size', type=int, default=768,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--freeze_backbone', action='store_true',
                        help='Freeze backbone, only train final_layers')
    parser.add_argument('--mixed_precision', action='store_true', default=True,
                        help='Use mixed precision training (FP16)')
    parser.add_argument('--no_mixed_precision', dest='mixed_precision', action='store_false',
                        help='Disable mixed precision training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from (e.g., checkpoint_best_768_combined_31_finetuned_last.pth)')
    
    args = parser.parse_args()
    
    try:
        model, best_loss = finetune_combined_model(
            model_path=args.model,
            annotations_file=args.annotations,
            images_dir=args.images,
            output_path=args.output,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            device=args.device,
            freeze_backbone=args.freeze_backbone,
            use_mixed_precision=args.mixed_precision,
            resume_checkpoint=args.resume
        )
        print("\n[OK] Fine-tuning completed successfully!")
    except Exception as e:
        print(f"\n[ERROR] Fine-tuning failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

