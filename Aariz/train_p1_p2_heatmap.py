"""
Train P1/P2 detector using heatmap approach - More accurate!
Target: < 10px error
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

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector


def generate_heatmap(coords, size, sigma=2.0):
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
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    # Gaussian
    heatmap = np.exp(-((x_grid - x_px)**2 + (y_grid - y_px)**2) / (2 * sigma**2))
    
    return heatmap


class P1P2HeatmapDataset(Dataset):
    """Dataset for heatmap-based training"""
    
    def __init__(self, annotations_file, images_dir, image_size=768, heatmap_size=192, augment=False):
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
        
        print(f"Loaded {len(self.samples)} samples")
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply transforms
        image = self.transform(image)
        
        # Generate heatmaps
        p1_heatmap = generate_heatmap(
            (sample['p1']['x'], sample['p1']['y']),
            (self.heatmap_size, self.heatmap_size),
            sigma=3.0
        )
        p2_heatmap = generate_heatmap(
            (sample['p2']['x'], sample['p2']['y']),
            (self.heatmap_size, self.heatmap_size),
            sigma=3.0
        )
        
        # Stack: (2, H, W)
        heatmaps = np.stack([p1_heatmap, p2_heatmap], axis=0)
        heatmaps = torch.tensor(heatmaps, dtype=torch.float32)
        
        # Ground truth coordinates for coordinate loss
        coords = torch.tensor([
            sample['p1']['x'], sample['p1']['y'],
            sample['p2']['x'], sample['p2']['y']
        ], dtype=torch.float32)
        
        return image, heatmaps, coords


class CombinedLoss(nn.Module):
    """Combined loss: Heatmap MSE + Coordinate L1"""
    def __init__(self, heatmap_weight=1.0, coord_weight=0.5):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.heatmap_loss = nn.MSELoss()
        self.coord_loss = nn.L1Loss()
    
    def forward(self, pred_heatmaps, gt_heatmaps, pred_coords, gt_coords):
        # Heatmap loss
        hm_loss = self.heatmap_loss(pred_heatmaps, gt_heatmaps)
        
        # Coordinate loss (from extracted coordinates)
        coord_loss = self.coord_loss(pred_coords, gt_coords)
        
        # Combined
        total_loss = self.heatmap_weight * hm_loss + self.coord_weight * coord_loss
        
        return total_loss, hm_loss, coord_loss


def train_heatmap_model(
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    output_dir='models',
    image_size=768,
    heatmap_size=192,
    batch_size=4,
    num_epochs=200,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train heatmap-based model"""
    
    print("="*70)
    print("HEATMAP-BASED P1/P2 TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  - Image Size: {image_size}x{image_size}")
    print(f"  - Heatmap Size: {heatmap_size}x{heatmap_size}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Device: {device}")
    print(f"  - Target: < 10px error")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset
    train_dataset = P1P2HeatmapDataset(
        annotations_file, images_dir,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False
    )
    
    # Split 80/20
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\nDataset split:")
    print(f"  - Training: {train_size} samples")
    print(f"  - Validation: {val_size} samples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )
    
    # Model
    print(f"\nCreating model...")
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=True,
        output_size=heatmap_size
    )
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(heatmap_weight=1.0, coord_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True, min_lr=1e-6
    )
    
    # Training
    best_val_loss = float('inf')
    best_pixel_error = float('inf')
    patience_counter = 0
    early_stop_patience = 30
    
    print(f"\n[TRAINING] Starting...")
    print("="*70)
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss_total = 0.0
        train_hm_loss = 0.0
        train_coord_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, gt_heatmaps, gt_coords in pbar:
            images = images.to(device)
            gt_heatmaps = gt_heatmaps.to(device)
            gt_coords = gt_coords.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            pred_heatmaps = model(images)
            pred_coords = model.extract_coordinates(pred_heatmaps)
            
            # Loss
            loss, hm_loss, coord_loss = criterion(
                pred_heatmaps, gt_heatmaps,
                pred_coords, gt_coords
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            train_hm_loss += hm_loss.item()
            train_coord_loss += coord_loss.item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'hm': f'{hm_loss.item():.4f}',
                'coord': f'{coord_loss.item():.4f}'
            })
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_hm = train_hm_loss / len(train_loader)
        avg_train_coord = train_coord_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss_total = 0.0
        pixel_errors = []
        
        with torch.no_grad():
            for images, gt_heatmaps, gt_coords in val_loader:
                images = images.to(device)
                gt_heatmaps = gt_heatmaps.to(device)
                gt_coords = gt_coords.to(device)
                
                pred_heatmaps = model(images)
                pred_coords = model.extract_coordinates(pred_heatmaps)
                
                loss, _, _ = criterion(pred_heatmaps, gt_heatmaps, pred_coords, gt_coords)
                val_loss_total += loss.item()
                
                # Calculate pixel error
                pred_px = pred_coords.cpu().numpy() * image_size
                gt_px = gt_coords.cpu().numpy() * image_size
                
                p1_errors = np.sqrt((pred_px[:, 0] - gt_px[:, 0])**2 + (pred_px[:, 1] - gt_px[:, 1])**2)
                p2_errors = np.sqrt((pred_px[:, 2] - gt_px[:, 2])**2 + (pred_px[:, 3] - gt_px[:, 3])**2)
                avg_errors = (p1_errors + p2_errors) / 2
                pixel_errors.extend(avg_errors.tolist())
        
        avg_val_loss = val_loss_total / len(val_loader)
        avg_pixel_error = np.mean(pixel_errors)
        
        scheduler.step(avg_val_loss)
        
        # Print
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f} (HM: {avg_train_hm:.4f}, Coord: {avg_train_coord:.4f})")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Avg Pixel Error: {avg_pixel_error:.2f} px")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best
        if avg_pixel_error < best_pixel_error:
            best_pixel_error = avg_pixel_error
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(output_dir, 'hrnet_p1p2_heatmap_best.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'pixel_error': avg_pixel_error,
                'image_size': image_size,
                'heatmap_size': heatmap_size,
            }, checkpoint_path)
            
            print(f"  >>> BEST MODEL! Pixel Error: {avg_pixel_error:.2f} px")
        else:
            patience_counter += 1
            print(f"  >>> No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs.")
            break
        
        print("-"*70)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nBest Pixel Error: {best_pixel_error:.2f} px")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"\nModel saved to: {os.path.join(output_dir, 'hrnet_p1p2_heatmap_best.pth')}")
    
    if best_pixel_error < 10:
        print("\n🎉 SUCCESS! Pixel error < 10px achieved!")
    elif best_pixel_error < 20:
        print("\n✅ Good! Pixel error < 20px")
    else:
        print("\n⚠️ Pixel error still > 20px. May need more data or tuning.")
    
    return model


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'Aariz/train/Cephalograms'  # Note: Script runs from Aariz/, so path is relative to Aariz/
    
    if not Path(annotations_file).exists():
        print(f"ERROR: {annotations_file} not found!")
        sys.exit(1)
    
    model = train_heatmap_model(
        annotations_file=annotations_file,
        images_dir=images_dir,
        image_size=1024,
        heatmap_size=256,  # 1/4 of image_size for better resolution
        batch_size=2,  # Reduced for 1024x1024 images
        num_epochs=200,
        learning_rate=0.001,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )













