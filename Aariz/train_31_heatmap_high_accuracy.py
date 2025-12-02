"""
Train 31-landmark heatmap model for sub-2px MRE accuracy
Based on hrnet_p1p2_heatmap_best.pth success but for all 31 landmarks
Target: < 2px MRE on CPU server deployment
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import argparse

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from model_heatmap_31 import HRNet31HeatmapDetector
from utils import heatmap_to_coordinates, calculate_mre


def generate_heatmap(coords, size, sigma=1.5):
    """
    Generate high-precision Gaussian heatmap
    Smaller sigma for sharper peaks (better accuracy)
    """
    H, W = size
    x, y = coords

    # Convert to pixel coordinates
    x_px = x * W
    y_px = y * H

    x_px = max(0, min(W - 1, x_px))
    y_px = max(0, min(H - 1, y_px))

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Smaller sigma for sharper, more accurate peaks
    heatmap = np.exp(-((x_grid - x_px)**2 + (y_grid - y_px)**2) / (2 * sigma**2))

    return heatmap


class Landmark31HeatmapDataset(Dataset):
    """High-precision dataset for 31 landmark heatmap training"""

    def __init__(self, annotations_file, images_dir, image_size=768, heatmap_size=384, augment=True):
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
                    landmarks = {}

                    # Extract all 31 landmarks
                    for result_item in annotation['result']:
                        if result_item.get('type') == 'keypointlabels':
                            value = result_item.get('value', {})
                            labels = value.get('keypointlabels', [])

                            # Map landmark labels to indices (1-31)
                            for label in labels:
                                if label.isdigit() and 1 <= int(label) <= 31:
                                    landmark_idx = int(label) - 1  # 0-based indexing
                                    landmarks[landmark_idx] = {
                                        'x': value.get('x', 0) / 100.0,
                                        'y': value.get('y', 0) / 100.0
                                    }

                    # Only include samples with all 31 landmarks
                    if len(landmarks) == 31:
                        image_url = item['data'].get('image', '')
                        image_filename = image_url.split('/')[-1]

                        # Handle UUID prefix
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
                            'landmarks': landmarks
                        })

        print(f"Loaded {len(self.samples)} samples with complete 31 landmarks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load and preprocess image
        image = Image.open(sample['image_path']).convert('RGB')

        # Calculate scaling to maintain aspect ratio
        aspect_ratio = image.width / image.height
        if aspect_ratio > 1:
            new_width = self.image_size
            new_height = int(self.image_size / aspect_ratio)
        else:
            new_height = self.image_size
            new_width = int(self.image_size * aspect_ratio)

        # Resize image
        image = image.resize((new_width, new_height), Image.LANCZOS)

        # Create square canvas
        square_image = Image.new('RGB', (self.image_size, self.image_size), (0, 0, 0))
        x_offset = (self.image_size - new_width) // 2
        y_offset = (self.image_size - new_height) // 2
        square_image.paste(image, (x_offset, y_offset))

        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(square_image)

        # Generate heatmaps for all 31 landmarks
        heatmaps = np.zeros((31, self.heatmap_size, self.heatmap_size), dtype=np.float32)

        # Scale factor for coordinate transformation
        scale_x = new_width / self.image_size
        scale_y = new_height / self.image_size

        for landmark_idx in range(31):
            if landmark_idx in sample['landmarks']:
                landmark = sample['landmarks'][landmark_idx]

                # Adjust coordinates for padding and scaling
                adjusted_x = (landmark['x'] * scale_x * self.image_size + x_offset) / self.image_size
                adjusted_y = (landmark['y'] * scale_y * self.image_size + y_offset) / self.image_size

                # Generate high-precision heatmap
                heatmap = generate_heatmap((adjusted_x, adjusted_y), (self.heatmap_size, self.heatmap_size), sigma=1.5)
                heatmaps[landmark_idx] = heatmap

        # Ground truth coordinates for coordinate loss
        coords = []
        for landmark_idx in range(31):
            if landmark_idx in sample['landmarks']:
                landmark = sample['landmarks'][landmark_idx]
                adjusted_x = (landmark['x'] * scale_x * self.image_size + x_offset) / self.image_size
                adjusted_y = (landmark['y'] * scale_y * self.image_size + y_offset) / self.image_size
                coords.extend([adjusted_x, adjusted_y])
            else:
                coords.extend([0.5, 0.5])  # Default center position

        return image_tensor, torch.tensor(heatmaps, dtype=torch.float32), torch.tensor(coords, dtype=torch.float32)


class AdvancedCombinedLoss(nn.Module):
    """Advanced combined loss for high accuracy training"""

    def __init__(self, heatmap_weight=1.0, coord_weight=5.0, focal_alpha=2.0, focal_gamma=2.0):
        super().__init__()
        self.heatmap_weight = heatmap_weight
        self.coord_weight = coord_weight
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        # Heatmap losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        # Coordinate loss with higher weight for precision
        self.coord_loss = nn.L1Loss()

    def focal_loss(self, pred, target):
        """Focal loss for better heatmap training"""
        bce_loss = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * bce_loss
        return focal_loss.mean()

    def forward(self, pred_heatmaps, gt_heatmaps, pred_coords, gt_coords):
        # Heatmap loss with focal component
        mse_loss = self.mse_loss(pred_heatmaps, gt_heatmaps)
        focal_loss = self.focal_loss(pred_heatmaps, gt_heatmaps)
        hm_loss = 0.7 * mse_loss + 0.3 * focal_loss

        # Coordinate loss with higher weight
        coord_loss = self.coord_loss(pred_coords, gt_coords)

        # Combined loss
        total_loss = self.heatmap_weight * hm_loss + self.coord_weight * coord_loss

        return total_loss, hm_loss, coord_loss


def train_31_heatmap_model(
    annotations_file='annotations_31.json',
    images_dir='Aariz/train/Cephalograms',
    output_dir='models',
    image_size=768,
    heatmap_size=384,  # Higher resolution for better accuracy
    batch_size=2,  # Smaller batch for stability
    num_epochs=300,
    learning_rate=0.0005,  # Lower LR for stability
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """Train high-accuracy 31-landmark heatmap model"""

    print("="*80)
    print("HIGH-ACCURACY 31-LANDMARK HEATMAP TRAINING")
    print("Target: < 2px MRE (similar to hrnet_p1p2_heatmap_best.pth)")
    print("="*80)

    print("\nConfiguration:")
    print(f"  - Image Size: {image_size}x{image_size}")
    print(f"  - Heatmap Size: {heatmap_size}x{heatmap_size}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Device: {device}")
    print(f"  - Target: < 2px MRE")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Dataset with augmentation
    train_dataset = Landmark31HeatmapDataset(
        annotations_file, images_dir,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=True
    )

    # Split 85/15 for better validation
    train_size = int(0.85 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print("\nDataset split:")
    print(f"  - Training: {train_size} samples")
    print(f"  - Validation: {val_size} samples")

    # DataLoaders with optimizations
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True if device == 'cuda' else False,
        drop_last=True  # Ensure consistent batch sizes
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if device == 'cuda' else False
    )

    # Model with larger HRNet variant
    print("\nCreating HRNet-W32 model...")
    model = HRNet31HeatmapDetector(
        num_landmarks=31,
        hrnet_variant='hrnet_w32',  # Larger model for better accuracy
        pretrained=True,
        output_size=heatmap_size
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")

    # Advanced loss and optimizer
    criterion = AdvancedCombinedLoss(
        heatmap_weight=1.0,
        coord_weight=5.0,  # Higher coordinate weight for sub-2px accuracy
        focal_alpha=2.0,
        focal_gamma=2.0
    )

    # Optimizer with different LRs for backbone vs head
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': learning_rate * 0.1},  # Lower LR for backbone
        {'params': head_params, 'lr': learning_rate}  # Higher LR for head
    ], weight_decay=1e-4)

    # Advanced scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6
    )

    # Training loop
    best_val_loss = float('inf')
    best_mre = float('inf')
    patience_counter = 0
    early_stop_patience = 50

    print("\n[TRAINING] Starting high-accuracy training...")
    print("="*80)

    for epoch in range(num_epochs):
        # Training phase
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

            # Forward pass
            pred_heatmaps = model(images)
            pred_coords = model.extract_coordinates(pred_heatmaps)[0]

            # Loss calculation
            loss, hm_loss, coord_loss = criterion(
                pred_heatmaps, gt_heatmaps,
                pred_coords, gt_coords
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            train_loss_total += loss.item()
            train_hm_loss += hm_loss.item()
            train_coord_loss += coord_loss.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.6f}",
                'hm': f"{hm_loss.item():.6f}",
                'coord': f"{coord_loss.item():.6f}"
            })

        # Validation phase
        model.eval()
        val_loss_total = 0.0
        val_pixel_errors = []

        with torch.no_grad():
            for images, gt_heatmaps, gt_coords in val_loader:
                images = images.to(device)
                gt_heatmaps = gt_heatmaps.to(device)
                gt_coords = gt_coords.to(device)

                pred_heatmaps = model(images)
                pred_coords = model.extract_coordinates(pred_heatmaps)[0]

                loss, _, _ = criterion(pred_heatmaps, gt_heatmaps, pred_coords, gt_coords)
                val_loss_total += loss.item()

                # Calculate pixel errors for MRE
                for b in range(images.shape[0]):
                    pred_coords_np = pred_coords[b].cpu().numpy().reshape(-1, 2)
                    gt_coords_np = gt_coords[b].cpu().numpy().reshape(-1, 2)

                    # Convert to pixel coordinates
                    pred_pixel = pred_coords_np * image_size
                    gt_pixel = gt_coords_np * image_size

                    # Calculate Euclidean distances
                    distances = np.sqrt(np.sum((pred_pixel - gt_pixel) ** 2, axis=1))
                    val_pixel_errors.extend(distances)

        # Calculate metrics
        avg_train_loss = train_loss_total / len(train_loader)
        avg_val_loss = val_loss_total / len(val_loader)
        mre = np.mean(val_pixel_errors) if val_pixel_errors else float('inf')

        # Calculate SDR (Success Detection Rate) - percentage of landmarks with error < 2px
        sdr_2px = np.mean(np.array(val_pixel_errors) < 2.0) * 100 if val_pixel_errors else 0.0
        sdr_4px = np.mean(np.array(val_pixel_errors) < 4.0) * 100 if val_pixel_errors else 0.0

        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(".6f")
        print(".6f")
        print(".2f")
        print(".1f")
        print(".1f")
        # Save best model
        if mre < best_mre:
            best_mre = mre
            best_val_loss = avg_val_loss
            patience_counter = 0

            checkpoint_path = os.path.join(output_dir, 'hrnet_31_heatmap_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'mre': mre,
                'config': {
                    'image_size': image_size,
                    'heatmap_size': heatmap_size,
                    'hrnet_variant': 'hrnet_w32'
                }
            }, checkpoint_path)
            print(".6f")
        # Early stopping
        patience_counter += 1
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

        # Scheduler step
        scheduler.step()

    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print("="*80)
    print(".2f")
    print(f"Model saved to: {os.path.join(output_dir, 'hrnet_31_heatmap_best.pth')}")

    return os.path.join(output_dir, 'hrnet_31_heatmap_best.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train high-accuracy 31-landmark heatmap model')
    parser.add_argument('--annotations', type=str, default='annotations_31.json',
                        help='Path to annotations file')
    parser.add_argument('--images', type=str, default='Aariz/train/Cephalograms',
                        help='Path to images directory')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate')

    args = parser.parse_args()

    train_31_heatmap_model(
        annotations_file=args.annotations,
        images_dir=args.images,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
