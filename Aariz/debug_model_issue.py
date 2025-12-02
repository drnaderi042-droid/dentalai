"""
بررسی دقیق مشکل مدل - چرا در تست خطا زیاد است اما در validation زمان train کم است؟
"""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import json
import cv2
from pathlib import Path
import os
import sys

sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector
from train_p1_p2_heatmap import P1P2HeatmapDataset


def load_model(checkpoint_path, device='cuda'):
    """Load trained heatmap model"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    image_size = checkpoint.get('image_size', 1024)
    heatmap_size = checkpoint.get('heatmap_size', 256)
    
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=True,  # Must match training configuration!
        output_size=heatmap_size
    )
    
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Check which keys match
    missing_keys = set(model_state.keys()) - set(state_dict.keys())
    unexpected_keys = set(state_dict.keys()) - set(model_state.keys())
    
    print(f"\n[DEBUG] Model loading:")
    print(f"  Model state keys: {len(model_state.keys())}")
    print(f"  Checkpoint keys: {len(state_dict.keys())}")
    print(f"  Missing keys: {len(missing_keys)}")
    print(f"  Unexpected keys: {len(unexpected_keys)}")
    
    if missing_keys:
        print(f"  Missing: {list(missing_keys)[:5]}...")
    if unexpected_keys:
        print(f"  Unexpected: {list(unexpected_keys)[:5]}...")
    
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  [OK] Model loaded (strict=False)")
    except Exception as e:
        print(f"  [ERROR] {e}")
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  [OK] Loaded {len(filtered_state)}/{len(state_dict)} matching keys")
    
    model = model.to(device)
    model.eval()
    
    return model, image_size, heatmap_size


def analyze_heatmaps(model, dataset, device, num_samples=5):
    """Analyze heatmap outputs in detail"""
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"\n{'='*70}")
    print("ANALYZING HEATMAP OUTPUTS")
    print(f"{'='*70}")
    
    for idx, (images, gt_heatmaps, gt_coords) in enumerate(dataloader):
        if idx >= num_samples:
            break
            
        images = images.to(device)
        gt_coords = gt_coords.to(device)
        
        with torch.no_grad():
            pred_heatmaps = model(images)
            pred_coords = model.extract_coordinates(pred_heatmaps)
        
        # Analyze heatmaps
        print(f"\n[Sample {idx+1}]")
        print(f"  Image shape: {images.shape}")
        print(f"  Pred heatmap shape: {pred_heatmaps.shape}")
        print(f"  GT heatmap shape: {gt_heatmaps.shape}")
        
        # P1 analysis
        p1_heatmap = pred_heatmaps[0, 0].cpu().numpy()
        p1_gt_heatmap = gt_heatmaps[0, 0].cpu().numpy()
        
        print(f"\n  P1 Heatmap:")
        print(f"    Pred - Min: {p1_heatmap.min():.6f}, Max: {p1_heatmap.max():.6f}, Sum: {p1_heatmap.sum():.6f}, Mean: {p1_heatmap.mean():.6f}")
        print(f"    GT   - Min: {p1_gt_heatmap.min():.6f}, Max: {p1_gt_heatmap.max():.6f}, Sum: {p1_gt_heatmap.sum():.6f}, Mean: {p1_gt_heatmap.mean():.6f}")
        
        # Find max location
        p1_max_idx = np.unravel_index(p1_heatmap.argmax(), p1_heatmap.shape)
        p1_gt_max_idx = np.unravel_index(p1_gt_heatmap.argmax(), p1_gt_heatmap.shape)
        print(f"    Pred max at: ({p1_max_idx[0]}, {p1_max_idx[1]}) = {p1_heatmap[p1_max_idx]:.6f}")
        print(f"    GT max at: ({p1_gt_max_idx[0]}, {p1_gt_max_idx[1]}) = {p1_gt_heatmap[p1_gt_max_idx]:.6f}")
        
        # P2 analysis
        p2_heatmap = pred_heatmaps[0, 1].cpu().numpy()
        p2_gt_heatmap = gt_heatmaps[0, 1].cpu().numpy()
        
        print(f"\n  P2 Heatmap:")
        print(f"    Pred - Min: {p2_heatmap.min():.6f}, Max: {p2_heatmap.max():.6f}, Sum: {p2_heatmap.sum():.6f}, Mean: {p2_heatmap.mean():.6f}")
        print(f"    GT   - Min: {p2_gt_heatmap.min():.6f}, Max: {p2_gt_heatmap.max():.6f}, Sum: {p2_gt_heatmap.sum():.6f}, Mean: {p2_gt_heatmap.mean():.6f}")
        
        # Find max location
        p2_max_idx = np.unravel_index(p2_heatmap.argmax(), p2_heatmap.shape)
        p2_gt_max_idx = np.unravel_index(p2_gt_heatmap.argmax(), p2_gt_heatmap.shape)
        print(f"    Pred max at: ({p2_max_idx[0]}, {p2_max_idx[1]}) = {p2_heatmap[p2_max_idx]:.6f}")
        print(f"    GT max at: ({p2_gt_max_idx[0]}, {p2_gt_max_idx[1]}) = {p2_gt_heatmap[p2_gt_max_idx]:.6f}")
        
        # Coordinates
        pred_coords_np = pred_coords.cpu().numpy()[0]
        gt_coords_np = gt_coords.cpu().numpy()[0]
        
        print(f"\n  Coordinates (normalized [0,1]):")
        print(f"    Pred: P1=({pred_coords_np[0]:.4f}, {pred_coords_np[1]:.4f}), P2=({pred_coords_np[2]:.4f}, {pred_coords_np[3]:.4f})")
        print(f"    GT:   P1=({gt_coords_np[0]:.4f}, {gt_coords_np[1]:.4f}), P2=({gt_coords_np[2]:.4f}, {gt_coords_np[3]:.4f})")
        
        # Pixel errors
        image_size = 1024
        pred_px = pred_coords_np * image_size
        gt_px = gt_coords_np * image_size
        
        p1_error = np.sqrt((pred_px[0] - gt_px[0])**2 + (pred_px[1] - gt_px[1])**2)
        p2_error = np.sqrt((pred_px[2] - gt_px[2])**2 + (pred_px[3] - gt_px[3])**2)
        
        print(f"\n  Pixel Errors:")
        print(f"    P1: {p1_error:.2f} px")
        print(f"    P2: {p2_error:.2f} px")
        print(f"    Avg: {(p1_error + p2_error)/2:.2f} px")
        
        # Check if heatmap is all zeros or very small
        if p1_heatmap.sum() < 1e-6:
            print(f"    [WARNING] P1 heatmap sum is near zero!")
        if p2_heatmap.sum() < 1e-6:
            print(f"    [WARNING] P2 heatmap sum is near zero!")


def compare_train_test_split():
    """Compare validation set used in training vs test set"""
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'Aariz/train/Cephalograms'
    image_size = 1024
    heatmap_size = 256
    
    # Create full dataset
    full_dataset = P1P2HeatmapDataset(
        annotations_file, images_dir,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False
    )
    
    # Use same split as training (80/20 with seed=42)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n{'='*70}")
    print("DATASET SPLIT INFO")
    print(f"{'='*70}")
    print(f"Total samples: {len(full_dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return val_dataset, full_dataset


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    checkpoint_path = 'models/hrnet_p1p2_heatmap_best.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"[ERROR] Model not found: {checkpoint_path}")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Using device: {device}")
    
    # Load model
    model, image_size, heatmap_size = load_model(checkpoint_path, device)
    
    print(f"\n[INFO] Model configuration:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Heatmap size: {heatmap_size}x{heatmap_size}")
    
    # Get validation dataset (same as training)
    val_dataset, full_dataset = compare_train_test_split()
    
    # Analyze heatmaps
    analyze_heatmaps(model, val_dataset, device, num_samples=5)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")

