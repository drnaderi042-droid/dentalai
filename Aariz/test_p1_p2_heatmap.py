"""
Test heatmap-based P1/P2 model
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    image_size = checkpoint.get('image_size', 768)
    heatmap_size = checkpoint.get('heatmap_size', 192)
    
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=True,  # Must match training configuration!
        output_size=heatmap_size
    )
    
    # Load with strict=False to handle model structure differences
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Try to load with strict=False first
    try:
        model.load_state_dict(state_dict, strict=False)
        print("  - Model loaded (some keys may be missing/unused)")
    except Exception as e:
        print(f"  - Warning: Error loading state dict: {e}")
        # Try to load only matching keys
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"  - Loaded {len(filtered_state)}/{len(state_dict)} matching keys")
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"  - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  - Pixel Error: {checkpoint.get('pixel_error', 0):.2f} px")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 0):.6f}")
    
    return model, image_size, heatmap_size


def test_model(model, annotations_file, images_dir, output_dir='test_results_heatmap', 
                image_size=1024, heatmap_size=256, device='cuda', use_validation_set=True):
    """Test model on dataset with ground truth comparison"""
    
    print("\n" + "="*70)
    print("TESTING HEATMAP MODEL - GROUND TRUTH COMPARISON")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset (no augmentation for testing)
    full_dataset = P1P2HeatmapDataset(
        annotations_file, images_dir,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False
    )
    
    # Use same split as training (80/20)
    if use_validation_set:
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        dataset = val_dataset
        print(f"Using validation set: {len(dataset)} samples (20% of total)")
    else:
        dataset = full_dataset
        print(f"Using full dataset: {len(dataset)} samples")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    errors = []
    p1_errors = []
    p2_errors = []
    
    print(f"\nTesting on {len(dataset)} samples...")
    
    for idx, (images, gt_heatmaps, gt_coords) in enumerate(dataloader):
        images = images.to(device)
        gt_coords = gt_coords.to(device)
        
        with torch.no_grad():
            # Predict
            pred_heatmaps = model(images)
            pred_coords = model.extract_coordinates(pred_heatmaps)
        
        # Debug: Check coordinate ranges
        if idx == 0:
            print(f"\nDebug - First sample:")
            print(f"  Pred coords (normalized): {pred_coords.cpu().numpy()[0]}")
            print(f"  GT coords (normalized): {gt_coords.cpu().numpy()[0]}")
            print(f"  Heatmap shape: {pred_heatmaps.shape}")
            print(f"  Heatmap min/max: {pred_heatmaps.min().item():.4f} / {pred_heatmaps.max().item():.4f}")
        
        # Calculate pixel errors
        pred_px = pred_coords.cpu().numpy()[0] * image_size
        gt_px = gt_coords.cpu().numpy()[0] * image_size
        
        # Debug: Check pixel coordinates
        if idx == 0:
            print(f"  Pred coords (pixels): {pred_px}")
            print(f"  GT coords (pixels): {gt_px}")
        
        p1_error = np.sqrt((pred_px[0] - gt_px[0])**2 + (pred_px[1] - gt_px[1])**2)
        p2_error = np.sqrt((pred_px[2] - gt_px[2])**2 + (pred_px[3] - gt_px[3])**2)
        avg_error = (p1_error + p2_error) / 2
        
        errors.append(avg_error)
        p1_errors.append(p1_error)
        p2_errors.append(p2_error)
        
        # Visualize (only for first 20 samples to save time)
        if idx < 20:
            # Get original sample from full dataset
            if use_validation_set and hasattr(dataset, 'indices'):
                # Get the actual index from validation split
                actual_idx = dataset.indices[idx]
                sample = full_dataset.samples[actual_idx]
            else:
                # For full dataset or if indices not available
                sample = full_dataset.samples[idx]
            
            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
            orig_h, orig_w = image.size
            
            # Resize to match model input
            img_resized = image.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img_resized)
            
            # Draw predictions (in resized coordinates)
            p1_pred = (int(pred_px[0]), int(pred_px[1]))
            p2_pred = (int(pred_px[2]), int(pred_px[3]))
            p1_gt = (int(gt_px[0]), int(gt_px[1]))
            p2_gt = (int(gt_px[2]), int(gt_px[3]))
            
            # Draw on image
            img_draw = img_array.copy()
            # Ground truth: Green
            cv2.circle(img_draw, p1_gt, 10, (0, 255, 0), -1)
            cv2.circle(img_draw, p1_gt, 12, (0, 255, 0), 2)
            cv2.circle(img_draw, p2_gt, 10, (0, 255, 0), -1)
            cv2.circle(img_draw, p2_gt, 12, (0, 255, 0), 2)
            
            # Predictions: Red
            cv2.circle(img_draw, p1_pred, 8, (0, 0, 255), -1)
            cv2.circle(img_draw, p1_pred, 10, (0, 0, 255), 2)
            cv2.circle(img_draw, p2_pred, 8, (0, 0, 255), -1)
            cv2.circle(img_draw, p2_pred, 10, (0, 0, 255), 2)
            
            # Draw line between p1 and p2
            cv2.line(img_draw, p1_gt, p2_gt, (0, 255, 0), 2)
            cv2.line(img_draw, p1_pred, p2_pred, (0, 0, 255), 2)
            
            # Draw error line
            cv2.line(img_draw, p1_gt, p1_pred, (255, 255, 0), 1)
            cv2.line(img_draw, p2_gt, p2_pred, (255, 255, 0), 1)
            
            # Add text
            filename = Path(image_path).name
            cv2.putText(img_draw, f"P1 Error: {p1_error:.1f}px", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_draw, f"P2 Error: {p2_error:.1f}px", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_draw, f"Avg Error: {avg_error:.1f}px", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(img_draw, "Green: Ground Truth", (10, image_size - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_draw, "Red: Prediction", (10, image_size - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Save
            output_path = os.path.join(output_dir, f"{Path(filename).stem}_pred.png")
            cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))
        
        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(dataset)}] Avg error: {np.mean(errors):.2f} px")
    
    # Statistics
    errors_array = np.array(errors)
    p1_errors_array = np.array(p1_errors)
    p2_errors_array = np.array(p2_errors)
    
    # Calculate percentiles
    p50 = np.percentile(errors_array, 50)
    p75 = np.percentile(errors_array, 75)
    p90 = np.percentile(errors_array, 90)
    p95 = np.percentile(errors_array, 95)
    p99 = np.percentile(errors_array, 99)
    
    # Count samples within error thresholds
    within_5px = np.sum(errors_array <= 5)
    within_10px = np.sum(errors_array <= 10)
    within_20px = np.sum(errors_array <= 20)
    
    print("\n" + "="*70)
    print("TEST RESULTS - GROUND TRUTH COMPARISON")
    print("="*70)
    print(f"\nSamples tested: {len(errors)}")
    print(f"Image size: {image_size}x{image_size}")
    print(f"Heatmap size: {heatmap_size}x{heatmap_size}")
    
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Average error: {np.mean(errors_array):.2f} px")
    print(f"Median error: {np.median(errors_array):.2f} px")
    print(f"Std deviation: {np.std(errors_array):.2f} px")
    print(f"Min error: {np.min(errors_array):.2f} px")
    print(f"Max error: {np.max(errors_array):.2f} px")
    
    print(f"\n{'='*70}")
    print("PERCENTILES")
    print(f"{'='*70}")
    print(f"50th percentile (Median): {p50:.2f} px")
    print(f"75th percentile: {p75:.2f} px")
    print(f"90th percentile: {p90:.2f} px")
    print(f"95th percentile: {p95:.2f} px")
    print(f"99th percentile: {p99:.2f} px")
    
    print(f"\n{'='*70}")
    print("ERROR THRESHOLDS")
    print(f"{'='*70}")
    print(f"Within 5px: {within_5px}/{len(errors)} ({100*within_5px/len(errors):.1f}%)")
    print(f"Within 10px: {within_10px}/{len(errors)} ({100*within_10px/len(errors):.1f}%)")
    print(f"Within 20px: {within_20px}/{len(errors)} ({100*within_20px/len(errors):.1f}%)")
    
    print(f"\n{'='*70}")
    print("PER-LANDMARK STATISTICS")
    print(f"{'='*70}")
    print(f"P1 average error: {np.mean(p1_errors_array):.2f} px")
    print(f"P1 median error: {np.median(p1_errors_array):.2f} px")
    print(f"P1 std deviation: {np.std(p1_errors_array):.2f} px")
    print(f"P1 min/max: {np.min(p1_errors_array):.2f} / {np.max(p1_errors_array):.2f} px")
    
    print(f"\nP2 average error: {np.mean(p2_errors_array):.2f} px")
    print(f"P2 median error: {np.median(p2_errors_array):.2f} px")
    print(f"P2 std deviation: {np.std(p2_errors_array):.2f} px")
    print(f"P2 min/max: {np.min(p2_errors_array):.2f} / {np.max(p2_errors_array):.2f} px")
    
    print(f"\nVisualizations saved to: {output_dir}/")
    print("="*70)
    
    # Save results to JSON
    results = {
        'samples_tested': len(errors),
        'image_size': image_size,
        'heatmap_size': heatmap_size,
        'avg_error': float(np.mean(errors_array)),
        'median_error': float(np.median(errors_array)),
        'std_error': float(np.std(errors_array)),
        'min_error': float(np.min(errors_array)),
        'max_error': float(np.max(errors_array)),
        'percentiles': {
            'p50': float(p50),
            'p75': float(p75),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99),
        },
        'error_thresholds': {
            'within_5px': int(within_5px),
            'within_10px': int(within_10px),
            'within_20px': int(within_20px),
            'pct_within_5px': float(100*within_5px/len(errors)),
            'pct_within_10px': float(100*within_10px/len(errors)),
            'pct_within_20px': float(100*within_20px/len(errors)),
        },
        'p1_stats': {
            'avg_error': float(np.mean(p1_errors_array)),
            'median_error': float(np.median(p1_errors_array)),
            'std_error': float(np.std(p1_errors_array)),
            'min_error': float(np.min(p1_errors_array)),
            'max_error': float(np.max(p1_errors_array)),
        },
        'p2_stats': {
            'avg_error': float(np.mean(p2_errors_array)),
            'median_error': float(np.median(p2_errors_array)),
            'std_error': float(np.std(p2_errors_array)),
            'min_error': float(np.min(p2_errors_array)),
            'max_error': float(np.max(p2_errors_array)),
        },
    }
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    checkpoint_path = 'models/hrnet_p1p2_heatmap_best.pth'
    
    if not Path(checkpoint_path).exists():
        print(f"ERROR: Model not found: {checkpoint_path}")
        print("Please train the model first using: python train_p1_p2_heatmap.py")
        sys.exit(1)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model, image_size, heatmap_size = load_model(checkpoint_path, device)
    
    print(f"\nModel configuration:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Heatmap size: {heatmap_size}x{heatmap_size}")
    
    results = test_model(
        model=model,
        annotations_file='annotations_p1_p2.json',
        images_dir='Aariz/train/Cephalograms',
        image_size=image_size,
        heatmap_size=heatmap_size,
        device=device,
        use_validation_set=True  # Test on validation set (20% of data)
    )
    
    print("\n" + "="*70)
    print("FINAL ASSESSMENT")
    print("="*70)
    avg_error = results['avg_error']
    pct_10px = results['error_thresholds']['pct_within_10px']
    
    if avg_error < 5 and pct_10px > 90:
        print("[EXCELLENT] Average error < 5px and >90% within 10px")
    elif avg_error < 10 and pct_10px > 80:
        print("[GREAT] Average error < 10px and >80% within 10px")
    elif avg_error < 15:
        print("[GOOD] Average error < 15px")
    else:
        print("[WARNING] Average error > 15px - may need more training or data")
    
    print(f"\nKey Metrics:")
    print(f"  - Average Error: {avg_error:.2f} px")
    print(f"  - Median Error: {results['median_error']:.2f} px")
    print(f"  - Samples within 10px: {pct_10px:.1f}%")
    print("="*70)













