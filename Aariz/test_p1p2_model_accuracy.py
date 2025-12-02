"""
Test P1/P2 Heatmap Model Accuracy - Ground Truth Comparison
Tests the trained model on dataset images and compares with ground truth
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
from tqdm import tqdm

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
        pretrained=False,
        output_size=heatmap_size
    )

    # Load with strict=False to handle model structure differences
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()

    try:
        model.load_state_dict(state_dict, strict=False)
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"⚠️ Warning: Error loading state dict: {e}")
        # Try to load only matching keys
        filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
        model.load_state_dict(filtered_state, strict=False)
        print(f"✓ Loaded {len(filtered_state)}/{len(state_dict)} matching keys")

    model = model.to(device)
    model.eval()

    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"  - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  - Best pixel error: {checkpoint.get('pixel_error', 0):.2f} px")
    print(f"  - Val loss: {checkpoint.get('val_loss', 0):.6f}")

    return model, image_size, heatmap_size


def test_model_accuracy(model, annotations_file, images_dir, output_dir='accuracy_test_results',
                       image_size=768, heatmap_size=192, device='cuda', num_samples=10):
    """Test model accuracy on dataset with detailed ground truth comparison"""

    print("\n" + "="*80)
    print("🧪 TESTING P1/P2 HEATMAP MODEL ACCURACY")
    print("="*80)

    os.makedirs(output_dir, exist_ok=True)

    # Create dataset (no augmentation for testing)
    full_dataset = P1P2HeatmapDataset(
        annotations_file, images_dir,
        image_size=image_size,
        heatmap_size=heatmap_size,
        augment=False
    )

    # Use validation set (20% of data) or first N samples
    if len(full_dataset) > 50:  # If we have enough data, use validation split
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        _, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        dataset = val_dataset
        print(f"✓ Using validation set: {len(dataset)} samples (20% of total)")
    else:
        # Use first N samples
        dataset = torch.utils.data.Subset(full_dataset, range(min(num_samples, len(full_dataset))))
        print(f"✓ Using first {len(dataset)} samples from dataset")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    errors = []
    p1_errors = []
    p2_errors = []
    detailed_results = []

    print(f"\n🔍 Testing on {len(dataset)} samples...")
    print("-"*80)

    for idx, (images, gt_heatmaps, gt_coords) in enumerate(tqdm(dataloader, desc="Testing")):
        images = images.to(device)
        gt_coords = gt_coords.to(device)

        with torch.no_grad():
            # Predict
            pred_heatmaps = model(images)
            pred_coords = model.extract_coordinates(pred_heatmaps)

        # Calculate pixel errors
        pred_px = pred_coords.cpu().numpy()[0] * image_size
        gt_px = gt_coords.cpu().numpy()[0] * image_size

        p1_error = np.sqrt((pred_px[0] - gt_px[0])**2 + (pred_px[1] - gt_px[1])**2)
        p2_error = np.sqrt((pred_px[2] - gt_px[2])**2 + (pred_px[3] - gt_px[3])**2)
        avg_error = (p1_error + p2_error) / 2

        errors.append(avg_error)
        p1_errors.append(p1_error)
        p2_errors.append(p2_error)

        # Store detailed results
        result = {
            'sample_idx': idx,
            'p1_pred': {'x': float(pred_px[0]), 'y': float(pred_px[1])},
            'p1_gt': {'x': float(gt_px[0]), 'y': float(gt_px[1])},
            'p2_pred': {'x': float(pred_px[2]), 'y': float(pred_px[3])},
            'p2_gt': {'x': float(gt_px[2]), 'y': float(gt_px[3])},
            'p1_error': float(p1_error),
            'p2_error': float(p2_error),
            'avg_error': float(avg_error)
        }
        detailed_results.append(result)

        # Visualize (only for first 10 samples)
        if idx < 10:
            # Get original sample from full dataset
            if hasattr(dataset, 'indices'):
                actual_idx = dataset.indices[idx]
                sample = full_dataset.samples[actual_idx]
            else:
                sample = full_dataset.samples[idx]

            image_path = sample['image_path']
            image = Image.open(image_path).convert('RGB')
            orig_h, orig_w = image.size

            # Resize to match model input
            img_resized = image.resize((image_size, image_size), Image.LANCZOS)
            img_array = np.array(img_resized)

            # Draw predictions and ground truth
            img_draw = img_array.copy()

            # Ground truth: Green
            cv2.circle(img_draw, (int(gt_px[0]), int(gt_px[1])), 12, (0, 255, 0), -1)  # P1 GT
            cv2.circle(img_draw, (int(gt_px[0]), int(gt_px[1])), 14, (0, 255, 0), 2)
            cv2.circle(img_draw, (int(gt_px[2]), int(gt_px[3])), 12, (0, 255, 0), -1)  # P2 GT
            cv2.circle(img_draw, (int(gt_px[2]), int(gt_px[3])), 14, (0, 255, 0), 2)

            # Predictions: Red
            cv2.circle(img_draw, (int(pred_px[0]), int(pred_px[1])), 10, (0, 0, 255), -1)  # P1 Pred
            cv2.circle(img_draw, (int(pred_px[0]), int(pred_px[1])), 12, (0, 0, 255), 2)
            cv2.circle(img_draw, (int(pred_px[2]), int(pred_px[3])), 10, (0, 0, 255), -1)  # P2 Pred
            cv2.circle(img_draw, (int(pred_px[2]), int(pred_px[3])), 12, (0, 0, 255), 2)

            # Draw line between p1 and p2
            cv2.line(img_draw, (int(gt_px[0]), int(gt_px[1])), (int(gt_px[2]), int(gt_px[3])), (0, 255, 0), 3)  # GT line
            cv2.line(img_draw, (int(pred_px[0]), int(pred_px[1])), (int(pred_px[2]), int(pred_px[3])), (0, 0, 255), 3)  # Pred line

            # Draw error lines
            cv2.line(img_draw, (int(gt_px[0]), int(gt_px[1])), (int(pred_px[0]), int(pred_px[1])), (255, 255, 0), 2)  # P1 error
            cv2.line(img_draw, (int(gt_px[2]), int(gt_px[3])), (int(pred_px[2]), int(pred_px[3])), (255, 255, 0), 2)  # P2 error

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
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255, 0), 2)

            # Save
            output_path = os.path.join(output_dir, f"sample_{idx:02d}_{Path(filename).stem}_accuracy.png")
            cv2.imwrite(output_path, cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR))

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
    within_1px = np.sum(errors_array <= 1)
    within_2px = np.sum(errors_array <= 2)
    within_5px = np.sum(errors_array <= 5)
    within_10px = np.sum(errors_array <= 10)

    print("\n" + "="*80)
    print("📊 ACCURACY TEST RESULTS - GROUND TRUTH COMPARISON")
    print("="*80)
    print(f"\n📈 Samples tested: {len(errors)}")
    print(f"📏 Image size: {image_size}x{image_size}")
    print(f"🗺️ Heatmap size: {heatmap_size}x{heatmap_size}")

    print(f"\n{'='*80}")
    print("📊 OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"🎯 Average error: {np.mean(errors_array):.2f} px")
    print(f"📏 Median error: {np.median(errors_array):.2f} px")
    print(f"📈 Std deviation: {np.std(errors_array):.2f} px")
    print(f"✅ Min error: {np.min(errors_array):.2f} px")
    print(f"❌ Max error: {np.max(errors_array):.2f} px")

    print(f"\n{'='*80}")
    print("📈 PERCENTILES")
    print(f"{'='*80}")
    print(f"50th percentile (Median): {p50:.2f} px")
    print(f"75th percentile: {p75:.2f} px")
    print(f"90th percentile: {p90:.2f} px")
    print(f"95th percentile: {p95:.2f} px")
    print(f"99th percentile: {p99:.2f} px")

    print(f"\n{'='*80}")
    print("🎯 ERROR THRESHOLDS")
    print(f"{'='*80}")
    print(f"Within 1px: {within_1px}/{len(errors)} ({100*within_1px/len(errors):.1f}%)")
    print(f"Within 2px: {within_2px}/{len(errors)} ({100*within_2px/len(errors):.1f}%)")
    print(f"Within 5px: {within_5px}/{len(errors)} ({100*within_5px/len(errors):.1f}%)")
    print(f"Within 10px: {within_10px}/{len(errors)} ({100*within_10px/len(errors):.1f}%)")

    print(f"\n{'='*80}")
    print("📍 PER-LANDMARK STATISTICS")
    print(f"{'='*80}")
    print(f"P1 average error: {np.mean(p1_errors_array):.2f} px")
    print(f"P1 median error: {np.median(p1_errors_array):.2f} px")
    print(f"P1 std deviation: {np.std(p1_errors_array):.2f} px")
    print(f"P1 min/max: {np.min(p1_errors_array):.2f} / {np.max(p1_errors_array):.2f} px")

    print(f"\nP2 average error: {np.mean(p2_errors_array):.2f} px")
    print(f"P2 median error: {np.median(p2_errors_array):.2f} px")
    print(f"P2 std deviation: {np.std(p2_errors_array):.2f} px")
    print(f"P2 min/max: {np.min(p2_errors_array):.2f} / {np.max(p2_errors_array):.2f} px")

    print(f"\n🖼️ Visualizations saved to: {output_dir}/")
    print("="*80)

    # Save detailed results to JSON
    results = {
        'test_info': {
            'samples_tested': len(errors),
            'image_size': image_size,
            'heatmap_size': heatmap_size,
            'model_checkpoint': str(Path(checkpoint_path).name)
        },
        'overall_stats': {
            'avg_error': float(np.mean(errors_array)),
            'median_error': float(np.median(errors_array)),
            'std_error': float(np.std(errors_array)),
            'min_error': float(np.min(errors_array)),
            'max_error': float(np.max(errors_array)),
        },
        'percentiles': {
            'p50': float(p50),
            'p75': float(p75),
            'p90': float(p90),
            'p95': float(p95),
            'p99': float(p99),
        },
        'error_thresholds': {
            'within_1px': int(within_1px),
            'within_2px': int(within_2px),
            'within_5px': int(within_5px),
            'within_10px': int(within_10px),
            'pct_within_1px': float(100*within_1px/len(errors)),
            'pct_within_2px': float(100*within_2px/len(errors)),
            'pct_within_5px': float(100*within_5px/len(errors)),
            'pct_within_10px': float(100*within_10px/len(errors)),
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
        'detailed_results': detailed_results
    }

    results_path = os.path.join(output_dir, 'accuracy_test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Detailed results saved to: {results_path}")

    # Save summary text file
    summary_path = os.path.join(output_dir, 'accuracy_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("P1/P2 HEATMAP MODEL ACCURACY TEST SUMMARY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Samples tested: {len(errors)}\n")
        f.write(f"Average error: {np.mean(errors_array):.2f} px\n")
        f.write(f"Median error: {np.median(errors_array):.2f} px\n")
        f.write(f"Within 5px: {100*within_5px/len(errors):.1f}%\n")
        f.write(f"Within 10px: {100*within_10px/len(errors):.1f}%\n")
        f.write(f"Min error: {np.min(errors_array):.2f} px\n")
        f.write(f"Max error: {np.max(errors_array):.2f} px\n")
        f.write("\nVisualizations saved in this directory\n")
        f.write("="*80 + "\n")

    print(f"📄 Summary saved to: {summary_path}")

    return results


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    checkpoint_path = 'models/hrnet_p1p2_heatmap_best.pth'

    if not Path(checkpoint_path).exists():
        print(f"❌ ERROR: Model not found: {checkpoint_path}")
        print("Please train the model first using: python train_p1_p2_heatmap.py")
        sys.exit(1)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️ Using device: {device}")
    if device == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")

    model, image_size, heatmap_size = load_model(checkpoint_path, device)

    print(f"\n🔧 Model configuration:")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Heatmap size: {heatmap_size}x{heatmap_size}")

    results = test_model_accuracy(
        model=model,
        annotations_file='annotations_p1_p2.json',
        images_dir='Aariz/train/Cephalograms',
        image_size=image_size,
        heatmap_size=heatmap_size,
        device=device,
        num_samples=20  # Test on 20 samples
    )

    print("\n" + "="*80)
    print("🎯 FINAL ASSESSMENT")
    print("="*80)
    avg_error = results['overall_stats']['avg_error']
    pct_5px = results['error_thresholds']['pct_within_5px']
    pct_10px = results['error_thresholds']['pct_within_10px']

    if avg_error < 2 and pct_5px > 95:
        print("🏆 EXCELLENT! Model accuracy is outstanding!")
        print("   - Average error < 2px")
        print("   - >95% within 5px")
    elif avg_error < 5 and pct_10px > 95:
        print("✅ VERY GOOD! Model meets production standards!")
        print("   - Average error < 5px")
        print("   - >95% within 10px")
    elif avg_error < 10:
        print("👍 GOOD! Model is acceptable for use!")
        print("   - Average error < 10px")
    else:
        print("⚠️ WARNING! Model may need improvement!")
        print("   - Average error > 10px")

    print(f"\n📊 Key Metrics:")
    print(f"  - Average Error: {avg_error:.2f} px")
    print(f"  - Median Error: {results['overall_stats']['median_error']:.2f} px")
    print(f"  - Samples within 5px: {pct_5px:.1f}%")
    print(f"  - Samples within 10px: {pct_10px:.1f}%")
    print("="*80)