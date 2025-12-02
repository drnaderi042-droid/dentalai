"""
Test the fine-tuned P1/P2 model and compare with ground truth.
"""

import os
import json
import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from collections import defaultdict

# Import model
sys.path.append(str(Path(__file__).parent))
from finetune_p1_p2_cldetection import P1P2ModelWithCLDetectionBackbone, P1P2DatasetFromJSON

# Change to script directory
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)


def load_model(checkpoint_path='checkpoint_p1_p2_cldetection.pth', 
               cldetection_model_path=None,
               device='cuda'):
    """Load the trained P1/P2 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get model parameters from checkpoint
    image_size = checkpoint.get('image_size', 1024)
    cldetection_path = checkpoint.get('cldetection_model_path', cldetection_model_path)
    
    if cldetection_path is None:
        cldetection_path = r"C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth"
    
    # Check which backbone was used by examining state_dict keys
    state_dict = checkpoint['model_state_dict']
    has_resnet_backbone = any('backbone.0.weight' in k or 'backbone.4.0.conv1.weight' in k for k in state_dict.keys())
    has_cldetection_backbone = any('backbone.stage' in k or 'backbone.stem' in k for k in state_dict.keys())
    
    print(f"Detecting backbone type from checkpoint...")
    if has_resnet_backbone:
        print(f"  Detected: ResNet18 backbone (from training)")
        # Force use ResNet18 fallback
        use_cldetection = False
    elif has_cldetection_backbone:
        print(f"  Detected: CLdetection2023 backbone")
        use_cldetection = True
    else:
        print(f"  Unknown backbone, trying CLdetection2023 first...")
        use_cldetection = True
    
    # Try to create model with detected backbone
    if use_cldetection:
        try:
            model = P1P2ModelWithCLDetectionBackbone(
                cldetection_model_path=cldetection_path,
                device=device,
                freeze_backbone=False
            )
            # Try loading
            model.load_state_dict(state_dict, strict=False)
            print(f"[OK] Loaded with CLdetection2023 backbone")
        except Exception as e:
            print(f"[WARNING] Failed to load with CLdetection2023: {e}")
            print(f"  Falling back to ResNet18...")
            use_cldetection = False
    
    if not use_cldetection:
        # Create model with ResNet18 (fallback)
        model = P1P2ModelWithCLDetectionBackbone(
            cldetection_model_path=None,  # Force fallback
            device=device,
            freeze_backbone=False
        )
        # Load weights (with strict=False to handle missing CLdetection keys)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[WARNING] Missing keys: {len(missing_keys)} (expected for ResNet18)")
        if unexpected_keys:
            print(f"[WARNING] Unexpected keys: {len(unexpected_keys)}")
        print(f"[OK] Loaded with ResNet18 backbone")
    
    model = model.to(device)
    model.eval()
    
    print(f"\n[OK] Model loaded from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.6f}")
    print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    print(f"  Image Size: {image_size}x{image_size}")
    
    return model, image_size


def predict_p1_p2(model, image_path, image_size=1024, device='cuda'):
    """Predict p1 and p2 on an image."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, (image_size, image_size))
    
    # Normalize (CLdetection2023 normalization)
    image_tensor = torch.FloatTensor(image_resized).permute(2, 0, 1)  # [C, H, W]
    mean = torch.tensor([121.25, 121.25, 121.25]).view(3, 1, 1)
    std = torch.tensor([76.5, 76.5, 76.5]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    
    # Output is normalized [0, 1], convert to pixel coordinates
    output = output.cpu().numpy()[0]  # [p1_x, p1_y, p2_x, p2_y]
    
    # Convert to pixel coordinates in resized image
    p1_x = output[0] * image_size
    p1_y = output[1] * image_size
    p2_x = output[2] * image_size
    p2_y = output[3] * image_size
    
    # Scale back to original image size
    scale_x = orig_w / image_size
    scale_y = orig_h / image_size
    
    p1 = {
        'x': p1_x * scale_x,
        'y': p1_y * scale_y
    }
    
    p2 = {
        'x': p2_x * scale_x,
        'y': p2_y * scale_y
    }
    
    return p1, p2


def load_ground_truth(annotations_json='annotations_p1_p2.json', image_dir='Aariz/train/Cephalograms'):
    """Load ground truth annotations from JSON file."""
    with open(annotations_json, 'r', encoding='utf-8') as f:
        annotations_data = json.load(f)
    
    image_dir = Path(image_dir)
    gt_data = {}
    
    for item in annotations_data:
        filename = item.get('file_upload', '')
        if not filename:
            continue
        
        # Extract p1 and p2
        p1 = None
        p2 = None
        
        if 'annotations' in item and len(item['annotations']) > 0:
            result = item['annotations'][0].get('result', [])
            for r in result:
                if r.get('type') == 'keypointlabels':
                    value = r.get('value', {})
                    labels = value.get('keypointlabels', [])
                    x = value.get('x', 0)  # Percentage (0-100)
                    y = value.get('y', 0)   # Percentage (0-100)
                    
                    if 'p1' in labels:
                        p1 = {'x': x, 'y': y}
                    elif 'p2' in labels:
                        p2 = {'x': x, 'y': y}
        
        if p1 and p2:
            # Find image path to get original size
            image_id = Path(filename).stem
            image_path = None
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                potential_path = image_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    image_path = potential_path
                    break
            
            if image_path and image_path.exists():
                # Load image to get size
                img = cv2.imread(str(image_path))
                if img is not None:
                    h, w = img.shape[:2]
                    # Convert percentage to pixel coordinates
                    gt_data[image_id] = {
                        'p1': {
                            'x': p1['x'] * w / 100.0,
                            'y': p1['y'] * h / 100.0
                        },
                        'p2': {
                            'x': p2['x'] * w / 100.0,
                            'y': p2['y'] * h / 100.0
                        },
                        'image_path': image_path,
                        'image_size': (h, w)
                    }
    
    return gt_data


def calculate_error(pred, gt):
    """Calculate pixel error between prediction and ground truth."""
    error_p1 = np.sqrt((pred['p1']['x'] - gt['p1']['x'])**2 + (pred['p1']['y'] - gt['p1']['y'])**2)
    error_p2 = np.sqrt((pred['p2']['x'] - gt['p2']['x'])**2 + (pred['p2']['y'] - gt['p2']['y'])**2)
    avg_error = (error_p1 + error_p2) / 2.0
    return error_p1, error_p2, avg_error


def test_model(model, annotations_json='annotations_p1_p2.json', 
               image_dir='Aariz/train/Cephalograms',
               image_size=1024, device='cuda', num_images=None):
    """Test model on all images and compare with ground truth."""
    
    print("="*80)
    print("Testing P1/P2 Model")
    print("="*80)
    
    # Load ground truth
    print("\nLoading ground truth annotations...")
    gt_data = load_ground_truth(annotations_json, image_dir)
    print(f"[OK] Loaded {len(gt_data)} ground truth annotations")
    
    if len(gt_data) == 0:
        print("ERROR: No ground truth data found!")
        return
    
    # Limit number of images if specified
    image_ids = list(gt_data.keys())
    if num_images:
        image_ids = image_ids[:num_images]
        print(f"Testing on {len(image_ids)} images (limited from {len(gt_data)})")
    else:
        print(f"Testing on all {len(image_ids)} images")
    
    # Test on each image
    results = []
    errors_p1 = []
    errors_p2 = []
    avg_errors = []
    
    print("\nRunning inference...")
    for image_id in tqdm(image_ids):
        gt = gt_data[image_id]
        image_path = gt['image_path']
        
        try:
            # Predict
            pred_p1, pred_p2 = predict_p1_p2(model, image_path, image_size, device)
            
            pred = {
                'p1': pred_p1,
                'p2': pred_p2
            }
            
            # Calculate errors
            error_p1, error_p2, avg_error = calculate_error(pred, gt)
            
            # Debug: print first image details
            if len(results) == 0:
                print(f"\n[DEBUG] First image: {image_id}")
                print(f"  GT: p1=({gt['p1']['x']:.1f}, {gt['p1']['y']:.1f}), p2=({gt['p2']['x']:.1f}, {gt['p2']['y']:.1f})")
                print(f"  Pred: p1=({pred_p1['x']:.1f}, {pred_p1['y']:.1f}), p2=({pred_p2['x']:.1f}, {pred_p2['y']:.1f})")
                print(f"  Image size: {gt['image_size']}")
                print(f"  Errors: p1={error_p1:.1f}px, p2={error_p2:.1f}px")
            
            errors_p1.append(error_p1)
            errors_p2.append(error_p2)
            avg_errors.append(avg_error)
            
            results.append({
                'image_id': image_id,
                'pred': pred,
                'gt': gt,
                'error_p1': error_p1,
                'error_p2': error_p2,
                'avg_error': avg_error
            })
            
        except Exception as e:
            print(f"\n[WARNING] Error processing {image_id}: {e}")
            continue
    
    if len(results) == 0:
        print("ERROR: No successful predictions!")
        return
    
    # Calculate statistics
    errors_p1 = np.array(errors_p1)
    errors_p2 = np.array(errors_p2)
    avg_errors = np.array(avg_errors)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    
    print(f"\nTotal images tested: {len(results)}")
    
    print(f"\n{'Metric':<20} {'P1':<15} {'P2':<15} {'Average':<15}")
    print("-" * 65)
    print(f"{'Mean Error (px)':<20} {np.mean(errors_p1):<15.2f} {np.mean(errors_p2):<15.2f} {np.mean(avg_errors):<15.2f}")
    print(f"{'Median Error (px)':<20} {np.median(errors_p1):<15.2f} {np.median(errors_p2):<15.2f} {np.median(avg_errors):<15.2f}")
    print(f"{'Std Dev (px)':<20} {np.std(errors_p1):<15.2f} {np.std(errors_p2):<15.2f} {np.std(avg_errors):<15.2f}")
    print(f"{'Min Error (px)':<20} {np.min(errors_p1):<15.2f} {np.min(errors_p2):<15.2f} {np.min(avg_errors):<15.2f}")
    print(f"{'Max Error (px)':<20} {np.max(errors_p1):<15.2f} {np.max(errors_p2):<15.2f} {np.max(avg_errors):<15.2f}")
    
    # Accuracy at different thresholds
    thresholds = [5, 10, 20, 30, 50]
    print(f"\n{'Accuracy at Threshold':<25} {'P1':<15} {'P2':<15} {'Average':<15}")
    print("-" * 70)
    for threshold in thresholds:
        acc_p1 = np.mean(errors_p1 <= threshold) * 100
        acc_p2 = np.mean(errors_p2 <= threshold) * 100
        acc_avg = np.mean(avg_errors <= threshold) * 100
        print(f"{f'< {threshold}px':<25} {acc_p1:<15.1f}% {acc_p2:<15.1f}% {acc_avg:<15.1f}%")
    
    # Best and worst predictions
    best_idx = np.argmin(avg_errors)
    worst_idx = np.argmax(avg_errors)
    
    print(f"\n[BEST] Best prediction:")
    print(f"   Image: {results[best_idx]['image_id']}")
    print(f"   Error: {results[best_idx]['avg_error']:.2f} px")
    print(f"   P1 error: {results[best_idx]['error_p1']:.2f} px")
    print(f"   P2 error: {results[best_idx]['error_p2']:.2f} px")
    
    print(f"\n[WORST] Worst prediction:")
    print(f"   Image: {results[worst_idx]['image_id']}")
    print(f"   Error: {results[worst_idx]['avg_error']:.2f} px")
    print(f"   P1 error: {results[worst_idx]['error_p1']:.2f} px")
    print(f"   P2 error: {results[worst_idx]['error_p2']:.2f} px")
    
    # Save results to JSON
    results_file = 'test_p1_p2_results.json'
    output_data = {
        'model_checkpoint': 'checkpoint_p1_p2_cldetection.pth',
        'image_size': image_size,
        'num_images_tested': len(results),
        'statistics': {
            'p1': {
                'mean': float(np.mean(errors_p1)),
                'median': float(np.median(errors_p1)),
                'std': float(np.std(errors_p1)),
                'min': float(np.min(errors_p1)),
                'max': float(np.max(errors_p1))
            },
            'p2': {
                'mean': float(np.mean(errors_p2)),
                'median': float(np.median(errors_p2)),
                'std': float(np.std(errors_p2)),
                'min': float(np.min(errors_p2)),
                'max': float(np.max(errors_p2))
            },
            'average': {
                'mean': float(np.mean(avg_errors)),
                'median': float(np.median(avg_errors)),
                'std': float(np.std(avg_errors)),
                'min': float(np.min(avg_errors)),
                'max': float(np.max(avg_errors))
            }
        },
        'accuracy_at_thresholds': {
            str(t): {
                'p1': float(np.mean(errors_p1 <= t) * 100),
                'p2': float(np.mean(errors_p2 <= t) * 100),
                'average': float(np.mean(avg_errors <= t) * 100)
            }
            for t in thresholds
        },
        'detailed_results': [
            {
                'image_id': r['image_id'],
                'error_p1': float(r['error_p1']),
                'error_p2': float(r['error_p2']),
                'avg_error': float(r['avg_error']),
                'pred_p1': r['pred']['p1'],
                'pred_p2': r['pred']['p2'],
                'gt_p1': r['gt']['p1'],
                'gt_p2': r['gt']['p2']
            }
            for r in results
        ]
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Detailed results saved to: {results_file}")
    
    # Visualize best and worst
    visualize_predictions(results[best_idx], results[worst_idx], image_dir)
    
    return results, output_data


def visualize_predictions(best_result, worst_result, image_dir):
    """Visualize best and worst predictions."""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        for idx, (result, title) in enumerate([(best_result, 'Best Prediction'), (worst_result, 'Worst Prediction')]):
            image_path = result['gt']['image_path']
            image = cv2.imread(str(image_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            ax = axes[idx]
            ax.imshow(image)
            ax.set_title(f"{title}\n{result['image_id']}\nError: {result['avg_error']:.2f}px", 
                        fontsize=12, fontweight='bold')
            
            # Ground truth (green)
            gt_p1 = result['gt']['p1']
            gt_p2 = result['gt']['p2']
            ax.plot(gt_p1['x'], gt_p1['y'], 'go', markersize=10, label='GT p1', alpha=0.7)
            ax.plot(gt_p2['x'], gt_p2['y'], 'go', markersize=10, label='GT p2', alpha=0.7)
            ax.plot([gt_p1['x'], gt_p2['x']], [gt_p1['y'], gt_p2['y']], 'g-', linewidth=2, alpha=0.5)
            
            # Prediction (red)
            pred_p1 = result['pred']['p1']
            pred_p2 = result['pred']['p2']
            ax.plot(pred_p1['x'], pred_p1['y'], 'ro', markersize=10, label='Pred p1', alpha=0.7)
            ax.plot(pred_p2['x'], pred_p2['y'], 'ro', markersize=10, label='Pred p2', alpha=0.7)
            ax.plot([pred_p1['x'], pred_p2['x']], [pred_p1['y'], pred_p2['y']], 'r-', linewidth=2, alpha=0.5)
            
            # Error lines
            ax.plot([gt_p1['x'], pred_p1['x']], [gt_p1['y'], pred_p1['y']], 'y--', linewidth=1, alpha=0.5)
            ax.plot([gt_p2['x'], pred_p2['x']], [gt_p2['y'], pred_p2['y']], 'y--', linewidth=1, alpha=0.5)
            
            ax.legend()
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('test_p1_p2_visualization.png', dpi=150, bbox_inches='tight')
        print(f"[OK] Visualization saved to: test_p1_p2_visualization.png")
        plt.close()
        
    except Exception as e:
        print(f"[WARNING] Could not create visualization: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test P1/P2 model')
    parser.add_argument('--checkpoint', type=str, default='checkpoint_p1_p2_cldetection.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--annotations', type=str, default='annotations_p1_p2.json',
                       help='Path to annotations JSON file')
    parser.add_argument('--image-dir', type=str, default='Aariz/train/Cephalograms',
                       help='Directory containing images')
    parser.add_argument('--num-images', type=int, default=None,
                       help='Number of images to test (None = all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"[DEVICE] Using: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}\n")
    
    # Load model
    try:
        model, image_size = load_model(args.checkpoint, device=device)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Test
    try:
        results, output_data = test_model(
            model,
            annotations_json=args.annotations,
            image_dir=args.image_dir,
            image_size=image_size,
            device=device,
            num_images=args.num_images
        )
        print("\n" + "="*80)
        print("Testing complete!")
        print("="*80)
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()

