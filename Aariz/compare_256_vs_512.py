"""
Compare 256x256 vs 512x512 model performance on ground truth
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
aariz_path = os.path.join(base_dir, 'Aariz')

if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

from inference import LandmarkPredictor

# Test image
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
TEST_IMAGE_PATH = os.path.join(aariz_path, "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    aariz_path, "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

# Model paths
CHECKPOINT_256_PATH = os.path.join(aariz_path, "checkpoint_best_256.pth")
CHECKPOINT_512_PATH = os.path.join(aariz_path, "checkpoints", "checkpoint_best.pth")
PIXEL_SIZE = 0.1  # mm/pixel

def load_ground_truth():
    """Load ground truth annotations"""
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    gt_landmarks = {}
    for lm in annotation['landmarks']:
        symbol = lm['symbol']
        gt_landmarks[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return gt_landmarks

def calculate_errors(predicted, ground_truth):
    """Calculate errors between predicted and ground truth"""
    errors = []
    
    for name in ground_truth.keys():
        if name in predicted:
            pred = predicted[name]
            gt = ground_truth[name]
            
            error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            error_mm = error_px * PIXEL_SIZE
            
            errors.append({
                'name': name,
                'error_px': error_px,
                'error_mm': error_mm,
                'pred': pred,
                'gt': gt,
                'diff_x': pred['x'] - gt['x'],
                'diff_y': pred['y'] - gt['y']
            })
    
    return errors

def calculate_metrics(errors):
    """Calculate MRE and SDR metrics"""
    if not errors:
        return {
            'mre_mm': 0.0,
            'median_mm': 0.0,
            'min_mm': 0.0,
            'max_mm': 0.0,
            'std_mm': 0.0,
            'sdr_1mm': 0.0,
            'sdr_1_5mm': 0.0,
            'sdr_2mm': 0.0,
            'sdr_2_5mm': 0.0,
            'sdr_3mm': 0.0,
            'sdr_4mm': 0.0
        }
    
    error_values_mm = [e['error_mm'] for e in errors]
    
    sdr_thresholds = {
        'sdr_1mm': 1.0,
        'sdr_1_5mm': 1.5,
        'sdr_2mm': 2.0,
        'sdr_2_5mm': 2.5,
        'sdr_3mm': 3.0,
        'sdr_4mm': 4.0
    }
    
    sdr_results = {}
    for key, threshold in sdr_thresholds.items():
        success = sum(1 for e_mm in error_values_mm if e_mm <= threshold)
        sdr_results[key] = (success / len(error_values_mm)) * 100
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'median_mm': float(np.median(error_values_mm)),
        'min_mm': float(np.min(error_values_mm)),
        'max_mm': float(np.max(error_values_mm)),
        'std_mm': float(np.std(error_values_mm)),
        **sdr_results
    }

def print_comparison_table(errors_256, errors_512, metrics_256, metrics_512):
    """Print detailed comparison table"""
    print("\n" + "="*120)
    print("DETAILED LANDMARK COMPARISON")
    print("="*120)
    
    # Get all landmark names
    landmark_names = sorted(set([e['name'] for e in errors_256] + [e['name'] for e in errors_512]))
    
    print(f"\n{'Landmark':<10} {'256x256 Error (mm)':<20} {'512x512 Error (mm)':<20} {'Improvement (mm)':<20} {'Better':<10}")
    print("-"*120)
    
    for name in landmark_names:
        err_256 = next((e for e in errors_256 if e['name'] == name), None)
        err_512 = next((e for e in errors_512 if e['name'] == name), None)
        
        if err_256 and err_512:
            err_256_mm = err_256['error_mm']
            err_512_mm = err_512['error_mm']
            improvement = err_256_mm - err_512_mm
            better = "512x512" if improvement > 0 else "256x256"
            
            print(f"{name:<10} {err_256_mm:<20.4f} {err_512_mm:<20.4f} {improvement:<20.4f} {better:<10}")
        elif err_256:
            print(f"{name:<10} {err_256['error_mm']:<20.4f} {'N/A':<20} {'N/A':<20} {'256 only':<10}")
        elif err_512:
            print(f"{name:<10} {'N/A':<20} {err_512['error_mm']:<20.4f} {'N/A':<20} {'512 only':<10}")

def print_metrics_comparison(metrics_256, metrics_512):
    """Print metrics comparison"""
    print("\n" + "="*120)
    print("METRICS COMPARISON")
    print("="*120)
    
    print(f"\n{'Metric':<30} {'256x256':<20} {'512x512':<20} {'Improvement':<20}")
    print("-"*90)
    
    # MRE
    mre_improvement = metrics_256['mre_mm'] - metrics_512['mre_mm']
    print(f"{'MRE (mm)':<30} {metrics_256['mre_mm']:<20.4f} {metrics_512['mre_mm']:<20.4f} {mre_improvement:<20.4f}")
    
    # Median
    median_improvement = metrics_256['median_mm'] - metrics_512['median_mm']
    print(f"{'Median Error (mm)':<30} {metrics_256['median_mm']:<20.4f} {metrics_512['median_mm']:<20.4f} {median_improvement:<20.4f}")
    
    # Min/Max
    print(f"{'Min Error (mm)':<30} {metrics_256['min_mm']:<20.4f} {metrics_512['min_mm']:<20.4f} {'-':<20}")
    print(f"{'Max Error (mm)':<30} {metrics_256['max_mm']:<20.4f} {metrics_512['max_mm']:<20.4f} {'-':<20}")
    print(f"{'Std Dev (mm)':<30} {metrics_256['std_mm']:<20.4f} {metrics_512['std_mm']:<20.4f} {'-':<20}")
    
    print("\n" + "-"*90)
    print("SUCCESS DETECTION RATE (SDR)")
    print("-"*90)
    
    sdr_keys = ['sdr_1mm', 'sdr_1_5mm', 'sdr_2mm', 'sdr_2_5mm', 'sdr_3mm', 'sdr_4mm']
    sdr_labels = ['SDR @ 1.0mm', 'SDR @ 1.5mm', 'SDR @ 2.0mm', 'SDR @ 2.5mm', 'SDR @ 3.0mm', 'SDR @ 4.0mm']
    
    for key, label in zip(sdr_keys, sdr_labels):
        sdr_256 = metrics_256[key]
        sdr_512 = metrics_512[key]
        sdr_improvement = sdr_512 - sdr_256
        print(f"{label:<30} {sdr_256:<20.2f}% {sdr_512:<20.2f}% {sdr_improvement:<20.2f}%")

def main():
    print("="*120)
    print("COMPARISON: 256x256 vs 512x512 Model Performance")
    print("="*120)
    print(f"\nTest Image: {TEST_IMAGE_ID}")
    print(f"Image Path: {TEST_IMAGE_PATH}")
    print(f"Ground Truth Path: {GROUND_TRUTH_PATH}")
    print(f"\nModel Paths:")
    print(f"  256x256: {CHECKPOINT_256_PATH}")
    print(f"  512x512: {CHECKPOINT_512_PATH}")
    
    # Check files exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\nERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\nERROR: Ground truth not found: {GROUND_TRUTH_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_256_PATH):
        print(f"\nERROR: 256x256 checkpoint not found: {CHECKPOINT_256_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_512_PATH):
        print(f"\nERROR: 512x512 checkpoint not found: {CHECKPOINT_512_PATH}")
        return
    
    # Load image and ground truth
    print(f"\nLoading image and ground truth...")
    img = Image.open(TEST_IMAGE_PATH).convert('RGB')
    img_size = img.size
    print(f"  Image size: {img_size[0]} x {img_size[1]} pixels")
    print(f"  Pixel size: {PIXEL_SIZE} mm/pixel")
    
    gt_landmarks = load_ground_truth()
    print(f"  Ground truth landmarks: {len(gt_landmarks)}")
    
    # Load models
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    print(f"\nLoading 256x256 model...")
    predictor_256 = LandmarkPredictor(
        checkpoint_path=CHECKPOINT_256_PATH,
        model_name='hrnet',
        device=device
    )
    print(f"  Model loaded successfully")
    
    print(f"\nLoading 512x512 model...")
    predictor_512 = LandmarkPredictor(
        checkpoint_path=CHECKPOINT_512_PATH,
        model_name='hrnet',
        device=device
    )
    print(f"  Model loaded successfully")
    
    # Run predictions
    print(f"\n" + "="*120)
    print("RUNNING PREDICTIONS")
    print("="*120)
    
    print(f"\nPredicting with 256x256 model...")
    result_256 = predictor_256.predict(img, target_size=(256, 256))
    predicted_256 = result_256['landmarks']
    print(f"  Valid landmarks: {len([v for v in predicted_256.values() if v is not None])}/29")
    
    print(f"\nPredicting with 512x512 model...")
    result_512 = predictor_512.predict(img, target_size=(512, 512))
    predicted_512 = result_512['landmarks']
    print(f"  Valid landmarks: {len([v for v in predicted_512.values() if v is not None])}/29")
    
    # Calculate errors
    print(f"\nCalculating errors...")
    errors_256 = calculate_errors(predicted_256, gt_landmarks)
    errors_512 = calculate_errors(predicted_512, gt_landmarks)
    
    # Calculate metrics
    metrics_256 = calculate_metrics(errors_256)
    metrics_512 = calculate_metrics(errors_512)
    
    # Print comparison
    print_metrics_comparison(metrics_256, metrics_512)
    print_comparison_table(errors_256, errors_512, metrics_256, metrics_512)
    
    # Summary
    print(f"\n" + "="*120)
    print("SUMMARY")
    print("="*120)
    
    print(f"\n256x256 Model:")
    print(f"  MRE: {metrics_256['mre_mm']:.4f} mm")
    print(f"  SDR @ 2mm: {metrics_256['sdr_2mm']:.2f}%")
    
    print(f"\n512x512 Model:")
    print(f"  MRE: {metrics_512['mre_mm']:.4f} mm")
    print(f"  SDR @ 2mm: {metrics_512['sdr_2mm']:.2f}%")
    
    mre_improvement = metrics_256['mre_mm'] - metrics_512['mre_mm']
    sdr_improvement = metrics_512['sdr_2mm'] - metrics_256['sdr_2mm']
    
    print(f"\nImprovement (512x512 vs 256x256):")
    print(f"  MRE reduction: {mre_improvement:.4f} mm ({'Better' if mre_improvement > 0 else 'Worse'})")
    print(f"  SDR increase: {sdr_improvement:.2f}% ({'Better' if sdr_improvement > 0 else 'Worse'})")
    
    if mre_improvement > 0 and sdr_improvement > 0:
        print(f"\nCONCLUSION: 512x512 model performs better than 256x256")
    elif mre_improvement < 0 and sdr_improvement < 0:
        print(f"\nCONCLUSION: 256x256 model performs better than 512x512")
    else:
        print(f"\nCONCLUSION: Mixed results - check detailed comparison above")
    
    # Save results
    output = {
        'test_image_id': TEST_IMAGE_ID,
        'image_size': {'width': img_size[0], 'height': img_size[1]},
        'pixel_size': PIXEL_SIZE,
        'model_256x256': {
            'checkpoint': CHECKPOINT_256_PATH,
            'metrics': metrics_256,
            'num_landmarks': len(errors_256)
        },
        'model_512x512': {
            'checkpoint': CHECKPOINT_512_PATH,
            'metrics': metrics_512,
            'num_landmarks': len(errors_512)
        },
        'comparison': {
            'mre_improvement': float(mre_improvement),
            'sdr_improvement': float(sdr_improvement),
            'better_model': '512x512' if (mre_improvement > 0 and sdr_improvement > 0) else '256x256' if (mre_improvement < 0 and sdr_improvement < 0) else 'mixed'
        },
        'per_landmark_errors': {
            '256x256': {e['name']: {'mm': e['error_mm'], 'px': e['error_px']} for e in errors_256},
            '512x512': {e['name']: {'mm': e['error_mm'], 'px': e['error_px']} for e in errors_512}
        }
    }
    
    output_file = 'compare_256_vs_512_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*120)

if __name__ == '__main__':
    main()

