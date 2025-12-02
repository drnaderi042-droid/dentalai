"""
Ensemble 256x256 and 512x512 models with landmark-specific weighting
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
PIXEL_SIZE = 0.1

# Landmark-specific model preferences based on comparison results
# True = use 512x512, False = use 256x256
LANDMARK_PREFERENCE = {
    # 512x512 better (facial landmarks, some mandibular)
    'Or': True, 'Sn': True, 'N': True, 'N`': True, 'A': True, 'Ar': True,
    'LIA': True, 'Me': True, 'Gn': True, 'Li': True, 'Ls': True, 'Pog': True,
    'Po': True, 'Pog`': True, 'Pn': True, 'LIT': True, 'UIT': True, 'UIA': True,
    'Go': True, 'ANS': True,
    
    # 256x256 better (mandibular region, cranial base)
    'LMT': False, 'UMT': False, 'UPM': False, 'S': False, 'LPM': False,
    'Co': False, 'R': False, 'PNS': False, 'B': False
}

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

def ensemble_predictions(pred_256, pred_512, preference_dict):
    """
    Ensemble predictions based on landmark-specific preferences
    """
    ensemble = {}
    
    for name in set(list(pred_256.keys()) + list(pred_512.keys())):
        # Get predictions from both models
        p256 = pred_256.get(name)
        p512 = pred_512.get(name)
        
        # Use preference if available
        if name in preference_dict:
            if preference_dict[name]:
                # Use 512x512
                ensemble[name] = p512 if p512 is not None else p256
            else:
                # Use 256x256
                ensemble[name] = p256 if p256 is not None else p512
        else:
            # Default: weighted average (60% better model, 40% other)
            # Or simple average if both available
            if p256 is not None and p512 is not None:
                # Average both predictions
                ensemble[name] = {
                    'x': (p256['x'] * 0.5 + p512['x'] * 0.5),
                    'y': (p256['y'] * 0.5 + p512['y'] * 0.5)
                }
            elif p256 is not None:
                ensemble[name] = p256
            elif p512 is not None:
                ensemble[name] = p512
            else:
                ensemble[name] = None
    
    return ensemble

def calculate_errors(predicted, ground_truth):
    """Calculate errors between predicted and ground truth"""
    errors = []
    
    for name in ground_truth.keys():
        if name in predicted and predicted[name] is not None:
            pred = predicted[name]
            gt = ground_truth[name]
            
            error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            error_mm = error_px * PIXEL_SIZE
            
            errors.append({
                'name': name,
                'error_px': error_px,
                'error_mm': error_mm,
                'pred': pred,
                'gt': gt
            })
    
    return errors

def calculate_metrics(errors):
    """Calculate MRE and SDR metrics"""
    if not errors:
        return {
            'mre_mm': 0.0,
            'median_mm': 0.0,
            'std_mm': 0.0,
            'sdr_2mm': 0.0
        }
    
    error_values_mm = [e['error_mm'] for e in errors]
    
    sdr_2mm = sum(1 for e_mm in error_values_mm if e_mm <= 2.0) / len(error_values_mm) * 100
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'median_mm': float(np.median(error_values_mm)),
        'std_mm': float(np.std(error_values_mm)),
        'sdr_2mm': float(sdr_2mm)
    }

def main():
    print("="*120)
    print("ENSEMBLE: 256x256 + 512x512 Model Combination")
    print("="*120)
    print(f"\nTest Image: {TEST_IMAGE_ID}")
    
    # Check files
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
    gt_landmarks = load_ground_truth()
    print(f"  Image size: {img.size[0]} x {img.size[1]} pixels")
    print(f"  Ground truth landmarks: {len(gt_landmarks)}")
    
    # Load models
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    print(f"\nLoading 256x256 model...")
    predictor_256 = LandmarkPredictor(CHECKPOINT_256_PATH, model_name='hrnet', device=device)
    print(f"  Model loaded successfully")
    
    print(f"\nLoading 512x512 model...")
    predictor_512 = LandmarkPredictor(CHECKPOINT_512_PATH, model_name='hrnet', device=device)
    print(f"  Model loaded successfully")
    
    # Run predictions
    print(f"\n" + "="*120)
    print("RUNNING PREDICTIONS")
    print("="*120)
    
    print(f"\nPredicting with 256x256 model...")
    result_256 = predictor_256.predict(img, target_size=(256, 256))
    pred_256 = result_256['landmarks']
    print(f"  Valid landmarks: {len([v for v in pred_256.values() if v is not None])}/29")
    
    print(f"\nPredicting with 512x512 model...")
    result_512 = predictor_512.predict(img, target_size=(512, 512))
    pred_512 = result_512['landmarks']
    print(f"  Valid landmarks: {len([v for v in pred_512.values() if v is not None])}/29")
    
    # Create ensemble
    print(f"\nCreating ensemble predictions...")
    ensemble_pred = ensemble_predictions(pred_256, pred_512, LANDMARK_PREFERENCE)
    print(f"  Ensemble landmarks: {len([v for v in ensemble_pred.values() if v is not None])}/29")
    
    # Calculate errors for all three
    print(f"\nCalculating errors...")
    errors_256 = calculate_errors(pred_256, gt_landmarks)
    errors_512 = calculate_errors(pred_512, gt_landmarks)
    errors_ensemble = calculate_errors(ensemble_pred, gt_landmarks)
    
    # Calculate metrics
    metrics_256 = calculate_metrics(errors_256)
    metrics_512 = calculate_metrics(errors_512)
    metrics_ensemble = calculate_metrics(errors_ensemble)
    
    # Print comparison
    print(f"\n" + "="*120)
    print("RESULTS COMPARISON")
    print("="*120)
    
    print(f"\n{'Model':<20} {'MRE (mm)':<15} {'Median (mm)':<15} {'Std Dev (mm)':<15} {'SDR @ 2mm (%)':<15}")
    print("-"*80)
    print(f"{'256x256':<20} {metrics_256['mre_mm']:<15.4f} {metrics_256['median_mm']:<15.4f} {metrics_256['std_mm']:<15.4f} {metrics_256['sdr_2mm']:<15.2f}")
    print(f"{'512x512':<20} {metrics_512['mre_mm']:<15.4f} {metrics_512['median_mm']:<15.4f} {metrics_512['std_mm']:<15.4f} {metrics_512['sdr_2mm']:<15.2f}")
    print(f"{'ENSEMBLE':<20} {metrics_ensemble['mre_mm']:<15.4f} {metrics_ensemble['median_mm']:<15.4f} {metrics_ensemble['std_mm']:<15.4f} {metrics_ensemble['sdr_2mm']:<15.2f}")
    
    # Improvement over best single model
    best_sdr = max(metrics_256['sdr_2mm'], metrics_512['sdr_2mm'])
    best_mre = min(metrics_256['mre_mm'], metrics_512['mre_mm'])
    
    ensemble_sdr_improvement = metrics_ensemble['sdr_2mm'] - best_sdr
    ensemble_mre_improvement = best_mre - metrics_ensemble['mre_mm']
    
    print(f"\n" + "="*120)
    print("ENSEMBLE IMPROVEMENT")
    print("="*120)
    print(f"\nBest single model SDR @ 2mm: {best_sdr:.2f}%")
    print(f"Ensemble SDR @ 2mm: {metrics_ensemble['sdr_2mm']:.2f}%")
    print(f"Improvement: {ensemble_sdr_improvement:+.2f}%")
    
    print(f"\nBest single model MRE: {best_mre:.4f} mm")
    print(f"Ensemble MRE: {metrics_ensemble['mre_mm']:.4f} mm")
    print(f"Improvement: {ensemble_mre_improvement:+.4f} mm")
    
    if ensemble_sdr_improvement > 0:
        print(f"\nCONCLUSION: Ensemble performs better than both single models!")
    elif ensemble_sdr_improvement == 0:
        print(f"\nCONCLUSION: Ensemble matches best single model")
    else:
        print(f"\nCONCLUSION: Ensemble performs worse - may need weight adjustment")
    
    # Save results
    output = {
        'test_image_id': TEST_IMAGE_ID,
        'ensemble_strategy': 'landmark-specific_selection',
        'metrics': {
            '256x256': metrics_256,
            '512x512': metrics_512,
            'ensemble': metrics_ensemble
        },
        'improvement': {
            'sdr_improvement': float(ensemble_sdr_improvement),
            'mre_improvement': float(ensemble_mre_improvement)
        }
    }
    
    output_file = 'ensemble_256_512_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*120)

if __name__ == '__main__':
    main()

