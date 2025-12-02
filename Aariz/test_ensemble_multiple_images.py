"""
Test Ensemble 256x256 + 512x512 on multiple images from validation set
"""

import os
import sys
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
aariz_path = os.path.join(base_dir, 'Aariz')

if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

from inference import LandmarkPredictor

# Paths
CEPHALOGRAMS_PATH = os.path.join(aariz_path, "Aariz", "valid", "Cephalograms")
ANNOTATIONS_PATH = os.path.join(aariz_path, "Aariz", "valid", "Annotations", "Cephalometric Landmarks", "Senior Orthodontists")
CHECKPOINT_256_PATH = os.path.join(aariz_path, "checkpoint_best_256.pth")
CHECKPOINT_512_PATH = os.path.join(aariz_path, "checkpoints", "checkpoint_best.pth")
PIXEL_SIZE = 0.1

# Landmark-specific model preferences
LANDMARK_PREFERENCE = {
    'Or': True, 'Sn': True, 'N': True, 'N`': True, 'A': True, 'Ar': True,
    'LIA': True, 'Me': True, 'Gn': True, 'Li': True, 'Ls': True, 'Pog': True,
    'Po': True, 'Pog`': True, 'Pn': True, 'LIT': True, 'UIT': True, 'UIA': True,
    'Go': True, 'ANS': True,
    'LMT': False, 'UMT': False, 'UPM': False, 'S': False, 'LPM': False,
    'Co': False, 'R': False, 'PNS': False, 'B': False
}

def get_image_files():
    """Get all image files from validation set"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
    image_files = []
    
    for ext in extensions:
        image_files.extend([f for f in os.listdir(CEPHALOGRAMS_PATH) if f.endswith(ext)])
    
    image_ids = list(set([os.path.splitext(f)[0] for f in image_files]))
    return sorted(image_ids)

def load_ground_truth(image_id):
    """Load ground truth for an image"""
    json_path = os.path.join(ANNOTATIONS_PATH, f"{image_id}.json")
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    gt_landmarks = {}
    for lm in annotation['landmarks']:
        symbol = lm['symbol']
        gt_landmarks[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return gt_landmarks

def find_image_path(image_id):
    """Find image path by ID"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
    
    for ext in extensions:
        img_path = os.path.join(CEPHALOGRAMS_PATH, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    
    return None

def ensemble_predictions(pred_256, pred_512, preference_dict):
    """Ensemble predictions based on landmark-specific preferences"""
    ensemble = {}
    
    for name in set(list(pred_256.keys()) + list(pred_512.keys())):
        p256 = pred_256.get(name)
        p512 = pred_512.get(name)
        
        if name in preference_dict:
            if preference_dict[name]:
                ensemble[name] = p512 if p512 is not None else p256
            else:
                ensemble[name] = p256 if p256 is not None else p512
        else:
            if p256 is not None and p512 is not None:
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
                'error_mm': error_mm
            })
    
    return errors

def calculate_metrics(errors):
    """Calculate MRE and SDR metrics"""
    if not errors:
        return {'mre_mm': 0.0, 'sdr_2mm': 0.0}
    
    error_values_mm = [e['error_mm'] for e in errors]
    sdr_2mm = sum(1 for e_mm in error_values_mm if e_mm <= 2.0) / len(error_values_mm) * 100
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'sdr_2mm': float(sdr_2mm)
    }

def main():
    print("="*100)
    print("Test Ensemble 256x256 + 512x512 on Multiple Images")
    print("="*100)
    
    # Get image IDs
    image_ids = get_image_files()
    print(f"\nNumber of images found: {len(image_ids)}")
    
    if not image_ids:
        print("ERROR: No images found!")
        return
    
    # Limit to first N images for testing
    max_images = 20
    if len(image_ids) > max_images:
        image_ids = image_ids[:max_images]
        print(f"Testing on first {max_images} images")
    
    # Load models
    print(f"\nLoading models...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")
    
    predictor_256 = LandmarkPredictor(CHECKPOINT_256_PATH, model_name='hrnet', device=device)
    print(f"  256x256 model loaded")
    
    predictor_512 = LandmarkPredictor(CHECKPOINT_512_PATH, model_name='hrnet', device=device)
    print(f"  512x512 model loaded")
    
    # Test on each image
    results_256 = []
    results_512 = []
    results_ensemble = []
    
    print(f"\n" + "="*100)
    print("Testing on images...")
    print("="*100)
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        img_path = find_image_path(image_id)
        if not img_path:
            continue
        
        gt_landmarks = load_ground_truth(image_id)
        if not gt_landmarks:
            continue
        
        try:
            img = Image.open(img_path).convert('RGB')
            
            # Predict with 256x256
            result_256 = predictor_256.predict(img, target_size=(256, 256))
            errors_256 = calculate_errors(result_256['landmarks'], gt_landmarks)
            metrics_256 = calculate_metrics(errors_256)
            
            # Predict with 512x512
            result_512 = predictor_512.predict(img, target_size=(512, 512))
            errors_512 = calculate_errors(result_512['landmarks'], gt_landmarks)
            metrics_512 = calculate_metrics(errors_512)
            
            # Ensemble
            ensemble_pred = ensemble_predictions(result_256['landmarks'], result_512['landmarks'], LANDMARK_PREFERENCE)
            errors_ensemble = calculate_errors(ensemble_pred, gt_landmarks)
            metrics_ensemble = calculate_metrics(errors_ensemble)
            
            results_256.append({
                'image_id': image_id,
                'mre_mm': metrics_256['mre_mm'],
                'sdr_2mm': metrics_256['sdr_2mm']
            })
            
            results_512.append({
                'image_id': image_id,
                'mre_mm': metrics_512['mre_mm'],
                'sdr_2mm': metrics_512['sdr_2mm']
            })
            
            results_ensemble.append({
                'image_id': image_id,
                'mre_mm': metrics_ensemble['mre_mm'],
                'sdr_2mm': metrics_ensemble['sdr_2mm']
            })
                
        except Exception as e:
            print(f"\nWARNING: Error processing {image_id}: {e}")
            continue
    
    if not results_256:
        print("\nERROR: No results obtained!")
        return
    
    # Calculate overall statistics
    mre_256 = [r['mre_mm'] for r in results_256]
    sdr_256 = [r['sdr_2mm'] for r in results_256]
    
    mre_512 = [r['mre_mm'] for r in results_512]
    sdr_512 = [r['sdr_2mm'] for r in results_512]
    
    mre_ensemble = [r['mre_mm'] for r in results_ensemble]
    sdr_ensemble = [r['sdr_2mm'] for r in results_ensemble]
    
    print(f"\n" + "="*100)
    print("OVERALL RESULTS")
    print("="*100)
    
    print(f"\nNumber of images tested: {len(results_256)}")
    
    print(f"\n{'Metric':<30} {'256x256':<20} {'512x512':<20} {'Ensemble':<20}")
    print("-"*90)
    print(f"{'Mean MRE (mm)':<30} {np.mean(mre_256):<20.4f} {np.mean(mre_512):<20.4f} {np.mean(mre_ensemble):<20.4f}")
    print(f"{'Median MRE (mm)':<30} {np.median(mre_256):<20.4f} {np.median(mre_512):<20.4f} {np.median(mre_ensemble):<20.4f}")
    print(f"{'Mean SDR @ 2mm (%)':<30} {np.mean(sdr_256):<20.2f} {np.mean(sdr_512):<20.2f} {np.mean(sdr_ensemble):<20.2f}")
    print(f"{'Median SDR @ 2mm (%)':<30} {np.median(sdr_256):<20.2f} {np.median(sdr_512):<20.2f} {np.median(sdr_ensemble):<20.2f}")
    
    # Improvement over best single model
    best_sdr_single = max(np.mean(sdr_256), np.mean(sdr_512))
    best_mre_single = min(np.mean(mre_256), np.mean(mre_512))
    
    ensemble_sdr_improvement = np.mean(sdr_ensemble) - best_sdr_single
    ensemble_mre_change = np.mean(mre_ensemble) - best_mre_single
    
    print(f"\n" + "="*100)
    print("ENSEMBLE IMPROVEMENT OVER BEST SINGLE MODEL")
    print("="*100)
    print(f"Best single model SDR @ 2mm: {best_sdr_single:.2f}%")
    print(f"Ensemble SDR @ 2mm: {np.mean(sdr_ensemble):.2f}%")
    print(f"Improvement: {ensemble_sdr_improvement:+.2f}%")
    
    print(f"\nBest single model MRE: {best_mre_single:.4f} mm")
    print(f"Ensemble MRE: {np.mean(mre_ensemble):.4f} mm")
    print(f"Change: {ensemble_mre_change:+.4f} mm")
    
    # Save results
    output = {
        'num_images': len(results_256),
        'overall_stats': {
            '256x256': {
                'mean_mre': float(np.mean(mre_256)),
                'median_mre': float(np.median(mre_256)),
                'mean_sdr_2mm': float(np.mean(sdr_256)),
                'median_sdr_2mm': float(np.median(sdr_256))
            },
            '512x512': {
                'mean_mre': float(np.mean(mre_512)),
                'median_mre': float(np.median(mre_512)),
                'mean_sdr_2mm': float(np.mean(sdr_512)),
                'median_sdr_2mm': float(np.median(sdr_512))
            },
            'ensemble': {
                'mean_mre': float(np.mean(mre_ensemble)),
                'median_mre': float(np.median(mre_ensemble)),
                'mean_sdr_2mm': float(np.mean(sdr_ensemble)),
                'median_sdr_2mm': float(np.median(sdr_ensemble))
            },
            'improvement': {
                'sdr_improvement': float(ensemble_sdr_improvement),
                'mre_change': float(ensemble_mre_change)
            }
        },
        'per_image_results': [
            {
                'image_id': results_256[i]['image_id'],
                '256x256': {'mre_mm': results_256[i]['mre_mm'], 'sdr_2mm': results_256[i]['sdr_2mm']},
                '512x512': {'mre_mm': results_512[i]['mre_mm'], 'sdr_2mm': results_512[i]['sdr_2mm']},
                'ensemble': {'mre_mm': results_ensemble[i]['mre_mm'], 'sdr_2mm': results_ensemble[i]['sdr_2mm']}
            }
            for i in range(len(results_256))
        ]
    }
    
    output_file = 'test_ensemble_multiple_images_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

