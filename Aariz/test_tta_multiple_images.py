"""
Test 512x512 model with TTA on multiple images from validation set
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
CHECKPOINT_PATH = os.path.join(aariz_path, "checkpoints", "checkpoint_best.pth")
PIXEL_SIZE = 0.1

def get_image_files():
    """Get all image files from validation set"""
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
    image_files = []
    
    for ext in extensions:
        image_files.extend([f for f in os.listdir(CEPHALOGRAMS_PATH) if f.endswith(ext)])
    
    # Remove duplicates and get unique IDs
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

def predict_with_tta(predictor, image, target_size=(512, 512)):
    """
    Prediction with Test-Time Augmentation
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size  # (width, height)
    w, h = original_size
    
    predictions = []
    weights = []
    
    # 1. Original image
    pred_orig = predictor.predict(image, target_size)
    predictions.append(pred_orig['landmarks'])
    weights.append(1.0)
    
    # 2. Horizontal flip
    img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    pred_flip = predictor.predict(img_flip, target_size)
    
    # Flip coordinates back
    pred_flip_flipped = {}
    for name, coords in pred_flip['landmarks'].items():
        if coords is not None:
            pred_flip_flipped[name] = {
                'x': w - coords['x'],
                'y': coords['y']
            }
    
    predictions.append(pred_flip_flipped)
    weights.append(1.0)
    
    # Weighted averaging
    final_landmarks = {}
    for name in predictions[0].keys():
        xs = []
        ys = []
        for pred, weight in zip(predictions, weights):
            if name in pred and pred[name] is not None:
                xs.extend([pred[name]['x']] * int(weight * 10))
                ys.extend([pred[name]['y']] * int(weight * 10))
        
        if xs:
            final_landmarks[name] = {
                'x': np.mean(xs),
                'y': np.mean(ys)
            }
    
    return final_landmarks

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
    print("Test 512x512 Model with TTA on Multiple Images")
    print("="*100)
    
    # Get image IDs
    image_ids = get_image_files()
    print(f"\nNumber of images found: {len(image_ids)}")
    
    if not image_ids:
        print("ERROR: No images found!")
        return
    
    # Limit to first N images for testing (adjust as needed)
    max_images = 20
    if len(image_ids) > max_images:
        image_ids = image_ids[:max_images]
        print(f"Testing on first {max_images} images")
    
    # Load model
    print(f"\nLoading model...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = LandmarkPredictor(CHECKPOINT_PATH, model_name='hrnet', device=device)
    print(f"  Model loaded on {device}")
    
    # Test on each image
    results_no_tta = []
    results_with_tta = []
    
    print(f"\n" + "="*100)
    print("Testing on images...")
    print("="*100)
    
    for image_id in tqdm(image_ids, desc="Processing images"):
        # Find image path
        img_path = find_image_path(image_id)
        if not img_path:
            continue
        
        # Load ground truth
        gt_landmarks = load_ground_truth(image_id)
        if not gt_landmarks:
            continue
        
        try:
            # Load image
            img = Image.open(img_path).convert('RGB')
            
            # Test without TTA
            result_no_tta = predictor.predict(img, target_size=(512, 512))
            errors_no_tta = calculate_errors(result_no_tta['landmarks'], gt_landmarks)
            metrics_no_tta = calculate_metrics(errors_no_tta)
            
            # Test with TTA
            pred_tta = predict_with_tta(predictor, img, target_size=(512, 512))
            errors_tta = calculate_errors(pred_tta, gt_landmarks)
            metrics_tta = calculate_metrics(errors_tta)
            
            results_no_tta.append({
                'image_id': image_id,
                'mre_mm': metrics_no_tta['mre_mm'],
                'sdr_2mm': metrics_no_tta['sdr_2mm']
            })
            
            results_with_tta.append({
                'image_id': image_id,
                'mre_mm': metrics_tta['mre_mm'],
                'sdr_2mm': metrics_tta['sdr_2mm']
            })
                
        except Exception as e:
            print(f"\nWARNING: Error processing {image_id}: {e}")
            continue
    
    if not results_no_tta:
        print("\nERROR: No results obtained!")
        return
    
    # Calculate overall statistics
    mre_no_tta = [r['mre_mm'] for r in results_no_tta]
    sdr_no_tta = [r['sdr_2mm'] for r in results_no_tta]
    
    mre_tta = [r['mre_mm'] for r in results_with_tta]
    sdr_tta = [r['sdr_2mm'] for r in results_with_tta]
    
    print(f"\n" + "="*100)
    print("OVERALL RESULTS")
    print("="*100)
    
    print(f"\nNumber of images tested: {len(results_no_tta)}")
    
    print(f"\n{'Metric':<30} {'Without TTA':<20} {'With TTA':<20} {'Improvement':<20}")
    print("-"*90)
    print(f"{'Mean MRE (mm)':<30} {np.mean(mre_no_tta):<20.4f} {np.mean(mre_tta):<20.4f} {np.mean(mre_no_tta) - np.mean(mre_tta):<20.4f}")
    print(f"{'Median MRE (mm)':<30} {np.median(mre_no_tta):<20.4f} {np.median(mre_tta):<20.4f} {np.median(mre_no_tta) - np.median(mre_tta):<20.4f}")
    print(f"{'Mean SDR @ 2mm (%)':<30} {np.mean(sdr_no_tta):<20.2f} {np.mean(sdr_tta):<20.2f} {np.mean(sdr_tta) - np.mean(sdr_no_tta):<20.2f}")
    print(f"{'Median SDR @ 2mm (%)':<30} {np.median(sdr_no_tta):<20.2f} {np.median(sdr_tta):<20.2f} {np.median(sdr_tta) - np.median(sdr_no_tta):<20.2f}")
    
    # Per-image improvement
    improvements = []
    for r_no, r_tta in zip(results_no_tta, results_with_tta):
        improvements.append({
            'image_id': r_no['image_id'],
            'mre_improvement': r_no['mre_mm'] - r_tta['mre_mm'],
            'sdr_improvement': r_tta['sdr_2mm'] - r_no['sdr_2mm']
        })
    
    improvements.sort(key=lambda x: x['sdr_improvement'], reverse=True)
    
    print(f"\n" + "="*100)
    print("PER-IMAGE RESULTS (Top 10 improvements)")
    print("="*100)
    print(f"\n{'Image ID':<35} {'No TTA MRE':<15} {'TTA MRE':<15} {'No TTA SDR':<15} {'TTA SDR':<15} {'SDR Improv.':<15}")
    print("-"*110)
    
    for imp in improvements[:10]:
        r_no = next(r for r in results_no_tta if r['image_id'] == imp['image_id'])
        r_tta = next(r for r in results_with_tta if r['image_id'] == imp['image_id'])
        print(f"{imp['image_id']:<35} {r_no['mre_mm']:<15.4f} {r_tta['mre_mm']:<15.4f} {r_no['sdr_2mm']:<15.2f} {r_tta['sdr_2mm']:<15.2f} {imp['sdr_improvement']:<15.2f}")
    
    # Save results
    output = {
        'num_images': len(results_no_tta),
        'overall_stats': {
            'without_tta': {
                'mean_mre': float(np.mean(mre_no_tta)),
                'median_mre': float(np.median(mre_no_tta)),
                'mean_sdr_2mm': float(np.mean(sdr_no_tta)),
                'median_sdr_2mm': float(np.median(sdr_no_tta))
            },
            'with_tta': {
                'mean_mre': float(np.mean(mre_tta)),
                'median_mre': float(np.median(mre_tta)),
                'mean_sdr_2mm': float(np.mean(sdr_tta)),
                'median_sdr_2mm': float(np.median(sdr_tta))
            },
            'improvement': {
                'mean_mre_reduction': float(np.mean(mre_no_tta) - np.mean(mre_tta)),
                'mean_sdr_increase': float(np.mean(sdr_tta) - np.mean(sdr_no_tta))
            }
        },
        'per_image_results': [
            {
                'image_id': r_no['image_id'],
                'without_tta': {'mre_mm': r_no['mre_mm'], 'sdr_2mm': r_no['sdr_2mm']},
                'with_tta': {'mre_mm': r_tta['mre_mm'], 'sdr_2mm': r_tta['sdr_2mm']},
                'improvement': {'mre_reduction': r_no['mre_mm'] - r_tta['mre_mm'], 'sdr_increase': r_tta['sdr_2mm'] - r_no['sdr_2mm']}
            }
            for r_no, r_tta in zip(results_no_tta, results_with_tta)
        ]
    }
    
    output_file = 'test_tta_multiple_images_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

