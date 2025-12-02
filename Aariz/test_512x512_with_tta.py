"""
تست مدل 512×512 با Test-Time Augmentation (TTA)
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

CHECKPOINT_PATH = os.path.join(aariz_path, "checkpoints", "checkpoint_best.pth")
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

def predict_with_tta(predictor, image, target_size=(512, 512)):
    """
    پیش‌بینی با Test-Time Augmentation
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
            # Coordinates are already scaled to original size
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
    
    return {
        'landmarks': final_landmarks,
        'image_size': original_size
    }

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

def print_comparison(errors_before, errors_after):
    """Compare results before and after TTA"""
    print(f"\n{'='*100}")
    print("📊 مقایسه: بدون TTA vs با TTA")
    print(f"{'='*100}")
    
    mre_before = np.mean([e['error_mm'] for e in errors_before])
    mre_after = np.mean([e['error_mm'] for e in errors_after])
    
    sdr_before = sum(1 for e in errors_before if e['error_mm'] <= 2.0) / len(errors_before) * 100
    sdr_after = sum(1 for e in errors_after if e['error_mm'] <= 2.0) / len(errors_after) * 100
    
    print(f"\n{'معیار':<20} {'بدون TTA':<20} {'با TTA':<20} {'بهبود':<20}")
    print("-"*80)
    print(f"{'MRE (mm)':<20} {mre_before:<20.4f} {mre_after:<20.4f} {mre_before - mre_after:<20.4f}")
    print(f"{'SDR @ 2mm (%)':<20} {sdr_before:<20.2f} {sdr_after:<20.2f} {sdr_after - sdr_before:<20.2f}")
    
    # بهبود برای لندمارک‌های مشکل‌دار
    print(f"\n{'='*100}")
    print("🔍 بهبود برای لندمارک‌های مشکل‌دار (>2mm):")
    print(f"{'='*100}")
    
    problematic = [e for e in errors_before if e['error_mm'] > 2.0]
    improvements = []
    
    for err_before in problematic:
        name = err_before['name']
        err_after = next((e for e in errors_after if e['name'] == name), None)
        
        if err_after:
            improvement = err_before['error_mm'] - err_after['error_mm']
            if improvement > 0:
                improvements.append({
                    'name': name,
                    'before': err_before['error_mm'],
                    'after': err_after['error_mm'],
                    'improvement': improvement
                })
    
    if improvements:
        improvements.sort(key=lambda x: x['improvement'], reverse=True)
        print(f"\n{'Landmark':<10} {'قبل (mm)':<15} {'بعد (mm)':<15} {'بهبود (mm)':<15}")
        print("-"*55)
        for imp in improvements[:10]:  # Top 10
            print(f"{imp['name']:<10} {imp['before']:<15.4f} {imp['after']:<15.4f} {imp['improvement']:<15.4f}")
    else:
        print("\n⚠️  بهبود خاصی برای لندمارک‌های مشکل‌دار مشاهده نشد")

def main():
    print("="*100)
    print("🧪 تست مدل 512×512 با Test-Time Augmentation (TTA)")
    print("="*100)
    print(f"\n📸 تصویر تست: {TEST_IMAGE_ID}")
    
    # Check files
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\n❌ ERROR: Ground truth not found: {GROUND_TRUTH_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        return
    
    # Load image and GT
    img = Image.open(TEST_IMAGE_PATH)
    gt_landmarks = load_ground_truth()
    
    print(f"   اندازه: {img.size[0]} × {img.size[1]} پیکسل")
    print(f"   Ground Truth: {len(gt_landmarks)} لندمارک")
    
    # Load model
    print(f"\n🤖 بارگذاری مدل...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = LandmarkPredictor(CHECKPOINT_PATH, model_name='hrnet', device=device)
    print(f"   ✅ Model loaded on {device}")
    
    # Test 1: Without TTA
    print(f"\n{'='*100}")
    print("🔬 TEST 1: بدون TTA")
    print(f"{'='*100}")
    result_no_tta = predictor.predict(img, target_size=(512, 512))
    errors_no_tta = calculate_errors(result_no_tta['landmarks'], gt_landmarks)
    
    mre_no_tta = np.mean([e['error_mm'] for e in errors_no_tta])
    sdr_no_tta = sum(1 for e in errors_no_tta if e['error_mm'] <= 2.0) / len(errors_no_tta) * 100
    
    print(f"\n   MRE: {mre_no_tta:.4f} mm")
    print(f"   SDR @ 2mm: {sdr_no_tta:.2f}%")
    
    # Test 2: With TTA
    print(f"\n{'='*100}")
    print("🔬 TEST 2: با TTA")
    print(f"{'='*100}")
    result_tta = predict_with_tta(predictor, img, target_size=(512, 512))
    errors_tta = calculate_errors(result_tta['landmarks'], gt_landmarks)
    
    mre_tta = np.mean([e['error_mm'] for e in errors_tta])
    sdr_tta = sum(1 for e in errors_tta if e['error_mm'] <= 2.0) / len(errors_tta) * 100
    
    print(f"\n   MRE: {mre_tta:.4f} mm")
    print(f"   SDR @ 2mm: {sdr_tta:.2f}%")
    
    # Comparison
    print_comparison(errors_no_tta, errors_tta)
    
    # Save results
    output = {
        'image_id': TEST_IMAGE_ID,
        'without_tta': {
            'mre_mm': float(mre_no_tta),
            'sdr_2mm': float(sdr_no_tta)
        },
        'with_tta': {
            'mre_mm': float(mre_tta),
            'sdr_2mm': float(sdr_tta)
        },
        'improvement': {
            'mre_reduction': float(mre_no_tta - mre_tta),
            'sdr_increase': float(sdr_tta - sdr_no_tta)
        }
    }
    
    output_file = 'test_512x512_tta_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    
    # Final summary
    print(f"\n{'='*100}")
    print("📋 خلاصه")
    print(f"{'='*100}")
    
    if mre_tta < mre_no_tta:
        print(f"✅ TTA موثر بود!")
        print(f"   MRE بهبود: {mre_no_tta - mre_tta:.4f} mm")
        print(f"   SDR بهبود: {sdr_tta - sdr_no_tta:.2f}%")
    else:
        print(f"⚠️  TTA تاثیر زیادی نداشت")
        print(f"   ممکن است نیاز به روش‌های دیگر باشد")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

