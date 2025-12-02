"""
تست مدل 512×512 با Ground Truth
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

# Model path (checkpoint جدید 512x512)
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

def print_results(errors, image_size):
    """Print comparison results"""
    print(f"\n{'='*100}")
    print("📊 مقایسه نتایج با Ground Truth")
    print(f"{'='*100}")
    print(f"\n{'Landmark':<10} {'Pred X':<12} {'Pred Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<10} {'Diff Y':<10} {'Error (px)':<12} {'Error (mm)':<12}")
    print("-"*100)
    
    errors.sort(key=lambda x: x['error_mm'], reverse=True)
    
    for err in errors:
        pred = err['pred']
        gt = err['gt']
        print(f"{err['name']:<10} {pred['x']:<12.2f} {pred['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {err['diff_x']:<10.2f} {err['diff_y']:<10.2f} {err['error_px']:<12.2f} {err['error_mm']:<12.4f}")
    
    # Statistics
    error_values_mm = [e['error_mm'] for e in errors]
    
    print(f"\n{'='*100}")
    print("📈 آمار خطاها")
    print(f"{'='*100}")
    print(f"\n✅ تعداد لندمارک‌های مقایسه شده: {len(errors)}")
    print(f"\n📊 خطا بر حسب میلی‌متر:")
    print(f"   میانگین (MRE): {np.mean(error_values_mm):.4f} mm")
    print(f"   میانه: {np.median(error_values_mm):.4f} mm")
    print(f"   کمینه: {np.min(error_values_mm):.4f} mm")
    print(f"   بیشینه: {np.max(error_values_mm):.4f} mm")
    print(f"   انحراف معیار: {np.std(error_values_mm):.4f} mm")
    
    # SDR
    thresholds_mm = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
    print(f"\n{'='*100}")
    print("📊 Success Detection Rate (SDR)")
    print(f"{'='*100}")
    for threshold_mm in thresholds_mm:
        success = sum(1 for e_mm in error_values_mm if e_mm <= threshold_mm)
        sdr = (success / len(error_values_mm)) * 100
        print(f"   SDR @ {threshold_mm}mm: {sdr:.2f}% ({success}/{len(errors)})")
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'errors': errors
    }

def main():
    print("="*100)
    print("🧪 تست مدل 512×512")
    print("="*100)
    print(f"\n📸 تصویر تست: {TEST_IMAGE_ID}")
    print(f"📂 مسیر تصویر: {TEST_IMAGE_PATH}")
    print(f"📂 مسیر Ground Truth: {GROUND_TRUTH_PATH}")
    print(f"📂 مسیر Checkpoint: {CHECKPOINT_PATH}")
    
    # Check files exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\n❌ ERROR: Ground truth not found: {GROUND_TRUTH_PATH}")
        return
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"\n❌ ERROR: Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"   لطفاً مطمئن شوید training تمام شده است!")
        return
    
    # Load image
    print(f"\n📸 بارگذاری تصویر...")
    img = Image.open(TEST_IMAGE_PATH)
    img_size = img.size
    print(f"   اندازه: {img_size[0]} × {img_size[1]} پیکسل")
    print(f"   Pixel Size: {PIXEL_SIZE} mm/pixel")
    
    # Load ground truth
    print(f"\n📂 بارگذاری Ground Truth...")
    gt_landmarks = load_ground_truth()
    print(f"   ✅ {len(gt_landmarks)} لندمارک در Ground Truth یافت شد")
    
    # Load model
    print(f"\n🤖 بارگذاری مدل 512×512...")
    print(f"   Checkpoint: {CHECKPOINT_PATH}")
    
    try:
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   Device: {device}")
        
        predictor = LandmarkPredictor(
            checkpoint_path=CHECKPOINT_PATH,
            model_name='hrnet',
            device=device
        )
        print(f"   ✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run prediction with 512×512
    print(f"\n🔍 اجرای تشخیص با 512×512...")
    try:
        # CRITICAL: Use 512×512 (matching training size)
        result = predictor.predict(img, target_size=(512, 512))
        predicted_landmarks = result['landmarks']
        
        print(f"   ✅ Detection complete!")
        print(f"   Valid landmarks: {len(predicted_landmarks)}/29")
        
    except Exception as e:
        print(f"\n❌ ERROR: Detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Calculate errors
    print(f"\n📊 محاسبه خطاها...")
    errors = calculate_errors(predicted_landmarks, gt_landmarks)
    
    if not errors:
        print("\n⚠️  هیچ لندمارک مشترکی برای مقایسه یافت نشد!")
        return
    
    # Print comparison
    stats = print_results(errors, img_size)
    
    # Save results
    output = {
        'image_id': TEST_IMAGE_ID,
        'image_size': {'width': img_size[0], 'height': img_size[1]},
        'model_input_size': '512×512',
        'pixel_size': PIXEL_SIZE,
        'checkpoint': CHECKPOINT_PATH,
        'stats': {
            'mre_mm': stats['mre_mm'],
            'median_mm': float(np.median([e['error_mm'] for e in stats['errors']])),
            'min_mm': float(np.min([e['error_mm'] for e in stats['errors']])),
            'max_mm': float(np.max([e['error_mm'] for e in stats['errors']]))
        },
        'errors': {e['name']: {'mm': e['error_mm'], 'px': e['error_px']} for e in stats['errors']}
    }
    
    output_file = 'test_512x512_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    
    # Final summary
    print(f"\n{'='*100}")
    print("📋 خلاصه")
    print(f"{'='*100}")
    print(f"   Model: HRNet (512×512)")
    print(f"   MRE: {stats['mre_mm']:.4f} mm")
    sdr_2mm = sum(1 for e_mm in [e['error_mm'] for e in stats['errors']] if e_mm <= 2.0) / len(stats['errors']) * 100
    print(f"   SDR @ 2mm: {sdr_2mm:.2f}%")
    
    # Compare with training results
    print(f"\n📊 مقایسه با نتایج Training:")
    print(f"   Training (Epoch 35): MRE=1.41mm, SDR @ 2mm=80.25%")
    print(f"   Test (این تصویر):   MRE={stats['mre_mm']:.4f}mm, SDR @ 2mm={sdr_2mm:.2f}%")
    
    if stats['mre_mm'] < 1.5:
        print(f"\n✅ نتایج عالی! MRE کمتر از 1.5mm است")
    elif stats['mre_mm'] < 2.0:
        print(f"\n✅ نتایج خوب! MRE کمتر از 2mm است")
    else:
        print(f"\n⚠️  MRE بالاست. ممکن است نیاز به بررسی بیشتر باشد")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

