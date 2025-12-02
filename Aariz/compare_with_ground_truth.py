"""
مقایسه نتایج Frontend با Ground Truth (Annotations واقعی)
"""

import json
import os
import numpy as np
import sys

# نتایج از Frontend (از console logs کاربر)
frontend_results = {
    "A": {"x": 311.34375, "y": 1116.845703125},
    "ANS": {"x": 1233.84375, "y": 599.70703125},
    "Ar": {"x": 311.34375, "y": 1082.080078125},
    "B": {"x": 1329.9375, "y": 1725.244140625},
    "Co": {"x": 434.34375, "y": 1325.439453125},
    "Gn": {"x": 280.59375, "y": 925.634765625},
    "Go": {"x": 295.96875, "y": 1377.587890625},
    "LIA": {"x": 1310.71875, "y": 1707.861328125},
    "LIT": {"x": 295.96875, "y": 995.166015625},
    "LMT": {"x": 1249.21875, "y": 1603.564453125},
    "LPM": {"x": 388.21875, "y": 1029.931640625},
    "Li": {"x": 1499.0625, "y": 1208.10546875},
    "Ls": {"x": 1264.59375, "y": 425.87890625},
    "Me": {"x": 1283.8125, "y": 1729.58984375},
    "N": {"x": 1329.9375, "y": 1712.20703125},
    "N`": {"x": 295.96875, "y": 1520.99609375},
    "Or": {"x": 388.21875, "y": 1273.291015625},
    "PNS": {"x": 453.5625, "y": 1290.673828125},
    "Pn": {"x": 1499.0625, "y": 1208.10546875},
    "Po": {"x": 1356.84375, "y": 1290.673828125},
    "Pog": {"x": 1329.9375, "y": 1725.244140625},
    "Pog`": {"x": 219.09375, "y": 856.103515625},
    "R": {"x": 588.09375, "y": 1186.376953125},
    "S": {"x": 987.84375, "y": 1499.267578125},
    "Sn": {"x": 1422.1875, "y": 1273.291015625},
    "UIA": {"x": 311.34375, "y": 1103.80859375},
    "UIT": {"x": 238.3125, "y": 995.166015625},
    "UMT": {"x": 418.96875, "y": 1034.27734375},
    "UPM": {"x": 1249.21875, "y": 1625.29296875}
}

def load_ground_truth(image_id, dataset_path="Aariz", annotation_type="Senior Orthodontists"):
    """بارگذاری annotation واقعی از dataset"""
    annotation_path = os.path.join(
        dataset_path, "Aariz", "train", "Annotations",
        "Cephalometric Landmarks", annotation_type,
        f"{image_id}.json"
    )
    
    if not os.path.exists(annotation_path):
        print(f"❌ Annotation not found: {annotation_path}")
        return None
    
    with open(annotation_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # تبدیل به دیکشنری ساده
    gt_landmarks = {}
    for lm in annotation['landmarks']:
        symbol = lm['symbol']
        gt_landmarks[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return gt_landmarks

def calculate_errors(frontend, ground_truth, pixel_size=0.1):
    """محاسبه خطاها"""
    errors = {}
    errors_mm = {}
    
    common_landmarks = set(frontend.keys()) & set(ground_truth.keys())
    
    for lm_name in common_landmarks:
        pred = frontend[lm_name]
        gt = ground_truth[lm_name]
        
        # خطا بر حسب پیکسل
        error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
        
        # خطا بر حسب میلی‌متر
        error_mm = error_px * pixel_size
        
        errors[lm_name] = {
            'pixels': error_px,
            'mm': error_mm,
            'pred': pred,
            'gt': gt,
            'diff_x': pred['x'] - gt['x'],
            'diff_y': pred['y'] - gt['y']
        }
        errors_mm[lm_name] = error_mm
    
    return errors, errors_mm

def main():
    image_id = "cks2ip8fq29yq0yufc4scftj8"
    pixel_size = 0.1  # از CSV
    
    print("="*80)
    print("🔍 مقایسه نتایج Frontend با Ground Truth")
    print("="*80)
    print(f"\nتصویر: {image_id}")
    print(f"Pixel Size: {pixel_size} mm/pixel")
    print(f"اندازه تصویر: 1968 × 2225 پیکسل\n")
    
    # بارگذاری ground truth
    print("📂 بارگذاری annotation واقعی...")
    gt_landmarks = load_ground_truth(image_id)
    
    if gt_landmarks is None:
        print("❌ نتوانست annotation را پیدا کند!")
        return
    
    print(f"✅ {len(gt_landmarks)} لندمارک در annotation واقعی یافت شد\n")
    
    # محاسبه خطاها
    errors, errors_mm = calculate_errors(frontend_results, gt_landmarks, pixel_size)
    
    # نمایش نتایج
    print("="*80)
    print("📊 مقایسه مختصات")
    print("="*80)
    print(f"\n{'Landmark':<10} {'Frontend X':<15} {'Frontend Y':<15} {'GT X':<15} {'GT Y':<15} {'Error (px)':<15} {'Error (mm)':<15}")
    print("-"*100)
    
    sorted_errors = sorted(errors.items(), key=lambda x: x[1]['mm'], reverse=True)
    
    for lm_name, error_info in sorted_errors:
        pred = error_info['pred']
        gt = error_info['gt']
        print(f"{lm_name:<10} {pred['x']:<15.2f} {pred['y']:<15.2f} {gt['x']:<15.2f} {gt['y']:<15.2f} {error_info['pixels']:<15.2f} {error_info['mm']:<15.4f}")
    
    # آمار
    error_values_px = [e['pixels'] for e in errors.values()]
    error_values_mm = [e['mm'] for e in errors.values()]
    
    print("\n" + "="*80)
    print("📈 آمار خطاها")
    print("="*80)
    print(f"\n✅ تعداد لندمارک‌های مقایسه شده: {len(errors)}")
    print(f"\n📊 خطا بر حسب پیکسل:")
    print(f"   میانگین (MRE): {np.mean(error_values_px):.2f} پیکسل")
    print(f"   میانه: {np.median(error_values_px):.2f} پیکسل")
    print(f"   کمینه: {np.min(error_values_px):.2f} پیکسل")
    print(f"   بیشینه: {np.max(error_values_px):.2f} پیکسل")
    print(f"   انحراف معیار: {np.std(error_values_px):.2f} پیکسل")
    
    print(f"\n📊 خطا بر حسب میلی‌متر:")
    print(f"   میانگین (MRE): {np.mean(error_values_mm):.4f} mm")
    print(f"   میانه: {np.median(error_values_mm):.4f} mm")
    print(f"   کمینه: {np.min(error_values_mm):.4f} mm")
    print(f"   بیشینه: {np.max(error_values_mm):.4f} mm")
    print(f"   انحراف معیار: {np.std(error_values_mm):.4f} mm")
    
    # لندمارک‌های با بیشترین خطا
    print("\n" + "="*80)
    print("⚠️  لندمارک‌های با بیشترین خطا (5 تا اول)")
    print("="*80)
    for i, (lm_name, error_info) in enumerate(sorted_errors[:5], 1):
        print(f"\n{i}. {lm_name}:")
        print(f"   خطا: {error_info['mm']:.4f} mm ({error_info['pixels']:.2f} پیکسل)")
        print(f"   Frontend: ({error_info['pred']['x']:.2f}, {error_info['pred']['y']:.2f})")
        print(f"   Ground Truth: ({error_info['gt']['x']:.2f}, {error_info['gt']['y']:.2f})")
        print(f"   تفاوت: X={error_info['diff_x']:.2f}px, Y={error_info['diff_y']:.2f}px")
    
    # لندمارک‌های با کمترین خطا
    print("\n" + "="*80)
    print("✅ لندمارک‌های با کمترین خطا (5 تا اول)")
    print("="*80)
    for i, (lm_name, error_info) in enumerate(sorted_errors[-5:], 1):
        print(f"\n{i}. {lm_name}:")
        print(f"   خطا: {error_info['mm']:.4f} mm ({error_info['pixels']:.2f} پیکسل)")
        print(f"   Frontend: ({error_info['pred']['x']:.2f}, {error_info['pred']['y']:.2f})")
        print(f"   Ground Truth: ({error_info['gt']['x']:.2f}, {error_info['gt']['y']:.2f})")
    
    # SDR calculation
    thresholds_mm = [1.0, 2.0, 2.5, 3.0, 4.0]
    print("\n" + "="*80)
    print("📊 Success Detection Rate (SDR)")
    print("="*80)
    for threshold_mm in thresholds_mm:
        success_count = sum(1 for e_mm in error_values_mm if e_mm <= threshold_mm)
        sdr = (success_count / len(error_values_mm)) * 100
        print(f"   SDR @ {threshold_mm}mm: {sdr:.2f}% ({success_count}/{len(error_values_mm)})")
    
    # ذخیره نتایج
    output = {
        'image_id': image_id,
        'pixel_size': pixel_size,
        'image_size': {'width': 1968, 'height': 2225},
        'mre_pixels': float(np.mean(error_values_px)),
        'mre_mm': float(np.mean(error_values_mm)),
        'errors': {k: {'pixels': float(v['pixels']), 'mm': float(v['mm'])} 
                  for k, v in errors.items()}
    }
    
    with open('comparison_result.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در comparison_result.json ذخیره شد")
    print("="*80)

if __name__ == '__main__':
    main()

