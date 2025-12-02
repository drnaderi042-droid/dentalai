"""
تحلیل خطای هر لندمارک از نتایج تست مدل‌های Aariz
"""

import os
import sys
import json
import numpy as np
from collections import defaultdict

# Add paths
base_dir = os.path.dirname(os.path.abspath(__file__))
aariz_path = os.path.join(base_dir, 'Aariz')

if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

from inference import LandmarkPredictor
from PIL import Image
import torch
import math

# Configuration
VALID_CEPHALOGRAMS_PATH = os.path.join(aariz_path, "Aariz", "valid", "Cephalograms")
VALID_ANNOTATIONS_PATH = os.path.join(
    aariz_path, "Aariz", "valid", "Annotations", 
    "Cephalometric Landmarks", "Senior Orthodontists"
)
CHECKPOINTS = {
    '256': os.path.join(aariz_path, "checkpoint_best_256.pth"),
    '512': os.path.join(aariz_path, "checkpoint_best_512.pth"),
    '768': os.path.join(aariz_path, "checkpoint_best_768.pth"),
}
PIXEL_SIZE = 0.1
NUM_TEST_IMAGES = 10

LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

def get_image_files(num_images=10):
    """دریافت لیست تصاویر"""
    if not os.path.exists(VALID_CEPHALOGRAMS_PATH):
        return []
    
    extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
    image_files = []
    
    for ext in extensions:
        image_files.extend([f for f in os.listdir(VALID_CEPHALOGRAMS_PATH) if f.endswith(ext)])
    
    image_ids = sorted(list(set([os.path.splitext(f)[0] for f in image_files])))
    return image_ids[:num_images]

def load_ground_truth(image_id):
    """بارگذاری Ground Truth"""
    json_path = os.path.join(VALID_ANNOTATIONS_PATH, f"{image_id}.json")
    
    if not os.path.exists(json_path):
        for mode in ["train", "test"]:
            alt_path = os.path.join(
                aariz_path, "Aariz", mode, "Annotations",
                "Cephalometric Landmarks", "Senior Orthodontists", f"{image_id}.json"
            )
            if os.path.exists(alt_path):
                json_path = alt_path
                break
    
    if not os.path.exists(json_path):
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    gt_landmarks = {}
    for lm in annotation.get('landmarks', []):
        symbol = lm['symbol']
        gt_landmarks[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return gt_landmarks

def find_image_path(image_id):
    """یافتن مسیر تصویر"""
    for mode in ["valid", "train", "test"]:
        mode_path = os.path.join(aariz_path, "Aariz", mode, "Cephalograms")
        if not os.path.exists(mode_path):
            continue
            
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG', '.BMP']
        for ext in extensions:
            img_path = os.path.join(mode_path, f"{image_id}{ext}")
            if os.path.exists(img_path):
                return img_path
    return None

def predict_no_tta(predictor, image, target_size):
    """پیش‌بینی بدون TTA"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    result = predictor.predict(image, target_size=target_size)
    return result.get('landmarks', {})

def predict_with_tta(predictor, image, target_size):
    """پیش‌بینی با TTA"""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size
    w, h = original_size
    
    pred_orig = predictor.predict(image, target_size)
    landmarks_orig = pred_orig.get('landmarks', {})
    
    img_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    pred_flip = predictor.predict(img_flip, target_size)
    landmarks_flip = pred_flip.get('landmarks', {})
    
    landmarks_flip_corrected = {}
    for name, coords in landmarks_flip.items():
        if coords is not None:
            landmarks_flip_corrected[name] = {
                'x': w - coords['x'],
                'y': coords['y'],
                'confidence': coords.get('confidence', 0.9)
            }
    
    landmarks_avg = {}
    all_symbols = set(list(landmarks_orig.keys()) + list(landmarks_flip_corrected.keys()))
    
    for symbol in all_symbols:
        orig = landmarks_orig.get(symbol)
        flip = landmarks_flip_corrected.get(symbol)
        
        if orig is not None and flip is not None:
            landmarks_avg[symbol] = {
                'x': (orig['x'] + flip['x']) / 2.0,
                'y': (orig['y'] + flip['y']) / 2.0,
                'confidence': (orig.get('confidence', 0.9) + flip.get('confidence', 0.9)) / 2.0
            }
        elif orig is not None:
            landmarks_avg[symbol] = orig
        elif flip is not None:
            landmarks_avg[symbol] = flip
    
    return landmarks_avg

def calculate_landmark_errors(predicted_landmarks, ground_truth_landmarks):
    """محاسبه خطا برای هر لندمارک"""
    errors = {}
    
    for symbol in LANDMARK_SYMBOLS:
        if symbol not in ground_truth_landmarks:
            continue
        
        gt = ground_truth_landmarks[symbol]
        
        if symbol in predicted_landmarks and predicted_landmarks[symbol] is not None:
            pred = predicted_landmarks[symbol]
            error_px = math.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            error_mm = error_px * PIXEL_SIZE
            errors[symbol] = error_mm
        else:
            errors[symbol] = None
    
    return errors

def analyze_all_models():
    """تحلیل همه مدل‌ها"""
    import sys
    import io
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("="*80)
    print("تحليل خطاي لندمارک‌ها")
    print("="*80)
    
    # دریافت تصاویر
    image_ids = get_image_files(num_images=NUM_TEST_IMAGES)
    print(f"تعداد تصاویر: {len(image_ids)}")
    
    # ذخیره خطاهای همه لندمارک‌ها برای همه مدل‌ها
    all_landmark_errors = defaultdict(lambda: defaultdict(list))
    
    for model_size in ['256', '512', '768']:
        checkpoint_path = CHECKPOINTS[model_size]
        if not os.path.exists(checkpoint_path):
            print(f"\n[SKIP] مدل {model_size} - checkpoint یافت نشد")
            continue
        
        print(f"\n{'='*80}")
        print(f"تحلیل مدل {model_size}x{model_size}")
        print(f"{'='*80}")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        predictor = LandmarkPredictor(
            checkpoint_path=checkpoint_path,
            model_name='hrnet',
            device=device
        )
        
        target_size = (int(model_size), int(model_size))
        
        # تست بدون TTA
        print(f"  بدون TTA...")
        for image_id in image_ids:
            gt_landmarks = load_ground_truth(image_id)
            if gt_landmarks is None:
                continue
            
            image_path = find_image_path(image_id)
            if image_path is None:
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                predicted_landmarks = predict_no_tta(predictor, image, target_size)
                errors = calculate_landmark_errors(predicted_landmarks, gt_landmarks)
                
                for symbol, error_mm in errors.items():
                    if error_mm is not None:
                        all_landmark_errors[f"{model_size}_no_tta"][symbol].append(error_mm)
            except Exception as e:
                continue
        
        # تست با TTA
        print(f"  با TTA...")
        for image_id in image_ids:
            gt_landmarks = load_ground_truth(image_id)
            if gt_landmarks is None:
                continue
            
            image_path = find_image_path(image_id)
            if image_path is None:
                continue
            
            try:
                image = Image.open(image_path).convert('RGB')
                predicted_landmarks = predict_with_tta(predictor, image, target_size)
                errors = calculate_landmark_errors(predicted_landmarks, gt_landmarks)
                
                for symbol, error_mm in errors.items():
                    if error_mm is not None:
                        all_landmark_errors[f"{model_size}_tta"][symbol].append(error_mm)
            except Exception as e:
                continue
    
    # محاسبه آمار برای هر لندمارک
    landmark_stats = defaultdict(lambda: defaultdict(dict))
    
    for config_name, landmark_errors in all_landmark_errors.items():
        for symbol, errors in landmark_errors.items():
            if errors:
                landmark_stats[config_name][symbol] = {
                    'mean': np.mean(errors),
                    'std': np.std(errors),
                    'min': np.min(errors),
                    'max': np.max(errors),
                    'count': len(errors)
                }
    
    # نمایش نتایج
    print("\n" + "="*80)
    print("نتایج تحلیل خطای لندمارک‌ها")
    print("="*80)
    
    # برای هر مدل، لندمارک‌های با بیشترین خطا
    for config_name in sorted(landmark_stats.keys()):
        print(f"\n{'='*80}")
        print(f"مدل: {config_name}")
        print(f"{'='*80}")
        
        stats = landmark_stats[config_name]
        if not stats:
            continue
        
        # مرتب‌سازی بر اساس میانگین خطا
        sorted_landmarks = sorted(stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print(f"\n⚠️  بدترین لندمارک‌ها (بیشترین خطا):")
        print(f"{'لندمارک':<10} {'میانگین (mm)':<15} {'Std (mm)':<15} {'Min (mm)':<12} {'Max (mm)':<12}")
        print("-" * 80)
        
        for i, (symbol, stat) in enumerate(sorted_landmarks[:10]):
            print(f"{symbol:<10} {stat['mean']:<15.3f} {stat['std']:<15.3f} "
                  f"{stat['min']:<12.3f} {stat['max']:<12.3f}")
        
        print(f"\n✅ بهترین لندمارک‌ها (کمترین خطا):")
        print(f"{'لندمارک':<10} {'میانگین (mm)':<15} {'Std (mm)':<15} {'Min (mm)':<12} {'Max (mm)':<12}")
        print("-" * 80)
        
        for i, (symbol, stat) in enumerate(sorted_landmarks[-10:]):
            print(f"{symbol:<10} {stat['mean']:<15.3f} {stat['std']:<15.3f} "
                  f"{stat['min']:<12.3f} {stat['max']:<12.3f}")
    
    # محاسبه میانگین کلی برای هر لندمارک در همه مدل‌ها
    print("\n" + "="*80)
    print("میانگین خطای هر لندمارک در تمام مدل‌ها")
    print("="*80)
    
    overall_landmark_errors = defaultdict(list)
    for config_name, landmark_errors in all_landmark_errors.items():
        for symbol, errors in landmark_errors.items():
            overall_landmark_errors[symbol].extend(errors)
    
    overall_stats = {}
    for symbol, errors in overall_landmark_errors.items():
        if errors:
            overall_stats[symbol] = {
                'mean': np.mean(errors),
                'std': np.std(errors),
                'min': np.min(errors),
                'max': np.max(errors),
                'count': len(errors)
            }
    
    sorted_overall = sorted(overall_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print(f"\n⚠️  لندمارک‌های با بیشترین خطا (میانگین در تمام مدل‌ها):")
    print(f"{'رتبه':<8} {'لندمارک':<10} {'میانگین (mm)':<15} {'Std (mm)':<15} {'Max (mm)':<12}")
    print("-" * 80)
    
    for i, (symbol, stat) in enumerate(sorted_overall[:15]):
        print(f"{i+1:<8} {symbol:<10} {stat['mean']:<15.3f} {stat['std']:<15.3f} "
              f"{stat['max']:<12.3f}")
    
    print(f"\n✅ لندمارک‌های با کمترین خطا (میانگین در تمام مدل‌ها):")
    print(f"{'رتبه':<8} {'لندمارک':<10} {'میانگین (mm)':<15} {'Std (mm)':<15} {'Max (mm)':<12}")
    print("-" * 80)
    
    for i, (symbol, stat) in enumerate(sorted_overall[-15:]):
        rank = len(sorted_overall) - i
        print(f"{rank:<8} {symbol:<10} {stat['mean']:<15.3f} {stat['std']:<15.3f} "
              f"{stat['max']:<12.3f}")
    
    # ذخیره نتایج
    output_data = {
        'landmark_stats': {
            config: {
                symbol: {
                    'mean': float(stats['mean']),
                    'std': float(stats['std']),
                    'min': float(stats['min']),
                    'max': float(stats['max']),
                    'count': stats['count']
                }
                for symbol, stats in landmark_data.items()
            }
            for config, landmark_data in landmark_stats.items()
        },
        'overall_stats': {
            symbol: {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'min': float(stats['min']),
                'max': float(stats['max']),
                'count': stats['count']
            }
            for symbol, stats in overall_stats.items()
        }
    }
    
    output_file = "landmark_errors_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] نتایج در فایل ذخیره شد: {output_file}")
    print("="*80)

if __name__ == "__main__":
    analyze_all_models()

