"""
مقایسه checkpoint قدیمی با جدید (بعد از training با difficult landmarks only)
"""

import os
import sys
import json
import numpy as np
from PIL import Image
import math
from tqdm import tqdm
import torch

# Add paths
base_dir = os.path.dirname(os.path.abspath(__file__))
aariz_path = os.path.join(base_dir, 'Aariz')

if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

from inference import LandmarkPredictor

# Configuration
VALID_CEPHALOGRAMS_PATH = os.path.join(aariz_path, "Aariz", "valid", "Cephalograms")
VALID_ANNOTATIONS_PATH = os.path.join(
    aariz_path, "Aariz", "valid", "Annotations", 
    "Cephalometric Landmarks", "Senior Orthodontists"
)

CHECKPOINTS = {
    'old': os.path.join(aariz_path, "checkpoint_best_768.pth"),
    'new': os.path.join(aariz_path, "checkpoints", "checkpoint_best.pth"),
}

PIXEL_SIZE = 0.1
NUM_TEST_IMAGES = 10  # برای تست سریع

LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

DIFFICULT_LANDMARKS = ['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT', 'LPM', 'Or', 'Co', 'PNS', 'Po', 'ANS']

def get_test_images(num_images=10):
    """دریافت تصاویر تست"""
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
    img_path = os.path.join(VALID_CEPHALOGRAMS_PATH, f"{image_id}.png")
    if os.path.exists(img_path):
        return img_path
    
    for ext in ['.jpg', '.jpeg', '.bmp']:
        img_path = os.path.join(VALID_CEPHALOGRAMS_PATH, f"{image_id}{ext}")
        if os.path.exists(img_path):
            return img_path
    
    return None

def test_checkpoint(checkpoint_path, image_ids):
    """تست یک checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = LandmarkPredictor(
        checkpoint_path=checkpoint_path,
        model_name='hrnet',
        device=device
    )
    
    all_errors = {}
    all_errors_difficult = {}
    
    for image_id in tqdm(image_ids, desc=f"Testing {os.path.basename(checkpoint_path)}"):
        gt_landmarks = load_ground_truth(image_id)
        if gt_landmarks is None:
            continue
        
        image_path = find_image_path(image_id)
        if image_path is None:
            continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            result = predictor.predict(image, target_size=(768, 768))
            predicted_landmarks = result.get('landmarks', {})
        except Exception as e:
            continue
        
        # محاسبه خطا برای هر لندمارک
        for symbol in LANDMARK_SYMBOLS:
            if symbol not in gt_landmarks:
                continue
            
            gt = gt_landmarks[symbol]
            
            if symbol in predicted_landmarks and predicted_landmarks[symbol] is not None:
                pred = predicted_landmarks[symbol]
                error_px = math.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
                error_mm = error_px * PIXEL_SIZE
                
                if symbol not in all_errors:
                    all_errors[symbol] = []
                all_errors[symbol].append(error_mm)
    
    # محاسبه میانگین برای هر لندمارک
    results = {}
    difficult_results = {}
    
    for symbol in LANDMARK_SYMBOLS:
        if symbol in all_errors and len(all_errors[symbol]) > 0:
            mean_error = np.mean(all_errors[symbol])
            results[symbol] = mean_error
            
            if symbol in DIFFICULT_LANDMARKS:
                difficult_results[symbol] = mean_error
    
    return results, difficult_results

def main():
    """تابع اصلی"""
    import sys
    import io
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    
    print("="*80)
    print("مقايسه Checkpoint قديمي با جديد (بعد از Training با Difficult Landmarks Only)")
    print("="*80)
    
    image_ids = get_test_images(num_images=NUM_TEST_IMAGES)
    print(f"\nتعداد تصاوير تست: {len(image_ids)}")
    
    # تست checkpoint قدیمی
    print("\n" + "="*80)
    print("تست Checkpoint قديمي (checkpoint_best_768.pth):")
    print("="*80)
    old_results, old_difficult = test_checkpoint(CHECKPOINTS['old'], image_ids)
    
    if old_results:
        old_all_mre = np.mean(list(old_results.values()))
        old_difficult_mre = np.mean(list(old_difficult.values())) if old_difficult else 0
        print(f"\nMRE (تمام لندمارک‌ها): {old_all_mre:.3f} mm")
        print(f"MRE (فقط مشکل‌دارها): {old_difficult_mre:.3f} mm")
    
    # تست checkpoint جدید
    print("\n" + "="*80)
    print("تست Checkpoint جديد (checkpoints/checkpoint_best.pth):")
    print("="*80)
    new_results, new_difficult = test_checkpoint(CHECKPOINTS['new'], image_ids)
    
    if new_results:
        new_all_mre = np.mean(list(new_results.values()))
        new_difficult_mre = np.mean(list(new_difficult.values())) if new_difficult else 0
        print(f"\nMRE (تمام لندمارک‌ها): {new_all_mre:.3f} mm")
        print(f"MRE (فقط مشکل‌دارها): {new_difficult_mre:.3f} mm")
    
    # مقایسه
    if old_results and new_results:
        print("\n" + "="*80)
        print("نتايج مقايسه:")
        print("="*80)
        
        print(f"\n📊 MRE (تمام لندمارک‌ها):")
        print(f"   قديمي: {old_all_mre:.3f} mm")
        print(f"   جديد: {new_all_mre:.3f} mm")
        improvement_all = ((old_all_mre - new_all_mre) / old_all_mre) * 100
        print(f"   بهبود: {improvement_all:+.2f}%")
        
        print(f"\n📊 MRE (فقط لندمارک‌های مشکل‌دار):")
        print(f"   قديمي: {old_difficult_mre:.3f} mm")
        print(f"   جديد: {new_difficult_mre:.3f} mm")
        improvement_difficult = ((old_difficult_mre - new_difficult_mre) / old_difficult_mre) * 100 if old_difficult_mre > 0 else 0
        print(f"   بهبود: {improvement_difficult:+.2f}%")
        
        print(f"\n📋 بهبود لندمارک‌های مشکل‌دار:")
        print(f"{'لندمارک':<10} {'قديمي (mm)':<15} {'جديد (mm)':<15} {'بهبود (%)':<15}")
        print("-" * 80)
        
        for symbol in DIFFICULT_LANDMARKS:
            if symbol in old_results and symbol in new_results:
                old_err = old_results[symbol]
                new_err = new_results[symbol]
                improvement = ((old_err - new_err) / old_err) * 100 if old_err > 0 else 0
                print(f"{symbol:<10} {old_err:<15.3f} {new_err:<15.3f} {improvement:+.2f}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()















