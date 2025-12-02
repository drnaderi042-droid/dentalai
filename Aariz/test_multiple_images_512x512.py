"""
تست مدل 512×512 روی چند تصویر مختلف از validation set
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
    return sorted(image_ids)[:10]  # Test first 10 images

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

def calculate_errors(predicted, ground_truth):
    """Calculate errors"""
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

def main():
    print("="*100)
    print("🧪 تست مدل 512×512 روی چند تصویر Validation Set")
    print("="*100)
    
    # Get image IDs
    image_ids = get_image_files()
    print(f"\n📸 تعداد تصاویر برای تست: {len(image_ids)}")
    
    if not image_ids:
        print("❌ هیچ تصویری یافت نشد!")
        return
    
    # Load model
    print(f"\n🤖 بارگذاری مدل...")
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predictor = LandmarkPredictor(CHECKPOINT_PATH, model_name='hrnet', device=device)
    print(f"   ✅ Model loaded on {device}")
    
    # Test on each image
    all_results = []
    
    print(f"\n{'='*100}")
    print("🔬 تست روی تصاویر...")
    print(f"{'='*100}")
    
    for image_id in tqdm(image_ids, desc="Testing images"):
        # Find image path
        img_path = find_image_path(image_id)
        if not img_path:
            continue
        
        # Load ground truth
        gt_landmarks = load_ground_truth(image_id)
        if not gt_landmarks:
            continue
        
        try:
            # Load and predict
            img = Image.open(img_path).convert('RGB')
            result = predictor.predict(img, target_size=(512, 512))
            
            # Calculate errors
            errors = calculate_errors(result['landmarks'], gt_landmarks)
            
            if errors:
                mre = np.mean([e['error_mm'] for e in errors])
                sdr_2mm = sum(1 for e in errors if e['error_mm'] <= 2.0) / len(errors) * 100
                
                all_results.append({
                    'image_id': image_id,
                    'mre_mm': float(mre),
                    'sdr_2mm': float(sdr_2mm),
                    'num_landmarks': len(errors)
                })
                
        except Exception as e:
            print(f"\n⚠️  خطا در پردازش {image_id}: {e}")
            continue
    
    if not all_results:
        print("\n❌ هیچ نتیجه‌ای بدست نیامد!")
        return
    
    # Statistics
    mre_values = [r['mre_mm'] for r in all_results]
    sdr_values = [r['sdr_2mm'] for r in all_results]
    
    print(f"\n{'='*100}")
    print("📊 نتایج کلی")
    print(f"{'='*100}")
    print(f"\n✅ تعداد تصاویر تست شده: {len(all_results)}")
    print(f"\n📊 MRE (میانگین روی همه تصاویر):")
    print(f"   میانگین: {np.mean(mre_values):.4f} mm")
    print(f"   میانه: {np.median(mre_values):.4f} mm")
    print(f"   کمینه: {np.min(mre_values):.4f} mm")
    print(f"   بیشینه: {np.max(mre_values):.4f} mm")
    print(f"   انحراف معیار: {np.std(mre_values):.4f} mm")
    
    print(f"\n📊 SDR @ 2mm (میانگین روی همه تصاویر):")
    print(f"   میانگین: {np.mean(sdr_values):.2f}%")
    print(f"   میانه: {np.median(sdr_values):.2f}%")
    print(f"   کمینه: {np.min(sdr_values):.2f}%")
    print(f"   بیشینه: {np.max(sdr_values):.2f}%")
    
    # Detailed results
    print(f"\n{'='*100}")
    print("📋 نتایج تفصیلی هر تصویر")
    print(f"{'='*100}")
    print(f"\n{'Image ID':<25} {'MRE (mm)':<15} {'SDR @ 2mm (%)':<20}")
    print("-"*60)
    
    for r in sorted(all_results, key=lambda x: x['mre_mm']):
        print(f"{r['image_id']:<25} {r['mre_mm']:<15.4f} {r['sdr_2mm']:<20.2f}")
    
    # Compare with training
    print(f"\n{'='*100}")
    print("📊 مقایسه با نتایج Training")
    print(f"{'='*100}")
    print(f"   Training (Validation Set): MRE=1.41mm, SDR @ 2mm=80.25%")
    print(f"   Test (این {len(all_results)} تصویر): MRE={np.mean(mre_values):.4f}mm, SDR @ 2mm={np.mean(sdr_values):.2f}%")
    
    diff_mre = np.mean(mre_values) - 1.41
    diff_sdr = np.mean(sdr_values) - 80.25
    
    if abs(diff_mre) < 0.3 and abs(diff_sdr) < 5:
        print(f"\n✅ نتایج سازگار با training هستند!")
    else:
        print(f"\n⚠️  تفاوت قابل توجه:")
        print(f"   MRE: {diff_mre:+.4f} mm")
        print(f"   SDR: {diff_sdr:+.2f}%")
    
    # Save results
    output = {
        'num_images': len(all_results),
        'overall_stats': {
            'mean_mre': float(np.mean(mre_values)),
            'median_mre': float(np.median(mre_values)),
            'mean_sdr_2mm': float(np.mean(sdr_values)),
            'median_sdr_2mm': float(np.median(sdr_values))
        },
        'per_image_results': all_results
    }
    
    output_file = 'test_multiple_images_512x512_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

