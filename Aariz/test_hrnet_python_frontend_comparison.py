"""
اسکریپت تست کامل HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth

این اسکریپت مدل HRNet را به سه حالت مختلف تست می‌کند:

## حالت‌های تست

1. **python**: فقط تست Python Direct (بدون API)
   - تست مستقیم مدل از طریق Python
   - مقایسه با Ground Truth
   
2. **frontend**: فقط تست Frontend API (شبیه‌سازی فرانت‌اند)
   - تست مدل از طریق API
   - مقایسه با Ground Truth
   
3. **all**: تست هر دو روش و مقایسه نتایج
   - تست Python Direct
   - تست Frontend API
   - مقایسه هر دو با Ground Truth
   - مقایسه دو روش با یکدیگر

## نحوه استفاده

```bash
# تست فقط Python Direct
python test_hrnet_python_frontend_comparison.py --mode python

# تست فقط Frontend API
python test_hrnet_python_frontend_comparison.py --mode frontend

# تست هر دو و مقایسه (پیش‌فرض)
python test_hrnet_python_frontend_comparison.py --mode all

# تست با تصویر دیگر
python test_hrnet_python_frontend_comparison.py --mode all --image-id IMAGE_ID
```
"""

import os
import sys
import json
import base64
import numpy as np
import requests
from PIL import Image
from datetime import datetime

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')
aariz_path = os.path.join(base_dir, 'Aariz')

if cephx_path not in sys.path:
    sys.path.insert(0, cephx_path)

# Add venv site-packages to path
venv_site_packages = os.path.join(cephx_path, 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Configuration
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"  # می‌توانید این را تغییر دهید
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

MODEL_PATH = os.path.join(cephx_path, "model", "hrnet_cephalometric.pth")
HRNET_API_URL = "http://localhost:5000"
PIXEL_SIZE = 0.1  # mm/pixel

# Mapping between HRNet landmarks and Ground Truth landmarks
LANDMARK_MAPPING = {
    'S': 'S',
    'N': 'N',
    'Or': 'Or',
    'Po': 'Po',
    'A': 'A',
    'B': 'B',
    'Pog': 'Pog',
    'Me': 'Me',
    'Gn': 'Gn',
    'Go': 'Go',
    'L1': 'LIT',  # Lower Incisor Tip
    'U1': 'UIT',  # Upper Incisor Tip
    'UL': 'Ls',   # Upper Lip (Labrale superius)
    'LL': 'Li',   # Lower Lip (Labrale inferius)
    'Sn': 'Sn',   # Subnasale
    'PogSoft': 'Pog`',  # Soft Tissue Pogonion
    'PNS': 'PNS',
    'ANS': 'ANS',
    'Ar': 'Ar',
}


def load_ground_truth():
    """بارگذاری Ground Truth از فایل JSON"""
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ ERROR: Ground Truth file not found: {GROUND_TRUTH_PATH}")
        return None
    
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


def image_to_base64(image_path):
    """تبدیل تصویر به base64"""
    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{base64_str}"


def test_python_direct():
    """تست مستقیم مدل از طریق Python (بدون API)"""
    print(f"\n{'='*100}")
    print("🔬 TEST 1: HRNet Python Direct (مستقیم از Python)")
    print(f"{'='*100}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: Model file not found: {MODEL_PATH}")
        return None
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ ERROR: Test image not found: {TEST_IMAGE_PATH}")
        return None
    
    try:
        # Import HRNet service
        from hrnet_production_service import HRNetProductionService
        
        # Check CUDA
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   🔧 Device: {device}")
        
        # Initialize service
        print(f"   📦 Loading model from: {MODEL_PATH}")
        hrnet_service = HRNetProductionService(MODEL_PATH, device=device)
        
        # Load image
        print(f"   📸 Loading image: {TEST_IMAGE_PATH}")
        img = Image.open(TEST_IMAGE_PATH)
        print(f"   📏 Image size: {img.size[0]} × {img.size[1]} pixels")
        
        # Detect landmarks
        print(f"   🔍 Running detection...")
        import time
        start_time = time.time()
        
        result = hrnet_service.detect(TEST_IMAGE_PATH)
        
        processing_time = time.time() - start_time
        
        if not result.get('success', True):
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
            return None
        
        landmarks = result.get('landmarks', {})
        metadata = result.get('metadata', {})
        
        # Convert to simple format
        result_landmarks = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result_landmarks[name] = {
                    'x': float(coords.get('x', 0)),
                    'y': float(coords.get('y', 0)),
                    'confidence': float(coords.get('confidence', 0))
                }
        
        print(f"   ✅ Detection complete!")
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print(f"   📊 Valid landmarks: {len([k for k, v in result_landmarks.items() if v])}/{len(result_landmarks)}")
        print(f"   🎯 Avg confidence: {metadata.get('avg_confidence', 0):.3f}")
        
        return {
            'landmarks': result_landmarks,
            'metadata': {
                **metadata,
                'processing_time': processing_time,
                'method': 'Python Direct',
                'device': device
            }
        }
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"   Make sure you're running from the correct directory")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_frontend_api():
    """تست مدل از طریق API (شبیه‌سازی فرانت‌اند)"""
    print(f"\n{'='*100}")
    print("🔬 TEST 2: HRNet Frontend API (از طریق API - شبیه‌سازی فرانت‌اند)")
    print(f"{'='*100}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ ERROR: Test image not found: {TEST_IMAGE_PATH}")
        return None
    
    try:
        # Convert image to base64
        print(f"   📸 Converting image to base64...")
        base64_image = image_to_base64(TEST_IMAGE_PATH)
        
        # Call API
        print(f"   🌐 Calling API: {HRNET_API_URL}/detect")
        import time
        start_time = time.time()
        
        response = requests.post(
            f"{HRNET_API_URL}/detect",
            json={'image_base64': base64_image},
            timeout=60
        )
        
        processing_time = time.time() - start_time
        
        if response.status_code != 200:
            print(f"❌ API Error: HTTP {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return None
        
        data = response.json()
        
        if not data.get('success', True):
            print(f"❌ API failed: {data.get('error', 'Unknown error')}")
            return None
        
        landmarks = data.get('landmarks', {})
        metadata = data.get('metadata', {})
        
        # Convert to simple format
        result_landmarks = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result_landmarks[name] = {
                    'x': float(coords.get('x', 0)),
                    'y': float(coords.get('y', 0)),
                    'confidence': float(coords.get('confidence', 0))
                }
        
        print(f"   ✅ API Response received!")
        print(f"   ⏱️  Processing time: {processing_time:.3f}s")
        print(f"   📊 Valid landmarks: {len([k for k, v in result_landmarks.items() if v])}/{len(result_landmarks)}")
        print(f"   🎯 Avg confidence: {metadata.get('avg_confidence', 0):.3f}")
        print(f"   🔧 Model type: {metadata.get('model_type', 'N/A')}")
        
        return {
            'landmarks': result_landmarks,
            'metadata': {
                **metadata,
                'processing_time': processing_time,
                'method': 'Frontend API'
            }
        }
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error! API service may not be running")
        print(f"   💡 Tip: Start the API server:")
        print(f"      cd {cephx_path}")
        print(f"      python app_hrnet_real.py")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def calculate_errors(predicted, ground_truth, method_name):
    """محاسبه خطاها بین پیش‌بینی و Ground Truth"""
    errors = []
    
    for hrnet_name, gt_name in LANDMARK_MAPPING.items():
        if hrnet_name in predicted and gt_name in ground_truth:
            pred = predicted[hrnet_name]
            gt = ground_truth[gt_name]
            
            # Calculate Euclidean distance
            error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            error_mm = error_px * PIXEL_SIZE
            
            errors.append({
                'name': hrnet_name,
                'gt_name': gt_name,
                'error_px': error_px,
                'error_mm': error_mm,
                'pred': pred,
                'gt': gt,
                'diff_x': pred['x'] - gt['x'],
                'diff_y': pred['y'] - gt['y'],
                'confidence': pred.get('confidence', 0)
            })
    
    return errors


def print_comparison_table(errors, method_name):
    """چاپ جدول مقایسه"""
    print(f"\n{'='*100}")
    print(f"📊 مقایسه نتایج {method_name} با Ground Truth")
    print(f"{'='*100}")
    print(f"\n{'Landmark':<12} {'Pred X':<12} {'Pred Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<10} {'Diff Y':<10} {'Error (px)':<12} {'Error (mm)':<12} {'Conf':<8}")
    print("-"*100)
    
    errors_sorted = sorted(errors, key=lambda x: x['error_mm'], reverse=True)
    
    for err in errors_sorted:
        pred = err['pred']
        gt = err['gt']
        print(f"{err['name']:<12} {pred['x']:<12.2f} {pred['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {err['diff_x']:<10.2f} {err['diff_y']:<10.2f} {err['error_px']:<12.2f} {err['error_mm']:<12.4f} {err['confidence']:<8.3f}")
    
    # Statistics
    error_values_mm = [e['error_mm'] for e in errors]
    
    print(f"\n{'='*100}")
    print(f"📈 آمار خطاها ({method_name})")
    print(f"{'='*100}")
    print(f"\n✅ تعداد لندمارک‌های مقایسه شده: {len(errors)}")
    print(f"\n📊 خطا بر حسب میلی‌متر:")
    print(f"   میانگین (MRE): {np.mean(error_values_mm):.4f} mm")
    print(f"   میانه: {np.median(error_values_mm):.4f} mm")
    print(f"   کمینه: {np.min(error_values_mm):.4f} mm")
    print(f"   بیشینه: {np.max(error_values_mm):.4f} mm")
    print(f"   انحراف معیار: {np.std(error_values_mm):.4f} mm")
    
    # SDR (Success Detection Rate)
    thresholds_mm = [1.0, 2.0, 2.5, 3.0, 4.0]
    print(f"\n{'='*100}")
    print(f"📊 Success Detection Rate (SDR) - {method_name}")
    print(f"{'='*100}")
    for threshold_mm in thresholds_mm:
        success = sum(1 for e_mm in error_values_mm if e_mm <= threshold_mm)
        sdr = (success / len(error_values_mm)) * 100
        print(f"   SDR @ {threshold_mm}mm: {sdr:.2f}% ({success}/{len(error_values_mm)})")
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'median_mm': float(np.median(error_values_mm)),
        'min_mm': float(np.min(error_values_mm)),
        'max_mm': float(np.max(error_values_mm)),
        'std_mm': float(np.std(error_values_mm)),
        'errors': errors,
        'sdr': {f'{t}mm': float(sum(1 for e_mm in error_values_mm if e_mm <= t) / len(error_values_mm) * 100) 
                for t in thresholds_mm}
    }


def compare_methods(python_stats, api_stats):
    """مقایسه نتایج Python Direct و Frontend API"""
    print(f"\n{'='*100}")
    print("🔄 مقایسه Python Direct vs Frontend API")
    print(f"{'='*100}")
    
    print(f"\n{'Metric':<30} {'Python Direct':<20} {'Frontend API':<20} {'Difference':<20}")
    print("-"*90)
    
    metrics = [
        ('MRE (mm)', 'mre_mm'),
        ('Median Error (mm)', 'median_mm'),
        ('Min Error (mm)', 'min_mm'),
        ('Max Error (mm)', 'max_mm'),
        ('Std Dev (mm)', 'std_mm'),
    ]
    
    for metric_name, metric_key in metrics:
        python_val = python_stats.get(metric_key, 0)
        api_val = api_stats.get(metric_key, 0)
        diff = python_val - api_val
        print(f"{metric_name:<30} {python_val:<20.4f} {api_val:<20.4f} {diff:<20.4f}")
    
    # SDR comparison
    print(f"\n{'='*100}")
    print("📊 SDR Comparison")
    print(f"{'='*100}")
    print(f"\n{'Threshold':<15} {'Python Direct':<20} {'Frontend API':<20} {'Difference':<20}")
    print("-"*75)
    
    python_sdr = python_stats.get('sdr', {})
    api_sdr = api_stats.get('sdr', {})
    
    for threshold in ['1.0mm', '2.0mm', '2.5mm', '3.0mm', '4.0mm']:
        python_val = python_sdr.get(threshold, 0)
        api_val = api_sdr.get(threshold, 0)
        diff = python_val - api_val
        print(f"{threshold:<15} {python_val:<20.2f}% {api_val:<20.2f}% {diff:<20.2f}%")
    
    # Determine which is better
    python_mre = python_stats.get('mre_mm', float('inf'))
    api_mre = api_stats.get('mre_mm', float('inf'))
    
    print(f"\n{'='*100}")
    if python_mre < api_mre:
        print(f"✅ Python Direct بهتر است! (MRE: {python_mre:.4f}mm vs {api_mre:.4f}mm)")
    elif api_mre < python_mre:
        print(f"✅ Frontend API بهتر است! (MRE: {api_mre:.4f}mm vs {python_mre:.4f}mm)")
    else:
        print(f"⚖️  نتایج تقریباً یکسان هستند! (MRE: {python_mre:.4f}mm)")
    print(f"{'='*100}")


def main():
    """تابع اصلی"""
    import argparse
    
    # Declare globals first
    global TEST_IMAGE_ID, TEST_IMAGE_PATH, GROUND_TRUTH_PATH
    
    parser = argparse.ArgumentParser(description='تست HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth')
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['python', 'frontend', 'all', 'compare'],
                       help='حالت تست: python (فقط Python Direct), frontend (فقط Frontend API), all (هر دو), compare (فقط مقایسه)')
    parser.add_argument('--image-id', type=str, default=TEST_IMAGE_ID,
                       help='ID تصویر تست')
    
    args = parser.parse_args()
    
    # Update test image if provided
    if args.image_id != TEST_IMAGE_ID:
        TEST_IMAGE_ID = args.image_id
        TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
        GROUND_TRUTH_PATH = os.path.join(
            base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
            "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
        )
    
    print("="*100)
    print("🧪 تست کامل HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth")
    print("="*100)
    print(f"\n📸 تصویر تست: {TEST_IMAGE_ID}")
    print(f"📂 مسیر تصویر: {TEST_IMAGE_PATH}")
    print(f"📂 مسیر Ground Truth: {GROUND_TRUTH_PATH}")
    print(f"📂 مسیر مدل: {MODEL_PATH}")
    print(f"🌐 API URL: {HRNET_API_URL}")
    print(f"📏 Pixel Size: {PIXEL_SIZE} mm/pixel")
    print(f"\n🔧 حالت تست: {args.mode}")
    print("   - python: فقط تست Python Direct")
    print("   - frontend: فقط تست Frontend API")
    print("   - all: تست هر دو و مقایسه")
    print("   - compare: فقط مقایسه نتایج (نیاز به فایل JSON قبلی)")
    
    # Check files exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\n❌ ERROR: Ground Truth not found: {GROUND_TRUTH_PATH}")
        return
    
    # Load ground truth (if needed)
    gt_landmarks = None
    if args.mode in ['python', 'frontend', 'all', 'compare']:
        print(f"\n📂 بارگذاری Ground Truth...")
        gt_landmarks = load_ground_truth()
        if gt_landmarks is None:
            return
        print(f"   ✅ {len(gt_landmarks)} لندمارک در Ground Truth یافت شد")
    
    python_result = None
    api_result = None
    python_stats = None
    api_stats = None
    
    # Test based on mode
    if args.mode == 'python' or args.mode == 'all':
        # Test 1: Python Direct
        python_result = test_python_direct()
        
        if python_result and gt_landmarks:
            print(f"\n📊 محاسبه خطاها برای Python Direct...")
            python_errors = calculate_errors(python_result['landmarks'], gt_landmarks, 'Python Direct')
            python_stats = print_comparison_table(python_errors, 'Python Direct')
    
    if args.mode == 'frontend' or args.mode == 'all':
        # Test 2: Frontend API
        api_result = test_frontend_api()
        
        if api_result and gt_landmarks:
            print(f"\n📊 محاسبه خطاها برای Frontend API...")
            api_errors = calculate_errors(api_result['landmarks'], gt_landmarks, 'Frontend API')
            api_stats = print_comparison_table(api_errors, 'Frontend API')
    
    if args.mode == 'all':
        # Compare methods
        if python_stats and api_stats:
            compare_methods(python_stats, api_stats)
    
    # Save results
    output = {
        'image_id': TEST_IMAGE_ID,
        'test_timestamp': datetime.now().isoformat(),
        'test_mode': args.mode,
        'pixel_size': PIXEL_SIZE,
        'ground_truth': {
            'num_landmarks': len(gt_landmarks) if gt_landmarks else 0,
            'landmarks': gt_landmarks if gt_landmarks else None
        },
        'python_direct': {
            'success': python_result is not None,
            'stats': python_stats,
            'metadata': python_result['metadata'] if python_result else None,
            'landmarks': python_result['landmarks'] if python_result else None
        },
        'frontend_api': {
            'success': api_result is not None,
            'stats': api_stats,
            'metadata': api_result['metadata'] if api_result else None,
            'landmarks': api_result['landmarks'] if api_result else None
        }
    }
    
    output_file = f'hrnet_test_results_{args.mode}_{TEST_IMAGE_ID}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج کامل در {output_file} ذخیره شد")
    
    # Final summary
    print(f"\n{'='*100}")
    print("📋 خلاصه نهایی")
    print(f"{'='*100}")
    
    if python_stats:
        print(f"\n✅ Python Direct:")
        print(f"   MRE: {python_stats['mre_mm']:.4f} mm")
        print(f"   SDR @ 2mm: {python_stats['sdr'].get('2.0mm', 0):.2f}%")
        if python_result:
            print(f"   Processing time: {python_result['metadata'].get('processing_time', 0):.3f}s")
    
    if api_stats:
        print(f"\n✅ Frontend API:")
        print(f"   MRE: {api_stats['mre_mm']:.4f} mm")
        print(f"   SDR @ 2mm: {api_stats['sdr'].get('2.0mm', 0):.2f}%")
        if api_result:
            print(f"   Processing time: {api_result['metadata'].get('processing_time', 0):.3f}s")
    
    if python_stats and api_stats:
        print(f"\n🔄 مقایسه:")
        python_mre = python_stats['mre_mm']
        api_mre = api_stats['mre_mm']
        if python_mre < api_mre:
            print(f"   ✅ Python Direct بهتر است! (MRE: {python_mre:.4f}mm vs {api_mre:.4f}mm)")
        elif api_mre < python_mre:
            print(f"   ✅ Frontend API بهتر است! (MRE: {api_mre:.4f}mm vs {python_mre:.4f}mm)")
        else:
            print(f"   ⚖️  نتایج تقریباً یکسان هستند! (MRE: {python_mre:.4f}mm)")
    
    print("\n" + "="*100)


if __name__ == '__main__':
    main()

