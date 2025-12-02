"""
اسکریپت تست کامل برای مدل HRNet
- تست از API
- تست از Frontend (simulated)
- مقایسه با Ground Truth
"""

import os
import sys
import json
import base64
import numpy as np
from PIL import Image
import requests

# Add paths
aariz_path = os.path.join(os.path.dirname(__file__))
if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

# Test image
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
# Use absolute paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

# HRNet API URL
HRNET_API_URL = "http://localhost:5000"
AARIZ_API_URL = "http://localhost:5001"

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

def image_to_base64(image_path):
    """Convert image to base64"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    
    base64_str = base64.b64encode(image_bytes).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    
    if ext == '.jpg' or ext == '.jpeg':
        mime_type = 'image/jpeg'
    elif ext == '.png':
        mime_type = 'image/png'
    else:
        mime_type = 'image/png'
    
    return f"data:{mime_type};base64,{base64_str}", base64_str

def test_hrnet_api(image_path, api_url):
    """Test HRNet API"""
    print(f"\n{'='*80}")
    print(f"🔬 TEST 1: HRNet API ({api_url})")
    print(f"{'='*80}")
    
    data_url, base64_str = image_to_base64(image_path)
    
    try:
        response = requests.post(
            f"{api_url}/detect",
            json={'image_base64': data_url},
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
        
        data = response.json()
        
        if not data.get('success', True):
            print(f"❌ Detection failed: {data.get('error', 'Unknown error')}")
            return None
        
        landmarks = data.get('landmarks', {})
        metadata = data.get('metadata', {})
        
        print(f"✅ API Response received")
        print(f"   Model: {metadata.get('model', 'N/A')}")
        print(f"   Valid landmarks: {len([k for k, v in landmarks.items() if v])}/{len(landmarks)}")
        print(f"   Processing time: {metadata.get('processing_time', 0):.3f}s")
        print(f"   Image size: {metadata.get('image_size', {})}")
        print(f"   Model input size: {metadata.get('model_input_size', 'N/A')}")
        
        # Convert to standard format
        result_landmarks = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result_landmarks[name] = {
                    'x': float(coords.get('x', 0)),
                    'y': float(coords.get('y', 0))
                }
        
        return result_landmarks, metadata
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error: Could not connect to {api_url}")
        print(f"   Make sure the service is running!")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_errors(predicted, ground_truth):
    """Calculate errors between predicted and ground truth"""
    errors = []
    
    # HRNet landmarks mapping to Aariz landmarks
    landmark_mapping = {
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
        'ANS': 'ANS',
        'PNS': 'PNS',
        'Ar': 'Ar',
    }
    
    for hrnet_name, aariz_name in landmark_mapping.items():
        if hrnet_name in predicted and aariz_name in ground_truth:
            pred = predicted[hrnet_name]
            gt = ground_truth[aariz_name]
            
            error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
            error_mm = error_px * PIXEL_SIZE
            
            errors.append({
                'name': hrnet_name,
                'error_px': error_px,
                'error_mm': error_mm,
                'pred': pred,
                'gt': gt,
                'diff_x': pred['x'] - gt['x'],
                'diff_y': pred['y'] - gt['y']
            })
    
    return errors

def print_comparison_table(errors):
    """Print comparison table"""
    print(f"\n{'='*80}")
    print("📊 مقایسه نتایج با Ground Truth")
    print(f"{'='*80}")
    print(f"\n{'Landmark':<10} {'Pred X':<12} {'Pred Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<10} {'Diff Y':<10} {'Error (px)':<12} {'Error (mm)':<12}")
    print("-"*100)
    
    errors.sort(key=lambda x: x['error_mm'], reverse=True)
    
    for err in errors:
        pred = err['pred']
        gt = err['gt']
        print(f"{err['name']:<10} {pred['x']:<12.2f} {pred['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {err['diff_x']:<10.2f} {err['diff_y']:<10.2f} {err['error_px']:<12.2f} {err['error_mm']:<12.4f}")
    
    # Statistics
    error_values_mm = [e['error_mm'] for e in errors]
    
    print(f"\n{'='*80}")
    print("📈 آمار خطاها")
    print(f"{'='*80}")
    print(f"\n✅ تعداد لندمارک‌های مقایسه شده: {len(errors)}")
    print(f"\n📊 خطا بر حسب میلی‌متر:")
    print(f"   میانگین (MRE): {np.mean(error_values_mm):.4f} mm")
    print(f"   میانه: {np.median(error_values_mm):.4f} mm")
    print(f"   کمینه: {np.min(error_values_mm):.4f} mm")
    print(f"   بیشینه: {np.max(error_values_mm):.4f} mm")
    print(f"   انحراف معیار: {np.std(error_values_mm):.4f} mm")
    
    # SDR
    thresholds_mm = [1.0, 2.0, 2.5, 3.0, 4.0]
    print(f"\n{'='*80}")
    print("📊 Success Detection Rate (SDR)")
    print(f"{'='*80}")
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
        'errors': errors
    }

def main():
    print("="*80)
    print("🧪 تست کامل HRNet Model")
    print("="*80)
    print(f"\n📸 تصویر تست: {TEST_IMAGE_ID}")
    print(f"📂 مسیر تصویر: {TEST_IMAGE_PATH}")
    print(f"📂 مسیر Ground Truth: {GROUND_TRUTH_PATH}")
    
    # Check files exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\n❌ ERROR: Ground truth not found: {GROUND_TRUTH_PATH}")
        return
    
    # Load image to get size
    img = Image.open(TEST_IMAGE_PATH)
    img_size = img.size
    print(f"\n📏 اندازه تصویر: {img_size[0]} × {img_size[1]} پیکسل")
    print(f"📏 Pixel Size: {PIXEL_SIZE} mm/pixel")
    
    # Load ground truth
    print(f"\n📂 بارگذاری Ground Truth...")
    gt_landmarks = load_ground_truth()
    print(f"✅ {len(gt_landmarks)} لندمارک در Ground Truth یافت شد")
    
    # Test HRNet API
    hrnet_result = test_hrnet_api(TEST_IMAGE_PATH, HRNET_API_URL)
    
    if hrnet_result:
        hrnet_landmarks, hrnet_metadata = hrnet_result
        
        # Calculate errors
        errors = calculate_errors(hrnet_landmarks, gt_landmarks)
        
        if errors:
            # Print comparison
            stats = print_comparison_table(errors)
            
            # Save results
            output = {
                'image_id': TEST_IMAGE_ID,
                'image_size': {'width': img_size[0], 'height': img_size[1]},
                'pixel_size': PIXEL_SIZE,
                'model': 'HRNet-W32',
                'model_input_size': hrnet_metadata.get('model_input_size', 'N/A'),
                'stats': {
                    'mre_mm': stats['mre_mm'],
                    'median_mm': stats['median_mm'],
                    'min_mm': stats['min_mm'],
                    'max_mm': stats['max_mm'],
                    'std_mm': stats['std_mm']
                },
                'errors': {e['name']: {'mm': e['error_mm'], 'px': e['error_px']} for e in stats['errors']}
            }
            
            output_file = 'hrnet_test_results.json'
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 نتایج در {output_file} ذخیره شد")
        else:
            print("\n⚠️  هیچ لندمارک مشترکی برای مقایسه یافت نشد!")
    else:
        print("\n⚠️  تست HRNet API انجام نشد!")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

