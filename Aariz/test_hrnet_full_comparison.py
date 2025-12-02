"""
اسکریپت تست کامل برای HRNet
- تست از API
- مقایسه با Ground Truth
- بررسی سایز تصویر
"""

import os
import sys
import json
import base64
import numpy as np
from PIL import Image
import requests

# Paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')

# Add venv site-packages to path (for easydict and other dependencies)
venv_site_packages = os.path.join(cephx_path, 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

HRNET_API_URL = "http://localhost:5000"
PIXEL_SIZE = 0.1

def load_ground_truth():
    """Load ground truth"""
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    gt = {}
    for lm in annotation['landmarks']:
        gt[lm['symbol']] = {'x': float(lm['value']['x']), 'y': float(lm['value']['y'])}
    return gt

def image_to_base64(image_path):
    """Convert to base64"""
    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{base64_str}"

def test_hrnet_api():
    """Test HRNet API"""
    print(f"\n{'='*80}")
    print("🔬 TEST: HRNet API")
    print(f"{'='*80}")
    
    data_url = image_to_base64(TEST_IMAGE_PATH)
    
    try:
        response = requests.post(f"{HRNET_API_URL}/detect", 
                               json={'image_base64': data_url}, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            return None
        
        data = response.json()
        if not data.get('success', True):
            print(f"❌ Failed: {data.get('error')}")
            return None
        
        landmarks = data.get('landmarks', {})
        metadata = data.get('metadata', {})
        
        print(f"✅ API Response")
        print(f"   Model: {metadata.get('model', 'N/A')}")
        print(f"   Model Type: {metadata.get('model_type', 'N/A')}")
        print(f"   Input Size: {metadata.get('model_input_size', 'N/A')}")
        print(f"   Valid landmarks: {len([k for k, v in landmarks.items() if v])}")
        
        # Check if it's mock or real
        model_type = metadata.get('model_type', 'unknown')
        if model_type == 'mock':
            print(f"\n⚠️  WARNING: Using MOCK model! Results will be inaccurate!")
            print(f"   Please use app_hrnet_real.py instead of app_hrnet.py")
        elif model_type == 'real':
            print(f"\n✅ Using REAL HRNet model")
        else:
            print(f"\n⚠️  Model type unknown: {model_type}")
        
        result = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result[name] = {'x': float(coords.get('x', 0)), 'y': float(coords.get('y', 0))}
        
        return result, metadata
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error! Make sure service is running on {HRNET_API_URL}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def calculate_errors(predicted, ground_truth):
    """Calculate errors"""
    mapping = {
        'S': 'S', 'N': 'N', 'Or': 'Or', 'Po': 'Po',
        'A': 'A', 'B': 'B', 'Pog': 'Pog', 'Me': 'Me',
        'Gn': 'Gn', 'Go': 'Go', 'L1': 'LIT', 'U1': 'UIT',
        'ANS': 'ANS', 'PNS': 'PNS', 'Ar': 'Ar'
    }
    
    errors = []
    for hrnet_name, aariz_name in mapping.items():
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
                'gt': gt
            })
    
    return errors

def main():
    print("="*80)
    print("🧪 تست کامل HRNet")
    print("="*80)
    print(f"\n📸 Image: {TEST_IMAGE_ID}")
    
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"❌ Image not found!")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"❌ Ground truth not found!")
        return
    
    img = Image.open(TEST_IMAGE_PATH)
    print(f"📏 Image size: {img.size[0]} × {img.size[1]} px")
    
    # Load GT
    gt = load_ground_truth()
    print(f"✅ Ground truth loaded: {len(gt)} landmarks")
    
    # Test API
    result = test_hrnet_api()
    
    if result:
        pred, metadata = result
        
        print(f"\n{'='*80}")
        print("📊 Model Info:")
        print(f"{'='*80}")
        print(f"   Model Type: {metadata.get('model_type', 'N/A')}")
        print(f"   Trained Image Size: 768×768 (from checkpoint)")
        
        model_input_size = metadata.get('model_input_size', None)
        if model_input_size:
            print(f"   Service Input Size: {model_input_size}")
            if isinstance(model_input_size, (list, tuple)) and len(model_input_size) >= 2:
                if list(model_input_size) == [768, 768] or model_input_size == (768, 768):
                    print(f"\n✅ Image size تطابق دارد!")
                else:
                    print(f"\n⚠️  عدم تطابق! Service: {model_input_size}, Expected: [768, 768]")
            else:
                print(f"\n⚠️  Format issue with model_input_size: {model_input_size}")
        else:
            print(f"   Service Input Size: N/A (not returned in metadata)")
            print(f"\n⚠️  model_input_size not found in metadata!")
        
        # Compare
        errors = calculate_errors(pred, gt)
        
        if errors:
            error_values_mm = [e['error_mm'] for e in errors]
            
            print(f"\n{'='*80}")
            print("📊 نتایج مقایسه")
            print(f"{'='*80}")
            print(f"{'Landmark':<10} {'Error (mm)':<15}")
            print("-"*30)
            
            errors.sort(key=lambda x: x['error_mm'], reverse=True)
            for e in errors:
                print(f"{e['name']:<10} {e['error_mm']:<15.4f}")
            
            print(f"\n📈 آمار:")
            print(f"   MRE: {np.mean(error_values_mm):.4f} mm")
            print(f"   Median: {np.median(error_values_mm):.4f} mm")
            print(f"   Min: {np.min(error_values_mm):.4f} mm")
            print(f"   Max: {np.max(error_values_mm):.4f} mm")
            
            # SDR
            print(f"\n📊 SDR:")
            for thresh in [1.0, 2.0, 2.5, 3.0, 4.0]:
                sdr = sum(1 for e in error_values_mm if e <= thresh) / len(error_values_mm) * 100
                print(f"   @ {thresh}mm: {sdr:.2f}%")
            
            # Save
            output = {
                'image_id': TEST_IMAGE_ID,
                'model': 'HRNet-W32',
                'mre_mm': float(np.mean(error_values_mm)),
                'errors': {e['name']: e['error_mm'] for e in errors}
            }
            
            with open('hrnet_test_results.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Results saved to hrnet_test_results.json")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

