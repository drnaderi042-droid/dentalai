"""
اسکریپت تست کامل HRNet - مقایسه API vs Direct vs Ground Truth
"""

import os
import sys
import json
import base64
import numpy as np
import requests
from PIL import Image

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')
aariz_path = os.path.join(base_dir, 'Aariz')

if cephx_path not in sys.path:
    sys.path.insert(0, cephx_path)

# Add venv site-packages to path (for easydict and other dependencies)
venv_site_packages = os.path.join(cephx_path, 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Test image
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

MODEL_PATH = os.path.join(cephx_path, "model", "hrnet_cephalometric.pth")
HRNET_API_URL = "http://localhost:5000"
PIXEL_SIZE = 0.1

def load_ground_truth():
    """Load ground truth"""
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    return {lm['symbol']: {'x': float(lm['value']['x']), 'y': float(lm['value']['y'])} 
            for lm in annotation['landmarks']}

def image_to_base64(image_path):
    """Convert to base64"""
    with open(image_path, 'rb') as f:
        base64_str = base64.b64encode(f.read()).decode('utf-8')
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/png' if ext == '.png' else 'image/jpeg'
    return f"data:{mime};base64,{base64_str}"

def test_api():
    """Test via API"""
    print(f"\n{'='*80}")
    print("🔬 TEST 1: HRNet API")
    print(f"{'='*80}")
    
    data_url = image_to_base64(TEST_IMAGE_PATH)
    
    try:
        response = requests.post(f"{HRNET_API_URL}/detect", 
                               json={'image_base64': data_url}, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ API Error: {response.status_code}")
            return None
        
        data = response.json()
        if not data.get('success', True):
            print(f"❌ Failed: {data.get('error')}")
            return None
        
        landmarks = data.get('landmarks', {})
        metadata = data.get('metadata', {})
        
        print(f"✅ API Response")
        print(f"   Model Type: {metadata.get('model_type', 'N/A')}")
        print(f"   Input Size: {metadata.get('model_input_size', 'N/A')}")
        print(f"   Valid landmarks: {len([k for k, v in landmarks.items() if v])}")
        
        result = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result[name] = {'x': float(coords.get('x', 0)), 'y': float(coords.get('y', 0))}
        
        return result, metadata, 'API'
        
    except requests.exceptions.ConnectionError:
        print(f"❌ Connection error! Service may not be running")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def test_direct():
    """Test directly (without API)"""
    print(f"\n{'='*80}")
    print("🔬 TEST 2: HRNet Direct (Real Model)")
    print(f"{'='*80}")
    
    try:
        from hrnet_production_service import HRNetProductionService
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return None
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        return None
    
    try:
        print(f"📂 Loading model: {MODEL_PATH}")
        
        # Check if CUDA is available
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"   🔥 Using GPU (CUDA)")
        else:
            device = 'cpu'
            print(f"   💻 Using CPU (CUDA not available)")
        
        service = HRNetProductionService(MODEL_PATH, device=device)
        print(f"✅ Model loaded!")
        print(f"   Input Size: {service.input_size}")
        print(f"   Accuracy: {service.accuracy_mre:.4f}mm")
        
        result = service.detect(TEST_IMAGE_PATH)
        landmarks = result.get('landmarks', {})
        metadata = result.get('metadata', {})
        
        print(f"✅ Detection complete!")
        print(f"   Valid landmarks: {len([k for k, v in landmarks.items() if v])}")
        
        result_dict = {}
        for name, coords in landmarks.items():
            if coords and isinstance(coords, dict):
                result_dict[name] = {'x': float(coords.get('x', 0)), 'y': float(coords.get('y', 0))}
        
        return result_dict, metadata, 'Direct'
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
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

def print_results(errors, test_name):
    """Print results"""
    if not errors:
        print(f"\n⚠️  No landmarks to compare for {test_name}")
        return None
    
    error_values_mm = [e['error_mm'] for e in errors]
    
    print(f"\n{'='*80}")
    print(f"📊 نتایج {test_name}")
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
    
    print(f"\n📊 SDR:")
    for thresh in [1.0, 2.0, 2.5, 3.0, 4.0]:
        sdr = sum(1 for e in error_values_mm if e <= thresh) / len(error_values_mm) * 100
        print(f"   @ {thresh}mm: {sdr:.2f}%")
    
    return {
        'mre_mm': float(np.mean(error_values_mm)),
        'errors': errors
    }

def main():
    print("="*80)
    print("🧪 تست کامل HRNet - مقایسه API vs Direct")
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
    
    gt = load_ground_truth()
    print(f"✅ Ground truth: {len(gt)} landmarks")
    
    # Test API
    api_result = test_api()
    
    # Test Direct
    direct_result = test_direct()
    
    # Compare with Ground Truth
    print(f"\n{'='*80}")
    print("📊 مقایسه با Ground Truth")
    print(f"{'='*80}")
    
    if api_result:
        pred_api, metadata_api, _ = api_result
        errors_api = calculate_errors(pred_api, gt)
        stats_api = print_results(errors_api, "API")
    else:
        stats_api = None
    
    if direct_result:
        pred_direct, metadata_direct, _ = direct_result
        errors_direct = calculate_errors(pred_direct, gt)
        stats_direct = print_results(errors_direct, "Direct (Real Model)")
    else:
        stats_direct = None
    
    # Summary
    print(f"\n{'='*80}")
    print("📋 خلاصه")
    print(f"{'='*80}")
    
    if stats_api:
        print(f"\nAPI:")
        print(f"   MRE: {stats_api['mre_mm']:.4f} mm")
        print(f"   Model Type: {metadata_api.get('model_type', 'N/A')}")
    
    if stats_direct:
        print(f"\nDirect (Real Model):")
        print(f"   MRE: {stats_direct['mre_mm']:.4f} mm")
        print(f"   Input Size: {metadata_direct.get('model_input_size', 'N/A')}")
    
    if stats_api and stats_direct:
        print(f"\n💡 تفاوت:")
        diff = abs(stats_api['mre_mm'] - stats_direct['mre_mm'])
        print(f"   تفاوت MRE: {diff:.4f} mm")
        if diff < 0.1:
            print(f"   ✅ نتایج بسیار مشابه هستند!")
        else:
            print(f"   ⚠️  تفاوت وجود دارد!")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

