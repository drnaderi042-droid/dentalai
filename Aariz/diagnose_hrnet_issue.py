"""
اسکریپت تشخیصی برای بررسی مشکل MRE بالا در HRNet
بررسی می‌کند که آیا padding درست اعمال می‌شود و مختصات به درستی scale می‌شوند
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')
aariz_path = os.path.join(base_dir, 'Aariz')

if cephx_path not in sys.path:
    sys.path.insert(0, cephx_path)

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

def analyze_detection():
    """تحلیل دقیق مشکل detection"""
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("="*100)
    print("تحليل دقيق مشكل Detection در HRNet")
    print("="*100)
    
    # Load image
    img = Image.open(TEST_IMAGE_PATH)
    orig_w, orig_h = img.size
    print(f"\nتصوير اصلي:")
    print(f"   اندازه: {orig_w} × {orig_h} پیکسل")
    print(f"   Aspect Ratio: {orig_w/orig_h:.4f}")
    
    # Load ground truth
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_landmarks = {lm['symbol']: {'x': float(lm['value']['x']), 'y': float(lm['value']['y'])} 
                    for lm in gt_data['landmarks']}
    
    print(f"\nGround Truth:")
    print(f"   تعداد لندمارک‌ها: {len(gt_landmarks)}")
    x_coords = [lm['x'] for lm in gt_landmarks.values()]
    y_coords = [lm['y'] for lm in gt_landmarks.values()]
    print(f"   X range: {min(x_coords):.0f} - {max(x_coords):.0f}")
    print(f"   Y range: {min(y_coords):.0f} - {max(y_coords):.0f}")
    
    # Load model and detect
    from hrnet_production_service import HRNetProductionService
    
    print(f"\nبارگذاري مدل...")
    service = HRNetProductionService(MODEL_PATH)
    
    print(f"\nانجام Detection...")
    result = service.detect(TEST_IMAGE_PATH, preserve_aspect_ratio=True)
    
    pred_landmarks = result['landmarks']
    metadata = result['metadata']
    padding_info = metadata.get('padding_info')
    
    print(f"\nنتايج Detection:")
    print(f"   تعداد لندمارک‌ها: {len(pred_landmarks)}")
    print(f"   Preserve Aspect Ratio: {metadata.get('preserve_aspect_ratio', False)}")
    
    if padding_info:
        print(f"\nاطلاعات Padding:")
        print(f"   Scale: {padding_info.get('scale', 'N/A')}")
        print(f"   Padding X: {padding_info.get('padding_x', 'N/A')}")
        print(f"   Padding Y: {padding_info.get('padding_y', 'N/A')}")
        print(f"   Resized Size: {padding_info.get('resized_size', 'N/A')}")
    
    # Compare with ground truth
    print(f"\n{'='*100}")
    print("مقايسه مختصات")
    print(f"{'='*100}")
    
    # Mapping
    LANDMARK_MAPPING = {
        'S': 'S', 'N': 'N', 'Or': 'Or', 'Po': 'Po', 'A': 'A', 'B': 'B',
        'Pog': 'Pog', 'Me': 'Me', 'Gn': 'Gn', 'Go': 'Go',
        'L1': 'LIT', 'U1': 'UIT', 'UL': 'Ls', 'LL': 'Li',
        'Sn': 'Sn', 'PogSoft': 'Pog`', 'PNS': 'PNS', 'ANS': 'ANS', 'Ar': 'Ar',
    }
    
    print(f"\n{'Landmark':<12} {'Pred X':<12} {'Pred Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<12} {'Diff Y':<12} {'Error (px)':<12}")
    print("-"*100)
    
    errors = []
    for hrnet_name, gt_name in LANDMARK_MAPPING.items():
        if hrnet_name in pred_landmarks and gt_name in gt_landmarks:
            pred = pred_landmarks[hrnet_name]
            gt = gt_landmarks[gt_name]
            
            diff_x = pred['x'] - gt['x']
            diff_y = pred['y'] - gt['y']
            error_px = np.sqrt(diff_x**2 + diff_y**2)
            
            errors.append({
                'name': hrnet_name,
                'pred': pred,
                'gt': gt,
                'error_px': error_px,
                'diff_x': diff_x,
                'diff_y': diff_y
            })
            
            print(f"{hrnet_name:<12} {pred['x']:<12.2f} {pred['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {diff_x:<12.2f} {diff_y:<12.2f} {error_px:<12.2f}")
    
    # Analyze errors
    print(f"\n{'='*100}")
    print("تحليل خطاها")
    print(f"{'='*100}")
    
    error_values = [e['error_px'] for e in errors]
    print(f"\nآمار خطا (پیکسل):")
    print(f"   میانگین: {np.mean(error_values):.2f} px")
    print(f"   میانه: {np.median(error_values):.2f} px")
    print(f"   کمینه: {np.min(error_values):.2f} px")
    print(f"   بیشینه: {np.max(error_values):.2f} px")
    
    # Check if there's a systematic offset
    print(f"\nبررسی Offset سيستماتيك:")
    diff_x_values = [e['diff_x'] for e in errors]
    diff_y_values = [e['diff_y'] for e in errors]
    print(f"   میانگین Diff X: {np.mean(diff_x_values):.2f} px")
    print(f"   میانگین Diff Y: {np.mean(diff_y_values):.2f} px")
    
    # Check if predictions are in wrong scale
    print(f"\nبررسی Scale:")
    pred_x_range = [pred_landmarks[k]['x'] for k in pred_landmarks.keys() if k in LANDMARK_MAPPING]
    pred_y_range = [pred_landmarks[k]['y'] for k in pred_landmarks.keys() if k in LANDMARK_MAPPING]
    print(f"   Pred X range: {min(pred_x_range):.0f} - {max(pred_x_range):.0f} (image width: {orig_w})")
    print(f"   Pred Y range: {min(pred_y_range):.0f} - {max(pred_y_range):.0f} (image height: {orig_h})")
    print(f"   GT X range: {min(x_coords):.0f} - {max(x_coords):.0f}")
    print(f"   GT Y range: {min(y_coords):.0f} - {max(y_coords):.0f}")
    
    # Check checkpoint original size
    import torch
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    checkpoint_config = checkpoint.get('config', {})
    input_config = checkpoint_config.get('INPUT', {})
    checkpoint_orig_size = input_config.get('ORIGINAL_SIZE', None)
    
    if checkpoint_orig_size:
        print(f"\nبررسی ORIGINAL_SIZE از Checkpoint:")
        print(f"   Checkpoint ORIGINAL_SIZE: {checkpoint_orig_size}")
        print(f"   تصویر تست: [{orig_w}, {orig_h}]")
        print(f"   Aspect Ratio Checkpoint: {checkpoint_orig_size[0]/checkpoint_orig_size[1]:.4f}")
        print(f"   Aspect Ratio Test Image: {orig_w/orig_h:.4f}")
        
        if abs(checkpoint_orig_size[0]/checkpoint_orig_size[1] - orig_w/orig_h) > 0.1:
            print(f"   ⚠️  تفاوت قابل توجه در Aspect Ratio!")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    analyze_detection()

