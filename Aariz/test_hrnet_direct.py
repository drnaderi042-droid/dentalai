"""
اسکریپت تست مستقیم HRNet (بدون API)
- استفاده مستقیم از HRNetProductionService
- مقایسه با Ground Truth
"""

import os
import sys
import json
import numpy as np
from PIL import Image

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')

# Add cephx_service to path
if cephx_path not in sys.path:
    sys.path.insert(0, cephx_path)

# Add venv site-packages to path (for easydict and other dependencies)
venv_site_packages = os.path.join(cephx_path, 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

# Import HRNet service
try:
    from hrnet_production_service import HRNetProductionService
except ImportError as e:
    print(f"❌ Error importing HRNetProductionService: {e}")
    print(f"   Make sure you're running from the correct directory")
    print(f"   Trying to use venv from: {venv_site_packages}")
    print(f"\n💡 Tip: Use run_hrnet_direct_test.bat or activate venv first:")
    print(f"   cd {cephx_path}")
    print(f"   .\\venv\\Scripts\\Activate.ps1")
    print(f"   cd ..\\Aariz")
    print(f"   python test_hrnet_direct.py")
    sys.exit(1)

# Test image
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
GROUND_TRUTH_PATH = os.path.join(
    base_dir, "Aariz", "Aariz", "train", "Annotations", "Cephalometric Landmarks",
    "Senior Orthodontists", f"{TEST_IMAGE_ID}.json"
)

# Model path
MODEL_PATH = os.path.join(cephx_path, "model", "hrnet_cephalometric.pth")

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
    thresholds_mm = [1.0, 2.0, 2.5, 3.0, 4.0]
    print(f"\n{'='*100}")
    print("📊 Success Detection Rate (SDR)")
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
        'errors': errors
    }

def main():
    print("="*100)
    print("🧪 تست مستقیم HRNet Model (Real Model - بدون API)")
    print("="*100)
    print(f"\n📸 تصویر تست: {TEST_IMAGE_ID}")
    print(f"📂 مسیر تصویر: {TEST_IMAGE_PATH}")
    print(f"📂 مسیر Ground Truth: {GROUND_TRUTH_PATH}")
    print(f"📂 مسیر مدل: {MODEL_PATH}")
    
    # Check files exist
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"\n❌ ERROR: Image not found: {TEST_IMAGE_PATH}")
        return
    
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"\n❌ ERROR: Ground truth not found: {GROUND_TRUTH_PATH}")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model checkpoint not found: {MODEL_PATH}")
        print(f"   Please make sure the checkpoint exists!")
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
    
    # Load HRNet service
    print(f"\n🤖 بارگذاری HRNet Model...")
    print(f"   Checkpoint: {MODEL_PATH}")
    
    try:
        # Check if CUDA is available
        import torch
        if torch.cuda.is_available():
            device = 'cuda'
            print(f"   🔥 Using GPU (CUDA)")
        else:
            device = 'cpu'
            print(f"   💻 Using CPU (CUDA not available or PyTorch not compiled with CUDA)")
        
        hrnet_service = HRNetProductionService(MODEL_PATH, device=device)
        print(f"   ✅ Model loaded successfully!")
        print(f"   Model Type: REAL (not mock)")
        print(f"   Input Size: {hrnet_service.input_size}")
        print(f"   Heatmap Size: {hrnet_service.heatmap_size}")
        print(f"   Landmarks: {len(hrnet_service.LANDMARKS)}")
        print(f"   Accuracy (from checkpoint): {hrnet_service.accuracy_mre:.4f}mm")
    except Exception as e:
        print(f"\n❌ ERROR: Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run detection
    print(f"\n🔍 اجرای تشخیص...")
    try:
        result = hrnet_service.detect(TEST_IMAGE_PATH)
        
        if not result.get('success', True):
            print(f"❌ Detection failed: {result.get('error', 'Unknown error')}")
            return
        
        predicted_landmarks = result['landmarks']
        metadata = result['metadata']
        
        print(f"   ✅ Detection complete!")
        print(f"   Valid landmarks: {len([k for k, v in predicted_landmarks.items() if v])}/{len(predicted_landmarks)}")
        print(f"   Processing time: {metadata.get('processing_time', 0):.3f}s")
        print(f"   Avg confidence: {metadata.get('avg_confidence', 0):.3f}")
        print(f"   Model Input Size: {metadata.get('model_input_size', 'N/A')}")
        
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
    stats = print_comparison_table(errors)
    
    # Save results
    output = {
        'image_id': TEST_IMAGE_ID,
        'image_size': {'width': img_size[0], 'height': img_size[1]},
        'pixel_size': PIXEL_SIZE,
        'model': 'HRNet-W32',
        'model_type': 'real',
        'model_input_size': list(hrnet_service.input_size) if hasattr(hrnet_service, 'input_size') else 'N/A',
        'stats': {
            'mre_mm': stats['mre_mm'],
            'median_mm': stats['median_mm'],
            'min_mm': stats['min_mm'],
            'max_mm': stats['max_mm'],
            'std_mm': stats['std_mm']
        },
        'errors': {e['name']: {'mm': e['error_mm'], 'px': e['error_px']} for e in stats['errors']}
    }
    
    output_file = 'hrnet_direct_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    
    # Final summary
    print(f"\n{'='*100}")
    print("📋 خلاصه")
    print(f"{'='*100}")
    print(f"   Model: HRNet-W32 (REAL)")
    print(f"   Input Size: {hrnet_service.input_size}")
    print(f"   MRE: {stats['mre_mm']:.4f} mm")
    print(f"   SDR @ 2mm: {sum(1 for e_mm in [e['error_mm'] for e in stats['errors']] if e_mm <= 2.0) / len(stats['errors']) * 100:.2f}%")
    
    if stats['mre_mm'] < 2.0:
        print(f"\n✅ نتایج عالی! MRE کمتر از 2mm است")
    elif stats['mre_mm'] < 5.0:
        print(f"\n✅ نتایج خوب! MRE کمتر از 5mm است")
    else:
        print(f"\n⚠️  MRE بالاست. ممکن است نیاز به بررسی بیشتر باشد")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    main()

