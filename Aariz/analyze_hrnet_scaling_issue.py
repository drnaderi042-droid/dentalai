"""
اسکریپت تست و بررسی مشکل Scaling در HRNet
"""

import os
import sys
import json
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

def analyze_scaling():
    """تحلیل مشکل scaling"""
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    print("="*100)
    print("تحليل مشكل Scaling در HRNet")
    print("="*100)
    
    # Load image
    img = Image.open(TEST_IMAGE_PATH)
    orig_w, orig_h = img.size
    print(f"\nتصوير اصلي:")
    print(f"   اندازه: {orig_w} × {orig_h} پیکسل")
    print(f"   Aspect Ratio: {orig_w/orig_h:.4f}")
    
    # Model config
    from hrnet_config_cephalometric import get_hrnet_w32_cephalometric_config
    cfg = get_hrnet_w32_cephalometric_config()
    model_input_w, model_input_h = cfg.MODEL.IMAGE_SIZE
    heatmap_w, heatmap_h = cfg.MODEL.HEATMAP_SIZE
    
    print(f"\nتنظيمات مدل:")
    print(f"   Model Input Size: {model_input_w} × {model_input_h}")
    print(f"   Heatmap Size: {heatmap_w} × {heatmap_h}")
    print(f"   Model Aspect Ratio: {model_input_w/model_input_h:.4f}")
    
    # Calculate scales
    print(f"\nمحاسبه Scale Factors:")
    
    # Old method (مستقیم از heatmap به original)
    scale_x_old = orig_w / heatmap_w
    scale_y_old = orig_h / heatmap_h
    print(f"\nروش قديمي (مستقيم از heatmap به original):")
    print(f"   scale_x = {orig_w} / {heatmap_w} = {scale_x_old:.4f}")
    print(f"   scale_y = {orig_h} / {heatmap_h} = {scale_y_old:.4f}")
    print(f"   ⚠️  مشکل: این روش aspect ratio را در نظر نمی‌گیرد!")
    
    # New method (دو مرحله‌ای)
    scale_x_hm_to_model = model_input_w / heatmap_w
    scale_y_hm_to_model = model_input_h / heatmap_h
    scale_x_model_to_orig = orig_w / model_input_w
    scale_y_model_to_orig = orig_h / model_input_h
    
    print(f"\nروش جديد (دو مرحله اي):")
    print(f"   مرحله 1: Heatmap -> Model Input")
    print(f"     scale_x = {model_input_w} / {heatmap_w} = {scale_x_hm_to_model:.4f}")
    print(f"     scale_y = {model_input_h} / {heatmap_h} = {scale_y_hm_to_model:.4f}")
    print(f"   مرحله 2: Model Input -> Original")
    print(f"     scale_x = {orig_w} / {model_input_w} = {scale_x_model_to_orig:.4f}")
    print(f"     scale_y = {orig_h} / {model_input_h} = {scale_y_model_to_orig:.4f}")
    
    # Example calculation
    print(f"\nمثال محاسبه:")
    print(f"   فرض: يك لندمارك در heatmap در موقعيت (96, 96)")
    
    # Old method
    x_old = 96 * scale_x_old
    y_old = 96 * scale_y_old
    print(f"\n   روش قديمي:")
    print(f"     x = 96 x {scale_x_old:.4f} = {x_old:.2f}")
    print(f"     y = 96 x {scale_y_old:.4f} = {y_old:.2f}")
    
    # New method
    x_model = 96 * scale_x_hm_to_model
    y_model = 96 * scale_y_hm_to_model
    x_new = x_model * scale_x_model_to_orig
    y_new = y_model * scale_y_model_to_orig
    print(f"\n   روش جديد:")
    print(f"     مرحله 1: (96, 96) -> ({x_model:.2f}, {y_model:.2f})")
    print(f"     مرحله 2: ({x_model:.2f}, {y_model:.2f}) -> ({x_new:.2f}, {y_new:.2f})")
    
    print(f"\n   تفاوت:")
    print(f"     Δx = {abs(x_new - x_old):.2f} پیکسل")
    print(f"     Δy = {abs(y_new - y_old):.2f} پیکسل")
    
    # Load ground truth for comparison
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    
    gt_landmarks = {lm['symbol']: {'x': float(lm['value']['x']), 'y': float(lm['value']['y'])} 
                    for lm in gt_data['landmarks']}
    
    print(f"\nبررسي محدوده مختصات Ground Truth:")
    x_coords = [lm['x'] for lm in gt_landmarks.values()]
    y_coords = [lm['y'] for lm in gt_landmarks.values()]
    print(f"   X range: {min(x_coords):.0f} - {max(x_coords):.0f} (image width: {orig_w})")
    print(f"   Y range: {min(y_coords):.0f} - {max(y_coords):.0f} (image height: {orig_h})")
    
    print(f"\nنكته مهم:")
    print(f"   وقتی تصویر به {model_input_w}×{model_input_h} resize می‌شود،")
    print(f"   aspect ratio از {orig_w/orig_h:.4f} به {model_input_w/model_input_h:.4f} تغییر می‌کند.")
    print(f"   این باعث distortion می‌شود و ممکن است دقت را کاهش دهد.")
    
    print("\n" + "="*100)

if __name__ == '__main__':
    analyze_scaling()

