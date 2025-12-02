"""
اسکریپت برای مقایسه خروجی مدل اصلی و مدل ترکیبی
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel
from utils import heatmap_to_coordinates


def preprocess_image(image_path, target_size=(768, 768)):
    """
    پیش‌پردازش تصویر مشابه inference.py
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    original_size = image.size  # (width, height)
    
    # Resize
    image_resized = image.resize((target_size[1], target_size[0]), Image.BILINEAR)
    
    # Normalize (مشابه inference.py)
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1] and CHW format
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Match training
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)  # Add batch dimension
    
    return image_tensor, original_size


def test_models(main_model_path, combined_model_path, image_path):
    """
    مقایسه خروجی مدل اصلی و مدل ترکیبی
    """
    print("="*80)
    print("Testing Model Outputs")
    print("="*80)
    
    # بارگذاری checkpoint ها
    print("\n[1/5] Loading checkpoints...")
    main_checkpoint = torch.load(main_model_path, map_location='cpu', weights_only=False)
    combined_checkpoint = torch.load(combined_model_path, map_location='cpu', weights_only=False)
    
    main_state_dict = main_checkpoint.get('model_state_dict', main_checkpoint)
    combined_state_dict = combined_checkpoint.get('model_state_dict', combined_checkpoint)
    
    # ایجاد مدل‌ها
    print("\n[2/5] Creating models...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    main_model = HRNetLandmarkModel(num_landmarks=29, width=32)
    combined_model = HRNetLandmarkModel(num_landmarks=31, width=32)
    
    main_model.load_state_dict(main_state_dict, strict=False)
    combined_model.load_state_dict(combined_state_dict, strict=False)
    
    main_model = main_model.to(device)
    combined_model = combined_model.to(device)
    main_model.eval()
    combined_model.eval()
    
    # پیش‌پردازش تصویر
    print("\n[3/5] Preprocessing image...")
    image_tensor, original_size = preprocess_image(image_path, target_size=(768, 768))
    image_tensor = image_tensor.to(device)
    print(f"  Image tensor shape: {image_tensor.shape}")
    print(f"  Original size: {original_size}")
    
    # پیش‌بینی با مدل اصلی
    print("\n[4/5] Running inference with main model (29 landmarks)...")
    with torch.no_grad():
        main_heatmaps = main_model(image_tensor)
        print(f"  Main model heatmaps shape: {main_heatmaps.shape}")
        
        # اگر heatmap ها کوچک‌تر از تصویر بودند، resize کن
        if main_heatmaps.shape[2:] != (768, 768):
            main_heatmaps = F.interpolate(
                main_heatmaps,
                size=(768, 768),
                mode='bilinear',
                align_corners=False
            )
            print(f"  Resized main heatmaps to: {main_heatmaps.shape}")
        
        # تبدیل به numpy
        main_heatmaps_np = torch.sigmoid(main_heatmaps).cpu().numpy()[0]
        
        # استخراج مختصات
        h, w = main_heatmaps_np.shape[1], main_heatmaps_np.shape[2]
        main_coordinates = heatmap_to_coordinates(main_heatmaps_np, h, w)
        
        # Scale به اندازه اصلی
        scale_x = original_size[0] / w
        scale_y = original_size[1] / h
        main_coordinates_scaled = main_coordinates.copy()
        main_coordinates_scaled[:, 0] *= scale_x
        main_coordinates_scaled[:, 1] *= scale_y
    
    # پیش‌بینی با مدل ترکیبی
    print("\n[5/5] Running inference with combined model (31 landmarks)...")
    with torch.no_grad():
        combined_heatmaps = combined_model(image_tensor)
        print(f"  Combined model heatmaps shape: {combined_heatmaps.shape}")
        
        # اگر heatmap ها کوچک‌تر از تصویر بودند، resize کن
        if combined_heatmaps.shape[2:] != (768, 768):
            combined_heatmaps = F.interpolate(
                combined_heatmaps,
                size=(768, 768),
                mode='bilinear',
                align_corners=False
            )
            print(f"  Resized combined heatmaps to: {combined_heatmaps.shape}")
        
        # تبدیل به numpy
        combined_heatmaps_np = torch.sigmoid(combined_heatmaps).cpu().numpy()[0]
        
        # استخراج مختصات
        h, w = combined_heatmaps_np.shape[1], combined_heatmaps_np.shape[2]
        combined_coordinates = heatmap_to_coordinates(combined_heatmaps_np, h, w)
        
        # Scale به اندازه اصلی
        scale_x = original_size[0] / w
        scale_y = original_size[1] / h
        combined_coordinates_scaled = combined_coordinates.copy()
        combined_coordinates_scaled[:, 0] *= scale_x
        combined_coordinates_scaled[:, 1] *= scale_y
    
    # مقایسه 29 لندمارک اول
    print("\n" + "="*80)
    print("COMPARISON: First 29 Landmarks")
    print("="*80)
    
    differences = []
    for i in range(29):
        main_coord = main_coordinates_scaled[i]
        combined_coord = combined_coordinates_scaled[i]
        
        diff = np.sqrt(np.sum((main_coord - combined_coord) ** 2))
        differences.append(diff)
        
        if diff > 10:  # اگر تفاوت بیشتر از 10 پیکسل باشد
            print(f"  Landmark {i}: Main={main_coord}, Combined={combined_coord}, Diff={diff:.2f} px")
    
    avg_diff = np.mean(differences)
    max_diff = np.max(differences)
    print(f"\n  Average difference: {avg_diff:.2f} px")
    print(f"  Max difference: {max_diff:.2f} px")
    
    if avg_diff < 1.0:
        print(f"  [OK] First 29 landmarks match closely!")
    elif avg_diff < 5.0:
        print(f"  [WARNING] First 29 landmarks have some differences")
    else:
        print(f"  [ERROR] First 29 landmarks have significant differences!")
    
    # بررسی heatmap ها
    print("\n" + "="*80)
    print("HEATMAP ANALYSIS")
    print("="*80)
    
    # مقایسه heatmap های 29 لندمارک اول
    heatmap_diffs = []
    for i in range(29):
        main_hm = main_heatmaps_np[i]
        combined_hm = combined_heatmaps_np[i]
        
        # محاسبه تفاوت
        diff = np.abs(main_hm - combined_hm).mean()
        heatmap_diffs.append(diff)
        
        if diff > 0.1:  # اگر تفاوت بیشتر از 0.1 باشد
            print(f"  Landmark {i}: Heatmap diff={diff:.4f}")
            print(f"    Main: min={main_hm.min():.4f}, max={main_hm.max():.4f}, mean={main_hm.mean():.4f}")
            print(f"    Combined: min={combined_hm.min():.4f}, max={combined_hm.max():.4f}, mean={combined_hm.mean():.4f}")
    
    avg_hm_diff = np.mean(heatmap_diffs)
    print(f"\n  Average heatmap difference: {avg_hm_diff:.4f}")
    
    if avg_hm_diff < 0.01:
        print(f"  [OK] Heatmaps match closely!")
    elif avg_hm_diff < 0.1:
        print(f"  [WARNING] Heatmaps have some differences")
    else:
        print(f"  [ERROR] Heatmaps have significant differences!")
    
    # بررسی P1/P2 (2 لندمارک آخر)
    print("\n" + "="*80)
    print("P1/P2 Analysis (Last 2 Landmarks)")
    print("="*80)
    
    for i in range(29, 31):
        combined_coord = combined_coordinates_scaled[i]
        combined_hm = combined_heatmaps_np[i]
        
        print(f"  Landmark {i} (P1/P2):")
        print(f"    Coordinates: ({combined_coord[0]:.2f}, {combined_coord[1]:.2f})")
        print(f"    Heatmap: min={combined_hm.min():.4f}, max={combined_hm.max():.4f}, mean={combined_hm.mean():.4f}")
        
        # بررسی اینکه آیا heatmap فعال است یا نه
        if combined_hm.max() < 0.1:
            print(f"    [WARNING] Heatmap is very weak (max < 0.1)")
        elif combined_hm.max() < 0.3:
            print(f"    [WARNING] Heatmap is weak (max < 0.3)")
        else:
            print(f"    [OK] Heatmap looks reasonable")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"First 29 landmarks:")
    print(f"  - Average coordinate difference: {avg_diff:.2f} px")
    print(f"  - Max coordinate difference: {max_diff:.2f} px")
    print(f"  - Average heatmap difference: {avg_hm_diff:.4f}")
    
    if avg_diff < 1.0 and avg_hm_diff < 0.01:
        print(f"\n[OK] Models produce similar outputs for first 29 landmarks!")
        return True
    else:
        print(f"\n[WARNING] Models produce different outputs!")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model outputs')
    parser.add_argument('--main_model', type=str, default='checkpoint_best_768.pth',
                        help='Path to main model (29 landmarks)')
    parser.add_argument('--combined_model', type=str, default='checkpoint_best_768_combined_31.pth',
                        help='Path to combined model (31 landmarks)')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.main_model):
        print(f"[ERROR] Main model not found: {args.main_model}")
        sys.exit(1)
    
    if not os.path.exists(args.combined_model):
        print(f"[ERROR] Combined model not found: {args.combined_model}")
        sys.exit(1)
    
    if not os.path.exists(args.image):
        print(f"[ERROR] Image not found: {args.image}")
        sys.exit(1)
    
    try:
        result = test_models(args.main_model, args.combined_model, args.image)
        if result:
            print("\n[OK] Test completed successfully!")
        else:
            print("\n[WARNING] Test completed with warnings!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




