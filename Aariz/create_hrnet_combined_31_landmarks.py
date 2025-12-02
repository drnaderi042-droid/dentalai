"""
ایجاد مدل ترکیبی 31 لندمارک با معماری HRNetLandmarkModel
ترکیب checkpoint_best_768.pth (29 لندمارک) و hrnet_p1p2_heatmap_best.pth (2 لندمارک)
با استفاده از معماری مشابه checkpoint_best_768.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel


def create_combined_hrnet_model(main_model_path, p1p2_model_path, output_path, width=32):
    """
    ایجاد مدل ترکیبی HRNetLandmarkModel با 31 لندمارک
    
    Args:
        main_model_path: مسیر checkpoint_best_768.pth (29 لندمارک)
        p1p2_model_path: مسیر hrnet_p1p2_heatmap_best.pth (2 لندمارک)
        output_path: مسیر ذخیره مدل ترکیبی
        width: عرض HRNet (default: 32)
    """
    print("="*80)
    print("Creating Combined HRNet Model: 29 + 2 = 31 Landmarks")
    print("="*80)
    
    # بررسی وجود فایل‌ها
    if not Path(main_model_path).exists():
        raise FileNotFoundError(f"Main model not found: {main_model_path}")
    
    if not Path(p1p2_model_path).exists():
        raise FileNotFoundError(f"P1/P2 model not found: {p1p2_model_path}")
    
    print(f"\n[1/5] Loading main model (29 landmarks)...")
    main_checkpoint = torch.load(main_model_path, map_location='cpu', weights_only=False)
    main_state_dict = main_checkpoint.get('model_state_dict', main_checkpoint)
    
    print(f"[2/5] Loading P1/P2 model (2 landmarks)...")
    p1p2_checkpoint = torch.load(p1p2_model_path, map_location='cpu', weights_only=False)
    p1p2_state_dict = p1p2_checkpoint.get('model_state_dict', p1p2_checkpoint)
    
    # بررسی ساختار مدل اصلی
    main_keys = list(main_state_dict.keys())
    print(f"\n[3/5] Analyzing model structures...")
    print(f"  Main model keys: {len(main_keys)}")
    print(f"  First 5 main keys: {main_keys[:5]}")
    
    # تشخیص width از مدل اصلی (اگر در checkpoint ذخیره شده)
    detected_width = width
    if 'width' in main_checkpoint:
        detected_width = main_checkpoint['width']
        print(f"  Detected width from checkpoint: {detected_width}")
    else:
        # سعی در تشخیص از ساختار
        # معمولاً width در stage1 استفاده می‌شود
        for key in main_keys:
            if 'stage1.0.branches.0.0.weight' in key:
                # این کلید معمولاً shape دارد که می‌تواند width را نشان دهد
                shape = main_state_dict[key].shape
                if len(shape) >= 2:
                    detected_width = shape[0]
                    print(f"  Detected width from model structure: {detected_width}")
                    break
    
    print(f"\n[4/5] Creating combined HRNetLandmarkModel with {detected_width} width...")
    # ایجاد مدل ترکیبی با 31 لندمارک
    combined_model = HRNetLandmarkModel(num_landmarks=31, width=detected_width)
    combined_state_dict = combined_model.state_dict()
    
    print(f"  Combined model keys: {len(combined_state_dict.keys())}")
    
    # کپی وزن‌های backbone از مدل اصلی
    print(f"\n[5/5] Copying weights...")
    copied_keys = []
    skipped_keys = []
    
    # کپی وزن‌های backbone (stem, stages, transitions)
    backbone_prefixes = ['stem.', 'stage1.', 'stage2.', 'stage3.', 'stage4.', 
                         'transition1.', 'transition2.', 'transition3.']
    
    for key in combined_state_dict.keys():
        # اگر کلید مربوط به backbone است
        if any(key.startswith(prefix) for prefix in backbone_prefixes):
            if key in main_state_dict:
                if combined_state_dict[key].shape == main_state_dict[key].shape:
                    combined_state_dict[key] = main_state_dict[key]
                    copied_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch)")
            else:
                skipped_keys.append(f"{key} (not in main model)")
        
        # کپی وزن‌های final_layers برای 29 لندمارک اول
        elif key.startswith('final_layers.'):
            # ساختار: final_layers.{layer_idx}.{module_idx}.{param_name}
            # مثال: final_layers.0.0.weight -> final_layers[0][0].weight (اولین Conv2d)
            #       final_layers.0.3.weight -> final_layers[0][3].weight (آخرین Conv2d - خروجی)
            #       final_layers.0.3.bias -> final_layers[0][3].bias (bias آخرین Conv2d)
            
            parts = key.split('.')
            if len(parts) >= 3 and parts[0] == 'final_layers':
                try:
                    layer_idx = int(parts[1])
                    module_idx = int(parts[2])
                    param_name = '.'.join(parts[3:]) if len(parts) > 3 else None
                    
                    # ساخت کلید معادل در مدل اصلی
                    main_key = '.'.join(['final_layers', str(layer_idx), str(module_idx)] + (parts[3:] if len(parts) > 3 else []))
                    
                    if main_key in main_state_dict:
                        main_weight = main_state_dict[main_key]
                        combined_weight = combined_state_dict[key]
                        
                        # تشخیص اینکه آیا این آخرین Conv2d است که خروجی heatmap را تولید می‌کند
                        # برای final_layers[0]: آخرین Conv2d در index 3 است
                        # برای final_layers[1,2,3]: آخرین Conv2d در index 4 است
                        # راه تشخیص: اگر out_channels == num_landmarks باشد، این آخرین Conv2d است
                        is_output_conv = False
                        if len(main_weight.shape) >= 1:
                            # اگر out_channels == 29 (برای مدل اصلی) یا == 31 (برای مدل ترکیبی)
                            if main_weight.shape[0] == 29 or combined_weight.shape[0] == 31:
                                # بررسی اینکه آیا این آخرین Conv2d در final_layer است
                                if layer_idx == 0:
                                    is_output_conv = (module_idx == 3)
                                else:
                                    is_output_conv = (module_idx == 4)
                        
                        # اگر این آخرین Conv2d است که خروجی heatmap را تولید می‌کند
                        if is_output_conv:
                            # main_weight shape: (29, ...) برای weight یا (29,) برای bias
                            # combined_weight shape: (31, ...) برای weight یا (31,) برای bias
                            
                            if len(main_weight.shape) == 4:  # Conv2d weight: (out_channels, in_channels, H, W)
                                if main_weight.shape[0] == 29 and combined_weight.shape[0] == 31:
                                    # کپی 29 لندمارک اول از مدل اصلی
                                    combined_weight = combined_weight.clone()
                                    combined_weight[:29] = main_weight
                                    
                                    # برای 2 لندمارک آخر (P1/P2)، از وزن‌های مدل P1/P2 استفاده می‌کنیم
                                    p1p2_decoder_key = 'heatmap_decoder.16.weight'
                                    if p1p2_decoder_key in p1p2_state_dict:
                                        p1p2_weight = p1p2_state_dict[p1p2_decoder_key]
                                        # p1p2_weight shape: (2, 32, 1, 1)
                                        # combined_weight shape: (31, in_channels, 1, 1)
                                        # باید مطمئن شویم که in_channels مطابقت دارد
                                        if p1p2_weight.shape[1] == combined_weight.shape[1]:
                                            combined_weight[29:31] = p1p2_weight
                                            copied_keys.append(f"{key} (29 from main, 2 from P1/P2 model)")
                                        else:
                                            # اگر in_channels متفاوت است، از میانگین استفاده می‌کنیم
                                            mean_weight = main_weight.mean(dim=0, keepdim=True)
                                            combined_weight[29:31] = mean_weight.expand(2, -1, -1, -1)
                                            copied_keys.append(f"{key} (29 from main, 2 initialized from mean - in_channels mismatch)")
                                    else:
                                        # اگر وزن P1/P2 پیدا نشد، از میانگین استفاده می‌کنیم
                                        mean_weight = main_weight.mean(dim=0, keepdim=True)
                                        combined_weight[29:31] = mean_weight.expand(2, -1, -1, -1)
                                        copied_keys.append(f"{key} (29 from main, 2 initialized from mean - P1/P2 weight not found)")
                                    
                                    combined_state_dict[key] = combined_weight
                                elif main_weight.shape == combined_weight.shape:
                                    combined_state_dict[key] = main_weight
                                    copied_keys.append(key)
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {main_weight.shape} vs {combined_weight.shape})")
                            
                            elif len(main_weight.shape) == 1:  # Conv2d bias: (out_channels,)
                                if main_weight.shape[0] == 29 and combined_weight.shape[0] == 31:
                                    # کپی 29 لندمارک اول از مدل اصلی
                                    combined_weight = combined_weight.clone()
                                    combined_weight[:29] = main_weight
                                    
                                    # برای 2 لندمارک آخر (P1/P2)، از bias مدل P1/P2 استفاده می‌کنیم
                                    p1p2_bias_key = 'heatmap_decoder.16.bias'
                                    if p1p2_bias_key in p1p2_state_dict:
                                        p1p2_bias = p1p2_state_dict[p1p2_bias_key]
                                        # p1p2_bias shape: (2,)
                                        combined_weight[29:31] = p1p2_bias
                                        copied_keys.append(f"{key} (29 from main, 2 from P1/P2 model)")
                                    else:
                                        # اگر bias P1/P2 پیدا نشد، از میانگین استفاده می‌کنیم
                                        mean_bias = main_weight.mean()
                                        combined_weight[29:31] = mean_bias
                                        copied_keys.append(f"{key} (29 from main, 2 initialized from mean - P1/P2 bias not found)")
                                    
                                    combined_state_dict[key] = combined_weight
                                elif main_weight.shape == combined_weight.shape:
                                    combined_state_dict[key] = main_weight
                                    copied_keys.append(key)
                                else:
                                    skipped_keys.append(f"{key} (shape mismatch: {main_weight.shape} vs {combined_weight.shape})")
                            
                            else:
                                # برای سایر موارد، اگر shape مطابقت دارد، مستقیماً کپی کن
                                if main_weight.shape == combined_weight.shape:
                                    combined_state_dict[key] = main_weight
                                    copied_keys.append(key)
                                else:
                                    skipped_keys.append(f"{key} (unexpected shape: {main_weight.shape})")
                        
                        else:
                            # برای لایه‌های دیگر (نه خروجی)، مستقیماً کپی کن
                            if main_weight.shape == combined_weight.shape:
                                combined_state_dict[key] = main_weight
                                copied_keys.append(key)
                            else:
                                skipped_keys.append(f"{key} (shape mismatch: {main_weight.shape} vs {combined_weight.shape})")
                    else:
                        skipped_keys.append(f"{key} (main_key not found: {main_key})")
                except (ValueError, IndexError) as e:
                    skipped_keys.append(f"{key} (parsing error: {e})")
            else:
                skipped_keys.append(f"{key} (unexpected format)")
    
    print(f"\n  Copied {len(copied_keys)} keys")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys)} keys")
        if len(skipped_keys) <= 10:
            for sk in skipped_keys:
                print(f"    - {sk}")
        else:
            for sk in skipped_keys[:10]:
                print(f"    - {sk}")
            print(f"    ... and {len(skipped_keys) - 10} more")
    
    # بارگذاری وزن‌ها در مدل
    combined_model.load_state_dict(combined_state_dict, strict=False)
    
    # تست مدل
    print(f"\n[Testing] Testing combined model...")
    combined_model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 768, 768)
        output = combined_model(dummy_input)
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: (1, 31, H, W)")
        if output.shape[1] == 31:
            print(f"  [OK] Model output shape is correct!")
        else:
            print(f"  [WARNING] Expected 31 landmarks, got {output.shape[1]}")
    
    # ذخیره checkpoint
    print(f"\n[Saving] Saving combined model to {output_path}...")
    combined_checkpoint = {
        'model_state_dict': combined_model.state_dict(),
        'num_landmarks': 31,
        'width': detected_width,
        'image_size': main_checkpoint.get('image_size', 768),
        'architecture': 'HRNetLandmarkModel',
        'source_models': {
            'main_model': main_model_path,
            'p1p2_model': p1p2_model_path
        },
        'best_mre': main_checkpoint.get('best_mre', None),
        'epoch': main_checkpoint.get('epoch', 0),
    }
    
    # کپی سایر اطلاعات مفید از checkpoint اصلی
    if 'best_loss' in main_checkpoint:
        combined_checkpoint['best_loss'] = main_checkpoint['best_loss']
    
    torch.save(combined_checkpoint, output_path)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  [OK] Model saved! Size: {file_size_mb:.2f} MB")
    
    print("\n" + "="*80)
    print("[OK] Combined model created successfully!")
    print("="*80)
    print(f"\nModel details:")
    print(f"  - Architecture: HRNetLandmarkModel")
    print(f"  - Width: {detected_width}")
    print(f"  - Number of landmarks: 31 (29 anatomical + 2 calibration)")
    print(f"  - Input size: 768x768")
    print(f"  - Output: Heatmaps for 31 landmarks")
    print(f"  - File: {output_path}")
    print(f"  - Size: {file_size_mb:.2f} MB")
    
    return combined_model, combined_checkpoint


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create combined HRNet model with 31 landmarks')
    parser.add_argument('--main_model', type=str, default='checkpoint_best_768.pth',
                        help='Path to main model (29 landmarks)')
    parser.add_argument('--p1p2_model', type=str, default='models/hrnet_p1p2_heatmap_best.pth',
                        help='Path to P1/P2 model (2 landmarks)')
    parser.add_argument('--output', type=str, default='checkpoint_best_768_combined_31.pth',
                        help='Output path for combined model')
    parser.add_argument('--width', type=int, default=32,
                        help='HRNet width (default: 32)')
    
    args = parser.parse_args()
    
    try:
        model, checkpoint = create_combined_hrnet_model(
            main_model_path=args.main_model,
            p1p2_model_path=args.p1p2_model,
            output_path=args.output,
            width=args.width
        )
        print("\n[OK] Success! Combined model is ready to use.")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

