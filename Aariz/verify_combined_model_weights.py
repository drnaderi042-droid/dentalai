"""
اسکریپت برای بررسی وزن‌های کپی شده در مدل ترکیبی
"""
import torch
import numpy as np
from pathlib import Path
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import HRNetLandmarkModel


def compare_weights(main_model_path, combined_model_path):
    """
    مقایسه وزن‌های مدل اصلی و مدل ترکیبی
    """
    print("="*80)
    print("Verifying Combined Model Weights")
    print("="*80)
    
    # بارگذاری checkpoint ها
    print("\n[1/4] Loading checkpoints...")
    main_checkpoint = torch.load(main_model_path, map_location='cpu', weights_only=False)
    combined_checkpoint = torch.load(combined_model_path, map_location='cpu', weights_only=False)
    
    main_state_dict = main_checkpoint.get('model_state_dict', main_checkpoint)
    combined_state_dict = combined_checkpoint.get('model_state_dict', combined_checkpoint)
    
    print(f"  Main model keys: {len(main_state_dict)}")
    print(f"  Combined model keys: {len(combined_state_dict)}")
    
    # ایجاد مدل‌ها برای بررسی ساختار
    print("\n[2/4] Creating model instances...")
    main_model = HRNetLandmarkModel(num_landmarks=29, width=32)
    combined_model = HRNetLandmarkModel(num_landmarks=31, width=32)
    
    main_model.load_state_dict(main_state_dict, strict=False)
    combined_model.load_state_dict(combined_state_dict, strict=False)
    
    # بررسی وزن‌های backbone
    print("\n[3/4] Checking backbone weights...")
    backbone_prefixes = ['stem.', 'stage1.', 'stage2.', 'stage3.', 'stage4.', 
                         'transition1.', 'transition2.', 'transition3.']
    
    backbone_matches = 0
    backbone_mismatches = 0
    backbone_missing = 0
    
    for key in combined_state_dict.keys():
        if any(key.startswith(prefix) for prefix in backbone_prefixes):
            if key in main_state_dict:
                main_weight = main_state_dict[key]
                combined_weight = combined_state_dict[key]
                
                if torch.equal(main_weight, combined_weight):
                    backbone_matches += 1
                else:
                    backbone_mismatches += 1
                    if backbone_mismatches <= 5:  # فقط 5 مورد اول را نمایش بده
                        print(f"  [MISMATCH] {key}: shapes match but values differ")
            else:
                backbone_missing += 1
                if backbone_missing <= 5:
                    print(f"  [MISSING] {key}: not in main model")
    
    print(f"  Backbone weights: {backbone_matches} matches, {backbone_mismatches} mismatches, {backbone_missing} missing")
    
    # بررسی وزن‌های final_layers
    print("\n[4/4] Checking final_layers weights...")
    final_layers_matches = 0
    final_layers_mismatches = 0
    final_layers_issues = []
    
    for key in combined_state_dict.keys():
        if key.startswith('final_layers.'):
            parts = key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                module_idx = int(parts[2])
                
                # تشخیص آخرین Conv2d
                is_output_conv = False
                if layer_idx == 0:
                    is_output_conv = (module_idx == 3)
                else:
                    is_output_conv = (module_idx == 4)
                
                main_key = '.'.join(['final_layers', str(layer_idx), str(module_idx)] + (parts[3:] if len(parts) > 3 else []))
                
                if main_key in main_state_dict:
                    main_weight = main_state_dict[main_key]
                    combined_weight = combined_state_dict[key]
                    
                    if is_output_conv:
                        # برای آخرین Conv2d، باید 29 لندمارک اول مطابقت داشته باشند
                        if len(main_weight.shape) >= 1:
                            if main_weight.shape[0] == 29 and combined_weight.shape[0] == 31:
                                # بررسی 29 لندمارک اول
                                if torch.equal(main_weight, combined_weight[:29]):
                                    final_layers_matches += 1
                                else:
                                    final_layers_mismatches += 1
                                    final_layers_issues.append({
                                        'key': key,
                                        'type': 'output_conv_29_mismatch',
                                        'layer': layer_idx,
                                        'module': module_idx
                                    })
                            elif main_weight.shape == combined_weight.shape:
                                if torch.equal(main_weight, combined_weight):
                                    final_layers_matches += 1
                                else:
                                    final_layers_mismatches += 1
                                    final_layers_issues.append({
                                        'key': key,
                                        'type': 'output_conv_shape_match_value_mismatch',
                                        'layer': layer_idx,
                                        'module': module_idx
                                    })
                            else:
                                final_layers_issues.append({
                                    'key': key,
                                    'type': 'output_conv_shape_mismatch',
                                    'layer': layer_idx,
                                    'module': module_idx,
                                    'main_shape': main_weight.shape,
                                    'combined_shape': combined_weight.shape
                                })
                    else:
                        # برای لایه‌های دیگر، باید کاملاً مطابقت داشته باشند
                        if main_weight.shape == combined_weight.shape:
                            if torch.equal(main_weight, combined_weight):
                                final_layers_matches += 1
                            else:
                                final_layers_mismatches += 1
                                final_layers_issues.append({
                                    'key': key,
                                    'type': 'non_output_conv_mismatch',
                                    'layer': layer_idx,
                                    'module': module_idx
                                })
                        else:
                            final_layers_issues.append({
                                'key': key,
                                'type': 'non_output_conv_shape_mismatch',
                                'layer': layer_idx,
                                'module': module_idx,
                                'main_shape': main_weight.shape,
                                'combined_shape': combined_weight.shape
                            })
                else:
                    final_layers_issues.append({
                        'key': key,
                        'type': 'not_in_main',
                        'layer': layer_idx,
                        'module': module_idx
                    })
    
    print(f"  Final layers weights: {final_layers_matches} matches, {final_layers_mismatches} mismatches")
    
    # نمایش مشکلات
    if final_layers_issues:
        print(f"\n  Found {len(final_layers_issues)} issues in final_layers:")
        for i, issue in enumerate(final_layers_issues[:10]):  # فقط 10 مورد اول
            print(f"    [{i+1}] {issue['key']}")
            print(f"        Type: {issue['type']}, Layer: {issue['layer']}, Module: {issue['module']}")
            if 'main_shape' in issue:
                print(f"        Main shape: {issue['main_shape']}, Combined shape: {issue['combined_shape']}")
        if len(final_layers_issues) > 10:
            print(f"    ... and {len(final_layers_issues) - 10} more issues")
    
    # خلاصه
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Backbone weights:")
    print(f"  - Matches: {backbone_matches}")
    print(f"  - Mismatches: {backbone_mismatches}")
    print(f"  - Missing: {backbone_missing}")
    print(f"\nFinal layers weights:")
    print(f"  - Matches: {final_layers_matches}")
    print(f"  - Mismatches: {final_layers_mismatches}")
    print(f"  - Issues: {len(final_layers_issues)}")
    
    # نتیجه نهایی
    total_backbone = backbone_matches + backbone_mismatches + backbone_missing
    if backbone_mismatches == 0 and backbone_missing == 0:
        print(f"\n[OK] All backbone weights copied correctly!")
    else:
        print(f"\n[WARNING] Some backbone weights have issues!")
    
    if final_layers_mismatches == 0 and len(final_layers_issues) == 0:
        print(f"[OK] All final_layers weights copied correctly!")
    else:
        print(f"[WARNING] Some final_layers weights have issues!")
    
    # بررسی وزن‌های P1/P2 (2 لندمارک آخر)
    print("\n" + "="*80)
    print("Checking P1/P2 weights initialization (last 2 landmarks)")
    print("="*80)
    
    p1p2_issues = []
    for key in combined_state_dict.keys():
        if key.startswith('final_layers.'):
            parts = key.split('.')
            if len(parts) >= 3:
                layer_idx = int(parts[1])
                module_idx = int(parts[2])
                
                is_output_conv = False
                if layer_idx == 0:
                    is_output_conv = (module_idx == 3)
                else:
                    is_output_conv = (module_idx == 4)
                
                if is_output_conv:
                    combined_weight = combined_state_dict[key]
                    if len(combined_weight.shape) >= 1 and combined_weight.shape[0] == 31:
                        # بررسی وزن‌های P1/P2 (index 29 و 30)
                        p1p2_weights = combined_weight[29:31]
                        
                        # بررسی اینکه آیا P1/P2 با میانگین وزن‌های موجود مقداردهی شده‌اند
                        main_29_weights = combined_weight[:29]
                        mean_weight = main_29_weights.mean(dim=0, keepdim=True)
                        
                        # مقایسه P1/P2 با میانگین
                        if len(combined_weight.shape) == 4:  # Conv2d weight
                            expected_p1p2 = mean_weight.expand(2, -1, -1, -1)
                            if torch.allclose(p1p2_weights, expected_p1p2, atol=1e-5):
                                print(f"  [OK] {key}: P1/P2 weights initialized from mean (layer {layer_idx})")
                            else:
                                p1p2_issues.append({
                                    'key': key,
                                    'layer': layer_idx,
                                    'issue': 'P1/P2 weights not matching mean initialization'
                                })
                        elif len(combined_weight.shape) == 1:  # Conv2d bias
                            mean_bias = main_29_weights.mean()
                            expected_p1p2 = torch.full((2,), mean_bias.item())
                            if torch.allclose(p1p2_weights, expected_p1p2, atol=1e-5):
                                print(f"  [OK] {key}: P1/P2 bias initialized from mean (layer {layer_idx})")
                            else:
                                p1p2_issues.append({
                                    'key': key,
                                    'layer': layer_idx,
                                    'issue': 'P1/P2 bias not matching mean initialization'
                                })
    
    if p1p2_issues:
        print(f"\n[WARNING] Found {len(p1p2_issues)} issues with P1/P2 initialization:")
        for issue in p1p2_issues:
            print(f"  - {issue['key']}: {issue['issue']}")
    else:
        print(f"\n[OK] All P1/P2 weights initialized correctly from mean!")
    
    print("\n" + "="*80)
    return {
        'backbone_matches': backbone_matches,
        'backbone_mismatches': backbone_mismatches,
        'backbone_missing': backbone_missing,
        'final_layers_matches': final_layers_matches,
        'final_layers_mismatches': final_layers_mismatches,
        'final_layers_issues': len(final_layers_issues),
        'p1p2_issues': len(p1p2_issues)
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Verify combined model weights')
    parser.add_argument('--main_model', type=str, default='checkpoint_best_768.pth',
                        help='Path to main model (29 landmarks)')
    parser.add_argument('--combined_model', type=str, default='checkpoint_best_768_combined_31.pth',
                        help='Path to combined model (31 landmarks)')
    
    args = parser.parse_args()
    
    if not Path(args.main_model).exists():
        print(f"[ERROR] Main model not found: {args.main_model}")
        sys.exit(1)
    
    if not Path(args.combined_model).exists():
        print(f"[ERROR] Combined model not found: {args.combined_model}")
        sys.exit(1)
    
    try:
        results = compare_weights(args.main_model, args.combined_model)
        print("\n[OK] Verification completed!")
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




