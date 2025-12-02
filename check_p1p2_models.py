"""
اسکریپت برای بررسی مدل‌های P1/P2 و پیدا کردن مدل با 100 تصویر
"""
import torch
import os
from pathlib import Path

def check_checkpoint(path):
    """بررسی اطلاعات یک checkpoint"""
    try:
        path_str = str(path) if isinstance(path, Path) else path
        if not os.path.exists(path_str):
            return {
                'path': path_str,
                'exists': False,
                'error': 'File not found'
            }
        checkpoint = torch.load(path_str, map_location='cpu', weights_only=False)
        
        info = {
            'path': path,
            'exists': True,
            'keys': list(checkpoint.keys()),
            'epoch': checkpoint.get('epoch', 'N/A'),
            'pixel_error': checkpoint.get('pixel_error', 'N/A'),
            'image_size': checkpoint.get('image_size', 'N/A'),
            'heatmap_size': checkpoint.get('heatmap_size', 'N/A'),
            'num_images': checkpoint.get('num_images', checkpoint.get('num_train_images', 'N/A')),
            'num_epochs': checkpoint.get('num_epochs', 'N/A'),
        }
        return info
    except Exception as e:
        return {
            'path': path,
            'exists': False,
            'error': str(e)
        }

# لیست مسیرهای ممکن برای مدل‌های P1/P2
base_dir = Path(__file__).parent
checkpoint_paths = [
    base_dir / 'aariz' / 'models' / 'hrnet_p1p2_heatmap_best.pth',
    base_dir / 'Aariz' / 'models' / 'hrnet_p1p2_heatmap_best.pth',
    base_dir / 'checkpoints_p1_p2' / 'checkpoint_best.pth' / 'checkpoint_latest.pth',
    base_dir / 'checkpoints_p1_p2_v2' / 'checkpoint_best.pth' / 'checkpoint_latest.pth',
    base_dir / 'checkpoints_p1_p2_v2' / 'checkpoint_epoch_20.pth' / 'checkpoint_latest.pth',
    base_dir / 'checkpoints_p1_p2_v2' / 'checkpoint_epoch_10.pth' / 'checkpoint_latest.pth',
    base_dir / 'checkpoints_p1_p2' / 'checkpoint_epoch_10.pth' / 'checkpoint_latest.pth',
]

print("=" * 80)
print("Checking P1/P2 Models")
print("=" * 80)
print()

found_models = []
for path in checkpoint_paths:
    print(f"Checking: {path}")
    info = check_checkpoint(path)
    
    if info['exists']:
        print(f"   [OK] Found")
        print(f"   Epoch: {info['epoch']}")
        print(f"   Pixel Error: {info['pixel_error']}")
        print(f"   Image Size: {info['image_size']}")
        print(f"   Heatmap Size: {info['heatmap_size']}")
        print(f"   Num Images: {info['num_images']}")
        print(f"   Keys: {', '.join(info['keys'][:5])}...")
        found_models.append(info)
    else:
        print(f"   [NOT FOUND]: {info.get('error', 'File not found')}")
    print()

print("=" * 80)
print("Summary of found models:")
print("=" * 80)
for i, model in enumerate(found_models, 1):
    print(f"{i}. {model['path']}")
    print(f"   - Epoch: {model['epoch']}, Pixel Error: {model['pixel_error']}, Num Images: {model['num_images']}")
    print()

# Find models with 100 images
models_with_100 = [m for m in found_models if m['num_images'] == 100]
if models_with_100:
    print("=" * 80)
    print("Models trained with 100 images:")
    print("=" * 80)
    for model in models_with_100:
        print(f"   - {model['path']} (Epoch: {model['epoch']}, Pixel Error: {model['pixel_error']})")
else:
    print("WARNING: No model with 100 images found")

