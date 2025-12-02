"""
بررسی و اصلاح مشکل لود کردن مدل
مشکل: وقتی pretrained=True استفاده می‌کنیم، timm مدل را با pretrained weights لود می‌کند
اما سپس state_dict train شده را لود می‌کنیم که ممکن است conflict ایجاد کند
"""
import torch
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector


def load_model_correctly(checkpoint_path, device='cuda'):
    """Load model correctly - first create without pretrained, then load weights"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    image_size = checkpoint.get('image_size', 1024)
    heatmap_size = checkpoint.get('heatmap_size', 256)
    
    # Strategy 1: Create model WITHOUT pretrained weights first
    # Then load the trained weights
    print("[INFO] Creating model WITHOUT pretrained weights...")
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=False,  # Don't load pretrained weights
        output_size=heatmap_size
    )
    
    # Now load the trained weights
    state_dict = checkpoint['model_state_dict']
    model_state = model.state_dict()
    
    # Filter state_dict to only include keys that exist in model
    # and have matching shapes
    filtered_state = {}
    for k, v in state_dict.items():
        if k in model_state:
            if model_state[k].shape == v.shape:
                filtered_state[k] = v
            else:
                print(f"  [SKIP] Shape mismatch: {k} - model: {model_state[k].shape}, checkpoint: {v.shape}")
        else:
            print(f"  [SKIP] Key not in model: {k}")
    
    print(f"[INFO] Loading {len(filtered_state)}/{len(state_dict)} matching keys")
    
    # Load filtered weights
    model.load_state_dict(filtered_state, strict=False)
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully")
    print(f"  - Image size: {image_size}x{image_size}")
    print(f"  - Heatmap size: {heatmap_size}x{heatmap_size}")
    print(f"  - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  - Pixel Error (from checkpoint): {checkpoint.get('pixel_error', 0):.2f} px")
    
    return model, image_size, heatmap_size


if __name__ == '__main__':
    checkpoint_path = 'models/hrnet_p1p2_heatmap_best.pth'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model, image_size, heatmap_size = load_model_correctly(checkpoint_path, device)
    
    # Test with a dummy input
    print("\n[TEST] Testing model with dummy input...")
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    
    with torch.no_grad():
        heatmaps = model(dummy_input)
        coords = model.extract_coordinates(heatmaps)
    
    print(f"  Heatmap shape: {heatmaps.shape}")
    print(f"  Heatmap min/max: {heatmaps.min().item():.4f} / {heatmaps.max().item():.4f}")
    print(f"  Heatmap has NaN: {torch.isnan(heatmaps).any().item()}")
    print(f"  Coords: {coords.cpu().numpy()[0]}")
    print(f"  Coords has NaN: {torch.isnan(coords).any().item()}")
    
    if torch.isnan(heatmaps).any() or torch.isnan(coords).any():
        print("\n[ERROR] Model output contains NaN!")
    else:
        print("\n[OK] Model works correctly!")






