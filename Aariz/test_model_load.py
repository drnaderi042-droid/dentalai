"""
Test script to debug model loading issues
"""
import torch
import sys
from pathlib import Path

# Add aariz to path
aariz_dir = Path(__file__).parent
sys.path.insert(0, str(aariz_dir))

print("="*70)
print("Testing P1/P2 Model Loading")
print("="*70)

# Check timm
try:
    import timm
    print(f"[OK] timm available: {timm.__version__}")
except ImportError:
    print("[ERROR] timm not available")
    sys.exit(1)

# Check model file
model_path = aariz_dir / 'models' / 'hrnet_p1p2_heatmap_best.pth'
if not model_path.exists():
    print(f"[ERROR] Model not found: {model_path}")
    sys.exit(1)

print(f"[OK] Model file exists: {model_path}")

# Load checkpoint
print("\nLoading checkpoint...")
try:
    checkpoint = torch.load(model_path, map_location='cpu')
    print(f"[OK] Checkpoint loaded")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Image size: {checkpoint.get('image_size', 'unknown')}")
    print(f"   Heatmap size: {checkpoint.get('heatmap_size', 'unknown')}")
except Exception as e:
    print(f"[ERROR] Error loading checkpoint: {e}")
    sys.exit(1)

# Import model
print("\nImporting model class...")
try:
    from model_heatmap import HRNetP1P2HeatmapDetector
    print("[OK] Model class imported")
except ImportError as e:
    print(f"[ERROR] Error importing model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Create model
print("\nCreating model...")
try:
    image_size = checkpoint.get('image_size', 768)
    heatmap_size = checkpoint.get('heatmap_size', 192)
    
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=False,
        output_size=heatmap_size
    )
    print("[OK] Model created")
    
    # Get model keys
    model_keys = set(model.state_dict().keys())
    checkpoint_keys = set(checkpoint['model_state_dict'].keys())
    
    print(f"\nModel keys count: {len(model_keys)}")
    print(f"Checkpoint keys count: {len(checkpoint_keys)}")
    
    missing_keys = model_keys - checkpoint_keys
    unexpected_keys = checkpoint_keys - model_keys
    
    if missing_keys:
        print(f"\n[WARNING] Missing keys in checkpoint ({len(missing_keys)}):")
        for key in list(missing_keys)[:10]:
            print(f"   - {key}")
        if len(missing_keys) > 10:
            print(f"   ... and {len(missing_keys) - 10} more")
    
    if unexpected_keys:
        print(f"\n[WARNING] Unexpected keys in checkpoint ({len(unexpected_keys)}):")
        for key in list(unexpected_keys)[:10]:
            print(f"   - {key}")
        if len(unexpected_keys) > 10:
            print(f"   ... and {len(unexpected_keys) - 10} more")
    
    if not missing_keys and not unexpected_keys:
        print("\n[OK] All keys match!")
    
    # Try loading
    print("\nLoading state dict...")
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print("[OK] Model loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"[WARNING] Strict loading failed: {e}")
        print("\nTrying with strict=False...")
        try:
            result = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print("[OK] Model loaded successfully (strict=False)")
            if result.missing_keys:
                print(f"   Missing keys: {len(result.missing_keys)}")
            if result.unexpected_keys:
                print(f"   Unexpected keys: {len(result.unexpected_keys)}")
        except Exception as e2:
            print(f"[ERROR] Failed even with strict=False: {e2}")
            sys.exit(1)
    
    print("\n" + "="*70)
    print("[OK] All tests passed! Model can be loaded successfully.")
    print("="*70)
    
except Exception as e:
    print(f"[ERROR] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

