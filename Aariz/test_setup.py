"""
Quick test to verify the optimized setup is working correctly
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'

print("Testing imports...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA version: {torch.version.cuda}")
except Exception as e:
    print(f"✗ PyTorch error: {e}")
    exit(1)

try:
    from dataset import create_dataloaders
    print("✓ dataset module")
except Exception as e:
    print(f"✗ dataset error: {e}")
    exit(1)

try:
    from model import get_model
    print("✓ model module")
except Exception as e:
    print(f"✗ model error: {e}")
    exit(1)

try:
    from utils import heatmap_to_coordinates, calculate_mre, calculate_sdr, save_checkpoint, load_checkpoint
    print("✓ utils module")
except Exception as e:
    print(f"✗ utils error: {e}")
    exit(1)

print("\nTesting model creation...")
try:
    model = get_model('hrnet', num_landmarks=29)
    print(f"✓ HRNet model created")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model has {total_params:,} parameters ({total_params/1e6:.1f}M)")
except Exception as e:
    print(f"✗ Model creation error: {e}")
    exit(1)

print("\nTesting dataset...")
try:
    if not os.path.exists('Aariz'):
        print("⚠ Dataset folder 'Aariz' not found!")
        print("  Please ensure the dataset is in the correct location")
    else:
        print("✓ Dataset folder found")
        
        # Try to load a small batch
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset_folder_path='Aariz',
            batch_size=2,
            num_workers=0,
            image_size=(512, 512),
            use_heatmap=True
        )
        print(f"✓ DataLoaders created")
        print(f"  Train: {len(train_loader.dataset)} samples")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test: {len(test_loader.dataset)} samples")
        
        # Try to load one batch
        batch = next(iter(train_loader))
        print(f"✓ Sample batch loaded")
        print(f"  Image shape: {batch['image'].shape}")
        print(f"  Target shape: {batch['target'].shape}")
except Exception as e:
    print(f"✗ Dataset error: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting GPU...")
if torch.cuda.is_available():
    try:
        device = torch.device('cuda')
        model = model.to(device)
        print(f"✓ Model moved to GPU")
        
        # Test forward pass
        test_input = torch.randn(1, 3, 512, 512).to(device)
        with torch.no_grad():
            output = model(test_input)
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output.shape}")
        
        # Check memory
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"✓ GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
    except Exception as e:
        print(f"✗ GPU error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ CUDA not available - will train on CPU (very slow)")

print("\n" + "="*60)
print("SETUP TEST COMPLETE!")
print("="*60)

if torch.cuda.is_available() and os.path.exists('Aariz'):
    print("\n✓ Everything looks good! You can now run:")
    print("  python train_optimized.py --mixed_precision --use_ema")
else:
    if not torch.cuda.is_available():
        print("\n⚠ Warning: CUDA not available")
        print("  Training will be very slow on CPU")
    if not os.path.exists('Aariz'):
        print("\n⚠ Warning: Dataset not found")
        print("  Please place the Aariz dataset in this directory")