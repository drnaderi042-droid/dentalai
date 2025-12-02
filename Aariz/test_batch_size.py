"""
Script to test maximum batch size for GPU
اسکریپت برای تست حداکثر batch size قابل استفاده
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse

from dataset import create_dataloaders
from model import get_model


def test_batch_size(model_name='resnet', image_size=(128, 128), 
                    start_batch=8, max_batch=64, mixed_precision=False):
    """
    Test different batch sizes to find maximum
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    print(f"\nTesting batch sizes from {start_batch} to {max_batch}")
    print(f"Model: {model_name}, Image Size: {image_size}")
    print(f"Mixed Precision: {mixed_precision}\n")
    
    # Create model
    model = get_model(model_name, num_landmarks=29)
    model = model.to(device)
    
    # Create a small dummy dataset for testing
    try:
        train_loader, _, _ = create_dataloaders(
            dataset_folder_path='Aariz',
            batch_size=start_batch,
            num_workers=0,  # No workers for testing
            image_size=image_size,
            use_heatmap=True,
            annotation_type='Senior Orthodontists'
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Loss function (dummy for testing)
    criterion = nn.MSELoss()
    
    # Mixed precision scaler
    scaler = None
    if mixed_precision and device.type == 'cuda':
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
    
    # Test different batch sizes
    working_batch_sizes = []
    failed_batch_sizes = []
    
    for batch_size in range(start_batch, max_batch + 1, 4):
        try:
            print(f"Testing batch_size={batch_size}...", end=" ")
            
            # Create new dataloader with this batch size
            train_loader, _, _ = create_dataloaders(
                dataset_folder_path='Aariz',
                batch_size=batch_size,
                num_workers=0,
                image_size=image_size,
                use_heatmap=True,
                annotation_type='Senior Orthodontists'
            )
            
            # Clear cache
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Try one forward + backward pass
            model.train()
            batch = next(iter(train_loader))
            images = batch['image'].to(device, non_blocking=True)
            targets = batch['target'].to(device, non_blocking=True)
            
            # Forward pass
            if mixed_precision and scaler is not None:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Backward pass
                scaler.scale(loss).backward()
                scaler.step(torch.optim.Adam(model.parameters(), lr=1e-4))
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
            
            # Check VRAM usage
            if device.type == 'cuda':
                memory_used = torch.cuda.memory_allocated(0) / 1024**3
                memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
                print(f"[OK] (VRAM: {memory_used:.2f}GB used, {memory_reserved:.2f}GB reserved)")
            else:
                print(f"[OK]")
            
            working_batch_sizes.append(batch_size)
            
            # Clean up
            del images, targets, outputs, loss, batch
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                print(f"[FAIL] Out of Memory")
                failed_batch_sizes.append(batch_size)
                break  # Stop testing larger batch sizes
            else:
                print(f"[FAIL] Error: {e}")
                failed_batch_sizes.append(batch_size)
        except Exception as e:
            print(f"[FAIL] Error: {e}")
            failed_batch_sizes.append(batch_size)
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    
    # Results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    if working_batch_sizes:
        max_working = max(working_batch_sizes)
        print(f"[OK] Maximum working batch size: {max_working}")
        print(f"[OK] Working batch sizes: {working_batch_sizes}")
        print(f"\n[RECOMMENDATION] Use batch_size={max_working} for training")
    else:
        print("[FAIL] No working batch size found!")
        print("       Try reducing image_size or using a smaller model")
    
    if failed_batch_sizes:
        print(f"\n[FAIL] Failed at batch_size={min(failed_batch_sizes)}")
    
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test maximum batch size')
    parser.add_argument('--model', type=str, default='resnet', 
                       choices=['resnet', 'unet', 'hourglass'],
                       help='Model architecture')
    parser.add_argument('--image_size', type=int, nargs=2, default=[128, 128],
                       help='Image size (H W)')
    parser.add_argument('--start_batch', type=int, default=8,
                       help='Starting batch size')
    parser.add_argument('--max_batch', type=int, default=64,
                       help='Maximum batch size to test')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision')
    
    args = parser.parse_args()
    
    test_batch_size(
        model_name=args.model,
        image_size=tuple(args.image_size),
        start_batch=args.start_batch,
        max_batch=args.max_batch,
        mixed_precision=args.mixed_precision
    )

