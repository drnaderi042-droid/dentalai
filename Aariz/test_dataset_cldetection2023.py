"""Test script for CLdetection2023 dataset loader"""
from dataset_cldetection2023 import create_dataloaders
import sys

print('Testing dataset loader...')
try:
    train_loader, val_loader, test_loader = create_dataloaders(
        'C:/Users/Salah/Downloads/Compressed/Dentalai/main - Copy/CLdetection2023',
        batch_size=2,
        num_workers=0,
        image_size=(768, 768),
        use_heatmap=True
    )
    print(f'Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}')
    
    batch = next(iter(train_loader))
    print(f'Batch image shape: {batch["image"].shape}')
    print(f'Batch target shape: {batch["target"].shape}')
    print('Dataset loader works!')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)









