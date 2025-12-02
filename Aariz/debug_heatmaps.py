"""
Debug script to check heatmap values
"""
import torch
import numpy as np
from dataset import create_dataloaders
from model import get_model

# Create model and dataloader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model('resnet', num_landmarks=29)
model = model.to(device)
model.eval()

train_loader, _, _ = create_dataloaders(
    dataset_folder_path='Aariz',
    batch_size=1,
    num_workers=0,
    image_size=(128, 128),
    use_heatmap=True,
    annotation_type='Senior Orthodontists'
)

# Get one batch
batch = next(iter(train_loader))
images = batch['image'].to(device)
targets = batch['target'].to(device)
landmarks = batch['landmarks'].cpu().numpy()

# Forward pass
with torch.no_grad():
    outputs = model(images)
    
    # Apply sigmoid
    heatmaps_pred = torch.sigmoid(outputs).cpu().numpy()[0]
    heatmaps_target = targets.cpu().numpy()[0]

print("="*60)
print("HEATMAP ANALYSIS")
print("="*60)
print(f"Predicted heatmaps shape: {heatmaps_pred.shape}")
print(f"Target heatmaps shape: {heatmaps_target.shape}")
print()

# Check values
for i in range(min(5, heatmaps_pred.shape[0])):
    pred_hm = heatmaps_pred[i]
    target_hm = heatmaps_target[i]
    
    print(f"Landmark {i}:")
    print(f"  Predicted: min={pred_hm.min():.4f}, max={pred_hm.max():.4f}, mean={pred_hm.mean():.4f}")
    print(f"  Target:    min={target_hm.min():.4f}, max={target_hm.max():.4f}, mean={target_hm.mean():.4f}")
    
    # Find max location
    pred_max_idx = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
    target_max_idx = np.unravel_index(target_hm.argmax(), target_hm.shape)
    
    print(f"  Predicted max at: ({pred_max_idx[1]}, {pred_max_idx[0]}) = {pred_hm.max():.4f}")
    print(f"  Target max at:    ({target_max_idx[1]}, {target_max_idx[0]}) = {target_hm.max():.4f}")
    
    # Check landmark
    if landmarks[0, i, 0] >= 0:
        print(f"  True landmark: ({landmarks[0, i, 0]:.1f}, {landmarks[0, i, 1]:.1f})")
    else:
        print(f"  True landmark: Invalid")
    print()

print("="*60)
print("THRESHOLD TEST")
print("="*60)
# Test different thresholds
for threshold in [0.01, 0.05, 0.1, 0.2, 0.3]:
    valid_count = 0
    for i in range(heatmaps_pred.shape[0]):
        if heatmaps_pred[i].max() > threshold:
            valid_count += 1
    print(f"Threshold {threshold:.2f}: {valid_count}/{heatmaps_pred.shape[0]} landmarks valid")



