import os
import sys

# Check if the model file exists and what parameters it contains
model_path = 'Aariz/models/hrnet_p1p2_heatmap_best.pth'
print(f"Checking model file: {model_path}")
print(f"File exists: {os.path.exists(model_path)}")

if os.path.exists(model_path):
    try:
        import torch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Check what parameters are stored in the checkpoint
        print(f"Checkpoint has these keys: {list(checkpoint.keys())}")
        
        # Check for image_size and heatmap_size
        image_size = checkpoint.get('image_size', 'NOT FOUND')
        heatmap_size = checkpoint.get('heatmap_size', 'NOT FOUND')
        
        print(f"Stored image_size in checkpoint: {image_size}")
        print(f"Stored heatmap_size in checkpoint: {heatmap_size}")
        
        # Also check if there's an args section
        if 'args' in checkpoint:
            print(f"Args section exists: {checkpoint['args']}")
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
else:
    print("Model file not found!")