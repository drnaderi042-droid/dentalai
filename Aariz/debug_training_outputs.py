"""
Debug script to check what the model is actually predicting.
"""

import torch
import numpy as np
from pathlib import Path
import sys
import cv2

sys.path.append(str(Path(__file__).parent))
from finetune_p1_p2_cldetection import P1P2ModelWithCLDetectionBackbone, P1P2DatasetFromJSON

# Load checkpoint
checkpoint_path = 'checkpoint_p1_p2_cldetection.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

print("="*60)
print("DEBUG: Checking Model Outputs")
print("="*60)
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
print(f"MRE: {checkpoint.get('mre_avg', 'N/A'):.2f}px")
print(f"Image Size: {checkpoint.get('image_size', 'N/A')}")

# Create dataset
dataset = P1P2DatasetFromJSON(
    image_dir='Aariz/train/Cephalograms',
    annotations_json='annotations_p1_p2.json',
    image_size=1024
)

# Create model - check which backbone was used
device = 'cuda' if torch.cuda.is_available() else 'cpu'
state_dict = checkpoint['model_state_dict']
has_resnet_backbone = any('backbone.0.weight' in k or 'backbone.4.0.conv1.weight' in k for k in state_dict.keys())

if has_resnet_backbone:
    print("Detected: ResNet18 backbone")
    # Use ResNet18 fallback
    model = P1P2ModelWithCLDetectionBackbone(
        cldetection_model_path=None,  # Force fallback
        device=device,
        freeze_backbone=True
    )
else:
    print("Detected: CLdetection2023 backbone")
    cldetection_path = checkpoint.get('cldetection_model_path', 
        r"C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth")
    model = P1P2ModelWithCLDetectionBackbone(
        cldetection_model_path=cldetection_path,
        device=device,
        freeze_backbone=True
    )

model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model = model.to(device)
model.eval()

# Test on a few samples
print("\n" + "="*60)
print("Sample Predictions:")
print("="*60)

with torch.no_grad():
    for i in range(min(5, len(dataset))):
        image_tensor, landmarks_gt, image_id = dataset[i]
        image_tensor = image_tensor.unsqueeze(0).to(device)
        landmarks_gt = landmarks_gt.unsqueeze(0).to(device)
        
        # Predict
        outputs = model(image_tensor)
        
        # Convert to pixel coordinates
        outputs_px = outputs.cpu().numpy()[0] * 1024
        landmarks_gt_px = landmarks_gt.cpu().numpy()[0] * 1024
        
        # Calculate errors
        error_p1 = np.sqrt(np.sum((outputs_px[:2] - landmarks_gt_px[:2]) ** 2))
        error_p2 = np.sqrt(np.sum((outputs_px[2:] - landmarks_gt_px[2:]) ** 2))
        
        print(f"\nImage {i+1}: {image_id}")
        print(f"  GT P1: ({landmarks_gt_px[0]:.1f}, {landmarks_gt_px[1]:.1f})")
        print(f"  Pred P1: ({outputs_px[0]:.1f}, {outputs_px[1]:.1f})")
        print(f"  Error P1: {error_p1:.1f}px")
        print(f"  GT P2: ({landmarks_gt_px[2]:.1f}, {landmarks_gt_px[3]:.1f})")
        print(f"  Pred P2: ({outputs_px[2]:.1f}, {outputs_px[3]:.1f})")
        print(f"  Error P2: {error_p2:.1f}px")
        
        # Check normalized outputs
        print(f"  Normalized outputs: {outputs.cpu().numpy()[0]}")
        print(f"  Normalized GT: {landmarks_gt.cpu().numpy()[0]}")

print("\n" + "="*60)
print("Analysis:")
print("="*60)
print("If normalized outputs are all near 0.5, model is predicting center")
print("If normalized outputs are all near 0 or 1, model is predicting corners")
print("If errors are consistently high, model needs more training or different approach")

