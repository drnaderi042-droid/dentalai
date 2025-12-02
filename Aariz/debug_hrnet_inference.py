"""
Debug script to compare HRNet inference with training evaluation
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
import json

# Add paths
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cephx_path = os.path.join(base_dir, 'cephx_service')
aariz_path = os.path.join(base_dir, 'Aariz')

if cephx_path not in sys.path:
    sys.path.insert(0, cephx_path)

venv_site_packages = os.path.join(cephx_path, 'venv', 'Lib', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

if aariz_path not in sys.path:
    sys.path.insert(0, aariz_path)

# Test image
TEST_IMAGE_ID = "cks2ip8fq29yq0yufc4scftj8"
TEST_IMAGE_PATH = os.path.join(base_dir, "Aariz", "Aariz", "train", "Cephalograms", f"{TEST_IMAGE_ID}.png")
MODEL_PATH = os.path.join(cephx_path, "model", "hrnet_cephalometric.pth")

print("="*80)
print("🔍 Debug: HRNet Inference Comparison")
print("="*80)

# Load image
img = Image.open(TEST_IMAGE_PATH).convert('RGB')
print(f"\n📸 Image: {TEST_IMAGE_ID}")
print(f"   Original size: {img.size} (width × height)")

# Method 1: Using HRNetProductionService
print(f"\n{'='*80}")
print("Method 1: HRNetProductionService")
print(f"{'='*80}")

from hrnet_production_service import HRNetProductionService
import torch

device = 'cpu'  # Use CPU for comparison
service = HRNetProductionService(MODEL_PATH, device=device)

# Preprocess manually to see what's happening
img_tensor, original_size = service.preprocess_image(TEST_IMAGE_PATH)
print(f"\n📊 Preprocessing:")
print(f"   Original size: {original_size}")
print(f"   Input tensor shape: {img_tensor.shape}")
print(f"   Input tensor min: {img_tensor.min():.4f}, max: {img_tensor.max():.4f}")
print(f"   Input tensor mean: {img_tensor.mean():.4f}, std: {img_tensor.std():.4f}")

# Run inference
with torch.no_grad():
    outputs = service.model(img_tensor)
    print(f"\n📊 Model Output:")
    print(f"   Output shape: {outputs.shape}")
    print(f"   Output min: {outputs.min():.4f}, max: {outputs.max():.4f}")
    print(f"   Output mean: {outputs.mean():.4f}, std: {outputs.std():.4f}")
    
    # Apply sigmoid
    heatmaps = torch.sigmoid(outputs)
    print(f"\n📊 After Sigmoid:")
    print(f"   Heatmap shape: {heatmaps.shape}")
    print(f"   Heatmap min: {heatmaps.min():.4f}, max: {heatmaps.max():.4f}")
    print(f"   Heatmap mean: {heatmaps.mean():.4f}, std: {heatmaps.std():.4f}")
    
    # Check first few heatmaps
    heatmaps_np = heatmaps[0].cpu().numpy()
    print(f"\n📊 Sample Heatmap Stats (first 3 landmarks):")
    for i in range(min(3, heatmaps_np.shape[0])):
        hm = heatmaps_np[i]
        max_val = hm.max()
        max_pos = np.unravel_index(hm.argmax(), hm.shape)
        print(f"   Landmark {i} ({service.LANDMARKS[i]}):")
        print(f"      Max value: {max_val:.4f} at position {max_pos}")
        print(f"      Mean: {hm.mean():.4f}, Std: {hm.std():.4f}")

# Get predictions
result = service.detect(TEST_IMAGE_PATH)
print(f"\n📊 Predictions:")
for i, (name, coords) in enumerate(list(result['landmarks'].items())[:5]):
    print(f"   {name}: ({coords['x']:.2f}, {coords['y']:.2f}), conf={coords['confidence']:.4f}")

# Method 2: Using Aariz inference code (if available)
print(f"\n{'='*80}")
print("Method 2: Compare with Dataset preprocessing")
print(f"{'='*80}")

# Check what dataset preprocessing does
from dataset import AarizDataset

dataset = AarizDataset(
    dataset_folder_path=os.path.join(base_dir, "Aariz", "Aariz"),
    mode="TRAIN",
    annotation_type="Senior Orthodontists",
    image_size=(768, 768),  # Match HRNet input size
    use_heatmap=True,
    heatmap_sigma=3.0,
    augmentation=False
)

# Find image index
try:
    img_idx = dataset.image_ids.index(TEST_IMAGE_ID)
    sample = dataset[img_idx]
    
    print(f"\n📊 Dataset Preprocessing:")
    print(f"   Image tensor shape: {sample['image'].shape}")
    print(f"   Image tensor min: {sample['image'].min():.4f}, max: {sample['image'].max():.4f}")
    print(f"   Image tensor mean: {sample['image'].mean():.4f}, std: {sample['image'].std():.4f}")
    print(f"   Original size: {sample['orig_size']}")
    print(f"   Target size: {dataset.image_size}")
    
    # Compare normalization
    print(f"\n📊 Normalization Comparison:")
    print(f"   Service normalization: mean={service.mean}, std={service.std}")
    print(f"   Dataset normalization: mean=0.5, std=0.5")
    print(f"   Service input range: [{img_tensor.min():.3f}, {img_tensor.max():.3f}]")
    print(f"   Dataset input range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
    
except ValueError:
    print(f"   Image {TEST_IMAGE_ID} not found in dataset")

print(f"\n{'='*80}")
print("🔍 Debug Complete")
print(f"{'='*80}")

