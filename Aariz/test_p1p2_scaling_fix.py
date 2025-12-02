#!/usr/bin/env python3
"""
Test script to verify the scaling fix for P1/P2 model
Tests with CORRECT parameters (1024, 256) vs INCORRECT parameters (768, 192)
"""

import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import local modules
sys.path.append(str(Path(__file__).parent))
from model_heatmap import HRNetP1P2HeatmapDetector
from dataset_p1_p2 import P1P2Dataset
from train_p1_p2_heatmap import P1P2HeatmapDataset

def load_model_with_params(checkpoint_path, image_size, heatmap_size, device='cuda'):
    """Load model with specified parameters"""
    print(f"Loading model with image_size={image_size}, heatmap_size={heatmap_size}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=True,
        output_size=heatmap_size
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

def preprocess_image(image, image_size):
    """Preprocess image for model"""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def extract_coords_from_heatmap(heatmaps):
    """Extract coordinates from heatmaps"""
    batch_size, num_landmarks, H, W = heatmaps.shape
    
    # Create coordinate grids
    y_coords = torch.arange(H, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, H, 1)
    x_coords = torch.arange(W, dtype=heatmaps.dtype, device=heatmaps.device).view(1, 1, 1, W)
    
    # Normalize to [0, 1]
    y_coords = y_coords / (H - 1) if H > 1 else y_coords
    x_coords = x_coords / (W - 1) if W > 1 else x_coords
    
    # Weighted average
    heatmaps_sum = heatmaps.sum(dim=(2, 3), keepdim=True) + 1e-8
    
    x_mean = (heatmaps * x_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    y_mean = (heatmaps * y_coords).sum(dim=(2, 3)) / heatmaps_sum.squeeze()
    
    # Flatten: [batch, num_landmarks, 2] -> [batch, num_landmarks * 2]
    coords = torch.stack([x_mean, y_mean], dim=-1)  # [batch, num_landmarks, 2]
    coords = coords.view(batch_size, num_landmarks * 2)
    
    return coords

def load_annotation_from_json(image_id, annotations_file):
    """Load annotation from JSON file"""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Find the annotation for this image_id
    for item in annotations:
        if item['id'] == image_id:
            # Extract P1 and P2 coordinates from annotation results
            for result in item['annotations'][0]['result']:
                if result['type'] == 'keypointlabels':
                    keypoints = result['value']['keypoints']
                    # Find P1 and P2
                    p1_x, p1_y = None, None
                    p2_x, p2_y = None, None
                    
                    for i, label in enumerate(keypoints):
                        if label == 'P1':
                            p1_x = result['value']['x'][i] / 100.0  # Convert from percentage
                            p1_y = result['value']['y'][i] / 100.0
                        elif label == 'P2':
                            p2_x = result['value']['x'][i] / 100.0
                            p2_y = result['value']['y'][i] / 100.0
                    
                    if p1_x is not None and p2_x is not None:
                        return {
                            'p1': {'x': p1_x, 'y': p1_y},
                            'p2': {'x': p2_x, 'y': p2_y}
                        }
    return None

def calculate_pixel_error(pred_coords, gt_coords, image_size):
    """Calculate pixel error between prediction and ground truth"""
    pred_p1 = pred_coords[0:2]  # p1_x, p1_y
    pred_p2 = pred_coords[2:4]  # p2_x, p2_y
    gt_p1 = [gt_coords['p1']['x'], gt_coords['p1']['y']]
    gt_p2 = [gt_coords['p2']['x'], gt_coords['p2']['y']]
    
    # Convert normalized to pixel coordinates
    pred_p1_px = [pred_p1[0] * image_size, pred_p1[1] * image_size]
    pred_p2_px = [pred_p2[0] * image_size, pred_p2[1] * image_size]
    gt_p1_px = [gt_p1[0] * image_size, gt_p1[1] * image_size]
    gt_p2_px = [gt_p2[0] * image_size, gt_p2[1] * image_size]
    
    # Calculate Euclidean distance errors
    error_p1 = np.sqrt((pred_p1_px[0] - gt_p1_px[0])**2 + (pred_p1_px[1] - gt_p1_px[1])**2)
    error_p2 = np.sqrt((pred_p2_px[0] - gt_p2_px[0])**2 + (pred_p2_px[1] - gt_p2_px[1])**2)
    
    # Mean Radial Error (MRE)
    mre = (error_p1 + error_p2) / 2.0
    
    return {
        'p1_error': error_p1,
        'p2_error': error_p2,
        'mre': mre,
        'pred_p1_px': pred_p1_px,
        'pred_p2_px': pred_p2_px,
        'gt_p1_px': gt_p1_px,
        'gt_p2_px': gt_p2_px
    }

def test_scaling_fix():
    """Main test function"""
    # Configuration
    model_path = 'models/hrnet_p1_p2_heatmap_best.pth'
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'train/Cephalograms'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Test with INCORRECT parameters (current implementation)
    print("\n" + "="*60)
    print("TESTING WITH INCORRECT PARAMETERS (768, 192)")
    print("="*60)
    
    model_wrong = load_model_with_params(model_path, 768, 192, device)
    
    # Load a few test images
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    test_images = annotations[:5]  # Test first 5 images
    
    results_wrong = []
    
    for item in test_images:
        image_id = item['id']
        image_path = Path(images_dir) / f"{image_id}.jpg"
        
        if not image_path.exists():
            continue
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess_image(image, 768)
        
        # Run inference
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            heatmaps = model_wrong(image_tensor)  # (1, 2, H, W)
            coords = extract_coords_from_heatmap(heatmaps)  # (1, 4) normalized [0, 1]
            
            # Convert to pixel coordinates
            coords_px = coords.cpu().numpy()[0] * 768  # Scale to 768px
            
        # Load ground truth
        gt_coords = load_annotation_from_json(image_id, annotations_file)
        if gt_coords is None:
            continue
            
        # Calculate errors
        errors = calculate_pixel_error(coords_px, gt_coords, 768)
        errors['image_id'] = image_id
        results_wrong.append(errors)
        
        print(f"Image {image_id}: P1 Error = {errors['p1_error']:.1f}px, P2 Error = {errors['p2_error']:.1f}px, MRE = {errors['mre']:.1f}px")
    
    # Calculate average errors for wrong parameters
    avg_mre_wrong = np.mean([r['mre'] for r in results_wrong])
    print(f"\nINCORRECT PARAMETERS AVERAGE MRE: {avg_mre_wrong:.1f}px")
    
    # Test with CORRECT parameters (training configuration)
    print("\n" + "="*60)
    print("TESTING WITH CORRECT PARAMETERS (1024, 256)")
    print("="*60)
    
    model_correct = load_model_with_params(model_path, 1024, 256, device)
    
    results_correct = []
    
    for item in test_images:
        image_id = item['id']
        image_path = Path(images_dir) / f"{image_id}.jpg"
        
        if not image_path.exists():
            continue
            
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = preprocess_image(image, 1024)
        
        # Run inference
        with torch.no_grad():
            image_tensor = image_tensor.to(device)
            heatmaps = model_correct(image_tensor)  # (1, 2, H, W)
            coords = extract_coords_from_heatmap(heatmaps)  # (1, 4) normalized [0, 1]
            
            # Convert to pixel coordinates
            coords_px = coords.cpu().numpy()[0] * 1024  # Scale to 1024px
            
        # Load ground truth
        gt_coords = load_annotation_from_json(image_id, annotations_file)
        if gt_coords is None:
            continue
            
        # Calculate errors (scale ground truth to match 1024px)
        errors = calculate_pixel_error(coords_px, gt_coords, 1024)
        errors['image_id'] = image_id
        results_correct.append(errors)
        
        print(f"Image {image_id}: P1 Error = {errors['p1_error']:.1f}px, P2 Error = {errors['p2_error']:.1f}px, MRE = {errors['mre']:.1f}px")
    
    # Calculate average errors for correct parameters
    avg_mre_correct = np.mean([r['mre'] for r in results_correct])
    print(f"\nCORRECT PARAMETERS AVERAGE MRE: {avg_mre_correct:.1f}px")
    
    # Summary
    print("\n" + "="*60)
    print("SCALING FIX VERIFICATION RESULTS")
    print("="*60)
    print(f"INCORRECT parameters (768, 192): {avg_mre_wrong:.1f}px MRE")
    print(f"CORRECT parameters (1024, 256): {avg_mre_correct:.1f}px MRE")
    print(f"Error reduction: {avg_mre_wrong - avg_mre_correct:.1f}px ({((avg_mre_wrong - avg_mre_correct) / avg_mre_wrong * 100):.1f}%)")
    
    if avg_mre_correct < 10.0:
        print("✅ SUCCESS: Correct parameters achieve <10px error threshold!")
    else:
        print("❌ STILL HIGH: Additional issues may exist")
    
    if avg_mre_correct < 2.0:
        print("✅ EXCELLENT: Target <2px error achieved!")
    else:
        print("⚠️  ACCEPTABLE: <10px threshold met, but target <2px not reached")
    
    # Save detailed results
    results_summary = {
        'incorrect_parameters': {
            'image_size': 768,
            'heatmap_size': 192,
            'average_mre': avg_mre_wrong,
            'individual_results': results_wrong
        },
        'correct_parameters': {
            'image_size': 1024,
            'heatmap_size': 256,
            'average_mre': avg_mre_correct,
            'individual_results': results_correct
        },
        'improvement': {
            'error_reduction_px': avg_mre_wrong - avg_mre_correct,
            'error_reduction_percent': ((avg_mre_wrong - avg_mre_correct) / avg_mre_wrong * 100),
            'meets_10px_threshold': avg_mre_correct < 10.0,
            'meets_2px_target': avg_mre_correct < 2.0
        }
    }
    
    with open('p1p2_scaling_fix_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nDetailed results saved to: p1p2_scaling_fix_results.json")
    
    return results_summary

if __name__ == "__main__":
    import sys
    results = test_scaling_fix()