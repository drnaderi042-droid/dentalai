#!/usr/bin/env python3
"""
Simple test to verify the scaling fix for P1/P2 model
Directly tests the inference script with correct vs incorrect parameters
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys
import argparse

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector

def load_image_and_annotate(image_path, annotations_file):
    """Load image and its ground truth annotation"""
    image_id = image_path.stem
    
    # Load annotation
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Find annotation for this image
    for item in annotations:
        if item['id'] == image_id:
            # Extract P1 and P2 coordinates
            for result in item['annotations'][0]['result']:
                if result['type'] == 'keypointlabels':
                    keypoints = result['value']['keypoints']
                    x_coords = result['value']['x']
                    y_coords = result['value']['y']
                    
                    p1_x = p1_y = p2_x = p2_y = None
                    for i, label in enumerate(keypoints):
                        if label == 'P1':
                            p1_x = x_coords[i] / 100.0  # Convert from percentage
                            p1_y = y_coords[i] / 100.0
                        elif label == 'P2':
                            p2_x = x_coords[i] / 100.0
                            p2_y = y_coords[i] / 100.0
                    
                    if p1_x is not None and p2_x is not None:
                        return {
                            'p1': {'x': p1_x, 'y': p1_y},
                            'p2': {'x': p2_x, 'y': p2_y}
                        }
    return None

def preprocess_image(image, image_size):
    """Preprocess image for model"""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)

def extract_coordinates(heatmaps):
    """Extract coordinates from heatmaps using soft-argmax"""
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

def test_scaling_parameters():
    """Test the model with correct vs incorrect parameters"""
    # Configuration
    model_path = 'models/hrnet_p1p2_heatmap_best.pth'
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'train/Cephalograms'
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load annotations to get test images
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Get first 5 images that exist
    test_images = []
    images_path = Path(images_dir)
    for item in annotations[:10]:  # Check first 10
        image_id = item['id']
        image_path = images_path / f"{image_id}.jpg"
        if image_path.exists():
            test_images.append((image_id, image_path))
            if len(test_images) >= 5:
                break
    
    print(f"Found {len(test_images)} test images")
    
    # Test parameters
    test_configs = [
        {'name': 'INCORRECT (Current)', 'image_size': 768, 'heatmap_size': 192},
        {'name': 'CORRECT (Training)', 'image_size': 1024, 'heatmap_size': 256}
    ]
    
    all_results = {}
    
    for config in test_configs:
        print(f"\n{'='*60}")
        print(f"Testing with {config['name']} parameters")
        print(f"Image size: {config['image_size']}, Heatmap size: {config['heatmap_size']}")
        print('='*60)
        
        # Load model with this configuration
        checkpoint = torch.load(model_path, map_location=device)
        
        model = HRNetP1P2HeatmapDetector(
            num_landmarks=2,
            hrnet_variant='hrnet_w18',
            pretrained=True,
            output_size=config['heatmap_size']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        results = []
        
        for image_id, image_path in test_images:
            # Load image
            image = Image.open(image_path).convert('RGB')
            image_tensor = preprocess_image(image, config['image_size'])
            
            # Run inference
            with torch.no_grad():
                image_tensor = image_tensor.to(device)
                heatmaps = model(image_tensor)  # (1, 2, H, W)
                coords = extract_coordinates(heatmaps)  # (1, 4) normalized [0, 1]
                
                # Convert to pixel coordinates
                coords_px = coords.cpu().numpy()[0] * config['image_size']
            
            # Load ground truth
            gt_coords = load_image_and_annotate(image_path, annotations_file)
            if gt_coords is None:
                print(f"Warning: No ground truth for {image_id}")
                continue
            
            # Calculate errors
            errors = calculate_pixel_error(coords_px, gt_coords, config['image_size'])
            errors['image_id'] = image_id
            results.append(errors)
            
            print(f"Image {image_id}: P1={errors['p1_error']:.1f}px, P2={errors['p2_error']:.1f}px, MRE={errors['mre']:.1f}px")
        
        # Calculate average
        if results:
            avg_mre = np.mean([r['mre'] for r in results])
            avg_p1_error = np.mean([r['p1_error'] for r in results])
            avg_p2_error = np.mean([r['p2_error'] for r in results])
            
            print(f"\nAverage Errors:")
            print(f"  P1: {avg_p1_error:.1f}px")
            print(f"  P2: {avg_p2_error:.1f}px")
            print(f"  MRE: {avg_mre:.1f}px")
            
            all_results[config['name']] = {
                'image_size': config['image_size'],
                'heatmap_size': config['heatmap_size'],
                'average_mre': avg_mre,
                'average_p1_error': avg_p1_error,
                'average_p2_error': avg_p2_error,
                'individual_results': results
            }
        else:
            print("No valid results")
    
    # Final comparison
    if 'INCORRECT (Current)' in all_results and 'CORRECT (Training)' in all_results:
        wrong_mre = all_results['INCORRECT (Current)']['average_mre']
        correct_mre = all_results['CORRECT (Training)']['average_mre']
        
        print(f"\n{'='*60}")
        print("SCALING FIX VERIFICATION RESULTS")
        print('='*60)
        print(f"INCORRECT parameters (768, 192): {wrong_mre:.1f}px MRE")
        print(f"CORRECT parameters (1024, 256): {correct_mre:.1f}px MRE")
        print(f"Error reduction: {wrong_mre - correct_mre:.1f}px")
        
        if wrong_mre - correct_mre > 50:
            print("✅ SIGNIFICANT IMPROVEMENT: Scaling fix confirmed!")
        elif wrong_mre - correct_mre > 10:
            print("✅ IMPROVEMENT: Some improvement achieved")
        else:
            print("⚠️  MINIMAL IMPROVEMENT: Additional issues may exist")
        
        if correct_mre < 10.0:
            print("✅ SUCCESS: Correct parameters achieve <10px threshold!")
        else:
            print("❌ STILL HIGH: Additional work needed")
        
        if correct_mre < 2.0:
            print("🎯 EXCELLENT: Target <2px error achieved!")
        elif correct_mre < 5.0:
            print("✅ GOOD: Close to target <2px error")
        else:
            print("⚠️  ACCEPTABLE: <10px threshold met, but target <2px not reached")
        
        # Save results
        with open('scaling_fix_verification_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to: scaling_fix_verification_results.json")
    
    return all_results

if __name__ == "__main__":
    results = test_scaling_parameters()