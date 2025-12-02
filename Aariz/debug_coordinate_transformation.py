"""
Debug script to understand coordinate transformation
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from model_heatmap import HRNetP1P2HeatmapDetector
from train_p1_p2_heatmap import generate_heatmap

def debug_single_image():
    """Debug coordinate transformation for a single image"""
    script_dir = Path(__file__).parent
    
    model_path = script_dir / 'models' / 'hrnet_p1p2_heatmap_best.pth'
    annotations_file = script_dir / 'annotations_p1_p2.json'
    # Try different possible paths for images
    possible_paths = [
        script_dir / 'train' / 'Cephalograms',
        script_dir / 'Cephalograms',
        script_dir.parent / 'Aariz' / 'train' / 'Cephalograms',
    ]
    images_dir = None
    for path in possible_paths:
        if path.exists():
            images_dir = path
            break
    if images_dir is None:
        images_dir = possible_paths[0]  # Use first as default for error message
    
    # Load annotations
    samples = load_annotations(str(annotations_file))
    if len(samples) == 0:
        print("No samples found!")
        return
    
    # Get first sample
    sample = samples[0]
    image_path = images_dir / sample['image_filename']
    
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = image.size
    
    print("="*70)
    print("COORDINATE TRANSFORMATION DEBUG")
    print("="*70)
    print(f"Image: {sample['image_filename']}")
    print(f"Original size: {orig_width}x{orig_height}")
    print(f"Aspect ratio: {orig_width/orig_height:.3f}")
    print()
    
    # Ground truth coordinates (normalized [0,1] relative to original)
    gt_p1_norm = sample['p1']
    gt_p2_norm = sample['p2']
    
    # Convert to pixel coordinates
    gt_p1_px = {
        'x': gt_p1_norm['x'] * orig_width,
        'y': gt_p1_norm['y'] * orig_height
    }
    gt_p2_px = {
        'x': gt_p2_norm['x'] * orig_width,
        'y': gt_p2_norm['y'] * orig_height
    }
    
    print("Ground Truth (from annotations):")
    print(f"  P1 normalized: ({gt_p1_norm['x']:.4f}, {gt_p1_norm['y']:.4f})")
    print(f"  P1 pixels: ({gt_p1_px['x']:.1f}, {gt_p1_px['y']:.1f})")
    print(f"  P2 normalized: ({gt_p2_norm['x']:.4f}, {gt_p2_norm['y']:.4f})")
    print(f"  P2 pixels: ({gt_p2_px['x']:.1f}, {gt_p2_px['y']:.1f})")
    print()
    
    # Image size for model
    image_size = 1024
    heatmap_size = 256
    
    # Transform (same as training)
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Generate heatmap (same as training)
    print("Heatmap Generation (same as training):")
    print(f"  Using coordinates [0,1] relative to ORIGINAL image")
    print(f"  Heatmap size: {heatmap_size}x{heatmap_size}")
    
    p1_heatmap = generate_heatmap(
        (gt_p1_norm['x'], gt_p1_norm['y']),
        (heatmap_size, heatmap_size),
        sigma=3.0
    )
    p2_heatmap = generate_heatmap(
        (gt_p2_norm['x'], gt_p2_norm['y']),
        (heatmap_size, heatmap_size),
        sigma=3.0
    )
    
    # Find peak in heatmap
    p1_peak = np.unravel_index(np.argmax(p1_heatmap), p1_heatmap.shape)
    p2_peak = np.unravel_index(np.argmax(p2_heatmap), p2_heatmap.shape)
    
    print(f"  P1 heatmap peak: ({p1_peak[1]}, {p1_peak[0]}) in {heatmap_size}x{heatmap_size} heatmap")
    print(f"  P2 heatmap peak: ({p2_peak[1]}, {p2_peak[0]}) in {heatmap_size}x{heatmap_size} heatmap")
    print(f"  Expected peak x: {int(gt_p1_norm['x'] * heatmap_size)}, got: {p1_peak[1]}")
    print(f"  Expected peak y: {int(gt_p1_norm['y'] * heatmap_size)}, got: {p1_peak[0]}")
    print()
    
    # Load model and predict
    print("Model Prediction:")
    device = torch.device('cpu')
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=False,
        output_size=heatmap_size
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.inference_mode():
        pred_heatmaps = model(image_tensor)
        pred_coords = model.extract_coordinates(pred_heatmaps)
    
    pred_coords_norm = pred_coords.cpu().numpy()[0]
    
    print(f"  Predicted heatmaps shape: {pred_heatmaps.shape}")
    print(f"  Extracted coordinates [0,1]: p1=({pred_coords_norm[0]:.4f}, {pred_coords_norm[1]:.4f}), p2=({pred_coords_norm[2]:.4f}, {pred_coords_norm[3]:.4f})")
    
    # Find peak in predicted heatmap
    pred_p1_heatmap = pred_heatmaps[0, 0].cpu().numpy()
    pred_p2_heatmap = pred_heatmaps[0, 1].cpu().numpy()
    pred_p1_peak = np.unravel_index(np.argmax(pred_p1_heatmap), pred_p1_heatmap.shape)
    pred_p2_peak = np.unravel_index(np.argmax(pred_p2_heatmap), pred_p2_heatmap.shape)
    
    print(f"  Predicted P1 heatmap peak: ({pred_p1_peak[1]}, {pred_p1_peak[0]})")
    print(f"  Predicted P2 heatmap peak: ({pred_p2_peak[1]}, {pred_p2_peak[0]})")
    print()
    
    # Convert to pixel coordinates (current method)
    pred_p1_px_current = {
        'x': pred_coords_norm[0] * orig_width,
        'y': pred_coords_norm[1] * orig_height
    }
    pred_p2_px_current = {
        'x': pred_coords_norm[2] * orig_width,
        'y': pred_coords_norm[3] * orig_height
    }
    
    print("Current Transformation (direct scale to original):")
    print(f"  Pred P1 pixels: ({pred_p1_px_current['x']:.1f}, {pred_p1_px_current['y']:.1f})")
    print(f"  Pred P2 pixels: ({pred_p2_px_current['x']:.1f}, {pred_p2_px_current['y']:.1f})")
    print(f"  Error P1: {np.sqrt((pred_p1_px_current['x'] - gt_p1_px['x'])**2 + (pred_p1_px_current['y'] - gt_p1_px['y'])**2):.2f}px")
    print(f"  Error P2: {np.sqrt((pred_p2_px_current['x'] - gt_p2_px['x'])**2 + (pred_p2_px_current['y'] - gt_p2_px['y'])**2):.2f}px")
    print()
    
    # Try alternative: scale from heatmap peak to original
    pred_p1_px_alt = {
        'x': (pred_p1_peak[1] / heatmap_size) * orig_width,
        'y': (pred_p1_peak[0] / heatmap_size) * orig_height
    }
    pred_p2_px_alt = {
        'x': (pred_p2_peak[1] / heatmap_size) * orig_width,
        'y': (pred_p2_peak[0] / heatmap_size) * orig_height
    }
    
    print("Alternative Transformation (from heatmap peak):")
    print(f"  Pred P1 pixels: ({pred_p1_px_alt['x']:.1f}, {pred_p1_px_alt['y']:.1f})")
    print(f"  Pred P2 pixels: ({pred_p2_px_alt['x']:.1f}, {pred_p2_px_alt['y']:.1f})")
    print(f"  Error P1: {np.sqrt((pred_p1_px_alt['x'] - gt_p1_px['x'])**2 + (pred_p1_px_alt['y'] - gt_p1_px['y'])**2):.2f}px")
    print(f"  Error P2: {np.sqrt((pred_p2_px_alt['x'] - gt_p2_px['x'])**2 + (pred_p2_px_alt['y'] - gt_p2_px['y'])**2):.2f}px")

def load_annotations(annotations_file):
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            if 'annotations' in item and item['annotations']:
                annotation = item['annotations'][0]
                
                if 'result' in annotation:
                    p1 = None
                    p2 = None
                    
                    for result_item in annotation['result']:
                        if result_item.get('type') == 'keypointlabels':
                            value = result_item.get('value', {})
                            labels = value.get('keypointlabels', [])
                            
                            if 'p1' in labels:
                                p1 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                            elif 'p2' in labels:
                                p2 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                    
                    if p1 and p2:
                        image_url = item['data'].get('image', '')
                        image_filename = image_url.split('/')[-1]
                        
                        if '-' in image_filename:
                            parts = image_filename.split('-')
                            if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
                                image_filename = '-'.join(parts[1:])
                        
                        samples.append({
                            'image_filename': image_filename,
                            'p1': p1,
                            'p2': p2
                        })
        
        return samples

if __name__ == '__main__':
    debug_single_image()

