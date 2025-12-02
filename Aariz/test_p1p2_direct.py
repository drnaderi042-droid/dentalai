"""
Direct test of P1/P2 model to understand coordinate transformation
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


def load_model(model_path, image_size=1024, heatmap_size=256):
    """Load the trained model"""
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
    
    return model, device


def load_annotations(annotations_file):
    """Load annotations from JSON file"""
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


def test_model(model, device, annotations_file, images_dir, image_size=1024, heatmap_size=256, num_images=5):
    """Test model directly"""
    images_dir = Path(images_dir)
    samples = load_annotations(annotations_file)
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    errors_method1 = []  # Direct scale: coords * orig_size
    errors_method2 = []  # Two-step: coords * 1024, then scale to original
    
    for i, sample in enumerate(samples[:num_images]):
        image_path = images_dir / sample['image_filename']
        
        if not image_path.exists():
            alt_filename = sample['image_filename']
            if '-' in alt_filename:
                parts = alt_filename.split('-')
                if len(parts[0]) == 8:
                    alt_filename = '-'.join(parts[1:])
            alt_path = images_dir / alt_filename
            if alt_path.exists():
                image_path = alt_path
            else:
                continue
        
        try:
            image = Image.open(image_path).convert('RGB')
            orig_width, orig_height = image.size
            
            # Ground truth
            gt_p1_px = {
                'x': sample['p1']['x'] * orig_width,
                'y': sample['p1']['y'] * orig_height
            }
            gt_p2_px = {
                'x': sample['p2']['x'] * orig_width,
                'y': sample['p2']['y'] * orig_height
            }
            
            # Preprocess
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.inference_mode():
                heatmaps = model(image_tensor)
                coords = model.extract_coordinates(heatmaps)
            
            # Debug: Check heatmaps
            heatmaps_np = heatmaps.cpu().numpy()[0]
            print(f"  Heatmap shapes: {heatmaps_np.shape}")
            print(f"  Heatmap 1 (P1) - min: {heatmaps_np[0].min():.4f}, max: {heatmaps_np[0].max():.4f}, sum: {heatmaps_np[0].sum():.4f}, mean: {heatmaps_np[0].mean():.4f}")
            print(f"  Heatmap 2 (P2) - min: {heatmaps_np[1].min():.4f}, max: {heatmaps_np[1].max():.4f}, sum: {heatmaps_np[1].sum():.4f}, mean: {heatmaps_np[1].mean():.4f}")
            
            # Find peak manually - check all locations with max value
            p1_max_val = heatmaps_np[0].max()
            p2_max_val = heatmaps_np[1].max()
            
            if p1_max_val > 0.01:  # Threshold to avoid noise
                p1_peak_indices = np.where(heatmaps_np[0] == p1_max_val)
                if len(p1_peak_indices[0]) > 0:
                    p1_peak = (p1_peak_indices[0][0], p1_peak_indices[1][0])
                    p1_peak_val = p1_max_val
                else:
                    p1_peak = (0, 0)
                    p1_peak_val = 0.0
            else:
                p1_peak = (0, 0)
                p1_peak_val = 0.0
                
            if p2_max_val > 0.01:
                p2_peak_indices = np.where(heatmaps_np[1] == p2_max_val)
                if len(p2_peak_indices[0]) > 0:
                    p2_peak = (p2_peak_indices[0][0], p2_peak_indices[1][0])
                    p2_peak_val = p2_max_val
                else:
                    p2_peak = (0, 0)
                    p2_peak_val = 0.0
            else:
                p2_peak = (0, 0)
                p2_peak_val = 0.0
                
            print(f"  P1 peak at: {p1_peak}, value: {p1_peak_val:.4f}")
            print(f"  P2 peak at: {p2_peak}, value: {p2_peak_val:.4f}")
            
            # Check if heatmap is uniform (all same value)
            p1_unique = len(np.unique(heatmaps_np[0]))
            p2_unique = len(np.unique(heatmaps_np[1]))
            print(f"  P1 unique values: {p1_unique}, P2 unique values: {p2_unique}")
            
            # Check if P2 heatmap is uniform (all 1.0)
            if p2_unique == 1 and p2_max_val == 1.0:
                print(f"  WARNING: P2 heatmap is uniform (all 1.0)! Model may not be working correctly.")
            
            coords_normalized = coords.cpu().numpy()[0]
            print(f"  Extracted coords (before scaling): {coords_normalized}")
            
            # Method 1: Direct scale (current endpoint method)
            pred1_p1 = {
                'x': coords_normalized[0] * orig_width,
                'y': coords_normalized[1] * orig_height
            }
            pred1_p2 = {
                'x': coords_normalized[2] * orig_width,
                'y': coords_normalized[3] * orig_height
            }
            
            # Method 2: Two-step (as in validation)
            coords_resized_px = coords_normalized * image_size
            scale_x = orig_width / image_size
            scale_y = orig_height / image_size
            pred2_p1 = {
                'x': coords_resized_px[0] * scale_x,
                'y': coords_resized_px[1] * scale_y
            }
            pred2_p2 = {
                'x': coords_resized_px[2] * scale_x,
                'y': coords_resized_px[3] * scale_y
            }
            
            # Calculate errors
            err1_p1 = np.sqrt((pred1_p1['x'] - gt_p1_px['x'])**2 + (pred1_p1['y'] - gt_p1_px['y'])**2)
            err1_p2 = np.sqrt((pred1_p2['x'] - gt_p2_px['x'])**2 + (pred1_p2['y'] - gt_p2_px['y'])**2)
            err1_avg = (err1_p1 + err1_p2) / 2
            
            err2_p1 = np.sqrt((pred2_p1['x'] - gt_p1_px['x'])**2 + (pred2_p1['y'] - gt_p1_px['y'])**2)
            err2_p2 = np.sqrt((pred2_p2['x'] - gt_p2_px['x'])**2 + (pred2_p2['y'] - gt_p2_px['y'])**2)
            err2_avg = (err2_p1 + err2_p2) / 2
            
            errors_method1.append(err1_avg)
            errors_method2.append(err2_avg)
            
            print(f"\nImage {i+1}: {sample['image_filename']} ({orig_width}x{orig_height})")
            print(f"  GT: P1=({gt_p1_px['x']:.1f}, {gt_p1_px['y']:.1f}), P2=({gt_p2_px['x']:.1f}, {gt_p2_px['y']:.1f})")
            print(f"  Method 1 (direct): P1=({pred1_p1['x']:.1f}, {pred1_p1['y']:.1f}), P2=({pred1_p2['x']:.1f}, {pred1_p2['y']:.1f}) - Error: {err1_avg:.2f}px")
            print(f"  Method 2 (two-step): P1=({pred2_p1['x']:.1f}, {pred2_p1['y']:.1f}), P2=({pred2_p2['x']:.1f}, {pred2_p2['y']:.1f}) - Error: {err2_avg:.2f}px")
            print(f"  Coords normalized: p1=({coords_normalized[0]:.4f}, {coords_normalized[1]:.4f}), p2=({coords_normalized[2]:.4f}, {coords_normalized[3]:.4f})")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if errors_method1 and errors_method2:
        print("\n" + "="*70)
        print("COMPARISON RESULTS")
        print("="*70)
        print(f"Method 1 (direct scale): Mean={np.mean(errors_method1):.2f}px, Std={np.std(errors_method1):.2f}px")
        print(f"Method 2 (two-step): Mean={np.mean(errors_method2):.2f}px, Std={np.std(errors_method2):.2f}px")
        
        if np.mean(errors_method1) < np.mean(errors_method2):
            print("✅ Method 1 (direct scale) is better!")
        else:
            print("✅ Method 2 (two-step) is better!")


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    
    # Try different paths
    possible_model_paths = [
        script_dir / 'models' / 'hrnet_p1p2_heatmap_best.pth',
        script_dir.parent / 'Aariz' / 'models' / 'hrnet_p1p2_heatmap_best.pth',
    ]
    model_path = None
    for path in possible_model_paths:
        if path.exists():
            model_path = path
            break
    
    possible_annotations = [
        script_dir / 'annotations_p1_p2.json',
        script_dir.parent / 'Aariz' / 'annotations_p1_p2.json',
    ]
    annotations_file = None
    for path in possible_annotations:
        if path.exists():
            annotations_file = path
            break
    
    possible_image_dirs = [
        script_dir / 'train' / 'Cephalograms',
        script_dir / 'Cephalograms',
        script_dir.parent / 'Aariz' / 'train' / 'Cephalograms',
    ]
    images_dir_found = None
    for path in possible_image_dirs:
        if path.exists():
            images_dir_found = path
            break
    
    if not model_path:
        print(f"ERROR: Model not found. Tried: {possible_model_paths}")
        sys.exit(1)
    if not annotations_file:
        print(f"ERROR: Annotations not found. Tried: {possible_annotations}")
        sys.exit(1)
    if not images_dir_found:
        print(f"ERROR: Images directory not found. Tried: {possible_image_dirs}")
        sys.exit(1)
    
    print(f"Using model: {model_path}")
    print(f"Using annotations: {annotations_file}")
    print(f"Using images: {images_dir_found}")
    
    print("Loading model...")
    model, device = load_model(str(model_path), image_size=1024, heatmap_size=256)
    print("Model loaded!")
    
    test_model(model, device, str(annotations_file), str(images_dir_found), num_images=5)
