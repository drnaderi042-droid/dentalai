"""
Fix coordinate transformation for P1/P2 heatmap model
The issue: coordinates are [0,1] relative to original image, but image is resized to 1024x1024
We need to transform coordinates from original space to resized space before generating heatmap
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


def test_with_correct_transformation():
    """Test with correct coordinate transformation"""
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
        images_dir = possible_paths[0]
    
    # Load annotations
    samples = load_annotations(str(annotations_file))
    print(f"Loaded {len(samples)} samples")
    print("="*70)
    
    # Load model
    device = torch.device('cpu')
    model = HRNetP1P2HeatmapDetector(
        num_landmarks=2,
        hrnet_variant='hrnet_w18',
        pretrained=False,
        output_size=256
    )
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    image_size = 1024
    heatmap_size = 256
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    errors = []
    
    for i, sample in enumerate(samples[:10]):
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
            
            # Ground truth (normalized [0,1] relative to original)
            gt_p1_norm = sample['p1']
            gt_p2_norm = sample['p2']
            
            gt_p1_px = {
                'x': gt_p1_norm['x'] * orig_width,
                'y': gt_p1_norm['y'] * orig_height
            }
            gt_p2_px = {
                'x': gt_p2_norm['x'] * orig_width,
                'y': gt_p2_norm['y'] * orig_height
            }
            
            # Preprocess
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.inference_mode():
                pred_heatmaps = model(image_tensor)
                pred_coords = model.extract_coordinates(pred_heatmaps)
            
            pred_coords_norm = pred_coords.cpu().numpy()[0]
            
            # CRITICAL FIX: The model was trained incorrectly!
            # In training: coordinates [0,1] relative to original image
            # But image is resized to 1024x1024 (stretch)
            # Heatmap is generated from coordinates [0,1] relative to original
            # But model sees resized image (1024x1024)
            # 
            # The model learns a mapping where extracted coordinates from heatmap
            # correspond to coordinates in the RESIZED image space (1024x1024),
            # NOT the original image space!
            #
            # So we need to:
            # 1. Extract coordinates are [0,1] relative to resized image (1024x1024)
            # 2. Scale to resized image: pred_px_resized = pred_coords * 1024
            # 3. Then scale from resized to original
            
            # Method 1: Assume extract_coords are [0,1] relative to resized image
            pred_p1_px_resized = {
                'x': pred_coords_norm[0] * image_size,
                'y': pred_coords_norm[1] * image_size
            }
            pred_p2_px_resized = {
                'x': pred_coords_norm[2] * image_size,
                'y': pred_coords_norm[3] * image_size
            }
            
            # Scale from resized to original
            scale_x = orig_width / image_size
            scale_y = orig_height / image_size
            
            pred_p1_px = {
                'x': pred_p1_px_resized['x'] * scale_x,
                'y': pred_p1_px_resized['y'] * scale_y
            }
            pred_p2_px = {
                'x': pred_p2_px_resized['x'] * scale_x,
                'y': pred_p2_px_resized['y'] * scale_y
            }
            
            # Calculate errors
            p1_error = np.sqrt((pred_p1_px['x'] - gt_p1_px['x'])**2 + (pred_p1_px['y'] - gt_p1_px['y'])**2)
            p2_error = np.sqrt((pred_p2_px['x'] - gt_p2_px['x'])**2 + (pred_p2_px['y'] - gt_p2_px['y'])**2)
            avg_error = (p1_error + p2_error) / 2
            
            errors.append(avg_error)
            
            status = "OK" if avg_error < 10 else "HIGH"
            print(f"Image {i+1}: {sample['image_filename']} - Error: {avg_error:.2f}px [{status}]")
            print(f"  GT P1: ({gt_p1_px['x']:.1f}, {gt_p1_px['y']:.1f}), Pred: ({pred_p1_px['x']:.1f}, {pred_p1_px['y']:.1f})")
            print(f"  GT P2: ({gt_p2_px['x']:.1f}, {gt_p2_px['y']:.1f}), Pred: ({pred_p2_px['x']:.1f}, {pred_p2_px['y']:.1f})")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if errors:
        mean_error = np.mean(errors)
        print(f"\nMean Error: {mean_error:.2f}px")
        if mean_error < 10:
            print("SUCCESS! Error is acceptable")
        else:
            print("FAILURE! Error is too high")

if __name__ == '__main__':
    test_with_correct_transformation()

