"""
Test script for P1/P2 Heatmap Model
Tests the model on multiple images and calculates error metrics
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import sys

# Add parent to path
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
                                'x': value.get('x', 0) / 100.0,  # Convert from percentage to [0,1]
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


def test_model_on_images(model, device, annotations_file, images_dir, image_size=1024, heatmap_size=256):
    """Test model on images from annotations"""
    images_dir = Path(images_dir)
    
    # Load annotations
    samples = load_annotations(annotations_file)
    print(f"Loaded {len(samples)} samples from annotations")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    errors = []
    successful_tests = 0
    failed_tests = 0
    
    for i, sample in enumerate(samples[:20]):  # Test on first 20 images
        image_path = images_dir / sample['image_filename']
        
        if not image_path.exists():
            # Try alternative filename
            alt_filename = sample['image_filename']
            if '-' in alt_filename:
                parts = alt_filename.split('-')
                if len(parts[0]) == 8:
                    alt_filename = '-'.join(parts[1:])
            alt_path = images_dir / alt_filename
            if alt_path.exists():
                image_path = alt_path
            else:
                print(f"WARNING: Image not found: {sample['image_filename']}")
                failed_tests += 1
                continue
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            orig_width, orig_height = image.size
            
            # Ground truth coordinates (normalized [0,1])
            gt_p1 = sample['p1']
            gt_p2 = sample['p2']
            
            # Convert to pixel coordinates
            gt_p1_px = {
                'x': gt_p1['x'] * orig_width,
                'y': gt_p1['y'] * orig_height
            }
            gt_p2_px = {
                'x': gt_p2['x'] * orig_width,
                'y': gt_p2['y'] * orig_height
            }
            
            # Preprocess
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            # Predict
            with torch.inference_mode():
                heatmaps = model(image_tensor)
                coords = model.extract_coordinates(heatmaps)
            
            # Convert to numpy
            coords_normalized = coords.cpu().numpy()[0]  # [p1_x, p1_y, p2_x, p2_y] normalized [0,1]
            
            # Convert to pixel coordinates
            pred_p1_px = {
                'x': coords_normalized[0] * orig_width,
                'y': coords_normalized[1] * orig_height
            }
            pred_p2_px = {
                'x': coords_normalized[2] * orig_width,
                'y': coords_normalized[3] * orig_height
            }
            
            # Calculate errors
            p1_error = np.sqrt((pred_p1_px['x'] - gt_p1_px['x'])**2 + (pred_p1_px['y'] - gt_p1_px['y'])**2)
            p2_error = np.sqrt((pred_p2_px['x'] - gt_p2_px['x'])**2 + (pred_p2_px['y'] - gt_p2_px['y'])**2)
            avg_error = (p1_error + p2_error) / 2
            
            errors.append({
                'image': sample['image_filename'],
                'p1_error': p1_error,
                'p2_error': p2_error,
                'avg_error': avg_error,
                'gt_p1': gt_p1_px,
                'gt_p2': gt_p2_px,
                'pred_p1': pred_p1_px,
                'pred_p2': pred_p2_px
            })
            
            successful_tests += 1
            
            print(f"Image {i+1}/{min(20, len(samples))}: {sample['image_filename']}")
            print(f"  P1 Error: {p1_error:.2f}px, P2 Error: {p2_error:.2f}px, Avg: {avg_error:.2f}px")
            
        except Exception as e:
            print(f"ERROR: Error processing {sample['image_filename']}: {e}")
            failed_tests += 1
            continue
    
    # Calculate statistics
    if errors:
        avg_errors = [e['avg_error'] for e in errors]
        p1_errors = [e['p1_error'] for e in errors]
        p2_errors = [e['p2_error'] for e in errors]
        
        mean_error = np.mean(avg_errors)
        std_error = np.std(avg_errors)
        median_error = np.median(avg_errors)
        max_error = np.max(avg_errors)
        min_error = np.min(avg_errors)
        
        mean_p1_error = np.mean(p1_errors)
        mean_p2_error = np.mean(p2_errors)
        
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)
        print(f"Successful tests: {successful_tests}")
        print(f"Failed tests: {failed_tests}")
        print(f"\nAverage Error Statistics:")
        print(f"  Mean: {mean_error:.2f}px")
        print(f"  Std: {std_error:.2f}px")
        print(f"  Median: {median_error:.2f}px")
        print(f"  Min: {min_error:.2f}px")
        print(f"  Max: {max_error:.2f}px")
        print(f"\nPer-Landmark Errors:")
        print(f"  P1 Mean Error: {mean_p1_error:.2f}px")
        print(f"  P2 Mean Error: {mean_p2_error:.2f}px")
        
        # Check if error is acceptable
        if mean_error < 2.0:
            print(f"\nSUCCESS! Mean error ({mean_error:.2f}px) is less than 2px")
        elif mean_error < 10.0:
            print(f"\nWARNING! Mean error ({mean_error:.2f}px) is between 2px and 10px")
            print(f"   Model may need fine-tuning or more training data")
        else:
            print(f"\nFAILURE! Mean error ({mean_error:.2f}px) is greater than 10px")
            print(f"   Model implementation may be incorrect")
        
        # Show worst cases
        print(f"\nWorst 3 cases:")
        sorted_errors = sorted(errors, key=lambda x: x['avg_error'], reverse=True)
        for e in sorted_errors[:3]:
            print(f"  {e['image']}: {e['avg_error']:.2f}px")
        
        return {
            'success': True,
            'mean_error': mean_error,
            'std_error': std_error,
            'median_error': median_error,
            'max_error': max_error,
            'min_error': min_error,
            'mean_p1_error': mean_p1_error,
            'mean_p2_error': mean_p2_error,
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'errors': errors
        }
    else:
        print("\nERROR: No successful tests!")
        return {
            'success': False,
            'message': 'No successful tests'
        }


if __name__ == '__main__':
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
    
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)
    
    if not annotations_file.exists():
        print(f"ERROR: Annotations not found: {annotations_file}")
        sys.exit(1)
    
    if not images_dir.exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    print("="*70)
    print("P1/P2 Heatmap Model Testing")
    print("="*70)
    print(f"Model: {model_path}")
    print(f"Annotations: {annotations_file}")
    print(f"Images: {images_dir}")
    print("="*70)
    print()
    
    # Load model
    print("Loading model...")
    model, device = load_model(str(model_path), image_size=1024, heatmap_size=256)
    print("SUCCESS: Model loaded successfully")
    print()
    
    # Test
    results = test_model_on_images(
        model, device,
        str(annotations_file),
        str(images_dir),
        image_size=1024,
        heatmap_size=256
    )
    
    # Save results
    if results.get('success'):
        results_file = script_dir / 'test_results_p1p2.json'
        # Remove errors list for JSON serialization (too large)
        results_to_save = {k: v for k, v in results.items() if k != 'errors'}
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\nSUCCESS: Results saved to: {results_file}")

