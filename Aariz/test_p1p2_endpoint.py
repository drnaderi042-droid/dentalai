"""
Test script for P1/P2 Heatmap Model Endpoint
Tests the endpoint on multiple images and compares with ground truth
"""
import requests
import json
import base64
from PIL import Image
import numpy as np
from pathlib import Path
import sys

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


def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def test_endpoint(annotations_file, images_dir, endpoint_url='http://localhost:5001/detect-p1p2-heatmap', num_images=10):
    """Test the endpoint on images from annotations"""
    images_dir = Path(images_dir)
    
    # Load annotations
    samples = load_annotations(annotations_file)
    print(f"Loaded {len(samples)} samples from annotations")
    print(f"Testing on first {num_images} images...")
    print("="*70)
    
    errors = []
    successful_tests = 0
    failed_tests = 0
    
    for i, sample in enumerate(samples[:num_images]):
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
            # Load image to get original size
            image = Image.open(image_path).convert('RGB')
            orig_width, orig_height = image.size
            
            # Ground truth coordinates (normalized [0,1] relative to original image)
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
            
            # Convert image to base64
            image_base64 = image_to_base64(image_path)
            
            # Call endpoint
            response = requests.post(
                endpoint_url,
                json={'image_base64': image_base64},
                headers={'Content-Type': 'application/json'},
                timeout=60
            )
            
            if response.status_code != 200:
                print(f"ERROR: Request failed with status {response.status_code}: {response.text}")
                failed_tests += 1
                continue
            
            data = response.json()
            
            if not data.get('success'):
                print(f"ERROR: Detection failed: {data.get('error', 'Unknown error')}")
                failed_tests += 1
                continue
            
            # Get predicted coordinates
            landmarks = data.get('landmarks', {})
            if 'p1' not in landmarks or 'p2' not in landmarks:
                print(f"ERROR: Missing p1 or p2 in response")
                failed_tests += 1
                continue
            
            pred_p1_px = landmarks['p1']
            pred_p2_px = landmarks['p2']
            
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
                'pred_p2': pred_p2_px,
                'image_size': f"{orig_width}x{orig_height}"
            })
            
            successful_tests += 1
            
            print(f"Image {i+1}/{num_images}: {sample['image_filename']}")
            print(f"  Image size: {orig_width}x{orig_height}")
            print(f"  GT P1: ({gt_p1_px['x']:.1f}, {gt_p1_px['y']:.1f}), GT P2: ({gt_p2_px['x']:.1f}, {gt_p2_px['y']:.1f})")
            print(f"  Pred P1: ({pred_p1_px['x']:.1f}, {pred_p1_px['y']:.1f}), Pred P2: ({pred_p2_px['x']:.1f}, {pred_p2_px['y']:.1f})")
            print(f"  P1 Error: {p1_error:.2f}px, P2 Error: {p2_error:.2f}px, Avg: {avg_error:.2f}px")
            if avg_error > 10:
                print(f"  ⚠️  WARNING: Error > 10px!")
            print()
            
        except Exception as e:
            print(f"ERROR: Error processing {sample['image_filename']}: {e}")
            import traceback
            traceback.print_exc()
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
        
        print("="*70)
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
            print(f"\n✅ SUCCESS! Mean error ({mean_error:.2f}px) is less than 2px")
        elif mean_error < 10.0:
            print(f"\n⚠️  WARNING! Mean error ({mean_error:.2f}px) is between 2px and 10px")
            print(f"   Model may need fine-tuning or coordinate transformation fix")
        else:
            print(f"\n❌ FAILURE! Mean error ({mean_error:.2f}px) is greater than 10px")
            print(f"   Coordinate transformation is incorrect!")
        
        # Show worst cases
        print(f"\nWorst 3 cases:")
        sorted_errors = sorted(errors, key=lambda x: x['avg_error'], reverse=True)
        for e in sorted_errors[:3]:
            print(f"  {e['image']} ({e['image_size']}): {e['avg_error']:.2f}px")
            print(f"    GT: P1=({e['gt_p1']['x']:.1f}, {e['gt_p1']['y']:.1f}), P2=({e['gt_p2']['x']:.1f}, {e['gt_p2']['y']:.1f})")
            print(f"    Pred: P1=({e['pred_p1']['x']:.1f}, {e['pred_p1']['y']:.1f}), P2=({e['pred_p2']['x']:.1f}, {e['pred_p2']['y']:.1f})")
        
        # Save results
        results_file = Path(__file__).parent / 'test_endpoint_results.json'
        results_to_save = {
            'mean_error': float(mean_error),
            'std_error': float(std_error),
            'median_error': float(median_error),
            'max_error': float(max_error),
            'min_error': float(min_error),
            'mean_p1_error': float(mean_p1_error),
            'mean_p2_error': float(mean_p2_error),
            'successful_tests': successful_tests,
            'failed_tests': failed_tests,
            'errors': errors
        }
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n✅ Results saved to: {results_file}")
        
        return results_to_save
    else:
        print("\nERROR: No successful tests!")
        return None


if __name__ == '__main__':
    script_dir = Path(__file__).parent
    
    annotations_file = script_dir / 'annotations_p1_p2.json'
    images_dir = script_dir / 'train' / 'Cephalograms'
    
    # Try different possible paths for images
    possible_paths = [
        script_dir / 'train' / 'Cephalograms',
        script_dir / 'Cephalograms',
        script_dir.parent / 'Aariz' / 'train' / 'Cephalograms',
    ]
    images_dir_found = None
    for path in possible_paths:
        if path.exists():
            images_dir_found = path
            break
    if images_dir_found is None:
        images_dir_found = possible_paths[0]  # Use first as default for error message
    
    if not annotations_file.exists():
        print(f"ERROR: Annotations not found: {annotations_file}")
        sys.exit(1)
    
    if not images_dir_found.exists():
        print(f"ERROR: Images directory not found: {images_dir_found}")
        print(f"Tried paths: {possible_paths}")
        sys.exit(1)
    
    print("="*70)
    print("P1/P2 Heatmap Model Endpoint Testing")
    print("="*70)
    print(f"Annotations: {annotations_file}")
    print(f"Images: {images_dir_found}")
    print(f"Endpoint: http://localhost:5001/detect-p1p2-heatmap")
    print("="*70)
    print()
    
    # Test
    results = test_endpoint(
        str(annotations_file),
        str(images_dir_found),
        endpoint_url='http://localhost:5001/detect-p1p2-heatmap',
        num_images=10
    )
