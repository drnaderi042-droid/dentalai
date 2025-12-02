"""
مقایسه دقت p1/p2 detection:
1. از API endpoint
2. از inference script مستقیم
3. با ground truth
"""
import json
import os
import sys
import base64
import requests
import numpy as np
from PIL import Image
from pathlib import Path
import torch
import torchvision.transforms as transforms

# Add paths
base_dir = Path(__file__).resolve().parent
root_dir = base_dir.parent  # Go up one level to root
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / 'Aariz'))

# Import model
from model_heatmap import HRNetP1P2HeatmapDetector

# Configuration
API_URL = "http://localhost:5001/detect-p1p2"
# Annotation file is in current directory (Aariz)
ANNOTATION_FILE = base_dir / "project-6-at-2025-11-11-03-58-db627e7c.json"
# Dataset is in Aariz/Aariz/train/Cephalograms
DATASET_DIR = base_dir / "Aariz" / "train" / "Cephalograms"
MODEL_PATH = base_dir / "models" / "hrnet_p1p2_heatmap_best.pth"

# Debug: print paths (will be printed in main)

def load_ground_truth(annotation_file):
    """Load ground truth from Label Studio annotation file"""
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_dict = {}
    for task in data:
        if not task.get('annotations'):
            continue
        
        # Get image filename
        # file_upload format: "8a75433a-cks2ip8fq2a0j0yufdfssbc09.png"
        # We need to extract the actual filename: "cks2ip8fq2a0j0yufdfssbc09.png"
        file_upload = task.get('file_upload', '')
        if not file_upload:
            continue
        
        # Extract actual filename (remove prefix if exists)
        # Format: "prefix-filename.ext" -> "filename.ext"
        if '-' in file_upload:
            parts = file_upload.split('-', 1)
            if len(parts) == 2:
                image_filename = parts[1]  # Take the part after first dash
            else:
                image_filename = file_upload
        else:
            image_filename = file_upload
        
        # Extract p1 and p2 from annotations
        annotation = task['annotations'][0]
        result = annotation.get('result', [])
        
        p1 = None
        p2 = None
        original_width = None
        original_height = None
        
        for item in result:
            if item.get('type') == 'keypointlabels':
                value = item.get('value', {})
                labels = value.get('keypointlabels', [])
                
                if 'p1' in labels:
                    p1 = {
                        'x': value.get('x', 0),
                        'y': value.get('y', 0)
                    }
                    original_width = item.get('original_width')
                    original_height = item.get('original_height')
                elif 'p2' in labels:
                    p2 = {
                        'x': value.get('x', 0),
                        'y': value.get('y', 0)
                    }
                    if original_width is None:
                        original_width = item.get('original_width')
                        original_height = item.get('original_height')
        
        if p1 and p2 and original_width and original_height:
            # Convert from percentage to pixel coordinates
            p1_px = {
                'x': p1['x'] * original_width / 100.0,
                'y': p1['y'] * original_height / 100.0
            }
            p2_px = {
                'x': p2['x'] * original_width / 100.0,
                'y': p2['y'] * original_height / 100.0
            }
            
            gt_dict[image_filename] = {
                'p1': p1_px,
                'p2': p2_px,
                'width': original_width,
                'height': original_height
            }
    
    return gt_dict

def image_to_base64(image_path):
    """Convert image to base64 string"""
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    return base64.b64encode(image_bytes).decode('utf-8')

def detect_via_api(image_path, api_url):
    """Detect p1/p2 via API endpoint"""
    try:
        image_base64 = image_to_base64(image_path)
        
        response = requests.post(
            api_url,
            json={'image_base64': image_base64},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success') and result.get('p1') and result.get('p2'):
                return {
                    'p1': result['p1'],
                    'p2': result['p2'],
                    'confidence': result.get('confidence', 0),
                    'processing_time': result.get('processing_time', 0)
                }
        return None
    except Exception as e:
        print(f"  API Error: {e}")
        return None

def detect_via_script(image_path, model_path):
    """Detect p1/p2 via direct inference script"""
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_width, original_height = image.size
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = HRNetP1P2HeatmapDetector(num_landmarks=2)
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        model.to(device)
        model.eval()
        
        # Get image size from checkpoint
        image_size = checkpoint.get('image_size', 768)
        heatmap_size = checkpoint.get('heatmap_size', 192)
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            heatmaps = model(image_tensor)
            coords = model.extract_coordinates(heatmaps)
        
        # Convert to pixel coordinates
        # coords shape: (1, 4) -> [p1.x, p1.y, p2.x, p2.y] normalized [0, 1]
        coords_px = coords.cpu().numpy()[0] * image_size  # Shape: (4,)
        
        # Scale back to original image size
        scale_x = original_width / image_size
        scale_y = original_height / image_size
        
        # coords_px format: [p1.x, p1.y, p2.x, p2.y]
        p1 = {
            'x': float(coords_px[0] * scale_x),  # p1.x
            'y': float(coords_px[1] * scale_y)   # p1.y
        }
        p2 = {
            'x': float(coords_px[2] * scale_x),  # p2.x
            'y': float(coords_px[3] * scale_y)   # p2.y
        }
        
        return {
            'p1': p1,
            'p2': p2
        }
    except Exception as e:
        print(f"  Script Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_error(pred, gt):
    """Calculate pixel error between prediction and ground truth"""
    if pred is None or gt is None:
        return None
    
    p1_error = np.sqrt((pred['p1']['x'] - gt['p1']['x'])**2 + 
                       (pred['p1']['y'] - gt['p1']['y'])**2)
    p2_error = np.sqrt((pred['p2']['x'] - gt['p2']['x'])**2 + 
                       (pred['p2']['y'] - gt['p2']['y'])**2)
    
    avg_error = (p1_error + p2_error) / 2.0
    
    return {
        'p1_error': float(p1_error),
        'p2_error': float(p2_error),
        'avg_error': float(avg_error)
    }

def main():
    print("="*70)
    print("P1/P2 Detection Accuracy Comparison")
    print("="*70)
    
    # Debug paths
    print(f"\nDebug - Base dir: {base_dir}")
    print(f"Debug - Annotation file: {ANNOTATION_FILE}")
    print(f"Debug - Annotation file exists: {ANNOTATION_FILE.exists()}")
    print(f"Debug - Dataset dir: {DATASET_DIR}")
    print(f"Debug - Dataset dir exists: {DATASET_DIR.exists()}")
    print(f"Debug - Model path: {MODEL_PATH}")
    print(f"Debug - Model path exists: {MODEL_PATH.exists()}")
    
    # Load ground truth
    print("\n[1/4] Loading ground truth...")
    gt_dict = load_ground_truth(ANNOTATION_FILE)
    print(f"  Loaded {len(gt_dict)} ground truth annotations")
    
    # Get image files from ground truth annotations
    # We'll search for images in dataset based on ground truth filenames
    print(f"\n[2/4] Searching for images in dataset...")
    
    # Results storage
    results = {
        'api': [],
        'script': [],
        'comparison': []
    }
    
    # Find images that have ground truth
    image_files_found = []
    for image_filename in gt_dict.keys():
        # Extract base name without extension
        base_name = Path(image_filename).stem  # e.g., "cks2ip8fq2a0j0yufdfssbc09"
        
        # Try to find the image in dataset directory
        # Search for files that contain the base name (with or without prefix)
        image_path = None
        
        # Method 1: Direct match
        if (DATASET_DIR / image_filename).exists():
            image_path = DATASET_DIR / image_filename
        else:
            # Method 2: Search with glob (with prefix)
            for ext in ['png', 'jpg', 'jpeg']:
                matches = list(DATASET_DIR.glob(f"*-{base_name}.{ext}"))
                if matches:
                    image_path = matches[0]
                    break
            
            # Method 3: Search without prefix
            if not image_path:
                for ext in ['png', 'jpg', 'jpeg']:
                    matches = list(DATASET_DIR.glob(f"{base_name}.{ext}"))
                    if matches:
                        image_path = matches[0]
                        break
        
        if image_path:
            image_files_found.append((image_path, image_filename))
    
    print(f"  Found {len(image_files_found)} images with ground truth")
    
    # Process each image
    print("\n[3/4] Processing images...")
    for i, (image_path, image_filename) in enumerate(image_files_found[:5], 1):  # Test first 5 images
        print(f"\n  [{i}/{min(5, len(image_files_found))}] {image_filename}")
        
        # Get ground truth
        gt = gt_dict[image_filename]
        print(f"    Ground Truth: p1=({gt['p1']['x']:.1f}, {gt['p1']['y']:.1f}), "
              f"p2=({gt['p2']['x']:.1f}, {gt['p2']['y']:.1f})")
        
        # Detect via API
        print(f"    Detecting via API...")
        api_result = detect_via_api(image_path, API_URL)
        if api_result:
            print(f"      API Result: p1=({api_result['p1']['x']:.1f}, {api_result['p1']['y']:.1f}), "
                  f"p2=({api_result['p2']['x']:.1f}, {api_result['p2']['y']:.1f})")
            api_error = calculate_error(api_result, gt)
            if api_error:
                print(f"      API Error: p1={api_error['p1_error']:.2f}px, "
                      f"p2={api_error['p2_error']:.2f}px, avg={api_error['avg_error']:.2f}px")
                results['api'].append(api_error)
        else:
            print(f"      ⚠️  API detection failed")
        
        # Detect via script
        print(f"    Detecting via script...")
        script_result = detect_via_script(image_path, MODEL_PATH)
        if script_result:
            print(f"      Script Result: p1=({script_result['p1']['x']:.1f}, {script_result['p1']['y']:.1f}), "
                  f"p2=({script_result['p2']['x']:.1f}, {script_result['p2']['y']:.1f})")
            script_error = calculate_error(script_result, gt)
            if script_error:
                print(f"      Script Error: p1={script_error['p1_error']:.2f}px, "
                      f"p2={script_error['p2_error']:.2f}px, avg={script_error['avg_error']:.2f}px")
                results['script'].append(script_error)
        else:
            print(f"      ⚠️  Script detection failed")
        
        # Compare API vs Script
        if api_result and script_result:
            api_script_diff = {
                'p1_diff': np.sqrt((api_result['p1']['x'] - script_result['p1']['x'])**2 + 
                                   (api_result['p1']['y'] - script_result['p1']['y'])**2),
                'p2_diff': np.sqrt((api_result['p2']['x'] - script_result['p2']['x'])**2 + 
                                   (api_result['p2']['y'] - script_result['p2']['y'])**2)
            }
            print(f"      API vs Script Diff: p1={api_script_diff['p1_diff']:.2f}px, "
                  f"p2={api_script_diff['p2_diff']:.2f}px")
            results['comparison'].append(api_script_diff)
    
    # Summary
    print("\n" + "="*70)
    print("[4/4] Summary")
    print("="*70)
    
    if results['api']:
        api_avg_p1 = np.mean([r['p1_error'] for r in results['api']])
        api_avg_p2 = np.mean([r['p2_error'] for r in results['api']])
        api_avg_total = np.mean([r['avg_error'] for r in results['api']])
        print(f"\nAPI Detection (vs Ground Truth):")
        print(f"  Average P1 Error: {api_avg_p1:.2f}px")
        print(f"  Average P2 Error: {api_avg_p2:.2f}px")
        print(f"  Average Total Error: {api_avg_total:.2f}px")
    
    if results['script']:
        script_avg_p1 = np.mean([r['p1_error'] for r in results['script']])
        script_avg_p2 = np.mean([r['p2_error'] for r in results['script']])
        script_avg_total = np.mean([r['avg_error'] for r in results['script']])
        print(f"\nScript Detection (vs Ground Truth):")
        print(f"  Average P1 Error: {script_avg_p1:.2f}px")
        print(f"  Average P2 Error: {script_avg_p2:.2f}px")
        print(f"  Average Total Error: {script_avg_total:.2f}px")
    
    if results['comparison']:
        comp_avg_p1 = np.mean([r['p1_diff'] for r in results['comparison']])
        comp_avg_p2 = np.mean([r['p2_diff'] for r in results['comparison']])
        print(f"\nAPI vs Script Difference:")
        print(f"  Average P1 Difference: {comp_avg_p1:.2f}px")
        print(f"  Average P2 Difference: {comp_avg_p2:.2f}px")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

