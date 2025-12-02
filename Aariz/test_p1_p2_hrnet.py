"""
Test script for HRNet-based p1/p2 calibration point detector
Evaluates model performance on test images
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import json
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from model import HRNetP1P2Detector


def load_hrnet_p1_p2_model(checkpoint_path, hrnet_variant='hrnet_w18', device='cpu'):
    """Load trained HRNet model from checkpoint"""
    
    print(f"Loading HRNet model from: {checkpoint_path}")
    
    # Create model
    model = HRNetP1P2Detector(
        num_landmarks=2,
        hrnet_variant=hrnet_variant,
        pretrained=False  # We're loading trained weights
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"  Model loaded successfully!")
    print(f"  - Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  - Val Loss: {checkpoint.get('val_loss', 0):.6f}")
    print(f"  - Pixel Error: {checkpoint.get('pixel_error', 0):.2f} px")
    
    return model


def test_hrnet_on_dataset(
    model,
    annotations_file,
    images_dir,
    output_dir='test_results_hrnet',
    image_size=512,
    device='cpu',
    visualize=True
):
    """Test model on dataset and calculate metrics"""
    
    print(f"\n[TEST] Testing HRNet model on dataset...")
    print(f"  Annotations: {annotations_file}")
    print(f"  Images: {images_dir}")
    
    # Create output directory
    if visualize:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load annotations
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test metrics
    errors = []
    p1_errors = []
    p2_errors = []
    
    print(f"\n[TESTING] Processing {len(data)} samples...")
    
    for idx, item in enumerate(data):
        if 'annotations' in item and item['annotations']:
            annotation = item['annotations'][0]
            
            if 'result' in annotation:
                # Extract ground truth p1 and p2
                p1_gt = None
                p2_gt = None
                
                for result_item in annotation['result']:
                    if result_item.get('type') == 'keypointlabels':
                        value = result_item.get('value', {})
                        labels = value.get('keypointlabels', [])
                        
                        if 'p1' in labels:
                            p1_gt = {
                                'x': value.get('x', 0) / 100.0,
                                'y': value.get('y', 0) / 100.0
                            }
                        elif 'p2' in labels:
                            p2_gt = {
                                'x': value.get('x', 0) / 100.0,
                                'y': value.get('y', 0) / 100.0
                            }
                
                if p1_gt and p2_gt:
                    # Load and preprocess image
                    image_url = item['data'].get('image', '')
                    image_filename = image_url.split('/')[-1]
                    
                    # Remove UUID prefix if exists
                    if '-' in image_filename:
                        parts = image_filename.split('-')
                        if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
                            image_filename = '-'.join(parts[1:])
                    
                    image_path = Path(images_dir) / image_filename
                    
                    # Try alternative filename if not found
                    if not image_path.exists():
                        alt_filename = image_url.split('/')[-1]
                        alt_path = Path(images_dir) / alt_filename
                        if alt_path.exists():
                            image_path = alt_path
                        else:
                            print(f"  [SKIP] Image not found: {image_filename}")
                            continue
                    
                    image = Image.open(image_path).convert('RGB')
                    orig_width, orig_height = image.size
                    
                    # Transform image
                    image_tensor = transform(image).unsqueeze(0).to(device)
                    
                    # Predict
                    with torch.no_grad():
                        pred = model(image_tensor)
                        pred = pred.cpu().numpy()[0]  # [p1_x, p1_y, p2_x, p2_y]
                    
                    # Denormalize predictions
                    p1_pred = {'x': pred[0], 'y': pred[1]}
                    p2_pred = {'x': pred[2], 'y': pred[3]}
                    
                    # Calculate pixel errors
                    p1_error = np.sqrt(
                        ((p1_pred['x'] - p1_gt['x']) * image_size) ** 2 +
                        ((p1_pred['y'] - p1_gt['y']) * image_size) ** 2
                    )
                    
                    p2_error = np.sqrt(
                        ((p2_pred['x'] - p2_gt['x']) * image_size) ** 2 +
                        ((p2_pred['y'] - p2_gt['y']) * image_size) ** 2
                    )
                    
                    avg_error = (p1_error + p2_error) / 2
                    
                    errors.append(avg_error)
                    p1_errors.append(p1_error)
                    p2_errors.append(p2_error)
                    
                    print(f"  [{idx+1}] {image_filename}: "
                          f"p1_err={p1_error:.2f}px, p2_err={p2_error:.2f}px, "
                          f"avg={avg_error:.2f}px")
                    
                    # Visualize results
                    if visualize and idx < 10:  # Only visualize first 10 samples
                        vis_image = image.copy()
                        draw = ImageDraw.Draw(vis_image)
                        
                        # Draw ground truth (green)
                        p1_gt_px = (int(p1_gt['x'] * orig_width), int(p1_gt['y'] * orig_height))
                        p2_gt_px = (int(p2_gt['x'] * orig_width), int(p2_gt['y'] * orig_height))
                        
                        draw.ellipse([p1_gt_px[0]-10, p1_gt_px[1]-10,
                                     p1_gt_px[0]+10, p1_gt_px[1]+10],
                                    outline='green', width=3)
                        draw.ellipse([p2_gt_px[0]-10, p2_gt_px[1]-10,
                                     p2_gt_px[0]+10, p2_gt_px[1]+10],
                                    outline='green', width=3)
                        
                        # Draw predictions (red)
                        p1_pred_px = (int(p1_pred['x'] * orig_width), int(p1_pred['y'] * orig_height))
                        p2_pred_px = (int(p2_pred['x'] * orig_width), int(p2_pred['y'] * orig_height))
                        
                        draw.ellipse([p1_pred_px[0]-8, p1_pred_px[1]-8,
                                     p1_pred_px[0]+8, p1_pred_px[1]+8],
                                    outline='red', width=3)
                        draw.ellipse([p2_pred_px[0]-8, p2_pred_px[1]-8,
                                     p2_pred_px[0]+8, p2_pred_px[1]+8],
                                    outline='red', width=3)
                        
                        # Save visualization
                        output_path = Path(output_dir) / f"result_{idx+1}_{image_filename}"
                        vis_image.save(output_path)
    
    # Calculate statistics
    if errors:
        print(f"\n[RESULTS] Test Statistics:")
        print(f"  - Samples tested: {len(errors)}")
        print(f"  - Average error: {np.mean(errors):.2f} px")
        print(f"  - Median error: {np.median(errors):.2f} px")
        print(f"  - Std deviation: {np.std(errors):.2f} px")
        print(f"  - Min error: {np.min(errors):.2f} px")
        print(f"  - Max error: {np.max(errors):.2f} px")
        print(f"\n  - P1 avg error: {np.mean(p1_errors):.2f} px")
        print(f"  - P2 avg error: {np.mean(p2_errors):.2f} px")
        
        if visualize:
            print(f"\n  Visualizations saved to: {output_dir}/")
    else:
        print("\n[WARNING] No samples with p1/p2 annotations found!")
    
    return {
        'avg_error': np.mean(errors) if errors else 0,
        'median_error': np.median(errors) if errors else 0,
        'p1_avg_error': np.mean(p1_errors) if p1_errors else 0,
        'p2_avg_error': np.mean(p2_errors) if p2_errors else 0,
    }


if __name__ == '__main__':
    # Setup paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Use the new annotations file created by p1_p2_annotator.py
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'Aariz/train/Cephalograms'
    
    # Check files
    if not Path(annotations_file).exists():
        print(f"ERROR: Annotations file not found: {annotations_file}")
        print(f"\nPlease create annotations first using:")
        print(f"  annotate_p1_p2.bat")
        sys.exit(1)
    
    if not Path(images_dir).exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        sys.exit(1)
    
    # Find latest HRNet model
    model_files = list(Path('models').glob('hrnet_p1p2_best_*.pth'))
    
    if not model_files:
        print("ERROR: No trained HRNet model found in models/ directory")
        print("Please train a model first using train_p1_p2_hrnet.py")
        sys.exit(1)
    
    # Use the most recent model
    checkpoint_path = max(model_files, key=lambda p: p.stat().st_mtime)
    print(f"Using model: {checkpoint_path}")
    
    # Determine HRNet variant from filename
    hrnet_variant = 'hrnet_w18'  # default
    if 'w32' in checkpoint_path.name:
        hrnet_variant = 'hrnet_w32'
    elif 'w48' in checkpoint_path.name:
        hrnet_variant = 'hrnet_w48'
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_hrnet_p1_p2_model(checkpoint_path, hrnet_variant, device)
    
    # Test model (use same image size as training: 768px)
    results = test_hrnet_on_dataset(
        model=model,
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_dir='test_results_hrnet',
        image_size=768,  # Match training resolution
        device=device,
        visualize=True
    )
    
    print("\n[DONE] Testing complete!")

