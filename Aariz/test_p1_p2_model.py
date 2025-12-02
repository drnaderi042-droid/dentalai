"""
Test the trained P1/P2 model on the dataset.
"""

import os
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import model
import sys
sys.path.append(str(Path(__file__).parent))
from model import CephalometricLandmarkDetector

# Change to script directory
script_dir = Path(__file__).parent.resolve()
os.chdir(script_dir)

def load_p1_p2_model(checkpoint_path='checkpoint_p1_p2.pth', device='cuda'):
    """Load the trained P1/P2 model."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = CephalometricLandmarkDetector(num_landmarks=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint

def detect_p1_p2(model, image_path, image_size=512, device='cuda'):
    """Detect p1 and p2 on an image."""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, (image_size, image_size))
    
    # Normalize
    image_tensor = torch.FloatTensor(image_resized).permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
    
    # Denormalize
    landmarks = output[0].cpu().numpy() * image_size
    
    # Scale back to original size
    scale_x = orig_w / image_size
    scale_y = orig_h / image_size
    
    p1 = {'x': float(landmarks[0] * scale_x), 'y': float(landmarks[1] * scale_y)}
    p2 = {'x': float(landmarks[2] * scale_x), 'y': float(landmarks[3] * scale_y)}
    
    return p1, p2

def get_ground_truth(image_id, annotation_dir):
    """Load ground truth p1 and p2."""
    annotation_file = annotation_dir / f"{image_id}.json"
    
    with open(annotation_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    landmarks = {}
    for lm in data['landmarks']:
        if lm['symbol'] in ['p1', 'p2']:
            landmarks[lm['symbol']] = lm['value']
    
    return landmarks.get('p1'), landmarks.get('p2')

def calculate_distance(p1, p2):
    """Calculate Euclidean distance."""
    if p1 is None or p2 is None:
        return None
    dx = p1['x'] - p2['x']
    dy = p1['y'] - p2['y']
    return np.sqrt(dx**2 + dy**2)

def test_model():
    """Test the model on all P1/P2 images."""
    print("="*60)
    print("Testing P1/P2 Calibration Model")
    print("="*60)
    
    # Check if model exists
    if not Path('checkpoint_p1_p2.pth').exists():
        print("\n[ERROR] Model not found!")
        print("   Please train the model first: python train_p1_p2.py")
        return
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n[DEVICE] Using: {device}")
    
    # Load model
    print("[INFO] Loading model...")
    model, checkpoint = load_p1_p2_model(device=device)
    print(f"[SUCCESS] Model loaded (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")
    
    # Paths
    image_dir = Path('Aariz/train/Cephalograms')
    annotation_dir = Path('Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    
    # Test image IDs
    test_ids = [
        'cks2ip8fq29yq0yufc4scftj8',
        'cks2ip8fq29z00yufgnfla2tf',
        'cks2ip8fq29za0yuf0tqu1qjs',
        'cks2ip8fq2a0j0yufdfssbc09',
        'cks2ip8fq2a0t0yufgab484s9',
        'cks2ip8fq2a130yuf5gyh2nrs',
        'cks2ip8fq2a180yufh98ue4yo',
        'cks2ip8fq2a1i0yuf9ra939xh',
        'cks2ip8fq2a1n0yuf8nqt3ndt',
        'cks2ip8fq2a1x0yuffrma5nom',
        'cks2ip8fr2a2c0yuf3pc66vjh',
        'cks2ip8fr2a2h0yuf2r8o8teg',
        'cks2ip8fr2a2m0yuf7tz6ci2u',
        'cks2ip8fr2a2w0yuf49bu0v1w',
        'cks2ip8fr2a3b0yuff9a6ac73',
        'cks2ip8fr2a3l0yuf8pbcfolv',
        'cks2ip8fr2a3q0yufadyu84rc',
        'cks2ip8fr2a3v0yuf4hws1b5t',
    ]
    
    print(f"\n[INFO] Testing on {len(test_ids)} images...\n")
    
    # Test
    results = []
    
    for image_id in tqdm(test_ids, desc="Testing"):
        # Find image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = image_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path is None:
            print(f"[WARNING] Image not found: {image_id}")
            continue
        
        # Get ground truth
        gt_p1, gt_p2 = get_ground_truth(image_id, annotation_dir)
        
        if gt_p1 is None or gt_p2 is None:
            print(f"[WARNING] Ground truth missing: {image_id}")
            continue
        
        # Detect
        pred_p1, pred_p2 = detect_p1_p2(model, image_path, device=device)
        
        # Calculate errors
        p1_error = calculate_distance(pred_p1, gt_p1)
        p2_error = calculate_distance(pred_p2, gt_p2)
        
        results.append({
            'image_id': image_id,
            'p1_error': p1_error,
            'p2_error': p2_error,
            'avg_error': (p1_error + p2_error) / 2
        })
    
    # Summary
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60 + "\n")
    
    if results:
        p1_errors = [r['p1_error'] for r in results]
        p2_errors = [r['p2_error'] for r in results]
        avg_errors = [r['avg_error'] for r in results]
        
        print(f"Total images: {len(results)}")
        print(f"\nP1 Error:")
        print(f"   Mean: {np.mean(p1_errors):.2f} px")
        print(f"   Median: {np.median(p1_errors):.2f} px")
        print(f"   Max: {np.max(p1_errors):.2f} px")
        
        print(f"\nP2 Error:")
        print(f"   Mean: {np.mean(p2_errors):.2f} px")
        print(f"   Median: {np.median(p2_errors):.2f} px")
        print(f"   Max: {np.max(p2_errors):.2f} px")
        
        print(f"\nAverage Error:")
        print(f"   Mean: {np.mean(avg_errors):.2f} px")
        print(f"   Median: {np.median(avg_errors):.2f} px")
        
        # Accuracy
        accurate_5px = sum(1 for e in avg_errors if e < 5)
        accurate_10px = sum(1 for e in avg_errors if e < 10)
        accurate_20px = sum(1 for e in avg_errors if e < 20)
        
        print(f"\nAccuracy:")
        print(f"   < 5px:  {accurate_5px}/{len(results)} ({100*accurate_5px/len(results):.1f}%)")
        print(f"   < 10px: {accurate_10px}/{len(results)} ({100*accurate_10px/len(results):.1f}%)")
        print(f"   < 20px: {accurate_20px}/{len(results)} ({100*accurate_20px/len(results):.1f}%)")
        
        # Best/worst
        print("\n[BEST] Best predictions:")
        for r in sorted(results, key=lambda x: x['avg_error'])[:3]:
            print(f"   {r['image_id']}: {r['avg_error']:.2f}px")
        
        print("\n[WORST] Worst predictions:")
        for r in sorted(results, key=lambda x: x['avg_error'], reverse=True)[:3]:
            print(f"   {r['image_id']}: {r['avg_error']:.2f}px")
        
        # Visualize one example
        print("\n[VISUALIZE] Creating prediction visualization...")
        best = min(results, key=lambda x: x['avg_error'])
        visualize_prediction(best['image_id'], model, image_dir, annotation_dir, device)
        print(f"   Saved as: p1_p2_prediction_best.png")
    else:
        print("[ERROR] No results to display")

def visualize_prediction(image_id, model, image_dir, annotation_dir, device):
    """Visualize prediction vs ground truth."""
    # Find image
    image_path = None
    for ext in ['.png', '.jpg', '.jpeg']:
        potential_path = image_dir / f"{image_id}{ext}"
        if potential_path.exists():
            image_path = potential_path
            break
    
    # Load image
    image = cv2.imread(str(image_path))
    
    # Get predictions
    pred_p1, pred_p2 = detect_p1_p2(model, image_path, device=device)
    
    # Get ground truth
    gt_p1, gt_p2 = get_ground_truth(image_id, annotation_dir)
    
    # Draw
    # Ground truth (green)
    cv2.circle(image, (int(gt_p1['x']), int(gt_p1['y'])), 8, (0, 255, 0), 2)
    cv2.circle(image, (int(gt_p2['x']), int(gt_p2['y'])), 8, (0, 255, 0), 2)
    cv2.putText(image, 'GT p1', (int(gt_p1['x'])+10, int(gt_p1['y'])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'GT p2', (int(gt_p2['x'])+10, int(gt_p2['y'])), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Predictions (red)
    cv2.circle(image, (int(pred_p1['x']), int(pred_p1['y'])), 6, (0, 0, 255), -1)
    cv2.circle(image, (int(pred_p2['x']), int(pred_p2['y'])), 6, (0, 0, 255), -1)
    cv2.putText(image, 'Pred p1', (int(pred_p1['x'])+10, int(pred_p1['y'])-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, 'Pred p2', (int(pred_p2['x'])+10, int(pred_p2['y'])-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # Save
    cv2.imwrite('p1_p2_prediction_best.png', image)

if __name__ == '__main__':
    test_model()

