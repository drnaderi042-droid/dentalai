"""
Evaluate combined 31-landmark model on 5 test images
Compare predictions with ground truth annotations
"""
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import json
from pathlib import Path
from create_combined_model import SimplifiedCombinedModel
import os


def load_model(model_path='combined_31_landmarks.pth'):
    """Load the combined model"""
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SimplifiedCombinedModel(
        num_landmarks=31,
        backbone='hrnet_w18',
        output_size=checkpoint.get('heatmap_size', 192)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    return model, device


def load_annotation(annotation_path):
    """Load annotation file (29 anatomical landmarks only, excluding p1/p2)"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    landmarks = []
    for lm in data['landmarks']:
        # Skip p1 and p2
        if lm.get('symbol') in ['p1', 'p2']:
            continue
        
        if 'value' in lm and lm['value'] is not None:
            landmarks.append([lm['value']['x'], lm['value']['y']])
        else:
            landmarks.append([0, 0])  # Missing landmark
    
    return np.array(landmarks)


def load_p1p2_annotation(annotation_path):
    """Load P1/P2 annotation from extended format"""
    with open(annotation_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # P1 and P2 are in the landmarks array
    p1 = None
    p2 = None
    
    for lm in data.get('landmarks', []):
        if lm.get('symbol') == 'p1' and 'value' in lm:
            p1 = [lm['value']['x'], lm['value']['y']]
        elif lm.get('symbol') == 'p2' and 'value' in lm:
            p2 = [lm['value']['x'], lm['value']['y']]
    
    if p1 is not None and p2 is not None:
        return np.array([p1, p2])
    
    return None


def preprocess_image(image_path, target_size=768):
    """Preprocess image for model"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_h, original_w = img.shape[:2]
    
    img_resized = cv2.resize(img, (target_size, target_size))
    img_normalized = img_resized.astype(np.float32) / 255.0
    
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_normalized = (img_normalized - mean) / std
    
    img_tensor = torch.from_numpy(img_normalized).permute(2, 0, 1).unsqueeze(0)
    
    return img_tensor, (original_h, original_w)


def predict_landmarks(model, image_tensor, device):
    """Predict landmarks"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        heatmaps = model(image_tensor)
        coords = model.extract_coordinates(heatmaps)
    
    return coords.cpu().numpy()


def denormalize_coords(coords, original_size):
    """Convert normalized [0,1] to pixel coordinates"""
    original_h, original_w = original_size
    coords = coords.reshape(-1, 2)
    coords[:, 0] *= original_w
    coords[:, 1] *= original_h
    return coords


def calculate_mre(pred, gt):
    """Calculate Mean Radial Error"""
    distances = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    return np.mean(distances)


def calculate_sdr(pred, gt, thresholds=[2.0, 2.5, 3.0, 4.0]):
    """Calculate Success Detection Rate at different thresholds"""
    distances = np.sqrt(np.sum((pred - gt) ** 2, axis=1))
    sdrs = {}
    for thresh in thresholds:
        sdr = np.mean(distances <= thresh) * 100
        sdrs[f'SDR_{thresh}mm'] = sdr
    return sdrs


def visualize_comparison(image_path, pred_landmarks, gt_landmarks, output_path):
    """Visualize prediction vs ground truth"""
    img = cv2.imread(str(image_path))
    
    # Draw ground truth (green)
    for i, (x, y) in enumerate(gt_landmarks):
        if x > 0 and y > 0:  # Skip missing landmarks
            cv2.circle(img, (int(x), int(y)), 4, (0, 255, 0), -1)
    
    # Draw predictions (red)
    for i, (x, y) in enumerate(pred_landmarks):
        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # Add legend
    cv2.putText(img, 'Green: Ground Truth', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(img, 'Red: Prediction', (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imwrite(str(output_path), img)
    return img


def main():
    print("="*70)
    print("Evaluating Combined 31-Landmark Model on 5 Test Images")
    print("="*70)
    
    # Load model
    model, device = load_model('combined_31_landmarks.pth')
    
    # Find annotation files with P1/P2
    annotation_dir = Path('Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    annotation_files = list(annotation_dir.glob('*.json'))
    
    # Filter files that have p1 and p2
    files_with_p1p2 = []
    for ann_file in annotation_files:
        if ann_file.name == 'project-6-at-2025-11-11-03-58-db627e7c.json':
            continue  # Skip label studio format
        
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if landmarks array contains p1 and p2
            has_p1 = False
            has_p2 = False
            for lm in data.get('landmarks', []):
                if lm.get('symbol') == 'p1':
                    has_p1 = True
                elif lm.get('symbol') == 'p2':
                    has_p2 = True
            
            if has_p1 and has_p2:
                files_with_p1p2.append(ann_file)
        except Exception as e:
            continue
    
    print(f"\nFound {len(files_with_p1p2)} files with P1/P2 annotations")
    
    # Select 5 files for testing
    test_files = files_with_p1p2[:5]
    
    if len(test_files) == 0:
        print("ERROR: No files with P1/P2 annotations found!")
        return
    
    print(f"Testing on {len(test_files)} images\n")
    
    # Create output directory
    output_dir = Path('combined_test_results/evaluation_5_images')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = []
    total_mre_29 = []
    total_mre_p1p2 = []
    total_mre_all = []
    
    # Process each image
    for idx, ann_file in enumerate(test_files, 1):
        print(f"\n{'='*70}")
        print(f"Image {idx}/{len(test_files)}: {ann_file.stem}")
        print(f"{'='*70}")
        
        # Find corresponding image
        image_path = Path('Aariz/train/Cephalograms') / f"{ann_file.stem}.png"
        if not image_path.exists():
            print(f"WARNING: Image not found: {image_path}")
            continue
        
        # Load annotations
        gt_29_landmarks = load_annotation(ann_file)
        gt_p1p2 = load_p1p2_annotation(ann_file)
        
        if gt_p1p2 is None:
            print("WARNING: No P1/P2 in this file")
            continue
        
        # Combine ground truth
        gt_all = np.vstack([gt_29_landmarks, gt_p1p2])  # 31 landmarks
        
        print(f"Ground truth loaded: 29 + 2 = {len(gt_all)} landmarks")
        
        # Preprocess and predict
        image_tensor, original_size = preprocess_image(image_path)
        pred_coords = predict_landmarks(model, image_tensor, device)
        pred_landmarks = denormalize_coords(pred_coords, original_size)
        
        print(f"Predictions: {len(pred_landmarks)} landmarks")
        
        # Calculate errors
        # 29 anatomical landmarks
        mre_29 = calculate_mre(pred_landmarks[:29], gt_29_landmarks)
        sdr_29 = calculate_sdr(pred_landmarks[:29], gt_29_landmarks)
        
        # P1/P2 calibration points
        mre_p1p2 = calculate_mre(pred_landmarks[29:31], gt_p1p2)
        sdr_p1p2 = calculate_sdr(pred_landmarks[29:31], gt_p1p2)
        
        # All 31 landmarks
        mre_all = calculate_mre(pred_landmarks, gt_all)
        sdr_all = calculate_sdr(pred_landmarks, gt_all)
        
        # Store results
        result = {
            'image_id': ann_file.stem,
            'image_path': str(image_path),
            'anatomical_landmarks': {
                'count': 29,
                'mre_mm': float(mre_29),
                'sdr': {k: float(v) for k, v in sdr_29.items()}
            },
            'calibration_points': {
                'count': 2,
                'mre_mm': float(mre_p1p2),
                'sdr': {k: float(v) for k, v in sdr_p1p2.items()}
            },
            'all_landmarks': {
                'count': 31,
                'mre_mm': float(mre_all),
                'sdr': {k: float(v) for k, v in sdr_all.items()}
            }
        }
        
        all_results.append(result)
        total_mre_29.append(mre_29)
        total_mre_p1p2.append(mre_p1p2)
        total_mre_all.append(mre_all)
        
        # Print results
        print(f"\n29 Anatomical Landmarks:")
        print(f"  MRE: {mre_29:.2f} mm")
        print(f"  SDR@2mm: {sdr_29['SDR_2.0mm']:.1f}%")
        print(f"  SDR@3mm: {sdr_29['SDR_3.0mm']:.1f}%")
        print(f"  SDR@4mm: {sdr_29['SDR_4.0mm']:.1f}%")
        
        print(f"\n2 Calibration Points (P1/P2):")
        print(f"  MRE: {mre_p1p2:.2f} mm")
        print(f"  SDR@2mm: {sdr_p1p2['SDR_2.0mm']:.1f}%")
        print(f"  SDR@3mm: {sdr_p1p2['SDR_3.0mm']:.1f}%")
        
        print(f"\nAll 31 Landmarks:")
        print(f"  MRE: {mre_all:.2f} mm")
        print(f"  SDR@2mm: {sdr_all['SDR_2.0mm']:.1f}%")
        print(f"  SDR@3mm: {sdr_all['SDR_3.0mm']:.1f}%")
        
        # Visualize
        vis_path = output_dir / f"{ann_file.stem}_comparison.png"
        visualize_comparison(image_path, pred_landmarks, gt_all, vis_path)
        print(f"\nVisualization saved: {vis_path}")
    
    # Calculate average results
    print(f"\n{'='*70}")
    print("AVERAGE RESULTS ACROSS ALL IMAGES")
    print(f"{'='*70}")
    
    avg_mre_29 = np.mean(total_mre_29)
    avg_mre_p1p2 = np.mean(total_mre_p1p2)
    avg_mre_all = np.mean(total_mre_all)
    
    print(f"\n29 Anatomical Landmarks:")
    print(f"  Average MRE: {avg_mre_29:.2f} mm")
    
    print(f"\n2 Calibration Points (P1/P2):")
    print(f"  Average MRE: {avg_mre_p1p2:.2f} mm")
    
    print(f"\nAll 31 Landmarks:")
    print(f"  Average MRE: {avg_mre_all:.2f} mm")
    
    # Save results
    summary = {
        'model': 'combined_31_landmarks.pth',
        'num_test_images': len(test_files),
        'individual_results': all_results,
        'average_results': {
            'anatomical_landmarks_mre_mm': float(avg_mre_29),
            'calibration_points_mre_mm': float(avg_mre_p1p2),
            'all_landmarks_mre_mm': float(avg_mre_all)
        }
    }
    
    results_path = output_dir / 'evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*70}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {output_dir}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()