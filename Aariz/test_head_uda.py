"""
Test script for head_uda.pth model on Aariz dataset
Tests the model on lateral cephalometry images and compares with ground truth
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import json
import os
import glob
from tqdm import tqdm
import cv2
from collections import defaultdict

# Standard 19 landmarks for cephalometric detection (from UDA_Med_Landmark)
HEAD_19_LANDMARKS = [
    "S", "N", "Or", "A", "B", "PNS", "ANS", "U1", "L1", "Me",
    "U6", "L6", "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
]

# Mapping from 19 landmarks to 29 Aariz landmarks (if available)
LANDMARK_MAPPING_19_TO_29 = {
    "S": "S",
    "N": "N", 
    "Or": "Or",
    "A": "A",
    "B": "B",
    "PNS": "PNS",
    "ANS": "ANS",
    "U1": "UIT",  # Upper Incisor Tip
    "L1": "LIT",  # Lower Incisor Tip
    "Me": "Me",
    "U6": "UMT",  # Upper Molar Tip (approximate)
    "L6": "LMT",  # Lower Molar Tip (approximate)
    "Go": "Go",
    "Pog": "Pog",
    "Gn": "Gn",
    "Ar": "Ar",
    "Co": "Co",
    "Po": "Po",
    "R": "R"
}


def load_head_uda_model(checkpoint_path, device='cuda'):
    """Load head_uda.pth model"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # The model has a different structure (ResNet + Transformer + UDA)
    # We'll need to create a wrapper or use it directly
    # For now, let's try to understand the structure
    
    print(f"Model keys: {list(checkpoint.keys())[:20]}...")
    
    # Check if it's a state dict or full checkpoint
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Get model structure info
    cls_layer_shape = state_dict.get('cls_layer.weight', None)
    x_layer_shape = state_dict.get('x_layer.weight', None)
    y_layer_shape = state_dict.get('y_layer.weight', None)
    
    if cls_layer_shape is not None:
        num_landmarks = cls_layer_shape.shape[0]
        print(f"Number of landmarks: {num_landmarks}")
        print(f"cls_layer shape: {cls_layer_shape.shape}")
        print(f"x_layer shape: {x_layer_shape.shape if x_layer_shape is not None else 'N/A'}")
        print(f"y_layer shape: {y_layer_shape.shape if y_layer_shape is not None else 'N/A'}")
    
    return checkpoint, state_dict


def load_ground_truth(image_id, dataset_path="Aariz", annotation_type="Senior Orthodontists"):
    """Load ground truth landmarks for an image"""
    gt_path = os.path.join(
        dataset_path, "test", "Annotations", "Cephalometric Landmarks",
        annotation_type, f"{image_id}.json"
    )
    
    if not os.path.exists(gt_path):
        # Try train or valid
        for mode in ["train", "valid", "test"]:
            gt_path = os.path.join(
                dataset_path, mode, "Annotations", "Cephalometric Landmarks",
                annotation_type, f"{image_id}.json"
            )
            if os.path.exists(gt_path):
                break
    
    if not os.path.exists(gt_path):
        return None
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    landmarks = {}
    for lm in annotation.get('landmarks', []):
        symbol = lm['symbol']
        landmarks[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    return landmarks


def get_test_images(dataset_path="Aariz", num_images=5):
    """Get test images from dataset"""
    test_folder = os.path.join(dataset_path, "test", "Cephalograms")
    
    if not os.path.exists(test_folder):
        # Try other folders
        for mode in ["valid", "train"]:
            test_folder = os.path.join(dataset_path, mode, "Cephalograms")
            if os.path.exists(test_folder):
                break
    
    if not os.path.exists(test_folder):
        print(f"Test folder not found: {test_folder}")
        return []
    
    # Get image files
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(test_folder, ext)))
        image_files.extend(glob.glob(os.path.join(test_folder, ext.upper())))
    
    # Get unique image IDs
    image_ids = []
    for img_path in image_files[:num_images * 2]:  # Get more to ensure we have enough with GT
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        if img_id not in image_ids:
            # Check if GT exists
            gt = load_ground_truth(img_id, dataset_path)
            if gt is not None:
                image_ids.append(img_id)
                if len(image_ids) >= num_images:
                    break
    
    return image_ids


def preprocess_image(image_path, target_size=(512, 512)):
    """Preprocess image for model input"""
    img = Image.open(image_path).convert('RGB')
    original_size = img.size  # (width, height)
    
    # Resize
    img_resized = img.resize(target_size, Image.BILINEAR)
    
    # Convert to tensor
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor, original_size


def calculate_errors(pred_landmarks, gt_landmarks, pixel_size=0.1):
    """Calculate MRE and SDR for predictions"""
    errors_mm = []
    errors_pixels = []
    
    # Map 19 landmarks to 29 if needed
    for lm_19 in HEAD_19_LANDMARKS:
        if lm_19 not in pred_landmarks:
            continue
        
        # Map to 29 landmark name
        lm_29 = LANDMARK_MAPPING_19_TO_29.get(lm_19, None)
        if lm_29 is None or lm_29 not in gt_landmarks:
            continue
        
        pred = pred_landmarks[lm_19]
        gt = gt_landmarks[lm_29]
        
        if pred is None or gt is None:
            continue
        
        # Calculate error in pixels
        error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
        error_mm = error_px * pixel_size
        
        errors_pixels.append(error_px)
        errors_mm.append(error_mm)
    
    if len(errors_mm) == 0:
        return None
    
    mre_mm = np.mean(errors_mm)
    mre_px = np.mean(errors_pixels)
    
    # Calculate SDR (Success Detection Rate) at different thresholds
    sdr_2mm = sum(1 for e in errors_mm if e <= 2.0) / len(errors_mm) * 100
    sdr_2_5mm = sum(1 for e in errors_mm if e <= 2.5) / len(errors_mm) * 100
    sdr_3mm = sum(1 for e in errors_mm if e <= 3.0) / len(errors_mm) * 100
    sdr_4mm = sum(1 for e in errors_mm if e <= 4.0) / len(errors_mm) * 100
    
    return {
        'mre_mm': mre_mm,
        'mre_px': mre_px,
        'sdr_2mm': sdr_2mm,
        'sdr_2_5mm': sdr_2_5mm,
        'sdr_3mm': sdr_3mm,
        'sdr_4mm': sdr_4mm,
        'num_detected': len(errors_mm),
        'errors_mm': errors_mm
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test head_uda.pth model on Aariz dataset')
    parser.add_argument('--model_path', type=str, default='head_uda.pth', help='Path to head_uda.pth')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to Aariz dataset')
    parser.add_argument('--num_images', type=int, default=5, help='Number of test images')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint, state_dict = load_head_uda_model(args.model_path, device)
    
    print("\n" + "="*80)
    print("WARNING: The head_uda.pth model has a different architecture")
    print("(ResNet + Transformer + UDA) and cannot be directly used with")
    print("the current codebase without implementing the model architecture.")
    print("="*80)
    
    # Get test images
    print(f"\nGetting test images from {args.dataset_path}...")
    test_image_ids = get_test_images(args.dataset_path, args.num_images)
    
    if len(test_image_ids) == 0:
        print("No test images found with ground truth!")
        return
    
    print(f"Found {len(test_image_ids)} test images with ground truth")
    
    # Load ground truth for all images
    print("\nLoading ground truth...")
    all_gt = {}
    for img_id in test_image_ids:
        gt = load_ground_truth(img_id, args.dataset_path)
        if gt:
            all_gt[img_id] = gt
            print(f"  {img_id}: {len(gt)} landmarks")
    
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Model: {args.model_path}")
    print(f"Model architecture: ResNet + Transformer + UDA (19 landmarks)")
    print(f"Test images: {len(test_image_ids)}")
    print(f"\nTo use this model, you need to:")
    print("1. Implement the model architecture from UDA_Med_Landmark repository")
    print("2. Load the model weights")
    print("3. Run inference on test images")
    print("4. Map 19 landmarks to 29 Aariz landmarks")
    print("5. Compare with ground truth")
    print(f"\nThe model expects 19 landmarks: {HEAD_19_LANDMARKS}")
    print(f"Your dataset has 29 landmarks, so mapping is needed.")


if __name__ == "__main__":
    main()
















