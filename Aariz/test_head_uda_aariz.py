"""
Test head_uda.pth model on Aariz dataset using UDA_Med_Landmark repository
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'UDA_Med_Landmark', 'lib'))

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import json
import glob
from tqdm import tqdm
import importlib

# Import from UDA repository
from networks import TF_resnet
from functions import val_model, compute_sdr
import data_utils

# Standard 19 landmarks for head detection
HEAD_19_LANDMARKS = [
    "S", "N", "Or", "A", "B", "PNS", "ANS", "U1", "L1", "Me",
    "U6", "L6", "Go", "Pog", "Gn", "Ar", "Co", "Po", "R"
]

# Mapping from 19 landmarks to 29 Aariz landmarks
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

# Reverse mapping
LANDMARK_MAPPING_29_TO_19 = {v: k for k, v in LANDMARK_MAPPING_19_TO_29.items()}


class Config:
    """Configuration for head_uda model"""
    def __init__(self):
        self.task = 'head'
        self.semi_iter = 5
        self.curriculum = [0.2, 0.4, 0.6, 0.8, 1]
        self.origin_size = (2400, 1935)
        self.input_size = (800, 640)
        self.phy_dist = 0.1
        self.batch_size = 1
        self.init_lr = 0.0002
        self.num_epochs = 720
        self.decay_steps = [480, 640]
        self.backbone = 'resnet50'
        self.pretrained = True
        self.criterion_cls = 'l2'
        self.criterion_reg = 'l1'
        self.cls_loss_weight = 100
        self.xy_loss_weight = 0.02
        self.domain_loss_weight = 0.01
        self.gt_sigma = 2
        self.num_lms = 19
        self.use_gpu = True
        self.gpu_id = 0
        self.tf_dim = 256
        self.tf_en_num = 0
        self.tf_de_num = 3


def load_ground_truth_aariz(image_id, dataset_path="Aariz", annotation_type="Senior Orthodontists"):
    """Load ground truth landmarks from Aariz dataset"""
    for mode in ["test", "valid", "train"]:
        gt_path = os.path.join(
            dataset_path, mode, "Annotations", "Cephalometric Landmarks",
            annotation_type, f"{image_id}.json"
        )
        if os.path.exists(gt_path):
            break
    
    if not os.path.exists(gt_path):
        return None, None
    
    with open(gt_path, 'r', encoding='utf-8') as f:
        annotation = json.load(f)
    
    # Get pixel size
    pixel_size = 0.1  # default
    csv_path = os.path.join(dataset_path, "cephalogram_machine_mappings.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        df = pd.read_csv(csv_path)
        row = df[df['cephalogram_id'] == image_id]
        if len(row) > 0:
            pixel_size = row.iloc[0]['pixel_size']
    
    # Get original image size
    for mode in ["test", "valid", "train"]:
        img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(dataset_path, mode, "Cephalograms", f"{image_id}.jpg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            orig_size = img.size  # (width, height)
            break
    
    landmarks_29 = {}
    for lm in annotation.get('landmarks', []):
        symbol = lm['symbol']
        landmarks_29[symbol] = {
            'x': float(lm['value']['x']),
            'y': float(lm['value']['y'])
        }
    
    # Map to 19 landmarks
    landmarks_19 = {}
    target_array = []
    for lm_19 in HEAD_19_LANDMARKS:
        lm_29 = LANDMARK_MAPPING_19_TO_29.get(lm_19, None)
        if lm_29 and lm_29 in landmarks_29:
            lm_data = landmarks_29[lm_29]
            # Normalize coordinates (0-1)
            x_norm = lm_data['x'] / orig_size[0]
            y_norm = lm_data['y'] / orig_size[1]
            landmarks_19[lm_19] = {'x': lm_data['x'], 'y': lm_data['y'], 
                                   'x_norm': x_norm, 'y_norm': y_norm}
            target_array.extend([x_norm, y_norm])
        else:
            # Invalid landmark
            target_array.extend([-1.0, -1.0])
    
    target_array = np.array(target_array, dtype=np.float32)
    
    return target_array, pixel_size, orig_size


def get_aariz_test_images(dataset_path="Aariz", num_images=5):
    """Get test images from Aariz dataset"""
    test_images = []
    
    for mode in ["test", "valid", "train"]:
        images_folder = os.path.join(dataset_path, mode, "Cephalograms")
        if not os.path.exists(images_folder):
            continue
        
        # Get image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
            image_files.extend(glob.glob(os.path.join(images_folder, ext)))
            image_files.extend(glob.glob(os.path.join(images_folder, ext.upper())))
        
        for img_path in image_files[:num_images * 2]:
            img_id = os.path.splitext(os.path.basename(img_path))[0]
            
            # Check if GT exists
            target, pixel_size, orig_size = load_ground_truth_aariz(img_id, dataset_path)
            if target is not None:
                test_images.append({
                    'image_path': img_path,
                    'image_id': img_id,
                    'target': target,
                    'pixel_size': pixel_size,
                    'orig_size': orig_size
                })
                if len(test_images) >= num_images:
                    break
        
        if len(test_images) >= num_images:
            break
    
    return test_images


def create_aariz_dataset(test_images, cfg, transform):
    """Create a dataset compatible with UDA repository format"""
    class AarizDatasetAdapter(torch.utils.data.Dataset):
        def __init__(self, test_images, transform):
            self.test_images = test_images
            self.transform = transform
        
        def __getitem__(self, idx):
            item = self.test_images[idx]
            img = Image.open(item['image_path']).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            target = torch.from_numpy(item['target']).float()
            orig_size = torch.from_numpy(np.array(item['orig_size'])).float()
            phy_dist = torch.from_numpy(np.array([item['pixel_size']])).float()
            
            return img, target, orig_size, phy_dist
        
        def __len__(self):
            return len(self.test_images)
    
    return AarizDatasetAdapter(test_images, transform)


def test_head_uda_on_aariz(model_path, aariz_path="Aariz", num_images=5):
    """Test head_uda.pth model on Aariz dataset"""
    print("="*80)
    print("Testing head_uda.pth on Aariz Dataset")
    print("="*80)
    
    # Configuration
    cfg = Config()
    # Resize images to model input size
    cfg.input_size = (800, 640)  # (height, width) as expected by model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input size: {cfg.input_size}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    resnet50 = models.resnet50(pretrained=False)
    net = TF_resnet(resnet50, cfg)
    net = net.to(device)
    
    state_dict = torch.load(model_path, map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    print("Model loaded successfully!")
    
    # Normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Get test images
    print(f"\nLoading test images from {aariz_path}...")
    test_images = get_aariz_test_images(aariz_path, num_images)
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    # Create dataset and dataloader
    # Resize to model input size
    dataset = create_aariz_dataset(test_images, cfg, transforms.Compose([
        transforms.Resize(cfg.input_size),  # (800, 640)
        transforms.ToTensor(),
        normalize
    ]))
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=False, 
        num_workers=0, pin_memory=False
    )
    
    # Test model
    print("\nRunning inference...")
    all_distances = []
    all_mres = []
    
    with torch.no_grad():
        for batch_idx, (images, targets, orig_sizes, phy_dists) in enumerate(tqdm(loader)):
            images = images.to(device)
            targets = targets.to(device)
            orig_sizes = orig_sizes.to(device)
            phy_dists = phy_dists.to(device)
            
            # Forward pass
            outputs_map, outputs_x, outputs_y, outputs_coord, _ = net(images, 0)
            
            # Process outputs (similar to val_model function)
            batch_size = images.size(0)
            for b in range(batch_size):
                target = targets[b].cpu().numpy().reshape(-1, 2)
                output_coord = outputs_coord[b].cpu().numpy()
                orig_size = orig_sizes[b].cpu().numpy()
                phy_dist = phy_dists[b].cpu().numpy()[0]
                
                # Scale predictions to original image size
                output_coord_scaled = output_coord.copy()
                output_coord_scaled[:, 0] *= orig_size[0]
                output_coord_scaled[:, 1] *= orig_size[1]
                
                target_scaled = target.copy()
                target_scaled[:, 0] *= orig_size[0]
                target_scaled[:, 1] *= orig_size[1]
                
                # Calculate distances (only for valid landmarks)
                valid_mask = (target[:, 0] >= 0) & (target[:, 1] >= 0)
                if valid_mask.sum() > 0:
                    distances = np.sqrt(np.sum((output_coord_scaled[valid_mask] - target_scaled[valid_mask])**2, axis=1))
                    distances_mm = distances * phy_dist
                    
                    all_distances.append(distances_mm)
                    mre = np.mean(distances_mm)
                    all_mres.append(mre)
                    
                    print(f"\nImage {test_images[batch_size * batch_idx + b]['image_id']}:")
                    print(f"  MRE: {mre:.2f} mm")
                    print(f"  Valid landmarks: {valid_mask.sum()}/19")
    
    # Calculate overall metrics
    if len(all_mres) > 0:
        print("\n" + "="*80)
        print("OVERALL RESULTS")
        print("="*80)
        print(f"Mean MRE: {np.mean(all_mres):.2f} mm")
        print(f"Std MRE: {np.std(all_mres):.2f} mm")
        
        # Calculate SDR
        all_distances_flat = np.concatenate(all_distances)
        sdr_2mm = (all_distances_flat <= 2.0).sum() / len(all_distances_flat) * 100
        sdr_2_5mm = (all_distances_flat <= 2.5).sum() / len(all_distances_flat) * 100
        sdr_3mm = (all_distances_flat <= 3.0).sum() / len(all_distances_flat) * 100
        sdr_4mm = (all_distances_flat <= 4.0).sum() / len(all_distances_flat) * 100
        
        print(f"\nSDR @ 2mm: {sdr_2mm:.2f}%")
        print(f"SDR @ 2.5mm: {sdr_2_5mm:.2f}%")
        print(f"SDR @ 3mm: {sdr_3mm:.2f}%")
        print(f"SDR @ 4mm: {sdr_4mm:.2f}%")
        print("="*80)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Test head_uda.pth on Aariz dataset')
    parser.add_argument('--model_path', type=str, default='../UDA_Med_Landmark/weights/head_uda.pth',
                       help='Path to head_uda.pth model')
    parser.add_argument('--aariz_path', type=str, default='Aariz',
                       help='Path to Aariz dataset')
    parser.add_argument('--num_images', type=int, default=5,
                       help='Number of test images')
    
    args = parser.parse_args()
    
    test_head_uda_on_aariz(args.model_path, args.aariz_path, args.num_images)

