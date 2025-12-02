"""
Fine-tune CLdetection2023 model to add p1 and p2 calibration landmarks.
Uses the pretrained CLdetection2023 backbone and adds a new head for p1/p2.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import sys

# Import from existing files
sys.path.append(str(Path(__file__).parent))
from model import CephalometricLandmarkDetector

# Try to import MMPose for CLdetection2023
MMPOSE_AVAILABLE = False
CLDETECTION_REPO = None

# Try to find CLdetection2023 repository
possible_paths = [
    Path(__file__).parent.parent / "CLdetection2023",
    Path(__file__).parent / "CLdetection2023",
    Path("CLdetection2023"),
]

for path in possible_paths:
    if path.exists():
        CLDETECTION_REPO = str(path)
        break

if CLDETECTION_REPO:
    try:
        sys.path.insert(0, CLDETECTION_REPO)
        from mmpose.apis import init_model
        from mmpose.utils import register_all_modules
        register_all_modules()
        MMPOSE_AVAILABLE = True
        print(f"[OK] MMPose available. CLdetection2023 repo: {CLDETECTION_REPO}")
    except (ImportError, AssertionError, Exception) as e:
        print(f"[WARNING] MMPose not available: {e}")
        print(f"[WARNING] Will use alternative approach (ResNet18).")
        MMPOSE_AVAILABLE = False


class CLDetectionBackboneWrapper(nn.Module):
    """
    Wrapper to extract features from CLdetection2023 model backbone.
    """
    def __init__(self, cldetection_model_path, config_path=None, device='cuda'):
        super().__init__()
        self.device = device
        self.backbone = None
        self.feature_dim = None
        
        if MMPOSE_AVAILABLE and cldetection_model_path and os.path.exists(cldetection_model_path):
            try:
                # Try to load CLdetection2023 model
                if config_path is None:
                    config_path = os.path.join(CLDETECTION_REPO, "configs", "CLdetection2023", "srpose_s2.py")
                
                if os.path.exists(config_path):
                    print(f"Loading CLdetection2023 model from {cldetection_model_path}...")
                    self.cldetection_model = init_model(config_path, cldetection_model_path, device=device)
                    
                    # Extract backbone
                    if hasattr(self.cldetection_model, 'backbone'):
                        self.backbone = self.cldetection_model.backbone
                        # Freeze backbone initially
                        for param in self.backbone.parameters():
                            param.requires_grad = False
                        
                        # Get feature dimension (CLdetection2023 uses 1024x1024)
                        with torch.no_grad():
                            dummy_input = torch.randn(1, 3, 1024, 1024).to(device)
                            features = self.backbone(dummy_input)
                            if isinstance(features, (list, tuple)):
                                features = features[-1]
                            if isinstance(features, dict):
                                features = features.get('feat', list(features.values())[0])
                            # Global average pooling to get feature vector
                            if len(features.shape) == 4:
                                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                                features = features.view(features.size(0), -1)
                            self.feature_dim = features.shape[1]
                            print(f"[OK] CLdetection2023 backbone loaded. Feature dim: {self.feature_dim}")
                    else:
                        print("[WARNING] Could not extract backbone from CLdetection2023 model")
                        self._use_fallback()
                else:
                    print(f"[WARNING] Config file not found: {config_path}")
                    self._use_fallback()
            except Exception as e:
                print(f"[WARNING] Error loading CLdetection2023 model: {e}")
                self._use_fallback()
        else:
            self._use_fallback()
    
    def _use_fallback(self):
        """Use ResNet18 as fallback backbone"""
        print("Using ResNet18 as fallback backbone...")
        import torchvision.models as models
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512
        # Freeze backbone initially
        for param in self.backbone.parameters():
            param.requires_grad = False
        print(f"[OK] ResNet18 backbone loaded. Feature dim: {self.feature_dim}")
    
    def forward(self, x):
        """Extract features from backbone"""
        if self.backbone is None:
            raise RuntimeError("Backbone not initialized")
        
        features = self.backbone(x)
        
        # Handle different output formats
        if isinstance(features, (list, tuple)):
            features = features[-1]
        if isinstance(features, dict):
            features = features.get('feat', list(features.values())[0])
        
        # Global average pooling if needed
        if len(features.shape) == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features


class P1P2ModelWithCLDetectionBackbone(nn.Module):
    """
    Model that uses CLdetection2023 backbone and adds p1/p2 regression head.
    """
    def __init__(self, cldetection_model_path, config_path=None, device='cuda', freeze_backbone=True):
        super().__init__()
        
        # Load backbone
        self.backbone_wrapper = CLDetectionBackboneWrapper(
            cldetection_model_path, config_path, device
        )
        self.backbone_wrapper = self.backbone_wrapper.to(device)
        
        if freeze_backbone:
            for param in self.backbone_wrapper.parameters():
                param.requires_grad = False
        
        feature_dim = self.backbone_wrapper.feature_dim
        
        # Regression head for p1/p2 (4 coordinates: x1, y1, x2, y2)
        self.regressor = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 4),  # 4 outputs: p1_x, p1_y, p2_x, p2_y
            nn.Sigmoid()  # Output in [0, 1] range
        )
        
        # Initialize regression head with better initialization
        for module in self.regressor:
            if isinstance(module, nn.Linear):
                # Use He initialization for ReLU
                nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                # Scale down weights to prevent saturation
                module.weight.data *= 0.1
                if module.bias is not None:
                    # Initialize bias to center (0.5) but with small random variation
                    nn.init.uniform_(module.bias, 0.4, 0.6)  # Start near center with variation
    
    def forward(self, x):
        # Extract features
        features = self.backbone_wrapper(x)
        
        # Regress coordinates
        coords = self.regressor(features)
        
        return coords
    
    def unfreeze_backbone(self, unfreeze_layers=None):
        """
        Unfreeze backbone for fine-tuning.
        unfreeze_layers: list of layer names to unfreeze, or None to unfreeze all
        """
        for param in self.backbone_wrapper.parameters():
            param.requires_grad = True
        print("[OK] Backbone unfrozen for fine-tuning")


class P1P2DatasetFromJSON(Dataset):
    """Dataset for p1 and p2 landmarks from annotations_p1_p2.json file."""
    
    def __init__(self, image_dir, annotations_json, image_size=512, augment=False):
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augment = augment
        
        # Load annotations from JSON file
        with open(annotations_json, 'r', encoding='utf-8') as f:
            self.annotations_data = json.load(f)
        
        # Extract valid annotations (must have both p1 and p2)
        self.valid_annotations = []
        for item in self.annotations_data:
            filename = item.get('file_upload', '')
            if not filename:
                continue
            
            # Extract p1 and p2 from annotations
            p1 = None
            p2 = None
            
            if 'annotations' in item and len(item['annotations']) > 0:
                result = item['annotations'][0].get('result', [])
                for r in result:
                    if r.get('type') == 'keypointlabels':
                        value = r.get('value', {})
                        labels = value.get('keypointlabels', [])
                        x = value.get('x', 0)  # Percentage (0-100)
                        y = value.get('y', 0)   # Percentage (0-100)
                        
                        if 'p1' in labels:
                            p1 = {'x': x, 'y': y}
                        elif 'p2' in labels:
                            p2 = {'x': x, 'y': y}
            
            # Only add if both p1 and p2 are present
            if p1 and p2:
                self.valid_annotations.append({
                    'filename': filename,
                    'p1': p1,
                    'p2': p2
                })
        
        print(f"Loaded {len(self.valid_annotations)} valid annotations from {annotations_json}")
        
    def __len__(self):
        return len(self.valid_annotations)
    
    def __getitem__(self, idx):
        annotation = self.valid_annotations[idx]
        filename = annotation['filename']
        
        # Extract image ID (filename without extension)
        image_id = Path(filename).stem
        
        # Load image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            potential_path = self.image_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path is None:
            # Try with full filename
            potential_path = self.image_dir / filename
            if potential_path.exists():
                image_path = potential_path
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for {filename} (tried: {image_id})")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Convert percentage coordinates to pixel coordinates
        # p1 and p2 are in percentage (0-100), convert to normalized (0-1)
        p1_x_percent = annotation['p1']['x'] / 100.0  # 0-1
        p1_y_percent = annotation['p1']['y'] / 100.0  # 0-1
        p2_x_percent = annotation['p2']['x'] / 100.0  # 0-1
        p2_y_percent = annotation['p2']['y'] / 100.0  # 0-1
        
        # Convert to pixel coordinates in resized image
        p1_x = p1_x_percent * self.image_size
        p1_y = p1_y_percent * self.image_size
        p2_x = p2_x_percent * self.image_size
        p2_y = p2_y_percent * self.image_size
        
        # Create landmark tensor: [x1, y1, x2, y2] normalized to [0, 1]
        landmarks_tensor = torch.FloatTensor([p1_x, p1_y, p2_x, p2_y]) / self.image_size
        
        # Normalize image (same as CLdetection2023 preprocessing)
        # CLdetection2023 uses custom normalization: mean=[121.25,121.25,121.25], std=[76.5, 76.5, 76.5]
        image_tensor = torch.FloatTensor(image_resized).permute(2, 0, 1)  # [C, H, W], values in [0, 255]
        # Apply CLdetection2023 normalization
        mean = torch.tensor([121.25, 121.25, 121.25]).view(3, 1, 1)
        std = torch.tensor([76.5, 76.5, 76.5]).view(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        
        return image_tensor, landmarks_tensor, image_id


def train_p1_p2_with_cldetection(
    cldetection_model_path,
    annotations_json='annotations_p1_p2.json',
    image_dir='Aariz/train/Cephalograms',
    image_size=1024,
    batch_size=4,
    num_epochs=100,
    learning_rate=0.001,
    device='cuda',
    train_split=0.8,
    freeze_backbone=True,
    unfreeze_after_epochs=None,
    config_path=None
):
    """
    Fine-tune CLdetection2023 model to add p1/p2 detection.
    
    Args:
        cldetection_model_path: Path to CLdetection2023 pretrained model
        unfreeze_after_epochs: After how many epochs to unfreeze backbone (None = never)
    """
    
    print("="*60)
    print("Fine-tuning CLdetection2023 for P1/P2 Detection")
    print("="*60)
    
    # Create dataset
    full_dataset = P1P2DatasetFromJSON(image_dir, annotations_json, image_size=image_size)
    
    if len(full_dataset) == 0:
        raise ValueError(f"No valid annotations found in {annotations_json}")
    
    # Split into train and validation
    indices = list(range(len(full_dataset)))
    random.seed(42)
    random.shuffle(indices)
    
    split_idx = int(len(indices) * train_split)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"\nDataset Info:")
    print(f"   Total images: {len(full_dataset)}")
    print(f"   Training images: {len(train_dataset)}")
    print(f"   Validation images: {len(val_dataset)}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {image_size}x{image_size}")
    print(f"   Landmarks: 2 (p1, p2)")
    print(f"   Device: {device}")
    print(f"   Backbone frozen: {freeze_backbone}\n")
    
    # Create model
    model = P1P2ModelWithCLDetectionBackbone(
        cldetection_model_path=cldetection_model_path,
        config_path=config_path,
        device=device,
        freeze_backbone=freeze_backbone
    )
    model = model.to(device)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Loss function - Use SmoothL1Loss (Huber Loss) which is more robust than MSE
    # Use smaller beta to be more like L2 for small errors (better gradient flow)
    criterion = nn.SmoothL1Loss(beta=0.1)  # beta=0.1: L2 for |x| < 0.1, L1 for larger
    
    # Use different learning rates for backbone and head
    # Higher LR for head since it's randomly initialized
    if freeze_backbone:
        optimizer = optim.Adam(model.regressor.parameters(), lr=learning_rate * 5.0, weight_decay=1e-4)
    else:
        # Different LR for backbone and head
        optimizer = optim.Adam([
            {'params': model.backbone_wrapper.parameters(), 'lr': learning_rate * 0.1, 'weight_decay': 1e-4},  # Lower LR for backbone
            {'params': model.regressor.parameters(), 'lr': learning_rate * 5.0, 'weight_decay': 1e-4}  # Higher LR for head
        ])
    
    # More aggressive scheduler - reduce LR faster if no improvement
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    
    # Training loop
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(num_epochs):
        # Unfreeze backbone if specified
        if unfreeze_after_epochs and epoch == unfreeze_after_epochs and freeze_backbone:
            print(f"\n[Epoch {epoch+1}] Unfreezing backbone for fine-tuning...")
            model.unfreeze_backbone()
            freeze_backbone = False
            # Update optimizer to include backbone parameters
            optimizer = optim.Adam([
                {'params': model.backbone_wrapper.parameters(), 'lr': learning_rate * 0.1, 'weight_decay': 1e-4},
                {'params': model.regressor.parameters(), 'lr': learning_rate * 5.0, 'weight_decay': 1e-4}
            ])
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
            )
        
        # Training phase
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, landmarks, _ in pbar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            
            # Forward
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_pixel_errors = []
        all_pixel_errors_p1 = []
        all_pixel_errors_p2 = []
        
        with torch.no_grad():
            for images, landmarks, _ in val_loader:
                images = images.to(device)
                landmarks = landmarks.to(device)
                outputs = model(images)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()
                
                # Calculate pixel errors for metrics
                # Outputs and landmarks are normalized [0, 1], convert to pixel coordinates
                outputs_px = outputs.cpu().numpy() * image_size  # [B, 4] -> [p1_x, p1_y, p2_x, p2_y]
                landmarks_px = landmarks.cpu().numpy() * image_size
                
                # Calculate radial error for each landmark
                for i in range(outputs_px.shape[0]):
                    # P1 error
                    p1_pred = outputs_px[i, :2]
                    p1_gt = landmarks_px[i, :2]
                    error_p1 = np.sqrt(np.sum((p1_pred - p1_gt) ** 2))
                    all_pixel_errors_p1.append(error_p1)
                    
                    # P2 error
                    p2_pred = outputs_px[i, 2:]
                    p2_gt = landmarks_px[i, 2:]
                    error_p2 = np.sqrt(np.sum((p2_pred - p2_gt) ** 2))
                    all_pixel_errors_p2.append(error_p2)
                    
                    # Average error
                    avg_error = (error_p1 + error_p2) / 2.0
                    all_pixel_errors.append(avg_error)
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate metrics
        mre_p1 = np.mean(all_pixel_errors_p1) if all_pixel_errors_p1 else 0.0
        mre_p2 = np.mean(all_pixel_errors_p2) if all_pixel_errors_p2 else 0.0
        mre_avg = np.mean(all_pixel_errors) if all_pixel_errors else 0.0
        
        # SDR (Success Detection Rate) at different pixel thresholds
        # Common thresholds: 2mm, 2.5mm, 3mm, 4mm
        # For 1024x1024 images, approximate: 1mm ≈ 2-3 pixels (depends on image size)
        # Using pixel thresholds: 5px, 10px, 20px, 30px, 50px
        thresholds_px = [5, 10, 20, 30, 50]
        sdr_p1 = {}
        sdr_p2 = {}
        sdr_avg = {}
        
        for threshold in thresholds_px:
            sdr_p1[threshold] = (np.array(all_pixel_errors_p1) <= threshold).sum() / len(all_pixel_errors_p1) * 100 if all_pixel_errors_p1 else 0.0
            sdr_p2[threshold] = (np.array(all_pixel_errors_p2) <= threshold).sum() / len(all_pixel_errors_p2) * 100 if all_pixel_errors_p2 else 0.0
            sdr_avg[threshold] = (np.array(all_pixel_errors) <= threshold).sum() / len(all_pixel_errors) * 100 if all_pixel_errors else 0.0
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Metrics:")
        print(f"    MRE (px): P1={mre_p1:.2f}, P2={mre_p2:.2f}, Avg={mre_avg:.2f}")
        print(f"    SDR @ 5px:  P1={sdr_p1[5]:.1f}%, P2={sdr_p2[5]:.1f}%, Avg={sdr_avg[5]:.1f}%")
        print(f"    SDR @ 10px: P1={sdr_p1[10]:.1f}%, P2={sdr_p2[10]:.1f}%, Avg={sdr_avg[10]:.1f}%")
        print(f"    SDR @ 20px: P1={sdr_p1[20]:.1f}%, P2={sdr_p2[20]:.1f}%, Avg={sdr_avg[20]:.1f}%")
        print(f"    SDR @ 30px: P1={sdr_p1[30]:.1f}%, P2={sdr_p2[30]:.1f}%, Avg={sdr_avg[30]:.1f}%")
        print(f"    SDR @ 50px: P1={sdr_p1[50]:.1f}%, P2={sdr_p2[50]:.1f}%, Avg={sdr_avg[50]:.1f}%")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss = avg_train_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'num_landmarks': 2,
                'image_size': image_size,
                'num_train': len(train_dataset),
                'num_val': len(val_dataset),
                'cldetection_model_path': cldetection_model_path,
                'backbone_frozen': freeze_backbone,
                # Metrics
                'mre_p1': float(mre_p1),
                'mre_p2': float(mre_p2),
                'mre_avg': float(mre_avg),
                'sdr_p1': {k: float(v) for k, v in sdr_p1.items()},
                'sdr_p2': {k: float(v) for k, v in sdr_p2.items()},
                'sdr_avg': {k: float(v) for k, v in sdr_avg.items()},
            }, 'checkpoint_p1_p2_cldetection.pth')
            print(f"  [SAVED] Best model (val_loss: {avg_val_loss:.6f}, MRE: {mre_avg:.2f}px)")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Best model saved as: checkpoint_p1_p2_cldetection.pth")
    print(f"[RESULT] Best train loss: {best_loss:.6f}")
    print(f"[RESULT] Best val loss: {best_val_loss:.6f}")
    
    # Load and print final metrics from checkpoint
    try:
        checkpoint = torch.load('checkpoint_p1_p2_cldetection.pth', map_location='cpu', weights_only=False)
        if 'mre_avg' in checkpoint:
            print(f"\n[FINAL METRICS]")
            print(f"  MRE: P1={checkpoint.get('mre_p1', 0):.2f}px, P2={checkpoint.get('mre_p2', 0):.2f}px, Avg={checkpoint.get('mre_avg', 0):.2f}px")
            if 'sdr_avg' in checkpoint:
                sdr_avg = checkpoint['sdr_avg']
                print(f"  SDR @ 5px:  {sdr_avg.get(5, 0):.1f}%")
                print(f"  SDR @ 10px: {sdr_avg.get(10, 0):.1f}%")
                print(f"  SDR @ 20px: {sdr_avg.get(20, 0):.1f}%")
                print(f"  SDR @ 30px: {sdr_avg.get(30, 0):.1f}%")
                print(f"  SDR @ 50px: {sdr_avg.get(50, 0):.1f}%")
    except:
        pass


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fine-tune CLdetection2023 for P1/P2 detection')
    parser.add_argument('--cldetection-model', type=str, 
                       default=r'C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth',
                       help='Path to CLdetection2023 pretrained model')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to CLdetection2023 config file (auto-detected if not provided)')
    parser.add_argument('--annotations', type=str, default='annotations_p1_p2.json',
                       help='Path to annotations JSON file')
    parser.add_argument('--image-dir', type=str, default='Aariz/train/Cephalograms',
                       help='Directory containing images')
    parser.add_argument('--image-size', type=int, default=1024,
                       help='Image size for training (CLdetection2023 uses 1024x1024)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size (reduced for 1024x1024 images)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/validation split ratio')
    parser.add_argument('--unfreeze-after', type=int, default=None,
                       help='Unfreeze backbone after N epochs (None = keep frozen)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cuda/cpu/auto)')
    
    args = parser.parse_args()
    
    # Check CUDA
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"[DEVICE] Using: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Train
    train_p1_p2_with_cldetection(
        cldetection_model_path=args.cldetection_model,
        annotations_json=args.annotations,
        image_dir=args.image_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        train_split=args.train_split,
        freeze_backbone=True,
        unfreeze_after_epochs=args.unfreeze_after,
        config_path=args.config
    )

