"""
Training script for HRNet-based p1/p2 calibration point detector
Uses coordinate regression with HRNet backbone
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from model import HRNetP1P2Detector

# بدون استفاده از emojis برای جلوگیری از UnicodeEncodeError در Windows
print("=" * 60)
print("HRNet P1/P2 Calibration Point Detector - Training")
print("=" * 60)


class P1P2Dataset(Dataset):
    """Dataset for p1/p2 calibration points (1cm apart)"""
    
    def __init__(self, annotations_file, images_dir, transform=None, augment=False):
        """
        Args:
            annotations_file: JSON file with annotations (Aariz format)
            images_dir: Directory containing images
            transform: Image transforms
            augment: Whether to apply data augmentation
        """
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.augment = augment
        
        # Load annotations
        with open(annotations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Parse Aariz format annotations
        self.samples = []
        for item in data:
            if 'annotations' in item and item['annotations']:
                annotation = item['annotations'][0]  # First annotation
                
                if 'result' in annotation:
                    # Extract p1 and p2 from keypoint annotations
                    p1 = None
                    p2 = None
                    
                    for result_item in annotation['result']:
                        if result_item.get('type') == 'keypointlabels':
                            value = result_item.get('value', {})
                            labels = value.get('keypointlabels', [])
                            
                            if 'p1' in labels:
                                # Coordinates are in percentage (0-100)
                                p1 = {
                                    'x': value.get('x', 0) / 100.0,  # Normalize to [0, 1]
                                    'y': value.get('y', 0) / 100.0
                                }
                            elif 'p2' in labels:
                                p2 = {
                                    'x': value.get('x', 0) / 100.0,
                                    'y': value.get('y', 0) / 100.0
                                }
                    
                    # Only include if both p1 and p2 are found
                    if p1 and p2:
                        # Get image filename
                        # Format: /data/upload/6/8a75433a-cks2ip8fq2a0j0yufdfssbc09.png
                        # Actual file: cks2ip8fq2a0j0yufdfssbc09.png (without prefix)
                        image_url = item['data'].get('image', '')
                        image_filename = image_url.split('/')[-1]  # Get last part
                        
                        # Remove UUID prefix if exists (keep only the part after last "-")
                        # "8a75433a-cks2ip8fq2a0j0yufdfssbc09.png" -> "cks2ip8fq2a0j0yufdfssbc09.png"
                        if '-' in image_filename:
                            parts = image_filename.split('-')
                            # Check if first part is UUID-like (8 hex chars)
                            if len(parts[0]) == 8 and all(c in '0123456789abcdef' for c in parts[0].lower()):
                                image_filename = '-'.join(parts[1:])  # Remove UUID prefix
                        
                        image_path = self.images_dir / image_filename
                        
                        if image_path.exists():
                            self.samples.append({
                                'image_path': str(image_path),
                                'p1': p1,
                                'p2': p2
                            })
                        else:
                            # Try without prefix removal (in case file has full name)
                            alt_filename = image_url.split('/')[-1]
                            alt_path = self.images_dir / alt_filename
                            if alt_path.exists():
                                self.samples.append({
                                    'image_path': str(alt_path),
                                    'p1': p1,
                                    'p2': p2
                                })
                            else:
                                print(f"  [WARN] Image not found: {image_filename}")
                                print(f"        Also tried: {alt_filename}")
        
        print(f"Loaded {len(self.samples)} samples with p1/p2 annotations")
        
        # Data augmentation transforms
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.3, contrast=0.3),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # FIXED: Disable augmentation for now - it causes landmark mismatch!
        # Augmentation transforms the image but doesn't update landmark coordinates
        # TODO: Implement proper augmentation with landmark transformation
        
        # Apply transforms (resize, normalize, etc.)
        if self.transform:
            image = self.transform(image)
        
        # Ground truth coordinates (already normalized to [0, 1])
        # Order: [p1_x, p1_y, p2_x, p2_y]
        landmarks = torch.tensor([
            sample['p1']['x'], sample['p1']['y'],
            sample['p2']['x'], sample['p2']['y']
        ], dtype=torch.float32)
        
        return image, landmarks


def train_hrnet_p1_p2_model(
    annotations_file,
    images_dir,
    output_dir='models',
    hrnet_variant='hrnet_w18',
    image_size=512,
    batch_size=4,
    num_epochs=200,
    learning_rate=0.003,  # Increased from 0.001 for faster learning
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train HRNet-based p1/p2 detector
    
    Args:
        annotations_file: Path to JSON annotations
        images_dir: Directory containing images
        output_dir: Where to save models
        hrnet_variant: HRNet variant (hrnet_w18, hrnet_w32, hrnet_w48)
        image_size: Input image size
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Initial learning rate
        device: Device to train on
    """
    
    print(f"\n[CONFIG] Training Configuration")
    print(f"  - HRNet Variant: {hrnet_variant}")
    print(f"  - Image Size: {image_size}x{image_size}")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Device: {device}")
    print(f"  - Annotations: {annotations_file}")
    print(f"  - Images: {images_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Image transforms
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (80/20 train/val split)
    full_dataset = P1P2Dataset(
        annotations_file=annotations_file,
        images_dir=images_dir,
        transform=transform,
        augment=True  # Augmentation for training
    )
    
    # Split dataset
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    print(f"\n[DATA] Dataset split:")
    print(f"  - Training: {train_size} samples")
    print(f"  - Validation: {val_size} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    print(f"\n[MODEL] Creating HRNet model...")
    model = HRNetP1P2Detector(
        num_landmarks=2,
        hrnet_variant=hrnet_variant,
        pretrained=True
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    # Loss function (MSE for coordinate regression)
    criterion = nn.MSELoss()
    
    # Optimizer (Adam with weight decay)
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-5
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,  # Reduced from 20 for faster LR decay
        verbose=True,
        min_lr=1e-6
    )
    
    # Training loop
    print(f"\n[TRAINING] Starting training...")
    print("=" * 60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 30  # Reduced from 50 for faster early stopping
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for images, landmarks in train_pbar:
            images = images.to(device)
            landmarks = landmarks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f"{loss.item():.6f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        pixel_errors = []
        
        with torch.no_grad():
            for images, landmarks in val_loader:
                images = images.to(device)
                landmarks = landmarks.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()
                
                # Calculate pixel error correctly for each landmark (p1 and p2)
                pred_pixels = outputs.cpu().numpy() * image_size  # shape: (batch, 4)
                gt_pixels = landmarks.cpu().numpy() * image_size   # shape: (batch, 4)
                
                # Calculate error for p1 (x, y) and p2 (x, y) separately
                p1_errors = np.sqrt((pred_pixels[:, 0] - gt_pixels[:, 0])**2 + 
                                   (pred_pixels[:, 1] - gt_pixels[:, 1])**2)
                p2_errors = np.sqrt((pred_pixels[:, 2] - gt_pixels[:, 2])**2 + 
                                   (pred_pixels[:, 3] - gt_pixels[:, 3])**2)
                
                # Average error for both landmarks
                avg_errors = (p1_errors + p2_errors) / 2
                pixel_errors.extend(avg_errors.tolist())
        
        avg_val_loss = val_loss / len(val_loader)
        avg_pixel_error = np.mean(pixel_errors)
        
        # Update scheduler
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Avg Pixel Error: {avg_pixel_error:.2f} px")
        print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            checkpoint_path = os.path.join(output_dir, f'hrnet_p1p2_best_{hrnet_variant}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'pixel_error': avg_pixel_error,
                'hrnet_variant': hrnet_variant,
                'image_size': image_size,
                'num_landmarks': 2,
            }, checkpoint_path)
            
            print(f"  >>> Best model saved! (Val Loss: {best_val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  >>> No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        # Early stopping
        if patience_counter >= early_stop_patience:
            print(f"\n[EARLY STOP] No improvement for {early_stop_patience} epochs. Stopping...")
            break
        
        print("-" * 60)
    
    print("\n[COMPLETE] Training finished!")
    print(f"  Best validation loss: {best_val_loss:.6f}")
    print(f"  Model saved to: {checkpoint_path}")
    
    return model


if __name__ == '__main__':
    # Dataset paths
    script_dir = Path(__file__).parent
    os.chdir(script_dir)  # Change to script directory
    
    # Use the new annotations file created by p1_p2_annotator.py
    annotations_file = 'annotations_p1_p2.json'
    images_dir = 'Aariz/train/Cephalograms'
    
    # Check if files exist
    if not Path(annotations_file).exists():
        print(f"ERROR: Annotations file not found: {annotations_file}")
        print(f"Current directory: {Path.cwd()}")
        print(f"\nPlease create annotations first using:")
        print(f"  annotate_p1_p2.bat")
        sys.exit(1)
    
    if not Path(images_dir).exists():
        print(f"ERROR: Images directory not found: {images_dir}")
        print(f"Current directory: {Path.cwd()}")
        sys.exit(1)
    
    print("Starting HRNet p1/p2 training...")
    print(f"Annotations: {Path(annotations_file).absolute()}")
    print(f"Images: {Path(images_dir).absolute()}")
    
    # Train model
    # You can choose: 'hrnet_w18' (faster), 'hrnet_w32' (balanced), 'hrnet_w48' (best accuracy)
    # Optimized for RTX 3070 Ti (8GB VRAM)
    model = train_hrnet_p1_p2_model(
        annotations_file=annotations_file,
        images_dir=images_dir,
        output_dir='models',
        hrnet_variant='hrnet_w18',  # Good balance of speed and accuracy
        image_size=768,  # Higher resolution for better accuracy
        batch_size=2,  # Optimized for 3070 Ti with 768px images
        num_epochs=200,
        learning_rate=0.003,  # Increased for faster learning
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\n[DONE] HRNet training complete!")

