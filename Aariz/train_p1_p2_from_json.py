"""
Train a specialized model for detecting only p1 and p2 calibration landmarks.
This version uses annotations_p1_p2.json file (LabelStudio format).
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

# Import from existing files
import sys
sys.path.append(str(Path(__file__).parent))
from model import CephalometricLandmarkDetector

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
        
        # Normalize image
        image_tensor = torch.FloatTensor(image_resized).permute(2, 0, 1) / 255.0
        
        return image_tensor, landmarks_tensor, image_id

def train_p1_p2_model(
    annotations_json='annotations_p1_p2.json',
    image_dir='Aariz/train/Cephalograms',
    image_size=512,
    batch_size=16,
    num_epochs=100,
    learning_rate=0.001,
    device='cuda',
    train_split=0.8,
    save_best=True
):
    """Train a model for p1 and p2 detection from JSON annotations."""
    
    print("="*60)
    print("Training P1/P2 Calibration Landmark Detector")
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
    print(f"   Device: {device}\n")
    
    # Create model - only 2 landmarks (4 coordinates)
    model = CephalometricLandmarkDetector(num_landmarks=2)  # 2 landmarks = 4 outputs
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training loop
    best_loss = float('inf')
    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 20
    
    for epoch in range(num_epochs):
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
        with torch.no_grad():
            for images, landmarks, _ in val_loader:
                images = images.to(device)
                landmarks = landmarks.to(device)
                outputs = model(images)
                loss = criterion(outputs, landmarks)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_loss = avg_train_loss
            patience_counter = 0
            
            if save_best:
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
                }, 'checkpoint_p1_p2.pth')
                print(f"  [SAVED] Best model (val_loss: {avg_val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {max_patience} epochs)")
                break
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Best model saved as: checkpoint_p1_p2.pth")
    print(f"[RESULT] Best train loss: {best_loss:.6f}")
    print(f"[RESULT] Best val loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train P1/P2 model from annotations_p1_p2.json')
    parser.add_argument('--annotations', type=str, default='annotations_p1_p2.json',
                       help='Path to annotations JSON file')
    parser.add_argument('--image-dir', type=str, default='Aariz/train/Cephalograms',
                       help='Directory containing images')
    parser.add_argument('--image-size', type=int, default=512,
                       help='Image size for training')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Train/validation split ratio')
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
    train_p1_p2_model(
        annotations_json=args.annotations,
        image_dir=args.image_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        device=device,
        train_split=args.train_split
    )

