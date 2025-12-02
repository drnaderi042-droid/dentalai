"""
Train a specialized model for detecting only p1 and p2 calibration landmarks.
This model will be much faster and more accurate than the full 29-landmark model.
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

# Import from existing files
import sys
sys.path.append(str(Path(__file__).parent))
from model import CephalometricLandmarkDetector

class P1P2Dataset(Dataset):
    """Dataset for p1 and p2 landmarks only."""
    
    def __init__(self, image_dir, annotation_dir, image_ids, image_size=512, augment=False):
        self.image_dir = Path(image_dir)
        self.annotation_dir = Path(annotation_dir)
        self.image_ids = image_ids
        self.image_size = image_size
        self.augment = augment
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        
        # Load annotation
        annotation_file = self.annotation_dir / f"{image_id}.json"
        with open(annotation_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract p1 and p2
        landmarks = {}
        for lm in data['landmarks']:
            if lm['symbol'] in ['p1', 'p2']:
                landmarks[lm['symbol']] = lm['value']
        
        if 'p1' not in landmarks or 'p2' not in landmarks:
            raise ValueError(f"Missing p1 or p2 in {image_id}")
        
        # Load image
        image_path = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_path = self.image_dir / f"{image_id}{ext}"
            if potential_path.exists():
                image_path = potential_path
                break
        
        if image_path is None:
            raise FileNotFoundError(f"Image not found for {image_id}")
        
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]
        
        # Resize image
        image_resized = cv2.resize(image, (self.image_size, self.image_size))
        
        # Normalize landmarks
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        
        p1_x = landmarks['p1']['x'] * scale_x
        p1_y = landmarks['p1']['y'] * scale_y
        p2_x = landmarks['p2']['x'] * scale_x
        p2_y = landmarks['p2']['y'] * scale_y
        
        # Create landmark tensor: [x1, y1, x2, y2]
        landmarks_tensor = torch.FloatTensor([p1_x, p1_y, p2_x, p2_y])
        
        # Normalize to [0, 1]
        landmarks_tensor = landmarks_tensor / self.image_size
        
        # Normalize image
        image_tensor = torch.FloatTensor(image_resized).permute(2, 0, 1) / 255.0
        
        return image_tensor, landmarks_tensor, image_id

def train_p1_p2_model(
    train_ids,
    image_size=512,
    batch_size=4,
    num_epochs=100,
    learning_rate=0.001,
    device='cuda'
):
    """Train a model for p1 and p2 detection."""
    
    print("="*60)
    print("Training P1/P2 Calibration Landmark Detector")
    print("="*60)
    
    # Paths
    image_dir = Path('Aariz/train/Cephalograms')
    annotation_dir = Path('Aariz/train/Annotations/Cephalometric Landmarks/Senior Orthodontists')
    
    # Create dataset
    dataset = P1P2Dataset(image_dir, annotation_dir, train_ids, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"\nDataset Info:")
    print(f"   Training images: {len(dataset)}")
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
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
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
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'num_landmarks': 2,
                'image_size': image_size,
            }, 'checkpoint_p1_p2.pth')
            print(f"[SAVED] Best model (loss: {best_loss:.6f})")
    
    print(f"\n[SUCCESS] Training complete!")
    print(f"[INFO] Best model saved as: checkpoint_p1_p2.pth")
    print(f"[RESULT] Best loss: {best_loss:.6f}")

if __name__ == '__main__':
    # P1/P2 image IDs
    train_ids = [
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
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[DEVICE] Using: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")
    
    # Train
    train_p1_p2_model(
        train_ids=train_ids,
        image_size=512,
        batch_size=4,  # Adjust based on GPU memory
        num_epochs=100,
        learning_rate=0.001,
        device=device
    )

