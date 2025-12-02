#!/usr/bin/env python3
"""
Training script for adding p1 and p2 landmarks to Aariz model
Optimized for i5-9400F CPU and RTX 3070 Ti GPU
Uses 768 model as base with 18 annotated images
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import argparse
import json
from datetime import datetime

# Import Aariz modules
from dataset_extended import ExtendedAarizDataset
from model import get_model
from utils import load_checkpoint, save_checkpoint, heatmap_to_coordinates, calculate_mre, calculate_sdr

class ExtendedModel(nn.Module):
    """Extended model for adding new landmarks"""
    def __init__(self, base_model, base_num_landmarks=29, new_num_landmarks=31):
        super(ExtendedModel, self).__init__()
        self.base_model = base_model
        self.base_num_landmarks = base_num_landmarks
        self.new_num_landmarks = new_num_landmarks
        self.num_new_landmarks = new_num_landmarks - base_num_landmarks

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Simple extension: learnable parameters for new landmarks
        # We'll create new heatmaps by learning linear combinations of base heatmaps
        self.new_landmark_weights = nn.Parameter(
            torch.randn(self.num_new_landmarks, base_num_landmarks) * 0.1
        )
        self.new_landmark_bias = nn.Parameter(torch.zeros(self.num_new_landmarks))

    def forward(self, x):
        # Get base landmarks (29 landmarks)
        base_heatmaps = self.base_model(x)

        # Create new landmarks as linear combinations of base landmarks
        batch_size, num_base_landmarks, height, width = base_heatmaps.shape

        # Reshape for matrix multiplication: (batch, landmarks, H*W) -> (batch, H*W, landmarks)
        base_reshaped = base_heatmaps.view(batch_size, num_base_landmarks, -1).permute(0, 2, 1)

        # Apply linear transformation: (batch, H*W, landmarks) @ (landmarks, new_landmarks) -> (batch, H*W, new_landmarks)
        new_heatmaps_flat = torch.matmul(base_reshaped, self.new_landmark_weights.t()) + self.new_landmark_bias

        # Reshape back: (batch, H*W, new_landmarks) -> (batch, new_landmarks, H, W)
        new_heatmaps = new_heatmaps_flat.permute(0, 2, 1).view(batch_size, self.num_new_landmarks, height, width)

        # Apply sigmoid to ensure heatmap values are between 0 and 1
        new_heatmaps = torch.sigmoid(new_heatmaps)

        # Concatenate base and new landmarks
        all_heatmaps = torch.cat([base_heatmaps, new_heatmaps], dim=1)

        return all_heatmaps

class ExtendedAnnotationsDataset(torch.utils.data.Dataset):
    """Custom dataset for extended annotations only"""
    def __init__(self, annotations_dir, image_base_dir, additional_landmarks=None):
        self.annotations_dir = annotations_dir
        self.image_base_dir = image_base_dir
        self.additional_landmarks = additional_landmarks or []

        # Get all annotation files
        self.annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
        self.image_ids = [os.path.splitext(f)[0] for f in self.annotation_files]

        # Base landmarks
        self.base_landmark_symbols = [
            "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
            "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
            "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
        ]
        self.landmark_symbols = self.base_landmark_symbols + self.additional_landmarks
        self.num_landmarks = len(self.landmark_symbols)

        print(f"Extended annotations dataset: {len(self.image_ids)} images")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        image_path = os.path.join(self.image_base_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            image_path = os.path.join(self.image_base_dir, f"{image_id}.jpg")

        from PIL import Image
        image = Image.open(image_path).convert('RGB')

        # Load annotation
        annotation_path = os.path.join(self.annotations_dir, f"{image_id}.json")
        with open(annotation_path, 'r', encoding='utf-8') as f:
            annotation = json.load(f)

        # Extract landmarks
        landmarks_dict = {}
        if 'landmarks' in annotation:
            for lm in annotation['landmarks']:
                symbol = lm['symbol']
                landmarks_dict[symbol] = {
                    'x': float(lm['value']['x']),
                    'y': float(lm['value']['y'])
                }

        # Create landmark array
        landmarks = np.full((self.num_landmarks, 2), -1.0, dtype=np.float32)

        # Fill base landmarks
        for i, symbol in enumerate(self.base_landmark_symbols):
            if symbol in landmarks_dict:
                landmarks[i] = [
                    landmarks_dict[symbol]['x'],
                    landmarks_dict[symbol]['y']
                ]

        # Fill additional landmarks
        for i, symbol in enumerate(self.additional_landmarks):
            idx_pos = len(self.base_landmark_symbols) + i
            if symbol in landmarks_dict:
                landmarks[idx_pos] = [
                    landmarks_dict[symbol]['x'],
                    landmarks_dict[symbol]['y']
                ]

        # Simple transform (resize and normalize)
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        image_tensor = transform(image)

        # Generate heatmaps
        heatmaps = self._generate_heatmaps(landmarks, (512, 512))
        heatmaps = torch.from_numpy(heatmaps).float()

        return image_tensor, heatmaps, landmarks, image_id

    def _generate_heatmaps(self, landmarks, image_size):
        """Generate Gaussian heatmaps for landmarks"""
        h, w = image_size
        heatmaps = np.zeros((self.num_landmarks, h, w), dtype=np.float32)

        for i, landmark in enumerate(landmarks):
            if landmark[0] >= 0 and landmark[1] >= 0:
                x, y = landmark
                # Scale to image size if needed
                if x <= 1.0 and y <= 1.0:  # Normalized coordinates
                    x *= w
                    y *= h

                x, y = int(x), int(y)
                if 0 <= x < w and 0 <= y < h:
                    # Create Gaussian heatmap
                    y_coords, x_coords = np.ogrid[:h, :w]
                    dist_sq = (x_coords - x) ** 2 + (y_coords - y) ** 2
                    heatmaps[i] = np.exp(-dist_sq / (2 * 3.0 ** 2))

        return heatmaps

def create_data_loader(annotations_dir, batch_size=2, num_workers=4):
    """Create data loader optimized for i5-9400F"""
    from torch.utils.data import DataLoader

    # Create custom dataset for extended annotations only
    dataset = ExtendedAnnotationsDataset(
        annotations_dir=annotations_dir,
        image_base_dir="Aariz/Aariz/train/Cephalograms",
        additional_landmarks=["p1", "p2"]
    )

    # Create data loader with optimized settings for i5-9400F
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Optimized for i5-9400F (6 cores)
        pin_memory=True,  # GPU memory pinning
        persistent_workers=True if num_workers > 0 else False
    )

    return data_loader

def train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, accumulation_steps=4):
    """Training epoch with gradient accumulation for small batch size"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch in enumerate(pbar):
        if len(batch) == 4:
            images, heatmaps, landmarks, image_ids = batch
        else:
            images, landmarks, image_ids = batch
            heatmaps = None

        images = images.to(device)
        if heatmaps is not None:
            heatmaps = heatmaps.to(device)

        # Forward pass with mixed precision
        with autocast():
            pred_heatmaps = model(images)

            if heatmaps is not None:
                # Loss on base landmarks (29) - these are already trained
                base_num = model.base_num_landmarks
                base_loss = criterion(pred_heatmaps[:, :base_num], heatmaps[:, :base_num])

                # Loss on new landmarks (p1, p2) - these need training
                new_loss = criterion(pred_heatmaps[:, base_num:], heatmaps[:, base_num:])

                # Combine losses with higher weight on new landmarks
                loss = (base_loss + 5.0 * new_loss) / accumulation_steps
            else:
                loss = torch.tensor(0.0, device=device)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    # Handle remaining accumulated gradients
    if (batch_idx + 1) % accumulation_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return total_loss / num_batches if num_batches > 0 else 0.0

def validate(model, val_loader, device, pixel_size=0.1):
    """Validation with focus on new landmarks"""
    model.eval()
    all_errors = []
    new_landmark_errors = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            if len(batch) == 4:
                images, heatmaps, landmarks, image_ids = batch
            else:
                images, landmarks, image_ids = batch

            images = images.to(device)
            landmarks = landmarks.numpy()

            pred_heatmaps = model(images)
            pred_heatmaps_np = pred_heatmaps.cpu().numpy()

            batch_size = pred_heatmaps_np.shape[0]
            for i in range(batch_size):
                pred_coords = heatmap_to_coordinates(pred_heatmaps_np[i], 512, 512)
                gt_coords = landmarks[i]

                # Calculate errors for all valid landmarks
                valid_mask = (gt_coords[:, 0] >= 0) & (gt_coords[:, 1] >= 0)
                if valid_mask.sum() > 0:
                    errors = np.sqrt(np.sum((pred_coords[valid_mask] - gt_coords[valid_mask]) ** 2, axis=1))
                    errors_mm = errors * pixel_size
                    all_errors.extend(errors_mm.tolist())

                    # Separate errors for new landmarks (p1, p2)
                    base_landmarks = 29  # First 29 are base landmarks
                    new_mask = valid_mask[base_landmarks:]
                    if new_mask.sum() > 0:
                        new_errors = errors_mm[base_landmarks:][new_mask]
                        new_landmark_errors.extend(new_errors.tolist())

    if len(all_errors) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mre = np.mean(all_errors)
    sdr_2mm = (np.array(all_errors) <= 2.0).sum() / len(all_errors) * 100

    if new_landmark_errors:
        new_mre = np.mean(new_landmark_errors)
        new_sdr_2mm = (np.array(new_landmark_errors) <= 2.0).sum() / len(new_landmark_errors) * 100
    else:
        new_mre = 0.0
        new_sdr_2mm = 0.0

    return mre, sdr_2mm, new_mre, new_sdr_2mm

def main():
    parser = argparse.ArgumentParser(description='Train extended Aariz model with p1 and p2 landmarks')
    parser.add_argument('--annotations_dir', default='extended_annotations', help='Directory with extended annotations')
    parser.add_argument('--checkpoint', default='Aariz/checkpoint_best_768.pth', help='Base model checkpoint (768)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (optimized for 3070 Ti)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save_dir', default='checkpoints_p1_p2', help='Save directory')
    parser.add_argument('--log_dir', default='logs_p1_p2', help='Log directory')
    parser.add_argument('--accumulation_steps', type=int, default=4, help='Gradient accumulation steps')

    args = parser.parse_args()

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Create data loader
    print("Creating data loader...")
    train_loader = create_data_loader(args.annotations_dir, args.batch_size)

    # Load base model (768)
    print(f"Loading base model from: {args.checkpoint}")
    base_model = get_model('hrnet', num_landmarks=29)
    checkpoint = load_checkpoint(args.checkpoint, base_model)
    base_model = base_model.to(device)
    base_model.eval()

    # Create extended model
    print("Creating extended model with p1 and p2 landmarks...")
    model = ExtendedModel(base_model, base_num_landmarks=29, new_num_landmarks=31)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam([
        {'params': model.new_landmark_weights, 'lr': args.lr},
        {'params': model.new_landmark_bias, 'lr': args.lr}
    ], lr=args.lr)
    scaler = GradScaler()

    # TensorBoard
    writer = SummaryWriter(args.log_dir)

    # Training loop
    best_new_mre = float('inf')
    best_epoch = 0

    print(f"\\nStarting training...")
    print(f"  Total images: {len(train_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.accumulation_steps}")
    print(f"  Effective batch size: {args.batch_size * args.accumulation_steps}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch, args.accumulation_steps)

        # Validate
        mre, sdr_2mm, new_mre, new_sdr_2mm = validate(model, train_loader, device)

        # Log
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Valid/MRE', mre, epoch)
        writer.add_scalar('Valid/SDR_2mm', sdr_2mm, epoch)
        writer.add_scalar('Valid/New_MRE', new_mre, epoch)
        writer.add_scalar('Valid/New_SDR_2mm', new_sdr_2mm, epoch)

        print(f"Epoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Overall MRE: {mre:.2f}mm, SDR@2mm: {sdr_2mm:.2f}%")
        print(f"  New Landmarks MRE: {new_mre:.2f}mm, SDR@2mm: {new_sdr_2mm:.2f}%")
        # Save best model based on new landmark performance
        if new_mre < best_new_mre and new_mre > 0:
            best_new_mre = new_mre
            best_epoch = epoch
            checkpoint_path = os.path.join(args.save_dir, 'checkpoint_best.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mre': mre,
                'sdr_2mm': sdr_2mm,
                'new_mre': new_mre,
                'new_sdr_2mm': new_sdr_2mm,
                'num_landmarks': 31,
                'additional_landmarks': ['p1', 'p2']
            }, checkpoint_path)
            print(f"  ✓ Saved best checkpoint: {checkpoint_path}")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch}.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'mre': mre,
                'sdr_2mm': sdr_2mm,
                'new_mre': new_mre,
                'new_sdr_2mm': new_sdr_2mm
            }, checkpoint_path)

    print(f"\\nTraining completed!")
    print(f"Best new landmark MRE: {best_new_mre:.2f}mm (epoch {best_epoch})")
    print(f"Final model saved to: {args.save_dir}/checkpoint_best.pth")

if __name__ == "__main__":
    main()
