"""
Training Script for CVM Stage Classification
"""
import os
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast as amp_autocast, GradScaler as amp_GradScaler
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import shutil

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, AttributeError, Exception) as e:
    TENSORBOARD_AVAILABLE = False
    # Suppress tensorboard warning

from dataset_cvm import create_cvm_dataloaders
from model_cvm import get_cvm_model


class WarmupCosineScheduler:
    """
    Learning rate scheduler with warmup and cosine annealing with restarts
    """
    def __init__(self, optimizer, warmup_epochs=5, max_epochs=100, T_0=10, T_mult=2, eta_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.base_lr = optimizer.param_groups[0]['lr']
        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=T_0, T_mult=T_mult, eta_min=eta_min
        )
        self.current_epoch = 0
    
    def step(self, epoch=None):
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        if self.current_epoch < self.warmup_epochs:
            lr = self.base_lr * (self.current_epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr
        else:
            self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None, 
                use_mixed_precision=False, scaler=None, gradient_accumulation_steps=1):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        
        if use_mixed_precision:
            with amp_autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss = loss / gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / gradient_accumulation_steps
            
            loss.backward()
            
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        total_loss += loss.item() * gradient_accumulation_steps
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        accuracy = 100.0 * correct / total
        
        pbar.set_postfix({
            'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
            'acc': f'{accuracy:.2f}%'
        })
        
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item() * gradient_accumulation_steps, global_step)
            writer.add_scalar('Train/BatchAccuracy', accuracy, global_step)
    
    remaining = len(train_loader) % gradient_accumulation_steps
    if remaining != 0 and len(train_loader) > 0:
        if use_mixed_precision and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        elif not use_mixed_precision:
            optimizer.step()
            optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    avg_accuracy = 100.0 * correct / total
    
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device, epoch, writer=None, use_mixed_precision=False):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Valid]')
        
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)
            
            if use_mixed_precision:
                with amp_autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            accuracy = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{accuracy:.2f}%'})
    
    avg_loss = total_loss / len(val_loader)
    avg_accuracy = 100.0 * correct / total
    
    # Calculate per-class accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    per_class_acc = {}
    for class_id in range(6):
        mask = all_labels == class_id
        if mask.sum() > 0:
            class_acc = 100.0 * (all_predictions[mask] == all_labels[mask]).sum() / mask.sum()
            per_class_acc[f'stage_{class_id+1}'] = class_acc
    
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/Accuracy', avg_accuracy, epoch)
        for class_name, acc in per_class_acc.items():
            writer.add_scalar(f'Val/{class_name}', acc, epoch)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'per_class_accuracy': per_class_acc
    }
    
    return metrics


def save_checkpoint(state, save_dir, is_best=False, epoch=None):
    """Save checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(save_dir, 'checkpoint_latest.pth')
    torch.save(state, checkpoint_path)
    
    if epoch is not None:
        epoch_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state, epoch_path)
    
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pth')
        shutil.copyfile(checkpoint_path, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=30, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
        
        return self.early_stop


def main():
    parser = argparse.ArgumentParser(description='Train CVM Stage Classification Model')
    parser.add_argument('--dataset_path', type=str, default='Aariz', help='Path to dataset')
    parser.add_argument('--model', type=str, default='hrnet', choices=['hrnet', 'resnet18', 'resnet34', 'resnet50'],
                       help='Model architecture')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--warmup_epochs', type=int, default=5, help='Number of warmup epochs')
    parser.add_argument('--image_size', type=int, nargs=2, default=[768, 768], help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--save_dir', type=str, default='CVM/checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='CVM/logs', help='Directory for tensorboard logs')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (FP16)')
    parser.add_argument('--pretrained_backbone', type=str, default=None,
                       help='Path to pretrained backbone checkpoint')
    parser.add_argument('--early_stopping_patience', type=int, default=30,
                       help='Early stopping patience (epochs without improvement)')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataloaders
    print("Loading dataset...")
    train_loader, val_loader = create_cvm_dataloaders(
        dataset_folder_path=args.dataset_path,
        image_size=tuple(args.image_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    # Create model
    print(f"Creating model: {args.model}")
    model = get_cvm_model(
        model_name=args.model,
        num_classes=6,
        width=32,
        pretrained_backbone=args.pretrained_backbone
    )
    model = model.to(device)
    
    # Loss function with class weights for imbalanced dataset
    # Class distribution: Stage 1: 18, 2: 38, 3: 37, 4: 185, 5: 311, 6: 111 (total: 700)
    class_weights = torch.tensor([38.89, 18.42, 18.92, 3.78, 2.25, 6.31], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Mixed precision training
    use_mixed_precision = args.mixed_precision and device.type == 'cuda'
    scaler = None
    if use_mixed_precision:
        print("Using Mixed Precision Training (FP16)")
        scaler = amp_GradScaler('cuda')
    
    # Learning rate scheduler
    print(f"Using WarmupCosineScheduler (warmup={args.warmup_epochs} epochs)")
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, mode='max')
    
    # Tensorboard writer
    writer = None
    if TENSORBOARD_AVAILABLE:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        writer = SummaryWriter(os.path.join(args.log_dir, f'{args.model}_{timestamp}'))
    else:
        print("Tensorboard not available. Logging disabled.")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_accuracy = 0.0
    
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint.get('best_accuracy', 0.0)
        print(f"Resumed from epoch {start_epoch}")
        print(f"Best accuracy: {best_accuracy:.2f}%")
    
    # Training loop
    print("Starting training...")
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer,
            use_mixed_precision, scaler
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, writer, use_mixed_precision)
        
        # Update learning rate
        scheduler.step(epoch)
        
        # Log to tensorboard
        if writer:
            writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            writer.add_scalar('Train/EpochAccuracy', train_acc, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # Print metrics
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Accuracy: {val_metrics['accuracy']:.2f}%")
        print("Per-class accuracy:")
        for class_name, acc in val_metrics['per_class_accuracy'].items():
            print(f"  {class_name}: {acc:.2f}%")
        print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['accuracy'] > best_accuracy
        if is_best:
            best_accuracy = val_metrics['accuracy']
            print(f"Saved best model (Accuracy: {best_accuracy:.2f}%)")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'val_metrics': val_metrics,
            'args': vars(args)
        }
        
        save_checkpoint(checkpoint, args.save_dir, is_best, epoch)
        
        # Early stopping
        if early_stopping(val_metrics['accuracy']):
            print(f"Early stopping triggered after {epoch+1} epochs")
            print(f"Best accuracy: {best_accuracy:.2f}%")
            break
        
        print("-" * 80)
    
    if writer:
        writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()

