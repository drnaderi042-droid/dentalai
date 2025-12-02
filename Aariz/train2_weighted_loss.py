"""
نسخه بهبود یافته train2.py با Weighted Loss برای لندمارک‌های مشکل‌دار

این نسخه وزن بیشتری به لندمارک‌های با خطای بالا می‌دهد تا مدل بهتر یاد بگیرد.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from dataset import AarizDataset
from model import get_model
from utils import load_checkpoint, save_checkpoint, heatmap_to_coordinates, calculate_mre, calculate_sdr
from losses import AdaptiveWingLoss, CombinedLoss

# وزن‌های لندمارک‌های مشکل‌دار (بر اساس تحلیل خطا)
DIFFICULT_LANDMARK_WEIGHTS = {
    'UMT': 2.5,   # Upper Molar Tip - بیشترین خطا
    'UPM': 2.5,   # Upper Premolar
    'R': 2.0,     # Ramus point
    'Ar': 1.8,    # Articulare
    'Go': 1.8,    # Gonion
    'LMT': 1.6,   # Lower Molar Tip
    'LPM': 1.4,   # Lower Premolar
    'Or': 1.3,    # Orbitale
    'Co': 1.2,    # Condylion
    'PNS': 1.2,   # Posterior Nasal Spine
    # بقیه لندمارک‌ها وزن 1.0 دارند (پیش‌فرض)
}

# ترتیب لندمارک‌ها (باید با dataset یکسان باشد)
LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

def calculate_weighted_loss(outputs, targets, landmark_symbols, base_criterion, device):
    """
    محاسبه weighted loss برای لندمارک‌های مشکل‌دار
    
    Args:
        outputs: خروجی مدل [B, C, H, W]
        targets: targets [B, C, H, W]
        landmark_symbols: لیست نام لندمارک‌ها
        base_criterion: loss function پایه (AdaptiveWingLoss)
        device: device
    
    Returns:
        weighted_loss: loss وزن‌دار شده
    """
    batch_size = outputs.shape[0]
    num_landmarks = outputs.shape[1]
    
    total_loss = 0.0
    total_weight = 0.0
    
    for i in range(num_landmarks):
        if i >= len(landmark_symbols):
            weight = 1.0
        else:
            landmark_name = landmark_symbols[i]
            weight = DIFFICULT_LANDMARK_WEIGHTS.get(landmark_name, 1.0)
        
        # استخراج heatmap این لندمارک
        landmark_output = outputs[:, i:i+1, :, :]
        landmark_target = targets[:, i:i+1, :, :]
        
        # محاسبه loss برای این لندمارک
        landmark_loss = base_criterion(landmark_output, landmark_target)
        
        total_loss += weight * landmark_loss
        total_weight += weight
    
    # نرمال‌سازی بر اساس مجموع وزن‌ها
    weighted_loss = total_loss / total_weight if total_weight > 0 else total_loss / num_landmarks
    
    return weighted_loss


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, 
                use_adaptive_wing, use_mixed_precision, scaler=None, writer=None):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_focal = 0.0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    
    # استفاده از autocast برای mixed precision
    if use_mixed_precision:
        from torch.cuda.amp import autocast as amp_autocast
        amp_autocast_device = 'cuda'
        amp_autocast_dtype = torch.float16
    else:
        amp_autocast = torch._C._nn._parse_to(None, device, None, None)
        amp_autocast_device = None
        amp_autocast_dtype = None
    
    for batch_idx, batch in enumerate(pbar):
        optimizer.zero_grad()
        
        images = batch['image'].to(device, non_blocking=True)
        targets = batch['target'].to(device, non_blocking=True)
        
        if use_mixed_precision:
            with amp_autocast(device_type=amp_autocast_device, dtype=amp_autocast_dtype):
                outputs = model(images)
                
                # Resize outputs if needed
                if outputs.shape[2:] != targets.shape[2:]:
                    outputs_resized = torch.nn.functional.interpolate(
                        outputs, size=targets.shape[2:], 
                        mode='bilinear', align_corners=False
                    )
                else:
                    outputs_resized = outputs
                
                # استفاده از weighted loss
                if use_adaptive_wing:
                    loss = calculate_weighted_loss(
                        outputs_resized, targets,
                        LANDMARK_SYMBOLS,
                        criterion,
                        device
                    )
                    mse = torch.tensor(0.0)
                    focal = torch.tensor(0.0)
                else:
                    loss, mse, focal = criterion(outputs_resized, targets)
            
            # Backward pass with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (FP32)
            outputs = model(images)
            
            # Resize outputs to match target size if needed
            if outputs.shape[2:] != targets.shape[2:]:
                outputs = torch.nn.functional.interpolate(
                    outputs, size=targets.shape[2:], 
                    mode='bilinear', align_corners=False
                )
            
            # استفاده از weighted loss
            if use_adaptive_wing:
                loss = calculate_weighted_loss(
                    outputs, targets,
                    LANDMARK_SYMBOLS,
                    criterion,
                    device
                )
                mse = torch.tensor(0.0)
                focal = torch.tensor(0.0)
            else:
                loss, mse, focal = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Accumulate losses
        total_loss += loss.item()
        if isinstance(mse, torch.Tensor):
            total_mse += mse.item() if mse.numel() > 0 else 0.0
        else:
            total_mse += mse
        if isinstance(focal, torch.Tensor):
            total_focal += focal.item() if focal.numel() > 0 else 0.0
        else:
            total_focal += focal
        
        # Update progress bar
        if use_adaptive_wing:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        else:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mse': f'{mse.item() if isinstance(mse, torch.Tensor) else mse:.4f}',
                'focal': f'{focal.item() if isinstance(focal, torch.Tensor) else focal:.4f}'
            })
        
        # Log to tensorboard
        if writer and batch_idx % 100 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    avg_loss = total_loss / len(train_loader)
    avg_mse = total_mse / len(train_loader)
    avg_focal = total_focal / len(train_loader)
    
    return avg_loss, avg_mse, avg_focal


# باقی کد train2.py را کپی کنید و فقط train_epoch را جایگزین کنید
# برای استفاده:
# python train2_weighted_loss.py --resume checkpoints/checkpoint_best_768.pth --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 4 --lr 1e-5 --epochs 50 --loss adaptive_wing --mixed_precision

if __name__ == "__main__":
    print("="*80)
    print("Training با Weighted Loss برای لندمارک‌های مشکل‌دار")
    print("="*80)
    print("\nوزن‌های لندمارک‌های مشکل‌دار:")
    for landmark, weight in sorted(DIFFICULT_LANDMARK_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
        print(f"  {landmark}: {weight}x")
    print("\nبرای استفاده کامل، این کد را به train2.py اضافه کنید.")
    print("="*80)















