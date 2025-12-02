"""
اسکریپت برای بررسی heatmap‌ها در fine-tuning
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_combined_31_landmarks import Combined31LandmarkDataset, extract_coordinates_from_heatmaps
from model import HRNetLandmarkModel

# بارگذاری مدل
checkpoint_path = 'checkpoint_best_768_combined_31_finetuned.pth'
if not os.path.exists(checkpoint_path):
    print(f"Model not found: {checkpoint_path}")
    print("Using original combined model...")
    checkpoint_path = 'checkpoint_best_768_combined_31.pth'

checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint)

model = HRNetLandmarkModel(num_landmarks=31, width=32)
model.load_state_dict(state_dict, strict=False)
model = model.to('cuda')
model.eval()

# ایجاد dataset
dataset = Combined31LandmarkDataset(
    annotations_file='annotations_p1_p2.json',
    images_dir='Aariz/train/Cephalograms',
    image_size=768,
    heatmap_size=768,
    augment=False
)

dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

print("="*80)
print("Debugging Heatmaps")
print("="*80)

with torch.no_grad():
    for idx, (images, heatmaps_gt, coords_gt, image_paths) in enumerate(dataloader):
        if idx >= 3:  # فقط 3 نمونه اول
            break
        
        images = images.to('cuda')
        heatmaps_gt = heatmaps_gt.to('cuda')
        
        # پیش‌بینی
        heatmaps_pred = model(images)
        heatmaps_pred = torch.sigmoid(heatmaps_pred)
        
        # P1/P2 heatmaps
        p1p2_heatmaps_gt = heatmaps_gt[:, 29:31]
        p1p2_heatmaps_pred = heatmaps_pred[:, 29:31]
        
        print(f"\n[Sample {idx+1}]")
        print(f"  Image: {image_paths[0]}")
        print(f"  GT coords: P1=({coords_gt[0,0]:.4f}, {coords_gt[0,1]:.4f}), P2=({coords_gt[0,2]:.4f}, {coords_gt[0,3]:.4f})")
        
        # بررسی P1
        p1_gt_hm = p1p2_heatmaps_gt[0, 0].cpu().numpy()
        p1_pred_hm = p1p2_heatmaps_pred[0, 0].cpu().numpy()
        
        p1_gt_max = np.unravel_index(p1_gt_hm.argmax(), p1_gt_hm.shape)
        p1_pred_max = np.unravel_index(p1_pred_hm.argmax(), p1_pred_hm.shape)
        
        print(f"\n  P1 Heatmap:")
        print(f"    GT:  max={p1_gt_hm.max():.4f} at ({p1_gt_max[0]}, {p1_gt_max[1]}), mean={p1_gt_hm.mean():.6f}, sum={p1_gt_hm.sum():.2f}")
        print(f"    Pred: max={p1_pred_hm.max():.4f} at ({p1_pred_max[0]}, {p1_pred_max[1]}), mean={p1_pred_hm.mean():.6f}, sum={p1_pred_hm.sum():.2f}")
        
        # استخراج مختصات
        p1_coords_pred = extract_coordinates_from_heatmaps(p1p2_heatmaps_pred[0:1], 768)
        p1_coords_pred_norm = p1_coords_pred[0, 0]  # (x, y) normalized
        
        # محاسبه خطا
        p1_gt_norm = coords_gt[0, 0:2].numpy()
        p1_error_px = np.sqrt(np.sum((p1_coords_pred_norm - p1_gt_norm) ** 2)) * 768
        
        print(f"    Pred coords (normalized): ({p1_coords_pred_norm[0]:.4f}, {p1_coords_pred_norm[1]:.4f})")
        print(f"    GT coords (normalized): ({p1_gt_norm[0]:.4f}, {p1_gt_norm[1]:.4f})")
        print(f"    Error: {p1_error_px:.2f} px")
        
        # بررسی P2
        p2_gt_hm = p1p2_heatmaps_gt[0, 1].cpu().numpy()
        p2_pred_hm = p1p2_heatmaps_pred[0, 1].cpu().numpy()
        
        p2_gt_max = np.unravel_index(p2_gt_hm.argmax(), p2_gt_hm.shape)
        p2_pred_max = np.unravel_index(p2_pred_hm.argmax(), p2_pred_hm.shape)
        
        print(f"\n  P2 Heatmap:")
        print(f"    GT:  max={p2_gt_hm.max():.4f} at ({p2_gt_max[0]}, {p2_gt_max[1]}), mean={p2_gt_hm.mean():.6f}, sum={p2_gt_hm.sum():.2f}")
        print(f"    Pred: max={p2_pred_hm.max():.4f} at ({p2_pred_max[0]}, {p2_pred_max[1]}), mean={p2_pred_hm.mean():.6f}, sum={p2_pred_hm.sum():.2f}")
        
        # استخراج مختصات
        p2_coords_pred = extract_coordinates_from_heatmaps(p1p2_heatmaps_pred[0:1], 768)
        p2_coords_pred_norm = p2_coords_pred[0, 1]  # (x, y) normalized
        
        # محاسبه خطا
        p2_gt_norm = coords_gt[0, 2:4].numpy()
        p2_error_px = np.sqrt(np.sum((p2_coords_pred_norm - p2_gt_norm) ** 2)) * 768
        
        print(f"    Pred coords (normalized): ({p2_coords_pred_norm[0]:.4f}, {p2_coords_pred_norm[1]:.4f})")
        print(f"    GT coords (normalized): ({p2_gt_norm[0]:.4f}, {p2_gt_norm[1]:.4f})")
        print(f"    Error: {p2_error_px:.2f} px")
        
        # بررسی اینکه آیا heatmap flat است
        if p1_pred_hm.max() - p1_pred_hm.min() < 0.01:
            print(f"    [WARNING] P1 heatmap is FLAT (max-min < 0.01)!")
        if p2_pred_hm.max() - p2_pred_hm.min() < 0.01:
            print(f"    [WARNING] P2 heatmap is FLAT (max-min < 0.01)!")




