"""
اسکریپت برای بررسی مشکل loss بزرگ
"""
import torch
import torch.nn as nn
import numpy as np

# شبیه‌سازی loss
batch_size = 4
num_landmarks = 2
H, W = 768, 768

# Ground truth heatmap (در بازه [0, 1])
gt_heatmap = torch.rand(batch_size, num_landmarks, H, W) * 0.1  # بیشتر صفر، فقط یک ناحیه کوچک فعال
gt_heatmap[:, :, 384:394, 384:394] = torch.rand(batch_size, num_landmarks, 10, 10)  # یک ناحیه فعال

# Predicted heatmap - حالت 1: بدون sigmoid (ممکن است منفی یا بزرگ باشد)
pred_heatmap_no_sigmoid = torch.randn(batch_size, num_landmarks, H, W) * 2.0  # در بازه [-4, 4]

# Predicted heatmap - حالت 2: با sigmoid (در بازه [0, 1])
pred_heatmap_with_sigmoid = torch.sigmoid(pred_heatmap_no_sigmoid)

# محاسبه loss
criterion = nn.MSELoss()

loss_no_sigmoid = criterion(pred_heatmap_no_sigmoid, gt_heatmap)
loss_with_sigmoid = criterion(pred_heatmap_with_sigmoid, gt_heatmap)

print("="*80)
print("Debug Loss Issue")
print("="*80)
print(f"\nHeatmap shape: ({batch_size}, {num_landmarks}, {H}, {W})")
print(f"Total pixels per heatmap: {H * W:,}")
print(f"Total pixels for 2 heatmaps: {H * W * 2:,}")

print(f"\nGround Truth:")
print(f"  Min: {gt_heatmap.min():.6f}")
print(f"  Max: {gt_heatmap.max():.6f}")
print(f"  Mean: {gt_heatmap.mean():.6f}")

print(f"\nPredicted (NO sigmoid):")
print(f"  Min: {pred_heatmap_no_sigmoid.min():.6f}")
print(f"  Max: {pred_heatmap_no_sigmoid.max():.6f}")
print(f"  Mean: {pred_heatmap_no_sigmoid.mean():.6f}")
print(f"  Loss: {loss_no_sigmoid.item():.6f}")

print(f"\nPredicted (WITH sigmoid):")
print(f"  Min: {pred_heatmap_with_sigmoid.min():.6f}")
print(f"  Max: {pred_heatmap_with_sigmoid.max():.6f}")
print(f"  Mean: {pred_heatmap_with_sigmoid.mean():.6f}")
print(f"  Loss: {loss_with_sigmoid.item():.6f}")

print(f"\n" + "="*80)
print("Analysis:")
print("="*80)
print(f"If model output is NOT sigmoided:")
print(f"  - Output range: [-inf, +inf] (typically [-2, 2] or more)")
print(f"  - GT range: [0, 1]")
print(f"  - Expected loss: HIGH (hundreds or thousands)")
print(f"  - Your loss: 647 (matches this pattern!)")

print(f"\nIf model output IS sigmoided:")
print(f"  - Output range: [0, 1]")
print(f"  - GT range: [0, 1]")
print(f"  - Expected loss: LOW (< 0.1)")
print(f"  - This is what we want!")

print(f"\n" + "="*80)
print("Solution:")
print("="*80)
print("Apply sigmoid to model output before calculating loss:")
print("  heatmaps_pred = torch.sigmoid(model(images))")




