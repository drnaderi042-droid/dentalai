# Advanced Training Guide: HRNet-W48 + Multi-task + Two-stage

## 📋 Overview

This training script implements an advanced pipeline for cephalometric landmark detection with the following features:

### Architecture
- **Backbone**: HRNet-W48 (width=48)
- **Input Size**: 512×512 pixels
- **Heatmap Size**: 128×128 pixels
- **Multi-task Learning**: Landmark Detection + CVM (Cephalometric Vertical Measurement)

### Loss Functions
- **Combined Loss**: Dark-Pose Loss + Wing Loss
  - `loss = 0.5 * DarkLoss() + 0.5 * WingLoss()`
- **Multi-task Loss**: 
  - `loss_total = loss_landmark + 0.3 * loss_CVM`

### Data Augmentation
Total dataset: **6500 images**
- Original: 1000 images
- Flipped (horizontal): 1000 images
- Rotated ±7°: 2000 images
- Brightness adjusted: 1000 images
- Synthetic braces: 500 images

### Two-Stage Pipeline
1. **Stage-1**: HRNet-W48 → Initial landmark prediction (512×512 → 128×128 heatmaps)
2. **Stage-2**: HRNet-W32 → Refinement using 5 crops (256×256) around each predicted landmark

### Test-Time Augmentation (TTA)
- 10 augmentations per image:
  - Flip: [False, True]
  - Multi-scale: [0.8, 1.0, 1.2]
  - Rotation: [-3°, 0°, +3°]
- Final prediction: Average of all augmentations

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PyTorch 1.10+
- CUDA (for GPU training)
- RTX 3070 Ti (8GB VRAM) or better
- CPU: Intel i5 9400F (6 cores) or better

### Installation
```bash
pip install torch torchvision torchaudio
pip install albumentations
pip install tensorboard
pip install tqdm
pip install pandas
pip install opencv-python
```

### Training
```bash
# Windows
train_advanced.bat

# Linux/Mac
python train_advanced.py \
    --dataset_path Aariz \
    --batch_size 6 \
    --epochs 200 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --image_size 512 512 \
    --heatmap_size 128 \
    --num_workers 6 \
    --mixed_precision \
    --cvm_weight 0.3 \
    --save_dir checkpoints_advanced_rtx3070ti \
    --log_dir logs_advanced_rtx3070ti
```

## 📊 Training Parameters

### Recommended Settings for RTX 3070 Ti
- **Batch Size**: 6 (optimized for 8GB VRAM)
- **Learning Rate**: 1e-4
- **Warmup Epochs**: 10
- **Total Epochs**: 200
- **Mixed Precision**: Enabled (FP16)
- **Num Workers**: 6 (for CPU i5 9400F - 6 cores)

### Expected Training Time
- **Per Epoch**: ~12-15 minutes (RTX 3070 Ti)
- **Total Time**: ~40-50 hours (200 epochs)

## 📁 File Structure

```
Aariz/
├── train_advanced.py                    # Main training script
├── train_advanced.bat                   # Windows batch script
├── models_advanced.py                   # HRNet-W48 and HRNet-W32 models
├── advanced_losses.py                   # Dark-Pose + Wing Loss
├── dataset_advanced.py                 # Dataset with augmentations
└── checkpoints_advanced_rtx3070ti/     # Saved checkpoints (RTX 3070 Ti)
    └── checkpoint_best.pth
```

## 🔧 Advanced Features

### 1. Multi-task Learning
The model predicts both:
- **Landmarks**: 29 cephalometric landmarks (heatmaps)
- **CVM**: 10 cephalometric vertical measurements (regression)

### 2. Two-Stage Pipeline
- **Stage-1**: Coarse prediction with HRNet-W48
- **Stage-2**: Fine refinement with HRNet-W32 on 256×256 crops

### 3. Test-Time Augmentation
During evaluation, each image is augmented 10 times and predictions are averaged for better accuracy.

## 📈 Monitoring

### TensorBoard
```bash
tensorboard --logdir logs_advanced_rtx3070ti
```

### Metrics Tracked
- Train/Val Loss (total, landmark, CVM)
- Mean Radial Error (MRE) in millimeters
- Success Detection Rate (SDR) at different thresholds

## ⚙️ Customization

### Adjust CVM Weight
```bash
python train_advanced.py --cvm_weight 0.5  # Increase CVM importance
```

### Change Heatmap Size
```bash
python train_advanced.py --heatmap_size 256  # Larger heatmaps
```

### Resume Training
```bash
python train_advanced.py --resume checkpoints_advanced_rtx3070ti/checkpoint_best.pth
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
- Reduce batch size: `--batch_size 2`
- Disable mixed precision (remove `--mixed_precision`)
- Reduce num_workers: `--num_workers 2`

### Slow Training
- Increase num_workers: `--num_workers 6` (if CPU allows)
- Enable mixed precision: `--mixed_precision`
- Use smaller heatmap size: `--heatmap_size 64`

### Poor Convergence
- Increase learning rate: `--lr 2e-4`
- Increase warmup epochs: `--warmup_epochs 20`
- Check data augmentation quality

## 📝 Notes

- **Stage-2 training**: Currently only Stage-1 is implemented. Stage-2 (crop refinement) can be added as a separate training script.
- **CVM calculation**: The CVM values are currently placeholders. Implement actual CVM calculation based on your requirements.
- **Synthetic braces**: The current implementation is simplified. You can improve it with more realistic brace rendering.

## 🔗 References

- HRNet: High-Resolution Representations for Labeling Pixels and Regions
- Dark-Pose: Distribution-Aware Coordinate Representation for Human Pose Estimation
- Wing Loss: Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks

