@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Advanced Training Script
echo HRNet-W48 + Multi-task + Two-stage
echo ========================================
echo.
echo Configuration:
echo   - Backbone: HRNet-W48
echo   - Input Size: 512x512
echo   - Heatmap Size: 128x128
echo   - Loss: Dark-Pose + Wing Loss (0.5 + 0.5)
echo   - Multi-task: Landmark + CVM (weight: 0.3)
echo   - Data Augmentation: 6500 images total
echo   - Hardware: RTX 3070 Ti + CPU i5 9400F
echo.
echo Dataset Augmentation Types:
echo   - Original: 1000 images
echo   - Flipped: 1000 images
echo   - Rotated ±7°: 2000 images
echo   - Brightness: 1000 images
echo   - Synthetic braces: 500 images
echo   - Total: 6500 images
echo.
pause

echo.
echo Starting training...
echo.

python train_advanced.py ^
    --dataset_path Aariz ^
    --batch_size 6 ^
    --epochs 200 ^
    --lr 1e-4 ^
    --warmup_epochs 10 ^
    --image_size 512 512 ^
    --heatmap_size 128 ^
    --num_workers 6 ^
    --mixed_precision ^
    --cvm_weight 0.3 ^
    --save_dir checkpoints_advanced_rtx3070ti ^
    --log_dir logs_advanced_rtx3070ti

echo.
echo ========================================
echo Training completed!
echo ========================================
pause

