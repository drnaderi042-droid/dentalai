@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Training Aariz Model - 768x768
echo ========================================
echo.

REM Check if checkpoint exists
if exist "checkpoints\checkpoint_best.pth" (
    echo Checkpoint found
    echo.
    echo Options:
    echo   1. Fine-tuning from checkpoint 512x512 to 768x768 (Recommended)
    echo   2. Train from scratch with 768x768
    echo.
    set /p choice="Select (1 or 2): "
    
    if "!choice!"=="1" (
        echo.
        echo ========================================
        echo Fine-tuning with 768x768
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 768 x 768
        echo   - Batch Size: 4
        echo   - Learning Rate: 1e-5 (low for fine-tuning)
        echo   - Epochs: 50
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo.
        echo Estimated time: 6-8 hours
        echo.
        pause
        echo.
        echo Starting Fine-tuning...
        echo.
        python train2.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 4 --lr 1e-5 --warmup_epochs 3 --epochs 50 --loss adaptive_wing --mixed_precision
    ) else (
        echo.
        echo ========================================
        echo Training from scratch with 768x768
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 768 x 768
        echo   - Batch Size: 4
        echo   - Learning Rate: 5e-4
        echo   - Epochs: 100
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo.
        echo Estimated time: 12-16 hours
        echo.
        pause
        echo.
        echo Starting training...
        echo.
        python train2.py --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 4 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision
    )
) else (
    echo Checkpoint not found
    echo.
    echo ========================================
    echo Training from scratch with 768x768
    echo ========================================
    echo.
    echo Settings:
    echo   - Image Size: 768 x 768
    echo   - Batch Size: 4
    echo   - Learning Rate: 5e-4
    echo   - Epochs: 100
    echo   - Loss: Adaptive Wing
    echo   - Mixed Precision: Enabled
    echo.
    echo Estimated time: 12-16 hours
    echo.
    pause
    echo.
    echo Starting training...
    echo.
    python train2.py --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 4 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision
)

echo.
echo ========================================
echo Training completed!
echo ========================================
pause
