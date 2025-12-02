@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Training Aariz Model - 1024x1024
echo RTX 3070 Ti Optimized
echo ========================================
echo.

REM Check if checkpoint exists
if exist "checkpoints\checkpoint_best.pth" (
    echo Checkpoint found
    echo.
    echo Options:
    echo   1. Fine-tuning from checkpoint 512x512 to 1024x1024 (Recommended)
    echo   2. Train from scratch with 1024x1024
    echo.
    set /p choice="Select (1 or 2): "
    
    if "!choice!"=="1" (
        echo.
        echo ========================================
        echo Fine-tuning with 1024x1024
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 1024 x 1024
        echo   - Batch Size: 2 (reduced for 1024x1024)
        echo   - Gradient Accumulation: 3 (effective batch size: 6)
        echo   - Learning Rate: 1e-5 (low for fine-tuning)
        echo   - Epochs: 100
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo   - Num Workers: 2
        echo   - CPU RAM Cache: Limited (100 samples max to save RAM)
        echo.
        echo Estimated time: 8-12 hours
        echo.
        pause
        echo.
        echo Starting Fine-tuning...
        echo.
        python train_1024x1024.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 3 --lr 1e-5 --warmup_epochs 3 --epochs 100 --loss adaptive_wing --mixed_precision --num_workers 2 --limited_ram_cache 100 --save_dir checkpoints_1024x1024 --log_dir logs_1024x1024
    ) else (
        echo.
        echo ========================================
        echo Training from scratch with 1024x1024
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 1024 x 1024
        echo   - Batch Size: 2
        echo   - Gradient Accumulation: 3 (effective batch size: 6)
        echo   - Learning Rate: 5e-4
        echo   - Epochs: 100
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo   - Num Workers: 2
        echo   - CPU RAM Cache: Limited (100 samples max to save RAM)
        echo.
        echo Estimated time: 15-20 hours
        echo.
        pause
        echo.
        echo Starting training...
        echo.
        python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 3 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision --num_workers 2 --limited_ram_cache 100 --save_dir checkpoints_1024x1024 --log_dir logs_1024x1024
    )
) else (
    echo Checkpoint not found
    echo.
    echo ========================================
    echo Training from scratch with 1024x1024
    echo ========================================
    echo.
    echo Settings:
    echo   - Image Size: 1024 x 1024
    echo   - Batch Size: 2
    echo   - Gradient Accumulation: 3 (effective batch size: 6)
    echo   - Learning Rate: 5e-4
    echo   - Epochs: 100
    echo   - Loss: Adaptive Wing
    echo   - Mixed Precision: Enabled
    echo   - Num Workers: 2
    echo   - CPU RAM Cache: Enabled (48GB available)
    echo.
    echo Estimated time: 15-20 hours
    echo.
    pause
    echo.
    echo Starting training...
    echo.
    python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 3 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision --num_workers 2 --use_ram_cache --save_dir checkpoints_1024x1024 --log_dir logs_1024x1024
)

echo.
echo ========================================
echo Training completed!
echo ========================================
pause
