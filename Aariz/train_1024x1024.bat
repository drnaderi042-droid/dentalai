@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Training Aariz Model - 1024x1024 Multi-GPU (Optimized)
echo ========================================
echo.

REM Check GPU availability
python -c "import torch; print('GPUs:', torch.cuda.device_count() if torch.cuda.is_available() else 'No GPU found')" 2>nul
if errorlevel 1 (
    echo Warning: Could not check GPU status
)

echo.
echo Training settings (OPTIMIZED FOR SPEED):
echo   Image Size: 1024 x 1024
echo   Batch Size: 2 per GPU (with 2 GPUs = 4 total) - Safe for 12GB VRAM
echo   Gradient Accumulation: 4 steps
echo   Effective Batch Size: 16 (2 x 2 GPUs x 4 accumulation)
echo   Learning Rate: 3e-4 (optimized for 1024x1024)
echo   Epochs: 200
echo   Loss: Adaptive Wing
echo   Mixed Precision: Enabled (FP16)
echo   EMA: Enabled (for better accuracy)
echo   Multi-GPU: Enabled (using both RTX 3060 GPUs)
echo   Speed Optimizations:
echo     - torch.compile: Enabled (30-50%% speedup) - May not work on Windows
echo     - channels_last: Enabled (10-20%% speedup)
echo     - num_workers: 6 (optimized for Windows)
echo     - prefetch_factor: 1 (reduced to save memory)
echo     - Early Stopping: Enabled (patience=20)
echo   Checkpoints: checkpoints_1024x1024\
echo   Logs: logs_1024x1024\
echo.

REM Check for existing checkpoint
if exist "checkpoints_1024x1024\checkpoint_best.pth" (
    echo Checkpoint found in checkpoints_1024x1024\
    echo.
    set /p choice="Do you want to resume from previous checkpoint? (y/n): "
    
    if /i "!choice!"=="y" (
        echo.
        echo ========================================
        echo Resuming training from checkpoint
        echo ========================================
        echo.
        python train_1024x1024.py --resume checkpoints_1024x1024/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu --num_workers 6 --channels_last --early_stopping --patience 20 --save_frequency 5
    ) else (
        echo.
        echo ========================================
        echo Starting new training
        echo ========================================
        echo.
        python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu --num_workers 6 --channels_last --early_stopping --patience 20 --save_frequency 5
    )
) else (
    echo Checkpoint not found - starting new training
    echo.
    echo ========================================
    echo Starting new training - 1024x1024 Multi-GPU (Optimized)
    echo ========================================
    echo.
    echo Estimated time: 15-20 hours (with optimizations and 2x RTX 3060)
    echo   First epoch: ~10-15 minutes (includes compilation)
    echo   Subsequent epochs: ~5-8 minutes each
    echo.
    pause
    echo.
    echo Starting training...
    echo.
    python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu --num_workers 6 --channels_last --early_stopping --patience 20 --save_frequency 5
)

echo.
echo ========================================
echo Training completed!
echo ========================================
echo.
echo Results saved in checkpoints_1024x1024\ folder
echo.
pause

