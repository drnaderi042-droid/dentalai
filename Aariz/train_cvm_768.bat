@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Training CVM Stage Classification Model - 768x768
echo ========================================
echo.

REM Check if checkpoint exists
if exist "CVM\checkpoints\checkpoint_best.pth" (
    echo Checkpoint exists
    echo.
    echo Options:
    echo   1. Fine-tuning from existing checkpoint
    echo   2. Train from scratch
    echo.
    set /p choice="Choice (1 or 2): "
    
    if "!choice!"=="1" (
        echo.
        echo ========================================
        echo Fine-tuning with 768x768
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 768 x 768
        echo   - Batch Size: 8
        echo   - Learning Rate: 1e-5 (low for fine-tuning)
        echo   - Epochs: 100
        echo   - Early Stopping: 30 epochs
        echo   - Mixed Precision: Enabled
        echo.
        echo Estimated time: 6-8 hours
        echo.
        pause
        echo.
        echo Starting fine-tuning...
        echo.
        "C:\Users\Salah\AppData\Local\Programs\Python\Python38\python.exe" train_cvm.py --resume CVM/checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 8 --lr 1e-5 --warmup_epochs 3 --epochs 100 --early_stopping_patience 30 --mixed_precision --save_dir CVM/checkpoints --log_dir CVM/logs
    ) else (
        echo.
        echo ========================================
        echo Training from scratch with 768x768
        echo ========================================
        echo.
        echo Settings:
        echo   - Image Size: 768 x 768
        echo   - Batch Size: 8
        echo   - Learning Rate: 5e-4
        echo   - Epochs: 100
        echo   - Early Stopping: 30 epochs
        echo   - Mixed Precision: Enabled
        echo.
        echo Estimated time: 10-14 hours
        echo.
        pause
        echo.
        echo Starting training...
        echo.
        "C:\Users\Salah\AppData\Local\Programs\Python\Python38\python.exe" train_cvm.py --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 8 --lr 5e-4 --warmup_epochs 5 --epochs 100 --early_stopping_patience 30 --mixed_precision --save_dir CVM/checkpoints --log_dir CVM/logs
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
    echo   - Batch Size: 8
    echo   - Learning Rate: 5e-4
    echo   - Epochs: 100
    echo   - Early Stopping: 30 epochs
    echo   - Mixed Precision: Enabled
    echo.
    echo Estimated time: 10-14 hours
    echo.
    pause
    echo.
    echo Starting training...
    echo.
    "C:\Users\Salah\AppData\Local\Programs\Python\Python38\python.exe" train_cvm.py --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 8 --lr 5e-4 --warmup_epochs 5 --epochs 100 --early_stopping_patience 30 --mixed_precision --save_dir CVM/checkpoints --log_dir CVM/logs
)

echo.
echo ========================================
echo Training completed!
echo ========================================
pause








