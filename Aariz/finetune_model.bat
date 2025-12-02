@echo off
echo ========================================
echo Fine-tuning Aariz Model
echo ========================================
echo.

cd /d "%~dp0"

REM Backup checkpoint
echo [1/3] Creating backup...
if exist "checkpoints\checkpoint_best.pth" (
    copy "checkpoints\checkpoint_best.pth" "checkpoints\checkpoint_best_backup.pth" >nul
    echo    Backup created: checkpoint_best_backup.pth
) else (
    echo    ERROR: checkpoint_best.pth not found!
    pause
    exit /b 1
)

echo.
echo [2/3] Starting fine-tuning...
echo    Model: HRNet
echo    Image Size: 256x256
echo    Learning Rate: 1e-5 (fine-tuning)
echo    Epochs: 30
echo    Batch Size: 16
echo.

python train2.py ^
    --resume checkpoints/checkpoint_best.pth ^
    --dataset_path Aariz ^
    --model hrnet ^
    --image_size 256 256 ^
    --batch_size 16 ^
    --lr 1e-5 ^
    --warmup_epochs 2 ^
    --epochs 30 ^
    --loss adaptive_wing ^
    --mixed_precision

echo.
echo [3/3] Fine-tuning complete!
echo    Check logs/ for Tensorboard logs
echo    Check checkpoints/ for new checkpoints
echo.
pause

