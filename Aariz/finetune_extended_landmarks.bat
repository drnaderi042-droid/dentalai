@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Fine-tune Model with Additional Landmarks
echo ========================================
echo.
echo This script fine-tunes the existing 512x512 model
echo to add new landmarks like PT (Pterygoid)
echo.
echo Prerequisites:
echo   1. Annotated images with new landmarks in JSON format
echo   2. Base model checkpoint (512x512)
echo   3. CUDA GPU (recommended)
echo.
pause

echo.
echo Configuration:
echo   - Base landmarks: 29
echo   - Additional landmarks: PT (Pterygoid)
echo   - Model: HRNet
echo   - Image size: 512x512
echo.

REM Set your checkpoint path here
set CHECKPOINT_PATH=checkpoints\checkpoint_best_512x512.pth

REM Check if checkpoint exists
if not exist "%CHECKPOINT_PATH%" (
    echo ERROR: Checkpoint not found: %CHECKPOINT_PATH%
    echo Please update CHECKPOINT_PATH in this script.
    pause
    exit /b 1
)

echo Using checkpoint: %CHECKPOINT_PATH%
echo.

python finetune_extended_landmarks.py ^
    --checkpoint "%CHECKPOINT_PATH%" ^
    --additional_landmarks PT ^
    --batch_size 4 ^
    --epochs 50 ^
    --lr 1e-4 ^
    --image_size 512 512 ^
    --num_workers 4 ^
    --save_dir checkpoints_extended ^
    --log_dir logs_extended ^
    --model hrnet ^
    --mixed_precision

echo.
echo ========================================
echo Fine-tuning completed!
echo ========================================
echo.
echo Results saved to:
echo   - checkpoints_extended/checkpoint_best.pth
echo   - logs_extended/ (TensorBoard logs)
echo.
pause
















