@echo off
REM Fine-tuning script for combining Aariz (29 landmarks) and 3471833 (19 landmarks) datasets
REM This script fine-tunes a model trained on Aariz dataset with the new 3471833 dataset
REM Optimized for maximum GPU utilization

cd /d "%~dp0"

REM Configuration
set MODEL=hrnet
set RESUME=checkpoints\checkpoint_best.pth
set AARIZ_PATH=Aariz
set DATASET_3471833_PATH=..\3471833
set BATCH_SIZE=6
set EPOCHS=50
set LR=1e-5
set IMAGE_SIZE=512 512
REM num_workers: برای i5 9400F (6 cores) مقدار 4-6 مناسب است
REM اگر CPU شما متفاوت است، تنظیم کنید: 4-6 برای 6 cores، 8-12 برای 8+ cores
set NUM_WORKERS=6
set PREFETCH_FACTOR=4
set MIXED_PRECISION=--mixed_precision
REM torch.compile نیاز به Triton دارد که روی Windows مشکل دارد
REM به صورت پیش‌فرض غیرفعال است - اگر می‌خواهید فعال کنید، خط زیر را uncomment کنید
REM set USE_COMPILE=--use_compile
set USE_COMPILE=
set CHANNELS_LAST=--channels_last

REM Check if checkpoint exists
if not exist "%RESUME%" (
    echo ERROR: Checkpoint not found: %RESUME%
    echo Please train a model first or specify a valid checkpoint path.
    pause
    exit /b 1
)

echo ========================================
echo Fine-tuning Cephalometric Landmark Model
echo Optimized for Maximum GPU Utilization
echo ========================================
echo Model: %MODEL%
echo Checkpoint: %RESUME%
echo Aariz Path: %AARIZ_PATH%
echo 3471833 Path: %DATASET_3471833_PATH%
echo Batch Size: %BATCH_SIZE%
echo Epochs: %EPOCHS%
echo Learning Rate: %LR%
echo Image Size: %IMAGE_SIZE%
echo Num Workers: %NUM_WORKERS%
echo Prefetch Factor: %PREFETCH_FACTOR%
echo Optimizations: Mixed Precision, Channels Last
echo ========================================
echo.

python finetune_combined.py ^
    --model %MODEL% ^
    --resume %RESUME% ^
    --aariz_path %AARIZ_PATH% ^
    --dataset_3471833_path %DATASET_3471833_PATH% ^
    --batch_size %BATCH_SIZE% ^
    --epochs %EPOCHS% ^
    --lr %LR% ^
    --image_size %IMAGE_SIZE% ^
    --num_workers %NUM_WORKERS% ^
    --prefetch_factor %PREFETCH_FACTOR% ^
    --loss adaptive_wing ^
    --aariz_annotation_type "Senior Orthodontists" ^
    --dataset_3471833_annotation_type "400_senior" ^
    %MIXED_PRECISION% ^
    %USE_COMPILE% ^
    %CHANNELS_LAST%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Fine-tuning completed successfully!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Fine-tuning failed with error code: %ERRORLEVEL%
    echo ========================================
)

pause
