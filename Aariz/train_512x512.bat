@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Training Aariz Model - 512x512
echo ========================================
echo.

REM بررسی وجود checkpoint
if exist "checkpoints\checkpoint_best.pth" (
    echo ✅ Checkpoint موجود است
    echo.
    echo گزینه‌ها:
    echo   1. Fine-tuning از checkpoint 256x256 به 512x512 (پیشنهادی)
    echo   2. آموزش از اول با 512x512
    echo.
    set /p choice="انتخاب (1 یا 2): "
    
    if "!choice!"=="1" (
        echo.
        echo ========================================
        echo 🚀 Fine-tuning با 512x512
        echo ========================================
        echo.
        echo تنظیمات:
        echo   - Image Size: 512 x 512
        echo   - Batch Size: 8
        echo   - Learning Rate: 1e-5 (پایین برای fine-tuning)
        echo   - Epochs: 50
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo.
        echo زمان تقریبی: 4-6 ساعت
        echo.
        pause
        echo.
        echo شروع Fine-tuning...
        echo.
        python train2.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 1e-5 --warmup_epochs 3 --epochs 50 --loss adaptive_wing --mixed_precision
    ) else (
        echo.
        echo ========================================
        echo 🚀 آموزش از اول با 512x512
        echo ========================================
        echo.
        echo تنظیمات:
        echo   - Image Size: 512 x 512
        echo   - Batch Size: 8
        echo   - Learning Rate: 5e-4
        echo   - Epochs: 100
        echo   - Loss: Adaptive Wing
        echo   - Mixed Precision: Enabled
        echo.
        echo زمان تقریبی: 8-12 ساعت
        echo.
        pause
        echo.
        echo شروع آموزش...
        echo.
        python train2.py --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision
    )
) else (
    echo ⚠️  Checkpoint یافت نشد
    echo.
    echo ========================================
    echo 🚀 آموزش از اول با 512x512
    echo ========================================
    echo.
    echo تنظیمات:
    echo   - Image Size: 512 x 512
    echo   - Batch Size: 8
    echo   - Learning Rate: 5e-4
    echo   - Epochs: 100
    echo   - Loss: Adaptive Wing
    echo   - Mixed Precision: Enabled
    echo.
    echo زمان تقریبی: 8-12 ساعت
    echo.
    pause
    echo.
    echo شروع آموزش...
    echo.
    python train2.py --dataset_path Aariz --model hrnet --image_size 512 512 --batch_size 8 --lr 5e-4 --warmup_epochs 5 --epochs 100 --loss adaptive_wing --mixed_precision
)

echo.
echo ========================================
echo ✅ Training تمام شد!
echo ========================================
pause

