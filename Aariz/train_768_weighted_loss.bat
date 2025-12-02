@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Fine-tuning با Weighted Loss - 768x768
echo ========================================
echo.

REM بررسی وجود checkpoint
if not exist "checkpoint_best_768.pth" (
    echo [ERROR] Checkpoint يافت نشد: checkpoint_best_768.pth
    echo لطفا مطمئن شوید که checkpoint موجود است.
    pause
    exit /b 1
)

echo Settings:
echo   - Image Size: 768 x 768
echo   - Batch Size: 6 (افزایش برای استفاده بیشتر از GPU memory)
echo   - Learning Rate: 1e-6 (پایین برای fine-tuning)
echo   - Epochs: 60 (از epoch 36 تا 59)
echo   - Loss: Adaptive Wing (فقط لندمارک‌های مشکل‌دار)
echo   - Mixed Precision: Enabled (FP16)
echo   - num_workers: 6 (برای سرعت بیشتر در data loading)
echo   - Training: فقط 12 لندمارک مشکل‌دار (بقیه ignore می‌شوند)
echo.
echo لندمارک‌هایی که آموزش می‌بینند:
echo   - UMT, UPM, R, Ar, Go, LMT, LPM, Or, Co, PNS, Po, ANS
echo   - مجموع: 12 لندمارک از 29 لندمارک
echo.
echo لندمارک‌های ignore شده (Me, Gn, UIT, S, Pn, Li, Pog, B, A, N, ...):
echo   - 17 لندمارک که قبلا خوب بودند ignore می‌شوند
echo.
echo بهینه‌سازی‌ها برای مصرف بیشتر GPU Memory:
echo   - Batch Size: 6 (افزایش از 4 برای استفاده ~7.5GB)
echo   - num_workers: 6 (افزایش سرعت data loading)
echo   - Mixed Precision: فعال (FP16)
echo   - Prefetch Factor: افزایش برای buffering بیشتر
echo.
echo Estimated GPU Memory Usage: ~7.5 GB
echo Estimated time: 2-3 hours
echo.
pause
echo.
echo Starting Fine-tuning with Increased Memory Usage...
echo.

python train2.py --resume checkpoint_best_768.pth --dataset_path Aariz --model hrnet --image_size 768 768 --batch_size 6 --gradient_accumulation_steps 1 --lr 1e-6 --warmup_epochs 2 --epochs 60 --loss adaptive_wing --mixed_precision --num_workers 6

echo.
echo ========================================
echo Training completed!
echo ========================================
echo.
echo Checkpoint جدید در checkpoints/checkpoint_best.pth ذخیره شده است.
echo برای تست نتایج، اجرا کنید:
echo   python test_768_validation_full.py
echo.
pause

