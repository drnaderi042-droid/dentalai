@echo off
echo ===================================
echo Starting Optimized Training
echo ===================================
echo.

python test_setup.py
if errorlevel 1 (
    echo.
    echo ERROR: Setup test failed!
    echo Please fix the errors above before training.
    pause
    exit /b 1
)

echo.
echo ===================================
echo Setup OK! Starting training...
echo ===================================
echo.

python train_optimized.py --mixed_precision --use_ema

echo.
echo ===================================
echo Training finished or stopped
echo ===================================
echo Check training_log.txt for details
pause