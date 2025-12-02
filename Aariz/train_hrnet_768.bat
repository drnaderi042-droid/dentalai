@echo off
echo ========================================
echo HRNet P1/P2 Training - 768px Resolution
echo Optimized for RTX 3070 Ti (8GB VRAM)
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Configuration:
echo   - Model: HRNet-W18
echo   - Image Size: 768x768 px
echo   - Batch Size: 2
echo   - Epochs: 200
echo   - GPU: CUDA (RTX 3070 Ti)
echo   - Dataset: 100 images (80 train / 20 val)
echo.
echo Expected training time: 3-5 hours
echo.

REM Check if annotations exist
if not exist "annotations_p1_p2.json" (
    echo ERROR: annotations_p1_p2.json not found!
    echo.
    echo Please run annotation first:
    echo   annotate_p1_p2.bat
    echo.
    pause
    exit /b 1
)

REM Check if images directory exists
if not exist "Aariz\train\Cephalograms" (
    echo ERROR: Aariz/train/Cephalograms directory not found!
    echo.
    pause
    exit /b 1
)

echo Starting training...
echo.
echo Press Ctrl+C to stop training at any time.
echo Best model will be saved to: models/hrnet_p1p2_best_hrnet_w18.pth
echo.
echo ========================================
echo.

python train_p1_p2_hrnet.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Model saved to: models/hrnet_p1p2_best_hrnet_w18.pth
echo.
echo Next steps:
echo   1. Test model: test_hrnet.bat
echo   2. Check results: test_results_hrnet/
echo.
pause













