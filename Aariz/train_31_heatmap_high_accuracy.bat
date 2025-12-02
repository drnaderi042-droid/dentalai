@echo off
setlocal

echo ========================================
echo High-Accuracy 31-Landmark Heatmap Training
echo Target: < 2px MRE (Similar to hrnet_p1p2_heatmap_best.pth)
echo ========================================
echo.
echo This script trains a heatmap-based model for all 31 landmarks
echo using advanced techniques for sub-2px accuracy:
echo.
echo   ✓ HRNet-W32 backbone (larger than W18)
echo   ✓ Higher resolution heatmaps (384x384)
echo   ✓ Advanced combined loss (heatmap + coordinate)
echo   ✓ Focal loss for better heatmap training
echo   ✓ Cosine annealing with warm restarts
echo   ✓ Gradient clipping for stability
echo   ✓ Sub-pixel coordinate extraction
echo.

REM Change to script directory
cd /d "%~dp0"

REM Define paths
set "ANNOTATIONS=annotations_31.json"
set "IMAGES=Aariz\train\Cephalograms"
set "OUTPUT=models"

REM Check if annotations exist
if not exist "%ANNOTATIONS%" (
    echo ERROR: Annotations file not found: %ANNOTATIONS%
    echo Please ensure annotations_31.json exists.
    pause
    exit /b 1
)

echo Annotations: %ANNOTATIONS%
echo Images: %IMAGES%
echo Output: %OUTPUT%
echo.

echo Training configuration:
echo   - Model: HRNet-W32 (pretrained)
echo   - Image size: 768x768
echo   - Heatmap size: 384x384 (higher resolution)
echo   - Batch size: 2 (stable training)
echo   - Epochs: 300 (extensive training)
echo   - Learning rate: 0.0005 (stable)
echo   - Backbone LR: 0.00005 (frozen-like)
echo   - Coordinate loss weight: 5.0 (high precision)
echo   - Target: < 2px MRE
echo.

pause

echo.
echo Starting high-accuracy 31-landmark heatmap training...
echo This will take several hours. Be patient!
echo.

python train_31_heatmap_high_accuracy.py --annotations "%ANNOTATIONS%" --images "%IMAGES%" --output "%OUTPUT%" --epochs 300 --batch_size 2 --lr 0.0005

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: High-accuracy training failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo High-Accuracy Training Completed!
echo ========================================
echo.
echo Next steps:
echo 1. Test the model: python test_31_heatmap_model.py
echo 2. Optimize for CPU: python compress_31_model.py --input models/hrnet_31_heatmap_best.pth
echo 3. Deploy on server
echo.
pause
exit /b 0


