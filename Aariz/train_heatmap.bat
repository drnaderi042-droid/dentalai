@echo off
echo ========================================
echo HEATMAP-BASED P1/P2 Training
echo Target: ^< 10px error
echo ========================================
echo.

cd /d "%~dp0"

if not exist "annotations_p1_p2.json" (
    echo ERROR: annotations_p1_p2.json not found!
    echo Please run annotation first: annotate_p1_p2.bat
    pause
    exit /b 1
)

echo Starting heatmap-based training...
echo This approach is more accurate than direct regression.
echo Expected time: 3-5 hours
echo.
echo Press Ctrl+C to stop at any time.
echo.

python train_p1_p2_heatmap.py

echo.
echo ========================================
echo Training Complete!
echo ========================================
echo.
echo Model saved to: models/hrnet_p1p2_heatmap_best.pth
echo.
pause













