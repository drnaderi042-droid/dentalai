@echo off
echo ====================================
echo Test P1/P2 Calibration Model
echo ====================================
echo.

if not exist checkpoint_p1_p2.pth (
    echo ❌ Model not found!
    echo.
    echo Please train the model first:
    echo    train_p1_p2.bat
    echo.
    pause
    exit /b
)

echo Testing trained model on all 18 images...
echo.

python test_p1_p2_model.py

echo.
echo ====================================
echo Test complete!
echo ====================================
echo.
echo Check the visualization: p1_p2_prediction_best.png
echo.
pause

