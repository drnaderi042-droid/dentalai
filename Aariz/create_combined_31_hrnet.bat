@echo off
echo ========================================
echo Create Combined HRNet Model (31 Landmarks)
echo ========================================
echo.
echo This script creates a combined HRNet model with 31 landmarks:
echo   - 29 anatomical landmarks from checkpoint_best_768.pth
echo   - 2 calibration points (P1/P2) initialized from mean weights
echo.
echo Architecture: HRNetLandmarkModel (same as checkpoint_best_768.pth)
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if main model exists
if not exist "checkpoint_best_768.pth" (
    echo ERROR: Main model not found: checkpoint_best_768.pth
    echo.
    echo Please make sure checkpoint_best_768.pth exists in the Aariz directory.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

REM Check if P1/P2 model exists
if not exist "models\hrnet_p1p2_heatmap_best.pth" (
    echo ERROR: P1/P2 model not found: models\hrnet_p1p2_heatmap_best.pth
    echo.
    echo Please make sure hrnet_p1p2_heatmap_best.pth exists in the models directory.
    echo Current directory: %CD%
    echo.
    pause
    exit /b 1
)

echo Main model found: checkpoint_best_768.pth
echo P1/P2 model found: models\hrnet_p1p2_heatmap_best.pth
echo.

echo Creating combined model...
echo.

python create_hrnet_combined_31_landmarks.py --main_model checkpoint_best_768.pth --p1p2_model models/hrnet_p1p2_heatmap_best.pth --output checkpoint_best_768_combined_31.pth

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Done! Combined model created successfully.
    echo ========================================
    echo.
    echo Output file: checkpoint_best_768_combined_31.pth
    echo.
    echo You can now use this model in unified_ai_api_server.py
    echo by changing the model path to: checkpoint_best_768_combined_31.pth
    echo.
) else (
    echo.
    echo ========================================
    echo Error occurred during model creation.
    echo ========================================
    echo.
)

pause




