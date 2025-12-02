@echo off
setlocal

echo ========================================
echo Fine-tune Combined 31-Landmark Model
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Define paths
set "MODEL=checkpoint_best_768_combined_31.pth"
set "ANNOTATIONS=annotations_p1_p2.json"
set "IMAGES=Aariz\train\Cephalograms"
set "OUTPUT=checkpoint_best_768_combined_31_finetuned.pth"

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model not found: %MODEL%
    echo Please ensure checkpoint_best_768_combined_31.pth exists in the Aariz directory.
    pause
    exit /b 1
)

REM Check if annotations exist
if not exist "%ANNOTATIONS%" (
    echo ERROR: Annotations not found: %ANNOTATIONS%
    echo Please ensure annotations_p1_p2.json exists in the Aariz directory.
    pause
    exit /b 1
)

echo Model found: %MODEL%
echo Annotations found: %ANNOTATIONS%
echo Images directory: %IMAGES%
echo Output: %OUTPUT%
echo.

echo Training settings:
echo   - Image size: 768x768
echo   - Batch size: 8 (increased for faster training)
echo   - Epochs: 20
echo   - Learning rate: 1e-4
echo   - Freeze backbone: Yes (only train final_layers)
echo   - Mixed precision: Yes (FP16 for 2x speed)
echo   - DataLoader workers: 4
echo   - Estimated time: 15-30 minutes on RTX 3070 Ti
echo.

pause

echo.
echo Starting fine-tuning...
echo.

REM Run the Python script with optimized settings
python finetune_combined_31_landmarks.py --model "%MODEL%" --annotations "%ANNOTATIONS%" --images "%IMAGES%" --output "%OUTPUT%" --image_size 768 --batch_size 8 --epochs 100 --lr 1e-4 --device cuda --freeze_backbone --mixed_precision

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Fine-tuning failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Fine-tuning Completed Successfully!
echo Output: %OUTPUT%
echo ========================================
pause
exit /b 0

