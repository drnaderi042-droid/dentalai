@echo off
setlocal

echo ========================================
echo Improved Fine-tuning Combined 31-Landmark Model
echo ========================================
echo.
echo This script uses advanced techniques:
echo   1. Unfreeze backbone with lower learning rate
echo   2. Focal loss for heatmap prediction
echo   3. Data augmentation
echo   4. Cosine annealing scheduler
echo   5. Higher coordinate loss weight (20.0)
echo.

REM Change to script directory
cd /d "%~dp0"

REM Define paths
set "MODEL=checkpoint_best_768_combined_31_finetuned.pth"
set "ANNOTATIONS=annotations_p1_p2.json"
set "IMAGES=Aariz\train\Cephalograms"
set "OUTPUT=checkpoint_best_768_combined_31_improved.pth"

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model not found: %MODEL%
    echo Please ensure the fine-tuned model exists.
    pause
    exit /b 1
)

echo Model: %MODEL%
echo Annotations: %ANNOTATIONS%
echo Images: %IMAGES%
echo Output: %OUTPUT%
echo.

echo Training settings:
echo   - Image size: 768x768
echo   - Batch size: 8
echo   - Epochs: 50
echo   - Backbone LR: 1e-5 (lower for stable training)
echo   - Final layers LR: 1e-4 (higher for faster adaptation)
echo   - Focal loss: Enabled
echo   - Data augmentation: Enabled
echo   - Mixed precision: Enabled
echo.

pause

echo.
echo Starting improved fine-tuning...
echo.

python improve_finetune_combined_31.py --model "%MODEL%" --annotations "%ANNOTATIONS%" --images "%IMAGES%" --output "%OUTPUT%" --epochs 150 --backbone_lr 1e-5 --final_layers_lr 1e-4 --batch_size 8

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Improved fine-tuning failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Improved Fine-tuning Completed!
echo Output: %OUTPUT%
echo ========================================
pause
exit /b 0




