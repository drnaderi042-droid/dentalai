@echo off
setlocal

echo ========================================
echo Resume Fine-tuning Combined 31-Landmark Model
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Define paths
set "RESUME_CHECKPOINT=checkpoint_best_768_combined_31_finetuned_last.pth"
set "ANNOTATIONS=annotations_p1_p2.json"
set "IMAGES=Aariz\train\Cephalograms"
set "OUTPUT=checkpoint_best_768_combined_31_finetuned.pth"

REM Check if resume checkpoint exists
if not exist "%RESUME_CHECKPOINT%" (
    echo ERROR: Resume checkpoint not found: %RESUME_CHECKPOINT%
    echo Please ensure the checkpoint file exists.
    echo.
    echo To create a resume checkpoint, run finetune_combined_31.bat first.
    echo The last checkpoint is automatically saved as: %RESUME_CHECKPOINT%
    pause
    exit /b 1
)

echo Resume checkpoint found: %RESUME_CHECKPOINT%
echo Annotations: %ANNOTATIONS%
echo Images directory: %IMAGES%
echo Output: %OUTPUT%
echo.

echo Training settings:
echo   - Image size: 768x768
echo   - Batch size: 8
echo   - Epochs: 20 (will continue from last checkpoint)
echo   - Learning rate: 1e-4 (will use saved LR if available)
echo   - Freeze backbone: Yes
echo   - Mixed precision: Yes (FP16)
echo.

pause

echo.
echo Resuming fine-tuning...
echo.

REM Run the Python script with resume option
python finetune_combined_31_landmarks.py --annotations "%ANNOTATIONS%" --images "%IMAGES%" --output "%OUTPUT%" --image_size 768 --batch_size 8 --epochs 200 --lr 1e-4 --device cuda --freeze_backbone --mixed_precision --resume "%RESUME_CHECKPOINT%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Resume fine-tuning failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Resume Fine-tuning Completed Successfully!
echo Output: %OUTPUT%
echo ========================================
pause
exit /b 0




