@echo off
REM Fine-tuning مدل Hugging Face با Dataset Aariz

echo ================================================================================
echo Fine-tuning مدل HRNet از Hugging Face با Dataset Aariz
echo ================================================================================
echo.

REM Get the directory where this batch file is located
set SCRIPT_DIR=%~dp0
set BASE_DIR=%SCRIPT_DIR%..
set CEPHX_PATH=%BASE_DIR%\cephx_service
set AARIZ_PATH=%SCRIPT_DIR%

echo Base Directory: %BASE_DIR%
echo CephX Service: %CEPHX_PATH%
echo Aariz Directory: %AARIZ_PATH%
echo.

REM Check if venv exists
if exist "%CEPHX_PATH%\venv\Scripts\python.exe" (
    echo Virtual environment found
    echo Using Python from virtual environment...
    set PYTHON_EXE=%CEPHX_PATH%\venv\Scripts\python.exe
) else (
    echo Virtual environment not found at %CEPHX_PATH%\venv
    echo Using system Python...
    set PYTHON_EXE=python
    echo.
    echo WARNING: Make sure you have set up the virtual environment for best results
    echo.
)

REM Check if model exists
if exist "%CEPHX_PATH%\model\hrnet_cephalometric.pth" (
    echo Model found: %CEPHX_PATH%\model\hrnet_cephalometric.pth
) else (
    echo ERROR: Model not found at %CEPHX_PATH%\model\hrnet_cephalometric.pth
    echo Please download the model first!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo Fine-tuning Configuration:
echo ================================================================================
echo Model: HRNet
echo Dataset: Aariz
echo Image Size: 768x768
echo Batch Size: 4
echo Learning Rate: 1e-5 (low for fine-tuning)
echo Epochs: 50
echo Loss: Adaptive Wing
echo Mixed Precision: Enabled
echo.
echo ================================================================================
echo.

REM Change to Aariz directory
cd /d "%AARIZ_PATH%"

echo Starting fine-tuning...
echo.

REM Run fine-tuning
"%PYTHON_EXE%" train.py ^
  --model hrnet ^
  --resume "%CEPHX_PATH%\model\hrnet_cephalometric.pth" ^
  --dataset_path Aariz ^
  --image_size 768 768 ^
  --batch_size 4 ^
  --lr 1e-5 ^
  --epochs 50 ^
  --mixed_precision ^
  --loss adaptive_wing ^
  --annotation_type "Senior Orthodontists"

echo.
echo ================================================================================
echo Fine-tuning complete!
echo ================================================================================
echo.
echo Checkpoints saved in: %AARIZ_PATH%\checkpoints
echo Best model: %AARIZ_PATH%\checkpoints\checkpoint_best.pth
echo.
pause
















