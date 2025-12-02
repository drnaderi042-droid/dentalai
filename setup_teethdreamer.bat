@echo off
REM ========================================
REM Setup TeethDreamer - Complete Guide
REM ========================================

echo ============================================================
echo     TeethDreamer Setup - راه‌اندازی کامل
echo ============================================================
echo.

REM Check if conda is installed
where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: Conda is not installed!
    echo Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/products/individual
    pause
    exit /b 1
)

echo [1/7] Checking Conda installation...
conda --version
echo.

REM Check if TeethDreamer exists
if exist "TeethDreamer\" (
    echo [2/7] TeethDreamer folder exists.
    echo Do you want to remove and re-clone? (Y/N)
    set /p reinstall="Enter choice: "
    if /i "%reinstall%"=="Y" (
        echo Removing existing folder...
        rmdir /s /q TeethDreamer
    ) else (
        echo Using existing folder...
        goto :extract_models
    )
)

REM Clone repository
echo [2/7] Cloning TeethDreamer repository...
git clone https://github.com/ShanghaiTech-IMPACT/TeethDreamer.git
if errorlevel 1 (
    echo ERROR: Failed to clone repository!
    echo Please check your internet connection.
    pause
    exit /b 1
)
echo ✓ Repository cloned successfully!
echo.

:extract_models
REM Extract models
echo [3/7] Extracting model files...
echo This may take several minutes...
echo.

if not exist "3D\TeethDreamer.zip" (
    echo ERROR: TeethDreamer.zip not found in 3D folder!
    pause
    exit /b 1
)

REM Create ckpt directory
if not exist "TeethDreamer\ckpt\" (
    mkdir TeethDreamer\ckpt
)

REM Extract TeethDreamer model
echo Extracting TeethDreamer.ckpt...
powershell -Command "Expand-Archive -Path '3D\TeethDreamer.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
if errorlevel 1 (
    echo WARNING: Failed to extract TeethDreamer.zip
)

REM Extract zero123-xl
if exist "3D\zero123-xl.zip" (
    echo Extracting zero123-xl.ckpt...
    powershell -Command "Expand-Archive -Path '3D\zero123-xl.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

REM Extract ViT-L-14
if exist "3D\ViT-L-14.zip" (
    echo Extracting ViT-L-14.ckpt...
    powershell -Command "Expand-Archive -Path '3D\ViT-L-14.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

REM Extract SAM
if exist "3D\sam_vit_b_01ec64.zip" (
    echo Extracting sam_vit_b_01ec64.pth...
    powershell -Command "Expand-Archive -Path '3D\sam_vit_b_01ec64.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

echo.
echo Checking extracted models...
if exist "TeethDreamer\ckpt\TeethDreamer.ckpt" (
    echo ✓ TeethDreamer.ckpt found
) else (
    echo ✗ TeethDreamer.ckpt NOT found!
)

if exist "TeethDreamer\ckpt\zero123-xl.ckpt" (
    echo ✓ zero123-xl.ckpt found
) else (
    echo ✗ zero123-xl.ckpt NOT found!
)

if exist "TeethDreamer\ckpt\ViT-L-14.ckpt" (
    echo ✓ ViT-L-14.ckpt found
) else (
    echo ✗ ViT-L-14.ckpt NOT found!
)

if exist "TeethDreamer\ckpt\sam_vit_b_01ec64.pth" (
    echo ✓ sam_vit_b_01ec64.pth found
) else (
    echo ✗ sam_vit_b_01ec64.pth NOT found!
)

echo.
echo [4/7] Creating Conda environment...
call conda create -n TeethDreamer python=3.8 -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment!
    pause
    exit /b 1
)
echo ✓ Environment created!
echo.

echo [5/7] Installing PyTorch...
call conda activate TeethDreamer
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
if errorlevel 1 (
    echo WARNING: PyTorch installation may have failed.
    echo You may need to install manually.
)
echo.

echo [6/7] Installing requirements...
cd TeethDreamer
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages may have failed to install.
    echo Check requirements.txt manually.
)
echo.

echo [7/7] Setting up Instant-NSR...
cd instant-nsr-pl
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo WARNING: instant-nsr-pl requirements.txt not found
)
cd ..
echo.

echo ============================================================
echo Setup completed!
echo ============================================================
echo.
echo Next steps:
echo 1. Activate environment: conda activate TeethDreamer
echo 2. Prepare your 5 intra-oral images
echo 3. Run segmentation: python seg_teeth.py
echo 4. Run generation: python TeethDreamer.py --test
echo.
echo For detailed instructions, see: TEETHDREAMER_SETUP_FA.md
echo ============================================================
echo.

pause













