@echo off
REM ========================================
REM Setup TeethDreamer - Using pip (no conda)
REM ========================================

echo ============================================================
echo     TeethDreamer Setup - راه‌اندازی با pip
echo ============================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed!
    pause
    exit /b 1
)

echo [1/6] Checking Python installation...
python --version
echo.

REM Clone repository if not exists
if not exist "TeethDreamer\" (
    echo [2/6] Cloning TeethDreamer repository...
    git clone https://github.com/ShanghaiTech-IMPACT/TeethDreamer.git
    if errorlevel 1 (
        echo ERROR: Failed to clone repository!
        pause
        exit /b 1
    )
    echo ✓ Repository cloned!
) else (
    echo [2/6] TeethDreamer folder already exists.
)
echo.

REM Extract models
echo [3/6] Extracting model files...
echo This may take 10-20 minutes (large files)...
echo.

REM Create ckpt directory
if not exist "TeethDreamer\ckpt\" (
    mkdir TeethDreamer\ckpt
)

REM Extract models
if exist "3D\TeethDreamer.zip" (
    echo Extracting TeethDreamer.ckpt...
    powershell -Command "Expand-Archive -Path '3D\TeethDreamer.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

if exist "3D\zero123-xl.zip" (
    echo Extracting zero123-xl.ckpt...
    powershell -Command "Expand-Archive -Path '3D\zero123-xl.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

if exist "3D\ViT-L-14.zip" (
    echo Extracting ViT-L-14.ckpt...
    powershell -Command "Expand-Archive -Path '3D\ViT-L-14.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

if exist "3D\sam_vit_b_01ec64.zip" (
    echo Extracting sam_vit_b_01ec64.pth...
    powershell -Command "Expand-Archive -Path '3D\sam_vit_b_01ec64.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
)

echo.
echo Checking extracted models...
dir TeethDreamer\ckpt\*.ckpt 2>nul
dir TeethDreamer\ckpt\*.pth 2>nul
echo.

REM Create virtual environment
echo [4/6] Creating Python virtual environment...
if exist "venv_teethdreamer\" (
    echo Virtual environment already exists.
) else (
    python -m venv venv_teethdreamer
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo ✓ Virtual environment created!
)
echo.

REM Activate and install
echo [5/6] Activating environment and installing packages...
call venv_teethdreamer\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch (CUDA 11.6)...
echo This may take 10-15 minutes...
pip install torch==1.12.0+cu116 torchvision==0.13.0+cu116 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu116
if errorlevel 1 (
    echo WARNING: PyTorch installation failed!
    echo Trying CPU version instead...
    pip install torch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0
)
echo.

echo [6/6] Installing requirements...
cd TeethDreamer
pip install -r requirements.txt
if errorlevel 1 (
    echo WARNING: Some packages failed to install.
    echo You may need to install manually.
)
cd ..
echo.

REM Setup instant-nsr-pl
echo Setting up Instant-NSR...
cd TeethDreamer\instant-nsr-pl
if exist "requirements.txt" (
    pip install -r requirements.txt
)
cd ..\..
echo.

echo ============================================================
echo Setup completed!
echo ============================================================
echo.
echo Model files location: TeethDreamer\ckpt\
echo.
echo Next steps:
echo 1. Activate: venv_teethdreamer\Scripts\activate
echo 2. Prepare 5 intra-oral images (0.png to 4.png)
echo 3. Run: python TeethDreamer\seg_teeth.py --img your_images --seg output
echo 4. Run: python TeethDreamer\TeethDreamer.py --test
echo.
echo For details: TEETHDREAMER_SETUP_FA.md
echo ============================================================
echo.

pause













