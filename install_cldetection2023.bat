@echo off
REM Installation script for CLdetection2023 model dependencies (Windows)
REM This script installs MMPose and its dependencies required for CLdetection2023 model

echo ============================================================
echo CLdetection2023 Model Dependencies Installation (Windows)
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo Python found: 
python --version
echo.

REM Get the directory where this script is located
set SCRIPT_DIR=%~dp0
set CLDETECTION_DIR=%SCRIPT_DIR%CLdetection2023
set MMPOSE_DIR=%CLDETECTION_DIR%\mmpose_package\mmpose

REM Check if CLdetection2023 directory exists
if not exist "%CLDETECTION_DIR%" (
    echo ERROR: CLdetection2023 directory not found: %CLDETECTION_DIR%
    echo Make sure CLdetection2023 folder exists in the project root
    pause
    exit /b 1
)

REM Check if mmpose_package exists
if not exist "%MMPOSE_DIR%" (
    echo ERROR: MMPose package directory not found: %MMPOSE_DIR%
    echo Make sure CLdetection2023\mmpose_package\mmpose exists
    pause
    exit /b 1
)

echo ============================================================
echo Step 1: Installing openmim
echo ============================================================
python -m pip install -U openmim
if errorlevel 1 (
    echo WARNING: Failed to install openmim, but continuing...
)

echo.
echo ============================================================
echo Step 2: Installing mmengine
echo ============================================================
python -m mim install mmengine
if errorlevel 1 (
    echo WARNING: Failed to install mmengine, but continuing...
)

echo.
echo ============================================================
echo Step 3: Removing existing mmcv (if incompatible)
echo ============================================================
python -m pip uninstall mmcv mmcv-full -y
if errorlevel 1 (
    echo INFO: No existing mmcv to remove, or already removed
)

echo.
echo ============================================================
echo Step 4: Installing mmcv (compatible version)
echo ============================================================
python -m mim install "mmcv>=2.0.0rc4,<=2.1.0"
if errorlevel 1 (
    echo WARNING: Failed to install mmcv, but continuing...
)

echo.
echo ============================================================
echo Step 5: Installing MMPose
echo ============================================================
cd /d "%MMPOSE_DIR%"
python -m pip install -e .
if errorlevel 1 (
    echo WARNING: Failed to install MMPose, but continuing...
)
cd /d "%SCRIPT_DIR%"

echo.
echo ============================================================
echo Step 6: Upgrading numpy
echo ============================================================
python -m pip install --upgrade numpy
if errorlevel 1 (
    echo WARNING: Failed to upgrade numpy, but continuing...
)

echo.
echo ============================================================
echo Verifying Installation
echo ============================================================

python -c "import mmengine; print('✅ mmengine installed:', mmengine.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ mmengine not installed
)

python -c "import mmcv; print('✅ mmcv installed:', mmcv.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ mmcv not installed
)

python -c "import sys; sys.path.insert(0, r'%MMPOSE_DIR%'); import mmpose; print('✅ mmpose installed:', mmpose.__version__)" 2>nul
if errorlevel 1 (
    echo ❌ mmpose not installed
    echo Try running: cd CLdetection2023\mmpose_package\mmpose ^&^& pip install -e .
)

python -c "import sys; sys.path.insert(0, r'%MMPOSE_DIR%'); from mmpose.apis import init_model; print('✅ mmpose.apis imported successfully')" 2>nul
if errorlevel 1 (
    echo ❌ mmpose.apis import failed
)

echo.
echo ============================================================
echo SimpleITK Status (Optional)
echo ============================================================
python -c "import SimpleITK; print('✅ SimpleITK installed (optional)')" 2>nul
if errorlevel 1 (
    echo ℹ️  SimpleITK not installed (optional - only needed for training data)
    echo    Inference will work without SimpleITK
)

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo Next steps:
echo 1. Restart the unified_ai_api_server.py
echo 2. Test the CLdetection2023 model endpoint
echo 3. Check the server logs for any errors
echo.
pause

