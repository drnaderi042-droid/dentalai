@echo off
REM اسکریپت نصب Dependency های لازم برای بهینه‌سازی CLdetection2023
REM Install script for CLdetection2023 optimization dependencies (Windows)

echo ======================================================================
echo 🚀 CLdetection2023 Optimization Dependencies Installer (Windows)
echo ======================================================================
echo.

REM Check if .venv exists
if exist ".venv\Scripts\python.exe" (
    echo ✅ Virtual environment found
    set PYTHON_CMD=.venv\Scripts\python.exe
    set PIP_CMD=.venv\Scripts\pip.exe
) else (
    echo ⚠️  Virtual environment not found, using system Python
    set PYTHON_CMD=python
    set PIP_CMD=pip
)

echo.
echo 📦 Installing required packages...
echo.

REM Install PyTorch 2.0+ (CPU version)
echo Installing PyTorch 2.0+ (CPU)...
%PIP_CMD% install --upgrade torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

REM Install MKL for faster CPU operations
echo.
echo Installing Intel MKL...
%PIP_CMD% install --upgrade mkl mkl-service

REM Optional: Install ONNX (uncomment if needed)
REM echo.
REM echo Installing ONNX (optional)...
REM %PIP_CMD% install --upgrade onnx onnxruntime

REM Optional: Install Numba (uncomment if needed)
REM echo.
REM echo Installing Numba (optional)...
REM %PIP_CMD% install --upgrade numba

echo.
echo ======================================================================
echo ✅ Installation Complete!
echo ======================================================================
echo.
echo 📝 Next steps:
echo    1. Restart your Python server
echo    2. Test the optimizations with a CLdetection request
echo    3. Check logs for 'torch.compile enabled' message
echo.
echo 💡 Tips:
echo    - If torch.compile is not available, upgrade PyTorch: pip install torch>=2.0.0
echo    - For maximum speed, consider ONNX conversion
echo.
pause










