@echo off
echo ============================================
echo Installing dlib for Windows
echo ============================================
echo.
echo Step 1: Installing CMake
echo.
echo Please download CMake from: https://cmake.org/download/
echo Select "Windows x64 Installer"
echo.
echo IMPORTANT: During installation, select "Add CMake to system PATH"
echo.
pause

echo.
echo Step 2: Installing cmake Python package
pip install cmake

echo.
echo Step 3: Installing dlib
echo.
echo Option A: Try installing from source (requires CMake)
pip install dlib

echo.
echo If that fails, try Option B: Pre-built wheel
echo.
echo For Python 3.9:
echo pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl
echo.
echo For Python 3.10:
echo pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp310-cp310-win_amd64.whl
echo.
echo For Python 3.11:
echo pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp311-cp311-win_amd64.whl
echo.
pause

















