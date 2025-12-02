@echo off
echo ============================================
echo Quick dlib Installation for Python 3.8
echo ============================================
echo.

echo Method 1: Installing pre-built wheel (Recommended)
echo.
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp38-cp38-win_amd64.whl

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ dlib installed successfully!
    echo.
    echo Now download the shape predictor file:
    echo http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    echo Extract it and place in facial-landmark-detection folder
    pause
    exit /b 0
)

echo.
echo Method 1 failed. Trying Method 2: dlib-bin
echo.
pip install dlib-bin

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✅ dlib installed successfully!
    pause
    exit /b 0
)

echo.
echo Both methods failed. You need to install CMake first.
echo.
echo Please:
echo 1. Download CMake from https://cmake.org/download/
echo 2. Install it and select "Add CMake to system PATH"
echo 3. Restart this terminal
echo 4. Run this script again
echo.
pause

















