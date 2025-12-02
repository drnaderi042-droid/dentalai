@echo off
REM Install facial landmark detection dependencies in .venv
REM نصب وابستگی‌های تشخیص لندمارک صورت در .venv

echo ============================================
echo Installing Facial Landmark Dependencies...
echo نصب وابستگی‌های تشخیص لندمارک صورت...
echo ============================================
echo.

REM Change to project root directory
cd /d "%~dp0"

REM Check if .venv exists
if not exist ".venv\Scripts\python.exe" (
    echo Creating .venv...
    python -m venv .venv
)

echo Installing face-alignment, mediapipe, and scikit-image...
echo نصب face-alignment، mediapipe و scikit-image...
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install face-alignment mediapipe scikit-image

echo.
echo ============================================
echo Installation complete!
echo نصب کامل شد!
echo ============================================
echo.
echo Note: If you get "Access is denied" errors, please:
echo   1. Stop the Python server (close the "AI API Server" window)
echo   2. Run this script again
echo.
echo توجه: اگر خطای "Access is denied" دریافت کردید:
echo   1. Python server را متوقف کنید (پنجره "AI API Server" را ببندید)
echo   2. این اسکریپت را دوباره اجرا کنید
echo.
pause


