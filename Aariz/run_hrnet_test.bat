@echo off
cd /d "%~dp0"
echo ========================================
echo تست HRNet از طریق API
echo ========================================
echo.

REM Go to parent directory
cd /d "%~dp0\.."

REM Check if venv exists
if not exist "cephx_service\venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please make sure venv is set up in cephx_service directory
    pause
    exit /b 1
)

echo Using Python from virtual environment...
echo.

REM Use venv Python to run the test
cephx_service\venv\Scripts\python.exe Aariz\test_hrnet_full_comparison.py

pause

