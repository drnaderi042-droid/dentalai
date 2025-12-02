@echo off
REM اسکریپت اجرای تست کامل HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth

echo ================================================================================
echo تست کامل HRNet - مقایسه Python Direct vs Frontend API vs Ground Truth
echo ================================================================================
echo.
echo حالت‌های تست:
echo   1. python   - فقط تست Python Direct
echo   2. frontend - فقط تست Frontend API
echo   3. all      - تست هر دو و مقایسه (پیش‌فرض)
echo.
set /p MODE="لطفاً حالت تست را انتخاب کنید (1/2/3) [پیش‌فرض: 3]: "

if "%MODE%"=="1" set TEST_MODE=python
if "%MODE%"=="2" set TEST_MODE=frontend
if "%MODE%"=="" set TEST_MODE=all
if "%MODE%"=="3" set TEST_MODE=all
if not defined TEST_MODE set TEST_MODE=all

echo.
echo حالت انتخاب شده: %TEST_MODE%
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

REM Change to Aariz directory
cd /d "%AARIZ_PATH%"

echo.
echo Running test script with mode: %TEST_MODE%
echo Using Python: %PYTHON_EXE%
echo.

REM Run the test script using venv Python
"%PYTHON_EXE%" test_hrnet_python_frontend_comparison.py --mode %TEST_MODE%

echo.
echo ================================================================================
echo تست کامل شد!
echo ================================================================================
pause

