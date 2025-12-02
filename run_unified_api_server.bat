@echo off
REM Unified AI API Server Launcher
REM این اسکریپت سرور API یکپارچه را اجرا می‌کند

echo ========================================
echo Unified AI API Server
echo ========================================
echo.

REM بررسی نصب Flask
python -c "import flask" 2>nul
if errorlevel 1 (
    echo [ERROR] Flask نصب نشده است!
    echo در حال نصب Flask...
    python -m pip install flask flask-cors
    if errorlevel 1 (
        echo [ERROR] خطا در نصب Flask
        pause
        exit /b 1
    )
    echo [OK] Flask با موفقیت نصب شد
    echo.
)

REM بررسی سایر dependencies
echo بررسی dependencies...
python -c "import cv2" 2>nul
if errorlevel 1 (
    echo [WARNING] opencv-python نصب نشده است
    echo در حال نصب dependencies...
    python -m pip install -r requirements_unified_api.txt
    if errorlevel 1 (
        echo [ERROR] خطا در نصب dependencies
        pause
        exit /b 1
    )
    echo [OK] Dependencies با موفقیت نصب شدند
    echo.
) else (
    python -c "import numpy, PIL" 2>nul
    if errorlevel 1 (
        echo [WARNING] برخی dependencies نصب نشده‌اند
        echo در حال نصب dependencies...
        python -m pip install -r requirements_unified_api.txt
        if errorlevel 1 (
            echo [ERROR] خطا در نصب dependencies
            pause
            exit /b 1
        )
        echo [OK] Dependencies با موفقیت نصب شدند
        echo.
    )
)

REM اجرای سرور
echo شروع سرور...
echo.
python unified_ai_api_server.py

pause
