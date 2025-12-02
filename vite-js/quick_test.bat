@echo off
REM =================================================================
REM 🚀 اسکریپت تست سریع مدل‌های AI (Windows)
REM =================================================================
REM 
REM این اسکریپت به شما کمک می‌کند تا سریعاً مدل‌های مختلف را تست کنید
REM
REM استفاده:
REM   double-click روی فایل یا در CMD اجرا کنید
REM
REM =================================================================

echo.
echo 🦷 DentalAI - تست سریع مدل‌های AI
echo ==================================================
echo.

REM بررسی Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python نصب نشده است
    echo    لطفاً ابتدا Python را از python.org نصب کنید
    pause
    exit /b 1
)

echo ✅ Python یافت شد
python --version

REM بررسی requests
python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo 📦 در حال نصب requests...
    pip install requests
)

REM درخواست API Key
echo.
set /p API_KEY="لطفاً API Key خود را وارد کنید: "

if "%API_KEY%"=="" (
    echo ❌ API Key وارد نشده است
    pause
    exit /b 1
)

REM انتخاب تصویر
echo.
echo 📷 انتخاب تصویر:
echo 1. استفاده از تصویر نمونه
echo 2. وارد کردن مسیر تصویر
echo.
set /p IMAGE_CHOICE="انتخاب (1 یا 2): "

if "%IMAGE_CHOICE%"=="1" (
    REM پیدا کردن اولین تصویر
    for %%f in (..\minimal-api-dev-v6\uploads\radiology\*.jpg ..\minimal-api-dev-v6\uploads\radiology\*.png) do (
        set IMAGE_PATH=%%f
        goto :found_image
    )
    
    echo ❌ تصویر نمونه‌ای یافت نشد
    set /p IMAGE_PATH="لطفاً مسیر تصویر را وارد کنید: "
    
    :found_image
    echo ✅ استفاده از: %IMAGE_PATH%
) else (
    set /p IMAGE_PATH="مسیر تصویر: "
)

REM بررسی وجود تصویر
if not exist "%IMAGE_PATH%" (
    echo ❌ تصویر یافت نشد: %IMAGE_PATH%
    pause
    exit /b 1
)

REM ویرایش اسکریپت Python برای قرار دادن API Key
echo.
echo 🔧 در حال تنظیم API Key...

REM ایجاد نسخه موقت با API Key
powershell -Command "(gc test_openrouter_models.py) -replace 'OPENROUTER_API_KEY = \"sk-or-v1-...\"', 'OPENROUTER_API_KEY = \"%API_KEY%\"' | Out-File -encoding ASCII temp_test.py"

REM اجرای تست
echo.
echo 🧪 شروع تست...
echo ==================================================
echo.

python temp_test.py "%IMAGE_PATH%"

REM پاک کردن فایل موقت
del temp_test.py

echo.
echo ✅ تست کامل شد!
echo 📄 نتایج در فایل test_results_*.json ذخیره شد
echo.
pause

