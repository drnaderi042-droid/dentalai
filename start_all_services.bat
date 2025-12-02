@echo off
REM Start all services: vite-js, minimal-api-dev-v6, and unified_ai_api_server.py
REM اجرای همزمان تمام سرویس‌ها: vite-js، minimal-api-dev-v6، و unified_ai_api_server.py

echo ============================================
echo Starting All Services...
echo شروع تمام سرویس‌ها...
echo ============================================
echo.

REM Change to project root directory
cd /d "%~dp0"

REM 1. Start vite-js (Frontend)
echo [1/3] Starting vite-js (Frontend)...
start "Vite-js Frontend" cmd /k "cd vite-js && npm run dev"

REM Wait a bit before starting next service
timeout /t 2 /nobreak >nul

REM 2. Start minimal-api-dev-v6 (Backend API)
echo [2/3] Starting minimal-api-dev-v6 (Backend API on port 7272)...
start "Minimal API Dev v6" cmd /k "cd minimal-api-dev-v6 && npm run dev"

REM Wait a bit before starting next service
timeout /t 2 /nobreak >nul

REM 3. Start unified_ai_api_server.py (AI API Server)
echo [3/3] Starting unified_ai_api_server.py (AI API Server)...
start "Unified AI API Server" cmd /k ".venv\Scripts\python.exe unified_ai_api_server.py"

echo.
echo ============================================
echo All services started!
echo تمام سرویس‌ها شروع شدند!
echo ============================================
echo.
echo Services running in separate windows:
echo سرویس‌ها در پنجره‌های جداگانه در حال اجرا هستند:
echo   - Vite-js Frontend (usually on port 3030)
echo   - Minimal API Dev v6 (on port 7272)
echo   - Unified AI API Server (usually on port 5000)
echo.
echo Press any key to exit this window...
echo برای بستن این پنجره، یک کلید را فشار دهید...
pause >nul








