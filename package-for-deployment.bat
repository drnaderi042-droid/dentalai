@echo off
echo ============================================
echo    DentalAI - Package Files for Deployment
echo ============================================

echo.
echo This script will package all necessary files for deployment
echo.

set /p SERVER_IP="Enter your Ubuntu server IP: "
set /p SERVER_USER="Enter SSH username: "
set /p PROJECT_PATH="Project path (default: C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy): "

if "%PROJECT_PATH%"=="" set PROJECT_PATH=C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy

echo.
echo ============================================
echo Checking built files...
echo ============================================

REM Check if vite-js dist exists
if not exist "%PROJECT_PATH%\vite-js\dist" (
    echo ERROR: vite-js\dist not found! Please build vite-js first.
    echo Run: cd vite-js ^& npm run build
    pause
    exit /b 1
)

REM Check if minimal-api-dev-v6 .next exists
if not exist "%PROJECT_PATH%\minimal-api-dev-v6\.next" (
    echo ERROR: minimal-api-dev-v6\.next not found! Please build minimal-api-dev-v6 first.
    echo Run: cd minimal-api-dev-v6 ^& npm run build
    pause
    exit /b 1
)

REM Check if AI server exists
if not exist "%PROJECT_PATH%\unified_ai_api_server.py" (
    echo ERROR: unified_ai_api_server.py not found!
    pause
    exit /b 1
)

echo ✅ All required files found!

echo.
echo ============================================
echo Creating deployment package...
echo ============================================

REM Create temp directory for packaging
if exist "deployment-temp" rmdir /s /q "deployment-temp"
mkdir "deployment-temp"

echo Creating directory structure...
mkdir "deployment-temp\dentalai"
mkdir "deployment-temp\dentalai\frontend"
mkdir "deployment-temp\dentalai\backend"
mkdir "deployment-temp\dentalai\ai-server"
mkdir "deployment-temp\dentalai\config"

echo Copying frontend files...
xcopy "%PROJECT_PATH%\vite-js\dist" "deployment-temp\dentalai\frontend\" /E /I /H /Y >nul
if %errorlevel% neq 0 (
    echo ERROR: Failed to copy frontend files
    pause
    exit /b 1
)

echo Copying backend files...
xcopy "%PROJECT_PATH%\minimal-api-dev-v6\.next" "deployment-temp\dentalai\backend\.next\" /E /I /H /Y >nul
copy "%PROJECT_PATH%\minimal-api-dev-v6\package.json" "deployment-temp\dentalai\backend\" >nul
copy "%PROJECT_PATH%\minimal-api-dev-v6\next.config.mjs" "deployment-temp\dentalai\backend\" >nul
xcopy "%PROJECT_PATH%\minimal-api-dev-v6\prisma" "deployment-temp\dentalai\backend\prisma\" /E /I /H /Y >nul

echo Copying AI server files...
copy "%PROJECT_PATH%\unified_ai_api_server.py" "deployment-temp\dentalai\ai-server\" >nul
copy "%PROJECT_PATH%\requirements_unified_api.txt" "deployment-temp\dentalai\ai-server\" >nul
if exist "%PROJECT_PATH%\cephx_service" (
    xcopy "%PROJECT_PATH%\cephx_service" "deployment-temp\dentalai\ai-server\cephx_service\" /E /I /H /Y >nul
)
if exist "%PROJECT_PATH%\facial-landmark-detection" (
    xcopy "%PROJECT_PATH%\facial-landmark-detection" "deployment-temp\dentalai\ai-server\facial-landmark-detection\" /E /I /H /Y >nul
)
if exist "%PROJECT_PATH%\CLdetection2023" (
    xcopy "%PROJECT_PATH%\CLdetection2023" "deployment-temp\dentalai\ai-server\CLdetection2023\" /E /I /H /Y >nul
)

echo Copying configuration files...
if exist "env.example" copy "env.example" "deployment-temp\dentalai\config\" >nul
if exist "docker-compose.yml" copy "docker-compose.yml" "deployment-temp\dentalai\config\" >nul
if exist "nginx" xcopy "nginx" "deployment-temp\dentalai\config\nginx\" /E /I /H /Y >nul

echo Creating deployment archive...
if exist "dentalai-deployment.tar.gz" del "dentalai-deployment.tar.gz"
tar -czf "dentalai-deployment.tar.gz" -C "deployment-temp" .

echo Cleaning up temp files...
rmdir /s /q "deployment-temp"

echo.
echo ============================================
echo ✅ Deployment package created successfully!
echo ============================================
echo.
echo Package: dentalai-deployment.tar.gz
echo Size: 
dir "dentalai-deployment.tar.gz" | findstr "dentalai-deployment.tar.gz"
echo.
echo Contents:
echo   - Frontend build files (vite-js/dist)
echo   - Backend build files (minimal-api-dev-v6/.next)
echo   - AI server files (unified_ai_api_server.py + dependencies)
echo   - Configuration files (env, docker, nginx)
echo.
echo ============================================
echo Transfer to server:
echo ============================================
echo.
echo Option 1 - Direct transfer:
echo scp dentalai-deployment.tar.gz %SERVER_USER%@%SERVER_IP%:~/
echo.
echo Option 2 - Using WinSCP:
echo   1. Open WinSCP
echo   2. Connect to %SERVER_IP% as %SERVER_USER%
echo   3. Upload dentalai-deployment.tar.gz
echo.
echo ============================================
echo On server - Extract and deploy:
echo ============================================
echo.
echo 1. Extract: tar -xzf dentalai-deployment.tar.gz
echo 2. Deploy:  cd dentalai && docker-compose up -d
echo.
echo Or use the automated script: ./quick-start.sh
echo.
pause



