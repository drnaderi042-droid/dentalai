@echo off
echo ============================================
echo    DentalAI - Create Deployment Package
echo ============================================

echo.
echo This script creates a deployment package from built files
echo.

REM Change to script directory
cd /d "%~dp0"

echo Working directory: %CD%
echo.

REM Create deployment directory in current folder
set DEPLOY_DIR=%CD%\dentalai-deployment

echo ============================================
echo Checking built files...
echo ============================================

REM Check if vite-js dist exists
if not exist "vite-js\dist" (
    echo [ERROR] vite-js\dist not found!
    echo Please build vite-js first: cd vite-js ^& npm run build
    goto :error
)

REM Check if minimal-api-dev-v6 .next exists
if not exist "minimal-api-dev-v6\.next" (
    echo [ERROR] minimal-api-dev-v6\.next not found!
    echo Please build minimal-api-dev-v6 first: cd minimal-api-dev-v6 ^& npm run build
    goto :error
)

REM Check if AI server exists
if not exist "unified_ai_api_server.py" (
    echo [ERROR] unified_ai_api_server.py not found!
    goto :error
)

echo [OK] All required files found!
echo.

echo ============================================
echo Creating deployment structure...
echo ============================================

REM Remove old deployment directory if exists
if exist "%DEPLOY_DIR%" (
    echo Removing old deployment directory...
    rmdir /s /q "%DEPLOY_DIR%"
)

REM Create new directory structure
echo Creating directory structure...
mkdir "%DEPLOY_DIR%" 2>nul
mkdir "%DEPLOY_DIR%\frontend" 2>nul
mkdir "%DEPLOY_DIR%\backend" 2>nul
mkdir "%DEPLOY_DIR%\ai-server" 2>nul
mkdir "%DEPLOY_DIR%\config" 2>nul

if errorlevel 1 (
    echo [ERROR] Failed to create directories
    goto :error
)

echo [OK] Directory structure created
echo.

echo ============================================
echo Copying files...
echo ============================================

echo Copying frontend files...
xcopy "vite-js\dist" "%DEPLOY_DIR%\frontend\" /E /I /H /Y >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy frontend files
    goto :error
)
echo [OK] Frontend files copied
echo.

echo Copying backend files...
xcopy "minimal-api-dev-v6\.next" "%DEPLOY_DIR%\backend\.next\" /E /I /H /Y >nul
if errorlevel 1 (
    echo [ERROR] Failed to copy backend .next files
    goto :error
)

copy "minimal-api-dev-v6\package.json" "%DEPLOY_DIR%\backend\" >nul
copy "minimal-api-dev-v6\next.config.mjs" "%DEPLOY_DIR%\backend\" >nul
if exist "minimal-api-dev-v6\prisma" (
    xcopy "minimal-api-dev-v6\prisma" "%DEPLOY_DIR%\backend\prisma\" /E /I /H /Y >nul
)
echo [OK] Backend files copied
echo.

echo Copying AI server files...
copy "unified_ai_api_server.py" "%DEPLOY_DIR%\ai-server\" >nul
copy "requirements_unified_api.txt" "%DEPLOY_DIR%\ai-server\" >nul

if exist "cephx_service" (
    xcopy "cephx_service" "%DEPLOY_DIR%\ai-server\cephx_service\" /E /I /H /Y >nul
)
if exist "facial-landmark-detection" (
    xcopy "facial-landmark-detection" "%DEPLOY_DIR%\ai-server\facial-landmark-detection\" /E /I /H /Y >nul
)
if exist "CLdetection2023" (
    xcopy "CLdetection2023" "%DEPLOY_DIR%\ai-server\CLdetection2023\" /E /I /H /Y >nul
)
echo [OK] AI server files copied
echo.

echo Copying configuration files...
if exist "env.example" copy "env.example" "%DEPLOY_DIR%\config\" >nul
if exist "docker-compose.yml" copy "docker-compose.yml" "%DEPLOY_DIR%\config\" >nul
if exist "nginx" xcopy "nginx" "%DEPLOY_DIR%\config\nginx\" /E /I /H /Y >nul
if exist "quick-start.sh" copy "quick-start.sh" "%DEPLOY_DIR%\" >nul
echo [OK] Configuration files copied
echo.

echo ============================================
echo Creating archive...
echo ============================================

REM Create tar.gz archive
echo Creating dentalai-deployment.tar.gz...
if exist "dentalai-deployment.tar.gz" del "dentalai-deployment.tar.gz"

REM Use tar if available (Linux/WSL), otherwise use PowerShell compression
tar --version >nul 2>&1
if %errorlevel% equ 0 (
    tar -czf "dentalai-deployment.tar.gz" -C "%DEPLOY_DIR%" .
) else (
    echo Tar not available, using PowerShell compression...
    powershell "Compress-Archive -Path '%DEPLOY_DIR%\*' -DestinationPath 'dentalai-deployment.tar.gz' -Force"
)

if exist "dentalai-deployment.tar.gz" (
    echo [OK] Archive created successfully
) else (
    echo [ERROR] Failed to create archive
    goto :error
)

echo.
echo ============================================
echo ✅ DEPLOYMENT PACKAGE CREATED SUCCESSFULLY!
echo ============================================
echo.
echo Package location: %CD%\dentalai-deployment.tar.gz
echo.
for %%A in ("dentalai-deployment.tar.gz") do echo Package size: %%~zA bytes
echo.
echo ============================================
echo 📦 Package Contents:
echo ============================================
echo.
echo 📁 dentalai-deployment/
echo ├── 📁 frontend/          (vite-js build files)
echo ├── 📁 backend/           (Next.js API build files)
echo ├── 📁 ai-server/         (Python AI server files)
echo └── 📁 config/            (Docker configs, nginx, env)
echo.
echo ============================================
echo 🚀 Transfer to Ubuntu Server:
echo ============================================
echo.
echo Method 1 - SCP:
echo scp dentalai-deployment.tar.gz root@195.206.234.48:~/
echo.
echo Method 2 - WinSCP:
echo   1. Open WinSCP
echo   2. Connect to: root@195.206.234.48
echo   3. Upload: dentalai-deployment.tar.gz
echo.
echo ============================================
echo 🖥️  Deploy on Ubuntu Server:
echo ============================================
echo.
echo # On Ubuntu server:
echo cd ~
echo tar -xzf dentalai-deployment.tar.gz
echo cd dentalai-deployment
echo cp config/env.example .env
echo nano .env  # Edit environment variables
echo docker-compose -f config/docker-compose.yml up -d
echo.
echo # Or use automated script:
echo chmod +x quick-start.sh ^&^& ./quick-start.sh
echo.
echo ============================================
echo 🌐 Access URLs after deployment:
echo ============================================
echo.
echo Frontend:    http://195.206.234.48:3030
echo API:         http://195.206.234.48:7272
echo AI Server:   http://195.206.234.48:5000
echo.
echo Press any key to continue...
pause >nul
goto :end

:error
echo.
echo ❌ DEPLOYMENT FAILED!
echo.
echo Please check the errors above and try again.
echo.
pause
goto :end

:end



