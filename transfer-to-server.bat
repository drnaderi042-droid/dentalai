@echo off
echo ============================================
echo    DentalAI - Transfer to Ubuntu Server
echo ============================================

echo.
echo This script will help you transfer your project to Ubuntu server
echo.

set /p SERVER_IP="Enter your Ubuntu server IP: "
set /p SERVER_USER="Enter SSH username: "
set /p PROJECT_PATH="Enter project path (default: C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy): "

if "%PROJECT_PATH%"=="" set PROJECT_PATH=C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy

echo.
echo ============================================
echo Server Information:
echo IP: %SERVER_IP%
echo User: %SERVER_USER%
echo Project Path: %PROJECT_PATH%
echo ============================================

echo.
set /p CONFIRM="Is this information correct? (y/n): "

if /i not "%CONFIRM%"=="y" (
    echo Transfer cancelled.
    pause
    exit /b 1
)

echo.
echo Building project locally...

cd /d "%PROJECT_PATH%\vite-js"

if errorlevel 1 (
    echo Error: Cannot find vite-js directory in %PROJECT_PATH%
    echo Please check the path and try again.
    pause
    exit /b 1
)

echo Installing dependencies...
call npm install

if errorlevel 1 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo Building project...
call npm run build

if errorlevel 1 (
    echo Error: Build failed
    pause
    exit /b 1
)

echo.
echo Build completed! Checking dist folder...
if not exist "dist" (
    echo Error: dist folder not found after build
    pause
    exit /b 1
)

echo.
echo ============================================
echo Transferring files to server...
echo ============================================

echo Transferring vite-js folder...
scp -r "%PROJECT_PATH%\vite-js" %SERVER_USER%@%SERVER_IP%:/home/%SERVER_USER%/

if errorlevel 1 (
    echo Error: File transfer failed
    echo Make sure:
    echo 1. SSH is enabled on your server
    echo 2. You have correct permissions
    echo 3. The server IP and username are correct
    pause
    exit /b 1
)

echo.
echo ============================================
echo SUCCESS! Files transferred successfully
echo ============================================
echo.
echo Next steps on your Ubuntu server:
echo.
echo 1. Connect to server: ssh %SERVER_USER%@%SERVER_IP%
echo 2. Navigate to project: cd ~/vite-js
echo 3. Run deployment: ./deploy-ubuntu.sh
echo.
echo Or run these commands manually:
echo sudo apt update ^&^& sudo apt install -y nodejs npm
echo npm install ^&^& npm run build
echo sudo apt install nginx
echo sudo cp -r dist/* /var/www/html/
echo sudo systemctl restart nginx
echo.
echo Your app will be available at: http://%SERVER_IP%
echo.
pause



