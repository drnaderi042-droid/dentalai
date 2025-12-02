@echo off
echo ============================================
echo    DentalAI - Quick Ubuntu Deployment
echo ============================================
echo.
echo This script will transfer source code and deploy on Ubuntu server
echo Recommended method: Transfer first, build second
echo.

set /p SERVER_IP="Enter Ubuntu server IP: "
set /p SERVER_USER="Enter SSH username: "
set /p PROJECT_PATH="Project path (default: C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy): "

if "%PROJECT_PATH%"=="" set PROJECT_PATH=C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy

echo.
echo ============================================
echo Transferring source code to server...
echo ============================================

echo Transferring vite-js folder...
scp -r "%PROJECT_PATH%\vite-js" %SERVER_USER%@%SERVER_IP%:/home/%SERVER_USER%/

echo Transferring deployment scripts...
scp "%PROJECT_PATH%\server-setup.sh" %SERVER_USER%@%SERVER_IP%:/home/%SERVER_USER%/

if errorlevel 1 (
    echo.
    echo ❌ Transfer failed!
    echo Make sure:
    echo 1. SSH is enabled on server
    echo 2. You have correct permissions
    echo 3. Server IP and username are correct
    echo.
    pause
    exit /b 1
)

echo.
echo ============================================
echo ✅ Transfer completed successfully!
echo ============================================
echo.
echo Next steps on your Ubuntu server:
echo.
echo 1. Connect: ssh %SERVER_USER%@%SERVER_IP%
echo 2. Make script executable: chmod +x server-setup.sh
echo 3. Run deployment: ./server-setup.sh
echo.
echo Your app will be available at: http://%SERVER_IP%
echo.
echo Alternative manual commands:
echo sudo apt update ^&^& curl -fsSL https://deb.nodesource.com/setup_lts.x ^| sudo -E bash - ^&^& sudo apt-get install -y nodejs
echo cd vite-js ^&^& npm install ^&^& npm run build
echo sudo apt install nginx ^&^& sudo cp -r dist/* /var/www/html/ ^&^& sudo systemctl restart nginx
echo.
pause



