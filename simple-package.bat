@echo off
echo ============================================
echo    DentalAI - Simple Deployment Package
echo ============================================

cd /d "%~dp0"

echo Working in: %CD%
echo.

REM Create simple deployment folder
if exist "deploy" rmdir /s /q "deploy"
mkdir "deploy"

echo Copying essential files...

REM Frontend
if exist "vite-js\dist" (
    echo Copying frontend...
    xcopy "vite-js\dist" "deploy\frontend\" /E /I /H /Y >nul
) else (
    echo ERROR: Frontend dist not found!
    goto :error
)

REM Backend
if exist "minimal-api-dev-v6\.next" (
    echo Copying backend...
    xcopy "minimal-api-dev-v6\.next" "deploy\backend\.next\" /E /I /H /Y >nul
    copy "minimal-api-dev-v6\package.json" "deploy\backend\" >nul
    copy "minimal-api-dev-v6\next.config.mjs" "deploy\backend\" >nul
    if exist "minimal-api-dev-v6\prisma" (
        xcopy "minimal-api-dev-v6\prisma" "deploy\backend\prisma\" /E /I /H /Y >nul
    )
) else (
    echo ERROR: Backend .next not found!
    goto :error
)

REM AI Server
if exist "unified_ai_api_server.py" (
    echo Copying AI server...
    copy "unified_ai_api_server.py" "deploy\" >nul
    copy "requirements_unified_api.txt" "deploy\" >nul
) else (
    echo ERROR: AI server not found!
    goto :error
)

REM Config files
echo Copying configs...
if exist "env.example" copy "env.example" "deploy\" >nul
if exist "docker-compose.yml" copy "docker-compose.yml" "deploy\" >nul
if exist "quick-start.sh" copy "quick-start.sh" "deploy\" >nul

echo.
echo ============================================
echo ✅ Simple package created in 'deploy' folder
echo ============================================
echo.
echo Transfer the entire 'deploy' folder to your server
echo.
echo On server:
echo   cd /home/user
echo   scp -r user@windows:deploy .
echo   cd deploy
echo   docker-compose up -d
echo.
pause
goto :end

:error
echo ERROR: Build failed!
pause

:end



