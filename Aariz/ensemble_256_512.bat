@echo off
cd /d "%~dp0"
echo ========================================
echo Ensemble 256x256 + 512x512 Models
echo ========================================
echo.
python ensemble_256_512.py
pause

