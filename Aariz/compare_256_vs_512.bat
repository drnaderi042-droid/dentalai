@echo off
cd /d "%~dp0"
echo ========================================
echo Compare 256x256 vs 512x512 Models
echo ========================================
echo.
python compare_256_vs_512.py
pause

