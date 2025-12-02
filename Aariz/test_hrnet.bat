@echo off
echo ========================================
echo HRNet P1/P2 Testing
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Activate conda environment if exists
if exist "%CONDA_PREFIX%" (
    echo Using conda environment: %CONDA_PREFIX%
) else (
    echo No conda environment detected
)

echo.
echo Testing HRNet model...
echo.

python test_p1_p2_hrnet.py

echo.
echo ========================================
echo Testing complete!
echo Check test_results_hrnet/ for visualizations
echo ========================================
pause

