@echo off
echo ========================================
echo HRNet P1/P2 Training
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
echo Starting HRNet training...
echo This may take several hours depending on your GPU.
echo.

python train_p1_p2_hrnet.py

echo.
echo ========================================
echo Training complete!
echo ========================================
pause

