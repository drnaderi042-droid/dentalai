@echo off
REM ========================================
REM Run TeethDreamer Segmentation
REM ========================================

echo ============================================================
echo     TeethDreamer Segmentation
echo ============================================================
echo.

REM Activate virtual environment
echo [1/3] Activating virtual environment...
call venv_teethdreamer\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment!
    pause
    exit /b 1
)
echo OK: Environment activated
echo.

REM Navigate to TeethDreamer
echo [2/3] Navigating to TeethDreamer...
cd TeethDreamer
if errorlevel 1 (
    echo ERROR: TeethDreamer directory not found!
    pause
    exit /b 1
)
echo OK: In TeethDreamer directory
echo.

REM Run segmentation
echo [3/3] Running segmentation...
echo.
echo This will open interactive windows for each image.
echo Click LEFT mouse button to select teeth area.
echo Click RIGHT mouse button to remove unwanted area.
echo.
echo Processing 4 images for upper teeth, then 4 for lower teeth...
echo.

python seg_teeth.py --img ../my_images --seg ../output/segmented --suffix png

if errorlevel 1 (
    echo.
    echo ERROR: Segmentation failed!
    pause
    exit /b 1
) else (
    echo.
    echo ============================================================
    echo Segmentation completed successfully!
    echo ============================================================
    echo.
    echo Output saved to: output\segmented\
    echo.
)

pause













