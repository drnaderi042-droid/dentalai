@echo off
echo ========================================
echo P1/P2 Calibration Point Annotator
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set default values
set IMAGES_DIR=Aariz/train/Cephalograms
set OUTPUT_FILE=annotations_p1_p2.json
REM Set to 300 to continue from 100 existing annotations (100 + 200 new = 300 total)
set MAX_IMAGES=300

REM Override with command line arguments if provided
if not "%~1"=="" set IMAGES_DIR=%~1
if not "%~2"=="" set OUTPUT_FILE=%~2
if not "%~3"=="" set MAX_IMAGES=%~3

echo Configuration:
echo   Images directory: %IMAGES_DIR%
echo   Output file: %OUTPUT_FILE%
echo   Max images: %MAX_IMAGES%
echo.
echo Starting annotator...
echo.

python p1_p2_annotator.py "%IMAGES_DIR%" -o "%OUTPUT_FILE%" -n %MAX_IMAGES%

echo.
echo ========================================
echo Annotation complete!
echo ========================================
echo.
echo Results saved to: %OUTPUT_FILE%
echo.
pause

