@echo off
REM ========================================
REM Extract TeethDreamer Model Files
REM ========================================

echo ============================================================
echo     Extract TeethDreamer Models
echo ============================================================
echo.

REM Check if 3D folder exists
if not exist "3D\" (
    echo ERROR: 3D folder not found!
    pause
    exit /b 1
)

REM Create ckpt directory
if not exist "TeethDreamer\ckpt\" (
    echo Creating ckpt directory...
    mkdir TeethDreamer\ckpt
)

echo Extracting model files...
echo This will take 10-20 minutes (large files)...
echo.

REM Extract TeethDreamer.ckpt
if exist "3D\TeethDreamer.zip" (
    echo [1/4] Extracting TeethDreamer.ckpt...
    powershell -Command "Expand-Archive -Path '3D\TeethDreamer.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract TeethDreamer.zip
    ) else (
        echo ✓ TeethDreamer.ckpt extracted
    )
) else (
    echo ✗ TeethDreamer.zip not found!
)

echo.

REM Extract zero123-xl.ckpt
if exist "3D\zero123-xl.zip" (
    echo [2/4] Extracting zero123-xl.ckpt...
    powershell -Command "Expand-Archive -Path '3D\zero123-xl.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract zero123-xl.zip
    ) else (
        echo ✓ zero123-xl.ckpt extracted
    )
) else (
    echo ✗ zero123-xl.zip not found!
)

echo.

REM Extract ViT-L-14.ckpt
if exist "3D\ViT-L-14.zip" (
    echo [3/4] Extracting ViT-L-14.ckpt...
    powershell -Command "Expand-Archive -Path '3D\ViT-L-14.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract ViT-L-14.zip
    ) else (
        echo ✓ ViT-L-14.ckpt extracted
    )
) else (
    echo ✗ ViT-L-14.zip not found!
)

echo.

REM Extract sam_vit_b_01ec64.pth
if exist "3D\sam_vit_b_01ec64.zip" (
    echo [4/4] Extracting sam_vit_b_01ec64.pth...
    powershell -Command "Expand-Archive -Path '3D\sam_vit_b_01ec64.zip' -DestinationPath 'TeethDreamer\ckpt\' -Force"
    if errorlevel 1 (
        echo ERROR: Failed to extract sam_vit_b_01ec64.zip
    ) else (
        echo ✓ sam_vit_b_01ec64.pth extracted
    )
) else (
    echo ✗ sam_vit_b_01ec64.zip not found!
)

echo.
echo ============================================================
echo Checking extracted files...
echo ============================================================

REM Check for .ckpt files
if exist "TeethDreamer\ckpt\TeethDreamer.ckpt" (
    echo ✓ TeethDreamer.ckpt found
) else (
    echo ✗ TeethDreamer.ckpt NOT found
)

if exist "TeethDreamer\ckpt\zero123-xl.ckpt" (
    echo ✓ zero123-xl.ckpt found
) else (
    echo ✗ zero123-xl.ckpt NOT found
)

if exist "TeethDreamer\ckpt\ViT-L-14.ckpt" (
    echo ✓ ViT-L-14.ckpt found
) else (
    echo ✗ ViT-L-14.ckpt NOT found
)

if exist "TeethDreamer\ckpt\sam_vit_b_01ec64.pth" (
    echo ✓ sam_vit_b_01ec64.pth found
) else (
    echo ✗ sam_vit_b_01ec64.pth NOT found
)

echo.
echo ============================================================
echo Extraction completed!
echo ============================================================
echo.

pause













