@echo off
setlocal

echo ========================================
echo CPU Optimization for 31-Landmark Model
echo Target: Fast CPU inference for servers
echo ========================================
echo.
echo This script optimizes the trained model for CPU deployment:
echo.
echo   ✓ Dynamic quantization (2-4x speedup)
echo   ✓ ONNX conversion (cross-platform)
echo   ✓ Benchmarking and comparison
echo   ✓ Maintains accuracy while improving speed
echo.

REM Change to script directory
cd /d "%~dp0"

REM Default model path
set "MODEL=models/hrnet_31_heatmap_best.pth"
set "OUTPUT=models/optimized"

REM Check command line arguments
if "%~1" NEQ "" set "MODEL=%~1"
if "%~2" NEQ "" set "OUTPUT=%~2"

REM Check if model exists
if not exist "%MODEL%" (
    echo ERROR: Model not found: %MODEL%
    echo Please ensure the trained model exists.
    echo.
    echo Usage: optimize_31_model_for_cpu.bat [model_path] [output_dir]
    echo Example: optimize_31_model_for_cpu.bat models/hrnet_31_heatmap_best.pth models/cpu_optimized
    pause
    exit /b 1
)

echo Model: %MODEL%
echo Output: %OUTPUT%
echo.

echo Optimization methods:
echo   - Quantization: Reduces model size and improves CPU speed
echo   - ONNX: Cross-platform deployment format
echo.

pause

echo.
echo Starting CPU optimization...
echo This may take several minutes.
echo.

python optimize_31_model_for_cpu.py --input "%MODEL%" --output "%OUTPUT%"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: CPU optimization failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo CPU Optimization Completed!
echo ========================================
echo.
echo Optimized models are available in: %OUTPUT%
echo.
echo For server deployment:
echo 1. Use quantized model for best compatibility
echo 2. Use ONNX model for maximum performance
echo 3. Test inference speed before deployment
echo.
pause
exit /b 0


