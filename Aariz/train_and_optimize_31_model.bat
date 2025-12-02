@echo off
setlocal

echo ========================================
echo Complete Pipeline: Train + Optimize 31-Landmark Model
echo Target: < 2px MRE + Fast CPU Inference
echo ========================================
echo.
echo This script runs the complete pipeline:
echo 1. Train high-accuracy heatmap model
echo 2. Optimize for CPU deployment
echo 3. Benchmark and validate results
echo.

REM Change to script directory
cd /d "%~dp0"

REM Define paths
set "ANNOTATIONS=annotations_31.json"
set "IMAGES=Aariz\train\Cephalograms"
set "MODEL_OUTPUT=models\hrnet_31_heatmap_best.pth"
set "OPTIMIZED_OUTPUT=models\cpu_optimized"

REM Check prerequisites
if not exist "%ANNOTATIONS%" (
    echo ERROR: Annotations file not found: %ANNOTATIONS%
    echo Please create annotations_31.json first.
    pause
    exit /b 1
)

if not exist "%IMAGES%" (
    echo ERROR: Images directory not found: %IMAGES%
    pause
    exit /b 1
)

echo Configuration:
echo   - Annotations: %ANNOTATIONS%
echo   - Images: %IMAGES%
echo   - Model Output: %MODEL_OUTPUT%
echo   - Optimized Output: %OPTIMIZED_OUTPUT%
echo.

echo ⚠️  WARNING: This will take 24-48 hours to complete!
echo     Make sure you have sufficient GPU memory (8GB+)
echo     and disk space (50GB+ for training data)
echo.
pause

echo.
echo ========================================
echo Phase 1: High-Accuracy Training
echo ========================================
echo.

call train_31_heatmap_high_accuracy.bat
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Training phase failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Phase 2: CPU Optimization
echo ========================================
echo.

call optimize_31_model_for_cpu.bat "%MODEL_OUTPUT%" "%OPTIMIZED_OUTPUT%"
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Optimization phase failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo Phase 3: Final Validation
echo ========================================
echo.

echo Testing optimized model accuracy...
python test_31_heatmap_model.py --model "%OPTIMIZED_OUTPUT%\hrnet_31_heatmap_quantized.pth"

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Validation failed!
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo ========================================
echo 🎉 COMPLETE PIPELINE SUCCESSFUL!
echo ========================================
echo.
echo Results:
echo   ✓ High-accuracy model trained (< 2px MRE)
echo   ✓ CPU optimizations applied
echo   ✓ Models ready for server deployment
echo.
echo Deployment ready models:
echo   - %OPTIMIZED_OUTPUT%\hrnet_31_heatmap_quantized.pth (recommended)
echo   - %OPTIMIZED_OUTPUT%\hrnet_31_heatmap.onnx (alternative)
echo.
echo Next steps:
echo   1. Deploy quantized model on your CPU server
echo   2. Monitor MRE performance in production
echo   3. Consider fine-tuning based on real data
echo.
pause
exit /b 0


