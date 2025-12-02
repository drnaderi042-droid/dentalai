@echo off
echo ====================================
echo Test P1/P2 Model
echo ====================================
echo.
echo This will test the trained model and compare
echo predictions with ground truth annotations.
echo.
echo Press any key to start testing...
pause >nul

python test_p1_p2_cldetection.py ^
    --checkpoint checkpoint_p1_p2_cldetection.pth ^
    --annotations annotations_p1_p2.json ^
    --image-dir Aariz/train/Cephalograms

echo.
echo ====================================
echo Testing complete!
echo ====================================
echo.
echo Results saved to:
echo   - test_p1_p2_results.json
echo   - test_p1_p2_visualization.png
echo.
pause >nul


