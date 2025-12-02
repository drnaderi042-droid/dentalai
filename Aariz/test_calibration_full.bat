@echo off
echo ====================================
echo Full Calibration Detection Test
echo Testing all 18 images...
echo ====================================
echo.

python test_calibration_detection.py

echo.
echo Test complete! Check 'calibration_test_results' folder for visualizations.
echo.
echo Press any key to exit...
pause >nul

