@echo off
setlocal enabledelayedexpansion
echo ========================================
echo Test CLdetection2023 Model Accuracy
echo On Aariz Dataset - Common Landmarks
echo ========================================
echo.
echo This script will test the CLdetection2023 model accuracy
echo on common landmarks (15 landmarks) between CLdetection2023 and Aariz.
echo.
echo Prerequisites:
echo   1. CLdetection2023 repository cloned
echo   2. MMPose installed
echo   3. Model file: model_pretrained_on_train_and_val.pth
echo.
pause

echo.
echo Running test script...
echo.

python test_cldetection_accuracy.py --num_images 10

echo.
echo ========================================
echo Test completed!
echo ========================================
echo.
echo If MMPose is not installed, follow the instructions above
echo to set up the environment and run the test.
pause
















