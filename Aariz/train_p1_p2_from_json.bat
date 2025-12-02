@echo off
echo ====================================
echo Train P1/P2 Calibration Model
echo ====================================
echo.
echo This will train a specialized model for detecting
echo p1 and p2 calibration landmarks using annotations_p1_p2.json
echo.
echo Training details:
echo   - Uses annotations_p1_p2.json file
echo   - 2 landmarks (p1, p2)
echo   - Image size: 512x512
echo   - Train/Val split: 80/20
echo   - Estimated time: 10-20 minutes on RTX 3070 Ti
echo.
echo Press any key to start training...
pause >nul

python train_p1_p2_from_json.py --annotations annotations_p1_p2.json --image-dir Aariz/train/Cephalograms

echo.
echo ====================================
echo Training complete!
echo ====================================
echo.
echo Model saved as: checkpoint_p1_p2.pth
echo.
echo Next steps:
echo   1. Test the model: python test_p1_p2_model.py
echo   2. Integrate into frontend
echo.
echo Press any key to exit...
pause >nul


