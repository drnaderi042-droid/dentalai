@echo off
echo ====================================
echo Fine-tune CLdetection2023 for P1/P2
echo ====================================
echo.
echo This will fine-tune the CLdetection2023 model to add
echo p1 and p2 calibration landmarks detection.
echo.
echo Training details:
echo   - Uses CLdetection2023 pretrained backbone
echo   - Adds new head for p1/p2 detection
echo   - Uses annotations_p1_p2.json
echo   - Backbone frozen initially (only head trained)
echo   - Image size: 1024x1024 (CLdetection2023 default)
echo.
echo Note: If MMPose is not available, will use ResNet18 as fallback.
echo.
echo Press any key to start training...
pause >nul

python finetune_p1_p2_cldetection.py ^
    --cldetection-model "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth" ^
    --annotations annotations_p1_p2.json ^
    --image-dir Aariz/train/Cephalograms ^
    --batch-size 4 ^
    --epochs 100

echo.
echo ====================================
echo Training complete!
echo ====================================
echo.
echo Model saved as: checkpoint_p1_p2_cldetection.pth
echo.
echo Next steps:
echo   1. Test the model
echo   2. If needed, unfreeze backbone for further fine-tuning
echo.
echo Press any key to exit...
pause >nul

