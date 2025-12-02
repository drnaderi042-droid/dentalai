@echo off
echo ====================================
echo Continue Fine-tuning with Unfreeze
echo ====================================
echo.
echo This will continue training from the best checkpoint
echo and unfreeze the backbone for better results.
echo.
echo Strategy:
echo   - Loads checkpoint_p1_p2_cldetection.pth
echo   - Unfreezes backbone immediately
echo   - Uses lower learning rate (0.0001)
echo   - Trains for additional 50 epochs
echo.
echo Press any key to continue...
pause >nul

python finetune_p1_p2_cldetection.py ^
    --cldetection-model "C:\Users\Salah\Downloads\Compressed\Dentalai\main - Copy\CLdetection2023\model_pretrained_on_train_and_val.pth" ^
    --annotations annotations_p1_p2.json ^
    --image-dir Aariz/train/Cephalograms ^
    --batch-size 4 ^
    --epochs 50 ^
    --lr 0.0001 ^
    --unfreeze-after 0

echo.
echo ====================================
echo Training complete!
echo ====================================
pause >nul


