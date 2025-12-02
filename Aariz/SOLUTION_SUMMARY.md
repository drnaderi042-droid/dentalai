# راه حل مشکل MRE بالا

## خلاصه مشکل

- Loss کاهش می‌یابد اما MRE بهبود نمی‌یابد (~50-120mm)
- مدل heatmap ها را به درستی یاد نمی‌گیرد
- Soft-argmax برای heatmap های flat مشکل دارد

## راه حل‌های اعمال شده

### 1. بهبود `heatmap_to_coordinates` در `utils.py`
- ✅ اضافه کردن threshold برای استفاده از argmax به جای soft-argmax برای heatmap های flat
- ✅ استفاده از temperature scaling برای sharper distribution
- ✅ Fallback به argmax برای early training

### 2. بهبود Weight Initialization در `model.py`
- ✅ Initialize کردن final heatmap layers با weights کوچکتر
- ✅ Bias initialization به سمت predictions پایین‌تر

### 3. تنظیمات بهینه

**برای HRNet (توصیه شده):**
```powershell
python train2.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 16 --epochs 100
```

**اگر VRAM کافی ندارید:**
```powershell
python train2.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 8 --epochs 100
```

## نکات مهم

1. **Image Size**: 256×256 یا بزرگ‌تر استفاده کنید (نه 128×128)
2. **Learning Rate**: 5e-4 برای شروع مناسب است (نه 1e-3)
3. **Batch Size**: 16 یا 8 برای 256×256 مناسب است
4. **صبر کنید**: بعد از 10-20 epoch باید بهبود قابل توجهی ببینید

## انتظارات

- **Epoch 0-5**: MRE ممکن است 50-100mm باشد (طبیعی در warmup)
- **Epoch 5-15**: MRE باید به زیر 30mm برسد
- **Epoch 15-30**: MRE باید به زیر 15mm برسد  
- **Epoch 30+**: MRE باید به زیر 10mm برسد (هدف: زیر 5mm)

## اگر بهبود ندید

1. LR را به 1e-4 کاهش دهید
2. Image size را به 512×512 افزایش دهید (اگر VRAM دارید)
3. Batch size را کاهش دهید
4. بررسی کنید که dataset درست لود می‌شود

