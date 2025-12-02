# راهنمای Training با HRNet

## ✅ تغییرات اعمال شده

1. **HRNet اضافه شده**: مدل HRNet که قبلاً در کد بود، حالا قابل استفاده است
2. **Adaptive Heatmap Sigma**: برای image size های مختلف، sigma به صورت خودکار تنظیم می‌شود
   - 128×128: sigma ≈ 0.75
   - 256×256: sigma ≈ 1.5
   - 512×512: sigma ≈ 3.0

## 🚀 استفاده

### Training با HRNet:

```powershell
# با image size 256×256 (توصیه شده)
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 16 --epochs 100

# با image size 128×128 (سریع‌تر)
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 128 128 --batch_size 32 --epochs 100

# با image size 512×512 (بهترین دقت)
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 512 512 --batch_size 8 --epochs 100
```

## 📊 مقایسه مدل‌ها

| مدل | پارامترها | سرعت | دقت (انتظار) |
|-----|-----------|------|--------------|
| ResNet | ~25M | ⭐⭐⭐ | ⭐⭐⭐ |
| HRNet | ~6.35M | ⭐⭐ | ⭐⭐⭐⭐ |
| UNet | ~17M | ⭐⭐⭐ | ⭐⭐ |
| Hourglass | ~10M | ⭐ | ⭐⭐⭐⭐ |

## 💡 مزایای HRNet

1. **حفظ Resolution بالا**: در تمام لایه‌ها resolution بالا حفظ می‌شود
2. **Multi-scale Features**: از چندین resolution استفاده می‌کند
3. **بهتر برای Landmark Detection**: طراحی شده برای pose estimation و landmark detection
4. **Parameters کمتر**: ~6.35M پارامتر (کمتر از ResNet)

## ⚠️ نکات

1. **Image Size**: HRNet با image size بزرگ‌تر بهتر کار می‌کند
2. **Batch Size**: HRNet ممکن است VRAM بیشتری نیاز داشته باشد
3. **Learning Rate**: می‌توانید LR را کمی کمتر کنید (1e-4) اگر unstable بود

## 🎯 توصیه

برای بهترین نتیجه:
- از 256×256 شروع کنید
- Batch size را بر اساس VRAM تنظیم کنید
- اگر مشکل داشتید، LR را به 1e-4 کاهش دهید



