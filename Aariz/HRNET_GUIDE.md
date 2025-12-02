# راهنمای استفاده از HRNet

## ✅ HRNet اضافه شد!

HRNet (High-Resolution Network) بهترین مدل برای landmark detection است و نتایج بهتری نسبت به ResNet/UNet/Hourglass دارد.

## 🚀 استفاده

### شروع آموزش با HRNet:

```powershell
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 128 128 --batch_size 24 --epochs 100
```

### تنظیمات پیشنهادی:

#### برای سرعت و دقت متعادل:
```powershell
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 12 --epochs 100
```

#### برای دقت بالا (پیشنهادی):
```powershell
python train.py --model hrnet --lr 1e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 8 --epochs 150
```

#### برای سرعت بالا:
```powershell
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 128 128 --batch_size 32 --epochs 100
```

## 📊 مقایسه مدل‌ها:

| مدل | پارامترها | دقت | سرعت | توصیه |
|-----|-----------|-----|------|-------|
| **HRNet** | 6.35M | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **بهترین** |
| ResNet | ~25M | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | خوب |
| Hourglass | ~10M | ⭐⭐⭐⭐ | ⭐⭐⭐ | خوب |
| UNet | ~30M | ⭐⭐⭐ | ⭐⭐⭐⭐ | متوسط |

## ⚙️ ویژگی‌های HRNet:

1. **Multi-Resolution Features**: حفظ resolution بالا در تمام مراحل
2. **Feature Fusion**: ترکیب features از resolutions مختلف
3. **بهترین برای Landmark Detection**: طراحی شده برای pose estimation و landmark detection
4. **Memory Efficient**: با 6.35M پارامتر

## 💡 نکات مهم:

1. **Learning Rate**: برای HRNet، LR پایین‌تر بهتر است (`1e-4` تا `5e-4`)
2. **Batch Size**: با 128×128 می‌توانید batch_size=24-32 استفاده کنید
3. **Image Size**: 256×256 یا 128×128 توصیه می‌شود
4. **Epochs**: معمولاً 100-150 epoch نیاز دارد

## 🎯 انتظارات:

با HRNet باید به:
- **MRE < 2.5mm** برسید (خیلی بهتر از ResNet)
- **SDR @ 2mm > 50%** (بهبود قابل توجه)
- **SDR @ 3mm > 70%**

---

**شروع کنید با:**
```powershell
python train.py --model hrnet --lr 5e-4 --mixed_precision --loss adaptive_wing --warmup_epochs 5 --image_size 256 256 --batch_size 12 --epochs 100
```

