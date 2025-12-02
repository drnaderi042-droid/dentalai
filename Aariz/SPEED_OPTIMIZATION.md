# راهنمای بهینه‌سازی سرعت آموزش

## ⚠️ مشکل: زمان هر epoch 10 دقیقه (خیلی کند!)

## ✅ راه‌حل‌های سریع (از موثرترین به کم‌اثرترین)

### 1. کاهش Image Size (بزرگترین تاثیر) ⭐⭐⭐

**کاهش 512×512 به 256×256 = 4x سریع‌تر!**

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --image_size 256 256 --epochs 50
```

**تاثیر:**
- 512×512 → 256×256: **4x سریع‌تر** (از 10 min به ~2.5 min)
- دقت: ممکن است کمی کاهش یابد اما قابل قبول است

### 2. استفاده از Mixed Precision Training (FP16) ⭐⭐⭐

**~1.5-2x سریع‌تر + استفاده کمتر از VRAM!**

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --mixed_precision --epochs 50
```

**تاثیر:** 
- ~30-50% سریع‌تر
- استفاده کمتر از VRAM (می‌توانید batch_size را افزایش دهید)

### 3. افزایش Batch Size (اگر VRAM اجازه بدهد) ⭐⭐

```powershell
# ابتدا تست کنید که batch_size=12 یا 16 کار می‌کند:
python train.py --dataset_path Aariz --model resnet --batch_size 12 --image_size 512 512 --epochs 1 --mixed_precision
```

**اگر کار کرد:**
```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --batch_size 12 --lr 2e-4 --loss adaptive_wing --mixed_precision --epochs 50
```

**تاثیر:** ~20-30% سریع‌تر

### 4. ترکیب همه (بهترین) ⭐⭐⭐

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --batch_size 12 --image_size 256 256 --num_workers 2 --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --mixed_precision --epochs 50
```

**تاثیر:** از 10 min به **~1-1.5 min** در هر epoch!

## 📊 مقایسه زمان

| تنظیمات | زمان هر epoch | بهبود |
|---------|---------------|-------|
| فعلی (512×512, batch=8, FP32) | 10 min | Baseline |
| 256×256, batch=8, FP32 | ~2.5 min | ✅ **4x سریع‌تر** |
| 256×256, batch=12, FP32 | ~2.0 min | ✅ **5x سریع‌تر** |
| 256×256, batch=12, FP16 | ~1.5 min | ✅ **6.7x سریع‌تر** |
| 512×512, batch=8, FP16 | ~6-7 min | ✅ **1.4x سریع‌تر** |
| 384×384, batch=10, FP16 | ~2-3 min | ✅ **3-5x سریع‌تر** |

## 🎯 توصیه سریع

**برای سریع‌ترین نتیجه (ترکیب همه):**

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --batch_size 12 --image_size 256 256 --num_workers 2 --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --mixed_precision --epochs 50
```

این باید زمان را از 10 min به **~1-1.5 min** برساند! 🚀

**اگر فقط می‌خواهید Mixed Precision اضافه کنید (بدون تغییر image size):**

```powershell
python train.py --resume checkpoints/checkpoint_best.pth --dataset_path Aariz --model resnet --lr 2e-4 --warmup_epochs 3 --loss adaptive_wing --mixed_precision --epochs 50
```

این باید زمان را از 10 min به **~6-7 min** برساند.

## ⚠️ نکات مهم

1. **Image Size 256×256:**
   - دقت ممکن است کمی کاهش یابد
   - اما برای landmark detection معمولاً کافی است
   - اگر دقت مهم است، 384×384 را امتحان کنید

2. **Batch Size:**
   - ابتدا تست کنید: `--batch_size 12 --epochs 1`
   - اگر Out of Memory گرفتید، به 10 یا 8 برگردید

3. **Trade-off:**
   - سریعتر = ممکن است دقت کمی کمتر شود
   - اما 256×256 معمولاً برای این task کافی است

---

**پیشنهاد: از 256×256 شروع کنید! ✅**

