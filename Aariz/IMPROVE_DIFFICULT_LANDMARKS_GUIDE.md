# راهنمای کاهش خطای لندمارک‌های مشکل‌دار

## 📊 لندمارک‌های با بیشترین خطا

بر اساس تحلیل انجام شده، لندمارک‌های زیر بیشترین خطا را دارند:

| رتبه | لندمارک | میانگین خطا (mm) | توضیحات |
|------|---------|------------------|----------|
| 🥇 | **UMT** | 3.805 | Upper Molar Tip (نوک دندان آسیای بزرگ بالا) |
| 🥈 | **UPM** | 3.486 | Upper Premolar (دندان آسیای کوچک بالا) |
| 🥉 | **R** | 3.331 | Ramus point (نقطه شاخه فک) |
| 4 | **Ar** | 2.645 | Articulare (نقطه مفصل فک) |
| 5 | **Go** | 2.618 | Gonion (زاویه فک پایین) |
| 6 | **LMT** | 2.545 | Lower Molar Tip (نوک دندان آسیای بزرگ پایین) |

## 🎯 راهکارهای کاهش خطا (اولویت‌بندی شده)

### 1. ✅ Weighted Loss (سریع‌ترین و مؤثرترین)

**هدف:** دادن وزن بیشتر به لندمارک‌های مشکل‌دار در loss function

**پیاده‌سازی:**
```bash
# اجرای اسکریپت برای اعمال تغییرات
python apply_weighted_loss_to_train2.py
```

**انتظارات:**
- کاهش 30-40% در خطای لندمارک‌های مشکل‌دار
- کاهش 15-20% در MRE کلی

### 2. ✅ افزایش Augmentation برای مناطق مشکل‌دار

**تغییرات در `dataset.py`:**

```python
# در تابع _get_transforms، برای training:
if self.augmentation:
    return A.Compose([
        # Rotation بیشتر برای دندانی‌ها
        A.Rotate(limit=15, p=0.7),  # از 10 به 15 درجه
        
        # Contrast و Brightness بیشتر
        A.RandomBrightnessContrast(
            brightness_limit=0.3,  # از 0.2 به 0.3
            contrast_limit=0.3,
            p=0.7
        ),
        
        # Noise بیشتر
        A.GaussNoise(var_limit=(20, 80), p=0.5),
        
        # Elastic Transform قوی‌تر
        A.ElasticTransform(
            alpha=150,  # از 120 به 150
            sigma=150*0.05,
            p=0.4
        ),
        
        A.Resize(height=height, width=width),
        A.Normalize(mean=0.5, std=0.5),
        ToTensorV2(),
    ])
```

### 3. ✅ Hard Negative Mining

**شناسایی و تمرکز روی تصاویر مشکل‌دار:**

```python
# ایجاد اسکریپت برای شناسایی hard samples
def identify_hard_samples(model, val_loader, threshold_mm=2.5):
    """
    شناسایی تصاویری که لندمارک‌های مشکل‌دار خطای بالایی دارند
    """
    hard_samples = []
    DIFFICULT_LANDMARKS = ['UMT', 'UPM', 'R', 'Ar', 'Go', 'LMT']
    
    # ... کد شناسایی ...
    
    return hard_samples

# سپس در training، این samples را بیشتر تکرار کنید
```

### 4. ✅ Fine-tuning روی Subset مشکل‌دار

```bash
# Fine-tuning فقط روی تصاویر با خطای بالا
python train2.py \
    --resume checkpoints/checkpoint_best_768.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 768 768 \
    --batch_size 4 \
    --lr 1e-6 \
    --epochs 20 \
    --loss adaptive_wing \
    --mixed_precision \
    --hard_samples_only
```

### 5. ✅ Multi-Scale Training

Training در چند resolution مختلف:

```python
# در training loop
scales = [512, 768, 1024]
scale = random.choice(scales)
image_resized = resize_image(image, scale)
```

### 6. ✅ افزایش Resolution

Training در resolution بالاتر (1024x1024) برای دقت بیشتر:

```bash
python train_1024x1024.py \
    --resume checkpoints/checkpoint_best_768.pth \
    --dataset_path Aariz \
    --batch_size 2 \
    --lr 1e-6 \
    --epochs 30
```

## 🚀 برنامه اجرایی پیشنهادی

### مرحله 1: اعمال Weighted Loss (1-2 ساعت)
```bash
# 1. اعمال تغییرات
python apply_weighted_loss_to_train2.py

# 2. Fine-tuning با weighted loss
python train2.py \
    --resume checkpoints/checkpoint_best_768.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 768 768 \
    --batch_size 4 \
    --lr 1e-6 \
    --warmup_epochs 2 \
    --epochs 20 \
    --loss adaptive_wing \
    --mixed_precision

# 3. تست نتایج
python test_768_validation_full.py
```

### مرحله 2: بهبود Augmentation (30 دقیقه)
- ویرایش `dataset.py` و افزایش augmentation
- Retrain با augmentation جدید

### مرحله 3: Hard Negative Mining (2-3 ساعت)
- شناسایی hard samples
- Fine-tuning روی آن‌ها

## 📊 انتظارات از بهبود

بعد از اعمال **Weighted Loss + Augmentation**:

| لندمارک | خطای فعلی | خطای انتظاری | بهبود |
|---------|-----------|--------------|-------|
| UMT | 3.805 mm | ~2.3 mm | 40% ↓ |
| UPM | 3.486 mm | ~2.1 mm | 40% ↓ |
| R | 3.331 mm | ~2.2 mm | 34% ↓ |
| Ar | 2.645 mm | ~1.8 mm | 32% ↓ |
| Go | 2.618 mm | ~1.8 mm | 31% ↓ |
| **MRE کلی** | 1.575 mm | **~1.25 mm** | **20% ↓** |
| **SDR @ 2mm** | 76.21% | **~85%** | **+9%** |

## ⚠️ نکات مهم

1. **Backup قبل از تغییرات:** همیشه backup بگیرید
2. **Validation Monitoring:** همیشه validation loss را monitor کنید
3. **Gradual Increase:** به تدریج وزن‌ها را افزایش دهید
4. **A/B Testing:** نتایج را با و بدون تغییرات مقایسه کنید
5. **Overfitting:** مراقب overfitting باشید - اگر validation loss افزایش یافت، وزن‌ها را کاهش دهید

## 🔍 Troubleshooting

### اگر validation loss افزایش یافت:
- وزن‌ها را کاهش دهید (مثلاً از 2.5 به 2.0)
- Learning rate را کاهش دهید
- Epochs را کاهش دهید

### اگر بهبودی مشاهده نشد:
- بررسی کنید که weighted loss درست اعمال شده
- بررسی کنید که augmentation اعمال می‌شود
- تعداد epochs را افزایش دهید

## 📝 خلاصه

**سریع‌ترین راه:** اعمال Weighted Loss (1-2 ساعت کار)
**مؤثرترین راه:** Weighted Loss + Augmentation + Hard Negative Mining
**بهترین نتیجه:** Multi-Scale + High Resolution + Ensemble

شروع کنید با **Weighted Loss** که سریع‌ترین و مؤثرترین است!















