# خلاصه تغییرات: Weighted Loss برای لندمارک‌های مشکل‌دار

## ✅ تغییرات اعمال شده در train2.py

### 1. اضافه شدن وزن‌های لندمارک‌های مشکل‌دار

```python
DIFFICULT_LANDMARK_WEIGHTS = {
    'UMT': 2.5,   # Upper Molar Tip - بیشترین خطا (3.805 mm)
    'UPM': 2.5,   # Upper Premolar (3.486 mm)
    'R': 2.0,     # Ramus point (3.331 mm)
    'Ar': 1.8,    # Articulare (2.645 mm)
    'Go': 1.8,    # Gonion (2.618 mm)
    'LMT': 1.6,   # Lower Molar Tip (2.545 mm)
    'LPM': 1.4,   # Lower Premolar
    'Or': 1.3,    # Orbitale (2.326 mm)
    'Co': 1.2,    # Condylion (2.200 mm)
    'PNS': 1.2,   # Posterior Nasal Spine (2.155 mm)
}
```

### 2. اضافه شدن تابع calculate_weighted_loss

این تابع وزن بیشتری به لندمارک‌های مشکل‌دار می‌دهد تا مدل بهتر یاد بگیرد.

### 3. جایگزینی loss calculation در train_epoch

به جای:
```python
loss = criterion(outputs, targets)
```

حالا:
```python
loss = calculate_weighted_loss(
    outputs, targets,
    LANDMARK_SYMBOLS,
    criterion,
    device
)
```

## 🚀 نحوه استفاده

### Fine-tuning با Weighted Loss:

```bash
cd Aariz

# Fine-tuning از checkpoint 768x768
python train2.py \
    --resume checkpoint_best_768.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 768 768 \
    --batch_size 4 \
    --lr 1e-6 \
    --warmup_epochs 2 \
    --epochs 20 \
    --loss adaptive_wing \
    --mixed_precision
```

### یا استفاده از batch script:

```bash
.\train_768x768_rtx3070ti.bat
```

## 📊 انتظارات

بعد از fine-tuning با weighted loss:

| لندمارک | خطای فعلی | خطای انتظاری | بهبود |
|---------|-----------|--------------|-------|
| **UMT** | 3.805 mm | **~2.3 mm** | **40% ↓** |
| **UPM** | 3.486 mm | **~2.1 mm** | **40% ↓** |
| **R** | 3.331 mm | **~2.2 mm** | **34% ↓** |
| **Ar** | 2.645 mm | **~1.8 mm** | **32% ↓** |
| **Go** | 2.618 mm | **~1.8 mm** | **31% ↓** |
| **MRE کلی** | 1.575 mm | **~1.25 mm** | **20% ↓** |
| **SDR @ 2mm** | 76.21% | **~85%** | **+9%** |

## ⚠️ نکات مهم

1. **Learning Rate پایین:** از `1e-6` استفاده کنید (نه `1e-5`) برای fine-tuning
2. **Epochs:** 15-20 epoch معمولاً کافی است
3. **Monitoring:** validation loss را monitor کنید
4. **Backup:** فایل `train2.py.backup` ایجاد شده است

## 🔄 بازگشت به نسخه قبلی

اگر می‌خواهید به نسخه بدون weighted loss برگردید:

```bash
copy train2.py.backup train2.py
```

## 📝 راهکارهای اضافی

برای بهبود بیشتر، می‌توانید:

1. **افزایش Augmentation** در `dataset.py`
2. **Hard Negative Mining** - تمرکز روی تصاویر مشکل‌دار
3. **Multi-Scale Training** - training در چند resolution
4. **افزایش Resolution** - training در 1024x1024

برای جزئیات بیشتر، به `IMPROVE_DIFFICULT_LANDMARKS_GUIDE.md` مراجعه کنید.















