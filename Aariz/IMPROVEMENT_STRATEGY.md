# 🎯 راهنمای جامع بهبود نتایج مدل

## 📊 وضعیت فعلی

- **MRE**: 1.99 mm ✅ (خوب - زیر 2mm)
- **SDR @ 2mm**: 65.52% (19/29 لندمارک‌ها)
- **هدف**: MRE ~1.7mm، SDR @ 2mm ~72%
- **فاصله تا هدف**: 6.48% (13 لندمارک‌های بیشتر در محدوده 2mm)

## 🎯 استراتژی پیشنهادی (به ترتیب اولویت)

### 1️⃣ Fine-tuning (پیشنهاد اول) ⭐⭐⭐⭐⭐

**چرا Fine-tuning بهتر است:**
- ✅ **سریع‌تر**: فقط 20-50 epoch نیاز دارد
- ✅ **خطر کمتر**: مدل فعلی را بهبود می‌دهد، نه نابود
- ✅ **کارآمدتر**: از یادگیری قبلی استفاده می‌کند
- ✅ **احتمال موفقیت بالا**: مدل شما قبلاً خوب کار می‌کند

**زمان**: 2-4 ساعت (برای 30-50 epoch)

---

### 2️⃣ آموزش از اول (گزینه دوم) ⭐⭐⭐

**چرا ممکن است لازم باشد:**
- ❌ اگر fine-tuning جواب ندهد
- ❌ اگر می‌خواهید از تنظیمات کاملاً جدید استفاده کنید
- ❌ اگر checkpoint های قبلی مشکل دارند

**زمان**: 8-12 ساعت (برای 100 epoch)

---

### 3️⃣ بهبودهای دیگر ⭐⭐⭐⭐

- Augmentation بهتر
- Loss function بهتر
- Learning rate scheduling
- Post-processing

---

## 📝 1. Fine-tuning (مرحله به مرحله)

### مرحله 1: بررسی بهترین Checkpoint

```bash
cd Aariz
python find_best_checkpoint.py checkpoints
```

این کار به شما نشان می‌دهد که آیا `checkpoint_best.pth` واقعاً بهترین است یا نه.

### مرحله 2: Fine-tuning با Learning Rate پایین

**گزینه A: Fine-tuning ملایم (پیشنهادی)**

```bash
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 1e-5 \
    --warmup_epochs 2 \
    --epochs 30 \
    --loss adaptive_wing \
    --mixed_precision
```

**پارامترهای کلیدی:**
- `--lr 1e-5`: Learning rate خیلی پایین (10x کمتر از اولیه)
- `--epochs 30`: فقط 30 epoch
- `--warmup_epochs 2`: Warmup کوتاه

**گزینه B: Fine-tuning متوسط**

```bash
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 5e-5 \
    --warmup_epochs 3 \
    --epochs 50 \
    --loss adaptive_wing \
    --mixed_precision
```

### مرحله 3: رصد کردن نتایج

در طول آموزش، Tensorboard را اجرا کنید:

```bash
tensorboard --logdir logs
```

**نشانه‌های موفقیت:**
- ✅ Validation MRE کاهش می‌یابد (از 1.99mm به 1.7-1.8mm)
- ✅ Validation loss کاهش می‌یابد
- ✅ SDR افزایش می‌یابد

**نشانه‌های مشکل:**
- ❌ Validation MRE افزایش می‌یابد (overfitting)
- ❌ Loss نوسان دارد یا افزایش می‌یابد

### مرحله 4: توقف در صورت overfitting

اگر validation MRE شروع به افزایش کرد، آموزش را متوقف کنید:
- `Ctrl + C`
- از آخرین checkpoint قبل از افزایش استفاده کنید

---

## 📝 2. آموزش از اول

### اگر Fine-tuning جواب نداد:

**گزینه A: آموزش با تنظیمات بهینه**

```bash
python train2.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --epochs 100 \
    --loss adaptive_wing \
    --mixed_precision \
    --annotation_type "Senior Orthodontists"
```

**گزینه B: آموزش با Augmentation بیشتر**

ابتدا `dataset.py` را بررسی کنید و augmentation ها را تقویت کنید.

---

## 📝 3. بهبودهای تکمیلی

### A. Augmentation بهتر

در `dataset.py`، augmentation ها را تقویت کنید:

```python
# افزایش rotation
rotation_degrees: float = 8.0  # بود: 5.0

# افزایش brightness/contrast variation
brightness: float = 0.15  # بود: 0.1
contrast: float = 0.15    # بود: 0.1
```

### B. Loss Function بهتر

از Adaptive Wing Loss استفاده کنید (در حال حاضر استفاده می‌شود):

```python
--loss adaptive_wing
```

### C. Learning Rate Scheduling

از WarmupCosineScheduler استفاده کنید (در حال حاضر استفاده می‌شود).

### D. Post-processing

یک لایه post-processing اضافه کنید برای بهبود دقت:

```python
# در inference.py
def smooth_landmarks(landmarks, image_size):
    """Smooth landmarks using spatial constraints"""
    # TODO: پیاده‌سازی
    return landmarks
```

---

## 🎯 سناریوهای پیشنهادی

### سناریو 1: سریع (2-3 ساعت) ⭐⭐⭐⭐⭐

```bash
# Fine-tuning ملایم
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 1e-5 \
    --warmup_epochs 2 \
    --epochs 30 \
    --loss adaptive_wing \
    --mixed_precision
```

**انتظار**: SDR @ 2mm به 68-70% برسد

---

### سناریو 2: متعادل (4-6 ساعت) ⭐⭐⭐⭐

```bash
# Fine-tuning متوسط
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 5e-5 \
    --warmup_epochs 3 \
    --epochs 50 \
    --loss adaptive_wing \
    --mixed_precision
```

**انتظار**: SDR @ 2mm به 70-72% برسد

---

### سناریو 3: کامل (8-12 ساعت) ⭐⭐⭐

```bash
# آموزش از اول با تنظیمات بهینه
python train2.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 5e-4 \
    --warmup_epochs 5 \
    --epochs 100 \
    --loss adaptive_wing \
    --mixed_precision
```

**انتظار**: SDR @ 2mm به 72-75% برسد

---

## ⚠️ نکات مهم

### 1. همیشه Backup بگیرید

```bash
# قبل از شروع fine-tuning
cp checkpoints/checkpoint_best.pth checkpoints/checkpoint_best_backup.pth
```

### 2. Monitor کنید

همیشه Tensorboard را باز کنید تا ببینید چه اتفاقی می‌افتد.

### 3. Early Stopping

اگر validation MRE برای 10 epoch متوالی افزایش یافت، متوقف کنید.

### 4. Test کنید

بعد از هر fine-tuning، مدل را روی همان تصویر تست کنید:

```bash
python compare_new_results.py
```

---

## 📊 معیارهای موفقیت

| معیار | فعلی | هدف Fine-tuning | هدف آموزش از اول |
|-------|------|-----------------|-------------------|
| **MRE (mm)** | 1.99 | 1.7-1.8 | 1.5-1.7 |
| **SDR @ 2mm** | 65.52% | 70-72% | 72-75% |
| **لندمارک‌های < 2mm** | 19/29 | 21-22/29 | 22-24/29 |

---

## 🚀 شروع سریع

**پیشنهاد من: با Fine-tuning ملایم شروع کنید:**

```bash
cd Aariz

# 1. Backup
cp checkpoints/checkpoint_best.pth checkpoints/checkpoint_best_backup.pth

# 2. Fine-tuning
python train2.py \
    --resume checkpoints/checkpoint_best.pth \
    --model hrnet \
    --image_size 256 256 \
    --batch_size 16 \
    --lr 1e-5 \
    --warmup_epochs 2 \
    --epochs 30 \
    --loss adaptive_wing \
    --mixed_precision

# 3. Test (در terminal جداگانه)
tensorboard --logdir logs
```

**زمان**: ~2-3 ساعت  
**احتمال موفقیت**: 70-80%

---

## 💡 نکات پیشرفته

### اگر Fine-tuning خیلی آهسته پیش می‌رود:

LR را کمی افزایش دهید:
```bash
--lr 5e-5  # بود: 1e-5
```

### اگر Overfitting می‌شود:

LR را کاهش دهید یا epochs را کم کنید:
```bash
--lr 5e-6 \
--epochs 20
```

### اگر نتایج بهتر نشدند:

به آموزش از اول بروید یا augmentation را تقویت کنید.

---

## 📞 نتیجه‌گیری

**پیشنهاد اول**: Fine-tuning ملایم (30 epoch, LR 1e-5)  
**زمان**: 2-3 ساعت  
**احتمال موفقیت**: 70-80%  
**انتظار**: SDR @ 2mm به 68-70% برسد

**اگر جواب نداد**: Fine-tuning متوسط (50 epoch, LR 5e-5)

**آخرین راه**: آموزش از اول با تنظیمات بهینه

