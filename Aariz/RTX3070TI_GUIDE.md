# 🚀 راهنمای آموزش با RTX 3070 Ti (8GB VRAM)

## 📊 مشخصات RTX 3070 Ti

- **VRAM**: 8GB GDDR6X
- **CUDA Cores**: 6144
- **Memory Bandwidth**: 608 GB/s
- **Power**: ~290W TDP

---

## ⚙️ تنظیمات بهینه برای 8GB VRAM

### ⚠️ مهم: RTX 3070 Ti فقط 8GB VRAM دارد (نه 12GB مثل RTX 3060)

برای 1024x1024 با 8GB VRAM، باید تنظیمات محافظه‌کارانه‌تری استفاده کنیم.

---

## 🎯 تنظیمات پیشنهادی

### Safe (پیشنهادی) - ~6-7GB VRAM:

```bash
--batch_size 1                    # Conservative برای 8GB
--gradient_accumulation_steps 8    # افزایش برای effective batch size بهتر
--num_workers 2                   # کاهش برای صرفه‌جویی در VRAM
--mixed_precision                  # الزامی برای 8GB VRAM
--use_ema                         # برای دقت بهتر
```

**Effective Batch Size:** 8 (1 × 1 GPU × 8 accumulation)  
**VRAM:** ~6-7GB  
**ریسک OOM:** ✅ بسیار کم

### اگر می‌خواهید بیشتر استفاده کنید - ~7-8GB VRAM:

```bash
--batch_size 2                    # ممکن است OOM بگیرد
--gradient_accumulation_steps 4    # کاهش accumulation
--num_workers 2
--mixed_precision                  # الزامی
```

**Effective Batch Size:** 8 (2 × 1 GPU × 4 accumulation)  
**VRAM:** ~7-8GB  
**ریسک OOM:** ⚠️ متوسط

---

## 🚀 نحوه استفاده

### روش 1: استفاده از Script (توصیه می‌شود) ⭐

```cmd
train_1024x1024_rtx3070ti.bat
```

این script به صورت خودکار از تنظیمات بهینه برای RTX 3070 Ti استفاده می‌کند.

### روش 2: دستور مستقیم (Safe)

```cmd
python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 1 --gradient_accumulation_steps 8 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --num_workers 2
```

### روش 3: دستور مستقیم (Aggressive - ممکن است OOM بگیرد)

```cmd
python train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 2 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --num_workers 2
```

---

## 📊 جدول تنظیمات

| Batch Size | Gradient Accum | Effective Batch | VRAM | ریسک OOM | توصیه |
|------------|----------------|-----------------|------|----------|-------|
| 1 | 8 | 8 | ~6-7GB | ✅ Safe | ⭐ پیشنهادی |
| 1 | 10 | 10 | ~6-7GB | ✅ Safe | ✅ خوب |
| 2 | 4 | 8 | ~7-8GB | ⚠️ Medium | ⚠️ احتیاط |
| 2 | 6 | 12 | ~7-8GB | ⚠️ Medium | ⚠️ احتیاط |
| 3 | 4 | 12 | ~8-9GB | ❌ High | ❌ خطرناک |

---

## ⚠️ اگر OOM گرفتید

### گزینه 1: کاهش batch_size
```bash
--batch_size 1  # اگر batch_size=2 بود
```

### گزینه 2: افزایش gradient_accumulation_steps
```bash
--gradient_accumulation_steps 10  # افزایش از 8 به 10
```

### گزینه 3: کاهش num_workers
```bash
--num_workers 1  # کاهش از 2 به 1
```

### گزینه 4: کاهش image_size (موقت برای تست)
```bash
--image_size 512 512  # برای تست
```

---

## 💡 نکات مهم

1. **Mixed Precision الزامی است**: برای 8GB VRAM، `--mixed_precision` باید فعال باشد
2. **با batch_size=1 شروع کنید**: این safe است و OOM نمی‌گیرد
3. **Gradient Accumulation را افزایش دهید**: برای effective batch size بهتر
4. **num_workers را کم نگه دارید**: برای صرفه‌جویی در VRAM

---

## 🎯 توصیه نهایی

برای RTX 3070 Ti (8GB VRAM):
- ✅ از `batch_size=1` + `gradient_accumulation_steps=8` استفاده کنید
- ✅ `num_workers=2` تنظیم کنید
- ✅ `mixed_precision` را فعال کنید (الزامی)
- ✅ از script `train_1024x1024_rtx3070ti.bat` استفاده کنید

**این تنظیمات:**
- ✅ Effective batch size = 8
- ✅ بدون ریسک OOM
- ✅ استفاده از ~6-7GB VRAM (safe margin)
- ✅ Performance خوب با mixed precision

---

## 📈 مقایسه با RTX 3060 (12GB)

| GPU | VRAM | Batch Size | Gradient Accum | Effective Batch | VRAM Usage |
|-----|------|------------|----------------|-----------------|------------|
| RTX 3060 | 12GB | 2 | 4 | 16 | ~7-8GB |
| RTX 3070 Ti | 8GB | 1 | 8 | 8 | ~6-7GB |

**نتیجه:** RTX 3070 Ti با effective batch size کمتر کار می‌کند، اما با mixed precision performance خوبی دارد.

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ تنظیمات بهینه برای RTX 3070 Ti (8GB VRAM)
















