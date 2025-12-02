# 🚀 بهینه‌سازی استفاده از VRAM برای 1024x1024

## 📊 وضعیت فعلی

- ✅ GPU 0: ~7GB VRAM استفاده می‌شود
- ✅ GPU 1: ~6.5GB VRAM استفاده می‌شود
- 🎯 هدف: استفاده از ~10GB از هر GPU (حداکثر استفاده)

---

## ⚙️ تنظیمات بهینه برای استفاده از 10GB VRAM

### ⚠️ مهم: batch_size=3 ممکن است OOM بگیرد!

### روش 1: افزایش Gradient Accumulation (پیشنهادی - بی‌خطر) ⭐

```bash
--batch_size 2              # Safe batch size
--gradient_accumulation_steps 6  # افزایش از 4 به 6
--num_workers 4
```

**Effective Batch Size:** 24 (2 × 2 GPUs × 6 accumulation)  
**VRAM:** ~9-10GB per GPU  
**ریسک OOM:** ✅ بسیار کم

### روش 2: افزایش Batch Size (ریسک OOM دارد)

```bash
--batch_size 3              # ممکن است OOM بگیرد
--num_workers 4             # کاهش برای صرفه‌جویی در VRAM
--prefetch_factor 2         # کاهش برای صرفه‌جویی در VRAM
```

**Effective Batch Size:** 24 (3 × 2 GPUs × 4 accumulation)  
**VRAM:** ~10-11GB per GPU  
**ریسک OOM:** ⚠️ بالا

---

## 📈 تنظیمات مختلف برای VRAM

### Conservative (فعلی - 7GB):
```bash
--batch_size 2 --num_workers 4 --prefetch_factor 2
```
- VRAM: ~7GB per GPU
- Safe و stable

### Balanced ⭐ (پیشنهادی - 10GB):
```bash
--batch_size 3 --num_workers 6 --pin_memory --prefetch_factor 3
```
- VRAM: ~9-10GB per GPU
- بهترین تعادل بین استفاده و stability

### Aggressive (حداکثر - 11GB):
```bash
--batch_size 4 --num_workers 8 --pin_memory --prefetch_factor 4
```
- VRAM: ~10-11GB per GPU
- ⚠️ ممکن است OOM بگیرد

---

## 🚀 نحوه استفاده

### روش 1: استفاده از Max VRAM Script (توصیه می‌شود برای 10GB) ⭐

```bash
chmod +x train_1024x1024_max_vram.sh
./train_1024x1024_max_vram.sh
```

این script از gradient accumulation بیشتر استفاده می‌کند تا VRAM بیشتری مصرف شود بدون ریسک OOM.

### روش 2: استفاده از DDP Script (Safe - 7-8GB)

```bash
./train_1024x1024_ddp.sh
```

این script از تنظیمات safe استفاده می‌کند (~7-8GB VRAM).

### روش 2: دستور مستقیم (Max VRAM - پیشنهادی)

```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 6 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --use_ddp \
    --num_workers 4
```

**مزایا:**
- ✅ Effective batch size = 24 (همان batch_size=3)
- ✅ بدون ریسک OOM
- ✅ استفاده از ~9-10GB VRAM

---

## 📊 نتایج انتظاری

### با `batch_size=2` + `gradient_accumulation_steps=6`:
- **GPU 0**: ~9-10GB VRAM ✅
- **GPU 1**: ~9-10GB VRAM ✅
- **Effective Batch Size**: 24 (همان batch_size=3)
- **Performance**: مشابه batch_size=3 اما بدون ریسک OOM
- **ریسک OOM**: بسیار کم ✅

### با `batch_size=3`:
- **GPU 0**: ~10-11GB VRAM ⚠️
- **GPU 1**: ~10-11GB VRAM ⚠️
- **Effective Batch Size**: 24
- **ریسک OOM**: بالا ⚠️

---

## ⚠️ اگر OOM گرفتید

### گزینه 1: کاهش batch_size
```bash
--batch_size 2  # برگشت به تنظیمات قبلی
```

### گزینه 2: کاهش num_workers
```bash
--num_workers 4  # کاهش از 6 به 4
```

### گزینه 3: غیرفعال کردن pin_memory
```bash
# --pin_memory را حذف کنید
```

---

## 🎯 توصیه

برای استفاده از ~10GB VRAM **بدون ریسک OOM**:
- ✅ از `batch_size=2` + `gradient_accumulation_steps=6` استفاده کنید
- ✅ `num_workers=4` تنظیم کنید
- ✅ از script `train_1024x1024_max_vram.sh` استفاده کنید

**این روش:**
- ✅ Effective batch size = 24 (همان batch_size=3)
- ✅ بدون ریسک OOM
- ✅ استفاده از ~9-10GB VRAM
- ✅ Performance مشابه batch_size=3

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ تنظیمات بهینه برای 10GB VRAM اضافه شد

