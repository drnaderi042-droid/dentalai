# 📊 راهنمای تنظیم Batch Size برای 1024x1024

## ⚠️ مشکل: Out of Memory با batch_size=3

برای 1024x1024 با HRNet و 12GB VRAM، batch_size=3 ممکن است باعث OOM شود.

---

## ✅ تنظیمات پیشنهادی

### Safe (پیشنهادی) - ~7-8GB VRAM:
```bash
--batch_size 2
--num_workers 4
--prefetch_factor 2
--pin_memory False
```

**Effective Batch Size:** 16 (2 × 2 GPUs × 4 accumulation)

### اگر می‌خواهید بیشتر استفاده کنید - ~9-10GB VRAM:

#### گزینه 1: افزایش Gradient Accumulation
```bash
--batch_size 2
--gradient_accumulation_steps 5  # افزایش از 4 به 5
```
**Effective Batch Size:** 20 (2 × 2 GPUs × 5 accumulation)

#### گزینه 2: افزایش تدریجی Batch Size
```bash
# ابتدا با 2.5 امتحان کنید (با gradient accumulation)
--batch_size 2
--gradient_accumulation_steps 5  # Effective = 20

# یا اگر VRAM کافی دارید:
--batch_size 3
--num_workers 4  # کاهش num_workers
--prefetch_factor 2  # کاهش prefetch
```

---

## 🎯 استراتژی برای استفاده از 10GB VRAM

### روش 1: افزایش Gradient Accumulation (بی‌خطر)

```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --batch_size 2 \
    --gradient_accumulation_steps 5 \
    --num_workers 4 \
    ...
```

**مزایا:**
- ✅ Effective batch size = 20 (بهتر از 16)
- ✅ بدون ریسک OOM
- ✅ استفاده بیشتر از VRAM (~8-9GB)

### روش 2: افزایش Batch Size با تنظیمات محافظه‌کارانه

```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --batch_size 3 \
    --num_workers 3 \
    --prefetch_factor 2 \
    ...
```

**نکات:**
- ⚠️ ممکن است OOM بگیرد
- ✅ اگر کار کرد، ~10GB VRAM استفاده می‌شود

---

## 📊 جدول تنظیمات

| Batch Size | num_workers | prefetch | Gradient Accum | Effective Batch | VRAM/GPU | ریسک OOM |
|------------|-------------|----------|----------------|-----------------|----------|----------|
| 2 | 4 | 2 | 4 | 16 | ~7-8GB | ✅ Safe |
| 2 | 4 | 2 | 5 | 20 | ~8-9GB | ✅ Safe |
| 2 | 6 | 3 | 4 | 16 | ~8-9GB | ⚠️ Medium |
| 3 | 4 | 2 | 4 | 24 | ~10-11GB | ⚠️ High |
| 3 | 3 | 2 | 4 | 24 | ~9-10GB | ⚠️ Medium |

---

## 🚀 دستورات پیشنهادی

### Safe (پیشنهادی):
```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --num_workers 4 \
    --mixed_precision \
    --use_ema \
    --use_ddp
```

### برای استفاده بیشتر از VRAM (~9GB):
```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 5 \
    --num_workers 4 \
    --mixed_precision \
    --use_ema \
    --use_ddp
```

### Aggressive (~10GB - ممکن است OOM بگیرد):
```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 3 \
    --gradient_accumulation_steps 4 \
    --num_workers 3 \
    --mixed_precision \
    --use_ema \
    --use_ddp
```

---

## 💡 نکات مهم

1. **با batch_size=2 شروع کنید** - این safe است
2. **اگر می‌خواهید بیشتر استفاده کنید**: gradient_accumulation_steps را افزایش دهید
3. **اگر OOM گرفتید**: batch_size را کاهش دهید یا gradient_accumulation را افزایش دهید

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ تنظیمات Safe برای جلوگیری از OOM

















