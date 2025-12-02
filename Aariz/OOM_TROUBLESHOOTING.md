# 🔧 راهنمای عیب‌یابی Out of Memory (OOM) برای 1024x1024

## ⚠️ مشکل: CUDA out of memory

اگر با خطای `RuntimeError: CUDA error: out of memory` مواجه شدید، این راهنما را دنبال کنید.

---

## ✅ راه حل‌های سریع

### 1. کاهش Batch Size (پیشنهادی)

```bash
# Batch size را از 2 به 1 کاهش دهید
python3 train_1024x1024.py --batch_size 1 --num_workers 2 ...
```

### 2. کاهش num_workers

```bash
# num_workers را کاهش دهید
python3 train_1024x1024.py --num_workers 2 ...
```

### 3. غیرفعال کردن pin_memory

```bash
# pin_memory را غیرفعال کنید (اگر OOM در pin memory thread بود)
python3 train_1024x1024.py --pin_memory ...
```

### 4. افزایش Gradient Accumulation

```bash
# Batch size را کاهش دهید و gradient accumulation را افزایش دهید
python3 train_1024x1024.py --batch_size 1 --gradient_accumulation_steps 8 ...
```

---

## 📊 تنظیمات پیشنهادی برای 1024x1024

### برای RTX 3060 (12GB VRAM):

| تنظیمات | Batch Size | num_workers | Gradient Accum | Effective Batch |
|---------|------------|-------------|----------------|-----------------|
| **Conservative** | 1 | 2 | 8 | 16 |
| **Balanced** ⭐ | 2 | 4 | 4 | 16 |
| **Aggressive** | 2 | 6 | 4 | 16 |

---

## 🔍 تشخیص مشکل

### مشکل 1: OOM در Pin Memory Thread

```
RuntimeError: Caught RuntimeError in pin memory thread for device 0.
```

**راه حل:**
```bash
# pin_memory را غیرفعال کنید
python3 train_1024x1024.py --pin_memory ...
```

یا در کد:
```python
# در dataset.py، pin_memory=False تنظیم کنید
```

### مشکل 2: OOM در Forward Pass

```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**راه حل:**
- Batch size را کاهش دهید
- Gradient accumulation را افزایش دهید
- Mixed precision را فعال کنید (اگر نیست)

### مشکل 3: OOM در DataLoader

```
RuntimeError: CUDA error: out of memory (در DataLoader)
```

**راه حل:**
- num_workers را کاهش دهید
- prefetch_factor را کاهش دهید
- pin_memory را غیرفعال کنید

---

## 🎯 تنظیمات بهینه برای 1024x1024

### تنظیمات پیشنهادی (پیش‌فرض جدید):

```bash
python3 train_1024x1024.py \
    --batch_size 2 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --mixed_precision \
    --image_size 1024 1024
```

**Effective Batch Size:** 16 (2 × 2 GPUs × 4 accumulation)

---

## 📉 اگر هنوز OOM گرفتید

### گزینه 1: Batch Size = 1

```bash
python3 train_1024x1024.py \
    --batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_workers 2
```

**Effective Batch Size:** 16 (1 × 2 GPUs × 8 accumulation)

### گزینه 2: کاهش Image Size موقت

```bash
# ابتدا با 768x768 آموزش دهید
python3 train_1024x1024.py --image_size 768 768 --batch_size 4

# سپس fine-tune با 1024x1024
python3 train_1024x1024.py --image_size 1024 1024 --batch_size 2 --resume checkpoints_1024x1024/checkpoint_best.pth
```

### گزینه 3: استفاده از Gradient Checkpointing

```python
# در model.py، gradient checkpointing را فعال کنید
# (نیاز به تغییر کد دارد)
```

---

## 🔍 بررسی استفاده از Memory

```bash
# در terminal دیگر
watch -n 1 nvidia-smi
```

**باید ببینید:**
- GPU 0: ~8-10GB VRAM (نه 12GB)
- GPU 1: ~8-10GB VRAM (نه 12GB)
- هر دو GPU: Utilization مشابه

---

## ⚙️ تنظیمات پیشرفته

### 1. Clear CUDA Cache

```python
# در ابتدای training
torch.cuda.empty_cache()
```

### 2. Limit Memory Growth

```python
# در ابتدای training
torch.cuda.set_per_process_memory_fraction(0.9)  # 90% of VRAM
```

### 3. Use CPU Offloading (آخرین راه)

```python
# برای لایه‌های خاص از CPU استفاده کنید
# (کندتر می‌شود اما memory کمتری استفاده می‌کند)
```

---

## 📝 خلاصه دستورات

### تنظیمات Conservative (اگر OOM گرفتید):

```bash
python3 train_1024x1024.py \
    --batch_size 1 \
    --num_workers 2 \
    --gradient_accumulation_steps 8 \
    --mixed_precision \
    --use_ema \
    --multi_gpu
```

### تنظیمات Balanced (پیشنهادی):

```bash
python3 train_1024x1024.py \
    --batch_size 2 \
    --num_workers 4 \
    --gradient_accumulation_steps 4 \
    --mixed_precision \
    --use_ema \
    --multi_gpu
```

---

## ✅ چک‌لیست

قبل از شروع training:

- [ ] Batch size مناسب است (1 یا 2 برای 1024x1024)
- [ ] num_workers کاهش یافته (2-4)
- [ ] Mixed precision فعال است
- [ ] pin_memory غیرفعال است (اگر مشکل داشتید)
- [ ] Gradient accumulation تنظیم شده
- [ ] هر دو GPU در دسترس هستند

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ راه حل‌های OOM اضافه شد

















