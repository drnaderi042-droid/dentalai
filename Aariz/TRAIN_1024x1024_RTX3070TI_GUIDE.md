# راهنمای آموزش با سایز 1024x1024 برای RTX 3070 Ti

## 📊 خلاصه

این راهنما نحوه آموزش مدل با دیتاست Aariz در سایز **1024x1024** را برای کارت گرافیک **RTX 3070 Ti (8GB VRAM)** و **48GB RAM** توضیح می‌دهد.

## 🎯 پیش‌نیازها

- ✅ دیتاست Aariz در پوشه `Aariz`
- ✅ کارت گرافیک RTX 3070 Ti (8GB VRAM)
- ✅ RAM: 48GB (برای cache کردن dataset)
- ✅ PyTorch با پشتیبانی CUDA نصب شده باشد

## ⚙️ تنظیمات بهینه برای 1024x1024

### تنظیمات پیشنهادی:

```bash
--image_size 1024 1024
--batch_size 2                    # کاهش یافته برای جلوگیری از OOM
--gradient_accumulation_steps 3   # Effective batch size = 6
--lr 5e-4
--epochs 100
--loss adaptive_wing
--mixed_precision                 # الزامی برای 8GB VRAM
--num_workers 2                   # مشابه 768x768
--use_ram_cache                   # استفاده از RAM برای cache
```

**مشخصات:**
- **Batch Size:** 2
- **Gradient Accumulation:** 3
- **Effective Batch Size:** 6 (مشابه 768x768)
- **VRAM Usage:** ~7-7.5GB (با margin امن)
- **RAM Usage:** ~10-15GB برای cache (از 48GB موجود)
- **ریسک OOM:** ✅ بسیار کم
- **زمان تقریبی:** 15-20 ساعت (از scratch) یا 8-12 ساعت (fine-tuning)

## 💾 استفاده از RAM برای Cache

### چرا از RAM استفاده کنیم؟

1. **سرعت بیشتر**: خواندن از RAM خیلی سریع‌تر از هارد است
2. **کاهش I/O**: CPU کمتر منتظر خواندن داده می‌ماند
3. **GPU Utilization بهتر**: GPU بیشتر درگیر می‌شود

### چگونه کار می‌کند؟

```python
# با --use_ram_cache:
# 1. تمام dataset در RAM load می‌شود (یکبار)
# 2. در epoch‌های بعدی از RAM خوانده می‌شود (خیلی سریع)
# 3. GPU کمتر منتظر داده می‌ماند
```

### مصرف RAM:

- **Train dataset**: ~800 تصویر × ~12MB = ~10GB
- **Val dataset**: ~100 تصویر × ~12MB = ~1.2GB
- **Total**: ~11-15GB از 48GB موجود ✅

## 🚀 نحوه استفاده

### روش 1: استفاده از Script (پیشنهادی) ⭐

```cmd
cd Aariz
train_1024x1024_rtx3070ti.bat
```

### روش 2: دستور مستقیم (Fine-tuning)

```cmd
python train_1024x1024.py ^
    --resume checkpoints/checkpoint_best.pth ^
    --dataset_path Aariz ^
    --model hrnet ^
    --image_size 1024 1024 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 3 ^
    --lr 1e-5 ^
    --warmup_epochs 3 ^
    --epochs 100 ^
    --loss adaptive_wing ^
    --mixed_precision ^
    --num_workers 2 ^
    --use_ram_cache ^
    --save_dir checkpoints_1024x1024 ^
    --log_dir logs_1024x1024
```

### روش 3: دستور مستقیم (از scratch)

```cmd
python train_1024x1024.py ^
    --dataset_path Aariz ^
    --model hrnet ^
    --image_size 1024 1024 ^
    --batch_size 2 ^
    --gradient_accumulation_steps 3 ^
    --lr 5e-4 ^
    --warmup_epochs 5 ^
    --epochs 100 ^
    --loss adaptive_wing ^
    --mixed_precision ^
    --num_workers 2 ^
    --use_ram_cache ^
    --save_dir checkpoints_1024x1024 ^
    --log_dir logs_1024x1024
```

## 📊 جدول تنظیمات

| Batch Size | Grad Accum | Effective BS | VRAM | ریسک OOM | توصیه |
|------------|------------|--------------|------|----------|-------|
| 2 | 3 | 6 | ~7-7.5GB | ✅ Safe | ⭐ پیشنهادی |
| 2 | 2 | 4 | ~6.5-7GB | ✅ Safe | ✅ محافظه‌کارانه |
| 1 | 6 | 6 | ~6-6.5GB | ✅ Safe | ✅ اگر OOM گرفتید |

## 🔄 Gradient Accumulation

### چرا استفاده می‌کنیم؟

- **1024x1024** خیلی بزرگ است → batch_size=2 برای جلوگیری از OOM
- اما batch_size کوچک = ناپایداری آموزش
- **راه‌حل**: Gradient Accumulation

### چگونه کار می‌کند؟

```
Batch 1: forward → backward (gradient ذخیره می‌شود)
Batch 2: forward → backward (gradient اضافه می‌شود)
Batch 3: forward → backward (gradient اضافه می‌شود)
→ optimizer.step() (update با effective batch size = 6)
```

## ⚠️ نکات مهم

1. **اولین بار کند است**: اگر `--use_ram_cache` فعال باشد، اولین بار dataset را در RAM load می‌کند (~5-10 دقیقه)

2. **Epoch‌های بعدی سریع‌تر**: بعد از cache، هر epoch سریع‌تر می‌شود

3. **اگر OOM گرفتید**:
   - `batch_size` را به 1 کاهش دهید
   - `gradient_accumulation_steps` را به 6 افزایش دهید
   - `--use_ram_cache` را غیرفعال کنید

4. **RAM Cache**:
   - فقط اگر RAM کافی دارید (48GB) استفاده کنید
   - اگر RAM محدود است، `--use_ram_cache` را حذف کنید

## 📁 ساختار خروجی

```
checkpoints_1024x1024/
├── checkpoint_best.pth      # بهترین مدل
├── checkpoint_latest.pth     # آخرین checkpoint
└── checkpoint_epoch_*.pth    # Checkpoint های دوره‌ای

logs_1024x1024/
└── events.out.tfevents.*      # TensorBoard logs
```

## 🔍 مانیتورینگ

برای نظارت بر استفاده از منابع:

```bash
# GPU
nvidia-smi -l 1

# RAM (Task Manager > Performance > Memory)
```

## ✅ چک‌لیست

- [ ] `batch_size=2` تنظیم شده
- [ ] `gradient_accumulation_steps=3` تنظیم شده
- [ ] `mixed_precision` فعال است
- [ ] `num_workers=2` تنظیم شده
- [ ] `--use_ram_cache` فعال است (اگر RAM کافی دارید)
- [ ] VRAM Usage زیر 8GB است
- [ ] RAM Usage زیر 48GB است

## 🎯 نتیجه

با این تنظیمات:
- ✅ OOM نمی‌شود
- ✅ از RAM برای سرعت بیشتر استفاده می‌شود
- ✅ Effective batch size = 6 (مشابه 768x768)
- ✅ آموزش پایدار و سریع
















