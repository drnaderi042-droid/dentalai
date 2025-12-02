# 🔧 راهنمای رفع مشکل Multi-GPU (استفاده از هر دو GPU)

## ⚠️ مشکل: فقط یک GPU استفاده می‌شود

اگر فقط GPU 0 استفاده می‌شود و GPU 1 خالی است، از این راهنما استفاده کنید.

---

## ✅ راه حل: استفاده از DistributedDataParallel (DDP)

**DDP بهتر از DataParallel است** و مطمئناً از هر دو GPU استفاده می‌کند.

### روش 1: استفاده از torchrun (توصیه می‌شود) ⭐

```bash
# در WSL2
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz

# دادن permission
chmod +x train_1024x1024_ddp.sh

# اجرا
./train_1024x1024_ddp.sh
```

یا دستور مستقیم:

```bash
torchrun --nproc_per_node=2 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --use_ddp \
    --num_workers 4
```

### روش 2: استفاده از DataParallel (اگر DDP کار نکرد)

```bash
python3 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --multi_gpu \
    --num_workers 4
```

---

## 🔍 بررسی استفاده از GPU

در terminal دیگر:

```bash
watch -n 1 nvidia-smi
```

**با DDP باید ببینید:**
- GPU 0: ~45-55% utilization, ~6-8GB VRAM
- GPU 1: ~45-55% utilization, ~6-8GB VRAM
- هر دو GPU: Memory و Utilization مشابه

---

## 📊 مقایسه DataParallel vs DDP

| ویژگی | DataParallel | DDP (DistributedDataParallel) |
|-------|--------------|-------------------------------|
| استفاده از هر دو GPU | ⚠️ ممکن است مشکل داشته باشد | ✅ مطمئن |
| Performance | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| پیچیدگی | ساده | متوسط |
| WSL2 | ⚠️ مشکلات دارد | ✅ بهتر کار می‌کند |

---

## 🎯 توصیه

**برای WSL2: از DDP استفاده کنید!**

```bash
./train_1024x1024_ddp.sh
```

این مطمئناً از هر دو GPU استفاده می‌کند.

---

## 🐛 عیب‌یابی

### مشکل 1: torchrun پیدا نمی‌شود

```bash
# نصب PyTorch با CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### مشکل 2: DDP خطا می‌دهد

```bash
# استفاده از DataParallel به جای DDP
python3 train_1024x1024.py --multi_gpu --num_workers 4 ...
```

### مشکل 3: هنوز فقط یک GPU استفاده می‌شود

```bash
# بررسی تعداد GPU
python3 -c "import torch; print('GPUs:', torch.cuda.device_count())"

# باید 2 باشد
```

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ راه حل DDP اضافه شد

















