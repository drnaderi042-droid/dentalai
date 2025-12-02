# 🚀 راهنمای سریع WSL2 - Training 1024x1024

## ⚡ شروع سریع (3 مرحله)

### 1. ورود به WSL2

```bash
# در Windows PowerShell یا CMD
wsl
```

### 2. رفتن به دایرکتوری پروژه

```bash
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz
```

### 3. اجرای Training

```bash
# روش 1: استفاده از Shell Script
chmod +x train_1024x1024_wsl.sh
./train_1024x1024_wsl.sh

# روش 2: اجرای مستقیم
python3 train_1024x1024.py --dataset_path Aariz --model hrnet --image_size 1024 1024 --batch_size 4 --gradient_accumulation_steps 4 --epochs 200 --lr 3e-4 --warmup_epochs 10 --mixed_precision --use_ema --multi_gpu
```

---

## ✅ بررسی GPU در WSL2

در terminal دیگر:

```bash
# در WSL2
watch -n 1 nvidia-smi

# یا در Windows
nvidia-smi -l 1
```

**باید ببینید:**
- GPU 0: ~45-55% utilization
- GPU 1: ~45-55% utilization  
- هر دو GPU: Memory usage مشابه

---

## 📝 نکات مهم

1. **Path در WSL2**: فایل‌های Windows در `/mnt/c/...` هستند
2. **Performance**: WSL2 معمولاً 10-15% سریع‌تر است
3. **DataParallel**: در WSL2 بهتر کار می‌کند
4. **num_workers**: می‌توانید بیشتر از Windows استفاده کنید (مثلاً 8)

---

## 🔧 اگر مشکل داشتید

```bash
# بررسی CUDA
nvidia-smi

# بررسی PyTorch
python3 -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

---

**برای راهنمای کامل**: `WSL2_SETUP_GUIDE.md` را ببینید

















