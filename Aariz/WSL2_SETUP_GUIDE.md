# 🐧 راهنمای استفاده از WSL2 (Ubuntu 22) برای Training 1024x1024

## چرا WSL2؟

- ✅ **DataParallel بهتر کار می‌کند**: در Linux/WSL2، DataParallel بهینه‌تر است
- ✅ **استفاده بهتر از Multi-GPU**: مشکلات Windows حل می‌شود
- ✅ **Performance بهتر**: معمولاً 10-15% سریع‌تر از Windows
- ✅ **کمتر مشکل**: مشکلات multiprocessing و DataParallel کمتر است

---

## 📋 پیش‌نیازها

### 1. نصب WSL2 با Ubuntu 22.04

اگر هنوز نصب نکرده‌اید:

```powershell
# در PowerShell با Administrator
wsl --install -d Ubuntu-22.04
```

یا اگر قبلاً نصب کرده‌اید:

```powershell
wsl --set-default-version 2
wsl --install -d Ubuntu-22.04
```

### 2. نصب CUDA Toolkit برای WSL2

**مهم**: باید CUDA Toolkit برای WSL2 نصب شود (نه Windows)

```bash
# در Ubuntu WSL2
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-12-4-local_12.4.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-12-4-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

### 3. نصب PyTorch با CUDA در WSL2

```bash
# در Ubuntu WSL2
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

یا اگر از requirements.txt استفاده می‌کنید:

```bash
pip3 install -r requirements.txt
```

### 4. بررسی GPU در WSL2

```bash
# بررسی CUDA
nvidia-smi

# بررسی PyTorch
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

---

## 🚀 استفاده از اسکریپت

### روش 1: استفاده از Shell Script (توصیه می‌شود)

```bash
# در Ubuntu WSL2
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz

# دادن permission اجرا
chmod +x train_1024x1024_wsl.sh

# اجرا
./train_1024x1024_wsl.sh
```

### روش 2: اجرای مستقیم Python

```bash
# در Ubuntu WSL2
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz

python3 train_1024x1024.py \
    --dataset_path Aariz \
    --model hrnet \
    --image_size 1024 1024 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --epochs 200 \
    --lr 3e-4 \
    --warmup_epochs 10 \
    --mixed_precision \
    --use_ema \
    --multi_gpu
```

---

## 🔧 تنظیمات بهینه برای WSL2

### 1. افزایش num_workers (در Linux بهتر کار می‌کند)

```bash
--num_workers 8  # یا بیشتر (در Windows مشکل داشت)
```

### 2. استفاده از DistributedDataParallel (اختیاری - برای performance بهتر)

اگر می‌خواهید از DDP استفاده کنید (بهتر از DataParallel):

```bash
# نیاز به تغییر کد دارد - در آینده اضافه می‌شود
```

---

## 📊 بررسی استفاده از GPU

در terminal دیگر (در WSL2 یا Windows):

```bash
# در WSL2
watch -n 1 nvidia-smi

# یا در Windows PowerShell
nvidia-smi -l 1
```

**باید ببینید:**
- GPU 0: Utilization ~45-55%
- GPU 1: Utilization ~45-55%
- هر دو GPU: Memory usage مشابه (~6-8GB هر کدام)

---

## ⚠️ نکات مهم

### 1. Path در WSL2

فایل‌های Windows در `/mnt/c/...` قابل دسترسی هستند:

```bash
# مثال
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz
```

### 2. Performance

- WSL2 معمولاً 10-15% سریع‌تر از Windows است
- DataParallel بهتر کار می‌کند
- کمتر مشکل multiprocessing

### 3. File Permissions

اگر مشکل permission داشتید:

```bash
chmod +x train_1024x1024_wsl.sh
```

---

## 🐛 عیب‌یابی

### مشکل 1: CUDA not found در WSL2

```bash
# بررسی CUDA
nvidia-smi

# اگر کار نکرد، CUDA Toolkit را دوباره نصب کنید
```

### مشکل 2: PyTorch CUDA not available

```bash
# بررسی PyTorch
python3 -c "import torch; print(torch.cuda.is_available())"

# اگر False بود، PyTorch را با CUDA دوباره نصب کنید
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### مشکل 3: فقط یک GPU استفاده می‌شود

```bash
# بررسی تعداد GPU
python3 -c "import torch; print(torch.cuda.device_count())"

# باید 2 باشد
```

---

## 📝 خلاصه دستورات

```bash
# 1. ورود به WSL2
wsl

# 2. رفتن به دایرکتوری پروژه
cd /mnt/c/Users/Salah/Downloads/Compressed/Dentalai/main\ -\ Copy/Aariz

# 3. اجرای اسکریپت
chmod +x train_1024x1024_wsl.sh
./train_1024x1024_wsl.sh

# 4. بررسی GPU (در terminal دیگر)
watch -n 1 nvidia-smi
```

---

## ✅ مزایای WSL2

| ویژگی | Windows | WSL2 |
|-------|---------|------|
| DataParallel | ⚠️ مشکلات دارد | ✅ بهتر کار می‌کند |
| Multi-GPU | ⚠️ ممکن است مشکل داشته باشد | ✅ بهینه |
| Performance | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| Multiprocessing | ⚠️ محدودیت دارد | ✅ بهتر |
| num_workers | محدود (0-2) | می‌تواند بیشتر باشد (8+) |

---

**تاریخ**: 2024-11-01  
**وضعیت**: ✅ آماده برای استفاده در WSL2  
**سیستم عامل**: Ubuntu 22.04 LTS در WSL2

















