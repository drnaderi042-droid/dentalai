# راهنمای نصب (Installation Guide)

## مشکل رایج: BackendUnavailable Error

اگر هنگام نصب پکیج‌ها خطای `BackendUnavailable` دریافت کردید، این راهنما را دنبال کنید.

## راه حل‌ها

### 1️⃣ آپدیت pip (اولین قدم - اجباری)

```bash
python -m pip install --upgrade pip
```

### 2️⃣ نصب PyTorch به صورت جداگانه

PyTorch معمولاً مشکل‌ساز است. بهتر است آن را جداگانه نصب کنید:

#### برای CPU (اگر GPU ندارید):
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### برای GPU با CUDA:
ابتدا نسخه CUDA خود را بررسی کنید:
```bash
nvcc --version
```

سپس PyTorch مناسب را نصب کنید:
- **CUDA 11.8:**
  ```bash
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
  ```

- **CUDA 12.1:**
  ```bash
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  ```

- **CUDA 12.4:**
  ```bash
  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
  ```

### 3️⃣ نصب سایر پکیج‌ها

بعد از نصب PyTorch، بقیه پکیج‌ها را نصب کنید:

```bash
python -m pip install numpy "numpy<2.0.0"
python -m pip install Pillow
python -m pip install pandas
python -m pip install opencv-python
python -m pip install scikit-image
python -m pip install scipy
python -m pip install tqdm
python -m pip install tensorboard
python -m pip install matplotlib
```

### 4️⃣ استفاده از اسکریپت‌های خودکار

برای سهولت، دو فایل `.bat` ایجاد شده:

#### برای CPU:
```bash
install_requirements.bat
```

#### برای GPU:
```bash
install_requirements_gpu.bat
```

فقط روی آنها دبل‌کلیک کنید یا در PowerShell اجرا کنید.

## مشکلات رایج

### مشکل: "No module named 'torch'"
**راه حل:** PyTorch را دوباره نصب کنید (به روش بالا)

### مشکل: "CUDA out of memory"
**راه حل:** 
- `batch_size` را کاهش دهید (مثلاً از 8 به 4)
- `image_size` را کوچک‌تر کنید (مثلاً 384x384 به جای 512x512)

### مشکل: "pip is too old"
**راه حل:**
```bash
python -m pip install --upgrade pip setuptools wheel
```

### مشکل: "Microsoft Visual C++ 14.0 is required"
**راه حل:** 
1. Visual C++ Build Tools را نصب کنید:
   - دانلود از: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - یا فقط از wheel files استفاده کنید (پکیج‌های pre-built)

2. یا از pre-built wheels استفاده کنید:
```bash
python -m pip install --only-binary :all: torch torchvision
```

## بررسی نصب

برای بررسی اینکه همه چیز درست نصب شده:

```python
import torch
import torchvision
import numpy as np
import cv2
from PIL import Image

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

## نکات مهم

1. **همیشه pip را آپدیت کنید** قبل از نصب پکیج‌های جدید
2. **PyTorch را جداگانه نصب کنید** برای جلوگیری از مشکلات build
3. **نسخه CUDA را بررسی کنید** قبل از نصب PyTorch GPU
4. **از virtual environment استفاده کنید** برای جداسازی پروژه‌ها:

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install --upgrade pip
```

## پشتیبانی

اگر هنوز مشکل دارید:
1. لگ کامل خطا را بررسی کنید
2. نسخه Python خود را بررسی کنید (`python --version`)
3. مطمئن شوید Python 3.8+ دارید

---

**موفق باشید! 🚀**

