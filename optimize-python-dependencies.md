# راهنمای بهینه‌سازی Dependency های پایتون برای سرور CPU

## 📊 مشکل: حجم زیاد Dependency ها

### Dependency های غیرضروری که حذف می‌شوند:

| Dependency | حجم تقریبی | استفاده | وضعیت |
|------------|-------------|---------|-------|
| **fastapi** | ~50MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **uvicorn** | ~30MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **python-multipart** | ~5MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **scikit-image** | ~100MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **python-dateutil** | ~5MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **dlib** | ~50MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **face-alignment** | ~200MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **retina-face** | ~50MB | ❌ استفاده نمی‌شود | ❌ حذف |
| **torch (full)** | ~2GB | ⚠️ فقط CPU نیاز است | ✅ بهینه‌سازی |

**حجم کل صرفه‌جویی: ~2.5GB**

---

## 🎯 Dependency های ضروری:

### ✅ Core API:
- `flask` - Framework اصلی
- `flask-cors` - CORS support

### ✅ Image Processing:
- `opencv-python` - پردازش تصویر
- `Pillow` - کار با تصاویر
- `numpy` - محاسبات عددی

### ✅ Deep Learning:
- `torch` (CPU-only) - Framework اصلی
- `torchvision` (CPU-only) - Transformations

### ✅ AI Models:
- `ultralytics` - YOLO models
- `mediapipe` - Facial landmark detection
- `mmengine` - برای CLdetection2023
- `mmcv` - برای CLdetection2023
- `openmim` - برای نصب mmcv

### ✅ Optional:
- `scipy` - فقط اگر از LAB model استفاده می‌کنید

---

## 🚀 روش نصب بهینه:

### روش ۱: نصب PyTorch CPU-only (پیشنهادی)

```bash
# نصب PyTorch CPU-only (حجم کمتر)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# سپس بقیه dependency ها
pip install -r requirements_minimal.txt
```

### روش ۲: استفاده از requirements_minimal.txt

```bash
# نصب فقط dependency های ضروری
pip install -r requirements_minimal.txt
```

---

## 📦 مقایسه حجم:

### قبل (requirements_unified_api.txt):
```
torch (full):          ~2.0GB
torchvision (full):   ~500MB
fastapi:              ~50MB
uvicorn:              ~30MB
scikit-image:         ~100MB
سایر:                 ~200MB
─────────────────────────────
مجموع:                ~2.88GB
```

### بعد (requirements_minimal.txt):
```
torch (CPU-only):      ~150MB
torchvision (CPU-only): ~50MB
flask:                 ~5MB
سایر:                  ~100MB
─────────────────────────────
مجموع:                 ~305MB
```

**صرفه‌جویی: ~2.5GB (87% کاهش حجم)**

---

## 🔧 مراحل بهینه‌سازی:

### مرحله ۱: حذف dependency های غیرضروری

```bash
cd /home/salahk

# فعال کردن محیط مجازی
source venv/bin/activate

# حذف dependency های غیرضروری
pip uninstall -y fastapi uvicorn python-multipart scikit-image python-dateutil

# اگر dlib, face-alignment, retina-face نصب شده:
pip uninstall -y dlib face-alignment retina-face
```

### مرحله ۲: نصب PyTorch CPU-only

```bash
# حذف PyTorch کامل
pip uninstall -y torch torchvision

# نصب PyTorch CPU-only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### مرحله ۳: نصب dependency های ضروری

```bash
# نصب از فایل بهینه شده
pip install -r requirements_minimal.txt
```

---

## 📋 فایل requirements_minimal.txt:

```txt
# Core API
flask>=2.0.0,<2.3.0
flask-cors>=4.0.0

# Image Processing
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0

# Deep Learning (CPU-only)
torch>=2.0.0,<2.2.0
torchvision>=0.15.0,<0.17.0

# AI Models
ultralytics>=8.0.0
mediapipe>=0.10.0

# CLdetection2023
openmim
mmengine>=0.6.0,<1.0.0
mmcv>=2.0.0rc4,<=2.1.0

# Optional (فقط اگر LAB استفاده می‌کنید)
scipy>=1.7.0
```

---

## ⚡ اسکریپت خودکار بهینه‌سازی:

```bash
#!/bin/bash
# optimize-dependencies.sh

echo "🔧 Optimizing Python dependencies..."

# فعال کردن venv
source venv/bin/activate

# حذف dependency های غیرضروری
echo "Removing unnecessary dependencies..."
pip uninstall -y fastapi uvicorn python-multipart scikit-image python-dateutil dlib face-alignment retina-face

# حذف PyTorch کامل
echo "Removing full PyTorch..."
pip uninstall -y torch torchvision

# نصب PyTorch CPU-only
echo "Installing PyTorch CPU-only..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# نصب dependency های ضروری
echo "Installing minimal requirements..."
pip install -r requirements_minimal.txt

echo "✅ Optimization completed!"
echo "Disk space saved: ~2.5GB"
```

---

## 🎯 نتیجه:

با استفاده از `requirements_minimal.txt`:
- ✅ حجم نصب: از ~2.88GB به ~305MB (87% کاهش)
- ✅ زمان نصب: سریع‌تر
- ✅ مصرف RAM: کمتر
- ✅ عملکرد: بدون تغییر (فقط CPU استفاده می‌شود)

---

## ⚠️ نکات مهم:

1. **PyTorch CPU-only**: حتماً از نسخه CPU-only استفاده کنید
2. **mmcv**: ممکن است نصب آن زمان‌بر باشد، صبر کنید
3. **مدل‌ها**: مدل‌های AI (Aariz, CLdetection2023) باید در سرور باشند
4. **تست**: بعد از بهینه‌سازی، همه endpoint ها را تست کنید

---

## 🔍 بررسی dependency های نصب شده:

```bash
# لیست dependency های نصب شده
pip list

# بررسی حجم
pip show torch | grep Location
du -sh $(python -c "import torch; print(torch.__file__)")

# تست PyTorch CPU
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# باید CUDA available: False باشد
```

---

## ✅ چک‌لیست بهینه‌سازی:

- [ ] Dependency های غیرضروری حذف شده
- [ ] PyTorch CPU-only نصب شده
- [ ] requirements_minimal.txt استفاده شده
- [ ] حجم نصب کاهش یافته
- [ ] همه endpoint ها تست شده
- [ ] عملکرد بدون تغییر است

**با این بهینه‌سازی، حجم dependency ها 87% کاهش می‌یابد! 🚀**



