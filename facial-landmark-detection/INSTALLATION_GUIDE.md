# راهنمای نصب مدل‌های تشخیص لندمارک صورت

## نصب پایه

```bash
pip install fastapi uvicorn python-multipart opencv-python numpy>=1.24.4 Pillow
```

## نصب مدل‌های مختلف

### 1. MediaPipe (پیش‌فرض - توصیه می‌شود)

```bash
pip install mediapipe
```

**مزایا:**
- ✅ نصب ساده
- ✅ سریع
- ✅ بدون نیاز به فایل مدل اضافی
- ✅ 468 نقطه لندمارک

### 2. dlib (68 points)

```bash
# Windows (با cmake)
pip install cmake
pip install dlib

# یا از pre-built wheel
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl

# Linux/Mac
pip install dlib
```

**دانلود فایل مدل:**
```bash
# دانلود فایل shape predictor
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# یا در Windows:
# 1. دانلود از: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 2. با WinRAR یا 7-Zip extract کنید
# 3. فایل را در پوشه facial-landmark-detection قرار دهید
```

### 3. face-alignment (68 points - دقیق)

```bash
# ابتدا PyTorch را نصب کنید
pip install torch torchvision

# سپس face-alignment
pip install face-alignment
```

**نکته:** برای استفاده از GPU:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. RetinaFace (5 key points)

```bash
pip install retina-face
```

**نیاز به dependencies اضافی:**
```bash
pip install tensorflow  # یا tensorflow-gpu
```

## نصب همه مدل‌ها (یکجا)

```bash
pip install -r requirements.txt

# سپس دانلود فایل dlib
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

## بررسی نصب

بعد از نصب، API server را اجرا کنید:

```bash
python api_server.py
```

سپس در مرورگر:
- `http://localhost:8000/models` - لیست مدل‌های نصب شده
- `http://localhost:8000/docs` - مستندات API

## Troubleshooting

### dlib نصب نمی‌شود در Windows

```bash
# روش 1: استفاده از Visual Studio Build Tools
# دانلود از: https://visualstudio.microsoft.com/downloads/
# نصب "C++ build tools"

# روش 2: استفاده از pre-built wheel
pip install https://files.pythonhosted.org/packages/.../dlib-19.24.0-cp39-cp39-win_amd64.whl
```

### face-alignment کند است

- از GPU استفاده کنید
- یا از MediaPipe یا dlib استفاده کنید که سریع‌تر هستند

### RetinaFace خطا می‌دهد

```bash
# نصب tensorflow
pip install tensorflow

# یا برای GPU
pip install tensorflow-gpu
```

















