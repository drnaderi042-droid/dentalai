# Facial Landmark Detection API

API Server برای تشخیص لندمارک‌های صورت با **چندین مدل مختلف**

## مدل‌های پشتیبانی شده

### 1. **MediaPipe Face Mesh** (پیش‌فرض)
- **تعداد لندمارک**: 468 نقطه (20+ نقطه مهم)
- **سرعت**: ⭐⭐⭐⭐⭐ (خیلی سریع)
- **دقت**: ⭐⭐⭐⭐ (خوب)
- **نیاز به GPU**: ❌ (CPU فقط)
- **نصب**: `pip install mediapipe`

### 2. **dlib shape predictor** (68 points)
- **تعداد لندمارک**: 68 نقطه استاندارد
- **سرعت**: ⭐⭐⭐⭐ (سریع)
- **دقت**: ⭐⭐⭐⭐ (خوب)
- **نیاز به GPU**: ❌ (CPU فقط)
- **نصب**: `pip install dlib`
- **فایل مدل**: دانلود از [http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

### 3. **face-alignment** (68 points)
- **تعداد لندمارک**: 68 نقطه استاندارد
- **سرعت**: ⭐⭐⭐ (متوسط)
- **دقت**: ⭐⭐⭐⭐⭐ (عالی)
- **نیاز به GPU**: ⚠️ (اختیاری - بهتر است با GPU)
- **نصب**: `pip install face-alignment`
- **نیاز به PyTorch**: ✅

### 4. **RetinaFace** (5 key points)
- **تعداد لندمارک**: 5 نقطه کلیدی (چشم‌ها، بینی، گوشه‌های دهان)
- **سرعت**: ⭐⭐⭐⭐ (سریع)
- **دقت**: ⭐⭐⭐⭐⭐ (عالی برای تشخیص چهره)
- **نیاز به GPU**: ⚠️ (اختیاری)
- **نصب**: `pip install retina-face`

## نصب و راه‌اندازی

### روش 1: نصب پایه (فقط MediaPipe - توصیه می‌شود)

```bash
pip install -r requirements-basic.txt
```

این روش سریع‌ترین است و MediaPipe را نصب می‌کند که برای اکثر کاربردها کافی است.

### روش 2: نصب همه مدل‌ها

⚠️ **نکته**: نصب همه مدل‌ها نیاز به dependencies اضافی دارد:

#### برای Windows:
1. **CMake** را نصب کنید:
   - دانلود از [cmake.org](https://cmake.org/download/)
   - در حین نصب، گزینه "Add CMake to system PATH" را انتخاب کنید
   - یا از فایل `install_dlib_windows.bat` استفاده کنید

2. **PyTorch** را نصب کنید (برای face-alignment):
   ```bash
   pip install torch torchvision
   ```

3. سپس:
   ```bash
   pip install -r requirements-all.txt
   ```

#### برای Linux/Mac:
```bash
# نصب CMake
sudo apt install cmake  # Ubuntu/Debian
# یا
brew install cmake  # Mac

# نصب همه dependencies
pip install -r requirements-all.txt
```

### 3. نصب dlib (اختیاری)

#### برای Python 3.8 (Windows):
```bash
# روش سریع: استفاده از pre-built wheel
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp38-cp38-win_amd64.whl

# یا
pip install dlib-bin
```

یا فایل `install_dlib_quick.bat` را اجرا کنید.

#### برای Python 3.9+:
ابتدا CMake را نصب کنید (از cmake.org)، سپس:
```bash
pip install cmake
pip install dlib
```

### 4. دانلود فایل dlib shape predictor (برای استفاده از dlib)

```bash
# دانلود فایل
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Extract
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# یا در Windows:
# 1. دانلود از: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 2. با WinRAR یا 7-Zip extract کنید
# 3. فایل shape_predictor_68_face_landmarks.dat را در پوشه facial-landmark-detection قرار دهید
```

### 5. اجرای API Server

```bash
python api_server.py
```

یا:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

## استفاده

### لیست مدل‌های موجود

```bash
GET http://localhost:8000/models
```

### تشخیص لندمارک‌ها (با مدل پیش‌فرض)

```bash
POST http://localhost:8000/facial-landmark
Content-Type: multipart/form-data
file: [image file]
```

### تشخیص لندمارک‌ها (با انتخاب مدل)

```bash
POST http://localhost:8000/facial-landmark?model=face_alignment
Content-Type: multipart/form-data
file: [image file]
```

### مدل‌های قابل استفاده:
- `mediapipe` (پیش‌فرض)
- `dlib`
- `face_alignment`
- `retinaface`

## مثال استفاده در Python

```python
import requests

# با مدل پیش‌فرض (MediaPipe)
response = requests.post(
    'http://localhost:8000/facial-landmark',
    files={'file': open('face.jpg', 'rb')}
)

# با مدل خاص
response = requests.post(
    'http://localhost:8000/facial-landmark?model=face_alignment',
    files={'file': open('face.jpg', 'rb')}
)

result = response.json()
print(f"Found {result['total_landmarks']} landmarks")
print(f"Model used: {result['model']}")
```

## پاسخ API

```json
{
  "success": true,
  "landmarks": [
    {
      "x": 100,
      "y": 150,
      "name": "left_eye_outer",
      "index": 33
    },
    ...
  ],
  "total_landmarks": 68,
  "image_width": 640,
  "image_height": 480,
  "model": "face_alignment"
}
```

## مقایسه مدل‌ها

| مدل | لندمارک | سرعت | دقت | نیاز GPU | پایداری |
|-----|---------|------|-----|----------|---------|
| MediaPipe | 468 (20+) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐⭐ |
| dlib | 68 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ |
| face-alignment | 68 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | ⭐⭐⭐⭐ |
| RetinaFace | 5 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ | ⭐⭐⭐⭐ |

## نکات مهم

1. **MediaPipe**: بهترین انتخاب برای شروع - سریع و بدون نیاز به GPU
2. **dlib**: کلاسیک و پایدار - نیاز به فایل مدل دارد
3. **face-alignment**: بیشترین دقت - نیاز به PyTorch
4. **RetinaFace**: بهترین برای تشخیص چهره + لندمارک‌های کلیدی

## Troubleshooting

### dlib نصب نمی‌شود؟
```bash
# Windows
pip install cmake
pip install dlib

# یا از pre-built wheel استفاده کنید
pip install https://github.com/sachadee/Dlib/releases/download/v19.22/dlib-19.22.99-cp39-cp39-win_amd64.whl
```

### face-alignment نیاز به PyTorch دارد؟
```bash
pip install torch torchvision
pip install face-alignment
```

## API Endpoints

- `GET /` - اطلاعات API
- `GET /models` - لیست مدل‌های موجود
- `GET /health` - بررسی سلامت API
- `POST /facial-landmark` - تشخیص لندمارک‌ها
- `GET /docs` - Swagger UI documentation
