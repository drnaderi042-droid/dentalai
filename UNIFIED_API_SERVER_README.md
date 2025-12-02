# Unified AI API Server

سرور API یکپارچه برای همه آنالیزهای هوش مصنوعی

## 📋 ویژگی‌ها

این سرور API یکپارچه، سه نوع آنالیز را پشتیبانی می‌کند:

1. **آنالیز داخل دهان (Intra-Oral Analysis)** - استفاده از YOLOv8
2. **آنالیز صورت (Facial Landmark Detection)** - استفاده از MediaPipe/dlib/face-alignment/RetinaFace
3. **آنالیز لترال سفالومتری (Cephalometric Analysis)** - استفاده از HRNet/Aariz

## 🚀 نصب و راه‌اندازی

### 1. نصب وابستگی‌ها

```bash
# وابستگی‌های اصلی
pip install fastapi uvicorn python-multipart

# برای آنالیز داخل دهان
pip install ultralytics

# برای آنالیز صورت
pip install mediapipe
# یا
pip install dlib
# یا
pip install face-alignment
# یا
pip install retina-face

# برای آنالیز سفالومتری
# HRNet و Aariz نیاز به وابستگی‌های خاص خود دارند
# لطفاً به مستندات مربوطه مراجعه کنید
```

### 2. آماده‌سازی مدل‌ها

#### آنالیز داخل دهان
- مدل YOLO باید در مسیر `LATERAL ORTHO AI.v2i.yolov8/runs/detect/ortho_improved/weights/best.pt` قرار گیرد

#### آنالیز صورت
- برای dlib: فایل `shape_predictor_68_face_landmarks.dat` باید در مسیر `facial-landmark-detection/` قرار گیرد
  - **دانلود خودکار**: اجرای `python download_dlib_model.py` یا `download_dlib_model.bat`
  - **دانلود دستی**: از آدرس `http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2` دانلود کنید و extract کنید

#### آنالیز سفالومتری
- HRNet: مدل باید در مسیر `cephx_service/model/hrnet_cephalometric.pth` قرار گیرد
- Aariz: checkpoint باید در مسیر `Aariz/checkpoints/checkpoint_best.pth` قرار گیرد

### 3. اجرای سرور

```bash
python unified_ai_api_server.py
```

سرور روی پورت `8000` اجرا می‌شود.

## 📡 Endpoint‌ها

### 1. آنالیز داخل دهان

**POST** `/predict`

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "confidence=0.25" \
  -F "iou=0.45"
```

**پارامترها:**
- `file` (required): فایل تصویر
- `confidence` (optional): حداقل confidence (پیش‌فرض: 0.25)
- `iou` (optional): IOU threshold برای NMS (پیش‌فرض: 0.45)

**پاسخ:**
```json
{
  "success": true,
  "detections": [
    {
      "class_id": 0,
      "class_name": "class_name",
      "confidence": 0.95,
      "bbox": {
        "x1": 100,
        "y1": 200,
        "x2": 300,
        "y2": 400
      }
    }
  ],
  "summary": {...},
  "total_detections": 5,
  "image_size": {
    "width": 1920,
    "height": 1080
  }
}
```

### 2. آنالیز صورت

**POST** `/facial-landmark`

```bash
curl -X POST "http://localhost:8000/facial-landmark?model=mediapipe" \
  -F "file=@face.jpg"
```

**پارامترها:**
- `file` (required): فایل تصویر
- `model` (optional): نوع مدل (`mediapipe`, `dlib`, `face_alignment`, `retinaface`) - پیش‌فرض: `mediapipe`

**پاسخ:**
```json
{
  "success": true,
  "landmarks": [
    {
      "x": 100,
      "y": 200,
      "name": "nose_tip",
      "index": 1
    }
  ],
  "total_landmarks": 468,
  "image_width": 1920,
  "image_height": 1080,
  "model": "mediapipe"
}
```

### 3. آنالیز سفالومتری

**POST** `/detect`

```bash
curl -X POST "http://localhost:8000/detect?model=aariz" \
  -F "image_base64=base64_encoded_image"
```

**پارامترها:**
- `image_base64` (required): تصویر به صورت base64
- `model` (optional): نوع مدل (`hrnet` یا `aariz`) - پیش‌فرض: `aariz`
- `preserve_aspect_ratio` (optional): حفظ نسبت تصویر (فقط برای HRNet) - پیش‌فرض: `true`

**پاسخ:**
```json
{
  "success": true,
  "landmarks": {
    "S": {"x": 100, "y": 200, "confidence": 0.9},
    "N": {"x": 150, "y": 180, "confidence": 0.95}
  },
  "metadata": {
    "model": "Aariz Model",
    "num_landmarks": 29,
    "valid_landmarks": 25,
    "processing_time": 0.5,
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

### 4. بررسی سلامت

**GET** `/health`

```bash
curl http://localhost:8000/health
```

**پاسخ:**
```json
{
  "status": "healthy",
  "services": {
    "intra_oral": "ready",
    "facial_landmark": "ready",
    "cephalometric_hrnet": "ready",
    "cephalometric_aariz": "ready"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### 5. لیست مدل‌ها

**GET** `/models`

```bash
curl http://localhost:8000/models
```

**پاسخ:**
```json
{
  "intra_oral": {
    "available": true,
    "status": "ready"
  },
  "facial_landmark": {
    "available_models": ["mediapipe", "dlib"],
    "default": "mediapipe",
    "status": "ready"
  },
  "cephalometric": {
    "hrnet": {
      "available": true,
      "status": "ready"
    },
    "aariz": {
      "available": true,
      "status": "ready"
    }
  }
}
```

## 📚 مستندات API

برای مشاهده مستندات کامل API، به آدرس زیر مراجعه کنید:

```
http://localhost:8000/docs
```

## ⚙️ تنظیمات

می‌توانید پورت سرور را در فایل `unified_ai_api_server.py` تغییر دهید:

```python
API_PORT = 8000  # تغییر پورت
```

## 🔧 عیب‌یابی

### مدل بارگذاری نشده است

اگر مدلی بارگذاری نشده باشد، پیام هشدار در console نمایش داده می‌شود. لطفاً:
1. بررسی کنید که وابستگی‌های مربوطه نصب شده باشند
2. بررسی کنید که فایل‌های مدل در مسیرهای صحیح قرار گرفته باشند
3. لاگ‌های console را بررسی کنید

### dlib shape predictor پیدا نشد

اگر پیام `⚠️ dlib shape predictor not found` را می‌بینید:

**روش 1: دانلود خودکار (توصیه می‌شود)**
```bash
python download_dlib_model.py
```

یا در Windows:
```bash
download_dlib_model.bat
```

**روش 2: دانلود دستی**
1. فایل را از این آدرس دانلود کنید:
   ```
   http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   ```
2. فایل را extract کنید (با WinRAR یا 7-Zip)
3. فایل `shape_predictor_68_face_landmarks.dat` را در پوشه `facial-landmark-detection/` قرار دهید

**نکته**: اگر dlib نصب نشده باشد، می‌توانید از MediaPipe استفاده کنید که به صورت پیش‌فرض فعال است و نیازی به فایل مدل اضافی ندارد.

### خطای CORS

سرور به صورت پیش‌فرض CORS را برای همه origin‌ها فعال کرده است. در production، بهتر است origin‌های مجاز را محدود کنید.

## 📝 یادداشت‌ها

- این سرور جایگزین اسکریپت‌های جداگانه قبلی است:
  - `LATERAL ORTHO AI.v2i.yolov8/api_server.py` (آنالیز داخل دهان)
  - `facial-landmark-detection/api_server.py` (آنالیز صورت)
  - `cephx_service/app_hrnet_real.py` (سفالومتری HRNet)
  - `cephx_service/app_aariz.py` (سفالومتری Aariz)

- همه endpoint‌ها روی یک پورت (8000) در دسترس هستند
- مدل‌ها به صورت lazy loading بارگذاری می‌شوند (فقط در صورت نیاز)

## 🆘 پشتیبانی

در صورت بروز مشکل، لطفاً لاگ‌های console را بررسی کنید و به تیم توسعه اطلاع دهید.

