# ✅ یکپارچه‌سازی مدل Aariz با Frontend

## 🎉 خلاصه

مدل آموزش داده شده Aariz برای تشخیص لندمارک‌های سفالومتری با موفقیت به صفحه `/dashboard/ai-model-test` اضافه شد.

## 📦 تغییرات انجام شده

### 1️⃣ سرویس Backend (Flask API)

**فایل:** `cephx_service/app_aariz.py`

- سرویس Flask برای مدل Aariz ایجاد شد
- پورت: **5001** (متفاوت از HRNet که روی 5000 است)
- Endpoints:
  - `GET /health` - بررسی وضعیت سرویس
  - `POST /detect` - تشخیص لندمارک‌ها
  - `GET /info` - اطلاعات مدل

### 2️⃣ Frontend Integration

**فایل:** `vite-js/src/pages/dashboard/ai-model-test.jsx`

- مدل Aariz به لیست MODELS اضافه شد
- شناسه مدل: `local/aariz-model`
- منطق فراخوانی API در تابع `handleTest` اضافه شد

## 🚀 نحوه استفاده

### مرحله 1: راه‌اندازی سرویس Backend

**مهم:** باید از Python درون virtual environment استفاده کنید.

**گزینه 1: استفاده از فایل batch (توصیه می‌شود):**

```powershell
cd cephx_service
.\run_aariz_service.bat
```

**گزینه 2: اجرای مستقیم با Python از venv:**

```powershell
cd cephx_service
.\venv\Scripts\python.exe app_aariz.py
```

**گزینه 3: فعال کردن venv و سپس اجرا:**

```powershell
cd cephx_service
.\venv\Scripts\Activate.ps1
python app_aariz.py
```

**⚠️ توجه:** اگر از `python.exe` یا `python` بدون فعال کردن venv استفاده کنید، خطای `ModuleNotFoundError: No module named 'flask'` دریافت خواهید کرد.

سرویس روی `http://localhost:5001` اجرا می‌شود.

### مرحله 2: استفاده از Frontend

1. برو به: `http://localhost:3030/dashboard/ai-model-test`
2. از منوی انتخاب مدل، **"مدل Aariz (Local)"** را انتخاب کن
3. یک تصویر سفالومتری آپلود کن
4. روی دکمه **"شروع تست"** کلیک کن
5. نتایج با 29 لندمارک نمایش داده می‌شود

## 📊 مشخصات مدل

- **معماری:** HRNet (یا مدل آموزش داده شده - قابل تنظیم در `app_aariz.py`)
- **تعداد لندمارک‌ها:** 29
- **اندازه ورودی:** 512x512
- **Checkpoint:** `Aariz/checkpoints/checkpoint_best.pth`
- **Device:** CUDA (با fallback به CPU)

## 🔧 تنظیمات

### تغییر معماری مدل

در فایل `cephx_service/app_aariz.py`:

```python
MODEL_NAME = 'hrnet'  # یا 'resnet', 'unet', 'hourglass'
```

### تغییر پورت

در فایل `cephx_service/app_aariz.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5001)  # پورت را تغییر دهید
```

و در `vite-js/src/pages/dashboard/ai-model-test.jsx`:

```javascript
response = await fetch('http://localhost:5001/detect', {  // همان پورت را استفاده کنید
```

## ✅ تست سرویس

### Health Check

```bash
curl http://localhost:5001/health
```

### دریافت اطلاعات مدل

```bash
curl http://localhost:5001/info
```

### تست تشخیص

```bash
curl -X POST http://localhost:5001/detect \
  -H "Content-Type: application/json" \
  -d "{\"image_base64\": \"...\"}"
```

## 📝 نکات مهم

1. **بررسی Checkpoint:** مطمئن شوید فایل `Aariz/checkpoints/checkpoint_best.pth` وجود دارد
2. **پورت‌های باز:** مطمئن شوید پورت 5001 در دسترس است
3. **Dependencies:** تمام dependencies مورد نیاز در `venv` نصب شده باشند
4. **CUDA:** اگر GPU دارید، CUDA باید نصب باشد

## 🐛 عیب‌یابی

### خطا: "Could not import LandmarkPredictor"

- مطمئن شوید مسیر `Aariz` درست است
- فایل `Aariz/inference.py` وجود دارد

### خطا: "Checkpoint not found"

- بررسی کنید فایل `checkpoint_best.pth` در `Aariz/checkpoints/` موجود است
- یا از `checkpoint_latest.pth` استفاده کنید (تغییر در `app_aariz.py`)

### خطا: "Service not ready"

- لاگ‌های خطا را در کنسول بررسی کنید
- مطمئن شوید مدل به درستی load شده است

## 🎯 لیست لندمارک‌های شناسایی شده

مدل Aariz 29 لندمارک را تشخیص می‌دهد:

```
A, ANS, B, Me, N, Or, Pog, PNS, Pn, R,
S, Ar, Co, Gn, Go, Po, LPM, LIT, LMT, UPM,
UIA, UIT, UMT, LIA, Li, Ls, N`, Pog`, Sn
```

## 📚 فایل‌های مرتبط

- `cephx_service/app_aariz.py` - سرویس Flask
- `cephx_service/run_aariz_service.bat` - اسکریپت اجرا
- `vite-js/src/pages/dashboard/ai-model-test.jsx` - صفحه Frontend
- `Aariz/inference.py` - کلاس LandmarkPredictor
- `Aariz/model.py` - معماری مدل
- `Aariz/checkpoints/checkpoint_best.pth` - وزن‌های مدل

---

**تکمیل شد! 🎉**

مدل Aariz اکنون در صفحه `/dashboard/ai-model-test` قابل استفاده است.

