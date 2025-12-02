# 🐛 راهنمای عیب‌یابی مشکل تفاوت مختصات Aariz

## مشکل

نتایج از فرانت‌اند با نتایج مستقیم از API متفاوت است:

- **Direct/API Test**: A = (311.34, 1116.85)
- **Frontend**: A = (265.21875, 887.97265625)

همچنین ابعاد تصویر متفاوت است:
- **Direct/API**: 1968 × 2225
- **Frontend metadata**: 1968 × 2207

## تحلیل

این تفاوت نشان می‌دهد که:
1. **تصویر در frontend قبل از ارسال تغییر می‌کند** (resize/crop)
2. یا **مختصات بعد از دریافت scale می‌شوند** (حتی اگر autoScale خاموش باشد)

## راه‌حل‌های ممکن

### راه‌حل 1: بررسی console logs

بعد از اعمال تغییرات، در console مرورگر این لاگ‌ها را بررسی کنید:

```
🖼️ Image Debug Info:
   File name: ...
   File size: ... bytes
   File type: ...
   Frontend detected size: {width: ..., height: ...}
   Base64 length: ...

📊 API Response Debug:
   API image_size: {width: ..., height: ...}
   Frontend imageSize: {width: ..., height: ...}
   Sample landmark (A): {x: ..., y: ...}
```

و در Terminal (backend):
```
🔍 Processing image from API:
   Image size: ... × ... pixels
   Image mode: ...
   Image format: ...
   Result image_size: ...
```

### راه‌حل 2: بررسی تصویر در browser

1. در console مرورگر:
```javascript
// بررسی تصویر بعد از load
const img = new Image();
img.onload = () => {
  console.log('Image dimensions:', img.width, img.height);
  console.log('Natural dimensions:', img.naturalWidth, img.naturalHeight);
};
img.src = URL.createObjectURL(imageFile);
```

### راه‌حل 3: مقایسه دقیق

یک تصویر یکسان را تست کنید:

1. **Direct test**:
```powershell
cd cephx_service
.\venv\Scripts\python.exe test_aariz_simple.py --image "..\Aariz\Aariz\train\Cephalograms\cks2ip8fq29yq0yufc4scftj8.png"
```

2. **Frontend test**: همان تصویر را از frontend آپلود کنید

3. **مقایسه**:
   - آیا ابعاد تصویر یکسان است؟
   - آیا مختصات با یک ratio مشخص متفاوت هستند؟

## احتمالات مشکل

### احتمال 1: تصویر resize می‌شود

اگر در frontend تصویر قبل از ارسال resize می‌شود، باید:
1. بررسی کنید که `convertImageToBase64` تصویر را تغییر نمی‌دهد
2. بررسی کنید که هیچ image processing قبل از base64 انجام نمی‌شود

### احتمال 2: مشکل از AdvancedCephalometricVisualizer

اگر `AdvancedCephalometricVisualizer` مختصات را scale می‌کند:
- بررسی کنید که مختصات مستقیماً از API استفاده می‌شوند
- نه از canvas coordinates

### احتمال 3: Browser image compression

برخی مرورگرها هنگام load تصویر، آن را resize/compress می‌کنند.

**راه‌حل**: از `naturalWidth` و `naturalHeight` استفاده کنید نه از `width` و `height`.

## تغییرات اعمال شده

### 1. Backend (app_aariz.py)
- اضافه شدن logging برای اندازه تصویر دریافتی
- اضافه شدن logging برای نتیجه prediction

### 2. Frontend (ai-model-test.jsx)
- اضافه شدن logging برای اطلاعات تصویر قبل از ارسال
- اضافه شدن logging برای مقایسه ابعاد تصویر
- اضافه شدن `frontend_image_size` به metadata

## تست بعد از اعمال تغییرات

1. **Restart API server**:
```powershell
cd cephx_service
.\venv\Scripts\python.exe app_aariz.py
```

2. **Refresh frontend** (Ctrl+Shift+R برای hard refresh)

3. **آپلود تصویر و کلیک "شروع تست"**

4. **بررسی console logs**:
   - مقایسه `Frontend detected size` با `API image_size`
   - اگر متفاوت بودند، مشکل از frontend است

5. **بررسی Terminal (backend)**:
   - مقایسه `Image size` با اندازه واقعی تصویر
   - اگر متفاوت بودند، مشکل از تبدیل base64 است

## نتیجه‌گیری

اگر بعد از بررسی logs:
- **ابعاد یکسان بودند**: مشکل از scale کردن مختصات است
- **ابعاد متفاوت بودند**: مشکل از resize کردن تصویر است

در هر دو حالت، باید مطمئن شوید که:
1. تصویر بدون تغییر به API ارسال می‌شود
2. مختصات بدون scale اضافی نمایش داده می‌شوند

