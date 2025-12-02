# راهنمای نصب و اعمال تغییرات

## 🔧 تغییرات انجام شده

### 1. فشرده‌سازی تصاویر قبل از ارسال به AI
یک utility جدید برای فشرده‌سازی تصاویر ایجاد شد که مشکل حجم بیش از حد را حل می‌کند.

**فایل**: `vite-js/src/utils/image-compression.js`
- ✅ ایجاد شده و آماده است

### 2. اعمال تغییرات در کد

لطفاً تغییرات زیر را **دستی** در فایل `patient-orthodontics-view.jsx` اعمال کنید:

#### تغییر 1: اضافه کردن import
**محل**: اول فایل، بعد از سایر import ها

```javascript
import { compressMultipleImages, getCompressionSettingsForModel } from 'src/utils/image-compression';
```

#### تغییر 2: اصلاح تابع handleRunAICephalometric
**محل**: حدود خط 777

**قبل**:
```javascript
const handleRunAICephalometric = async () => {
```

**بعد**:
```javascript
const handleRunAICephalometric = async (selectedModel = 'cephx-v2') => {
```

#### تغییر 3: افزودن فشرده‌سازی تصاویر
**محل**: داخل تابع handleRunAICephalometric، بعد از ساخت imageUrls

**اضافه کنید**:
```javascript
// Get compression settings for selected model
const compressionSettings = getCompressionSettingsForModel(selectedModel);
console.log('🎯 Compression settings:', compressionSettings);

// Compress images before sending to API
console.log('🔄 Compressing images...');
const compressedImages = await compressMultipleImages(imageUrls, compressionSettings.targetSize);

// Use compressed data URLs
const processedImageUrls = compressedImages.map(img => img.dataUrl);

console.log('✅ Images compressed successfully:');
compressedImages.forEach((img, idx) => {
  console.log(`  Image ${idx + 1}: ${img.width}x${img.height}, ${(img.size / 1024 / 1024).toFixed(2)}MB, Quality: ${img.quality}%`);
});
```

#### تغییر 4: استفاده از تصاویر فشرده در API call
**محل**: داخل axios.post

**قبل**:
```javascript
images: imageUrls,
```

**بعد**:
```javascript
images: processedImageUrls, // Use compressed images
```

**و اضافه کنید**:
```javascript
aiModel: selectedModel, // Pass selected model to backend
```

#### تغییر 5: بهبود error handling
**محل**: بلوک catch در تابع handleRunAICephalometric

**قبل**:
```javascript
} catch (error) {
  console.error('AI Cephalometric error:', error);
  alert(`خطا در تحلیل تصویر cephalometric با AI: ${error.response?.data?.message || error.message}`);
}
```

**بعد**:
```javascript
} catch (error) {
  console.error('[AI Cephalometric] Error:', error);
  
  // More specific error messages
  if (error.message && error.message.includes('exceeds')) {
    alert('خطا: حجم تصویر بیش از حد مجاز است. لطفاً تصویر کوچک‌تری استفاده کنید یا دوباره تلاش کنید.');
  } else if (error.message && error.message.includes('Provider returned error')) {
    alert('خطا در ارتباط با سرویس هوش مصنوعی. لطفاً مدل دیگری را انتخاب کنید یا دوباره تلاش کنید.');
  } else {
    alert(`خطا در تحلیل تصویر cephalometric با AI: ${error.response?.data?.message || error.message}`);
  }
}
```

---

## ✅ فایل‌های کامل شده

- ✅ `vite-js/src/utils/image-compression.js` - آماده
- ✅ `vite-js/src/sections/orthodontics/cephalometric-analysis/ai-model-selector.jsx` - آماده
- ✅ `vite-js/src/sections/orthodontics/cephalometric-analysis/cephalometric-analysis-display.jsx` - آماده
- ✅ `vite-js/src/sections/orthodontics/patient/view/cephalometric-landmark-viewer.jsx` - آماده و بهبود یافته

---

## 🧪 تست کردن

بعد از اعمال تغییرات:

1. Restart کردن سرور Vite:
```bash
Ctrl+C
npm run dev
```

2. رفتن به صفحه بیمار و Cephalometric Analysis

3. کلیک روی دکمه AI و انتخاب مدل

4. مشاهده لاگ‌ها در console:
```
📦 Original images count: 1
🎯 Compression settings: ...
🔄 Compressing images...
✅ Images compressed successfully:
  Image 1: 1024x768, 3.2MB, Quality: 75%
📤 Sending compressed images...
```

---

## 🎯 نتایج مورد انتظار

- ✅ حجم تصاویر به زیر 5MB کاهش می‌یابد
- ✅ API بدون خطا کار می‌کند
- ✅ کیفیت تصویر مناسب حفظ می‌شود
- ✅ سرعت پردازش افزایش می‌یابد

---

## 🐛 عیب‌یابی

### خطا: "image exceeds 5 MB maximum"
- ✅ حل شد با image compression

### خطا: "Cannot find module 'src/utils/image-compression'"
- بررسی کنید import صحیح باشد
- مطمئن شوید فایل در مسیر صحیح است

### تصویر خیلی کوچک می‌شود
- در فایل `image-compression.js`، `MAX_DIMENSION` را از 2048 به 3072 تغییر دهید

---

## 📝 نکات

- فشرده‌سازی فقط برای ارسال به API است
- تصویر اصلی تغییر نمی‌کند
- هر مدل AI تنظیمات فشرده‌سازی خاص خود را دارد
- Claude: 5MB max → فشرده به 4MB
- GPT-4o: 20MB max → فشرده به 10MB

---

✨ با این تغییرات، دیگر مشکل حجم تصویر نخواهید داشت!

