# 🧪 راهنمای سریع تست کالیبراسیون

## 📁 مسیرهای دیتاست

تصاویر و annotation ها در مسیرهای زیر قرار دارند:

```
aariz/
└── Aariz/
    └── train/
        ├── Cephalograms/           # 700 تصویر (18 تا دارای p1/p2)
        │   ├── cks2ip8fq29yq0yufc4scftj8.png
        │   ├── cks2ip8fq29z00yufgnfla2tf.png
        │   └── ...
        └── Annotations/
            └── Cephalometric Landmarks/
                └── Senior Orthodontists/  # Annotation ها
                    ├── cks2ip8fq29yq0yufc4scftj8.json
                    ├── cks2ip8fq29z00yufgnfla2tf.json
                    └── ...
```

## ✅ بررسی ساختار دیتاست (مرحله اول - مهم!)

قبل از اجرای تست‌ها، ابتدا مطمئن شوید که ساختار دیتاست صحیح است:

```bash
cd aariz
python check_dataset_structure.py
```

یا در ویندوز:
```bash
cd aariz
check_dataset.bat
```

این اسکریپت:
- ✅ بررسی می‌کند که تمام پوشه‌های لازم وجود دارند
- ✅ شمارش می‌کند که چند annotation و تصویر p1/p2 موجود است
- ✅ به شما می‌گوید که آیا می‌توانید تست‌ها را اجرا کنید

---

## 🚀 اجرای تست‌ها

### 1. تست سریع (یک تصویر):

```bash
cd aariz
python quick_test_calibration.py
```

یا در ویندوز:
```bash
cd aariz
test_calibration_quick.bat
```

**خروجی:**
- نمایش console با اطلاعات p1/p2 شناسایی شده
- ذخیره تصویر `calibration_detection_result.png`
- نمایش تصویر در پنجره

### 2. تست کامل (18 تصویر):

```bash
cd aariz
python test_calibration_detection.py
```

یا در ویندوز:
```bash
cd aariz
test_calibration_full.bat
```

**خروجی:**
- تست همه 18 تصویر که p1/p2 دارند
- مقایسه با ground truth
- محاسبه accuracy
- ذخیره visualizations در `calibration_test_results/`
- گزارش خطای میانگین

## 📊 خروجی مورد انتظار

### تست سریع:
```
🧪 Testing: Aariz/train/Cephalograms/cks2ip8fq29yq0yufc4scftj8.png

📏 Image size: 1968x2225
🔍 Found 8 bright points

✅ Found calibration pair:
   p2 (upper): (1472, 181)
   p1 (lower): (1470, 274)
   Distance: 93.0 pixels
   Conversion: 0.1075 mm/pixel
   DPI: 236

💾 Saved visualization to: calibration_detection_result.png
```

### تست کامل:
```
📊 Testing 18 images with P1/P2 annotations
✅ cks2ip8fq29yq0yufc4scftj8.png: p1_error=3.2px, p2_error=2.8px - PASS
✅ cks2ip8fq29z00yufgnfla2tf.png: p1_error=5.1px, p2_error=4.2px - PASS
...

============================================================
📊 SUMMARY
============================================================
Total images: 18
Successful detections: 18/18
Correct detections (< 20px error): 16/18
Accuracy: 88.9%

Average error:
  p1: 7.23 pixels
  p2: 6.45 pixels

Average conversion: 0.1082 mm/pixel

💾 Visualizations saved to: calibration_test_results
```

## 🔧 عیب‌یابی

### اگر ارور "Dataset not found" گرفتید:
مطمئن شوید که در پوشه `aariz` هستید:
```bash
cd aariz
```

### اگر تصویر پیدا نشد:
بررسی کنید که پوشه `Aariz/train/Cephalograms/` وجود دارد و تصاویر در آن هستند.

### اگر نقاط کالیبراسیون پیدا نشدند:
پارامترها را در کد تنظیم کنید (راهنمای کامل در `CALIBRATION_DETECTION_GUIDE.md`)

## 📚 مستندات کامل

برای جزئیات بیشتر، الگوریتم، و تنظیمات پارامترها:
```
CALIBRATION_DETECTION_GUIDE.md
```

