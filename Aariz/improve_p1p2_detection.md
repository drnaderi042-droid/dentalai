# راه‌حل‌های بهبود تشخیص P1/P2

## مشکلات فعلی
- در بعضی تصاویر، موقعیت p1/p2 بسیار اشتباه است
- خطای متوسط: 4.16px (خوب است اما می‌تواند بهتر شود)

## راه‌حل‌های پیشنهادی

### 1. ✅ Image Cropping (پیاده‌سازی شده)
**مشکل**: مدل در کل تصویر جستجو می‌کند و ممکن است نقاط مشابه را در جای دیگر پیدا کند.

**راه‌حل**: Crop کردن تصویر به upper right corner قبل از ارسال به مدل:
- **X**: 80% تا 95% از عرض تصویر
- **Y**: 0% تا 20% از ارتفاع تصویر

**مزایا**:
- مدل فقط در ناحیه مورد نظر جستجو می‌کند
- کاهش false positives
- بهبود دقت (چون نویز کمتر است)
- سریع‌تر (تصویر کوچک‌تر)

**پیاده‌سازی**:
- در `/detect-p1p2` و `/detect-p1p2-tta` endpoints
- Coordinates بعد از detection به اندازه اصلی تصویر scale می‌شوند

### 2. ✅ Validation Checks (پیاده‌سازی شده)
- بررسی vertical alignment (dx < 30, 50 < dy < 200)
- بررسی confidence threshold (>= 0.3)
- بررسی موقعیت در upper right corner (75-98% افقی، 0-25% عمودی)

### 3. ✅ Test Time Augmentation (TTA) (پیاده‌سازی شده)
استفاده از TTA برای بهبود دقت:
- Flip horizontal
- Rotate ±5 degrees
- Average نتایج

### 4. Ensemble Methods
استفاده از چند مدل:
- مدل فعلی (heatmap-based)
- مدل قدیمی (coordinate regression)
- Average نتایج

### 5. بهبود Training
- افزایش dataset (از 100 به 200+ تصویر)
- Augmentation بهتر
- Hard negative mining

### 6. Post-processing
- Smoothing با median filter
- Outlier removal
- Temporal consistency (اگر چند تصویر از یک بیمار داریم)

## پیاده‌سازی TTA

می‌توانید یک endpoint جدید `/detect-p1p2-tta` اضافه کنید که:
1. تصویر را با چند augmentation پردازش می‌کند
2. نتایج را average می‌کند
3. دقت بیشتری دارد (اما کندتر است)

## پیاده‌سازی Ensemble

می‌توانید چند مدل را با هم ترکیب کنید:
1. مدل heatmap-based (فعلی)
2. مدل coordinate regression (اگر موجود است)
3. Weighted average نتایج

## بهبود Training

برای بهبود مدل:
1. افزایش dataset به 200+ تصویر
2. Augmentation بهتر (rotation, scale, brightness)
3. Hard negative mining (تمرکز روی تصاویر مشکل‌دار)
4. Transfer learning از مدل‌های pre-trained

