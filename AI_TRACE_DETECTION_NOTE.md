# 🤖 یادداشت: AI Detection برای Trace (دندان‌ها)

## 🎯 درخواست کاربر

> "برای بخش trace ، نیاز است تا خود ai بتواند حدود دندان ها را تشخیص بدهد. ضمن اینکه برای اینکار خطوط لزوما صاف نیستند و ممکن است انحنا دار باشند."

---

## 📊 تحلیل نیاز

### چالش اصلی:
مدل فعلی (HRNet) فقط **نقاط** را تشخیص می‌دهد، نه **مرزها/contours**.

```
HRNet Output:
┌─────────────────┐
│   • S           │  ← نقاط منفرد
│     • N         │
│   • A  • B      │
│     • Pog       │
└─────────────────┘

مورد نیاز:
┌─────────────────┐
│   ╭─╮  ╭─╮      │  ← مرزهای دندان‌ها
│   │ │  │ │      │     (انحنادار)
│   ╰─╯  ╰─╯      │
└─────────────────┘
```

---

## 🔍 راه‌حل‌های ممکن

### 1️⃣ استفاده از مدل Segmentation

#### مدل‌های پیشنهادی:

##### A) U-Net (برای Dental Segmentation)
```python
import torch
from segmentation_models_pytorch import Unet

# ساخت مدل
model = Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,  # X-ray grayscale
    classes=32,     # 32 دندان
)

# Training با Dataset دندان‌های برچسب‌گذاری شده
# Output: Mask برای هر دندان
```

**Dataset های موجود:**
- UFBA-UESC Dental Images Deep (Public)
- Tufts Dental Database
- Dental Panoramic X-ray Dataset (Kaggle)

**دقت مورد انتظار:** ~85-90% IoU

##### B) Mask R-CNN
```python
import detectron2
from detectron2.modeling import build_model

# برای تشخیص و Segmentation همزمان
model = build_model(cfg)
# Output: bbox + mask برای هر دندان
```

**مزایا:**
- تشخیص دندان‌های جداگانه
- Mask دقیق برای هر دندان
**معایب:**
- سنگین‌تر از U-Net
- نیاز به GPU قوی‌تر

##### C) DeepLabV3+ (Semantic Segmentation)
```python
import segmentation_models_pytorch as smp

model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    encoder_weights="imagenet",
    classes=3,  # background, upper teeth, lower teeth
)
```

**مزایا:**
- سریع‌تر
- دقت خوب برای مرزها

---

### 2️⃣ پردازش پس از Landmark Detection

#### روش: Active Contours (Snake)
```python
from skimage.segmentation import active_contour

# استفاده از نقاط HRNet به عنوان نقاط اولیه
init_points = [U1, L1, ...]  # از HRNet

# Active contour برای یافتن مرز دقیق
contour = active_contour(
    image,
    init_points,
    alpha=0.015,  # پیوستگی
    beta=10,      # صافی
    gamma=0.001,  # گام
)
```

**مزایا:**
- ✅ نیازی به مدل جدید نیست
- ✅ سریع

**معایب:**
- ❌ نیاز به نقاط اولیه خوب
- ❌ ممکن است برای شکل‌های پیچیده ناکافی باشد

---

### 3️⃣ ترکیبی: HRNet + Spline Interpolation

#### روش پیشنهادی (بهترین برای شروع):

```javascript
// 1. تشخیص نقاط کلیدی دندان‌ها با HRNet
const landmarks = {
  U1_tip: { x, y },
  U1_root: { x, y },
  L1_tip: { x, y },
  L1_root: { x, y },
  // ...
};

// 2. ایجاد خطوط انحنادار بین نقاط
function createToothBoundary(points) {
  // Catmull-Rom Spline for smooth curves
  const spline = new CatmullRomSpline(points);
  return spline.getPoints(100); // 100 نقطه برای نرمی
}

// 3. رسم روی Canvas
ctx.beginPath();
const boundary = createToothBoundary([U1_tip, U1_edge1, U1_root, U1_edge2]);
boundary.forEach((point, i) => {
  if (i === 0) ctx.moveTo(point.x, point.y);
  else ctx.lineTo(point.x, point.y);
});
ctx.closePath();
ctx.stroke();
```

**مزایا:**
- ✅ استفاده از مدل موجود (HRNet)
- ✅ خطوط صاف و انحنادار
- ✅ سریع و سبک
- ✅ قابل ویرایش توسط کاربر

**معایب:**
- ❌ نیمه خودکار (نه کاملاً AI)
- ❌ HRNet فعلی ممکن است نقاط دندان‌ها را نداشته باشد

---

## 💡 پیشنهاد نهایی

### مرحله 1: راه‌حل سریع (فعلی)
```
1. کاربر "Trace Mode" را فعال می‌کند
2. نقاط کلیدی را کلیک می‌کند (مثلاً 4 نقطه برای یک دندان)
3. سیستم با Catmull-Rom Spline خط انحنادار می‌کشد
4. کاربر می‌تواند نقاط را drag کند برای اصلاح
```

**پیاده‌سازی:**
```javascript
// فعلی - Tracing با نقاط دستی
const [tracingPoints, setTracingPoints] = useState([]);

// بهبود - افزودن Spline
function drawTracingWithSpline(points) {
  if (points.length < 3) return; // حداقل 3 نقطه
  
  const spline = createCatmullRomSpline(points);
  const smoothPoints = spline.getPoints(50);
  
  ctx.beginPath();
  smoothPoints.forEach((p, i) => {
    if (i === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.stroke();
}
```

### مرحله 2: راه‌حل AI (آینده)
```
1. Fine-tune کردن HRNet برای تشخیص نقاط دندان‌ها
   - افزودن Landmarks: U1_tip, U1_root, ... (×32 دندان)
   - Training با Dataset دندانی

2. یا استفاده از مدل U-Net جداگانه
   - تشخیص خودکار مرزهای دندان‌ها
   - خروجی: Binary mask برای هر دندان
   
3. نمایش نتیجه به کاربر با قابلیت ویرایش
```

---

## 📊 مقایسه روش‌ها

| روش | دقت | سرعت | پیچیدگی | نیاز به Dataset | هزینه |
|-----|-----|------|---------|----------------|-------|
| **دستی + Spline** | متوسط | خیلی سریع | کم | ❌ | رایگان |
| **Active Contour** | خوب | سریع | متوسط | ❌ | رایگان |
| **U-Net** | عالی | متوسط | زیاد | ✅ | GPU |
| **Mask R-CNN** | عالی | کند | خیلی زیاد | ✅ | GPU قوی |

---

## 🎯 توصیه

### برای Production فعلی:
✅ **دستی + Spline Interpolation**
- سریع برای پیاده‌سازی
- کاربر کنترل کامل دارد
- خطوط صاف و حرفه‌ای

### برای آینده (اگر بودجه و زمان هست):
✅ **HRNet Fine-tuning** یا **U-Net**
- تشخیص خودکار دندان‌ها
- صرفه‌جویی در وقت کاربر
- حرفه‌ای‌تر

---

## 🔧 پیاده‌سازی پیشنهادی (Spline)

### نصب کتابخانه:
```bash
npm install d3-shape
# یا
npm install cubic-spline
```

### کد:
```javascript
import { curveCatmullRom } from 'd3-shape';

function createSmoothCurve(points, ctx) {
  if (points.length < 2) return;
  
  const line = d3.line()
    .curve(curveCatmullRom.alpha(0.5))
    .context(ctx);
  
  ctx.beginPath();
  line(points.map(p => [p.x, p.y]));
  ctx.stroke();
}

// استفاده
createSmoothCurve(tracingPoints, ctx);
```

---

## 📝 نتیجه‌گیری

**وضعیت فعلی:**
- ✅ Trace دستی کار می‌کند
- ✅ می‌توان Spline اضافه کرد (برای خطوط انحنادار)

**برای AI کامل:**
- ⚠️ نیاز به مدل Segmentation
- ⚠️ نیاز به Dataset
- ⚠️ نیاز به Training
- ⚠️ زمان‌بر است (چند هفته)

**پیشنهاد:**
1. **فاز 1 (فعلی):** Trace دستی + Spline → اضافه کنیم؟
2. **فاز 2 (آینده):** مدل AI برای Segmentation دندان‌ها

---

تاریخ: 30 اکتبر 2025

