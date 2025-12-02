# استفاده از P1/P2 برای تبدیل پیکسل به میلی‌متر

## خلاصه

نقاط **p1** و **p2** نقاط کالیبراسیون هستند که فاصله آن‌ها **10 میلی‌متر (1 سانتی‌متر)** است. از این فاصله برای تبدیل دقیق پیکسل به میلی‌متر استفاده می‌شود.

---

## نحوه کار

### 1. تشخیص P1/P2
- مدل ML نقاط p1 و p2 را در upper right corner تصویر تشخیص می‌دهد
- p2: نقطه بالایی
- p1: نقطه پایینی
- فاصله بین آن‌ها: **10mm** (1cm)

### 2. محاسبه Conversion Factor
```javascript
// فاصله بین p1 و p2 در پیکسل
const distance = Math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2);

// تبدیل به میلی‌متر: 10mm / distance (pixels)
const mmPerPixel = 10 / distance;

// مثال: اگر distance = 90 pixels
// mmPerPixel = 10 / 90 = 0.111 mm/pixel
```

### 3. استفاده در آنالیزها
تمام آنالیزهایی که با **طول و فاصله** کار می‌کنند از این conversion factor استفاده می‌کنند:

#### McNamara Analysis
- **Co-A** (Maxillary Length): فاصله Co تا A
- **Co-Gn** (Mandibular Length): فاصله Co تا Gn
- **Lower Face Height**: فاصله ANS تا Me
- **Upper Face Height**: فاصله N تا ANS
- **Wits Appraisal**: فاصله افقی A تا B

#### Ricketts Analysis
- **Lower Face Height**: (ANS-Me/N-Me) × 100
- **Facial Height Ratio**: (ANS-Me/N-Me) × 100

#### Bjork Analysis
- **S-Go**: فاصله S تا Go
- **S-Ar**: فاصله S تا Ar
- **Go-Gn**: فاصله Go تا Gn
- **Go-Me**: فاصله Go تا Me

#### Jarabak Analysis
- **S-Go**: فاصله S تا Go
- **Ar-Go**: فاصله Ar تا Go

---

## کد پیاده‌سازی

### 1. محاسبه Conversion Factor
```javascript
const calculatePixelToMmConversion = (detectedPoints) => {
  if (detectedPoints.length < 2) {
    return 0.11; // fallback
  }
  
  const p2 = detectedPoints[0]; // Upper point
  const p1 = detectedPoints[1]; // Lower point
  
  const distance = Math.sqrt(
    (p2.x - p1.x)**2 + (p2.y - p1.y)**2
  );
  
  // فاصله p1-p2 = 10mm
  const mmPerPixel = 10 / distance;
  
  return mmPerPixel;
};
```

### 2. استفاده در محاسبه فاصله
```javascript
const calculateDistance = (p1, p2) => {
  const pixelDistance = Math.sqrt(
    (p2.x - p1.x)**2 + (p2.y - p1.y)**2
  );
  
  // تبدیل به میلی‌متر
  const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
  const mmDistance = pixelDistance * conversionFactor;
  
  return mmDistance;
};
```

### 3. استفاده در آنالیزها
```javascript
// مثال: Co-Gn در McNamara Analysis
const coLandmark = getLandmark(['Co', 'co', 'CO', 'condylion']);
const gnLandmark = getLandmark(['Gn', 'gn', 'GN', 'gnathion']);

if (coLandmark && gnLandmark) {
  const coGnDist = calculateDistance(coLandmark, gnLandmark);
  measures['Co-Gn'] = Math.round(coGnDist * 10) / 10; // در میلی‌متر
  measures['Mandibular Length'] = measures['Co-Gn'];
}
```

---

## مزایا

1. **دقت بالا**: استفاده از calibration واقعی (10mm) به جای تخمین
2. **سازگاری**: کار می‌کند با هر اندازه تصویر و DPI
3. **قابلیت اطمینان**: اگر calibration fail کند، از fallback (0.11 mm/pixel) استفاده می‌شود

---

## Fallback

اگر P1/P2 تشخیص داده نشوند یا validation fail کند:
- از conversion factor پیش‌فرض استفاده می‌شود: **0.11 mm/pixel**
- این مقدار برای اکثر radiograph‌ها مناسب است

---

## مثال

### تصویر با DPI = 228
- فاصله p1-p2 در پیکسل: 90 pixels
- Conversion factor: 10 / 90 = **0.111 mm/pixel**
- Co-Gn در پیکسل: 450 pixels
- Co-Gn در میلی‌متر: 450 × 0.111 = **50.0 mm**

### تصویر با DPI = 300
- فاصله p1-p2 در پیکسل: 118 pixels
- Conversion factor: 10 / 118 = **0.085 mm/pixel**
- Co-Gn در پیکسل: 588 pixels
- Co-Gn در میلی‌متر: 588 × 0.085 = **50.0 mm**

---

## نکات مهم

1. **P1/P2 باید عمودی باشند**: `dx < 50` و `50 < dy < 200`
2. **P1/P2 باید در upper right corner باشند**: 70-99% x, 0-30% y
3. **Conversion factor باید در محدوده واقعی باشد**: 0.05-0.20 mm/pixel
4. **تمام فاصله‌ها به میلی‌متر تبدیل می‌شوند**: برای دقت پزشکی

---

## آنالیزهای استفاده‌کننده

### McNamara
- ✅ Co-A (Maxillary Length)
- ✅ Co-Gn (Mandibular Length)
- ✅ Lower Face Height
- ✅ Upper Face Height
- ✅ Wits Appraisal

### Ricketts
- ✅ Lower Face Height
- ✅ Facial Height Ratio

### Bjork
- ✅ S-Go
- ✅ S-Ar
- ✅ Go-Gn
- ✅ Go-Me

### Jarabak
- ✅ S-Go
- ✅ Ar-Go

### Sassouni
- ✅ Ar-Co//Co-Gn (زاویه)

---

## نتیجه

با استفاده از P1/P2 برای calibration، تمام اندازه‌گیری‌های طول و فاصله در آنالیزها **دقیق** و **قابل اعتماد** هستند.













