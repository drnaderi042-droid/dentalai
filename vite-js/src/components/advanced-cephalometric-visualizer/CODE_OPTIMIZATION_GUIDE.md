# راهنمای بهینه‌سازی و ساده‌سازی کد

## مشکلات شناسایی شده

### 1. کدهای تکراری برای پیدا کردن لندمارک‌ها
**مشکل:** در هر آنالیز، لندمارک‌های مشابه با `findLandmarkVariations` پیدا می‌شوند.

**راهکار:**
```javascript
// ایجاد یک object مرکزی برای نام‌های لندمارک‌ها
const LANDMARK_VARIATIONS = {
  S: ['S', 's', 'Sella', 'sella'],
  N: ['N', 'n', 'Nasion', 'nasion'],
  A: ['A', 'a', 'Point A'],
  B: ['B', 'b', 'Point B'],
  Pog: ['Pog', 'pog', 'POG', 'pogonion', 'Pogonion'],
  Go: ['Go', 'go', 'GO', 'gonion', 'Gonion'],
  Me: ['Me', 'me', 'ME', 'menton', 'Menton'],
  Or: ['Or', 'or', 'OR', 'orbitale', 'Orbitale'],
  Po: ['Po', 'po', 'PO', 'porion', 'Porion'],
  UL: ['UL', 'ul', 'UL′', 'UL\'', 'ul′', 'ul\'', 'Ls', 'ls', 'LS', 'Labiale Superius'],
  LL: ['LL', 'll', 'LL′', 'LL\'', 'll′', 'll\'', 'Li', 'li', 'LI', 'Labiale Inferius'],
  'Pog\'': ['Pog′', 'Pog\'', 'pog′', 'pog\'', 'Soft tissue Pogonion'],
  'N\'': ['N′', 'N\'', 'n′', 'n\'', 'Nprime', 'nprime'],
  // ... سایر لندمارک‌ها
};

// تابع helper برای پیدا کردن لندمارک
const getLandmark = (name) => {
  const variations = LANDMARK_VARIATIONS[name];
  return variations ? findLandmarkVariations(variations) : null;
};

// استفاده:
const sLandmark = getLandmark('S');
const nLandmark = getLandmark('N');
```

**صرفه‌جویی:** کاهش ~30-40% کد تکراری

---

### 2. منطق تکراری برای تبدیل پیکسل به میلی‌متر
**مشکل:** کد تبدیل پیکسل به میلی‌متر در چندین جا تکرار شده است.

**راهکار:**
```javascript
// تابع مشترک برای تبدیل فاصله به میلی‌متر
const convertDistanceToMm = (pixelDistance, measurementKey, alternativeKeys = []) => {
  // اول از currentMeasurements بررسی کن
  if (currentMeasurements && currentMeasurements[measurementKey] !== undefined) {
    return parseFloat(currentMeasurements[measurementKey]);
  }
  
  // سپس از alternative keys
  for (const key of alternativeKeys) {
    if (currentMeasurements && currentMeasurements[key] !== undefined) {
      return parseFloat(currentMeasurements[key]);
    }
  }
  
  // در نهایت از pixelToMmConversion
  if (pixelToMmConversion && pixelToMmConversion > 0) {
    return pixelDistance * pixelToMmConversion;
  }
  
  return null;
};

// استفاده:
const sGoDistanceMm = convertDistanceToMm(
  sGoDistancePixels,
  'S-Go',
  ['Posterior Facial Height (S-Go)']
);
```

**صرفه‌جویی:** کاهش ~20-25% کد تکراری

---

### 3. شرط‌های تکراری برای بررسی analysisType
**مشکل:** شرط‌های `if (analysisType === 'xxx')` در چندین جا تکرار می‌شوند.

**راهکار:**
```javascript
// ایجاد یک object برای نگاشت آنالیزها به توابع رسم
const ANALYSIS_DRAWERS = {
  mcnamara: drawMcNamaraAnalysis,
  wits: drawWitsAnalysis,
  steiner: drawSteinerAnalysis,
  ricketts: drawRickettsAnalysis,
  holdaway: drawHoldawayAnalysis,
  jarabak: drawJarabakAnalysis,
  sassouni: drawSassouniAnalysis,
  leganBurstone: drawLeganBurstoneAnalysis,
  arnettMcLaughlin: drawArnettMcLaughlinAnalysis,
  softTissueAngular: drawSoftTissueAngularAnalysis,
  tweed: drawTweedAnalysis,
};

// استفاده:
if (showMeasurements) {
  const drawer = ANALYSIS_DRAWERS[analysisType];
  if (drawer) {
    drawer(ctx, landmarks, currentMeasurements, /* ... */);
  }
  
  // برای 'all' همه را رسم کن
  if (analysisType === 'all') {
    Object.values(ANALYSIS_DRAWERS).forEach(drawer => {
      drawer(ctx, landmarks, currentMeasurements, /* ... */);
    });
  }
}
```

**صرفه‌جویی:** کاهش ~15-20% کد شرطی

---

### 4. کدهای تکراری برای رسم خطوط با امتداد
**مشکل:** منطق رسم خطوط با امتداد در چندین جا تکرار شده است.

**راهکار:**
```javascript
// تابع مشترک برای رسم خط با امتداد (قبلاً ایجاد شده - drawLineWithExtensions)
// اما می‌توان آن را بهبود داد:

const drawLineWithExtensions = useCallback((
  startPos, 
  endPos, 
  startColor, 
  endColor, 
  lineWidth = actualLineWidth, 
  drawExtensions = true,
  label = null,
  value = null
) => {
  if (!startPos || !endPos) return;
  
  // رسم خط اصلی
  // ... کد موجود ...
  
  // اگر label و value وجود دارد، نمایش بده
  if (label && value !== null && value !== undefined) {
    const midX = (startPos.x + endPos.x) / 2;
    const midY = (startPos.y + endPos.y) / 2;
    const angle = Math.atan2(endPos.y - startPos.y, endPos.x - startPos.x);
    drawRotatedTextWithBackground(
      `${label}: ${Math.round(value)}mm`,
      midX, midY, angle, fontSize, '#FFD700'
    );
  }
}, [actualLineWidth, canvas, containerRef, ctx, useGradientLines, zoom, landmarkColors]);
```

**صرفه‌جویی:** کاهش ~10-15% کد تکراری

---

### 5. داده‌های Hard-coded برای خطوط hard tissue
**مشکل:** لیست خطوط hard tissue در کد hard-coded شده است.

**راهکار:**
```javascript
// ایجاد constant برای خطوط hard tissue
const HARD_TISSUE_LINES = [
  ['S', 'N'], ['Or', 'Po'], ['ANS', 'PNS'],
  ['N', 'A'], ['N', 'B'], ['N', 'Pog'],
  ['S', 'A'], ['S', 'B'], ['Go', 'Me'],
  ['Go', 'Gn'], ['Ar', 'Go'], ['S', 'Ar'],
  ['S', 'Go'], ['U1', 'L1'], ['U1', 'A'],
  ['L1', 'A'], ['S', 'U1'], ['L1', 'Me'],
  ['Or', 'L1'], ['Po', 'L1'], ['Gn', 'Pt'],
  ['Ba', 'Pt'], ['N', 'Me'], ['Co', 'Gn'],
  ['L1', 'LMT'], ['A', 'B'], ['L1A', 'L1'],
];

// تابع helper برای بررسی
const isHardTissueLine = (start, end) => {
  return HARD_TISSUE_LINES.some(([s, e]) => 
    (start === s && end === e) || (start === e && end === s)
  );
};

// استفاده:
if (isSoftTissueAnalysis && isHardTissueLine(start, end)) {
  return; // این خط را رسم نکن
}
```

**صرفه‌جویی:** کاهش ~5-10% کد

---

### 6. منطق تکراری برای محاسبه و نمایش زوایا
**مشکل:** منطق محاسبه زاویه و نمایش آن در چندین جا تکرار شده است.

**راهکار:**
```javascript
// تابع مشترک برای رسم زاویه با fallback محاسبه
const drawAngleWithFallback = (
  p1, vertex, p2,
  measurementKey,
  label,
  fontSize,
  radius,
  landmark1Name = null,
  landmark2Name = null
) => {
  let angleValue = currentMeasurements && currentMeasurements[measurementKey] 
    ? parseFloat(currentMeasurements[measurementKey]) 
    : undefined;
  
  // اگر مقدار وجود ندارد، محاسبه کن
  if (angleValue === undefined || angleValue === null || isNaN(angleValue)) {
    const calculatedAngle = calculateAngleBetweenLines(p1, vertex, vertex, p2);
    if (calculatedAngle !== null && !isNaN(calculatedAngle)) {
      angleValue = calculatedAngle;
    }
  }
  
  // رسم زاویه
  if (angleValue !== undefined && angleValue !== null && !isNaN(angleValue)) {
    drawAngle(p1, vertex, p2, angleValue, label, '#FFD700', fontSize, radius, landmark1Name, landmark2Name);
  }
};

// استفاده:
drawAngleWithFallback(
  nPos, aPos, pogPos,
  'N-A-Pog',
  'N-A-Pog',
  fontSize,
  radius,
  'N',
  'Pog'
);
```

**صرفه‌جویی:** کاهش ~15-20% کد تکراری

---

### 7. بهینه‌سازی تابع getLineColor
**مشکل:** تابع `getLineColor` در چندین جا فراخوانی می‌شود و منطق پیچیده‌ای دارد.

**راهکار:**
```javascript
// Cache برای رنگ‌های لندمارک‌ها
const landmarkColorCache = useMemo(() => {
  const cache = {};
  const getColorForLandmark = (name) => {
    if (cache[name]) return cache[name];
    
    // منطق موجود برای پیدا کردن رنگ
    const color = /* ... */;
    cache[name] = color;
    return color;
  };
  
  return { getColorForLandmark };
}, [landmarkColors]);

// استفاده از cache در getLineColor
const getLineColor = (startName, endName, startPos, endPos) => {
  const startColor = landmarkColorCache.getColorForLandmark(startName);
  const endColor = landmarkColorCache.getColorForLandmark(endName);
  // ...
};
```

**صرفه‌جویی:** بهبود عملکرد و کاهش ~5% کد

---

### 8. استخراج توابع بزرگ به توابع کوچک‌تر
**مشکل:** تابع `drawCanvas` خیلی بزرگ است (~4000+ خط).

**راهکار:**
```javascript
// استخراج منطق هر آنالیز به تابع جداگانه (بدون تقسیم فایل)
const drawMcNamaraAnalysis = useCallback((ctx, landmarks, currentMeasurements, /* ... */) => {
  // تمام کد مربوط به McNamara
}, [/* dependencies */]);

const drawWitsAnalysis = useCallback((ctx, landmarks, currentMeasurements, /* ... */) => {
  // تمام کد مربوط به Wits
}, [/* dependencies */]);

// در drawCanvas:
if (showMeasurements && analysisType === 'mcnamara') {
  drawMcNamaraAnalysis(ctx, landmarks, currentMeasurements, /* ... */);
}
```

**صرفه‌جویی:** بهبود خوانایی و کاهش پیچیدگی

---

### 9. استفاده از Map برای جستجوی سریع‌تر
**مشکل:** استفاده از `Array.some()` برای بررسی تکراری بودن خطوط.

**راهکار:**
```javascript
// استفاده از Set برای بررسی سریع‌تر
const drawnLinesSet = useRef(new Set());

const getLineKey = (start, end) => {
  return start < end ? `${start}-${end}` : `${end}-${start}`;
};

// بررسی تکراری بودن
const lineKey = getLineKey(start, end);
if (drawnLinesSet.current.has(lineKey)) {
  return; // تکراری است
}
drawnLinesSet.current.add(lineKey);
```

**صرفه‌جویی:** بهبود عملکرد برای خطوط زیاد

---

### 10. استفاده از Object برای نگاشت آنالیزها به خطوط
**مشکل:** شرط‌های تکراری برای تعیین خطوط هر آنالیز.

**راهکار:**
```javascript
// نگاشت آنالیزها به خطوط مورد نیاز
const ANALYSIS_LINES = {
  mcnamara: {
    exclude: [['Go', 'Me'], ['Me', 'Go']],
    include: [['S', 'Go'], ['N', 'Me'], ['N', 'A'], ['A', 'Pog']],
  },
  jarabak: {
    exclude: [['Or', 'Po'], ['Po', 'Or'], ['ANS', 'PNS'], ['PNS', 'ANS']],
    include: [['S', 'N'], ['N', 'ANS'], ['ANS', 'Me'], ['Me', 'Go'], ['Go', 'Ar'], ['Ar', 'S']],
  },
  // ...
};

// تابع helper
const shouldDrawLine = (start, end, analysisType) => {
  const config = ANALYSIS_LINES[analysisType];
  if (!config) return true;
  
  // بررسی exclude
  if (config.exclude?.some(([s, e]) => 
    (start === s && end === e) || (start === e && end === s)
  )) {
    return false;
  }
  
  // اگر include وجود دارد، فقط آن‌ها را رسم کن
  if (config.include) {
    return config.include.some(([s, e]) => 
      (start === s && end === e) || (start === e && end === s)
    );
  }
  
  return true;
};
```

**صرفه‌جویی:** کاهش ~10-15% کد شرطی

---

## خلاصه صرفه‌جویی

| بهینه‌سازی | کاهش حجم کد | اولویت |
|------------|------------|--------|
| 1. LANDMARK_VARIATIONS | 30-40% | بالا |
| 2. convertDistanceToMm | 20-25% | بالا |
| 3. ANALYSIS_DRAWERS | 15-20% | متوسط |
| 4. drawLineWithExtensions بهبود | 10-15% | متوسط |
| 5. HARD_TISSUE_LINES | 5-10% | پایین |
| 6. drawAngleWithFallback | 15-20% | بالا |
| 7. landmarkColorCache | 5% + بهبود عملکرد | متوسط |
| 8. استخراج توابع | بهبود خوانایی | بالا |
| 9. استفاده از Set | بهبود عملکرد | پایین |
| 10. ANALYSIS_LINES | 10-15% | متوسط |

**کل صرفه‌جویی پیش‌بینی شده: 40-60% کاهش حجم کد**

---

## نکات مهم

1. **تست کامل:** بعد از هر بهینه‌سازی، تمام آنالیزها را تست کنید
2. **تدریجی:** بهینه‌سازی‌ها را یکی یکی اعمال کنید
3. **بررسی عملکرد:** مطمئن شوید که بهینه‌سازی‌ها عملکرد را بهبود می‌دهند
4. **مستندسازی:** تغییرات را مستند کنید

