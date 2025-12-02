# راهنمای پیاده‌سازی سیستم پریودونتیکس 🦷

## ✅ وضعیت فعلی

### کامل شده:
- ✅ صفحه لیست بیماران (`/dashboard/periodontics`)
- ✅ افزودن/ویرایش/حذف بیمار
- ✅ نمایش اطلاعات پایه بیماران

### در حال توسعه:
- 🔄 صفحه جزئیات بیمار با 3 تب
- 🔄 چارت پریودونتال
- 🔄 سیستم آنالیز و طرح درمان

---

## 📁 ساختار فایل‌ها

```
vite-js/src/sections/periodontics/
├── view/
│   ├── periodontics-view.jsx          ✅ (لیست بیماران)
│   └── index.js
├── patient/
│   ├── view/
│   │   ├── patient-periodontics-view.jsx    📝 (صفحه اصلی بیمار - 3 تب)
│   │   └── index.js
│   └── components/
│       ├── patient-info-tab.jsx             📝 (تب اطلاعات + بیماری‌های زمینه‌ای)
│       ├── periodontal-chart-tab.jsx        📝 (تب چارت پریودونتال)
│       ├── analysis-tab.jsx                 📝 (تب آنالیز و طرح درمان)
│       └── index.js
├── components/
│   ├── periodontal-chart/
│   │   ├── periodontal-chart.jsx            📝 (کامپوننت اصلی چارت)
│   │   ├── tooth-chart.jsx                  📝 (چارت هر دندان)
│   │   ├── measurement-input.jsx            📝 (ورودی اندازه‌گیری‌ها)
│   │   └── index.js
│   ├── analysis/
│   │   ├── bop-analysis.jsx                 📝 (آنالیز BOP)
│   │   ├── attachment-loss-analysis.jsx     📝 (آنالیز Attachment Loss)
│   │   ├── pocket-depth-analysis.jsx        📝 (آنالیز عمق پاکت)
│   │   ├── disease-classification.jsx       📝 (طبقه‌بندی بیماری)
│   │   ├── treatment-plan.jsx               📝 (طرح درمان)
│   │   └── index.js
│   └── index.js
└── PERIODONTICS_IMPLEMENTATION_GUIDE.md
```

---

## 🎯 ویژگی‌های هر تب

### 1. تب اطلاعات کلی (`patient-info-tab.jsx`)

**محتوا**:
- نام، نام خانوادگی، سن، تلفن
- تاریخ شروع درمان، ویزیت بعدی
- تشخیص، طرح درمان
- **بیماری‌های زمینه‌ای** (Systemic Diseases):
  - ✅ دیابت (Diabetes)
  - ✅ فشار خون بالا (Hypertension)
  - ✅ بیماری‌های قلبی (Cardiovascular Disease)
  - ✅ آرتریت روماتوئید (Rheumatoid Arthritis)
  - ✅ استئوپروز (Osteoporosis)
  - ✅ HIV/AIDS
  - ✅ هپاتیت (Hepatitis)
  - ✅ مصرف سیگار (Smoking)
  - ✅ مصرف الکل (Alcohol)
  - ✅ استرس (Stress)
  - ✅ بارداری (Pregnancy)
  - ✅ سایر...

**فیلدهای Database**:
```javascript
{
  medicalHistory: {
    diabetes: boolean,
    hypertension: boolean,
    cardiovascularDisease: boolean,
    rheumatoidArthritis: boolean,
    osteoporosis: boolean,
    hiv: boolean,
    hepatitis: boolean,
    smoking: boolean,
    smokingPackYears: number, // اگر سیگاری است
    alcohol: boolean,
    stress: boolean,
    pregnancy: boolean,
    other: string, // سایر بیماری‌ها
  }
}
```

---

### 2. تب چارت پریودونتال (`periodontal-chart-tab.jsx`)

**مشابه**: https://www.periodontalchart-online.com/uk/

**اطلاعات هر دندان** (16 دندان فک بالا + 16 دندان فک پایین):

#### Facial (سطح رویی):
- **Pocket Depth** (عمق پاکت): 3 نقطه (Mesial, Central, Distal)
- **Gingival Margin** (حاشیه لثه): 3 نقطه
- **CAL** (Clinical Attachment Level): خودکار محاسبه می‌شود
- **Bleeding on Probing (BOP)**: 3 نقطه (checkbox)
- **Suppuration** (چرک): 3 نقطه (checkbox)
- **Furcation** (منشعب): برای دندان‌های چند ریشه
- **Mobility** (تحرک): Grade 0-3
- **Plaque**: checkbox

#### Lingual (سطح زبانی):
- همان موارد بالا

**ساختار Data**:
```javascript
{
  periodontalChart: {
    teeth: {
      "1": { // شماره دندان
        facial: {
          pocketDepth: [3, 3, 3], // mm
          gingivalMargin: [0, 0, 0], // mm
          bleeding: [false, false, false],
          suppuration: [false, false, false],
          furcation: null, // Grade I, II, III
          mobility: 0, // 0-3
          plaque: false
        },
        lingual: {
          // same as facial
        },
        missing: false,
        implant: false
      },
      // ... برای 32 دندان
    },
    date: Date,
    notes: string
  }
}
```

**ویژگی‌های بصری**:
- نمایش دندان‌ها مانند تصویر
- رنگ‌بندی:
  - سبز: سالم (Pocket Depth ≤ 3mm)
  - زرد: Gingivitis (Pocket Depth 4-5mm)
  - نارنجی: Periodontitis خفیف (6mm)
  - قرمز: Periodontitis شدید (≥7mm)
- نمایش BOP با نقاط قرمز
- Dropdown برای هر دندان: missing, implant, crown, etc.

---

### 3. تب آنالیز و طرح درمان (`analysis-tab.jsx`)

#### محاسبات خودکار:

**1. BOP % (Bleeding on Probing)**
```javascript
// درصد سطوحی که BOP داشته‌اند
BOP% = (تعداد سطوح با BOP / کل سطوح بررسی شده) × 100

// BOP% > 30%: التهاب فعال
// BOP% < 10%: سلامت پریودنشیال
```

**2. Attachment Loss**
```javascript
CAL = Pocket Depth + Gingival Margin

// میانگین CAL برای هر دندان و کل دهان
```

**3. Pocket Depth Analysis**
```javascript
// تعداد و درصد سطوح با عمق مختلف:
// - Healthy: ≤3mm
// - Mild: 4-5mm
// - Moderate: 6mm
// - Severe: ≥7mm
```

**4. Disease Extent** (گستردگی بیماری)
```javascript
// درصد دندان‌های درگیر
affected = (dents with CAL ≥ 3mm / total teeth) × 100

// Localized: < 30% دندان‌ها
// Generalized: ≥ 30% دندان‌ها
```

**5. Disease Severity** (شدت بیماری)
```javascript
// Stage I: CAL 1-2mm
// Stage II: CAL 3-4mm
// Stage III: CAL ≥5mm
// Stage IV: CAL ≥5mm + tooth loss
```

**6. Bone Loss Calculation**
```javascript
// تخمین از روی CAL
BoneLoss% = (Average CAL / Root Length) × 100

// Root Length معمولاً 10-14mm
```

#### نمودارها:

**1. BOP Distribution Chart** (نمودار توزیع BOP)
- Bar chart: BOP% در هر کادران (4 ربع)
- Line chart: روند BOP در طول زمان

**2. Pocket Depth Distribution** (توزیع عمق پاکت)
- Histogram: تعداد سطوح در هر بازه عمق
- Heat map: نمایش دندان‌ها با رنگ بر اساس عمق

**3. CAL Spider Chart** (نمودار عنکبوتی CAL)
- نمایش CAL در 6 سکستانت

**4. Disease Progression** (روند بیماری)
- Line chart: تغییر میانگین CAL در طول زمان
- Timeline: تاریخ‌های ویزیت و تغییرات

#### طرح درمان خودکار:

**الگوریتم تعیین طرح درمان**:

```javascript
function generateTreatmentPlan(chartData) {
  const plan = {
    phase1: [], // Initial Therapy
    phase2: [], // Surgical Phase
    phase3: [], // Restorative Phase
    phase4: [], // Maintenance
  };

  // Phase 1: Initial Therapy (همیشه)
  plan.phase1.push("Patient Education & Oral Hygiene Instruction");
  plan.phase1.push("Scaling & Root Planing (SRP)");
  
  // اگر BOP% > 30%
  if (bopPercentage > 30) {
    plan.phase1.push("Intensive Plaque Control");
    plan.phase1.push("Antimicrobial Mouth Rinse (Chlorhexidine 0.12%)");
  }
  
  // اگر Pocket Depth > 5mm
  if (hasPocketsOver5mm) {
    plan.phase1.push("Local Antibiotic Delivery (if needed)");
  }
  
  // Phase 2: Surgical (اگر لازم باشد)
  if (hasPocketsOver6mm || hasAttachmentLoss > 5) {
    plan.phase2.push("Re-evaluation (4-6 weeks after SRP)");
    
    if (hasPocketsOver6mm) {
      plan.phase2.push("Flap Surgery / Osseous Surgery");
    }
    
    if (hasRecession) {
      plan.phase2.push("Gingival Grafting (if indicated)");
    }
    
    if (hasBoneDefects) {
      plan.phase2.push("Bone Grafting / Guided Tissue Regeneration");
    }
  }
  
  // Phase 3: Restorative
  if (hasMobility) {
    plan.phase3.push("Splinting (if severe mobility)");
  }
  
  if (missingTeeth) {
    plan.phase3.push("Prosthetic Rehabilitation");
  }
  
  // Phase 4: Maintenance (همیشه)
  const maintenanceInterval = bopPercentage < 10 ? "6 months" : "3 months";
  plan.phase4.push(`Periodontal Maintenance (SPT) every ${maintenanceInterval}`);
  plan.phase4.push("Monitor BOP, CAL, and Pocket Depths");
  
  // توصیه‌های اضافی
  if (medicalHistory.diabetes) {
    plan.phase1.push("⚠️ Glycemic Control - Coordinate with physician");
  }
  
  if (medicalHistory.smoking) {
    plan.phase1.push("🚭 Smoking Cessation Counseling (CRITICAL)");
  }
  
  return plan;
}
```

**نمایش طرح درمان**:
```
Phase I: Initial Therapy
✓ Patient Education
✓ Oral Hygiene Instruction
✓ Scaling & Root Planing (Full Mouth)
✓ Antimicrobial Rinse (0.12% Chlorhexidine)

Phase II: Re-evaluation & Surgery (4-6 weeks)
○ Re-assessment of pocket depths
○ Flap surgery for teeth #3, #14, #19, #30
○ Bone grafting for #19

Phase III: Restorative
○ Crown for #3
○ Implant consultation for missing #18

Phase IV: Maintenance
✓ SPT every 3 months
✓ Monitor BOP and CAL
```

---

## 🎨 طراحی UI

### رنگ‌بندی وضعیت:
```javascript
const healthColors = {
  healthy: '#4CAF50',    // سبز
  mild: '#FFC107',       // زرد
  moderate: '#FF9800',   // نارنجی
  severe: '#F44336',     // قرمز
};
```

### نمودارها:
- استفاده از **Recharts** برای نمودارها
- نمودار میله‌ای برای BOP
- نمودار خطی برای روند
- Heat map برای نمایش دندان‌ها

---

## 💾 Database Schema

```prisma
model Patient {
  id                    String      @id @default(uuid())
  firstName             String
  lastName              String
  phone                 String?
  age                   Int?
  specialty             Specialty   @default(GENERAL)
  
  // Medical History
  medicalHistory        Json?       // بیماری‌های زمینه‌ای
  
  // Periodontal Charts (چند چارت برای follow-up)
  periodontalCharts     PeriodontalChart[]
  
  createdAt             DateTime    @default(now())
  updatedAt             DateTime    @updatedAt
}

model PeriodontalChart {
  id          String    @id @default(uuid())
  patientId   String
  patient     Patient   @relation(fields: [patientId], references: [id])
  
  date        DateTime  @default(now())
  teeth       Json      // ساختار JSON برای 32 دندان
  notes       String?
  
  // Calculated fields
  bopPercentage       Float?
  avgPocketDepth      Float?
  avgCAL              Float?
  diseaseExtent       String?  // Localized/Generalized
  diseaseSeverity     String?  // Stage I-IV
  
  // Treatment plan
  treatmentPlan       Json?
  
  createdAt   DateTime  @default(now())
  updatedAt   DateTime  @updatedAt
}
```

---

## 🚀 مراحل پیاده‌سازی

### مرحله 1: ساختار پایه ✅
- [x] صفحه لیست بیماران
- [ ] صفحه جزئیات بیمار با 3 تب

### مرحله 2: تب اطلاعات
- [ ] فرم اطلاعات پایه
- [ ] Checkboxes بیماری‌های زمینه‌ای
- [ ] ذخیره در database

### مرحله 3: چارت پریودونتال
- [ ] کامپوننت نمایش دندان‌ها
- [ ] Input fields برای measurements
- [ ] BOP checkboxes
- [ ] Mobility, Furcation inputs
- [ ] ذخیره چارت

### مرحله 4: آنالیز
- [ ] محاسبات خودکار (BOP%, CAL, etc.)
- [ ] نمودارها
- [ ] Disease classification
- [ ] Bone loss estimation

### مرحله 5: طرح درمان
- [ ] الگوریتم تولید خودکار
- [ ] نمایش فازهای درمان
- [ ] توصیه‌های شخصی‌سازی شده

### مرحله 6: Features اضافی
- [ ] مقایسه چارت‌های مختلف
- [ ] Export PDF
- [ ] گزارش پیشرفت
- [ ] یادآوری ویزیت

---

## 📝 نکات مهم

1. **Validation**: همه ورودی‌ها باید validate شوند (Pocket Depth: 0-15mm)
2. **Auto-calculation**: CAL باید خودکار محاسبه شود
3. **Color coding**: رنگ‌بندی واضح برای شدت بیماری
4. **Responsive**: UI باید در موبایل هم کار کند
5. **History**: ذخیره تاریخچه چارت‌ها برای مقایسه

---

**تاریخ**: 2025-11-02  
**وضعیت**: در حال توسعه  
**Priority**: High



