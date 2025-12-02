# Clinical RAG Service - راهنمای استفاده

## 🎯 چیست؟

یک سیستم RAG کامل و آماده که:
- ✅ سن، جنس و اطلاعات سفالومتری بیمار را دریافت می‌کند
- ✅ از **PDF‌های واقعی** (کتاب‌ها و مقالات) می‌خواند
- ✅ طرح درمان مبتنی بر شواهد ارائه می‌دهد
- ✅ توضیحات کامل با رفرنس‌های واقعی (شامل صفحه و فصل) می‌دهد

## 🚀 شروع سریع

### 1. استفاده ساده

```typescript
import { RealClinicalRAGService } from 'src/utils/rag/real-rag-service';
import { PatientData } from 'src/utils/rag/rag-types';

const service = new RealClinicalRAGService();

// راه‌اندازی (فقط یک بار)
await service.initialize('./knowledge-base/books', {
  useEmbeddings: false, // بدون Embeddings (رایگان و سریع)
});

const patientData: PatientData = {
  age: 14,
  gender: 'male',
  cephalometricMeasurements: {
    SNA: 85,
    SNB: 78,
    ANB: 7,  // کلاس II
    FMA: 30,
  },
};

const analysis = await service.analyzePatient(patientData);

console.log(analysis.diagnosis);        // "کلاس II اسکلتی با الگوی رشد عمودی"
console.log(analysis.treatmentPlan);    // طرح درمان کامل
console.log(analysis.references);       // رفرنس‌های علمی
```

### 2. استفاده در React

```jsx
import { ClinicalRAGAnalysis } from 'src/sections/orthodontics/patient/components/clinical-rag-analysis';

function PatientView() {
  const patientData = {
    age: 14,
    gender: 'male',
    cephalometricMeasurements: {
      SNA: 85,
      SNB: 78,
      ANB: 7,
    },
  };

  return (
    <ClinicalRAGAnalysis 
      patientData={patientData}
      onAnalysisComplete={(analysis) => {
        console.log('تحلیل کامل شد:', analysis);
      }}
    />
  );
}
```

## 📊 خروجی سیستم

سیستم یک تحلیل کامل برمی‌گرداند شامل:

```typescript
{
  diagnosis: string;              // تشخیص بالینی
  severity: 'mild' | 'moderate' | 'severe';
  issues: [                        // مشکلات شناسایی شده
    {
      parameter: 'ANB',
      value: 7,
      normalRange: { min: 2, max: 4 },
      deviation: 3,
      description: 'کلاس II اسکلتی',
      clinicalSignificance: 'نیاز به اصلاح رابطه اسکلتی...'
    }
  ],
  treatmentPlan: [                 // طرح درمان
    {
      phase: 'درمان اسکلتی - کلاس II',
      duration: '12-18 ماه',
      procedures: ['دستگاه فانکشنال', ...],
      goals: ['اصلاح رابطه اسکلتی', ...],
      evidence: [                  // شواهد علمی
        {
          authors: 'Proffit WR',
          year: 2019,
          title: 'Class II Malocclusion...',
          journal: 'Contemporary Orthodontics'
        }
      ],
      rationale: 'بر اساس Proffit (2019)...'
    }
  ],
  recommendations: [               // توصیه‌ها
    {
      recommendation: 'استفاده از دستگاه فانکشنال',
      evidence: [...],
      priority: 'high'
    }
  ],
  prognosis: string;               // پیش‌بینی
  references: [                    // رفرنس‌های استفاده شده
    {
      id: 'ref-001',
      authors: 'Proffit WR, Fields HW',
      year: 2019,
      title: 'Class II Malocclusion...',
      journal: 'Contemporary Orthodontics'
    }
  ],
  explanation: string;             // توضیحات کامل (Markdown)
}
```

## 📚 رفرنس‌های علمی

سیستم از **PDF‌های واقعی** می‌خواند. برای استفاده:

1. PDF‌های کتاب‌ها و مقالات را در پوشه `./knowledge-base/books` قرار دهید
2. سیستم به صورت خودکار PDF‌ها را پردازش می‌کند
3. رفرنس‌ها با شماره صفحه و فصل استخراج می‌شوند

## 🔍 ویژگی‌ها

### ✅ تحلیل خودکار
- شناسایی مشکلات بر اساس محدوده‌های نرمال
- محاسبه انحراف از نرمال
- تعیین شدت مشکل

### ✅ بازیابی هوشمند رفرنس‌ها
- جستجوی خودکار رفرنس‌های مرتبط
- اولویت‌بندی بر اساس ارتباط
- انتخاب بهترین شواهد علمی

### ✅ طرح درمان مبتنی بر شواهد
- هر فاز درمان با رفرنس علمی
- توجیه علمی برای هر روش
- مدت زمان درمان بر اساس شواهد

### ✅ توضیحات کامل
- توضیحات Markdown
- شامل تمام اطلاعات بیمار
- لیست کامل رفرنس‌ها

## 📖 مثال‌های کامل

به فایل `real-rag-example.ts` مراجعه کنید برای:
- مثال استفاده پایه
- استفاده با Embeddings
- افزودن PDF جدید
- استفاده در React Component

## 🎨 کامپوننت React

کامپوننت `ClinicalRAGAnalysis` آماده استفاده است:

```jsx
<ClinicalRAGAnalysis 
  patientData={patientData}
  onAnalysisComplete={(analysis) => {
    // ذخیره در دیتابیس
    // نمایش به کاربر
    // ...
  }}
/>
```

**ویژگی‌های کامپوننت:**
- ✅ نمایش خودکار تحلیل
- ✅ UI زیبا و قابل فهم
- ✅ نمایش رفرنس‌ها
- ✅ Accordion برای طرح درمان
- ✅ Loading و Error handling

## 🔧 تنظیمات

### افزودن PDF جدید

```typescript
// بعد از initialize
await ragService.addPDF('./knowledge-base/articles/new-article.pdf');
```

### استفاده با Embeddings (دقیق‌تر)

```typescript
await ragService.initialize('./knowledge-base/books', {
  useEmbeddings: true,
  apiKey: process.env.OPENAI_API_KEY,
});
```

## ⚠️ نکات مهم

1. **همیشه بررسی کنید**: نتایج را با متخصص ارتودنسی بررسی کنید
2. **رفرنس‌ها**: رفرنس‌ها بر اساس منابع معتبر هستند اما ممکن است نیاز به به‌روزرسانی باشد
3. **سن بیمار**: سیستم به سن بیمار توجه می‌کند و درمان مناسب را پیشنهاد می‌دهد
4. **دقت**: سیستم برای راهنمایی است، نه جایگزین تشخیص پزشک

## 📝 ساختار فایل‌ها

```
rag/
├── real-rag-service.ts            # سرویس اصلی (واقعی)
├── rag-types.ts                   # انواع داده‌ای
├── pdf-processor.ts                # پردازش PDF
├── real-rag-example.ts            # مثال‌ها
└── ...

components/
└── clinical-rag-analysis.jsx      # کامپوننت React
```

## 🆘 پشتیبانی

برای سوالات:
1. به `real-rag-example.ts` نگاه کنید
2. به `REAL_RAG_USAGE.md` مراجعه کنید
3. به کامپوننت `ClinicalRAGAnalysis` نگاه کنید

---

**آماده استفاده است! 🚀**

