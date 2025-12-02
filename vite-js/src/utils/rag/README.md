# سیستم RAG برای آنالیز سفالومتری

این پوشه شامل پیاده‌سازی سیستم RAG (Retrieval-Augmented Generation) برای تحلیل پرونده‌های بیمار و آنالیز سفالومتری است.

## ساختار فایل‌ها

```
rag/
├── README.md                      # این فایل
├── cephalometric-rag-service.ts   # سرویس اصلی RAG (بدون نیاز به API)
├── advanced-rag-service.ts        # سرویس پیشرفته با Embedding (نیاز به API)
├── rag-example.ts                 # مثال‌های استفاده
└── ...
```

## نصب وابستگی‌ها

### برای استفاده پایه (بدون API):

```bash
# هیچ وابستگی اضافی نیاز نیست
# فقط از cephalometric-rag-service.ts استفاده کنید
```

### برای استفاده پیشرفته (با Embedding و LLM):

```bash
npm install @langchain/openai @langchain/community chromadb
# یا
npm install openai
```

## استفاده سریع

### مثال 1: استفاده ساده

```typescript
import { CephalometricRAGService, PatientRecord } from 'src/utils/rag/cephalometric-rag-service';

const ragService = new CephalometricRAGService();

const patient: PatientRecord = {
  patientId: 'P001',
  age: 14,
  gender: 'male',
  cephalometricMeasurements: {
    SNA: 85,
    SNB: 78,
    ANB: 7, // کلاس II
    FMA: 30,
  },
};

const response = await ragService.analyzePatient(
  patient,
  'چه درمانی پیشنهاد می‌کنید؟'
);

console.log(response.diagnosis);
console.log(response.recommendations);
console.log(response.treatmentPlan);
```

### مثال 2: جستجوی موارد مشابه

```typescript
const similarCases = ragService.findSimilarCases(patient, caseDatabase);
```

### مثال 3: استفاده پیشرفته (با Embedding)

```typescript
import { AdvancedCephalometricRAGService } from 'src/utils/rag/advanced-rag-service';

const service = new AdvancedCephalometricRAGService({
  provider: 'openai',
  apiKey: process.env.OPENAI_API_KEY,
  vectorStoreType: 'chroma',
});

await service.initialize();
const response = await service.analyzePatient(patient, question);
```

## ویژگی‌ها

### ✅ CephalometricRAGService (پایه)

- ✅ بدون نیاز به API یا کتابخانه خارجی
- ✅ تحلیل خودکار بر اساس پارامترهای سفالومتری
- ✅ بازیابی راهنماهای بالینی مرتبط
- ✅ تولید توصیه‌های درمانی
- ✅ جستجوی موارد مشابه
- ✅ توضیحات پارامترها

### 🚀 AdvancedCephalometricRAGService (پیشرفته)

- 🚀 استفاده از Vector Embeddings
- 🚀 جستجوی معنایی پیشرفته
- 🚀 یکپارچه‌سازی با LLM (GPT-4)
- 🚀 ذخیره‌سازی برداری (Chroma/FAISS)
- 🚀 بازیابی دقیق‌تر اطلاعات

## API Reference

### CephalometricRAGService

#### `analyzePatient(patientRecord, question?)`

تحلیل کامل بیمار و تولید پاسخ.

**پارامترها:**
- `patientRecord: PatientRecord` - اطلاعات بیمار
- `question?: string` - سوال اختیاری

**بازگشت:**
```typescript
{
  diagnosis: string;
  recommendations: string[];
  treatmentPlan: string[];
  explanation: string;
  confidence: 'high' | 'medium' | 'low';
  sources: string[];
}
```

#### `findSimilarCases(patientRecord, caseDatabase)`

جستجوی موارد مشابه در پایگاه داده.

**پارامترها:**
- `patientRecord: PatientRecord` - بیمار فعلی
- `caseDatabase: PatientRecord[]` - پایگاه داده موارد

**بازگشت:** `PatientRecord[]` - لیست موارد مشابه

### Helper Functions

#### `patientRecordToText(patientRecord)`

تبدیل پرونده بیمار به متن.

#### `buildPrompt(patientRecord, analysis, context, question)`

ساخت prompt برای LLM.

## یکپارچه‌سازی با سیستم موجود

سیستم RAG از توابع موجود در `orthodontic-analysis.ts` استفاده می‌کند:

```typescript
import {
  generateComprehensiveAnalysis,
  analyzeCephalometricMeasurements,
} from 'src/utils/orthodontic-analysis';
```

## مثال‌های کامل

برای مثال‌های کامل، به فایل `rag-example.ts` مراجعه کنید.

## نکات مهم

1. **امنیت**: داده‌های بیماران باید رمزگذاری شوند
2. **دقت**: همیشه نتایج را با متخصصان بررسی کنید
3. **API Keys**: برای استفاده از Advanced Service، API key نیاز است
4. **هزینه**: استفاده از LLM هزینه‌بر است، از caching استفاده کنید

## منابع

- راهنمای کامل: `RAG_SYSTEM_GUIDE.md`
- مستندات LangChain: https://js.langchain.com/
- مستندات OpenAI: https://platform.openai.com/docs

## پشتیبانی

برای سوالات و مشکلات، به مستندات اصلی مراجعه کنید یا با تیم توسعه تماس بگیرید.





