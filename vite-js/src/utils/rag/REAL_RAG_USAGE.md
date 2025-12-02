# راهنمای استفاده از Real Clinical RAG Service

## نصب وابستگی‌ها

```bash
# برای پردازش PDF
npm install pdf-parse @types/pdf-parse

# اختیاری: برای Embeddings (نیاز به API key)
npm install @langchain/openai @langchain/community chromadb
```

## ساختار پوشه‌ها

```
project-root/
├── knowledge-base/
│   ├── books/
│   │   ├── contemporary-orthodontics.pdf
│   │   ├── textbook-of-orthodontics.pdf
│   │   └── ...
│   └── articles/
│       └── ...
└── vite-js/
    └── src/
        └── utils/
            └── rag/
                ├── real-rag-service.ts
                └── ...
```

## استفاده پایه (بدون Embeddings)

```typescript
import { RealClinicalRAGService } from 'src/utils/rag/real-rag-service';

// 1. ساخت سرویس
const ragService = new RealClinicalRAGService();

// 2. راه‌اندازی (فقط یک بار - زمان‌بر است)
await ragService.initialize('./knowledge-base/books', {
  useEmbeddings: false, // بدون Embeddings = رایگان و سریع
});

// 3. استفاده (سریع)
const patient = {
  age: 14,
  gender: 'male',
  cephalometricMeasurements: {
    SNA: 85,
    SNB: 78,
    ANB: 7,
  },
};

const analysis = await ragService.analyzePatient(patient);

// نتیجه شامل رفرنس‌های واقعی از PDF‌ها است
console.log(analysis.references); // با صفحه و فصل!
```

## استفاده با Embeddings (دقیق‌تر)

```typescript
await ragService.initialize('./knowledge-base/books', {
  useEmbeddings: true,
  apiKey: process.env.OPENAI_API_KEY,
  vectorStoreConfig: {
    collectionName: 'clinical-orthodontics',
  },
});
```

## ویژگی‌ها

### ✅ بدون نیاز به LLM برای Retrieval
- جستجوی ساده متنی (رایگان)
- یا جستجوی Embedding (دقیق‌تر، نیاز به API key)

### ✅ رفرنس‌های واقعی
- از PDF‌های واقعی استخراج می‌شود
- شامل صفحه و فصل
- قابل بررسی

### ✅ پردازش یک‌باره
- PDF‌ها فقط یک بار پردازش می‌شوند
- بعد از initialize، استفاده سریع است

## نکات مهم

1. **مسیر PDF‌ها**: مسیر را نسبت به root پروژه مشخص کنید
2. **کیفیت PDF**: PDF‌ها باید قابل خواندن باشند (نه اسکن شده)
3. **حق نشر**: مطمئن شوید حق استفاده از PDF‌ها را دارید
4. **ذخیره‌سازی**: Vector Store در Chroma ذخیره می‌شود

## مثال کامل

```typescript
// در یک فایل جداگانه (مثلاً scripts/init-rag.ts)
import { RealClinicalRAGService } from '../src/utils/rag/real-rag-service';

async function initRAG() {
  const ragService = new RealClinicalRAGService();
  
  await ragService.initialize('./knowledge-base/books', {
    useEmbeddings: false, // یا true با API key
  });
  
  console.log('✅ RAG initialized');
  console.log('Stats:', ragService.getStats());
}

initRAG();
```

## یکپارچه‌سازی با سیستم موجود

می‌توانید RealClinicalRAGService را جایگزین ClinicalRAGService کنید:

```typescript
// در cephalometric-analysis-view.jsx
import { RealClinicalRAGService } from 'src/utils/rag/real-rag-service';

// راه‌اندازی (یک بار)
const realRAG = new RealClinicalRAGService();
await realRAG.initialize('./knowledge-base/books');

// استفاده
const analysis = await realRAG.analyzePatient(patientData);
```

## تفاوت با ClinicalRAGService

| ویژگی | ClinicalRAGService | RealClinicalRAGService |
|-------|---------------------|------------------------|
| منبع داده | Mock Data | PDF‌های واقعی |
| رفرنس‌ها | Mock | واقعی از PDF |
| صفحه و فصل | Mock | واقعی |
| نیاز به PDF | ❌ | ✅ |
| هزینه | رایگان | رایگان (بدون Embeddings) |

## نتیجه

RealClinicalRAGService یک RAG واقعی است که:
- ✅ از PDF‌های واقعی می‌خواند
- ✅ رفرنس‌های دقیق با صفحه و فصل می‌دهد
- ✅ بدون نیاز به LLM برای Retrieval
- ✅ قابل اعتماد برای سیستم پزشکی





