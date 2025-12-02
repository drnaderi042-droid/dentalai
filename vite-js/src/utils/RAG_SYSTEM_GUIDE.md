# راهنمای ساخت سیستم RAG برای آنالیز سفالومتری

## مقدمه

این راهنمای جامع برای ساخت یک سیستم RAG (Retrieval-Augmented Generation) است که می‌تواند:
- پرونده‌های بیمار را بررسی کند
- اعداد آنالیز سفالومتری را بخواند
- بر اساس داده‌ها تصمیم‌گیری و راهنمایی ارائه دهد

## ✅ فایل‌های ایجاد شده

سیستم RAG به صورت کامل پیاده‌سازی شده است:

1. **`vite-js/src/utils/rag/cephalometric-rag-service.ts`**
   - سرویس اصلی RAG (بدون نیاز به API)
   - آماده استفاده و یکپارچه با سیستم موجود

2. **`vite-js/src/utils/rag/advanced-rag-service.ts`**
   - سرویس پیشرفته با Embedding و LLM
   - نیاز به نصب کتابخانه‌های LangChain

3. **`vite-js/src/utils/rag/rag-example.ts`**
   - مثال‌های کامل استفاده

4. **`vite-js/src/utils/rag/README.md`**
   - مستندات سریع

## 🚀 شروع سریع

```typescript
import { CephalometricRAGService } from 'src/utils/rag/cephalometric-rag-service';

const ragService = new CephalometricRAGService();
const response = await ragService.analyzePatient(patientRecord, question);
```

## معماری سیستم RAG

```
┌─────────────────┐
│  Patient Record │
│  + Cephalometric│
│    Measurements │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Processor │  ← آماده‌سازی و ساختاری‌سازی داده‌ها
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Embeddings    │  ← تبدیل به بردارهای عددی
│   Generator     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Vector Store   │  ← ذخیره‌سازی در پایگاه داده برداری
│  (Chroma/FAISS) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Retrieval     │  ← بازیابی اطلاعات مرتبط
│    Module       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   LLM (GPT-4)   │  ← تولید پاسخ بر اساس داده‌های بازیابی شده
│   + Context     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Diagnosis &    │
│  Recommendations│
└─────────────────┘
```

## مراحل پیاده‌سازی

### مرحله 1: آماده‌سازی داده‌ها

#### 1.1 ساختار داده‌های سفالومتری

داده‌های شما از قبل در `orthodontic-analysis.ts` تعریف شده‌اند:

```typescript
interface CephalometricMeasurements {
  SNA?: number;
  SNB?: number;
  ANB?: number;
  FMA?: number;
  FMIA?: number;
  IMPA?: number;
  'U1-SN'?: number;
  'L1-MP'?: number;
  GoGnSN?: number;
}
```

#### 1.2 ساختار پرونده بیمار

برای هر بیمار باید اطلاعات زیر را جمع‌آوری کنید:

```typescript
interface PatientRecord {
  patientId: string;
  age: number;
  gender: 'male' | 'female';
  cephalometricMeasurements: CephalometricMeasurements;
  medicalHistory?: string;
  previousTreatments?: string[];
  images?: string[];
  analysisHistory?: AnalysisRecord[];
}
```

### مرحله 2: انتخاب ابزارها و کتابخانه‌ها

#### 2.1 کتابخانه‌های پیشنهادی

**برای RAG:**
- **LangChain**: فریمورک اصلی برای ساخت RAG
- **Chroma** یا **FAISS**: برای ذخیره‌سازی برداری
- **OpenAI Embeddings**: برای تولید embedding
- **OpenAI GPT-4**: برای تولید پاسخ

**برای پردازش داده:**
- **Pandas**: برای پردازش داده‌های ساختاریافته
- **NumPy**: برای محاسبات عددی

#### 2.2 نصب وابستگی‌ها

```bash
npm install langchain @langchain/openai chromadb
# یا
pip install langchain openai chromadb pandas numpy
```

### مرحله 3: ساخت پایگاه دانش (Knowledge Base)

#### 3.1 منابع دانش

برای سیستم RAG نیاز به منابع دانش دارید:

1. **راهنماهای بالینی ارتودنسی**
   - محدوده‌های نرمال پارامترهای سفالومتری
   - پروتکل‌های درمانی
   - مقالات علمی مرتبط

2. **داده‌های تاریخی**
   - پرونده‌های قبلی بیماران
   - نتایج درمان‌های موفق
   - الگوهای تشخیصی

3. **دانش تخصصی**
   - تعاریف پارامترهای سفالومتری
   - روابط بین پارامترها
   - استثناها و موارد خاص

#### 3.2 ساختار پایگاه دانش

```typescript
interface KnowledgeBase {
  clinicalGuidelines: ClinicalGuideline[];
  caseStudies: CaseStudy[];
  parameterDefinitions: ParameterDefinition[];
  treatmentProtocols: TreatmentProtocol[];
}
```

### مرحله 4: پیاده‌سازی ماژول بازیابی (Retrieval)

#### 4.1 تولید Embedding

```typescript
import { OpenAIEmbeddings } from '@langchain/openai';

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
  modelName: 'text-embedding-3-small', // یا 'text-embedding-3-large'
});
```

#### 4.2 ذخیره‌سازی برداری

```typescript
import { Chroma } from '@langchain/community/vectorstores/chroma';

// ایجاد vector store
const vectorStore = await Chroma.fromDocuments(
  documents,
  embeddings,
  {
    collectionName: 'cephalometric-knowledge',
  }
);
```

#### 4.3 بازیابی اطلاعات مرتبط

```typescript
// بازیابی با similarity search
const relevantDocs = await vectorStore.similaritySearch(
  query,
  k=5 // تعداد اسناد مرتبط
);

// بازیابی با MMR (Maximum Marginal Relevance)
const diverseDocs = await vectorStore.maxMarginalRelevanceSearch(
  query,
  { k: 5, fetchK: 20 }
);
```

### مرحله 5: پیاده‌سازی ماژول تولید (Generation)

#### 5.1 ساخت Prompt Template

```typescript
import { ChatPromptTemplate } from '@langchain/core/prompts';

const prompt = ChatPromptTemplate.fromMessages([
  ['system', `شما یک متخصص ارتودنسی با تجربه هستید.
  بر اساس داده‌های سفالومتری و پرونده بیمار، 
  تشخیص و توصیه‌های درمانی ارائه دهید.`],
  ['human', `پرونده بیمار:
  سن: {age}
  جنسیت: {gender}
  
  اندازه‌گیری‌های سفالومتری:
  {measurements}
  
  سوال: {question}`],
]);
```

#### 5.2 اتصال به LLM

```typescript
import { ChatOpenAI } from '@langchain/openai';

const llm = new ChatOpenAI({
  modelName: 'gpt-4-turbo-preview',
  temperature: 0.3, // برای دقت بیشتر
  openAIApiKey: process.env.OPENAI_API_KEY,
});
```

#### 5.3 ساخت Chain

```typescript
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';

// Chain برای ترکیب اسناد
const combineDocsChain = await createStuffDocumentsChain({
  llm,
  prompt,
});

// Chain نهایی RAG
const ragChain = await createRetrievalChain({
  combineDocsChain,
  retriever: vectorStore.asRetriever(),
});
```

### مرحله 6: یکپارچه‌سازی با سیستم موجود

#### 6.1 استفاده از توابع موجود

سیستم شما از قبل توابع مفیدی دارد:

```typescript
// از orthodontic-analysis.ts
import {
  analyzeCephalometricMeasurements,
  generateComprehensiveAnalysis,
  generateTreatmentPlan,
} from 'src/utils/orthodontic-analysis';
```

#### 6.2 ساخت RAG Service

```typescript
class CephalometricRAGService {
  private vectorStore: VectorStore;
  private llm: ChatOpenAI;
  private retriever: Retriever;

  async initialize() {
    // راه‌اندازی vector store و LLM
  }

  async analyzePatient(patientRecord: PatientRecord, question: string) {
    // 1. تحلیل اولیه با توابع موجود
    const analysis = generateComprehensiveAnalysis(
      patientRecord.cephalometricMeasurements,
      patientRecord.facialLandmarks,
      patientRecord.age
    );

    // 2. بازیابی اطلاعات مرتبط از پایگاه دانش
    const context = await this.retrieveRelevantContext(
      patientRecord,
      analysis
    );

    // 3. تولید پاسخ با LLM
    const response = await this.generateResponse(
      patientRecord,
      analysis,
      context,
      question
    );

    return response;
  }
}
```

## سیستم‌های RAG موجود در حوزه پزشکی

### 1. Medical Graph RAG
- **مقاله**: "Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation"
- **ویژگی**: استفاده از گراف‌های دانش برای بهبود دقت
- **لینک**: https://arxiv.org/abs/2408.04187

### 2. M-Eval Framework
- **مقاله**: "M-Eval: A Heterogeneity-Based Framework for Multi-evidence Validation in Medical RAG Systems"
- **ویژگی**: اعتبارسنجی چندگانه برای سیستم‌های RAG پزشکی
- **لینک**: https://arxiv.org/abs/2510.23995

### 3. سیستم‌های عمومی RAG
- **LangChain**: فریمورک جامع برای ساخت RAG
- **LlamaIndex**: فریمورک تخصصی برای RAG
- **Haystack**: فریمورک برای جستجوی معنایی

## نکات مهم برای پیاده‌سازی

### 1. امنیت و حریم خصوصی
- داده‌های بیماران باید رمزگذاری شوند
- از API keys محافظت کنید
- مطابق با قوانین HIPAA/GDPR عمل کنید

### 2. دقت و اعتبار
- همیشه نتایج را با متخصصان ارتودنسی بررسی کنید
- از چند منبع برای اعتبارسنجی استفاده کنید
- محدودیت‌های سیستم را به کاربران اطلاع دهید

### 3. بهینه‌سازی عملکرد
- از caching برای کاهش هزینه API استفاده کنید
- Embedding‌ها را از قبل محاسبه کنید
- از batch processing برای پردازش چندین بیمار استفاده کنید

### 4. ارزیابی سیستم
- دقت تشخیص را با داده‌های تست ارزیابی کنید
- زمان پاسخ را اندازه‌گیری کنید
- رضایت کاربران را جمع‌آوری کنید

## مثال استفاده

```typescript
// ایجاد سرویس RAG
const ragService = new CephalometricRAGService();
await ragService.initialize();

// تحلیل بیمار
const patientRecord = {
  patientId: '123',
  age: 14,
  gender: 'male',
  cephalometricMeasurements: {
    SNA: 85,
    SNB: 78,
    ANB: 7,
    FMA: 32,
  },
};

const question = 'این بیمار چه نوع ناهنجاری دارد و چه درمانی پیشنهاد می‌کنید؟';

const response = await ragService.analyzePatient(patientRecord, question);
console.log(response);
```

## منابع بیشتر

1. **مستندات LangChain**: https://js.langchain.com/docs/
2. **مستندات Chroma**: https://docs.trychroma.com/
3. **OpenAI Embeddings**: https://platform.openai.com/docs/guides/embeddings
4. **مقالات RAG پزشکی**: جستجو در arXiv با کلیدواژه "medical RAG"

## نتیجه‌گیری

ساخت یک سیستم RAG برای آنالیز سفالومتری نیاز به:
- آماده‌سازی دقیق داده‌ها
- ساخت پایگاه دانش جامع
- پیاده‌سازی صحیح ماژول‌های بازیابی و تولید
- یکپارچه‌سازی با سیستم موجود
- ارزیابی و بهبود مستمر

با دنبال کردن این راهنما می‌توانید یک سیستم RAG کارآمد و دقیق بسازید.

