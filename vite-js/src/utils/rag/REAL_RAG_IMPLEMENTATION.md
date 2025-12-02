# راهنمای پیاده‌سازی RAG واقعی با خواندن PDF

## وضعیت فعلی

**⚠️ هشدار:** RAG فعلی از Mock Data استفاده می‌کند. رفرنس‌ها واقعی هستند اما محتوا از PDF خوانده نشده است.

## نیازمندی‌ها برای RAG واقعی

### 1. کتابخانه‌های مورد نیاز

```bash
npm install pdf-parse pdfjs-dist
npm install @langchain/openai @langchain/community
npm install chromadb
npm install mammoth  # برای فایل‌های Word
```

### 2. ساختار پیشنهادی

```
rag/
├── real-rag-service.ts          # سرویس RAG واقعی
├── pdf-processor.ts              # پردازش PDF
├── knowledge-base/               # پوشه PDF‌ها و مقالات
│   ├── books/                    # کتاب‌های PDF
│   │   ├── contemporary-orthodontics.pdf
│   │   ├── textbook-of-orthodontics.pdf
│   │   └── ...
│   └── articles/                 # مقالات PDF
│       ├── class-ii-treatment.pdf
│       └── ...
├── vector-store/                 # ذخیره‌سازی برداری
└── references.json               # فهرست رفرنس‌های واقعی
```

## مرحله 1: پردازش PDF

### فایل: `pdf-processor.ts`

```typescript
import fs from 'fs';
import path from 'path';
import pdf from 'pdf-parse';

export interface PDFDocument {
  title: string;
  authors: string;
  year: number;
  pages: PDFPage[];
  metadata: {
    totalPages: number;
    filePath: string;
  };
}

export interface PDFPage {
  pageNumber: number;
  content: string;
  chapter?: string;
  section?: string;
}

/**
 * خواندن PDF و استخراج محتوا
 */
export async function parsePDF(filePath: string): Promise<PDFDocument> {
  const dataBuffer = fs.readFileSync(filePath);
  const pdfData = await pdf(dataBuffer);
  
  // استخراج metadata از نام فایل یا محتوای PDF
  const metadata = extractMetadata(filePath, pdfData);
  
  // تقسیم به صفحات
  const pages: PDFPage[] = [];
  const contentPerPage = pdfData.text.split(/\f/); // تقسیم بر اساس page break
  
  contentPerPage.forEach((content, index) => {
    pages.push({
      pageNumber: index + 1,
      content: content.trim(),
      chapter: extractChapter(content),
      section: extractSection(content),
    });
  });
  
  return {
    title: metadata.title,
    authors: metadata.authors,
    year: metadata.year,
    pages,
    metadata: {
      totalPages: pdfData.numpages,
      filePath,
    },
  };
}

/**
 * استخراج metadata از PDF
 */
function extractMetadata(filePath: string, pdfData: any) {
  const fileName = path.basename(filePath, '.pdf');
  
  // استخراج از info (اگر موجود باشد)
  const info = pdfData.info || {};
  
  return {
    title: info.Title || fileName,
    authors: info.Author || 'Unknown',
    year: extractYear(info.CreationDate) || extractYear(fileName) || new Date().getFullYear(),
  };
}

/**
 * استخراج سال از متن
 */
function extractYear(text: string): number | null {
  const match = text.match(/\b(19|20)\d{2}\b/);
  return match ? parseInt(match[0], 10) : null;
}

/**
 * استخراج فصل از محتوا
 */
function extractChapter(content: string): string | undefined {
  // جستجوی الگوهای فصل
  const patterns = [
    /Chapter\s+(\d+)/i,
    /فصل\s+(\d+)/i,
    /Chapter\s+([IVX]+)/i,
  ];
  
  for (const pattern of patterns) {
    const match = content.match(pattern);
    if (match) {
      return `Chapter ${match[1]}`;
    }
  }
  
  return undefined;
}

/**
 * استخراج بخش از محتوا
 */
function extractSection(content: string): string | undefined {
  // جستجوی الگوهای بخش
  const patterns = [
    /Section\s+(\d+\.\d+)/i,
    /بخش\s+(\d+)/i,
  ];
  
  for (const pattern of patterns) {
    const match = content.match(pattern);
    if (match) {
      return match[1];
    }
  }
  
  return undefined;
}

/**
 * پردازش همه PDF‌ها در یک پوشه
 */
export async function processAllPDFs(directory: string): Promise<PDFDocument[]> {
  const files = fs.readdirSync(directory);
  const pdfFiles = files.filter(f => f.endsWith('.pdf'));
  
  const documents: PDFDocument[] = [];
  
  for (const file of pdfFiles) {
    const filePath = path.join(directory, file);
    try {
      const doc = await parsePDF(filePath);
      documents.push(doc);
      console.log(`✅ Processed: ${file}`);
    } catch (error) {
      console.error(`❌ Error processing ${file}:`, error);
    }
  }
  
  return documents;
}
```

## مرحله 2: ساخت Vector Store واقعی

### فایل: `real-rag-service.ts`

```typescript
import { Chroma } from '@langchain/community/vectorstores/chroma';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
import { PDFDocument, parsePDF, processAllPDFs } from './pdf-processor';
import { ChatOpenAI } from '@langchain/openai';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';

export class RealClinicalRAGService {
  private vectorStore: Chroma | null = null;
  private embeddings: OpenAIEmbeddings;
  private llm: ChatOpenAI;
  private documents: Document[] = [];

  constructor(apiKey: string) {
    this.embeddings = new OpenAIEmbeddings({
      openAIApiKey: apiKey,
      modelName: 'text-embedding-3-large', // برای دقت بیشتر
    });
    
    this.llm = new ChatOpenAI({
      modelName: 'gpt-4-turbo-preview',
      temperature: 0.3,
      openAIApiKey: apiKey,
    });
  }

  /**
   * راه‌اندازی: خواندن PDF‌ها و ساخت Vector Store
   */
  async initialize(pdfDirectory: string) {
    console.log('📚 Reading PDFs from:', pdfDirectory);
    
    // خواندن همه PDF‌ها
    const pdfDocuments = await processAllPDFs(pdfDirectory);
    
    // تبدیل به Document format برای LangChain
    this.documents = [];
    
    pdfDocuments.forEach((pdfDoc) => {
      pdfDoc.pages.forEach((page) => {
        // ساخت metadata شامل اطلاعات کامل
        const metadata = {
          source: pdfDoc.title,
          authors: pdfDoc.authors,
          year: pdfDoc.year,
          page: page.pageNumber,
          chapter: page.chapter,
          section: page.section,
          filePath: pdfDoc.metadata.filePath,
        };
        
        // تقسیم محتوا به chunks (هر chunk حدود 1000 کلمه)
        const chunks = this.splitIntoChunks(page.content, 1000);
        
        chunks.forEach((chunk, chunkIndex) => {
          this.documents.push(
            new Document({
              pageContent: chunk,
              metadata: {
                ...metadata,
                chunkIndex,
              },
            })
          );
        });
      });
    });
    
    console.log(`✅ Processed ${this.documents.length} document chunks from ${pdfDocuments.length} PDFs`);
    
    // ساخت Vector Store
    this.vectorStore = await Chroma.fromDocuments(
      this.documents,
      this.embeddings,
      {
        collectionName: 'clinical-orthodontics-knowledge',
      }
    );
    
    console.log('✅ Vector Store created successfully');
  }

  /**
   * تقسیم متن به chunks
   */
  private splitIntoChunks(text: string, maxWords: number): string[] {
    const words = text.split(/\s+/);
    const chunks: string[] = [];
    
    for (let i = 0; i < words.length; i += maxWords) {
      chunks.push(words.slice(i, i + maxWords).join(' '));
    }
    
    return chunks;
  }

  /**
   * تحلیل بیمار با استفاده از RAG واقعی
   */
  async analyzePatient(patientData: PatientData): Promise<ClinicalAnalysis> {
    if (!this.vectorStore) {
      throw new Error('Vector Store not initialized. Call initialize() first.');
    }

    // ساخت query از اطلاعات بیمار
    const query = this.buildQuery(patientData);

    // بازیابی اسناد مرتبط
    const retriever = this.vectorStore.asRetriever({
      k: 10, // 10 سند مرتبط
      searchType: 'mmr', // Maximum Marginal Relevance برای تنوع بیشتر
    });

    const relevantDocs = await retriever.getRelevantDocuments(query);

    // ساخت context
    const context = relevantDocs.map(doc => ({
      content: doc.pageContent,
      source: doc.metadata.source,
      page: doc.metadata.page,
      chapter: doc.metadata.chapter,
      authors: doc.metadata.authors,
      year: doc.metadata.year,
    }));

    // ساخت prompt
    const prompt = this.buildPrompt(patientData, context);

    // تولید پاسخ با LLM
    const response = await this.llm.invoke(prompt);

    // پارس کردن پاسخ
    return this.parseResponse(response.content, context);
  }

  /**
   * ساخت query از اطلاعات بیمار
   */
  private buildQuery(patientData: PatientData): string {
    let query = `Patient analysis: Age ${patientData.age}, Gender ${patientData.gender}. `;
    query += `Cephalometric measurements: `;
    
    Object.entries(patientData.cephalometricMeasurements).forEach(([param, value]) => {
      query += `${param}: ${value}°, `;
    });
    
    query += `What is the diagnosis and treatment plan?`;
    
    return query;
  }

  /**
   * ساخت prompt برای LLM
   */
  private buildPrompt(patientData: PatientData, context: any[]): string {
    let prompt = `You are an expert orthodontist. Analyze this patient based on the following clinical references:\n\n`;
    
    prompt += `Patient Information:\n`;
    prompt += `- Age: ${patientData.age} years\n`;
    prompt += `- Gender: ${patientData.gender}\n`;
    prompt += `- Cephalometric Measurements:\n`;
    
    Object.entries(patientData.cephalometricMeasurements).forEach(([param, value]) => {
      prompt += `  - ${param}: ${value}°\n`;
    });
    
    prompt += `\nClinical References:\n`;
    context.forEach((ref, index) => {
      prompt += `${index + 1}. ${ref.source} (${ref.authors}, ${ref.year})\n`;
      if (ref.chapter) prompt += `   Chapter: ${ref.chapter}\n`;
      if (ref.page) prompt += `   Page: ${ref.page}\n`;
      prompt += `   Content: ${ref.content.substring(0, 500)}...\n\n`;
    });
    
    prompt += `\nPlease provide:\n`;
    prompt += `1. Clinical diagnosis\n`;
    prompt += `2. Identified issues with explanations\n`;
    prompt += `3. Evidence-based treatment plan\n`;
    prompt += `4. References with page numbers\n`;
    
    return prompt;
  }

  /**
   * پارس کردن پاسخ LLM
   */
  private parseResponse(response: string, context: any[]): ClinicalAnalysis {
    // این تابع باید پاسخ LLM را پارس کند
    // برای سادگی، از ساختار قبلی استفاده می‌کنیم
    // در واقعیت، باید از structured output استفاده کنید
    
    return {
      diagnosis: 'Extracted from LLM response',
      severity: 'moderate',
      issues: [],
      treatmentPlan: [],
      recommendations: [],
      prognosis: '',
      references: context.map(ref => ({
        id: `ref-${ref.source}-${ref.page}`,
        title: ref.source,
        authors: ref.authors,
        year: ref.year,
        page: ref.page?.toString(),
        chapter: ref.chapter,
        content: ref.content,
        tags: [],
        category: 'treatment' as const,
        isReal: true,
      })),
      explanation: response,
    };
  }
}
```

## مرحله 3: استفاده

```typescript
// راه‌اندازی
const ragService = new RealClinicalRAGService(process.env.OPENAI_API_KEY);
await ragService.initialize('./knowledge-base/books');

// استفاده
const analysis = await ragService.analyzePatient({
  age: 14,
  gender: 'male',
  cephalometricMeasurements: {
    SNA: 85,
    SNB: 78,
    ANB: 7,
  },
});
```

## نکات مهم

1. **حق نشر:** مطمئن شوید که حق استفاده از PDF‌ها را دارید
2. **کیفیت PDF:** PDF‌ها باید قابل خواندن باشند (نه اسکن شده)
3. **هزینه:** استفاده از OpenAI API هزینه دارد
4. **ذخیره‌سازی:** Vector Store را ذخیره کنید تا نیازی به پردازش مجدد نباشد

## منابع PDF پیشنهادی

1. **Contemporary Orthodontics** - Proffit (PDF)
2. **Textbook of Orthodontics** - Bishara (PDF)
3. **American Journal of Orthodontics** - مقالات
4. **Seminars in Orthodontics** - مقالات

## مرحله بعدی

برای پیاده‌سازی کامل، باید:
1. PDF‌های واقعی را تهیه کنید
2. کد بالا را پیاده‌سازی کنید
3. تست کنید
4. بهینه‌سازی کنید





