/**
 * Real Clinical RAG Service
 * سرویس RAG واقعی که از PDF‌های واقعی می‌خواند
 * 
 * این سرویس:
 * - PDF‌های واقعی را پردازش می‌کند
 * - Vector Store می‌سازد
 * - اطلاعات واقعی را بازیابی می‌کند
 * - رفرنس‌های دقیق با صفحه و فصل ارائه می‌دهد
 */

import { generateComprehensiveAnalysis } from '../orthodontic-analysis.ts';
import { parsePDF, PDFDocument, processAllPDFs } from './pdf-processor.ts';
import {
  PatientData,
  TreatmentPlan,
  ClinicalAnalysis,
  ClinicalReference,
} from './rag-types.ts';

// ============================================================================
// Types
// ============================================================================

interface DocumentChunk {
  content: string;
  metadata: {
    source: string;
    authors: string;
    year: number;
    page: number;
    chapter?: string;
    section?: string;
    filePath: string;
    chunkIndex: number;
  };
}

interface VectorStoreConfig {
  collectionName?: string;
  persistDirectory?: string;
}

// ============================================================================
// Real Clinical RAG Service
// ============================================================================

export class RealClinicalRAGService {
  private documents: DocumentChunk[] = [];

  private pdfDocuments: PDFDocument[] = [];

  private isInitialized: boolean = false;

  private vectorStore: any = null; // Chroma vector store

  private embeddings: any = null; // Embeddings

  /**
   * راه‌اندازی: خواندن PDF‌ها و ساخت Vector Store
   * 
   * @param pdfDirectory مسیر پوشه PDF‌ها
   * @param useEmbeddings آیا از Embeddings استفاده کنیم (نیاز به API key)
   */
  async initialize(
    pdfDirectory: string,
    options?: {
      useEmbeddings?: boolean;
      apiKey?: string;
      vectorStoreConfig?: VectorStoreConfig;
    }
  ): Promise<void> {
    console.log('📚 [Real RAG] Reading PDFs from:', pdfDirectory);

    try {
      // خواندن همه PDF‌ها
      this.pdfDocuments = await processAllPDFs(pdfDirectory);

      if (this.pdfDocuments.length === 0) {
        console.warn('⚠️ [Real RAG] No PDFs found in directory:', pdfDirectory);
        this.isInitialized = true; // Mark as initialized even if no PDFs
        return;
      }

      console.log(`✅ [Real RAG] Found ${this.pdfDocuments.length} PDFs`);

      // تبدیل به DocumentChunk
      this.documents = [];
      this.pdfDocuments.forEach((pdfDoc) => {
        pdfDoc.pages.forEach((page) => {
          // تقسیم به chunks (هر chunk حدود 500 کلمه)
          const chunks = RealClinicalRAGService.splitIntoChunks(page.content, 500);

          chunks.forEach((chunk, chunkIndex) => {
            this.documents.push({
              content: chunk,
              metadata: {
                source: pdfDoc.title,
                authors: pdfDoc.authors,
                year: pdfDoc.year,
                page: page.pageNumber,
                chapter: page.chapter,
                section: page.section,
                filePath: pdfDoc.metadata.filePath,
                chunkIndex,
              },
            });
          });
        });
      });

      console.log(`✅ [Real RAG] Processed ${this.documents.length} document chunks`);

      // اگر useEmbeddings فعال باشد، Vector Store بساز
      if (options?.useEmbeddings && options?.apiKey) {
        await this.buildVectorStore(options.apiKey, options.vectorStoreConfig);
      } else {
        console.log('ℹ️ [Real RAG] Using simple text search (no embeddings)');
      }

      this.isInitialized = true;
      console.log('✅ [Real RAG] Initialization complete');
    } catch (error) {
      console.error('❌ [Real RAG] Error during initialization:', error);
      throw error;
    }
  }

  /**
   * ساخت Vector Store با Embeddings
   * 
   * نکته: این متد نیاز به پکیج‌های LangChain دارد که اختیاری هستند.
   * اگر پکیج‌ها نصب نشده باشند، به جستجوی ساده متنی fallback می‌کند.
   * 
   * برای نصب (فقط در صورت نیاز به Embeddings):
   * npm install @langchain/openai @langchain/community @langchain/core chromadb
   */
  private async buildVectorStore(
    apiKey: string,
    config?: VectorStoreConfig
  ): Promise<void> {
    try {
      // Dynamic import برای LangChain (اگر نصب نشده باشد خطا ندهد)
      // استفاده از Function constructor برای جلوگیری از static analysis توسط Vite
      // این روش باعث می‌شود Vite نتواند import را در زمان build تحلیل کند
      // eslint-disable-next-line no-new-func
      const dynamicImport = new Function('specifier', 'return import(specifier)');
      
      const langchainOpenaiModule = '@langchain/openai';
      const langchainChromaModule = '@langchain/community/vectorstores/chroma';
      const langchainCoreModule = '@langchain/core/documents';
      
      const { OpenAIEmbeddings } = await dynamicImport(langchainOpenaiModule);
      const { Chroma } = await dynamicImport(langchainChromaModule);
      const { Document } = await dynamicImport(langchainCoreModule);

      this.embeddings = new OpenAIEmbeddings({
        openAIApiKey: apiKey,
        modelName: 'text-embedding-3-small', // کوچکتر = ارزان‌تر
      });

      // تبدیل به LangChain Document format
      const langchainDocs = this.documents.map(
        (doc) =>
          new Document({
            pageContent: doc.content,
            metadata: doc.metadata,
          })
      );

      // ساخت Vector Store
      this.vectorStore = await Chroma.fromDocuments(langchainDocs, this.embeddings, {
        collectionName: config?.collectionName || 'clinical-orthodontics-knowledge',
      });

      console.log('✅ [Real RAG] Vector Store created successfully');
    } catch (error) {
      console.error('❌ [Real RAG] Error building vector store:', error);
      console.warn('⚠️ [Real RAG] Falling back to simple text search');
      this.vectorStore = null;
    }
  }

  /**
   * تقسیم متن به chunks
   */
  private static splitIntoChunks(text: string, maxWords: number): string[] {
    const words = text.split(/\s+/);
    const chunks: string[] = [];

    for (let i = 0; i < words.length; i += maxWords) {
      chunks.push(words.slice(i, i + maxWords).join(' '));
    }

    return chunks.length > 0 ? chunks : [text];
  }

  /**
   * تحلیل بیمار با استفاده از RAG واقعی
   */
  async analyzePatient(patientData: PatientData): Promise<ClinicalAnalysis> {
    if (!this.isInitialized) {
      throw new Error('RAG Service not initialized. Call initialize() first.');
    }

    if (this.documents.length === 0) {
      // Fallback: استفاده از تحلیل ساده
      return this.fallbackAnalysis(patientData);
    }

    // 1. ساخت query از اطلاعات بیمار
    const query = RealClinicalRAGService.buildQuery(patientData);

    // 2. بازیابی اسناد مرتبط
    const relevantDocs = await this.retrieveRelevantDocuments(query);

    // 3. استخراج اطلاعات از اسناد
    const issues = this.extractIssues(patientData, relevantDocs);
    const treatmentInfo = RealClinicalRAGService.extractTreatmentInfo(relevantDocs, patientData);
    const references = RealClinicalRAGService.buildReferences(relevantDocs);

    // 4. ساخت تحلیل
    return this.buildAnalysis(patientData, issues, treatmentInfo, references);
  }

  /**
   * ساخت query از اطلاعات بیمار
   */
  private static buildQuery(patientData: PatientData): string {
    let query = `orthodontic patient analysis `;
    query += `age ${patientData.age} years `;
    query += `gender ${patientData.gender} `;

    // اضافه کردن پارامترهای مهم
    const measurements = patientData.cephalometricMeasurements as Record<string, number>;
    const importantParams = ['SNA', 'SNB', 'ANB', 'FMA', 'U1-SN', 'IMPA'];

    importantParams.forEach((param) => {
      if (measurements[param] !== undefined) {
        query += `${param} ${measurements[param]} `;
      }
    });

    // اضافه کردن همه پارامترها
    Object.entries(measurements).forEach(([param, value]) => {
      if (!importantParams.includes(param)) {
        query += `${param} ${value} `;
      }
    });

    query += `diagnosis treatment plan`;

    return query;
  }

  /**
   * بازیابی اسناد مرتبط
   */
  private async retrieveRelevantDocuments(query: string): Promise<DocumentChunk[]> {
    // اگر Vector Store موجود باشد، از آن استفاده کن
    if (this.vectorStore) {
      try {
        const retriever = this.vectorStore.asRetriever({
          k: 15, // 15 سند مرتبط
        });

        const docs = await retriever.getRelevantDocuments(query);
        return docs.map((doc: any) => ({
          content: doc.pageContent,
          metadata: doc.metadata,
        }));
      } catch (error) {
        console.error('Error using vector store, falling back to text search:', error);
      }
    }

    // Fallback: جستجوی ساده متنی
    return this.simpleTextSearch(query);
  }

  /**
   * جستجوی ساده متنی (بدون Embeddings)
   */
  private simpleTextSearch(query: string): DocumentChunk[] {
    const queryLower = query.toLowerCase();
    const queryWords = queryLower.split(/\s+/).filter((w) => w.length > 2);

    // امتیازدهی به هر document
    const scoredDocs = this.documents.map((doc) => {
      const contentLower = doc.content.toLowerCase();
      let score = 0;

      // شمارش کلمات مشترک
      queryWords.forEach((word) => {
        const matches = (contentLower.match(new RegExp(word, 'g')) || []).length;
        score += matches;
      });

      // امتیاز بیشتر برای پارامترهای خاص
      const paramMatches = ['SNA', 'SNB', 'ANB', 'FMA', 'class II', 'class III', 'treatment'];
      paramMatches.forEach((param) => {
        if (contentLower.includes(param.toLowerCase())) {
          score += 5;
        }
      });

      return { doc, score };
    });

    // مرتب‌سازی و برگرداندن 15 تا برتر
    return scoredDocs
      .sort((a, b) => b.score - a.score)
      .slice(0, 15)
      .map((item) => item.doc);
  }

  /**
   * استخراج مشکلات از اسناد
   */
  private static extractIssues(
    patientData: PatientData,
    relevantDocs: DocumentChunk[]
  ): ClinicalAnalysis['issues'] {
    const issues: ClinicalAnalysis['issues'] = [];
    const measurements = patientData.cephalometricMeasurements as Record<string, number>;

    // استفاده از تحلیل اولیه
    const basicAnalysis = generateComprehensiveAnalysis(
      measurements as any,
      undefined
    );

    // ترکیب با اطلاعات از PDF‌ها
    Object.entries(measurements).forEach(([param, value]) => {
      if (value === undefined || value === null || isNaN(value)) return;

      // جستجوی اطلاعات در PDF‌ها
      const pdfInfo = RealClinicalRAGService.findParameterInfo(param, value, relevantDocs);

      if (pdfInfo) {
        issues.push({
          parameter: param,
          value,
          normalRange: pdfInfo.normalRange || { min: 0, max: 100 },
          deviation: pdfInfo.deviation || 0,
          description: pdfInfo.description || `${param} خارج از محدوده نرمال`,
          clinicalSignificance: pdfInfo.clinicalSignificance || 'نیاز به بررسی بالینی',
        });
      }
    });

    return issues;
  }

  /**
   * پیدا کردن اطلاعات پارامتر در PDF‌ها
   */
  private static findParameterInfo(
    param: string,
    value: number,
    relevantDocs: DocumentChunk[]
  ): {
    normalRange?: { min: number; max: number };
    deviation?: number;
    description?: string;
    clinicalSignificance?: string;
  } | null {
    // جستجو در اسناد مرتبط
    for (const doc of relevantDocs) {
      const content = doc.content.toLowerCase();
      const paramLower = param.toLowerCase();

      // اگر پارامتر در محتوا باشد
      if (content.includes(paramLower)) {
        // استخراج محدوده نرمال
        const normalRange = RealClinicalRAGService.extractNormalRange(content, param);
        const description = RealClinicalRAGService.extractDescription(content, param, value);
        const clinicalSignificance = RealClinicalRAGService.extractClinicalSignificance(content, param, value);

        if (normalRange || description) {
          const deviation = normalRange
            ? value > normalRange.max
              ? value - normalRange.max
              : value < normalRange.min
              ? normalRange.min - value
              : 0
            : 0;

          return {
            normalRange: normalRange || undefined,
            deviation,
            description,
            clinicalSignificance,
          };
        }
      }
    }

    return null;
  }

  /**
   * استخراج محدوده نرمال از متن
   */
  private static extractNormalRange(content: string, param: string): { min: number; max: number } | null {
    // الگوهای مختلف برای محدوده نرمال
    const patterns = [
      new RegExp(`${param}[^\\d]*(\\d+)[^\\d]*-[^\\d]*(\\d+)`, 'i'),
      new RegExp(`(\\d+)[^\\d]*-[^\\d]*(\\d+)[^\\d]*${param}`, 'i'),
      new RegExp(`${param}[^\\d]*(\\d+)[^\\d]*±[^\\d]*(\\d+)`, 'i'),
    ];

    for (const pattern of patterns) {
      const match = content.match(pattern);
      if (match) {
        const mean = parseFloat(match[1]);
        const sd = parseFloat(match[2]);
        if (!isNaN(mean) && !isNaN(sd)) {
          return {
            min: mean - 2 * sd,
            max: mean + 2 * sd,
          };
        }
      }
    }

    return null;
  }

  /**
   * استخراج توضیحات از متن
   */
  private static extractDescription(content: string, param: string, value: number): string | undefined {
    // جستجوی جملات مرتبط
    const sentences = content.split(/[.!?]/);
    for (const sentence of sentences) {
      if (sentence.toLowerCase().includes(param.toLowerCase())) {
        // استخراج توضیحات
        if (sentence.includes('indicates') || sentence.includes('shows') || sentence.includes('نشان')) {
          return sentence.trim();
        }
      }
    }
    return undefined;
  }

  /**
   * استخراج اهمیت بالینی از متن
   */
  private static extractClinicalSignificance(
    content: string,
    param: string,
    value: number
  ): string | undefined {
    // جستجوی جملات درمانی
    const sentences = content.split(/[.!?]/);
    for (const sentence of sentences) {
      if (
        sentence.toLowerCase().includes('treatment') ||
        sentence.toLowerCase().includes('requires') ||
        sentence.toLowerCase().includes('نیاز')
      ) {
        if (sentence.toLowerCase().includes(param.toLowerCase())) {
          return sentence.trim();
        }
      }
    }
    return undefined;
  }

  /**
   * استخراج اطلاعات درمان از اسناد
   */
  private static extractTreatmentInfo(
    relevantDocs: DocumentChunk[],
    patientData: PatientData
  ): {
    procedures: string[];
    duration: string;
    goals: string[];
  } {
    const procedures: string[] = [];
    const goals: string[] = [];
    let duration = '18-24 months';

    // جستجو در اسناد برای اطلاعات درمان
    relevantDocs.forEach((doc) => {
      const content = doc.content.toLowerCase();

      // استخراج روش‌های درمانی
      if (content.includes('treatment') || content.includes('درمان')) {
        // جستجوی دستگاه‌ها
        const appliances = [
          'twin block',
          'herbst',
          'face mask',
          'headgear',
          'miniscrew',
          'fixed appliance',
          'براکت',
          'دستگاه فانکشنال',
        ];

        appliances.forEach((appliance) => {
          if (content.includes(appliance) && !procedures.includes(appliance)) {
            procedures.push(appliance);
          }
        });

        // استخراج مدت زمان
        const durationMatch = content.match(/(\d+)[^\\d]*-[^\\d]*(\d+)[^\\d]*month/i);
        if (durationMatch) {
          duration = `${durationMatch[1]}-${durationMatch[2]} months`;
        }
      }
    });

    return {
      procedures: procedures.length > 0 ? procedures : ['Fixed appliances', 'Regular follow-up'],
      duration,
      goals: ['Correct malocclusion', 'Improve facial profile', 'Achieve stable occlusion'],
    };
  }

  /**
   * ساخت رفرنس‌ها از اسناد
   */
  private static buildReferences(relevantDocs: DocumentChunk[]): ClinicalReference[] {
    const references: ClinicalReference[] = [];
    const seenRefs = new Set<string>();

    relevantDocs.forEach((doc) => {
      const refKey = `${doc.metadata.source}-${doc.metadata.page}`;
      if (!seenRefs.has(refKey)) {
        seenRefs.add(refKey);

        // استخراج محتوای مرتبط (اولین 500 کاراکتر)
        const relevantContent = doc.content.substring(0, 500);

        references.push({
          id: `ref-${doc.metadata.source}-${doc.metadata.page}-${doc.metadata.chunkIndex}`,
          title: doc.metadata.source,
          authors: doc.metadata.authors,
          year: doc.metadata.year,
          journal: doc.metadata.source,
          content: relevantContent,
          tags: RealClinicalRAGService.extractTags(doc.content),
          category: 'treatment',
          page: doc.metadata.page.toString(),
          chapter: doc.metadata.chapter,
          isReal: true, // این رفرنس واقعی است!
        });
      }
    });

    return references;
  }

  /**
   * استخراج تگ‌ها از محتوا
   */
  private static extractTags(content: string): string[] {
    const tags: string[] = [];
    const contentLower = content.toLowerCase();

    if (contentLower.includes('class ii') || contentLower.includes('کلاس ii')) {
      tags.push('class-II');
    }
    if (contentLower.includes('class iii') || contentLower.includes('کلاس iii')) {
      tags.push('class-III');
    }
    if (contentLower.includes('vertical') || contentLower.includes('عمودی')) {
      tags.push('vertical-growth');
    }
    if (contentLower.includes('treatment') || contentLower.includes('درمان')) {
      tags.push('treatment');
    }
    if (contentLower.includes('diagnosis') || contentLower.includes('تشخیص')) {
      tags.push('diagnosis');
    }

    return tags;
  }

  /**
   * ساخت تحلیل نهایی
   */
  private buildAnalysis(
    patientData: PatientData,
    issues: ClinicalAnalysis['issues'],
    treatmentInfo: { procedures: string[]; duration: string; goals: string[] },
    references: ClinicalReference[]
  ): ClinicalAnalysis {
    // ساخت تشخیص
    const diagnosis = this.buildDiagnosis(issues, patientData);

    // ساخت طرح درمان
    const treatmentPlan: TreatmentPlan[] = [
      {
        phase: 'درمان فعال',
        duration: treatmentInfo.duration,
        procedures: treatmentInfo.procedures,
        goals: treatmentInfo.goals,
        evidence: references.slice(0, 3),
        rationale: `بر اساس ${references[0]?.source || 'منابع علمی'} (${references[0]?.year || ''})`,
      },
    ];

    // ساخت توصیه‌ها
    const recommendations = issues.map((issue) => ({
      recommendation: issue.clinicalSignificance,
      evidence: references.filter((ref) =>
        ref.content.toLowerCase().includes(issue.parameter.toLowerCase())
      ),
      priority: issue.deviation > 5 ? ('high' as const) : issue.deviation > 2.5 ? ('medium' as const) : ('low' as const),
    }));

    // ساخت پیش‌بینی
    const prognosis = this.buildPrognosis(patientData, issues);

    // ساخت توضیحات
    const explanation = this.buildExplanation(patientData, issues, diagnosis, treatmentPlan, references);

    return {
      diagnosis,
      severity: this.determineSeverity(issues),
      issues,
      treatmentPlan,
      recommendations,
      prognosis,
      references,
      explanation,
    };
  }

  /**
   * ساخت تشخیص
   */
  private static buildDiagnosis(issues: ClinicalAnalysis['issues'], patientData: PatientData): string {
    const classII = issues.find((i) => i.parameter === 'ANB' && i.value > 4);
    const classIII = issues.find((i) => i.parameter === 'ANB' && i.value < 2);
    const vertical = issues.find((i) => i.parameter === 'FMA' && i.value > 28);

    if (classII) {
      return vertical ? 'کلاس II اسکلتی با الگوی رشد عمودی' : 'کلاس II اسکلتی';
    }
    if (classIII) {
      return 'کلاس III اسکلتی';
    }
    return 'ناهنجاری دندانی';
  }

  /**
   * ساخت پیش‌بینی
   */
  private static buildPrognosis(patientData: PatientData, issues: ClinicalAnalysis['issues']): string {
    const isGrowing = patientData.age < 15;
    const severeIssues = issues.filter((i) => i.deviation > 5);

    if (isGrowing && severeIssues.length === 0) {
      return 'پیش‌بینی عالی: با درمان مناسب و همکاری بیمار، نتایج مطلوب حاصل خواهد شد.';
    }
    if (isGrowing && severeIssues.length > 0) {
      return 'پیش‌بینی خوب: با درمان زودهنگام و مناسب، می‌توان نتایج خوبی کسب کرد.';
    }
    return 'پیش‌بینی محتاطانه: ممکن است نیاز به درمان طولانی‌تر باشد.';
  }

  /**
   * تعیین شدت
   */
  private static determineSeverity(issues: ClinicalAnalysis['issues']): 'mild' | 'moderate' | 'severe' {
    const severeCount = issues.filter((i) => i.deviation > 5).length;
    const moderateCount = issues.filter((i) => i.deviation > 2.5 && i.deviation <= 5).length;

    if (severeCount > 0) return 'severe';
    if (moderateCount > 2 || issues.length > 4) return 'moderate';
    return 'mild';
  }

  /**
   * ساخت توضیحات کامل
   */
  private static buildExplanation(
    patientData: PatientData,
    issues: ClinicalAnalysis['issues'],
    diagnosis: string,
    treatmentPlan: TreatmentPlan[],
    references: ClinicalReference[]
  ): string {
    let explanation = `# تحلیل بالینی بیمار (از PDF‌های واقعی)\n\n`;

    explanation += `## اطلاعات بیمار\n`;
    explanation += `- سن: ${patientData.age} سال\n`;
    explanation += `- جنسیت: ${patientData.gender === 'male' ? 'مرد' : 'زن'}\n\n`;

    explanation += `## تشخیص\n`;
    explanation += `${diagnosis}\n\n`;

    explanation += `## مشکلات شناسایی شده\n`;
    issues.forEach((issue, index) => {
      explanation += `${index + 1}. **${issue.parameter}**: ${issue.value.toFixed(1)}°\n`;
      explanation += `   - محدوده نرمال: ${issue.normalRange.min}-${issue.normalRange.max}°\n`;
      explanation += `   - انحراف: ${issue.deviation.toFixed(1)}°\n`;
      explanation += `   - توضیح: ${issue.description}\n`;
      explanation += `   - اهمیت بالینی: ${issue.clinicalSignificance}\n\n`;
    });

    explanation += `## طرح درمان\n`;
    treatmentPlan.forEach((phase, index) => {
      explanation += `### فاز ${index + 1}: ${phase.phase}\n`;
      explanation += `- مدت زمان: ${phase.duration}\n`;
      explanation += `- روش‌ها: ${phase.procedures.join(', ')}\n`;
      if (phase.evidence.length > 0) {
        explanation += `- شواهد علمی:\n`;
        phase.evidence.forEach((ref) => {
          explanation += `  - ${ref.authors} (${ref.year}): ${ref.title}`;
          if (ref.page) explanation += `, صفحه ${ref.page}`;
          if (ref.chapter) explanation += `, ${ref.chapter}`;
          explanation += `\n`;
        });
      }
      explanation += `\n`;
    });

    explanation += `## منابع و رفرنس‌های واقعی\n`;
    references.slice(0, 5).forEach((ref, index) => {
      explanation += `${index + 1}. ${ref.authors} (${ref.year}). ${ref.title}`;
      if (ref.page) explanation += `. صفحه ${ref.page}`;
      if (ref.chapter) explanation += `. ${ref.chapter}`;
      explanation += `\n`;
      explanation += `   ✅ این رفرنس از PDF واقعی استخراج شده است.\n\n`;
    });

    return explanation;
  }

  /**
   * Fallback: تحلیل ساده وقتی PDF نداریم
   */
  private static fallbackAnalysis(patientData: PatientData): ClinicalAnalysis {
    const basicAnalysis = generateComprehensiveAnalysis(
      patientData.cephalometricMeasurements as any,
      undefined
    );

    return {
      diagnosis: basicAnalysis.diagnosis,
      severity: 'moderate',
      issues: [],
      treatmentPlan: basicAnalysis.treatmentPlan.map((phase) => ({
        phase: phase.phase,
        duration: phase.duration,
        procedures: phase.procedures,
        goals: phase.goals,
        evidence: [],
        rationale: 'تحلیل اولیه',
      })),
      recommendations: [],
      prognosis: basicAnalysis.prognosis,
      references: [],
      explanation: '⚠️ هیچ PDF یافت نشد. از تحلیل اولیه استفاده شد.',
    };
  }

  /**
   * افزودن PDF جدید (بدون پردازش مجدد همه)
   */
  async addPDF(filePath: string): Promise<void> {
    try {
      const pdfDoc = await parsePDF(filePath);
      this.pdfDocuments.push(pdfDoc);

      // اضافه کردن به documents
      pdfDoc.pages.forEach((page) => {
        const chunks = this.splitIntoChunks(page.content, 500);
        chunks.forEach((chunk, chunkIndex) => {
          this.documents.push({
            content: chunk,
            metadata: {
              source: pdfDoc.title,
              authors: pdfDoc.authors,
              year: pdfDoc.year,
              page: page.pageNumber,
              chapter: page.chapter,
              section: page.section,
              filePath: pdfDoc.metadata.filePath,
              chunkIndex,
            },
          });
        });
      });

      console.log(`✅ [Real RAG] Added PDF: ${pdfDoc.title}`);
    } catch (error) {
      console.error('❌ [Real RAG] Error adding PDF:', error);
      throw error;
    }
  }

  /**
   * دریافت آمار
   */
  getStats() {
    return {
      pdfCount: this.pdfDocuments.length,
      documentChunks: this.documents.length,
      totalPages: this.pdfDocuments.reduce((sum, doc) => sum + doc.pages.length, 0),
      isInitialized: this.isInitialized,
      hasVectorStore: this.vectorStore !== null,
    };
  }
}

