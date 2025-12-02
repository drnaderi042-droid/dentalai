/**
 * Cephalometric RAG Service
 * سرویس RAG برای تحلیل پرونده‌های بیمار و آنالیز سفالومتری
 */

import {
  generateComprehensiveAnalysis,
} from '../orthodontic-analysis.ts';

// ============================================================================
// Types
// ============================================================================

export interface PatientRecord {
  patientId: string;
  age: number;
  gender: 'male' | 'female';
  cephalometricMeasurements: CephalometricMeasurements;
  medicalHistory?: string;
  previousTreatments?: string[];
  images?: string[];
  analysisHistory?: AnalysisRecord[];
  facialLandmarks?: any;
}

export interface AnalysisRecord {
  date: string;
  measurements: CephalometricMeasurements;
  diagnosis?: string;
  treatmentPlan?: string[];
}

export interface RAGContext {
  patientAnalysis: ComprehensiveAnalysis;
  relevantGuidelines: string[];
  similarCases: string[];
  parameterExplanations: Record<string, string>;
}

export interface RAGResponse {
  diagnosis: string;
  recommendations: string[];
  treatmentPlan: string[];
  explanation: string;
  confidence: 'high' | 'medium' | 'low';
  sources: string[];
}

// ============================================================================
// Knowledge Base (پایگاه دانش)
// ============================================================================

/**
 * محدوده‌های نرمال پارامترهای سفالومتری
 */
export const NORMAL_RANGES = {
  SNA: { min: 80, max: 84, description: 'موقعیت قدامی-خلفی فک بالا' },
  SNB: { min: 78, max: 82, description: 'موقعیت قدامی-خلفی فک پایین' },
  ANB: { min: 2, max: 4, description: 'رابطه قدامی-خلفی فک بالا و پایین' },
  FMA: { min: 22, max: 28, description: 'الگوی رشد عمودی' },
  FMIA: { min: 65, max: 75, description: 'زاویه دندان قدامی پایین با صفحه فرانکفورت' },
  IMPA: { min: 90, max: 95, description: 'زاویه دندان قدامی پایین با صفحه مندیبولار' },
  'U1-SN': { min: 102, max: 108, description: 'زاویه دندان قدامی بالا با خط SN' },
  'L1-MP': { min: 90, max: 95, description: 'زاویه دندان قدامی پایین با صفحه مندیبولار' },
};

/**
 * راهنماهای بالینی
 */
export const CLINICAL_GUIDELINES = {
  classII: {
    description: 'کلاس II اسکلتی (ANB > 4)',
    treatment: [
      'در بیماران در حال رشد: استفاده از دستگاه‌های فانکشنال (Face Mask, Twin Block)',
      'براکت ثابت برای اصلاح چیدمان دندانی',
      'مینی ایمپلانت در صورت نیاز برای کنترل انکوریج',
      'مدت زمان درمان: 12-18 ماه در بیماران در حال رشد، 18-24 ماه در بزرگسالان',
    ],
  },
  classIII: {
    description: 'کلاس III اسکلتی (ANB < 2)',
    treatment: [
      'در بیماران در حال رشد: دستگاه‌های فانکشنال پیش‌برنده (Chin Cup, Reverse Headgear)',
      'براکت ثابت',
      'در بزرگسالان: ممکن است نیاز به جراحی ارتوگناتیک باشد',
      'مدت زمان درمان: 18-24 ماه در بیماران در حال رشد، 24-30 ماه در بزرگسالان',
    ],
  },
  verticalGrowth: {
    description: 'الگوی رشد عمودی (FMA > 28)',
    treatment: [
      'کنترل رشد عمودی با هایپولای',
      'بایت پلیت برای کنترل ارتفاع صورت',
      'اجتناب از اکستروژن دندان‌های خلفی',
      'استفاده از مینی ایمپلانت برای کنترل عمودی',
    ],
  },
  horizontalGrowth: {
    description: 'الگوی رشد افقی (FMA < 22)',
    treatment: [
      'استفاده از اکستروژن دندان‌های خلفی',
      'اجتناب از اینتروژن دندان‌های قدامی',
      'کنترل رشد افقی',
    ],
  },
  proclinedIncisors: {
    description: 'دندان‌های قدامی به سمت جلو (Proclined)',
    treatment: [
      'رترکشن دندان‌های قدامی',
      'استفاده از اسپرینگ رترکشن',
      'مینی ایمپلانت برای انکوریج',
    ],
  },
  retroclinedIncisors: {
    description: 'دندان‌های قدامی به سمت عقب (Retroclined)',
    treatment: [
      'پروتراکشن دندان‌های قدامی',
      'استفاده از اسپرینگ پروتراکشن',
    ],
  },
};

/**
 * توضیحات پارامترها
 */
export const PARAMETER_EXPLANATIONS: Record<string, string> = {
  SNA: 'زاویه بین نقاط Sella (S)، Nasion (N) و Point A. نشان‌دهنده موقعیت قدامی-خلفی فک بالا (ماگزیلا) نسبت به قاعده جمجمه است.',
  SNB: 'زاویه بین نقاط Sella (S)، Nasion (N) و Point B. نشان‌دهنده موقعیت قدامی-خلفی فک پایین (مندیبل) نسبت به قاعده جمجمه است.',
  ANB: 'اختلاف بین SNA و SNB. نشان‌دهنده رابطه قدامی-خلفی فک بالا و پایین است. افزایش: کلاس II، کاهش: کلاس III',
  FMA: 'زاویه بین صفحه فرانکفورت (Or-Po) و صفحه مندیبولار (Go-Me). نشان‌دهنده الگوی رشد عمودی است.',
  FMIA: 'زاویه بین صفحه فرانکفورت و محور طولی دندان قدامی پایین.',
  IMPA: 'زاویه بین صفحه مندیبولار و محور طولی دندان قدامی پایین.',
  'U1-SN': 'زاویه بین محور طولی دندان قدامی بالا و خط SN.',
  'L1-MP': 'زاویه بین محور طولی دندان قدامی پایین و صفحه مندیبولار.',
};

// ============================================================================
// Cephalometric RAG Service
// ============================================================================

export class CephalometricRAGService {
  private knowledgeBase: {
    normalRanges: typeof NORMAL_RANGES;
    guidelines: typeof CLINICAL_GUIDELINES;
    explanations: typeof PARAMETER_EXPLANATIONS;
  };

  constructor() {
    this.knowledgeBase = {
      normalRanges: NORMAL_RANGES,
      guidelines: CLINICAL_GUIDELINES,
      explanations: PARAMETER_EXPLANATIONS,
    };
  }

  /**
   * تحلیل کامل بیمار با استفاده از RAG
   */
  static async analyzePatient(
    patientRecord: PatientRecord,
    question?: string
  ): Promise<RAGResponse> {
    // 1. تحلیل اولیه با توابع موجود
    const patientAnalysis = generateComprehensiveAnalysis(
      patientRecord.cephalometricMeasurements,
      patientRecord.facialLandmarks,
      patientRecord.age
    );

    // 2. بازیابی اطلاعات مرتبط از پایگاه دانش
    const context = CephalometricRAGService.retrieveRelevantContext(patientRecord, patientAnalysis);

    // 3. تولید پاسخ
    const response = CephalometricRAGService.generateResponse(
      patientRecord,
      patientAnalysis,
      context,
      question || 'لطفاً تحلیل کامل این بیمار را ارائه دهید'
    );

    return response;
  }

  /**
   * بازیابی اطلاعات مرتبط از پایگاه دانش
   */
  private static retrieveRelevantContext(
    patientRecord: PatientRecord,
    analysis: ComprehensiveAnalysis
  ): RAGContext {
    const relevantGuidelines: string[] = [];
    const similarCases: string[] = [];
    const parameterExplanations: Record<string, string> = {};

    // شناسایی مشکلات
    const { issues } = analysis;

    // بازیابی راهنماهای مرتبط
    issues.forEach((issue) => {
      if (issue.parameter === 'ANB') {
        if (issue.value && issue.value > 4) {
          relevantGuidelines.push(CLINICAL_GUIDELINES.classII.description);
          relevantGuidelines.push(...CLINICAL_GUIDELINES.classII.treatment);
        } else if (issue.value && issue.value < 2) {
          relevantGuidelines.push(CLINICAL_GUIDELINES.classIII.description);
          relevantGuidelines.push(...CLINICAL_GUIDELINES.classIII.treatment);
        }
      }

      if (issue.parameter === 'FMA') {
        if (issue.value && issue.value > 28) {
          relevantGuidelines.push(CLINICAL_GUIDELINES.verticalGrowth.description);
          relevantGuidelines.push(...CLINICAL_GUIDELINES.verticalGrowth.treatment);
        } else if (issue.value && issue.value < 22) {
          relevantGuidelines.push(CLINICAL_GUIDELINES.horizontalGrowth.description);
          relevantGuidelines.push(...CLINICAL_GUIDELINES.horizontalGrowth.treatment);
        }
      }

      if (issue.description.includes('Proclined') || issue.description.includes('جلو')) {
        relevantGuidelines.push(CLINICAL_GUIDELINES.proclinedIncisors.description);
        relevantGuidelines.push(...CLINICAL_GUIDELINES.proclinedIncisors.treatment);
      }

      if (issue.description.includes('Retroclined') || issue.description.includes('عقب')) {
        relevantGuidelines.push(CLINICAL_GUIDELINES.retroclinedIncisors.description);
        relevantGuidelines.push(...CLINICAL_GUIDELINES.retroclinedIncisors.treatment);
      }

      // افزودن توضیحات پارامترها
      if (issue.parameter && PARAMETER_EXPLANATIONS[issue.parameter]) {
        parameterExplanations[issue.parameter] = PARAMETER_EXPLANATIONS[issue.parameter];
      }
    });

    // افزودن توضیحات برای تمام پارامترهای موجود
    Object.keys(patientRecord.cephalometricMeasurements).forEach((param) => {
      if (PARAMETER_EXPLANATIONS[param] && !parameterExplanations[param]) {
        parameterExplanations[param] = PARAMETER_EXPLANATIONS[param];
      }
    });

    return {
      patientAnalysis: analysis,
      relevantGuidelines,
      similarCases,
      parameterExplanations,
    };
  }

  /**
   * تولید پاسخ بر اساس تحلیل و اطلاعات بازیابی شده
   */
  private static generateResponse(
    patientRecord: PatientRecord,
    analysis: ComprehensiveAnalysis,
    context: RAGContext,
    question: string
  ): RAGResponse {
    // ساخت تشخیص
    let { diagnosis } = analysis;
    if (analysis.issues.length > 0) {
      const severeIssues = analysis.issues.filter((i) => i.severity === 'severe');
      const moderateIssues = analysis.issues.filter((i) => i.severity === 'moderate');

      if (severeIssues.length > 0) {
        diagnosis += ` با ${severeIssues.length} مشکل شدید`;
      } else if (moderateIssues.length > 0) {
        diagnosis += ` با ${moderateIssues.length} مشکل متوسط`;
      }
    }

    // ساخت توصیه‌ها
    const recommendations: string[] = [];

    // افزودن توصیه‌های از راهنماهای بالینی
    if (context.relevantGuidelines.length > 0) {
      recommendations.push(...context.relevantGuidelines);
    }

    // افزودن توصیه‌های از تحلیل
    if (analysis.recommendations.length > 0) {
      recommendations.push(...analysis.recommendations);
    }

    // ساخت طرح درمان
    const treatmentPlan: string[] = [];
    analysis.treatmentPlan.forEach((phase) => {
      treatmentPlan.push(`${phase.phase} (${phase.duration})`);
      treatmentPlan.push(...phase.procedures.map((p) => `  - ${p}`));
    });

    // ساخت توضیحات
    let explanation = `تحلیل بیمار ${patientRecord.patientId}:\n\n`;
    explanation += `سن: ${patientRecord.age} سال\n`;
    explanation += `جنسیت: ${patientRecord.gender === 'male' ? 'مرد' : 'زن'}\n\n`;

    explanation += `تشخیص: ${diagnosis}\n\n`;

    if (analysis.issues.length > 0) {
      explanation += `مشکلات شناسایی شده:\n`;
      analysis.issues.forEach((issue, index) => {
        explanation += `${index + 1}. ${issue.description}`;
        if (issue.parameter && issue.value !== undefined) {
          const normal = NORMAL_RANGES[issue.parameter as keyof typeof NORMAL_RANGES];
          if (normal) {
            explanation += ` (${issue.parameter}: ${issue.value.toFixed(1)}°، محدوده نرمال: ${normal.min}-${normal.max}°)`;
          } else {
            explanation += ` (${issue.parameter}: ${issue.value.toFixed(1)}°)`;
          }
        }
        explanation += `\n`;
      });
      explanation += `\n`;
    }

    // افزودن توضیحات پارامترها
    if (Object.keys(context.parameterExplanations).length > 0) {
      explanation += `توضیحات پارامترها:\n`;
      Object.entries(context.parameterExplanations).forEach(([param, desc]) => {
        explanation += `- ${param}: ${desc}\n`;
      });
      explanation += `\n`;
    }

    explanation += `پیش‌بینی: ${analysis.prognosis}\n`;

    // محاسبه سطح اطمینان
    let confidence: 'high' | 'medium' | 'low' = 'medium';
    if (analysis.issues.length === 0) {
      confidence = 'high';
    } else if (analysis.issues.some((i) => i.severity === 'severe')) {
      confidence = 'high'; // مشکلات شدید معمولاً واضح‌تر هستند
    } else if (analysis.issues.length > 5) {
      confidence = 'low'; // مشکلات متعدد ممکن است پیچیده باشند
    }

    // منابع
    const sources = [
      'تحلیل خودکار بر اساس پارامترهای سفالومتری',
      'راهنماهای بالینی ارتودنسی',
    ];

    return {
      diagnosis,
      recommendations,
      treatmentPlan,
      explanation,
      confidence,
      sources,
    };
  }

  /**
   * جستجوی موارد مشابه
   */
  static findSimilarCases(
    patientRecord: PatientRecord,
    caseDatabase: PatientRecord[]
  ): PatientRecord[] {
    const currentMeasurements = patientRecord.cephalometricMeasurements;
    const similarCases: Array<{ patient: PatientRecord; similarity: number }> = [];

    caseDatabase.forEach((caseRecord) => {
      if (caseRecord.patientId === patientRecord.patientId) return;

      let similarity = 0;
      let count = 0;

      Object.keys(currentMeasurements).forEach((param) => {
        const currentValue = currentMeasurements[param as keyof CephalometricMeasurements];
        const caseValue = caseRecord.cephalometricMeasurements[param as keyof CephalometricMeasurements];

        if (currentValue !== undefined && caseValue !== undefined) {
          const normal = NORMAL_RANGES[param as keyof typeof NORMAL_RANGES];
          if (normal) {
            const currentDiff = Math.abs(currentValue - (normal.min + normal.max) / 2);
            const caseDiff = Math.abs(caseValue - (normal.min + normal.max) / 2);
            const similarityScore = 1 - Math.abs(currentDiff - caseDiff) / 10;
            similarity += Math.max(0, similarityScore);
          } else {
            const diff = Math.abs(currentValue - caseValue);
            const similarityScore = 1 - diff / 20;
            similarity += Math.max(0, similarityScore);
          }
          count++;
        }
      });

      if (count > 0) {
        similarity /= count;
        similarCases.push({ patient: caseRecord, similarity });
      }
    });

    return similarCases
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, 5)
      .map((item) => item.patient);
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * تبدیل پرونده بیمار به متن برای embedding
 */
export function patientRecordToText(patientRecord: PatientRecord): string {
  let text = `بیمار ${patientRecord.patientId}\n`;
  text += `سن: ${patientRecord.age} سال\n`;
  text += `جنسیت: ${patientRecord.gender === 'male' ? 'مرد' : 'زن'}\n\n`;

  text += `اندازه‌گیری‌های سفالومتری:\n`;
  Object.entries(patientRecord.cephalometricMeasurements).forEach(([param, value]) => {
    if (value !== undefined) {
      const normal = NORMAL_RANGES[param as keyof typeof NORMAL_RANGES];
      if (normal) {
        text += `${param}: ${value.toFixed(1)}° (محدوده نرمال: ${normal.min}-${normal.max}°)\n`;
      } else {
        text += `${param}: ${value.toFixed(1)}°\n`;
      }
    }
  });

  if (patientRecord.medicalHistory) {
    text += `\nسابقه پزشکی: ${patientRecord.medicalHistory}\n`;
  }

  if (patientRecord.previousTreatments && patientRecord.previousTreatments.length > 0) {
    text += `\nدرمان‌های قبلی: ${patientRecord.previousTreatments.join(', ')}\n`;
  }

  return text;
}

/**
 * ساخت prompt برای LLM
 */
export function buildPrompt(
  patientRecord: PatientRecord,
  analysis: ComprehensiveAnalysis,
  context: RAGContext,
  question: string
): string {
  let prompt = `شما یک متخصص ارتودنسی با تجربه هستید. بر اساس داده‌های زیر، تحلیل و توصیه ارائه دهید.\n\n`;

  prompt += `=== اطلاعات بیمار ===\n`;
  prompt += patientRecordToText(patientRecord);
  prompt += `\n`;

  prompt += `=== تحلیل اولیه ===\n`;
  prompt += `تشخیص: ${analysis.diagnosis}\n`;
  if (analysis.issues.length > 0) {
    prompt += `مشکلات: ${analysis.issues.map((i) => i.description).join(', ')}\n`;
  }
  prompt += `\n`;

  if (context.relevantGuidelines.length > 0) {
    prompt += `=== راهنماهای بالینی مرتبط ===\n`;
    context.relevantGuidelines.forEach((guideline) => {
      prompt += `- ${guideline}\n`;
    });
    prompt += `\n`;
  }

  prompt += `=== سوال ===\n`;
  prompt += `${question}\n\n`;

  prompt += `لطفاً پاسخ جامع و دقیق ارائه دهید که شامل:\n`;
  prompt += `1. تحلیل دقیق وضعیت بیمار\n`;
  prompt += `2. توضیح مشکلات شناسایی شده\n`;
  prompt += `3. توصیه‌های درمانی\n`;
  prompt += `4. پیش‌بینی نتایج درمان\n`;

  return prompt;
}

