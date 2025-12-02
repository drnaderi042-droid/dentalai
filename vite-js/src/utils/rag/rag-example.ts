/**
 * مثال استفاده از سیستم RAG برای آنالیز سفالومتری
 */

import {
  PatientRecord,
  patientRecordToText,
  CephalometricRAGService,
} from './cephalometric-rag-service.ts';

// ============================================================================
// مثال 1: تحلیل بیمار با کلاس II اسکلتی
// ============================================================================

export async function exampleClassIIPatient() {
  const ragService = new CephalometricRAGService();

  const patient: PatientRecord = {
    patientId: 'P001',
    age: 14,
    gender: 'male',
    cephalometricMeasurements: {
      SNA: 85, // افزایش یافته (نرمال: 80-84)
      SNB: 78, // نرمال
      ANB: 7, // افزایش یافته (نرمال: 2-4) - کلاس II
      FMA: 30, // افزایش یافته (نرمال: 22-28)
      'U1-SN': 110, // افزایش یافته (نرمال: 102-108)
      IMPA: 100, // افزایش یافته (نرمال: 90-95)
    },
    medicalHistory: 'هیچ سابقه بیماری خاصی ندارد',
    previousTreatments: [],
  };

  const question = 'این بیمار چه نوع ناهنجاری دارد و چه درمانی پیشنهاد می‌کنید؟';

  const response = await ragService.analyzePatient(patient, question);

  console.log('=== پاسخ RAG ===');
  console.log('تشخیص:', response.diagnosis);
  console.log('\nتوصیه‌ها:');
  response.recommendations.forEach((rec, i) => {
    console.log(`${i + 1}. ${rec}`);
  });
  console.log('\nطرح درمان:');
  response.treatmentPlan.forEach((item) => {
    console.log(item);
  });
  console.log('\nتوضیحات:');
  console.log(response.explanation);
  console.log('\nسطح اطمینان:', response.confidence);
  console.log('\nمنابع:', response.sources.join(', '));

  return response;
}

// ============================================================================
// مثال 2: تحلیل بیمار با کلاس III اسکلتی
// ============================================================================

export async function exampleClassIIIPatient() {
  const ragService = new CephalometricRAGService();

  const patient: PatientRecord = {
    patientId: 'P002',
    age: 16,
    gender: 'female',
    cephalometricMeasurements: {
      SNA: 78, // کاهش یافته
      SNB: 82, // افزایش یافته
      ANB: -4, // کاهش یافته (نرمال: 2-4) - کلاس III
      FMA: 20, // کاهش یافته
      'U1-SN': 95, // کاهش یافته
      IMPA: 85, // کاهش یافته
    },
    medicalHistory: 'سابقه ارتودنسی در کودکی',
    previousTreatments: ['براکت متحرک در سن 10 سالگی'],
  };

  const question = 'آیا این بیمار نیاز به جراحی دارد یا می‌توان با درمان ارتودنسی مشکل را حل کرد؟';

  const response = await ragService.analyzePatient(patient, question);

  return response;
}

// ============================================================================
// مثال 3: جستجوی موارد مشابه
// ============================================================================

export async function exampleSimilarCases() {
  const ragService = new CephalometricRAGService();

  const currentPatient: PatientRecord = {
    patientId: 'P003',
    age: 13,
    gender: 'male',
    cephalometricMeasurements: {
      SNA: 83,
      SNB: 79,
      ANB: 4,
      FMA: 25,
    },
  };

  // پایگاه داده موارد قبلی (در واقعیت از دیتابیس می‌آید)
  const caseDatabase: PatientRecord[] = [
    {
      patientId: 'P001',
      age: 14,
      gender: 'male',
      cephalometricMeasurements: {
        SNA: 85,
        SNB: 78,
        ANB: 7,
        FMA: 30,
      },
    },
    {
      patientId: 'P002',
      age: 16,
      gender: 'female',
      cephalometricMeasurements: {
        SNA: 78,
        SNB: 82,
        ANB: -4,
        FMA: 20,
      },
    },
    {
      patientId: 'P004',
      age: 12,
      gender: 'male',
      cephalometricMeasurements: {
        SNA: 82,
        SNB: 79,
        ANB: 3,
        FMA: 24,
      },
    },
  ];

  const similarCases = ragService.findSimilarCases(currentPatient, caseDatabase);

  console.log('=== موارد مشابه ===');
  similarCases.forEach((similarCase, index) => {
    console.log(`\n${index + 1}. بیمار ${similarCase.patientId}`);
    console.log(`   سن: ${similarCase.age} سال`);
    console.log(`   اندازه‌گیری‌ها:`, similarCase.cephalometricMeasurements);
  });

  return similarCases;
}

// ============================================================================
// مثال 4: استفاده از توابع helper
// ============================================================================

export function exampleHelperFunctions() {
  const patient: PatientRecord = {
    patientId: 'P005',
    age: 15,
    gender: 'female',
    cephalometricMeasurements: {
      SNA: 81,
      SNB: 79,
      ANB: 2,
      FMA: 26,
    },
  };

  // تبدیل به متن
  const text = patientRecordToText(patient);
  console.log('=== متن پرونده ===');
  console.log(text);

  // ساخت prompt (نیاز به تحلیل دارد)
  // const analysis = generateComprehensiveAnalysis(...);
  // const context = ...;
  // const prompt = buildPrompt(patient, analysis, context, 'سوال؟');
  // console.log('=== Prompt ===');
  // console.log(prompt);
}

// ============================================================================
// مثال 5: استفاده در کامپوننت React
// ============================================================================

/**
 * مثال استفاده در یک کامپوننت React
 * 
 * import { useState, useEffect } from 'react';
 * import { CephalometricRAGService, PatientRecord } from 'src/utils/rag/cephalometric-rag-service';
 * 
 * function PatientAnalysisComponent({ patientId }) {
 *   const [response, setResponse] = useState(null);
 *   const [loading, setLoading] = useState(false);
 * 
 *   useEffect(() => {
 *     async function analyze() {
 *       setLoading(true);
 *       const ragService = new CephalometricRAGService();
 *       
 *       // دریافت اطلاعات بیمار از API
 *       const patientData = await fetchPatientData(patientId);
 *       
 *       const result = await ragService.analyzePatient(
 *         patientData,
 *         'لطفاً تحلیل کامل این بیمار را ارائه دهید'
 *       );
 *       
 *       setResponse(result);
 *       setLoading(false);
 *     }
 *     
 *     analyze();
 *   }, [patientId]);
 * 
 *   if (loading) return <div>در حال تحلیل...</div>;
 *   if (!response) return null;
 * 
 *   return (
 *     <div>
 *       <h2>تشخیص: {response.diagnosis}</h2>
 *       <h3>توصیه‌ها:</h3>
 *       <ul>
 *         {response.recommendations.map((rec, i) => (
 *           <li key={i}>{rec}</li>
 *         ))}
 *       </ul>
 *       <h3>طرح درمان:</h3>
 *       <pre>{response.treatmentPlan.join('\n')}</pre>
 *       <p>{response.explanation}</p>
 *     </div>
 *   );
 * }
 */

// ============================================================================
// اجرای مثال‌ها
// ============================================================================

if (typeof window === 'undefined') {
  // فقط در Node.js اجرا شود
  (async () => {
    console.log('=== مثال 1: بیمار کلاس II ===\n');
    await exampleClassIIPatient();

    console.log('\n\n=== مثال 2: بیمار کلاس III ===\n');
    await exampleClassIIIPatient();

    console.log('\n\n=== مثال 3: موارد مشابه ===\n');
    await exampleSimilarCases();

    console.log('\n\n=== مثال 4: توابع Helper ===\n');
    exampleHelperFunctions();
  })();
}


