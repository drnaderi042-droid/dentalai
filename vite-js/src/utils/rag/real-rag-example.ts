/**
 * مثال استفاده از Real Clinical RAG Service
 */

import { PatientData } from './rag-types.ts';
import { RealClinicalRAGService } from './real-rag-service.ts';

// ============================================================================
// مثال 1: راه‌اندازی و استفاده پایه
// ============================================================================

export async function exampleBasicUsage() {
  const ragService = new RealClinicalRAGService();

  // راه‌اندازی (فقط یک بار)
  // مسیر پوشه PDF‌ها را مشخص کنید
  const pdfDirectory = './knowledge-base/books'; // یا مسیر واقعی PDF‌ها

  try {
    await ragService.initialize(pdfDirectory, {
      useEmbeddings: false, // بدون Embeddings (رایگان و سریع)
    });

    console.log('✅ RAG Service initialized');
    console.log('Stats:', ragService.getStats());

    // استفاده
    const patient: PatientData = {
      age: 14,
      gender: 'male',
      cephalometricMeasurements: {
        SNA: 85,
        SNB: 78,
        ANB: 7,
        FMA: 30,
      },
    };

    const analysis = await ragService.analyzePatient(patient);

    console.log('=== تحلیل از PDF‌های واقعی ===');
    console.log('تشخیص:', analysis.diagnosis);
    console.log('رفرنس‌ها:', analysis.references.length);
    analysis.references.forEach((ref, i) => {
      console.log(`${i + 1}. ${ref.authors} (${ref.year}) - صفحه ${ref.page}`);
    });
  } catch (error) {
    console.error('Error:', error);
    console.log('💡 Tip: Make sure PDFs are in the knowledge-base/books directory');
  }
}

// ============================================================================
// مثال 2: استفاده با Embeddings (دقیق‌تر اما نیاز به API key)
// ============================================================================

export async function exampleWithEmbeddings() {
  const ragService = new RealClinicalRAGService();

  await ragService.initialize('./knowledge-base/books', {
    useEmbeddings: true,
    apiKey: process.env.OPENAI_API_KEY, // نیاز به API key
    vectorStoreConfig: {
      collectionName: 'clinical-orthodontics',
    },
  });

  const patient: PatientData = {
    age: 14,
    gender: 'male',
    cephalometricMeasurements: {
      SNA: 85,
      SNB: 78,
      ANB: 7,
    },
  };

  const analysis = await ragService.analyzePatient(patient);
  return analysis;
}

// ============================================================================
// مثال 3: افزودن PDF جدید
// ============================================================================

export async function exampleAddPDF() {
  const ragService = new RealClinicalRAGService();
  await ragService.initialize('./knowledge-base/books');

  // افزودن PDF جدید بدون پردازش مجدد همه
  await ragService.addPDF('./knowledge-base/articles/new-article.pdf');

  console.log('Stats after adding PDF:', ragService.getStats());
}

// ============================================================================
// مثال 4: استفاده در React Component
// ============================================================================

/**
 * مثال استفاده در React:
 * 
 * import { useState, useEffect } from 'react';
 * import { RealClinicalRAGService } from 'src/utils/rag/real-rag-service';
 * 
 * function PatientRAGAnalysis({ patientData }) {
 *   const [analysis, setAnalysis] = useState(null);
 *   const [loading, setLoading] = useState(false);
 *   const [ragService] = useState(() => new RealClinicalRAGService());
 * 
 *   useEffect(() => {
 *     async function init() {
 *       // راه‌اندازی (فقط یک بار)
 *       await ragService.initialize('./knowledge-base/books');
 *     }
 *     init();
 *   }, []);
 * 
 *   const handleAnalyze = async () => {
 *     setLoading(true);
 *     const result = await ragService.analyzePatient(patientData);
 *     setAnalysis(result);
 *     setLoading(false);
 *   };
 * 
 *   return (
 *     <div>
 *       <button onClick={handleAnalyze}>تحلیل با RAG واقعی</button>
 *       {analysis && (
 *         <div>
 *           <h2>{analysis.diagnosis}</h2>
 *           <h3>رفرنس‌های واقعی:</h3>
 *           {analysis.references.map(ref => (
 *             <div key={ref.id}>
 *               {ref.authors} ({ref.year}) - صفحه {ref.page}
 *             </div>
 *           ))}
 *         </div>
 *       )}
 *     </div>
 *   );
 * }
 */

