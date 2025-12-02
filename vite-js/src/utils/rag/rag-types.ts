/**
 * RAG Types
 * انواع داده‌ای مشترک برای سرویس‌های RAG
 */

import { CephalometricMeasurements } from '../orthodontic-analysis.ts';

// ============================================================================
// Types
// ============================================================================

export interface PatientData {
  age: number;
  gender: 'male' | 'female';
  cephalometricMeasurements: CephalometricMeasurements | Record<string, number>; // پشتیبانی از همه پارامترها
  medicalHistory?: string;
  previousTreatments?: string[];
  cephalometricTable?: Record<string, { mean?: string | number; sd?: string | number; note?: string }>; // برای دسترسی به mean و sd
}

export interface ClinicalReference {
  id: string;
  title: string;
  authors: string;
  year: number;
  journal?: string;
  content: string;
  tags: string[];
  category: 'diagnosis' | 'treatment' | 'guideline' | 'case-study';
  page?: string; // شماره صفحه
  chapter?: string; // شماره یا نام فصل
  volume?: string; // شماره جلد (برای مجلات)
  isReal?: boolean; // آیا از منبع واقعی استخراج شده یا mock data
}

export interface TreatmentPlan {
  phase: string;
  duration: string;
  procedures: string[];
  goals: string[];
  evidence: ClinicalReference[];
  rationale: string;
}

export interface ClinicalAnalysis {
  diagnosis: string;
  severity: 'mild' | 'moderate' | 'severe';
  issues: Array<{
    parameter: string;
    value: number;
    normalRange: { min: number; max: number };
    deviation: number;
    description: string;
    clinicalSignificance: string;
  }>;
  treatmentPlan: TreatmentPlan[];
  recommendations: Array<{
    recommendation: string;
    evidence: ClinicalReference[];
    priority: 'high' | 'medium' | 'low';
  }>;
  prognosis: string;
  references: ClinicalReference[];
  explanation: string;
}





