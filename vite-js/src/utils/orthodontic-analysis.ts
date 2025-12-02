/**
 * Orthodontic Analysis Utility Functions
 * توابع تحلیل ارتودنسی برای شناسایی مشکلات و تولید طرح درمان
 */

export interface CephalometricMeasurements {
  SNA?: number;
  SNB?: number;
  ANB?: number;
  FMA?: number;
  FMIA?: number;
  IMPA?: number;
  'U1-SN'?: number;
  'L1-MP'?: number;
  GoGnSN?: number;
  [key: string]: number | undefined;
}

export interface FacialLandmarkAnalysis {
  landmarks: any[];
  measurements?: {
    facialAngle?: number;
    nasolabialAngle?: number;
    lipCompetence?: boolean;
    [key: string]: any;
  };
}

export interface OrthodonticIssue {
  type: 'skeletal' | 'dental' | 'soft_tissue' | 'functional';
  severity: 'mild' | 'moderate' | 'severe';
  description: string;
  parameter?: string;
  value?: number;
  normalRange?: { min: number; max: number };
}

export interface TreatmentPlan {
  phase: string;
  duration: string;
  procedures: string[];
  goals: string[];
}

export interface ComprehensiveAnalysis {
  issues: OrthodonticIssue[];
  diagnosis: string;
  treatmentPlan: TreatmentPlan[];
  recommendations: string[];
  prognosis: string;
}

/**
 * Analyze cephalometric measurements and identify issues
 */
export function analyzeCephalometricMeasurements(
  measurements: CephalometricMeasurements
): OrthodonticIssue[] {
  const issues: OrthodonticIssue[] = [];

  // Normal ranges for cephalometric parameters
  const normalRanges: Record<string, { min: number; max: number }> = {
    SNA: { min: 80, max: 84 },
    SNB: { min: 78, max: 82 },
    ANB: { min: 2, max: 4 },
    FMA: { min: 22, max: 28 },
    FMIA: { min: 65, max: 75 },
    IMPA: { min: 90, max: 95 },
    'U1-SN': { min: 102, max: 108 },
    'L1-MP': { min: 90, max: 95 },
  };

  // Helper function to check parameter
  const checkParameter = (
    param: string,
    value: number | undefined,
    type: OrthodonticIssue['type'],
    descriptions: {
      high: string;
      low: string;
      severity: (diff: number) => OrthodonticIssue['severity'];
    }
  ) => {
    if (value === undefined || value === null) return;

    const normal = normalRanges[param];
    if (!normal) return;

    const diffHigh = value - normal.max;
    const diffLow = normal.min - value;

    if (value > normal.max) {
      const diff = Math.abs(diffHigh);
      issues.push({
        type,
        severity: descriptions.severity(diff),
        description: descriptions.high,
        parameter: param,
        value,
        normalRange: normal,
      });
    } else if (value < normal.min) {
      const diff = Math.abs(diffLow);
      issues.push({
        type,
        severity: descriptions.severity(diff),
        description: descriptions.low,
        parameter: param,
        value,
        normalRange: normal,
      });
    }
  };

  // SNA - Maxillary position
  checkParameter('SNA', measurements.SNA, 'skeletal', {
    high: 'ماگزیلا در موقعیت قدامی قرار دارد (Protrusion)',
    low: 'ماگزیلا در موقعیت خلفی قرار دارد (Retrusion)',
    severity: (diff) => (diff > 5 ? 'severe' : diff > 2 ? 'moderate' : 'mild'),
  });

  // SNB - Mandibular position
  checkParameter('SNB', measurements.SNB, 'skeletal', {
    high: 'مندیبل در موقعیت قدامی قرار دارد (Prognathism)',
    low: 'مندیبل در موقعیت خلفی قرار دارد (Retrognathism)',
    severity: (diff) => (diff > 5 ? 'severe' : diff > 2 ? 'moderate' : 'mild'),
  });

  // ANB - Skeletal class
  checkParameter('ANB', measurements.ANB, 'skeletal', {
    high: 'کلاس II اسکلتی (Mandibular deficiency)',
    low: 'کلاس III اسکلتی (Mandibular excess)',
    severity: (diff) => (diff > 3 ? 'severe' : diff > 1.5 ? 'moderate' : 'mild'),
  });

  // FMA - Vertical growth pattern
  checkParameter('FMA', measurements.FMA, 'skeletal', {
    high: 'الگوی رشد عمودی (Vertical growth pattern) - باز بودن صورت',
    low: 'الگوی رشد افقی (Horizontal growth pattern) - بسته بودن صورت',
    severity: (diff) => (diff > 6 ? 'severe' : diff > 3 ? 'moderate' : 'mild'),
  });

  // IMPA - Lower incisor position
  checkParameter('IMPA', measurements.IMPA, 'dental', {
    high: 'دندان‌های قدامی پایین به سمت جلو (Proclined)',
    low: 'دندان‌های قدامی پایین به سمت عقب (Retroclined)',
    severity: (diff) => (diff > 8 ? 'severe' : diff > 4 ? 'moderate' : 'mild'),
  });

  // U1-SN - Upper incisor position
  checkParameter('U1-SN', measurements['U1-SN'], 'dental', {
    high: 'دندان‌های قدامی بالا به سمت جلو (Proclined)',
    low: 'دندان‌های قدامی بالا به سمت عقب (Retroclined)',
    severity: (diff) => (diff > 8 ? 'severe' : diff > 4 ? 'moderate' : 'mild'),
  });

  // FMIA - Lower incisor to Frankfort plane
  checkParameter('FMIA', measurements.FMIA, 'dental', {
    high: 'زاویه FMIA بالا - نیاز به کنترل عمودی',
    low: 'زاویه FMIA پایین - نیاز به اصلاح موقعیت دندان‌های قدامی',
    severity: (diff) => (diff > 8 ? 'severe' : diff > 4 ? 'moderate' : 'mild'),
  });

  return issues;
}

/**
 * Generate treatment plan based on identified issues
 */
export function generateTreatmentPlan(issues: OrthodonticIssue[]): TreatmentPlan[] {
  const plan: TreatmentPlan[] = [];

  // Identify primary issues
  const skeletalIssues = issues.filter((i) => i.type === 'skeletal');
  const dentalIssues = issues.filter((i) => i.type === 'dental');
  const severeIssues = issues.filter((i) => i.severity === 'severe');

  // Phase 1: Initial assessment and preparation
  plan.push({
    phase: 'مرحله ۱: ارزیابی و آماده‌سازی',
    duration: '۲-۳ ماه',
    procedures: [
      'عکاسی رادیولوژی کامل (پانوراما و لترال سفالومتری)',
      'تهیه مدل‌های دندانی و آنالیز',
      'بررسی بهداشت دهان و دندان',
      'درمان مشکلات دندانی موجود (در صورت نیاز)',
    ],
    goals: ['ارزیابی کامل وضعیت بیمار', 'تهیه طرح درمان دقیق'],
  });

  // Phase 2: Active treatment based on issues
  if (skeletalIssues.length > 0) {
    const hasClassII = issues.some(
      (i) => i.description.includes('کلاس II') || i.parameter === 'ANB' && (i.value || 0) > 4
    );
    const hasClassIII = issues.some(
      (i) => i.description.includes('کلاس III') || i.parameter === 'ANB' && (i.value || 0) < 2
    );

    if (hasClassII) {
      plan.push({
        phase: 'مرحله ۲: درمان کلاس II اسکلتی',
        duration: '۱۲-۱۸ ماه',
        procedures: [
          'استفاده از دستگاه‌های فانکشنال (در صورت نیاز)',
          'مینی ایمپلانت برای کنترل لنف',
          'براکت‌گذاری با سیستم پیشرفته',
          'کشیدن دندان (در صورت نیاز به فضای اضافی)',
        ],
        goals: [
          'اصلاح رابطه اسکلتی',
          'بهبود موقعیت مندیبل',
          'اصلاح چیدمان دندانی',
        ],
      });
    } else if (hasClassIII) {
      plan.push({
        phase: 'مرحله ۲: درمان کلاس III اسکلتی',
        duration: '۱۸-۲۴ ماه',
        procedures: [
          'استفاده از دستگاه‌های فانکشنال پیش‌برنده',
          'مینی ایمپلانت برای کنترل عمودی',
          'براکت‌گذاری با سیستم پیشرفته',
          'جراحی ارتوگناتیک (در صورت نیاز)',
        ],
        goals: [
          'اصلاح رابطه اسکلتی',
          'بهبود موقعیت ماگزیلا و مندیبل',
          'اصلاح چیدمان دندانی',
        ],
      });
    } else {
      plan.push({
        phase: 'مرحله ۲: درمان ارتودنسی فعال',
        duration: '۱۲-۱۸ ماه',
        procedures: [
          'براکت‌گذاری با سیستم پیشرفته',
          'اصلاح موقعیت دندان‌ها',
          'بهبود چیدمان دندانی',
        ],
        goals: ['اصلاح چیدمان دندانی', 'بهبود عملکرد جویدن'],
      });
    }
  } else if (dentalIssues.length > 0) {
    plan.push({
      phase: 'مرحله ۲: درمان ارتودنسی فعال',
      duration: '۹-۱۲ ماه',
      procedures: [
        'براکت‌گذاری با سیستم پیشرفته',
        'اصلاح موقعیت دندان‌های قدامی',
        'بهبود چیدمان دندانی',
      ],
      goals: ['اصلاح موقعیت دندان‌ها', 'بهبود چیدمان دندانی'],
    });
  }

  // Phase 3: Finishing and detailing
  plan.push({
    phase: 'مرحله ۳: تکمیل و جزئیات',
    duration: '۶-۹ ماه',
    procedures: [
      'اصلاح جزئیات چیدمان دندانی',
      'بهبود روابط اکلوزال',
      'بهینه‌سازی زیبایی',
    ],
    goals: ['تکمیل درمان', 'بهینه‌سازی نتایج'],
  });

  // Phase 4: Retention
  plan.push({
    phase: 'مرحله ۴: نگهداری (Retention)',
    duration: '۲-۳ سال',
    procedures: [
      'استفاده از پلاک‌های ریتنینگ ثابت و متحرک',
      'ویزیت‌های پیگیری منظم',
      'کنترل بهداشت دهان و دندان',
    ],
    goals: ['حفظ نتایج درمان', 'پیشگیری از بازگشت'],
  });

  return plan;
}

/**
 * Generate comprehensive analysis report
 */
export function generateComprehensiveAnalysis(
  cephalometricMeasurements: CephalometricMeasurements,
  facialLandmarks?: FacialLandmarkAnalysis
): ComprehensiveAnalysis {
  // Analyze issues
  const issues = analyzeCephalometricMeasurements(cephalometricMeasurements);

  // Generate diagnosis
  const skeletalIssues = issues.filter((i) => i.type === 'skeletal');
  const dentalIssues = issues.filter((i) => i.type === 'dental');
  const severeIssues = issues.filter((i) => i.severity === 'severe');

  let diagnosis = '';
  if (skeletalIssues.length > 0 || dentalIssues.length > 0) {
    diagnosis = 'ناهنجاری اسکلتی و دندانی';
  } else {
    diagnosis = 'وضعیت نسبتاً نرمال';
  }

  // Generate treatment plan
  const treatmentPlan = generateTreatmentPlan(issues);

  // Generate recommendations (removed general recommendations)
  const recommendations: string[] = [];

  // Generate prognosis
  let prognosis = '';
  if (severeIssues.length > 0) {
    prognosis = 'پیش‌آگهی: درمان کامل ممکن است ۱۸-۲۴ ماه طول بکشد. نیاز به همکاری کامل بیمار و پیگیری منظم.';
  } else if (issues.length > 0) {
    prognosis = 'پیش‌آگهی: درمان کامل ممکن است ۱۲-۱۸ ماه طول بکشد. با همکاری بیمار نتایج مطلوب حاصل خواهد شد.';
  } else {
    prognosis = 'پیش‌آگهی: وضعیت نسبتاً نرمال. درمان ممکن است کوتاه‌مدت (۶-۹ ماه) باشد.';
  }

  return {
    issues,
    diagnosis,
    treatmentPlan,
    recommendations,
    prognosis,
  };
}

/**
 * Calculate measurements from landmarks
 */
export function calculateMeasurementsFromLandmarks(landmarks: Record<string, { x: number; y: number } | null>): CephalometricMeasurements {
  const measurements: CephalometricMeasurements = {};

  // Helper function to get landmark
  const getLandmark = (name: string) => {
    // Try exact match first
    if (landmarks[name]) return landmarks[name];
    
    // Try case-insensitive match
    const keys = Object.keys(landmarks);
    const found = keys.find(k => k.toLowerCase() === name.toLowerCase());
    if (found) return landmarks[found];
    
    // Try partial match
    const partialMatches: Record<string, string[]> = {
      'S': ['s', 'sella'],
      'N': ['n', 'nasion'],
      'A': ['a', 'subspinale'],
      'B': ['b', 'supramental'],
      'Or': ['or', 'orbitale'],
      'Po': ['po', 'porion'],
      'Go': ['go', 'gonion'],
      'Me': ['me', 'menton'],
      'Gn': ['gn', 'gnathion'],
      'Pog': ['pog', 'pogonion'],
      'U1': ['u1', 'upper_incisor', 'upper incisor', 'uit', 'upper incisor tip'],
      'U1A': ['u1a', 'uia', 'upper_incisor_apex', 'upper incisor apex'],
      'L1': ['l1', 'lower_incisor', 'lower incisor', 'lit', 'lower incisor tip'],
      'L1A': ['l1a', 'lia', 'lower_incisor_apex', 'lower incisor apex'],
    };
    
    if (partialMatches[name]) {
      for (const partial of partialMatches[name]) {
        const found = keys.find(k => k.toLowerCase().includes(partial) || partial.includes(k.toLowerCase()));
        if (found) return landmarks[found];
      }
    }
    
    return null;
  };

  // Helper function to calculate angle between three points
  const calculateAngle = (p1: { x: number; y: number }, vertex: { x: number; y: number }, p2: { x: number; y: number }): number => {
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
    let angle = Math.abs((angle1 - angle2) * (180 / Math.PI));
    if (angle > 180) angle = 360 - angle;
    return Math.round(angle * 10) / 10;
  };

  // Helper function to calculate line angle
  const calculateLineAngle = (p1: { x: number; y: number }, p2: { x: number; y: number }): number => {
    const angle = Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
    return angle;
  };

  // Helper function to calculate angle between two lines
  const calculateAngleBetweenLines = (
    line1Start: { x: number; y: number },
    line1End: { x: number; y: number },
    line2Start: { x: number; y: number },
    line2End: { x: number; y: number }
  ): number => {
    const v1x = line1End.x - line1Start.x;
    const v1y = line1End.y - line1Start.y;
    const v2x = line2End.x - line2Start.x;
    const v2y = line2End.y - line2Start.y;
    
    const dotProduct = v1x * v2x + v1y * v2y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    
    if (mag1 === 0 || mag2 === 0) return 0;
    
    const cosAngle = dotProduct / (mag1 * mag2);
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    let angle = Math.acos(clampedCos) * (180 / Math.PI);
    
    if (angle > 180) angle = 360 - angle;
    
    return Math.round(angle * 10) / 10;
  };

  // SNA - Angle between S-N-A
  const s = getLandmark('S');
  const n = getLandmark('N');
  const a = getLandmark('A');
  if (s && n && a) {
    measurements.SNA = calculateAngle(s, n, a);
  }

  // SNB - Angle between S-N-B
  const b = getLandmark('B');
  if (s && n && b) {
    measurements.SNB = calculateAngle(s, n, b);
  }

  // ANB - Difference between SNA and SNB
  if (measurements.SNA !== undefined && measurements.SNB !== undefined) {
    measurements.ANB = Math.round((measurements.SNA - measurements.SNB) * 10) / 10;
  }

  // FMA - Angle between Frankfort plane (Or-Po) and Mandibular plane (Go-Me)
  const or = getLandmark('Or');
  const po = getLandmark('Po');
  const go = getLandmark('Go');
  const me = getLandmark('Me');
  if (or && po && go && me) {
    measurements.FMA = calculateAngleBetweenLines(or, po, go, me);
  }

  // IMPA - Angle between Lower incisor (L1) and Mandibular plane (Go-Me)
  const l1 = getLandmark('L1');
  if (l1 && go && me) {
    // IMPA is angle between L1-Me line and Go-Me line
    measurements.IMPA = calculateAngleBetweenLines(go, me, l1, me);
  }

  // FMIA - Angle between Frankfort plane (Or-Po) and Lower incisor (L1-Me)
  if (or && po && l1 && me) {
    measurements.FMIA = calculateAngleBetweenLines(or, po, l1, me);
  }

  // U1-SN - Angle between Upper incisor line (U1A-U1) and SN line
  // U1A = Upper Incisor Apex, U1 = Upper Incisor Tip
  // Try to get U1A first, then U1
  const u1a = getLandmark('U1A') || getLandmark('UIA');
  const u1 = getLandmark('U1') || getLandmark('UIT');
  
  if (u1a && u1 && s && n) {
    // Calculate angle of incisor line (U1A-U1) and SN line (S-N), then find the difference
    // U1A should be the apex (root), U1 should be the tip (crown)
    const u1Angle = calculateLineAngle(u1a, u1);
    const snAngle = calculateLineAngle(s, n);
    let angleDiff = Math.abs(u1Angle - snAngle);
    if (angleDiff > 180) angleDiff = 360 - angleDiff;
    // عدد بدست آمده باید از 180 کم شود
    measurements['U1-SN'] = Math.round(Math.max(0, Math.min(180, 180 - angleDiff)) * 10) / 10;
  } else if (u1 && s && n) {
    // Fallback: if U1A not available, use U1 point with N
    const u1Angle = calculateLineAngle(u1, n);
    const snAngle = calculateLineAngle(s, n);
    let angleDiff = Math.abs(u1Angle - snAngle);
    if (angleDiff > 180) angleDiff = 360 - angleDiff;
    // عدد بدست آمده باید از 180 کم شود
    measurements['U1-SN'] = Math.round(Math.max(0, Math.min(180, 180 - angleDiff)) * 10) / 10;
  }

  // L1-MP - Angle between Lower incisor line (L1A-L1) and Mandibular plane (Go-Me)
  // L1A = Lower Incisor Apex, L1 = Lower Incisor Tip
  // Note: l1 is already defined above (line 446), so we reuse it here
  const l1a = getLandmark('L1A');
  if (l1a && l1 && go && me) {
    // Calculate angle between incisor line (L1A-L1) and mandibular plane (Go-Me)
    const calculatedAngle = calculateAngleBetweenLines(go, me, l1a, l1);
    // عدد بدست آمده باید از 180 کم شود
    measurements['L1-MP'] = Math.round(Math.max(0, Math.min(180, 180 - calculatedAngle)) * 10) / 10;
  } else if (measurements.IMPA !== undefined) {
    // Fallback: use IMPA if L1A not available
    measurements['L1-MP'] = measurements.IMPA;
  }

  // GoGnSN - Angle between Go-Gn line and SN line
  const gn = getLandmark('Gn');
  if (go && gn && s && n) {
    const goGnAngle = calculateLineAngle(go, gn);
    const snAngle = calculateLineAngle(s, n);
    measurements.GoGnSN = Math.abs(snAngle - goGnAngle);
  }

  return measurements;
}

/**
 * Generate mechanotherapy recommendations based on issues
 */
export function generateMechanotherapy(issues: OrthodonticIssue[]): string[] {
  const appliances: string[] = [];
  
  const hasClassII = issues.some(
    (i) => i.description.includes('کلاس II') || i.parameter === 'ANB' && (i.value || 0) > 4
  );
  const hasClassIII = issues.some(
    (i) => i.description.includes('کلاس III') || i.parameter === 'ANB' && (i.value || 0) < 2
  );
  const hasMaxillaryRetrusion = issues.some((i) => i.description.includes('ماگزیلا') && i.description.includes('خلفی'));
  const hasMandibularRetrusion = issues.some((i) => i.description.includes('مندیبل') && i.description.includes('خلفی'));
  const hasVerticalGrowth = issues.some((i) => i.description.includes('عمودی') || i.description.includes('باز بودن'));
  const hasRetroclined = issues.some((i) => i.description.includes('Retroclined') || i.description.includes('عقب'));
  
  if (hasClassII || hasMandibularRetrusion) {
    appliances.push('فیس ماسک');
    appliances.push('لبیال بو');
    appliances.push('زد اسپرینگ');
  }
  
  if (hasClassIII || hasMaxillaryRetrusion) {
    appliances.push('چین کپ');
    appliances.push('رپید پالاتال اکسپندر');
  }
  
  if (hasVerticalGrowth) {
    appliances.push('هایپولای');
    appliances.push('بایت پلیت');
  }
  
  if (hasRetroclined) {
    appliances.push('اسپرینگ رترکشن');
  }
  
  // Always include basic appliances
  if (appliances.length === 0) {
    appliances.push('براکت ثابت');
  }
  
  return appliances;
}

/**
 * Format analysis for display
 */
export function formatAnalysisForDisplay(analysis: ComprehensiveAnalysis): string {
  let report = ``;

  // Diagnosis
  report += `تشخیص: ${analysis.diagnosis}\n\n`;

  // Issues
  if (analysis.issues.length > 0) {
    report += `مشکلات شناسایی شده:\n`;
    analysis.issues.forEach((issue, index) => {
      report += `${index + 1}. ${issue.description}`;
      if (issue.parameter && issue.value !== undefined) {
        report += ` - ${issue.parameter}: ${issue.value.toFixed(1)}°`;
      }
      report += `\n`;
    });
    report += `\n`;
  } else {
    report += `وضعیت: پارامترهای اصلی در محدوده نرمال قرار دارند.\n\n`;
  }

  // Treatment plan
  if (analysis.treatmentPlan.length > 0) {
    report += `طرح درمان\n`;
    analysis.treatmentPlan.forEach((phase, index) => {
      report += `${phase.phase} - مدت زمان: ${phase.duration}\n`;
      if (phase.goals.length > 0) {
        report += `اهداف: ${phase.goals.join('، ')}\n`;
      }
      if (phase.procedures.length > 0) {
        report += `روش‌های درمانی: ${phase.procedures.join('، ')}\n`;
      }
      report += `\n`;
    });
  }

  // Mechanotherapy
  const mechanotherapy = generateMechanotherapy(analysis.issues);
  if (mechanotherapy.length > 0) {
    report += `مکانوتراپی\n`;
    mechanotherapy.forEach((appliance, index) => {
      report += `${index + 1}. ${appliance}\n`;
    });
  }

  return report;
}

