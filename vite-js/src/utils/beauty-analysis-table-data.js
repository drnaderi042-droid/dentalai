/**
 * داده‌های جدول آنالیز زیبایی صورت
 * شامل مقادیر ایده‌آل و توضیحات برای هر پارامتر
 */

export const beautyAnalysisTableData = [
  // Macro-Esthetics
  {
    category: 'Macro-Esthetics',
    parameter: 'نسبت عرض به ارتفاع صورت',
    idealValue: '0.70 - 0.75',
    description: 'نسبت ایده‌آل عرض صورت به ارتفاع آن. نسبت‌های بالاتر نشان‌دهنده صورت عریض‌تر است.',
    getValue: (analysis) => analysis.facialProportions?.widthToHeightRatio,
    getScore: (analysis) => analysis.facialProportions?.widthToHeightScore,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'نسبت Glabella به Nose Tip (از Glabella)',
    idealValue: '50%',
    description: 'نسبت ارتفاع از Glabella تا نوک بینی به ارتفاع کل از Glabella تا چانه. چون لندمارک بالای پیشانی در دسترس نیست، از Glabella به عنوان نقطه شروع استفاده می‌شود.',
    getValue: (analysis) => analysis.facialProportions?.glabellaToNoseRatio ? `${analysis.facialProportions.glabellaToNoseRatio}%` : null,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'نسبت Nose Tip به Chin (از Glabella)',
    idealValue: '50%',
    description: 'نسبت ارتفاع از نوک بینی تا چانه به ارتفاع کل از Glabella تا چانه.',
    getValue: (analysis) => analysis.facialProportions?.noseToChinRatio ? `${analysis.facialProportions.noseToChinRatio}%` : null,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'انحراف بینی',
    idealValue: '< 2%',
    description: 'انحراف بینی از خط مرکزی صورت. انحراف بیش از 5% نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.facialAsymmetry?.noseDeviationPercent ? `${analysis.facialAsymmetry.noseDeviationPercent}%` : null,
    getGrade: (analysis) => analysis.facialAsymmetry?.noseDeviationGrade,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'انحراف چانه',
    idealValue: '< 2%',
    description: 'انحراف چانه از خط مرکزی صورت. انحراف زیاد ممکن است نشان‌دهنده عدم تقارن باشد.',
    getValue: (analysis) => analysis.facialAsymmetry?.chinDeviationPercent ? `${analysis.facialAsymmetry.chinDeviationPercent}%` : null,
    getGrade: (analysis) => analysis.facialAsymmetry?.chinDeviationGrade,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'نسبت MFH/LFH (Middle/Lower Face Height)',
    idealValue: '0.60 - 0.70',
    description: 'نسبت ارتفاع صورت میانی (از نوک بینی تا بالای دهان) به ارتفاع صورت پایینی (از بالای دهان تا چانه). این نسبت به جای LFH/UFH استفاده می‌شود چون لندمارک بالای پیشانی در دسترس نیست.',
    getValue: (analysis) => analysis.faceHeight?.mfhToLfhRatio,
    getScore: (analysis) => analysis.faceHeight?.mfhToLfhScore,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'نوع پروفایل',
    idealValue: 'Straight',
    description: 'پروفایل مستقیم ایده‌آل است. پروفایل محدب (Convex) نشان‌دهنده پیش آمدگی و پروفایل مقعر (Concave) نشان‌دهنده پس رفتگی است.',
    getValue: (analysis) => analysis.profile?.profileDescription,
    getGrade: (analysis) => analysis.profile?.profileType === 'Straight' ? 'Ideal' : analysis.profile?.profileType,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'برجستگی لب',
    idealValue: 'Normal',
    description: 'برجستگی طبیعی لب ایده‌آل است. Protrusion (پیش آمدگی) یا Retrusion (پس رفتگی) نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.profile?.lipProminenceDescription,
    getGrade: (analysis) => analysis.profile?.lipProminenceType,
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'عدم بسته شدن لب (Incompetence)',
    idealValue: 'No',
    description: 'در حالت استراحت، لب‌ها باید به هم برسند. عدم بسته شدن ممکن است نیاز به درمان داشته باشد.',
    getValue: (analysis) => analysis.lipDetails?.lipIncompetence,
    getGrade: (analysis) => analysis.lipDetails?.lipIncompetence === 'No' ? 'Normal' : 'Abnormal',
  },
  {
    category: 'Macro-Esthetics',
    parameter: 'تقارن چین نازولابیال',
    idealValue: 'Symmetric',
    description: 'چین نازولابیال باید متقارن باشد. عدم تقارن ممکن است نشان‌دهنده مشکلات ساختاری باشد.',
    getValue: (analysis) => analysis.nasolabialFold?.nasolabialSymmetry,
  },
  {
    // Mini-Esthetics
    category: 'Mini-Esthetics',
    parameter: 'نمایش دندان پیشین در استراحت',
    idealValue: 'Normal',
    description: 'در حالت استراحت، مقدار کمی از دندان‌های پیشین بالا باید قابل مشاهده باشد (2-4mm).',
    getValue: (analysis) => analysis.incisorShow?.restIncisorShow,
  },
  {
    category: 'Mini-Esthetics',
    parameter: 'نمایش لثه',
    idealValue: 'Normal',
    description: 'نمایش بیش از حد لثه (Gummy Smile) ممکن است نیاز به درمان داشته باشد.',
    getValue: (analysis) => analysis.gingivalShow?.gingivalShow,
  },
  {
    category: 'Mini-Esthetics',
    parameter: 'قوس لبخند',
    idealValue: 'Consonant',
    description: 'قوس هماهنگ (Consonant) ایده‌آل است. قوس صاف (Flat) یا معکوس (Reverse) ممکن است نیاز به درمان داشته باشد.',
    getValue: (analysis) => analysis.smileArc?.smileArcDescription,
    getGrade: (analysis) => analysis.smileArc?.smileArcType === 'Consonant' ? 'Ideal' : analysis.smileArc?.smileArcType,
  },
  {
    category: 'Mini-Esthetics',
    parameter: 'راهروهای گونه‌ای (Buccal Corridors)',
    idealValue: '10-15%',
    description: 'فاصله بین گوشه دهان و کنار صورت در حالت لبخند. راهروهای باریک (کمتر از 10%) یا عریض (بیش از 15%) ایده‌آل نیستند.',
    getValue: (analysis) => analysis.smileWidth?.buccalCorridorRatio ? `${analysis.smileWidth.buccalCorridorRatio}%` : null,
    getGrade: (analysis) => analysis.smileWidth?.buccalCorridorType === 'Ideal' ? 'Ideal' : analysis.smileWidth?.buccalCorridorType,
  },
  // Basic Analysis
  {
    category: 'Basic Analysis',
    parameter: 'تقارن کلی صورت',
    idealValue: '> 90%',
    description: 'تقارن کلی صورت باید بالای 90% باشد. تقارن پایین‌تر ممکن است نیاز به درمان داشته باشد.',
    getValue: (analysis) => analysis.symmetry?.overall ? `${analysis.symmetry.overall.toFixed(1)}%` : null,
  },
  {
    category: 'Basic Analysis',
    parameter: 'تقارن چشم‌ها',
    idealValue: '> 90%',
    description: 'چشم‌ها باید متقارن باشند. عدم تقارن ممکن است نشان‌دهنده مشکلات ساختاری باشد.',
    getValue: (analysis) => analysis.symmetry?.eyes ? `${analysis.symmetry.eyes.toFixed(1)}%` : null,
  },
  {
    category: 'Basic Analysis',
    parameter: 'نسبت طلایی عمودی',
    idealValue: '1.618',
    description: 'نسبت طلایی (Golden Ratio) برای تقسیم‌بندی عمودی صورت. نسبت 1:1.618 ایده‌آل است.',
    getValue: (analysis) => analysis.goldenRatio?.verticalRatio?.ratio,
    getScore: (analysis) => analysis.goldenRatio?.verticalRatio?.score,
  },
  {
    category: 'Basic Analysis',
    parameter: 'نسبت طلایی افقی',
    idealValue: '1.618',
    description: 'نسبت طلایی برای تقسیم‌بندی افقی صورت.',
    getValue: (analysis) => analysis.goldenRatio?.horizontalRatio?.ratio,
    getScore: (analysis) => analysis.goldenRatio?.horizontalRatio?.score,
  },
  {
    category: 'Basic Analysis',
    parameter: 'نسبت بینی',
    idealValue: '1.5 - 2.0',
    description: 'نسبت ارتفاع به عرض بینی. نسبت 1.5-2.0 ایده‌آل است.',
    getValue: (analysis) => analysis.nose?.noseRatio,
    getScore: (analysis) => analysis.nose?.noseRatioScore,
  },
  // Frontal Analysis Lines
  {
    category: 'Frontal Analysis',
    parameter: 'انحراف خط میانی صورت (Midfacial Line)',
    idealValue: '≤ 2%',
    description: 'خط میانی صورت باید از Glabella → Subnasale → Pogonion عبور کند. انحراف بیش از 2% نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.midfacialLineDeviation?.deviationPercent ? `${analysis.midfacialLineDeviation.deviationPercent}%` : null,
    getScore: (analysis) => analysis.midfacialLineDeviation?.score,
    getGrade: (analysis) => analysis.midfacialLineDeviation?.grade,
  },
  {
    category: 'Frontal Analysis',
    parameter: 'زاویه خط بین مردمک‌ها (Interpupillary Line)',
    idealValue: '0° (افقی)',
    description: 'خط بین مردمک‌ها باید کاملاً افقی و موازی با زمین باشد. انحراف بیش از 3 درجه نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.interpupillaryLineAngle?.angleDeviation ? `${analysis.interpupillaryLineAngle.angleDeviation}°` : null,
    getScore: (analysis) => analysis.interpupillaryLineAngle?.score,
    getGrade: (analysis) => analysis.interpupillaryLineAngle?.grade,
  },
  {
    category: 'Frontal Analysis',
    parameter: 'زاویه خط دهان (Commissural Line)',
    idealValue: 'موازی با Interpupillary Line',
    description: 'خط بین گوشه‌های دهان باید موازی با خط بین مردمک‌ها باشد. تفاوت بیش از 3 درجه نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.commissuralLineAngle?.angleDifference ? `${analysis.commissuralLineAngle.angleDifference}°` : null,
    getScore: (analysis) => analysis.commissuralLineAngle?.score,
    getGrade: (analysis) => analysis.commissuralLineAngle?.grade,
  },
  {
    category: 'Frontal Analysis',
    parameter: 'انحراف خط دندانی از خط میانی صورت',
    idealValue: '≤ 2%',
    description: 'خط دندانی باید با خط میانی صورت منطبق باشد. انحراف بیش از 2% نیاز به بررسی دارد.',
    getValue: (analysis) => analysis.dentalMidlineDeviation?.deviationPercent ? `${analysis.dentalMidlineDeviation.deviationPercent}%` : null,
    getScore: (analysis) => analysis.dentalMidlineDeviation?.score,
    getGrade: (analysis) => analysis.dentalMidlineDeviation?.grade,
  },
  // Vertical Proportions
  {
    category: 'Vertical Proportions',
    parameter: 'نسبت بخش میانی به پایینی صورت (Glabella–Subnasale : Subnasale–Menton)',
    idealValue: '1:1 (50% : 50%)',
    description: 'نسبت بخش میانی صورت (از Glabella تا Subnasale) به بخش پایینی (از Subnasale تا Menton) باید تقریباً برابر باشد. اگر اختلاف زیادی وجود داشته باشد، نشان‌دهنده عدم تعادل در نسبت‌های صورت است.',
    getValue: (analysis) => {
      if (analysis.verticalProportions?.middleToLowerRatio) {
        const ratio = parseFloat(analysis.verticalProportions.middleToLowerRatio);
        return `${ratio.toFixed(2)}:1`;
      }
      if (analysis.verticalProportions?.middleThirdRatio && analysis.verticalProportions?.lowerThirdRatio) {
        return `${analysis.verticalProportions.middleThirdRatio}% : ${analysis.verticalProportions.lowerThirdRatio}%`;
      }
      return null;
    },
    getScore: (analysis) => analysis.verticalProportions?.score,
    getGrade: (analysis) => analysis.verticalProportions?.grade,
  },
  {
    category: 'Vertical Proportions',
    parameter: 'نسبت بخش میانی صورت (Glabella–Subnasale)',
    idealValue: '50%',
    description: 'بخش میانی صورت باید تقریباً 50% از ارتفاع کل (از Glabella تا Menton) باشد.',
    getValue: (analysis) => analysis.verticalProportions?.middleThirdRatio ? `${analysis.verticalProportions.middleThirdRatio}%` : null,
    getScore: (analysis) => analysis.verticalProportions?.score,
  },
  {
    category: 'Vertical Proportions',
    parameter: 'نسبت بخش پایینی صورت (Subnasale–Menton)',
    idealValue: '50%',
    description: 'بخش پایینی صورت باید تقریباً 50% از ارتفاع کل (از Glabella تا Menton) باشد.',
    getValue: (analysis) => analysis.verticalProportions?.lowerThirdRatio ? `${analysis.verticalProportions.lowerThirdRatio}%` : null,
    getScore: (analysis) => analysis.verticalProportions?.score,
  },
  {
    category: 'Vertical Proportions',
    parameter: 'نسبت لب بالا به بخش پایینی صورت',
    idealValue: '33.3%',
    description: 'لب بالا باید یک سوم بخش پایینی صورت باشد (نسبت 1:2 با لب پایین + چانه).',
    getValue: (analysis) => analysis.verticalProportions?.upperLipRatio ? `${analysis.verticalProportions.upperLipRatio}%` : null,
    getScore: (analysis) => analysis.verticalProportions?.lowerFaceScore,
  },
  {
    category: 'Vertical Proportions',
    parameter: 'نسبت لب پایین + چانه به بخش پایینی صورت',
    idealValue: '66.7%',
    description: 'لب پایین + چانه باید دو سوم بخش پایینی صورت باشد.',
    getValue: (analysis) => analysis.verticalProportions?.lowerLipChinRatio ? `${analysis.verticalProportions.lowerLipChinRatio}%` : null,
  },
  // Facial Index
  {
    category: 'Facial Index',
    parameter: 'نسبت قدامی-تحتانی (Facial Index)',
    idealValue: '1.3 ± 0.1',
    description: 'نسبت ارتفاع صورت به عرض صورت. نرمال: 1.3 ± 0.1. بالاتر → صورت کشیده (Long Face)، پایین‌تر → صورت پهن (Short Face).',
    getValue: (analysis) => analysis.facialIndex?.facialIndex,
    getScore: (analysis) => analysis.facialIndex?.score,
    getGrade: (analysis) => analysis.facialIndex?.grade,
  },
  // Transverse Proportions
  {
    category: 'Transverse Proportions',
    parameter: 'نسبت عرض بینی به فاصله بین چشم‌ها',
    idealValue: '≈ 1:1',
    description: 'عرض بینی (Alar width) باید تقریباً برابر با فاصله بین گوشه داخلی چشم‌ها (Intercanthal distance) باشد.',
    getValue: (analysis) => analysis.transverseProportions?.alarToIntercanthalRatio,
    getScore: (analysis) => analysis.transverseProportions?.alarToIntercanthalScore,
  },
  {
    category: 'Transverse Proportions',
    parameter: 'نسبت عرض دهان به فاصله بین مردمک‌ها',
    idealValue: '≈ 1:1',
    description: 'عرض دهان باید تقریباً برابر با فاصله بین مردمک‌ها باشد.',
    getValue: (analysis) => analysis.transverseProportions?.mouthToPupillaryRatio,
    getScore: (analysis) => analysis.transverseProportions?.mouthToPupillaryScore,
  },
  {
    category: 'Transverse Proportions',
    parameter: 'نسبت عرض دهان به عرض بینی',
    idealValue: '1.3',
    description: 'عرض دهان باید تقریباً 1.3 برابر عرض بینی باشد.',
    getValue: (analysis) => analysis.transverseProportions?.noseToMouthWidthRatio,
    getScore: (analysis) => analysis.transverseProportions?.noseToMouthWidthScore,
  },
  // Lip Proportions
  {
    category: 'Lip Proportions',
    parameter: 'طول لب بالا (Sn–Stm)',
    idealValue: '20-22 mm',
    description: 'طول لب بالا باید بین 20 تا 22 میلی‌متر باشد.',
    getValue: (analysis) => analysis.lipProportions?.upperLipLengthMM ? `${analysis.lipProportions.upperLipLengthMM} mm` : null,
    getScore: (analysis) => analysis.lipProportions?.upperLipLengthScore,
  },
  {
    category: 'Lip Proportions',
    parameter: 'طول لب پایین + چانه (Stm–Me)',
    idealValue: '40-44 mm',
    description: 'طول لب پایین + چانه باید بین 40 تا 44 میلی‌متر باشد.',
    getValue: (analysis) => analysis.lipProportions?.lowerLipChinLengthMM ? `${analysis.lipProportions.lowerLipChinLengthMM} mm` : null,
    getScore: (analysis) => analysis.lipProportions?.lowerLipChinLengthScore,
  },
  {
    category: 'Lip Proportions',
    parameter: 'نسبت لب پایین به لب بالا',
    idealValue: '2.0 (1:2)',
    description: 'نسبت طول لب پایین به لب بالا باید 2:1 باشد (لب پایین دو برابر لب بالا).',
    getValue: (analysis) => analysis.lipProportions?.upperToLowerLipRatio,
    getScore: (analysis) => analysis.lipProportions?.upperToLowerLipScore,
  },
];




