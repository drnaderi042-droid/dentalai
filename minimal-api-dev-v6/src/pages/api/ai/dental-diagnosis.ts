import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

import { prisma } from 'src/lib/prisma';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// ----------------------------------------------------------------------

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

    // Verify authentication
    const { authorization } = req.headers;
    if (!authorization) {
      return res.status(401).json({ message: 'Authorization token missing' });
    }

    const accessToken = `${authorization}`.split(' ')[1];
    const data = verify(accessToken, JWT_SECRET) as { userId: string };

    if (!data.userId) {
      return res.status(401).json({ message: 'Invalid authorization token' });
    }

    // Get user info
    const user = await prisma.user.findUnique({
      where: { id: data.userId },
    });

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    const { images, patientInfo, aiModel } = req.body;

    if (!images || !Array.isArray(images) || images.length === 0) {
      return res.status(400).json({ message: 'At least one image is required' });
    }

    console.log(`[AI Diagnosis] Processing ${images.length} images for patient: ${patientInfo?.name || 'Unknown'}`);
    console.log(`[AI Diagnosis] Selected AI Model: ${aiModel || 'default (gemini-flash-1.5)'}`);

    try {
      console.log(`[AI Diagnosis] Starting analysis for type: ${patientInfo?.analysisType || 'general'}`);

      // Use OpenRouter API for AI analysis
      const aiDiagnosis = await generateAIDiagnosis(images, patientInfo, aiModel);

      console.log('[AI Diagnosis] AI Response received successfully:', {
        analysisType: patientInfo?.analysisType,
        hasDiagnosis: !!aiDiagnosis.diagnosis,
        hasCeph: !!aiDiagnosis.cephalometricAnalysis,
        diagnosisLength: aiDiagnosis.diagnosis?.length || 0,
        cephLength: aiDiagnosis.cephalometricAnalysis?.length || 0,
        responsePreview: `${aiDiagnosis.diagnosis?.substring(0, 100)  }...`
      });

      return res.status(200).json(aiDiagnosis);
    } catch (aiError) {
      console.error('[AI Diagnosis] AI call failed with error:', aiError);
      console.error('[AI Diagnosis] Error details:', {
        message: aiError.message,
        status: aiError.status,
        response: aiError.response?.data
      });

      // Return error to client - NO MOCK FALLBACK
      return res.status(500).json({
        message: 'خطا در تحلیل AI',
        error: aiError.message || 'Unknown error',
        details: 'لطفاً دوباره تلاش کنید یا مدل دیگری انتخاب کنید.'
      });
    }

  } catch (error) {
    console.error('[AI Dental Diagnosis]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  }
  // MEMORY OPTIMIZATION: Don't disconnect prisma - use singleton connection pool
}

// Helper function to convert image URL to base64
async function imageUrlToBase64(url: string): Promise<string> {
  try {
    // If it's a localhost URL, read the file directly
    if (url.includes('localhost') || url.includes('127.0.0.1')) {
      const fs = await import('fs/promises');
      const path = await import('path');
      
      // Extract the file path from the URL
      const urlPath = url.split('localhost:7272')[1] || url.split('127.0.0.1:7272')[1];
      const filePath = path.join(process.cwd(), urlPath);
      
      console.log(`[AI Diagnosis] Reading image from: ${filePath}`);
      
      const fileBuffer = await fs.readFile(filePath);
      const base64 = fileBuffer.toString('base64');
      
      // Determine MIME type based on file extension
      const ext = path.extname(filePath).toLowerCase();
      const mimeTypes: { [key: string]: string } = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.bmp': 'image/bmp',
        '.tiff': 'image/tiff',
        '.tif': 'image/tiff',
      };
      const mimeType = mimeTypes[ext] || 'image/jpeg';
      
      return `data:${mimeType};base64,${base64}`;
    }
    
    // For external URLs, fetch and convert
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);
    const base64 = buffer.toString('base64');
    
    const contentType = response.headers.get('content-type') || 'image/jpeg';
    return `data:${contentType};base64,${base64}`;
  } catch (error) {
    console.error(`[AI Diagnosis] Error converting image to base64:`, error);
    throw new Error(`Failed to process image: ${url}`);
  }
}

// AI diagnosis function using OpenRouter API with Gemini 1.5 Flash
async function generateAIDiagnosis(images: string[], patientInfo: any, selectedModel?: string) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OpenRouter API key not configured');
  }

  const isCephalometricAnalysis = patientInfo?.analysisType === 'cephalometric';
  const analysisType = patientInfo?.analysisType || 'general';
  
  // Map friendly model names to actual OpenRouter model IDs
  // Using correct OpenRouter model names (check https://openrouter.ai/models)
  const modelMapping: Record<string, string> = {
    'cephx-v1': 'google/gemini-flash-1.5-8b', // Correct Gemini model name
    'cephx-v2': 'anthropic/claude-3.5-sonnet:beta', // Claude 3.5 Sonnet
    'deepceph': 'anthropic/claude-3-opus:beta', // Claude 3 Opus
    'gpt-4o-vision': 'openai/gpt-4o', // GPT-4o
    'gpt-4o': 'openai/gpt-4o', // GPT-4o
    'claude-3.5': 'anthropic/claude-3.5-sonnet:beta', // Claude 3.5 Sonnet
    'claude-vision': 'anthropic/claude-3.5-sonnet:beta',
    'gemini-flash': 'google/gemini-flash-1.5-8b',
    'gemini-pro': 'google/gemini-pro-1.5',
    'local': 'google/gemini-flash-1.5-8b', // Fallback to Gemini for local
  };
  
  // Use selected model or default to Claude 3.5 Sonnet for cephalometric analysis
  const defaultModel = isCephalometricAnalysis ? 'anthropic/claude-3.5-sonnet:beta' : 'google/gemini-flash-1.5-8b';
  const requestedModel = selectedModel || process.env.AI_MODEL || defaultModel;
  const modelToUse = modelMapping[requestedModel] || requestedModel;
  
  console.log(`[AI Diagnosis] Analysis Type: ${analysisType}`);
  console.log(`[AI Diagnosis] Requested model: ${requestedModel}`);
  console.log(`[AI Diagnosis] Using OpenRouter model: ${modelToUse}`);

  // MEMORY OPTIMIZATION: Limit number of images and process sequentially for very large images
  // Convert image URLs to base64 with limit
  const maxImages = Math.min(images.length, 5); // Limit to 5 images max
  console.log(`[AI Diagnosis] Converting ${maxImages} images to base64 (limited for memory optimization)...`);
  
  // Process images sequentially to avoid memory spikes
  const base64Images: string[] = [];
  for (let i = 0; i < maxImages; i++) {
    try {
      const base64 = await imageUrlToBase64(images[i]);
      base64Images.push(base64);
      // Allow garbage collection between images
      if (i < maxImages - 1 && global.gc) {
        global.gc();
      }
    } catch (error) {
      console.error(`[AI Diagnosis] Failed to convert image ${i + 1}:`, error);
      // Continue with other images
    }
  }
  
  console.log(`[AI Diagnosis] Successfully converted ${base64Images.length} images to base64`);
  console.log('[AI Diagnosis] Image data samples:', base64Images.map((img, i) => ({
    index: i,
    preview: `${img.substring(0, 50)  }...`,
    length: img.length,
    hasDataPrefix: img.startsWith('data:'),
    hasBase64: img.includes('base64')
  })));

  let prompt;
  if (isCephalometricAnalysis) {
    prompt = `🔍 CRITICAL INSTRUCTION: You MUST analyze the actual cephalometric radiograph image provided. DO NOT use example or mock coordinates!

You are an expert orthodontist analyzing a lateral cephalometric radiograph. You MUST:
1. **LOOK at the actual image** - examine the skull structure carefully
2. **IDENTIFY anatomical landmarks** - locate each landmark on the image
3. **MEASURE pixel coordinates** - provide REAL x,y coordinates based on the actual image dimensions
4. **CALCULATE angles** - compute actual cephalometric measurements

شما یک متخصص ارتودنسی با تجربه می‌باشید و باید تصویر lateral cephalometry واقعی که ارائه شده را تحلیل کنید. هرگز از مقادیر نمونه یا فرضی استفاده نکنید!

**مشخصات بیمار:**
- نام: ${patientInfo?.name || 'نامشخص'}
- سن: ${patientInfo?.age || 'مشخص نشده'}
- تاریخ: ${new Date().toLocaleDateString('fa-IR')}
- تحلیلگر: متخصص AI واقعیت افزوده ارتو دونتیکس

**🚨 نقطه عطف بسیار مهم:** شما باید مختصات دقیق پیکسلی هر لندمارک را در تصویر مشخص کنید. تصویر دارای اندازه معینی است که باید بر اساس آن اندازه‌گیری کنید:

**لیست اصلی لندمارک‌ها با مختصات پیکسلی:**

## 🎯 **لندمارک‌های استخوانی (Hard Tissue Landmarks)**
1. **Sella (S)**: مرکز سلا ترخرمیک
2. **Nasion (N)**: اتصال فرونتونیسال
3. **Point A (A)**: عمیق‌ترین نقطه آلوئولار ماکسیلا
4. **Point B (B)**: عمیق‌ترین نقطه آلوئولار مندیبولار
5. **Pogonion (Pg)**: نقطه قدامی‌ترین چانه
6. **Gnathion (Gn)**: نقطه قدامی-تحتانی‌ترین چانه
7. **Menton (Me)**: نقطه تحتانی‌ترین سفیفیس مندیبولار
8. **Gonion (Go)**: زاویه پسترو-تحتانی مندیبولار
9. **Porion (Po)**: نقطه برتر کانال اکستنال آودیو
10. **Orbitale (Or)**: پست‌ترین نقطه مارجین اوربیتال
11. **Anterior Nasal Spine (ANS)**: نوک ESP قدامی نازال
12. **Posterior Nasal Spine (PNS)**: نوک ESP پستیور نازال

## 💄 **لندمارک‌های بافت نرم (Soft Tissue Landmarks)**
13. **Subnasale (Sn)**: نقطه اتصال کولوملا و لب بالا
14. **Labrale Superius (Ls)**: نقطه قدامی‌ترین لب بالا
15. **Labrale Inferius (Li)**: نقطه قدامی‌ترین لب پایین
16. **Supramentale (Sm)**: نقطه قوس‌مانند CONVEX مندیبولار

**📏 اندازه‌گیری مختصات:**
- ارائه مختصات هر لندمارک به صورت (x,y) بر اساس پس‌زمینه تصویر
- مختصات x از سمت چپ، y از سمت بالا
- تصویر دارای حاشیه و اندازه مشخص

**🎯 خروجی مورد انتظار:**
باید پاسخ خود را در دو بخش ارائه دهید:

**SECTION 1: CEHALOMETRIC_MEASUREMENTS_JSON**

🚨 **CRITICAL: You MUST analyze the ACTUAL radiograph image! Step-by-step:**
1. First, confirm you can see the skull/cephalometric image
2. Identify the orientation (left profile, right profile)
3. Locate each anatomical landmark visually in the image
4. Measure the pixel position of each landmark
5. Provide REAL coordinates, NOT example values!

این بخش باید یک شیء JSON معتبر باشد که مختصات واقعی از تصویر را نشان دهد:

{
  "landmarks": {
    "S": {"x": [number], "y": [number]},
    "N": {"x": [number], "y": [number]},
    "A": {"x": [number], "y": [number]},
    "B": {"x": [number], "y": [number]},
    "Pg": {"x": [number], "y": [number]},
    "Gn": {"x": [number], "y": [number]},
    "Me": {"x": [number], "y": [number]},
    "Go": {"x": [number], "y": [number]},
    "ANS": {"x": [number], "y": [number]},
    "PNS": {"x": [number], "y": [number]},
    "Po": {"x": [number], "y": [number]},
    "Or": {"x": [number], "y": [number]},
    "Sn": {"x": [number], "y": [number]},
    "Ls": {"x": [number], "y": [number]},
    "Li": {"x": [number], "y": [number]},
    "Sm": {"x": [number], "y": [number]}
  },
  "measurements": {
    "SNA": "82.5°",
    "SNB": "79.2°",
    "ANB": "3.3°",
    "FMA": "27.8°",
    "MMPA": "29.5°"
  }
}

📐 **COORDINATE EXTRACTION PROCESS (MANDATORY):**
1. **IMAGE DIMENSIONS**: First state the image dimensions you see (e.g., "Image is 1200x900 pixels")
2. **SKULL IDENTIFICATION**: Describe what you see ("I see a lateral cephalometric radiograph showing...")
3. **LANDMARK LOCATION**: For each landmark, describe its visual location BEFORE giving coordinates
4. **COORDINATE SYSTEM**: 
   - x = 0 at LEFT edge, increases rightward
   - y = 0 at TOP edge, increases downward
   - Coordinates must match actual image dimensions
5. **VERIFICATION**: Coordinates should be DIFFERENT from example values (not 400, 150, 480, etc.)
6. If you CANNOT see or analyze the image, explicitly state: "I cannot process this image"

⚠️ **واجب است:** قبل از ارائه JSON، ابعاد تصویر و توضیح مختصری از آنچه در تصویر می‌بینید را بنویسید.

**SECTION 2: PROFESSIONAL_CEPHALOMETRIC_ANALYSIS**
تحلیل کامل حرفه‌ای با رعایت نکات زیر:
- زاویه‌های اسکلتی (SNA: 82°, SNB: 79°, ANB: 3°)
- زاویه‌های دندانی (IMPA: 95°, FMIA: 58°)
- اندازه‌گیری‌های خطی (ارتفاع صورت قدامی کل: 118mm)
- ارزیابی بافت نرم (خط زیبایی E، پروفایل نرم)
- طبقه‌بندی مشکلات و تفسیر بالینی

**⚠️ نکات بسیار مهم:**
- 🚨 **شما باید تصویر را واقعاً تحلیل کنید - از مقادیر فرضی یا نمونه استفاده نکنید!**
- اولین بخشی که تحت عنوان CEHALOMETRIC_MEASUREMENTS_JSON قرار می‌گیرد باید دقیقاً یک JSON معتبر باشد
- مختصات باید از روی تصویر واقعی استخراج شوند نه مقادیر تخمینی
- هر لندمارک را با دقت در تصویر پیدا کنید و موقعیت پیکسلی آن را ثبت کنید
- اگر نتوانستید لندمارک خاصی را پیدا کنید، آن را در JSON قرار ندهید
- تحلیل حرفه‌ای باید کامل و بدون خلاصه باشد
- 📏 ابعاد تصویر را در نظر بگیرید - تصاویر cephalometric معمولاً بین 800x600 تا 2000x1500 پیکسل هستند`;
  } else {
    prompt = `شما یک متخصص ارتودنسی هستید و باید بیمار ${patientInfo?.name || 'نامشخص'} (${patientInfo?.age || 'سن نامشخص'}) ساله را بر اساس تصاویر ارائه شده به صورت جامع تحلیل کنید.

**⚠️ دستورالعمل مهم: پاسخ خود را به زبان پارسی و بیست تا سی هزار کاراکتر کامل ارائه دهید. از خلاصه‌نویسی خودداری کنید و تحلیل کاملی ارائه دهید.**

## 📋 **آنالیز جامع ارتودنسی - فاز تشخیصی**

### 👨‍⚕️ **۱. ارزیابی اولیه بیمار**
- **مشخصات**: نام، سن، جنس، تاریخ پیشگیری
- **سابقه پزشکی**: هرگونه سابقه پزشکی مرتبط
- **شکل اصلی مراجعه**: دلیل اصلی مراجعه به متخصص

### 🔍 **۲. آنالیز داخل دهانی (Intraoral Analysis)**
اگر تصویر داخل دهانی وجود دارد:
- **شاکل دهانی**: have/not نامناسب بودن شاکل دندان‌های بالا و پایین
- **Class Classification**: ارزیابی Class I/II/III (چهار عدد، شش عدد)
- **Overjet/Overbite**: میزان دست‌اندازی افقی و عمودی
- **شرایط مرکزی**: رابطه خطوط وسطی دندانی و پروفایل
- **شلوغی دندانی**: ارزیابی Crowding در هر چهار یک چهارم
- **فضاهای تخلیه**:鉴定 هفته‌های موجود و مورد نیاز

### 🦴 **۳. آنالیز بافت نرم (Soft Tissue Analysis)**
بر اساس تصاویر پروفایل و frontal:
- **خط زیبایی E**: ارزیابی خط از لب بالایی تا لب پایینی
- **پروفایل**: convex/concave/straight assessment
- **لب‌های**: competent/incompetent ارزیابی وضعیت
- **چین لب‌چانه**: mentolabial sulcus ارزیابی
- **سایر لندمارک‌های نرم**: گوهر و غیره

### 📏 **۴. آنالیز رادیولوژی (Radiological Analysis)**
اگر OPG یا Lateral Ceph وجود دارد:
- **عدد دندان**: دندان‌های حال حاضر و غایب
- **توسعه قدانی**: ارزیابی سن فونکسیون و calcification
- **وضعیت roots**: curvature و length نسبت به normal
- **بیماری پریودونتال**: ارزیابی crestal bone و غیره

### 💀 **۵. آنالیز Cephalometric (اگر تصویر lateral وجود دارد)**
- **SNA**: اندازه‌گیری و تفسیر (Normal: 82°)
- **SNB**: اندازه‌گیری و تفسیر (Normal: 80°)
- **ANB**: اندازه‌گیری و تفسیر (Normal: 2-4°)
- **MMPA**: اندازه‌گیری و تفسیر (Normal: 25-30°)
- **FMA**: اندازه‌گیری و تفسیر (Normal: 25-30°)
- **Z-Angle**: اندازه‌گیری و تفسیر (Normal: 75-80°)
- **Other angles**: U1-NA, L1-NB, IMPA و غیره

### 🎯 **۶. تشخیصی نهایی (Comprehensive Diagnosis)**
- **مشکل اصلی**: تعیین مشکلی که بیشتر نیاز به درمان دارد
- **مشکلات فرعی**: سایر مشکلات شناسایی شده
- **شدت مشکلات**: mild/moderate/severe classification
- **تأثیر روی عملکرد**: chewing, speech, aesthetics
- **رابطه با age و growth**: ارزیابی تأثیر پتانسیل رشد

### 🏗️ **۷. طرح درمان پیشنهادی (Treatment Plan)**
- **فاز I**: اگر در سن رشد، interceptive treatment
- **فاز II**: comprehensive orthodontic treatment
- **اپلاینس‌ها**: لیست دستگاه‌های مورد نیاز
- **زمان مورد انتظار**: مدت درمان تقریبی
- **پیگیری**: زمان‌بندی ویزیت‌های کنترل
- **جراحی**: نیاز به جراحی یا خیر و توضیح

### 📈 **۸. پیش‌آگهی درمان (Prognosis)**
- **فاکتورهای مثبت**: آنچه در درمان کمک می‌کند
- **فاکتورهای منفی**: موانع و مشکلات احتمالی
- **نتیجه انتهایی**: وضعیت نهایی آسیب‌پس
- **نگهداری نتایج**: روش‌های retention planning

### 📝 **۹. توصیه‌های اضافی**
- **همکاری متخصصین**: referral به چه متخصصانی
- **آموزش بیمار**: نکات مهم برای رعایت درمان
- **پیشگیری**: اقدامات پیشگیرانه آینده
- **پیروی**: اهمیت compliance بیمار

**⚠️ نکته: پاسخ باید کاملاً حرفه‌ای باشد و از اصطلاحات تخصصی درسیده استفاده کند. نتیجه باید یک گزارش کامل مانند گزارش یک متخصص واقعی باشد.**`;
  }

  try {
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
        'HTTP-Referer': process.env.OPENROUTER_SITE_URL || 'http://localhost:7272',
        'X-Title': process.env.OPENROUTER_SITE_NAME || 'DentalAI Local Development',
      },
      body: JSON.stringify({
        model: modelToUse,
        messages: [
          {
            role: 'system',
            content: isCephalometricAnalysis 
              ? 'You are an expert orthodontist with vision analysis capabilities. You MUST analyze the cephalometric radiograph images provided and extract precise landmark coordinates. شما یک متخصص ارتودنسی هستید که قادر به تحلیل تصاویر رادیوگرافی هستید. باید تصاویر را تحلیل کنید و مختصات دقیق landmarks را استخراج کنید.'
              : 'شما یک متخصص دندانپزشکی ایرانی هستید که همیشه به زبان پارسی پاسخ می‌دهید. تمام پاسخ‌های شما باید به زبان پارسی باشد و از اصطلاحات پزشکی تخصصی پارسی استفاده کنید.',
          },
          {
            role: 'user',
            content: [
              {
                type: 'text',
                text: prompt,
              },
              // Add base64 images to the content
              ...base64Images.map(base64Url => ({
                type: 'image_url' as const,
                image_url: {
                  url: base64Url,
                  detail: 'high' as const, // Request high-resolution analysis
                },
              })),
            ],
          },
        ],
        max_tokens: parseInt(process.env.AI_MAX_TOKENS || '4000', 10), // Increased for detailed analysis
        temperature: parseFloat(process.env.AI_TEMPERATURE || '0.1'), // Low temperature for accuracy
      }),
    });

    if (!response.ok) {
      let errorDetails = '';
      try {
        const errorResponse = await response.clone().json();
        errorDetails = JSON.stringify(errorResponse, null, 2);
        console.error('[AI Diagnosis] OpenRouter API Error Response:', errorDetails);
      } catch (parseError) {
        const errorText = await response.text();
        errorDetails = errorText || response.statusText;
        console.error('[AI Diagnosis] OpenRouter API Error Text:', errorDetails);
      }
      throw new Error(`OpenRouter API error: ${response.status} ${response.statusText} - ${errorDetails}`);
    }

    const data = await response.json();
    const aiResponse = data.choices[0]?.message?.content;

    if (!aiResponse) {
      throw new Error('Empty response from AI model');
    }

    console.log(`[AI Diagnosis] ✅ Received response from ${modelToUse}`);

    // Parse the AI response and structure it appropriately
    return parseAIResponse(aiResponse, patientInfo);

  } catch (error) {
    console.error('[AI Diagnosis] OpenRouter API error:', error);
    throw error;
  }
}

// Parse AI response and structure it into our expected format
function parseAIResponse(aiResponse: string, patientInfo: any) {
  const isCephalometricAnalysis = patientInfo?.analysisType === 'cephalometric';

  if (isCephalometricAnalysis) {
    console.log('[AI Parser] ==================== CEPHALOMETRIC ANALYSIS ====================');
    console.log('[AI Parser] Full AI Response Length:', aiResponse.length);
    console.log('[AI Parser] Response Preview:', `${aiResponse.substring(0, 500)  }...`);
    console.log('[AI Parser] ================================================================');

    // Extract landmark coordinates from the response
    let landmarkData = {};
    let measurements = {};
    let analysisText = aiResponse;

    // Try to extract JSON section from the response
    // Use a more robust regex to capture complete JSON including nested objects
    const jsonMatch = aiResponse.match(/CEHALOMETRIC_MEASUREMENTS_JSON[\s\S]*?```(?:json)?\s*(\{[\s\S]*?\})\s*```/i) ||
                      aiResponse.match(/CEHALOMETRIC_MEASUREMENTS_JSON[\s\S]*?(\{[\s\S]*?\n\})/i);
    
    console.log('[AI Parser] JSON Match Found:', !!jsonMatch);
    
    if (jsonMatch && jsonMatch[1]) {
      try {
        // Clean up JSON before parsing - remove trailing commas and fix common issues
        let jsonStr = jsonMatch[1].trim();
        
        // Remove trailing commas before closing braces/brackets
        jsonStr = jsonStr.replace(/,(\s*[}\]])/g, '$1');
        
        // Ensure proper quotes around property names
        jsonStr = jsonStr.replace(/(\w+):/g, '"$1":');
        
        console.log('[AI Parser] Cleaned JSON:', `${jsonStr.substring(0, 300)  }...`);
        
        const parsedJson = JSON.parse(jsonStr);
        console.log('[AI Parser] Successfully parsed landmark JSON:', parsedJson);

        if (parsedJson.landmarks) {
          landmarkData = parsedJson.landmarks;
        }
        if (parsedJson.measurements) {
          measurements = parsedJson.measurements;
        }

        // Remove the JSON section from analysis text
        analysisText = aiResponse.replace(/SECTION 1: CEHALOMETRIC_MEASUREMENTS_JSON[\s\S]*?SECTION 2:/i, '');
      } catch (parseError) {
        console.error('[AI Parser] Failed to parse landmark JSON:', parseError);
        console.log('[AI Parser] Raw JSON text:', `${jsonMatch[1].substring(0, 200)  }...`);

        // Try to extract coordinates manually from the text
        console.log('[AI Parser] Attempting manual coordinate extraction...');
        const manualExtraction = extractCoordinatesFromText(jsonMatch[1]);
        if (Object.keys(manualExtraction).length > 0) {
          console.log('[AI Parser] Manual extraction successful, found', Object.keys(manualExtraction).length, 'landmarks');
          landmarkData = manualExtraction;
        }

        // For analysis text, try to remove JSON section even if it failed to parse
        analysisText = aiResponse.replace(/SECTION 1: CEHALOMETRIC_MEASUREMENTS_JSON[\s\S]*?(SECTION 2:|[\n\s]*?$)/i, '');
      }
    }

    // If no landmark data was found, try to extract from the entire response text
    if (Object.keys(landmarkData).length === 0) {
      console.log('[AI Parser] No structured JSON found, trying fallback extraction...');
      console.log('[AI Parser] Searching for coordinates in full response...');
      
      const fallbackLandmarks = extractCoordinatesFromText(aiResponse);
      console.log('[AI Parser] Fallback extraction results:', {
        foundCount: Object.keys(fallbackLandmarks).length,
        landmarks: Object.keys(fallbackLandmarks)
      });
      
      if (Object.keys(fallbackLandmarks).length > 0) {
        console.log('[AI Parser] ✅ Fallback extraction successful with', Object.keys(fallbackLandmarks).length, 'landmarks');
        landmarkData = fallbackLandmarks;
      } else {
        // Check if AI understands the request but can't process the image
        if (aiResponse.toLowerCase().includes('نمی‌توانم') || 
            aiResponse.toLowerCase().includes('قادر نیستم') ||
            aiResponse.toLowerCase().includes('متاسفانه') ||
            aiResponse.toLowerCase().includes('cannot') ||
            aiResponse.toLowerCase().includes('unable') ||
            aiResponse.toLowerCase().includes('i don\'t have') ||
            aiResponse.toLowerCase().includes('i can\'t')) {
          console.error('[AI Parser] ❌ AI explicitly stated inability to process image');
          console.error('[AI Parser] AI Response:', aiResponse);
          
          // Provide specific error message based on model
          throw new Error(`❌ مدل ${patientInfo?.aiModel || 'AI'} قادر به تحلیل تصویر نیست!\n\n💡 راه‌حل‌های پیشنهادی:\n1️⃣ از مدل "CephX v2.0" (Claude 3.5 Sonnet) استفاده کنید\n2️⃣ از مدل "DeepCeph" (Claude 3 Opus) استفاده کنید\n3️⃣ از مدل "Gemini Flash" استفاده کنید\n\n⚠️ برخی مدل‌ها از طریق OpenRouter محدودیت تحلیل تصویر دارند.`);
        }
        
        // DO NOT use mock data - throw error instead
        console.error('[AI Parser] ❌ Failed to extract any landmark coordinates from AI response');
        console.error('[AI Parser] Response contains SECTION 1?', aiResponse.includes('SECTION 1'));
        console.error('[AI Parser] Response contains CEHALOMETRIC?', aiResponse.includes('CEHALOMETRIC'));
        console.error('[AI Parser] Response contains landmarks?', aiResponse.includes('landmarks'));
        console.error('[AI Parser] Full raw response:', aiResponse);
        throw new Error('AI did not provide valid landmark coordinates. Please ensure the image is a lateral cephalometric radiograph and try again.');
      }
    }

    const cleanText = (text: string) => text.trim().replace(/\n{3,}/g, '\n\n').substring(0, 4000);

    // Create cephalometric table from measurements
    let cephalometricTable = {};
    if (Object.keys(measurements).length > 0) {
      // Generate table from AI measurements
      cephalometricTable = generateCephalometricTableFromMeasurements(measurements);
      console.log('[AI Parser] ✅ Generated cephalometric table with', Object.keys(measurements).length, 'measurements');
    } else if (Object.keys(landmarkData).length > 0) {
      // If we have landmarks but no measurements, calculate them
      console.log('[AI Parser] 📐 Calculating measurements from', Object.keys(landmarkData).length, 'landmarks');
      measurements = calculateMeasurementsFromLandmarks(landmarkData);
      if (Object.keys(measurements).length > 0) {
        cephalometricTable = generateCephalometricTableFromMeasurements(measurements);
        console.log('[AI Parser] ✅ Calculated', Object.keys(measurements).length, 'measurements');
      }
    } else {
      console.warn('[AI Parser] ⚠️ No measurements or landmarks available for cephalometric table');
      // Return empty table instead of mock data
      cephalometricTable = {};
    }

    return {
      diagnosis: 'آنالیز cephalometric انجام شد. برای جزئیات به بخش آنالیز cephalometric مراجعه کنید.',
      softTissueAnalysis: '',
      cephalometricAnalysis: cleanText(analysisText),
      cephalometricTable,
      cephalometricMeasurements: landmarkData, // Add landmark coordinates
      treatmentPlan: '',
      summary: `آنالیز cephalometric برای بیمار ${patientInfo?.name || 'نامشخص'} تکمیل شد.`,
    };
  }

  // For general analysis, try to parse into different sections
  let diagnosis = '';
  let softTissueAnalysis = '';
  let cephalometricAnalysis = '';
  let treatmentPlan = '';
  let summary = '';

  // Split response by sections
  const sections = aiResponse.split(/\d+\.\s*\*\*|\n\s*\*\*|\n\s*\n/);

  for (const section of sections) {
    const sectionLower = section.toLowerCase();

    if (sectionLower.includes('تشخیص') || sectionLower.includes('diagnosis')) {
      diagnosis += `${section.trim()  }\n`;
    } else if (sectionLower.includes('بافت نرم') || sectionLower.includes('soft tissue')) {
      softTissueAnalysis += `${section.trim()  }\n`;
    } else if (sectionLower.includes('سفالومتری') || sectionLower.includes('cephalometric')) {
      cephalometricAnalysis += `${section.trim()  }\n`;
    } else if (sectionLower.includes('طرح درمان') || sectionLower.includes('treatment plan')) {
      treatmentPlan += `${section.trim()  }\n`;
    } else if (sectionLower.includes('خلاصه') || sectionLower.includes('summary')) {
      summary += `${section.trim()  }\n`;
    } else if (section.length > 50) {
      // Fallback: add to diagnosis if it's substantial content
      diagnosis += `${section.trim()  }\n`;
    }
  }

  // Ensure we have at least basic responses
  if (!diagnosis.trim()) {
    diagnosis = aiResponse.substring(0, 500) + (aiResponse.length > 500 ? '...' : '');
  }

  if (!summary.trim()) {
    summary = `تحلیل AI برای بیمار ${patientInfo?.name || 'نامشخص'} تکمیل شد. برای جزئیات بیشتر به بخش‌های مربوطه مراجعه کنید.`;
  }

  // Clean up and format responses
  const cleanText = (text: string) => text.trim().replace(/\n{3,}/g, '\n\n').substring(0, 10000);

  return {
    diagnosis: cleanText(diagnosis),
    softTissueAnalysis: cleanText(softTissueAnalysis),
    cephalometricAnalysis: cleanText(cephalometricAnalysis),
    treatmentPlan: cleanText(treatmentPlan),
    summary: cleanText(summary),
  };
}

// Helper function to generate cephalometric table from AI measurements
function generateCephalometricTableFromMeasurements(measurements: any) {
  const table: Record<string, any> = {};

  // Example of how to build table from measurements
  // This is a basic implementation - you could make it more sophisticated
  Object.entries(measurements).forEach(([key, value]: [string, any]) => {
    const measured = value || '';
    let severity = 'نرمال';
    let note = '';

    // Basic interpretation logic - can be enhanced
    if (typeof measured === 'string') {
      const numericValue = parseFloat(measured.replace('°', ''));
      if (!isNaN(numericValue)) {
        switch (key) {
          case 'SNA':
            severity = numericValue < 80 ? 'زیاد' : numericValue > 85 ? 'کم' : 'نرمال';
            note = 'زاویه فک بالا نسبت به پایه جمجمه';
            break;
          case 'SNB':
            severity = numericValue < 78 ? 'زیاد' : numericValue > 82 ? 'کم' : 'نرمال';
            note = 'زاویه فک پایین نسبت به پایه جمجمه';
            break;
          case 'ANB':
            severity = numericValue < 1 ? 'کم' : numericValue > 5 ? 'زیاد' : 'نرمال';
            note = 'تفاوت SNA و SNB (کلاس اسکلتی)';
            break;
        }
      }
    }

    table[key] = {
      mean: getNormalValue(key),
      severity,
      note,
      measured
    };
  });

  return table;
}

// Helper function to get normal values for cephalometric measurements
function getNormalValue(measurement: string) {
  const normalValues: { [key: string]: string } = {
    'SNA': '82° ± 2°',
    'SNB': '80° ± 2°',
    'ANB': '2°-4°',
    'MMPA': '25°-30°',
    'FMA': '25°-30°',
    'FMIA': '65°-75°',
    'Z-Angle': '75°-80°',
    'U1-NA': '22°',
    'L1-NB': '25°',
    'IMPA': '90°-95°'
  };

  return normalValues[measurement] || '-';
}

// Function to extract coordinates manually from text (when JSON parsing fails)
function extractCoordinatesFromText(text: string) {
  const landmarks: Record<string, { x: number, y: number }> = {};

  // Regular expressions to find landmark coordinates in various formats
  const patterns = [
    /(\w+):\s*\{"x":\s*(\d+),\s*"y":\s*(\d+)\}/gi,
    /(\w+):\s*\("(\d+)",\s*"(\d+)"\)/gi,
    /(\w+):\s*\[(\d+),\s*(\d+)\]/gi,
    /(\w+)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)/gi,
    /(\w+).*?coordinates.*?(\d+).*?(\d+)/gi,
  ];

  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      const landmark = match[1];
      const x = parseInt(match[2], 10);
      const y = parseInt(match[3], 10);

      if (landmark && !isNaN(x) && !isNaN(y)) {
        landmarks[landmark] = { x, y };
      }
    }
  }

  return landmarks;
}

// Calculate cephalometric measurements from landmarks
function calculateMeasurementsFromLandmarks(landmarks: any) {
  const measurements: any = {};
  
  // Helper function to calculate angle between three points
  const calculateAngle = (p1: any, vertex: any, p2: any): number => {
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
    let angle = Math.abs((angle1 - angle2) * 180 / Math.PI);
    if (angle > 180) angle = 360 - angle;
    return Math.round(angle * 10) / 10;
  };
  
  // Helper to check if all required landmarks exist
  const hasLandmarks = (...names: string[]) => names.every(name => landmarks[name]);
  
  try {
    // SNA angle: S-N-A
    if (hasLandmarks('S', 'N', 'A')) {
      measurements.SNA = calculateAngle(landmarks.S, landmarks.N, landmarks.A);
    }
    
    // SNB angle: S-N-B
    if (hasLandmarks('S', 'N', 'B')) {
      measurements.SNB = calculateAngle(landmarks.S, landmarks.N, landmarks.B);
    }
    
    // ANB angle: difference between SNA and SNB
    if (measurements.SNA && measurements.SNB) {
      measurements.ANB = Math.round((measurements.SNA - measurements.SNB) * 10) / 10;
    }
    
    // FMA (Frankfort Mandibular Plane Angle)
    // This requires more specific landmarks, simplified here
    if (hasLandmarks('Go', 'Me', 'N')) {
      measurements.FMA = calculateAngle(landmarks.Go, landmarks.Me, landmarks.N);
    }
    
    // GoGn-SN (Mandibular Plane Angle)
    if (hasLandmarks('Go', 'Gn', 'S', 'N')) {
      const gnAngle = calculateAngle(landmarks.Go, landmarks.Gn, landmarks.S);
      measurements.MMPA = Math.round(gnAngle * 10) / 10;
    }
    
    console.log('[AI Calculation] Calculated measurements:', measurements);
  } catch (error) {
    console.error('[AI Calculation] Error calculating measurements:', error);
  }
  
  return measurements;
}
