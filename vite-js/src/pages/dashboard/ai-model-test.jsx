import { Helmet } from 'react-helmet-async';
import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import Switch from '@mui/material/Switch';
import Slider from '@mui/material/Slider';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import MenuItem from '@mui/material/MenuItem';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import CardContent from '@mui/material/CardContent';
import LinearProgress from '@mui/material/LinearProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';

import { CONFIG } from 'src/config-global';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { AdvancedCephalometricVisualizer } from 'src/components/advanced-cephalometric-visualizer';

// ----------------------------------------------------------------------

// رنگ‌های مدرن و جذاب از OPG Visualizer
const MODERN_COLORS = [
  '#FF6B6B', // قرمز صورتی مدرن
  '#4ECDC4', // آبی سبز نئونی
  '#45B7D1', // آبی روشن مدرن
  '#FFA07A', // نارنجی صورتی
  '#98D8C8', // سبز آبی ملایم
  '#F7DC6F', // زرد طلایی
  '#BB8FCE', // بنفش روشن
  '#85C1E9', // آبی آسمانی
  '#F8C471', // زرد نارنجی
  '#82E0AA', // سبز روشن
  '#F1948A', // صورتی روشن
  '#AED6F1', // آبی خیلی روشن
  '#FF8A65', // نارنجی قرمز
  '#81C784', // سبز روشن
  '#64B5F6', // آبی روشن
  '#BA68C8', // بنفش صورتی
  '#4DB6AC', // سبز آبی
  '#FFB74D', // زرد نارنجی روشن
  '#F48FB1', // صورتی
  '#90CAF9', // آبی خیلی روشن
  '#A5D6A7', // سبز خیلی روشن
  '#CE93D8', // بنفش خیلی روشن
  '#FFAB91', // نارنجی صورتی روشن
  '#80CBC4', // سبز آبی روشن
  '#9FA8DA', // آبی بنفش
  '#FFE082', // زرد خیلی روشن
  '#EF9A9A', // قرمز صورتی روشن
  '#B39DDB', // بنفش آبی
  '#BCAAA4', // قهوه‌ای روشن
  '#EEEEEE', // خاکستری خیلی روشن
];

const MODELS = [
  {
    id: 'local/hrnet-w32',
    name: 'HRNet-W32 (Local)',
    provider: 'Local Server',
    description: '⭐ مدل تخصصی Cephalometric - دقت عالی (MRE 0.63mm)',
    color: MODERN_COLORS[0], // قرمز صورتی مدرن
    isLocal: true,
    requiresApiKey: false,
  },
  {
    id: 'local/aariz-256',
    name: 'Aariz 256x256 (Local)',
    provider: 'Local Server',
    description: 'Aariz Model 256x256 - Fast (SDR: 61.55%)',
    color: MODERN_COLORS[1], // آبی سبز نئونی
    isLocal: true,
    requiresApiKey: false,
  },
  {
    id: 'local/aariz-512',
    name: 'Aariz 512x512 (Local)',
    provider: 'Local Server',
    description: 'Aariz Model 512x512 - Balanced (SDR: 73.45%)',
    color: MODERN_COLORS[2], // آبی روشن مدرن
    isLocal: true,
    requiresApiKey: false,
  },
  {
    id: 'local/aariz-512-tta',
    name: 'Aariz 512x512 + TTA (Local)',
    provider: 'Local Server',
    description: '⭐ BEST - Aariz 512x512 with TTA (SDR: 74.83%)',
    color: MODERN_COLORS[3], // نارنجی صورتی
    isLocal: true,
    requiresApiKey: false,
  },
  {
    id: 'local/aariz-ensemble',
    name: 'Aariz Ensemble (Local)',
    provider: 'Local Server',
    description: 'Aariz Ensemble 256+512 (SDR: 71.90%)',
    color: MODERN_COLORS[4], // سبز آبی ملایم
    isLocal: true,
    requiresApiKey: false,
  },
  {
    id: 'local/aariz-ensemble-tta',
    name: 'Aariz Ensemble + TTA (Local)',
    provider: 'Local Server',
    description: 'Aariz Ensemble 256+512 with TTA (SDR: 72.41%)',
    color: MODERN_COLORS[5], // زرد طلایی
    isLocal: true,
    requiresApiKey: false,
  },
];

const PROMPT_TEMPLATE = `You are an expert in cephalometric analysis. Analyze this lateral cephalometric radiograph and identify the following anatomical landmarks with their exact pixel coordinates:

Required landmarks:
1. S (Sella) - Center of sella turcica
2. N (Nasion) - Most anterior point of frontonasal suture
3. A (Point A) - Deepest point on maxilla between ANS and prosthion
4. B (Point B) - Deepest point on mandible between infradentale and pogonion
5. Pog (Pogonion) - Most anterior point of chin
6. Go (Gonion) - Most posterior-inferior point of mandibular angle
7. Me (Menton) - Most inferior point of mandibular symphysis
8. Or (Orbitale) - Lowest point of orbital margin
9. Po (Porion) - Superior point of external auditory meatus
10. ANS (Anterior Nasal Spine) - Tip of anterior nasal spine
11. PNS (Posterior Nasal Spine) - Tip of posterior nasal spine
12. U1 (Upper Incisor) - Incisal edge of upper central incisor
13. L1 (Lower Incisor) - Incisal edge of lower central incisor

Please respond ONLY with a valid JSON object in this exact format:
{
  "landmarks": {
    "S": {"x": 0, "y": 0},
    "N": {"x": 0, "y": 0},
    "A": {"x": 0, "y": 0},
    "B": {"x": 0, "y": 0},
    "Pog": {"x": 0, "y": 0},
    "Go": {"x": 0, "y": 0},
    "Me": {"x": 0, "y": 0},
    "Or": {"x": 0, "y": 0},
    "Po": {"x": 0, "y": 0},
    "ANS": {"x": 0, "y": 0},
    "PNS": {"x": 0, "y": 0},
    "U1": {"x": 0, "y": 0},
    "L1": {"x": 0, "y": 0}
  },
  "confidence": 0.0,
  "notes": "any observations"
}

Do not include any text outside the JSON object.`;

// ----------------------------------------------------------------------

export default function AIModelTestPage() {
  const [selectedModel, setSelectedModel] = useState(MODELS[3].id); // Aariz 512x512 + TTA (بهترین دقت)
  const [apiKey, setApiKey] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prompt, setPrompt] = useState(PROMPT_TEMPLATE);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [testHistory, setTestHistory] = useState([]);
  const [imageSize, setImageSize] = useState(null);
  const [autoScale, setAutoScale] = useState(true);
  const [isSavingToDb, setIsSavingToDb] = useState(false);
  const [dbTestHistory, setDbTestHistory] = useState([]);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Contour detection state variables
  const [availableContourRegions, setAvailableContourRegions] = useState([]);
  const [customContourSettings, setCustomContourSettings] = useState({});
  const [enableContours, setEnableContours] = useState(false);
  const [contours, setContours] = useState(null);
  const [contourError, setContourError] = useState(null);
  const [isDetectingContours, setIsDetectingContours] = useState(false);
  const [contourMethod, setContourMethod] = useState('advanced');
  const [selectedContourRegions, setSelectedContourRegions] = useState([]);


  // Default settings for reference
  const defaultSettings = {
    soft_tissue_profile: {
      search_area_ratio: 0.4,
      edge_threshold_low: 30,
      edge_threshold_high: 100,
      smooth_sigma: 2.0,
      contrast_alpha: 1.5,
      contrast_beta: -30,
      max_points: 200,
      simplify_epsilon: 1.5,
      use_circle_fit: false,
      use_ellipse_fit: false,
    },
    mandible_border: {
      search_area_ratio: 0.5,
      edge_threshold_low: 40,
      edge_threshold_high: 120,
      smooth_sigma: 1.5,
      contrast_alpha: 1.8,
      contrast_beta: -40,
      max_points: 150,
      simplify_epsilon: 2.0,
      use_circle_fit: false,
      use_ellipse_fit: false,
    },
    maxilla_border: {
      search_area_ratio: 0.4,
      edge_threshold_low: 35,
      edge_threshold_high: 110,
      smooth_sigma: 1.5,
      contrast_alpha: 1.6,
      contrast_beta: -35,
      max_points: 150,
      simplify_epsilon: 2.0,
      use_circle_fit: false,
      use_ellipse_fit: false,
    },
    sella_turcica: {
      search_area_ratio: 0.25,
      edge_threshold_low: 25,
      edge_threshold_high: 90,
      smooth_sigma: 1.0,
      contrast_alpha: 2.0,
      contrast_beta: -50,
      max_points: 64,
      simplify_epsilon: 2.0,
      use_circle_fit: false,
      use_ellipse_fit: true,  // استفاده از ellipse به جای circle
      ellipse_fit_points: 48,
    },
    orbital_rim: {
      search_area_ratio: 0.3,
      edge_threshold_low: 30,
      edge_threshold_high: 100,
      smooth_sigma: 2.0,
      contrast_alpha: 1.7,
      contrast_beta: -40,
      max_points: 100,
      simplify_epsilon: 2.5,
      use_circle_fit: true,
      circle_fit_points: 48,
      use_ellipse_fit: false,
    },
    upper_tooth: {
      search_area_ratio: 0.2,
      edge_threshold_low: 40,
      edge_threshold_high: 130,
      smooth_sigma: 1.0,
      contrast_alpha: 2.2,
      contrast_beta: -50,
      max_points: 80,
      simplify_epsilon: 1.5,
      use_circle_fit: false,
      use_ellipse_fit: false,
    },
    lower_tooth: {
      search_area_ratio: 0.2,
      edge_threshold_low: 40,
      edge_threshold_high: 130,
      smooth_sigma: 1.0,
      contrast_alpha: 2.2,
      contrast_beta: -50,
      max_points: 80,
      simplify_epsilon: 1.5,
      use_circle_fit: false,
      use_ellipse_fit: false,
    },
  };
  
  // Initialize custom settings with defaults
  useEffect(() => {
    if (availableContourRegions.length > 0 && Object.keys(customContourSettings).length === 0) {
      const initialSettings = {};
      availableContourRegions.forEach(region => {
        if (defaultSettings[region]) {
          initialSettings[region] = { ...defaultSettings[region] };
        }
      });
      setCustomContourSettings(initialSettings);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [availableContourRegions]);

  // بارگذاری تاریخچه از دیتابیس
  useEffect(() => {
    loadTestHistory();
  }, []);

  // بارگذاری لیست نواحی قابل تشخیص برای contours (برای همه مدل‌های محلی)
  const loadAvailableContourRegions = useCallback(async () => {
    try {
      // تعیین port و endpoint بر اساس مدل انتخاب شده
      let contourApiUrl = null;
      if (selectedModel === 'local/hrnet-w32') {
        contourApiUrl = `${CONFIG.hrnetServerUrl}/contour-regions`;
      } else if (selectedModel && selectedModel.startsWith('local/aariz')) {
        contourApiUrl = `${CONFIG.aiServerUrl}/contour-regions`;
      }
      
      // اگر مدل محلی نیست، نیازی به بارگذاری نیست
      if (!contourApiUrl) {
        return;
      }
      
      const response = await fetch(contourApiUrl);
      const data = await response.json();
      if (data.success) {
        setAvailableContourRegions(Object.keys(data.regions || {}));
        // به طور پیش‌فرض تمام نواحی را انتخاب کن
        setSelectedContourRegions(Object.keys(data.regions || {}));
      }
    } catch (err) {
      // خطا را فقط به صورت warning نمایش بده
      const isLocalModel = selectedModel && (selectedModel === 'local/hrnet-w32' || selectedModel.startsWith('local/aariz'));
      if (isLocalModel) {
        console.warn('⚠️ Contour service not available (contours disabled):', err.message);
      }
    }
  }, [selectedModel]);

  // بارگذاری نواحی contours وقتی مدل تغییر می‌کند
  useEffect(() => {
    loadAvailableContourRegions();
  }, [loadAvailableContourRegions]);

  const loadTestHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const response = await fetch(`${CONFIG.site.serverUrl || 'http://localhost:7272'}/api/ai-model-tests?limit=20`);
      const data = await response.json();
      
      if (data.success) {
        setDbTestHistory(data.data);
      }
    } catch (err) {
      console.error('خطا در بارگذاری تاریخچه:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  const saveTestToDb = async (testResult) => {
    setIsSavingToDb(true);
    try {
      // اطمینان از اینکه model وجود دارد
      if (!testResult?.model) {
        console.error('❌ خطا: model در testResult وجود ندارد');
        return;
      }

      // اگر imagePreview یک blob URL است، آن را null می‌کنیم (نمی‌توان blob URL را در دیتابیس ذخیره کرد)
      let imageUrlForDb = null;
      if (imagePreview && !imagePreview.startsWith('blob:')) {
        imageUrlForDb = imagePreview;
      }

      const payload = {
        modelId: testResult.model.id || 'unknown',
        modelName: testResult.model.name || 'Unknown Model',
        modelProvider: testResult.model.provider || 'Unknown Provider',
        imageUrl: imageUrlForDb,
        imageSize: imageSize ? JSON.stringify(imageSize) : null,
        prompt: prompt || '',
        success: testResult.success || false,
        landmarks: testResult.success && testResult.response?.landmarks 
          ? JSON.stringify(testResult.response.landmarks) 
          : null,
        rawResponse: testResult.rawResponse || null,
        error: testResult.error || null,
        processingTime: testResult.metadata?.processingTime 
          ? parseFloat(testResult.metadata.processingTime) 
          : null,
        tokensUsed: testResult.metadata?.tokensUsed 
          ? JSON.stringify(testResult.metadata.tokensUsed) 
          : null,
        scalingInfo: testResult.metadata?.scaling 
          ? JSON.stringify(testResult.metadata.scaling) 
          : null,
        confidence: testResult.response?.confidence || testResult.confidence || null,
        userId: null, // TODO: دریافت از auth context
      };

      const response = await fetch(`${CONFIG.site.serverUrl || 'http://localhost:7272'}/api/ai-model-tests`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      
      if (data.success) {
        console.log('✅ تست در دیتابیس ذخیره شد:', data.data.id);
        // به‌روزرسانی تاریخچه
        await loadTestHistory();
      } else {
        console.error('❌ خطا در ذخیره تست:', data.error || 'Unknown error');
      }
    } catch (err) {
      console.error('❌ خطا در ذخیره تست:', err);
    } finally {
      setIsSavingToDb(false);
    }
  };

  const handleDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImageFile(file);
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      setError(null);
      setResult(null);
      
      // دریافت ابعاد واقعی تصویر (از naturalWidth/Height استفاده می‌کنیم)
      const img = new Image();
      img.onload = () => {
        // استفاده از naturalWidth/naturalHeight برای اندازه واقعی (نه display size)
        const naturalWidth = img.naturalWidth || img.width;
        const naturalHeight = img.naturalHeight || img.height;
        setImageSize({ width: naturalWidth, height: naturalHeight });
        console.log(`📏 ابعاد تصویر: ${naturalWidth} × ${naturalHeight} (natural: ${img.naturalWidth} × ${img.naturalHeight}, display: ${img.width} × ${img.height})`);
      };
      img.src = previewUrl;
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setImageFile(null);
    setImagePreview(null);
  }, []);

  const convertImageToBase64 = (file) => 
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const calculateScalingFactor = (landmarks, actualSize) => {
    if (!landmarks || !actualSize) return 1;

    // پیدا کردن محدوده landmarks
    const xCoords = Object.values(landmarks).map(lm => lm.x);
    const yCoords = Object.values(landmarks).map(lm => lm.y);
    
    const maxX = Math.max(...xCoords);
    const maxY = Math.max(...yCoords);
    
    // محاسبه ضریب scaling بر اساس نسبت
    const scaleX = actualSize.width / maxX;
    const scaleY = actualSize.height / maxY;
    
    // استفاده از میانگین برای دقت بیشتر
    const scalingFactor = (scaleX + scaleY) / 2;
    
    console.log(`🔢 Scaling calculation:
      - Image size: ${actualSize.width} × ${actualSize.height}
      - Landmark range: ${maxX.toFixed(0)} × ${maxY.toFixed(0)}
      - Scale X: ${scaleX.toFixed(4)}
      - Scale Y: ${scaleY.toFixed(4)}
      - Final factor: ${scalingFactor.toFixed(4)}`);
    
    return scalingFactor;
  };

  // تابع تشخیص contours با تنظیمات پیش‌فرض
  const detectContours = async (base64Image, landmarks) => {
    if (!base64Image || !landmarks) {
      setContourError('ابتدا landmarks را تشخیص دهید');
      return;
    }
    
    setIsDetectingContours(true);
    setContourError(null);
    
    try {
      // تعیین endpoint بر اساس مدل انتخاب شده
      let contourApiUrl = null;
      if (selectedModel === 'local/hrnet-w32') {
        contourApiUrl = `${CONFIG.hrnetServerUrl}/detect-contours`;
      } else if (selectedModel && selectedModel.startsWith('local/aariz')) {
        contourApiUrl = `${CONFIG.aiServerUrl}/detect-contours`;
      } else {
        throw new Error('Contour detection only available for local models');
      }
      
      console.log(`[Contour] Using method: ${contourMethod}`);
      
      const payload = {
        image_base64: base64Image,
        landmarks,
        method: contourMethod, // *** اضافه کردن method ***
      };
      
      const response = await fetch(contourApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      console.log('🔍 Contour API Response:', data);
      
      if (data.success && data.contours) {
        // ساختار response: { success: true, contours: { success: true, contours: { region1: {...}, ... } } }
        let contoursDict = {};
        
        // بررسی ساختار nested
        if (data.contours && typeof data.contours === 'object') {
          if (data.contours.contours && typeof data.contours.contours === 'object') {
            // ساختار کامل: { success: true, contours: { contours: { region1: {...}, ... } } }
            contoursDict = data.contours.contours;
          } else if (!data.contours.success && Object.keys(data.contours).length > 0) {
            // ساختار مستقیم: { success: true, contours: { region1: {...}, region2: {...} } }
            // یعنی data.contours خودش dict of regions است
            contoursDict = data.contours;
          } else {
            // fallback
            contoursDict = data.contours;
          }
        }
        
        // فیلتر کردن regions که success: false ندارند
        const validContours = {};
        Object.keys(contoursDict).forEach(region => {
          const regionData = contoursDict[region];
          if (regionData && typeof regionData === 'object' && regionData.contour && Array.isArray(regionData.contour) && regionData.contour.length > 0) {
            validContours[region] = regionData;
          }
        });
        
        if (Object.keys(validContours).length === 0) {
          console.warn('⚠️ No valid contours found in response');
          setContourError('هیچ کانتور معتبری یافت نشد');
          setContours(null);
          return;
        }
        
        // تبدیل به فرمت مورد انتظار frontend
        const contoursData = {
          success: true,
          num_regions: Object.keys(validContours).length,
          contours: validContours,
        };
        setContours(contoursData);
        console.log(`✅ Contours detected: ${contoursData.num_regions} regions`, contoursData);
      } else {
        const errorMsg = data.error || data.contours?.error || 'Contour detection failed';
        throw new Error(errorMsg);
      }
    } catch (err) {
      console.error('خطا در تشخیص contours:', err);
      setContourError(err.message || 'خطا در تشخیص contours');
      setContours(null);
    } finally {
      setIsDetectingContours(false);
    }
  };
  
  // تابع تشخیص contours با تنظیمات سفارشی
  const detectContoursWithCustomSettings = async () => {
    if (!imageFile || !result?.success || !result?.response?.landmarks) {
      setContourError('ابتدا landmarks را تشخیص دهید');
      return;
    }
    
    setIsDetectingContours(true);
    setContourError(null);
    
    try {
      // تعیین endpoint بر اساس مدل انتخاب شده
      let contourApiUrl = null;
      if (selectedModel === 'local/hrnet-w32') {
        contourApiUrl = `${CONFIG.hrnetServerUrl}/detect-contours`;
      } else if (selectedModel && selectedModel.startsWith('local/aariz')) {
        contourApiUrl = `${CONFIG.aiServerUrl}/detect-contours`;
      } else {
        throw new Error('Contour detection only available for local models');
      }
      
      const base64Image = await convertImageToBase64(imageFile);
      const payload = {
        image_base64: base64Image,
        landmarks: result.response.landmarks,
        custom_configs: customContourSettings,
      };
      
      const response = await fetch(contourApiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      const data = await response.json();
      
      console.log('🔍 Contour API Response (custom):', data);
      
      if (data.success && data.contours) {
        // ساختار response: { success: true, contours: { success: true, contours: { region1: {...}, ... } } }
        let contoursDict = {};
        
        // بررسی ساختار nested
        if (data.contours && typeof data.contours === 'object') {
          if (data.contours.contours && typeof data.contours.contours === 'object') {
            // ساختار کامل: { success: true, contours: { contours: { region1: {...}, ... } } }
            contoursDict = data.contours.contours;
          } else if (!data.contours.success && Object.keys(data.contours).length > 0) {
            // ساختار مستقیم: { success: true, contours: { region1: {...}, region2: {...} } }
            contoursDict = data.contours;
          } else {
            contoursDict = data.contours;
          }
        }
        
        // فیلتر کردن regions که success: false ندارند
        const validContours = {};
        Object.keys(contoursDict).forEach(region => {
          const regionData = contoursDict[region];
          if (regionData && typeof regionData === 'object' && regionData.contour && Array.isArray(regionData.contour) && regionData.contour.length > 0) {
            validContours[region] = regionData;
          }
        });
        
        if (Object.keys(validContours).length === 0) {
          console.warn('⚠️ No valid contours found in response');
          setContourError('هیچ کانتور معتبری یافت نشد');
          setContours(null);
          return;
        }
        
        // تبدیل به فرمت مورد انتظار frontend
        const contoursData = {
          success: true,
          num_regions: Object.keys(validContours).length,
          contours: validContours,
        };
        setContours(contoursData);
        console.log(`✅ Contours detected with custom settings: ${contoursData.num_regions} regions`, contoursData);
      } else {
        const errorMsg = data.error || data.contours?.error || 'Contour detection failed';
        throw new Error(errorMsg);
      }
    } catch (err) {
      console.error('خطا در تشخیص contours:', err);
      setContourError(err.message || 'خطا در تشخیص contours');
      setContours(null);
    } finally {
      setIsDetectingContours(false);
    }
  };
  
  // Helper function to update custom settings
  const updateCustomSetting = (region, key, value) => {
    setCustomContourSettings(prev => ({
      ...prev,
      [region]: {
        ...prev[region],
        [key]: value,
      },
    }));
  };

  const scaleLoadmarks = (landmarks, scalingFactor) => {
    const scaled = {};
    Object.entries(landmarks).forEach(([name, coords]) => {
      scaled[name] = {
        x: Math.round(coords.x * scalingFactor * 100) / 100,
        y: Math.round(coords.y * scalingFactor * 100) / 100,
      };
    });
    return scaled;
  };

  const handleTest = async () => {
    if (!imageFile) {
      setError('لطفاً ابتدا یک تصویر آپلود کنید');
      return;
    }

    const selectedModelInfo = MODELS.find(m => m.id === selectedModel);
    const isLocalModel = selectedModelInfo?.isLocal;

    if (!isLocalModel && !apiKey) {
      setError('لطفاً API Key را وارد کنید');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    const startTime = Date.now();

    try {
      // تبدیل تصویر به base64
      const base64Image = await convertImageToBase64(imageFile);
      
      // Log for debugging
      console.log('🖼️ Image Debug Info:');
      console.log('   File name:', imageFile.name);
      console.log('   File size:', imageFile.size, 'bytes');
      console.log('   File type:', imageFile.type);
      console.log('   Frontend detected size:', imageSize);
      console.log('   Base64 length:', base64Image.length);

      let response; let data; let content; let parsedContent; let processingTime;

      // اگر مدل Local HRNet است
      if (isLocalModel && selectedModel === 'local/hrnet-w32') {
        console.log('🤖 Using Local HRNet Service...');
        
        // ارسال درخواست به سرویس محلی HRNet
        response = await fetch(`${CONFIG.hrnetServerUrl}/detect`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_base64: base64Image
          }),
        });

        const endTime = Date.now();
        processingTime = (endTime - startTime) / 1000;

        if (!response.ok) {
          throw new Error(`HRNet Service Error: ${response.status}`);
        }

        data = await response.json();
        
        if (!data.success) {
          throw new Error(data.error || 'Detection failed');
        }

        // تبدیل فرمت HRNet به فرمت استاندارد
        parsedContent = {
          landmarks: data.landmarks,
          confidence: data.metadata?.avg_confidence || 0.9,
          notes: `HRNet-W32 Detection - ${data.metadata?.model_type || 'mock'} mode`,
          measurements: data.measurements,
        };

        content = JSON.stringify(parsedContent);

      } else if (isLocalModel && selectedModel.startsWith('local/aariz')) {
        console.log(`🤖 Using Local Aariz Model Service (${selectedModel})...`);
        
        // Determine endpoint based on model type
        let endpoint = `${CONFIG.aiServerUrl}/detect`;
        if (selectedModel === 'local/aariz-256') {
          endpoint = `${CONFIG.aiServerUrl}/detect-256`;
        } else if (selectedModel === 'local/aariz-512') {
          endpoint = `${CONFIG.aiServerUrl}/detect-512`;
        } else if (selectedModel === 'local/aariz-512-tta') {
          endpoint = `${CONFIG.aiServerUrl}/detect-512-tta`;
        } else if (selectedModel === 'local/aariz-ensemble') {
          endpoint = `${CONFIG.aiServerUrl}/detect-ensemble`;
        } else if (selectedModel === 'local/aariz-ensemble-tta') {
          endpoint = `${CONFIG.aiServerUrl}/detect-ensemble-tta`;
        }
        
        // ارسال درخواست به سرویس محلی Aariz
        response = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            image_base64: base64Image
          }),
        });

        const endTime = Date.now();
        processingTime = (endTime - startTime) / 1000;

        if (!response.ok) {
          const errorText = await response.text();
          throw new Error(`Aariz Model Service Error: ${response.status} - ${errorText}`);
        }

        data = await response.json();
        
        if (!data.success) {
          throw new Error(data.error || 'Detection failed');
        }

        // Log image size for debugging
        console.log('📊 API Response Debug:');
        console.log('   API image_size:', data.metadata?.image_size);
        console.log('   Frontend imageSize:', imageSize);
        console.log('   Sample landmark (A):', data.landmarks?.A);
        console.log('   Model:', data.metadata?.model);

        // تبدیل فرمت Aariz به فرمت استاندارد
        parsedContent = {
          landmarks: data.landmarks,
          confidence: data.metadata?.avg_confidence || 0.9,
          notes: `${data.metadata?.model || 'Aariz Model'} - ${data.metadata?.model_architecture || 'trained'} - ${data.metadata?.valid_landmarks || 0}/29 لندمارک`,
          metadata: {
            ...data.metadata,
            frontend_image_size: imageSize, // Add frontend image size for comparison
          },
        };

        content = JSON.stringify(parsedContent);

      } else {
        // برای مدل‌های OpenRouter
        console.log('🌐 Using OpenRouter API...');
        
        response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${apiKey}`,
            'Content-Type': 'application/json',
            'HTTP-Referer': window.location.origin,
            'X-Title': 'DentalAI - Cephalometric Analysis',
          },
          body: JSON.stringify({
            model: selectedModel,
            messages: [
              {
                role: 'user',
                content: [
                  {
                    type: 'text',
                    text: prompt,
                  },
                  {
                    type: 'image_url',
                    image_url: {
                      url: base64Image,
                    },
                  },
                ],
              },
            ],
            max_tokens: 2000,
            temperature: 0.1,
          }),
        });

        const endTime = Date.now();
        processingTime = (endTime - startTime) / 1000;

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error?.message || `HTTP ${response.status}: ${response.statusText}`);
        }

        data = await response.json();
        
        // استخراج پاسخ
        const { choices } = data;
        const { message } = choices[0];
        const { content: messageContent } = message;
        content = messageContent;

        // پاک کردن markdown code blocks
        if (content.includes('```json')) {
          content = content.split('```json')[1].split('```')[0];
        } else if (content.includes('```')) {
          content = content.split('```')[1].split('```')[0];
        }

        // Parse کردن JSON
        try {
          parsedContent = JSON.parse(content.trim());
        } catch (e) {
          parsedContent = { raw: content, parseError: e.message };
        }
      }

      // اعمال auto-scaling اگر فعال باشد
      let scaledContent = parsedContent;
      let scalingInfo = null;
      
      // Log for debugging
      console.log('🔍 Auto-scale check:');
      console.log('   autoScale enabled:', autoScale);
      console.log('   imageSize:', imageSize);
      console.log('   has landmarks:', !!parsedContent.landmarks);
      
      if (autoScale && imageSize && parsedContent.landmarks) {
        const scalingFactor = calculateScalingFactor(parsedContent.landmarks, imageSize);
        
        console.log('   calculated scalingFactor:', scalingFactor);
        
        if (scalingFactor > 1.5 || scalingFactor < 0.5) {
          // فقط اگر scaling قابل توجه باشد
          console.log(`⚠️ Auto-scaling applied: ${scalingFactor.toFixed(4)}x`);
          scaledContent = {
            ...parsedContent,
            landmarks: scaleLoadmarks(parsedContent.landmarks, scalingFactor),
            original_landmarks: parsedContent.landmarks, // ذخیره مقادیر اصلی
          };
          
          scalingInfo = {
            factor: scalingFactor.toFixed(4),
            imageSize,
            applied: true,
          };
        } else {
          console.log('   ✅ Scaling factor within normal range, no scaling applied');
        }
      } else {
        console.log('   ✅ Auto-scaling disabled or conditions not met - using original landmarks');
      }

      const testResult = {
        success: true,
        model: MODELS.find(m => m.id === selectedModel),
        response: scaledContent,
        metadata: {
          processingTime: processingTime.toFixed(2),
          tokensUsed: data.usage,
          timestamp: new Date().toLocaleString('fa-IR'),
          scaling: scalingInfo,
        },
        rawResponse: content,
      };

      setResult(testResult);
      
      // اضافه به تاریخچه محلی
      setTestHistory(prev => [testResult, ...prev.slice(0, 9)]);
      
      // اگر contour detection فعال است و مدل Local است (HRNet یا Aariz)
      if (enableContours && isLocalModel && (selectedModel === 'local/hrnet-w32' || selectedModel.startsWith('local/aariz')) && testResult.success) {
        console.log('🔄 Auto-detecting contours...');
        await detectContours(base64Image, scaledContent.landmarks || parsedContent.landmarks);
      }
      
      // ذخیره در دیتابیس
      await saveTestToDb(testResult);

    } catch (err) {
      console.error('Test error:', err);
      setError(err.message || 'خطا در تست مدل');
      
      const errorResult = {
        success: false,
        model: MODELS.find(m => m.id === selectedModel),
        error: err.message,
        metadata: {
          timestamp: new Date().toLocaleString('fa-IR'),
        },
      };
      
      setTestHistory(prev => [errorResult, ...prev.slice(0, 9)]);
      
      // ذخیره خطا در دیتابیس
      await saveTestToDb(errorResult);
    } finally {
      setIsLoading(false);
    }
  };

  const selectedModelInfo = MODELS.find(m => m.id === selectedModel);

  return (
    <>
      <Helmet>
        <title>تست مدل‌های AI | DentalAI</title>
      </Helmet>

      <Container maxWidth="xl">
        <Stack spacing={3}>
          {/* Header */}
          <Stack direction="row" alignItems="center" spacing={2}>
            <Iconify icon="carbon:ai-status" width={40} />
            <Box>
              <Typography variant="h4">تست مدل‌های AI</Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                آزمایش و مقایسه مدل‌های مختلف برای تشخیص Cephalometric Landmarks
              </Typography>
            </Box>
          </Stack>

          {/* Main Content */}
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
            {/* Left Panel - Configuration */}
            <Stack spacing={3} sx={{ width: { xs: '100%', md: '400px' } }}>
              {/* Model Selection */}
              <Card>
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h6">⚙️ تنظیمات</Typography>

                    <FormControl fullWidth>
                      <InputLabel>مدل AI</InputLabel>
                      <Select
                        value={selectedModel}
                        label="مدل AI"
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        {MODELS.map((model) => (
                          <MenuItem key={model.id} value={model.id}>
                            <Stack direction="row" spacing={1} alignItems="center" sx={{ width: '100%' }}>
                              <Box
                                sx={{
                                  width: 8,
                                  height: 8,
                                  borderRadius: '50%',
                                  bgcolor: model.color,
                                }}
                              />
                              <Box sx={{ flex: 1 }}>
                                <Typography variant="body2">{model.name}</Typography>
                                <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                  {model.provider}
                                </Typography>
                              </Box>
                            </Stack>
                          </MenuItem>
                        ))}
                      </Select>
                    </FormControl>

                    {selectedModelInfo && (
                      <Alert 
                        severity={selectedModelInfo.isLocal ? "success" : "info"} 
                        icon={<Iconify icon={selectedModelInfo.isLocal ? "carbon:checkmark-filled" : "carbon:information"} />}
                        sx={{ display: 'none' }}
                      >
                        {selectedModelInfo.description}
                        {selectedModelInfo.isLocal && (
                          <Typography variant="caption" component="div" sx={{ mt: 1 }}>
                            💡 این مدل روی سرور محلی شما اجرا می‌شود. نیازی به API Key نیست!
                            <br />
                            🔗 اطمینان حاصل کنید سرویس روی سرور در حال اجراست.
                          </Typography>
                        )}
                      </Alert>
                    )}

                    {/* API Key فقط برای مدل‌های غیر محلی */}
                    {!selectedModelInfo?.isLocal && (
                      <TextField
                        fullWidth
                        type="password"
                        label="OpenRouter API Key"
                        value={apiKey}
                        onChange={(e) => setApiKey(e.target.value)}
                        placeholder="sk-or-v1-..."
                        helperText={
                          <Stack direction="row" spacing={0.5} component="span">
                            <span>دریافت از</span>
                            <Box
                              component="a"
                              href="https://openrouter.ai/keys"
                              target="_blank"
                              rel="noopener noreferrer"
                              sx={{ color: 'primary.main', textDecoration: 'none' }}
                            >
                              openrouter.ai/keys
                            </Box>
                          </Stack>
                        }
                      />
                    )}

                    <FormControlLabel
                      control={
                        <Switch
                          checked={autoScale}
                          onChange={(e) => setAutoScale(e.target.checked)}
                          color="primary"
                        />
                      }
                      label={
                        <Stack>
                          <Typography variant="body2">تصحیح خودکار مقیاس</Typography>
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            اصلاح مختصات بر اساس ابعاد تصویر
                          </Typography>
                        </Stack>
                      }
                      sx={{ display: 'none' }}
                    />

                    {/* Contour Detection Switch - برای همه مدل‌های محلی */}
                    {selectedModelInfo?.isLocal && (selectedModel === 'local/hrnet-w32' || selectedModel.startsWith('local/aariz')) && (
                      <>
                        <Divider sx={{ display: 'none' }} />
                        <FormControlLabel
                          control={
                            <Switch
                              checked={enableContours}
                              onChange={(e) => {
                                setEnableContours(e.target.checked);
                                if (!e.target.checked) {
                                  setContours(null);
                                  setContourError(null);
                                }
                              }}
                              color="primary"
                            />
                          }
                          label={
                            <Stack>
                              <Typography variant="body2">
                                🎨 تشخیص خودکار حدود (Contours)
                              </Typography>
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                تشخیص حدود بافت نرم، استخوان‌ها و دندان‌ها
                              </Typography>
                            </Stack>
                          }
                          sx={{ display: 'none' }}
                        />

                      </>
                    )}

                    {imageSize && (
                      <Alert severity="info" icon={<Iconify icon="carbon:ruler" />}>
                        <Typography variant="caption">
                          ابعاد تصویر: {imageSize.width} × {imageSize.height} پیکسل
                        </Typography>
                      </Alert>
                    )}

                    <Divider />

                    <Typography variant="h6">📷 آپلود تصویر</Typography>

                    <Upload
                      file={imageFile}
                      onDrop={handleDrop}
                      onDelete={handleRemoveFile}
                      accept={{ 'image/*': [] }}
                    />

                    {imagePreview && (
                      <Box
                        component="img"
                        src={imagePreview}
                        sx={{
                          width: '100%',
                          height: 200,
                          objectFit: 'contain',
                          borderRadius: 1,
                          bgcolor: 'background.neutral',
                        }}
                      />
                    )}
                  </Stack>
                </CardContent>
              </Card>

              {/* Test Button */}
              <Button
                fullWidth
                size="large"
                variant="contained"
                color="primary"
                onClick={handleTest}
                disabled={isLoading || !imageFile || (!selectedModelInfo?.isLocal && !apiKey)}
                startIcon={
                  isLoading ? (
                    <Iconify icon="line-md:loading-loop" />
                  ) : (
                    <Iconify icon="carbon:play-filled" />
                  )
                }
              >
                {isLoading ? 'در حال تست...' : 'شروع تست'}
              </Button>
            </Stack>

            {/* Right Panel - Results */}
            <Stack spacing={3} sx={{ flex: 1 }}>
              {/* Loading */}
              {isLoading && (
                <Card>
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="h6">⏳ در حال پردازش...</Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                          {selectedModelInfo?.name}
                        </Typography>
                      </Stack>
                      <LinearProgress />
                    </Stack>
                  </CardContent>
                </Card>
              )}

              {/* Error */}
              {error && (
                <Alert severity="error" onClose={() => setError(null)}>
                  <Typography variant="subtitle2">خطا</Typography>
                  <Typography variant="body2">{error}</Typography>
                </Alert>
              )}

              {/* Contour Detection Status */}
              {enableContours && selectedModelInfo?.isLocal && (selectedModel === 'local/hrnet-w32' || selectedModel.startsWith('local/aariz')) && (
                <Card>
                    <CardContent>
                      <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Stack>
                          <Typography variant="h6">🎨 تشخیص حدود</Typography>
                          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            روش: RyanChoi Advanced (پیشرفته)
                          </Typography>
                        </Stack>
                        {isDetectingContours && (
                          <Chip 
                            icon={<Iconify icon="line-md:loading-loop" />}
                            label="در حال تشخیص..." 
                            color="info" 
                            size="small" 
                          />
                        )}
                      </Stack>
                        
                        {contourError && (
                          <Alert severity="error" onClose={() => setContourError(null)}>
                            <Typography variant="body2">{contourError}</Typography>
                          </Alert>
                        )}
                        
                        {contours && contours.success && (
                          <Alert severity="success">
                            <Typography variant="body2">
                              ✅ {contours.num_regions} ناحیه تشخیص داده شد
                            </Typography>
                            <Stack direction="row" spacing={1} sx={{ mt: 1, flexWrap: 'wrap', gap: 0.5 }}>
                              {Object.entries(contours.contours).map(([region, data]) => (
                                <Chip
                                  key={region}
                                  label={`${region.replace(/_/g, ' ')}: ${data.success ? `${data.num_points || 0} نقاط` : 'ناموفق'}`}
                                  color={data.success ? 'success' : 'error'}
                                  size="small"
                                  variant="outlined"
                                />
                              ))}
                            </Stack>
                          </Alert>
                        )}
                      </Stack>
                    </CardContent>
                  </Card>
              )}

              {/* Custom Contour Settings */}
              {enableContours && selectedModelInfo?.isLocal && (selectedModel === 'local/hrnet-w32' || selectedModel.startsWith('local/aariz')) && result?.success && (
                <Card>
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">⚙️ تنظیمات سفارشی کانتورها</Typography>
                        <Button
                          variant="contained"
                          color="primary"
                          size="small"
                          onClick={detectContoursWithCustomSettings}
                          disabled={isDetectingContours}
                          startIcon={<Iconify icon="carbon:play-filled" />}
                        >
                          رسم با تنظیمات جدید
                        </Button>
                      </Stack>
                      
                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                        تنظیمات هر ناحیه را تغییر دهید و با دکمه بالا نتیجه را مشاهده کنید
                      </Typography>
                      
                      {availableContourRegions.map((region) => {
                        const settings = customContourSettings[region] || {};
                        const regionName = region.replace(/_/g, ' ');
                        
                        return (
                          <Accordion key={region}>
                            <AccordionSummary expandIcon={<Iconify icon="carbon:chevron-down" />}>
                              <Typography variant="subtitle2">{regionName}</Typography>
                            </AccordionSummary>
                            <AccordionDetails>
                              <Stack spacing={2}>
                                {/* Search Area Ratio */}
                                <Box>
                                  <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                    Search Area Ratio: {settings.search_area_ratio?.toFixed(2) || 0}
                                  </Typography>
                                  <Slider
                                    value={settings.search_area_ratio || 0.3}
                                    min={0.1}
                                    max={1.0}
                                    step={0.05}
                                    onChange={(e, value) => updateCustomSetting(region, 'search_area_ratio', value)}
                                    marks={[
                                      { value: 0.1, label: '0.1' },
                                      { value: 0.5, label: '0.5' },
                                      { value: 1.0, label: '1.0' },
                                    ]}
                                  />
                                </Box>
                                
                                {/* Edge Thresholds */}
                                <Stack direction="row" spacing={2}>
                                  <Box sx={{ flex: 1 }}>
                                    <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                      Edge Low: {settings.edge_threshold_low || 30}
                                    </Typography>
                                    <Slider
                                      value={settings.edge_threshold_low || 30}
                                      min={10}
                                      max={100}
                                      step={5}
                                      onChange={(e, value) => updateCustomSetting(region, 'edge_threshold_low', value)}
                                    />
                                  </Box>
                                  <Box sx={{ flex: 1 }}>
                                    <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                      Edge High: {settings.edge_threshold_high || 100}
                                    </Typography>
                                    <Slider
                                      value={settings.edge_threshold_high || 100}
                                      min={50}
                                      max={200}
                                      step={5}
                                      onChange={(e, value) => updateCustomSetting(region, 'edge_threshold_high', value)}
                                    />
                                  </Box>
                                </Stack>
                                
                                {/* Smooth Sigma */}
                                <Box>
                                  <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                    Smooth Sigma: {settings.smooth_sigma?.toFixed(1) || 1.0}
                                  </Typography>
                                  <Slider
                                    value={settings.smooth_sigma || 1.0}
                                    min={0.5}
                                    max={5.0}
                                    step={0.1}
                                    onChange={(e, value) => updateCustomSetting(region, 'smooth_sigma', value)}
                                  />
                                </Box>
                                
                                {/* Contrast */}
                                <Stack direction="row" spacing={2}>
                                  <Box sx={{ flex: 1 }}>
                                    <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                      Contrast Alpha: {settings.contrast_alpha?.toFixed(1) || 1.5}
                                    </Typography>
                                    <Slider
                                      value={settings.contrast_alpha || 1.5}
                                      min={0.5}
                                      max={3.0}
                                      step={0.1}
                                      onChange={(e, value) => updateCustomSetting(region, 'contrast_alpha', value)}
                                    />
                                  </Box>
                                  <Box sx={{ flex: 1 }}>
                                    <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                      Contrast Beta: {settings.contrast_beta || -30}
                                    </Typography>
                                    <Slider
                                      value={settings.contrast_beta || -30}
                                      min={-100}
                                      max={100}
                                      step={5}
                                      onChange={(e, value) => updateCustomSetting(region, 'contrast_beta', value)}
                                    />
                                  </Box>
                                </Stack>
                                
                                {/* Max Points */}
                                <Box>
                                  <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                    Max Points: {settings.max_points || 100}
                                  </Typography>
                                  <Slider
                                    value={settings.max_points || 100}
                                    min={20}
                                    max={500}
                                    step={10}
                                    onChange={(e, value) => updateCustomSetting(region, 'max_points', value)}
                                  />
                                </Box>
                                
                                {/* Simplify Epsilon */}
                                <Box>
                                  <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                    Simplify Epsilon: {settings.simplify_epsilon?.toFixed(1) || 1.5}
                                  </Typography>
                                  <Slider
                                    value={settings.simplify_epsilon || 1.5}
                                    min={0.5}
                                    max={5.0}
                                    step={0.1}
                                    onChange={(e, value) => updateCustomSetting(region, 'simplify_epsilon', value)}
                                  />
                                </Box>
                                
                                {/* Circle/Ellipse Fit Options */}
                                <Stack spacing={1}>
                                  <FormControlLabel
                                    control={
                                      <Switch
                                        checked={settings.use_circle_fit || false}
                                        onChange={(e) => {
                                          updateCustomSetting(region, 'use_circle_fit', e.target.checked);
                                          if (e.target.checked) {
                                            updateCustomSetting(region, 'use_ellipse_fit', false);
                                          }
                                        }}
                                        size="small"
                                      />
                                    }
                                    label="استفاده از Circle Fit"
                                  />
                                  {settings.use_circle_fit && (
                                    <Box sx={{ pl: 4 }}>
                                      <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                        Circle Points: {settings.circle_fit_points || 32}
                                      </Typography>
                                      <Slider
                                        value={settings.circle_fit_points || 32}
                                        min={16}
                                        max={128}
                                        step={4}
                                        onChange={(e, value) => updateCustomSetting(region, 'circle_fit_points', value)}
                                      />
                                    </Box>
                                  )}
                                  
                                  <FormControlLabel
                                    control={
                                      <Switch
                                        checked={settings.use_ellipse_fit || false}
                                        onChange={(e) => {
                                          updateCustomSetting(region, 'use_ellipse_fit', e.target.checked);
                                          if (e.target.checked) {
                                            updateCustomSetting(region, 'use_circle_fit', false);
                                          }
                                        }}
                                        size="small"
                                      />
                                    }
                                    label="استفاده از Ellipse Fit"
                                  />
                                  {settings.use_ellipse_fit && (
                                    <Box sx={{ pl: 4 }}>
                                      <Typography variant="caption" sx={{ mb: 1, display: 'block' }}>
                                        Ellipse Points: {settings.ellipse_fit_points || 48}
                                      </Typography>
                                      <Slider
                                        value={settings.ellipse_fit_points || 48}
                                        min={24}
                                        max={128}
                                        step={4}
                                        onChange={(e, value) => updateCustomSetting(region, 'ellipse_fit_points', value)}
                                      />
                                    </Box>
                                  )}
                                </Stack>
                                
                                {/* Reset to Default */}
                                <Button
                                  size="small"
                                  variant="outlined"
                                  onClick={() => {
                                    if (defaultSettings[region]) {
                                      setCustomContourSettings(prev => ({
                                        ...prev,
                                        [region]: { ...defaultSettings[region] },
                                      }));
                                    }
                                  }}
                                  startIcon={<Iconify icon="carbon:reset" />}
                                >
                                  بازنشانی به پیش‌فرض
                                </Button>
                              </Stack>
                            </AccordionDetails>
                          </Accordion>
                        );
                      })}
                    </Stack>
                  </CardContent>
                </Card>
              )}

              {/* Visualization */}
              {result && result.success && result.response?.landmarks && imagePreview && (
                <Card>
                  <CardContent>
                    <Stack spacing={2}>
                      <Typography variant="h6">📊 نمایش بصری و ویرایش Landmarks</Typography>
                      <AdvancedCephalometricVisualizer
                        imageUrl={imagePreview}
                        landmarks={result.response.landmarks}
                        imageSize={imageSize}
                        contours={contours?.contours || null}
                        onLandmarksChange={(updatedLandmarks) => {
                          setResult({
                            ...result,
                            response: {
                              ...result.response,
                              landmarks: updatedLandmarks
                            }
                          });
                        }}
                        showMeasurements
                      />
                    </Stack>
                  </CardContent>
                </Card>
              )}

              {/* Results */}
              {result && (
                <Card
                  sx={{
                    bgcolor: result.success ? alpha('#00FF00', 0.05) : alpha('#FF0000', 0.05),
                    border: `1px solid ${result.success ? alpha('#00FF00', 0.2) : alpha('#FF0000', 0.2)}`,
                  }}
                >
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">
                          {result.success ? '✅ نتیجه' : '❌ خطا'}
                        </Typography>
                        <Stack direction="row" spacing={1}>
                          {result.metadata?.processingTime && (
                            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                              ⏱️ {result.metadata.processingTime}s
                            </Typography>
                          )}
                          {result.metadata?.scaling?.applied && (
                            <Chip
                              size="small"
                              label={`Scaled ${result.metadata.scaling.factor}x`}
                              color="success"
                              variant="outlined"
                            />
                          )}
                        </Stack>
                      </Stack>

                      {result.success && (
                        <Stack spacing={1}>
                          {result.metadata?.tokensUsed && (
                            <Stack direction="row" spacing={2}>
                              <Typography variant="caption">
                                📊 Tokens: {result.metadata.tokensUsed.total_tokens || 'N/A'}
                              </Typography>
                              {result.metadata.tokensUsed.prompt_tokens && (
                                <Typography variant="caption">
                                  Prompt: {result.metadata.tokensUsed.prompt_tokens}
                                </Typography>
                              )}
                              {result.metadata.tokensUsed.completion_tokens && (
                                <Typography variant="caption">
                                  Completion: {result.metadata.tokensUsed.completion_tokens}
                                </Typography>
                              )}
                            </Stack>
                          )}
                          
                          {result.metadata?.scaling?.applied && (
                            <Alert severity="success" sx={{ py: 0.5 }}>
                              <Typography variant="caption">
                                🔧 مقیاس خودکار اعمال شد: ضریب {result.metadata.scaling.factor}
                                {' '}(تصویر {result.metadata.scaling.imageSize.width}×{result.metadata.scaling.imageSize.height})
                              </Typography>
                            </Alert>
                          )}
                        </Stack>
                      )}

                      <TextField
                        fullWidth
                        multiline
                        rows={20}
                        value={JSON.stringify(result.success ? result.response : { error: result.error }, null, 2)}
                        InputProps={{
                          readOnly: true,
                          sx: {
                            fontFamily: 'monospace',
                            fontSize: '0.875rem',
                            bgcolor: 'background.neutral',
                          },
                        }}
                      />

                      <Stack direction="row" spacing={1}>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => {
                            navigator.clipboard.writeText(JSON.stringify(result.response, null, 2));
                          }}
                          startIcon={<Iconify icon="solar:copy-bold" />}
                        >
                          کپی JSON
                        </Button>
                        <Button
                          size="small"
                          variant="outlined"
                          onClick={() => {
                            const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
                            const url = URL.createObjectURL(blob);
                            const a = document.createElement('a');
                            a.href = url;
                            a.download = `test-result-${Date.now()}.json`;
                            a.click();
                          }}
                          startIcon={<Iconify icon="solar:download-bold" />}
                        >
                          دانلود
                        </Button>
                      </Stack>
                    </Stack>
                  </CardContent>
                </Card>
              )}

            </Stack>
          </Stack>

          {/* Test History Section - Bottom of Page */}
          <Card>
            <CardContent>
              <Stack spacing={3}>
                <Stack direction="row" justifyContent="space-between" alignItems="center">
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Iconify icon="solar:history-bold" width={32} />
                    <Box>
                      <Typography variant="h5">تاریخچه آنالیزها</Typography>
                      <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
                        تمام تست‌های انجام شده در دیتابیس ذخیره می‌شوند
                      </Typography>
                    </Box>
                  </Stack>
                  <Stack direction="row" spacing={1}>
                    {isSavingToDb && (
                      <Chip 
                        icon={<Iconify icon="solar:diskette-bold" width={16} />}
                        label="در حال ذخیره..." 
                        color="info" 
                        size="small"
                      />
                    )}
                    <Button
                      variant="outlined"
                      size="small"
                      onClick={loadTestHistory}
                      disabled={isLoadingHistory}
                      startIcon={<Iconify icon="solar:refresh-bold" />}
                    >
                      {isLoadingHistory ? 'در حال بارگذاری...' : 'به‌روزرسانی'}
                    </Button>
                  </Stack>
                </Stack>

                <Divider />

                {isLoadingHistory ? (
                  <Box sx={{ py: 4 }}>
                    <LinearProgress />
                    <Typography variant="body2" sx={{ textAlign: 'center', mt: 2, color: 'text.secondary' }}>
                      در حال بارگذاری تاریخچه...
                    </Typography>
                  </Box>
                ) : dbTestHistory.length === 0 ? (
                  <Box sx={{ py: 6, textAlign: 'center' }}>
                    <Iconify icon="solar:history-bold" width={64} sx={{ color: 'text.disabled', mb: 2 }} />
                    <Typography variant="body1" sx={{ color: 'text.secondary', mb: 1 }}>
                      هنوز هیچ آنالیزی ذخیره نشده است
                    </Typography>
                    <Typography variant="caption" sx={{ color: 'text.disabled' }}>
                      پس از انجام اولین تست، نتایج در اینجا نمایش داده می‌شوند
                    </Typography>
                  </Box>
                ) : (
                  <Stack spacing={1.5}>
                    {dbTestHistory.map((test) => {
                      let landmarks = null;
                      try {
                        landmarks = test.landmarks 
                          ? (typeof test.landmarks === 'string' ? JSON.parse(test.landmarks) : test.landmarks) 
                          : null;
                      } catch (e) {
                        console.error('خطا در parse کردن landmarks:', e);
                        landmarks = null;
                      }
                      const landmarkCount = landmarks ? Object.keys(landmarks).length : 0;
                      
                      return (
                        <Card
                          key={test.id}
                          sx={{
                            cursor: 'pointer',
                            transition: 'all 0.2s',
                            border: `1px solid ${test.success ? alpha('#10B981', 0.3) : alpha('#EF4444', 0.3)}`,
                            bgcolor: test.success ? alpha('#10B981', 0.02) : alpha('#EF4444', 0.02),
                            '&:hover': {
                              bgcolor: test.success ? alpha('#10B981', 0.05) : alpha('#EF4444', 0.05),
                              border: `1px solid ${test.success ? alpha('#10B981', 0.5) : alpha('#EF4444', 0.5)}`,
                              boxShadow: 2,
                            },
                          }}
                          onClick={() => {
                            if (test.success && landmarks) {
                              setResult({
                                success: true,
                                model: {
                                  id: test.modelId,
                                  name: test.modelName,
                                  provider: test.modelProvider,
                                },
                                response: { landmarks },
                                metadata: {
                                  processingTime: test.processingTime?.toFixed(2),
                                  timestamp: new Date(test.createdAt).toLocaleString('fa-IR'),
                                },
                                rawResponse: test.rawResponse,
                              });
                              // Scroll to top to see result
                              window.scrollTo({ top: 0, behavior: 'smooth' });
                            }
                          }}
                        >
                          <CardContent sx={{ p: 2.5 }}>
                            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} justifyContent="space-between" alignItems={{ xs: 'flex-start', sm: 'center' }}>
                              <Stack direction="row" spacing={2} alignItems="center" sx={{ flex: 1 }}>
                                <Box
                                  sx={{
                                    width: 40,
                                    height: 40,
                                    borderRadius: 1,
                                    display: 'flex',
                                    alignItems: 'center',
                                    justifyContent: 'center',
                                    bgcolor: test.success ? alpha('#10B981', 0.1) : alpha('#EF4444', 0.1),
                                  }}
                                >
                                  <Iconify 
                                    icon={test.success ? "solar:check-circle-bold" : "solar:close-circle-bold"} 
                                    width={24} 
                                    sx={{ color: test.success ? '#10B981' : '#EF4444' }}
                                  />
                                </Box>
                                
                                <Stack spacing={0.5} sx={{ flex: 1 }}>
                                  <Stack direction="row" spacing={1} alignItems="center">
                                    <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
                                      {test.modelName}
                                    </Typography>
                                    <Chip 
                                      label={test.modelProvider} 
                                      size="small" 
                                      variant="outlined"
                                      sx={{ height: 20, fontSize: '0.7rem' }}
                                    />
                                  </Stack>
                                  
                                  <Stack direction="row" spacing={2} alignItems="center" sx={{ flexWrap: 'wrap', gap: 1 }}>
                                    {test.success && landmarkCount > 0 && (
                                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                        {landmarkCount} لندمارک تشخیص داده شده
                                      </Typography>
                                    )}
                                    {test.processingTime && (
                                      <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                        ⏱️ {parseFloat(test.processingTime).toFixed(2)} ثانیه
                                      </Typography>
                                    )}
                                    {test.confidence && (
                                      <Chip
                                        size="small"
                                        label={`اعتماد: ${Math.round(test.confidence * 100)}%`}
                                        color="success"
                                        variant="outlined"
                                        sx={{ height: 20 }}
                                      />
                                    )}
                                    {test.error && (
                                      <Typography variant="caption" sx={{ color: 'error.main' }}>
                                        {test.error.substring(0, 50)}...
                                      </Typography>
                                    )}
                                  </Stack>
                                </Stack>
                              </Stack>

                              <Stack direction="row" spacing={2} alignItems="center">
                                <Typography variant="caption" sx={{ color: 'text.secondary', whiteSpace: 'nowrap' }}>
                                  {new Date(test.createdAt).toLocaleString('fa-IR', {
                                    year: 'numeric',
                                    month: 'long',
                                    day: 'numeric',
                                    hour: '2-digit',
                                    minute: '2-digit',
                                  })}
                                </Typography>
                                {test.success && (
                                  <Iconify 
                                    icon="solar:arrow-left-bold" 
                                    width={20} 
                                    sx={{ color: 'text.secondary' }}
                                  />
                                )}
                              </Stack>
                            </Stack>
                          </CardContent>
                        </Card>
                      );
                    })}
                  </Stack>
                )}
              </Stack>
            </CardContent>
          </Card>
        </Stack>
      </Container>
    </>
  );
}
