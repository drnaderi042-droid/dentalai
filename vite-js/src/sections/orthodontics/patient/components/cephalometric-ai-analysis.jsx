import { memo, useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import Switch from '@mui/material/Switch';
import Dialog from '@mui/material/Dialog';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import Skeleton from '@mui/material/Skeleton';
import { useTheme } from '@mui/material/styles';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import IconButton from '@mui/material/IconButton';
import FormControl from '@mui/material/FormControl';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import ToggleButton from '@mui/material/ToggleButton';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import CircularProgress from '@mui/material/CircularProgress';
import FormControlLabel from '@mui/material/FormControlLabel';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';

import { getApiUrl, getImageUrl, getAiServiceUrl } from 'src/utils/url-helpers';

import { CONFIG } from 'src/config-global';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { AdvancedCephalometricVisualizer } from 'src/components/advanced-cephalometric-visualizer';

// Settings Icon SVG Component
const SettingsIcon = () => (
  <Box
    component="svg"
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    sx={{
      width: '24px',
      height: '24px',
      color: 'inherit',
      '& path': {
        fill: 'currentColor',
      }
    }}
  >
    <g clipPath="url(#clip0_4418_5060)">
      <path opacity="0.4" d="M2 12.8794V11.1194C2 10.0794 2.85 9.21945 3.9 9.21945C5.71 9.21945 6.45 7.93945 5.54 6.36945C5.02 5.46945 5.33 4.29945 6.24 3.77945L7.97 2.78945C8.76 2.31945 9.78 2.59945 10.25 3.38945L10.36 3.57945C11.26 5.14945 12.74 5.14945 13.65 3.57945L13.76 3.38945C14.23 2.59945 15.25 2.31945 16.04 2.78945L17.77 3.77945C18.68 4.29945 18.99 5.46945 18.47 6.36945C17.56 7.93945 18.3 9.21945 20.11 9.21945C21.15 9.21945 22.01 10.0694 22.01 11.1194V12.8794C22.01 13.9194 21.16 14.7794 20.11 14.7794C18.3 14.7794 17.56 16.0594 18.47 17.6294C18.99 18.5394 18.68 19.6994 17.77 20.2194L16.04 21.2094C15.25 21.6794 14.23 21.3995 13.76 20.6094L13.65 20.4194C12.75 18.8494 11.27 18.8494 10.36 20.4194L10.25 20.6094C9.78 21.3995 8.76 21.6794 7.97 21.2094L6.24 20.2194C5.33 19.6994 5.02 18.5294 5.54 17.6294C6.45 16.0594 5.71 14.7794 3.9 14.7794C2.85 14.7794 2 13.9194 2 12.8794Z" fill="currentColor"/>
      <path d="M12 15.25C13.7949 15.25 15.25 13.7949 15.25 12C15.25 10.2051 13.7949 8.75 12 8.75C10.2051 8.75 8.75 10.2051 8.75 12C8.75 13.7949 10.2051 15.25 12 15.25Z" fill="currentColor"/>
    </g>
    <defs>
      <clipPath id="clip0_4418_5060">
        <rect width="24" height="24" fill="white"/>
      </clipPath>
    </defs>
  </Box>
);

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
    id: 'local/aariz-768',
    name: 'Aariz 768x768',
    provider: '',
    description: 'Aariz Model 768x768 - High Resolution (MRE: 1.118mm, SDR: 86.92%)',
    color: MODERN_COLORS[0],
    isLocal: true,
    requiresApiKey: false,
    modelPath: 'Aariz/checkpoint_best_768.pth',
  },
  {
    id: 'local/cldetection2023',
    name: 'CLdetection2023',
    provider: '',
    description: 'CLdetection2023 Model - 38 landmarks (MICCAI CLDetection2023 Challenge)',
    color: MODERN_COLORS[1],
    isLocal: true,
    requiresApiKey: false,
    modelPath: 'CLdetection2023/model_pretrained_on_train_and_val.pth',
  },
];

// ----------------------------------------------------------------------

export function CephalometricAIAnalysis({
  onLandmarksDetected,
  lateralImageUrl,
  isEditMode: externalIsEditMode,
  onEditModeChange,
  initialLandmarks,
  showCoordinateSystem = false,
  onSaveAnalysis,
  saving = false,
  hasPatient = false,
  analysisHistory = [],
  selectedAnalysisIndex = null,
  onSelectedAnalysisIndexChange,
  onDeleteAnalysis,
  deleteDialogOpen = false,
  onDeleteDialogOpenChange,
  analysisToDelete = null,
  onAnalysisToDeleteChange,
  deleting = false,
  // New props for image selection
  selectedImageIndex = 0,
  onSelectedImageIndexChange,
  lateralImages = [],
  onImageUpload,
  patientId = null,
  onDeleteImage,
  isUploadingImage = false, // 🔧 FIX: New prop to track upload state
  viewMode: externalViewMode = 'normal', // View mode: 'normal', 'coordinate', 'hard-tissue-only'
  onViewModeChange, // Callback to update view mode in parent
  cephalometricTable = null, // Full table data from parent
  selectedAnalysisType = 'steiner', // Analysis type for displaying lines
}) {
  // Internal view mode state (if parent doesn't control it)
  const [internalViewMode, setInternalViewMode] = useState(externalViewMode || 'normal');
  const viewMode = externalViewMode !== undefined ? externalViewMode : internalViewMode;
  const handleViewModeChange = (newMode) => {
    if (onViewModeChange) {
      onViewModeChange(newMode);
    } else {
      setInternalViewMode(newMode);
    }
  };
  // Log props on component mount/update (only in development, and only on significant changes)
  const prevPropsRef = useRef({ lateralImageUrl, initialLandmarks });
  useEffect(() => {
  if (process.env.NODE_ENV === 'development') {
      const propsChanged = prevPropsRef.current.lateralImageUrl !== lateralImageUrl ||
                          prevPropsRef.current.initialLandmarks !== initialLandmarks;
      if (propsChanged) {
        prevPropsRef.current = { lateralImageUrl, initialLandmarks };
  }
    }
  }, [lateralImageUrl, initialLandmarks]);
  
  // Default to CLdetection2023 model
  const [selectedModel, setSelectedModel] = useState(MODELS.find(m => m.id === 'local/cldetection2023')?.id || MODELS[0].id);
  const [apiKey, setApiKey] = useState('');
  const [imageFile, setImageFile] = useState(null);
  const [imageFiles, setImageFiles] = useState([]); // Array for multiple uploaded files
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [autoScale, setAutoScale] = useState(true);
  const [isSavingToDb, setIsSavingToDb] = useState(false);
  const [lateralImageLoaded, setLateralImageLoaded] = useState(false);
  
  // Contour detection states
  const [enableContours, setEnableContours] = useState(true);
  const [contourMethod, setContourMethod] = useState('auto');
  const [showAdvancedSettings, setShowAdvancedSettings] = useState(false);
  const [selectedContourRegions, setSelectedContourRegions] = useState([]);
  const [isDetectingContours, setIsDetectingContours] = useState(false);
  const [contours, setContours] = useState(null);
  const [contourError, setContourError] = useState(null);
  const [availableContourRegions, setAvailableContourRegions] = useState([]);
  
  // Calibration detection states
  const [calibrationPoints, setCalibrationPoints] = useState([]);
  const [PIXEL_TO_MM_CONVERSION, setPIXEL_TO_MM_CONVERSION] = useState(0.11);
  const [isDetectingCalibration, setIsDetectingCalibration] = useState(false);
  const [calibrationError, setCalibrationError] = useState(null);
  
  // Computer Vision parameters for p1/p2 detection
  const [cvParams, setCvParams] = useState({
    enabled: true, // 🔧 FIX: استفاده از Computer Vision به صورت پیش‌فرض برای هر دو مدل فعال است
    selectedMethod: 'auto', // 'auto', 'local_maxima', 'adaptive_threshold', 'gradient', 'multiscale', 'histogram_peak', 'watershed', 'template_matching'
    searchArea: {
      x: 0.50,  // 50% از چپ
      y: 0.00,  // 0% از بالا
      width: 0.50,  // 50% عرض (50-100%)
      height: 0.50,  // 50% ارتفاع (0-50%)
    },
    brightness: {
      threshold: 150,  // حداقل brightness
      minContrast: 25,  // حداقل contrast با neighbors
    },
    verticalAlignment: {
      maxDx: 20,  // حداکثر فاصله افقی (pixels)
      minDy: 0.03,  // حداقل فاصله عمودی (3% از ارتفاع تصویر)
      maxDy: 0.08,  // حداکثر فاصله عمودی (8% از ارتفاع تصویر)
    },
    detection: {
      maxPoints: 20,  // حداکثر تعداد نقاط برای بررسی
      duplicateThreshold: 10,  // فاصله minimum برای duplicate detection (pixels)
    }
  });
  
  // Custom contour settings per region
  const [customContourSettings, setCustomContourSettings] = useState({});
  
  // Track unsaved changes
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);
  const [isSavingChanges, setIsSavingChanges] = useState(false);
  const [displayAnalysisType, setDisplayAnalysisType] = useState(selectedAnalysisType || 'steiner');
  
  // Update displayAnalysisType when selectedAnalysisType prop changes
  useEffect(() => {
    if (selectedAnalysisType) {
      setDisplayAnalysisType(selectedAnalysisType);
    }
  }, [selectedAnalysisType]);
  
  // Theme for dark mode detection
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  // Edit mode state - به صورت پیش‌فرض غیرفعال (view mode)
  // اگر از parent پاس داده شده باشد، از آن استفاده کن، در غیر این صورت state محلی
  const [internalIsEditMode, setInternalIsEditMode] = useState(false);
  const isEditMode = externalIsEditMode !== undefined ? externalIsEditMode : internalIsEditMode;
  const setIsEditMode = useCallback((value) => {
    if (onEditModeChange) {
      onEditModeChange(value);
    } else {
      setInternalIsEditMode(value);
    }
  }, [onEditModeChange]);
  
  // محاسبه ضریب تبدیل پیکسل به میلی‌متر بر اساس نقاط کالیبراسیون (p1, p2)
  // این نقاط فاصله 1cm (10mm) دارند و برای تعیین مقیاس واقعی تصویر استفاده می‌شوند
  const calculatePixelToMmConversion = useCallback((detectedPoints) => {
    if (detectedPoints.length < 2) {
      return 0.11; // More realistic fallback based on typical radiograph DPI
    }
    
    
    // اگر دقیقاً 2 نقطه داریم، آن‌ها p2 و p1 هستند (به ترتیب)
    // p2 بالایی است، p1 پایینی است، و فاصله آن‌ها 1cm (10mm) است
    if (detectedPoints.length === 2) {
      const p2 = detectedPoints[0]; // Upper point
      const p1 = detectedPoints[1]; // Lower point
      
      const dx = Math.abs(p2.x - p1.x);
      const dy = Math.abs(p2.y - p1.y);
      
      // 🔧 ساده‌سازی: چون p1 و p2 عمودی هستند (x یکسان)، فقط از dy استفاده می‌کنیم
      // فاصله = فقط تفاوت عمودی (dy)
      const distance = dy;
      
      
      // 🔧 حذف validation - همیشه از نقاط استفاده کن (بدون بررسی isVertical یا mmPerPixel range)
      // Calculate conversion assuming these points are 10mm apart
      // استفاده از dy به جای distance کامل (چون x یکسان است)
      const mmPerPixel = 10 / distance;
      
      // 🔧 حذف validation - همیشه mmPerPixel را برگردان (بدون بررسی range)
      const dpi = 25.4 / mmPerPixel;
      return mmPerPixel;
    }
    
    // Enhanced algorithm for more than 2 points
    // Look for pairs that are vertically aligned and ~10mm apart
    let bestConversion = null;
    let bestPair = null;
    
    for (let i = 0; i < detectedPoints.length; i++) {
      for (let j = i + 1; j < detectedPoints.length; j++) {
        const p1 = detectedPoints[i];
        const p2 = detectedPoints[j];
        
        const dx = Math.abs(p2.x - p1.x);
        const dy = Math.abs(p2.y - p1.y);
        const distance = Math.sqrt(dx * dx + dy * dy);
        
        if (distance < 15 || distance > 200) continue; // Skip unrealistic distances
        
        // Prefer vertically aligned pairs (small dx relative to dy)
        const isVertical = dx < 30 && dy > 50;
        
        // For calibration markers 10mm apart, typical pixel distance is 60-120 pixels
        // This means conversion factor should be 0.08-0.17 mm/pixel
        const mmPerPixel = 10 / distance;
        
        // Accept conversion factors in the realistic range for medical radiographs
        if (mmPerPixel >= 0.08 && mmPerPixel <= 0.17) {
          
          bestConversion = mmPerPixel;
          bestPair = [p1, p2];
          
          // If this is a vertical pair, use it immediately (better match)
          if (isVertical) break;
        }
      }
      if (bestConversion && bestPair && Math.abs(bestPair[0].x - bestPair[1].x) < 30) break;
    }
    
    if (bestConversion) {
      return bestConversion;
    }
    
    // If no valid pair found, use enhanced fallback based on typical radiograph settings
    // Most digital radiographs use 150-300 DPI, which translates to 0.1-0.17 mm/pixel
    return 0.11; // 0.11 mm/pixel is typical for medical radiographs
  }, []);
  
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
      use_ellipse_fit: true,
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


  // Ref to track the last loaded lateralImageUrl to avoid unnecessary reloads
  const lastLoadedLateralImageUrlRef = useRef(null);
  
  // Ref to track the last loaded initialLandmarks to avoid unnecessary reloads
  const lastLoadedInitialLandmarksRef = useRef(null);
  
  // 🔧 FIX: Track the image URL for which landmarks were last loaded
  // This prevents loading old landmarks for a new image
  const lastLoadedLandmarksImageUrlRef = useRef(null);
  
  // Ref to track if landmarks have been manually changed (prevent overwrite)
  const hasManualLandmarkChangesRef = useRef(false);
  
  // Ref to track if component just mounted (for forcing initial load)
  const isInitialMountRef = useRef(true);
  
  // Ref to prevent multiple simultaneous fetches
  const isLoadingImageRef = useRef(false);
  
  // 🔧 FIX: Ref to track if a new analysis was just completed (to prevent overwriting with initialLandmarks)
  const hasNewAnalysisRef = useRef(false);
  
  // 🔧 FIX: Track the last result we notified about to prevent re-notification loops
  const lastNotifiedResultRef = useRef(null);
  
  // Ref to track last processed selectedAnalysisIndex to prevent duplicate processing
  const lastProcessedAnalysisIndexRef = useRef(null);
  
  // 🔧 FIX: دیگر نیازی به currentBlobUrlRef نیست چون از base64 استفاده می‌کنیم
  
  // Reset refs and result when component mounts or key changes
  // This ensures that when the component remounts (e.g., after page refresh), landmarks are reloaded
  useEffect(() => {
    // Don't reset refs here - let them be reset naturally when landmarks/image are loaded
    // But reset result to ensure landmarks are reloaded
    setResult(null);
    // Reset manual changes flag on mount
    hasManualLandmarkChangesRef.current = false;
    // 🔧 FIX: Reset new analysis flag on mount
    hasNewAnalysisRef.current = false;
    // 🔧 FIX: Reset notification ref on mount
    lastNotifiedResultRef.current = null;
    // Reset processed analysis index ref
    lastProcessedAnalysisIndexRef.current = null;
  }, []); // Empty dependency array - only run on mount
  
  // Reset manual changes flag when image changes
  useEffect(() => {
    hasManualLandmarkChangesRef.current = false;
  }, [imagePreview]);

  // 🔧 FIX: Ensure newest image is selected by default when images change
  useEffect(() => {
    if (lateralImages && lateralImages.length > 0 && onSelectedImageIndexChange) {
      // Check if current selected index is out of bounds or invalid
      const isOutOfBounds = selectedImageIndex === null ||
                           selectedImageIndex >= lateralImages.length ||
                           selectedImageIndex < 0;
      
      // If out of bounds, automatically select the newest image (index 0)
      if (isOutOfBounds) {
        onSelectedImageIndexChange(0);
        
        // 🔧 FIX: Clear analysis history selection when changing images
        if (onSelectedAnalysisIndexChange && selectedAnalysisIndex !== null) {
          onSelectedAnalysisIndexChange('');
        }
      }
    }
  }, [lateralImages, onSelectedImageIndexChange, selectedImageIndex, onSelectedAnalysisIndexChange, selectedAnalysisIndex]);

  // 🔧 FIX: Load landmarks when analysis is selected from history
  useEffect(() => {
    // Skip if this analysis index was already processed
    if (lastProcessedAnalysisIndexRef.current === selectedAnalysisIndex) {
      return;
    }
    

    // 🔧 FIX: Don't load analysis from history if a new image is being uploaded
    // Each analysis is specific to one image - when a new image is uploaded, old analyses should not be loaded
    if (isUploadingImage) {
      return;
    }

    if (selectedAnalysisIndex !== null &&
        selectedAnalysisIndex !== undefined &&
        analysisHistory &&
        analysisHistory.length > 0 &&
        selectedAnalysisIndex < analysisHistory.length) {
      
      const selectedAnalysis = analysisHistory[selectedAnalysisIndex];

      if (selectedAnalysis?.landmarks && Object.keys(selectedAnalysis.landmarks).length > 0) {
        // Load the landmarks from the selected analysis
        const landmarksFromHistory = selectedAnalysis.landmarks;
        

        // 🔧 FIX: Create new result WITHOUT clearing first (no setTimeout, no intermediate null state)
        const newResult = {
          success: true,
          response: {
            landmarks: landmarksFromHistory,
          },
          metadata: {
            source: 'history',
            analysisType: selectedAnalysis.analysisType || selectedAnalysis.currentAnalysisType || 'steiner',
            timestamp: selectedAnalysis.timestamp,
            selectedAnalysisIndex: selectedAnalysisIndex
          }
        };

        // Update refs
        hasManualLandmarkChangesRef.current = false;
        hasNewAnalysisRef.current = false;
        const landmarksString = JSON.stringify(landmarksFromHistory);
        lastLoadedInitialLandmarksRef.current = landmarksString;

        // Update calibration points if available
        if (landmarksFromHistory.p1 && landmarksFromHistory.p2) {
          const calibrationPointsArray = [
            landmarksFromHistory.p2, // p2 (top)
            landmarksFromHistory.p1, // p1 (bottom)
          ];
          setCalibrationPoints(calibrationPointsArray);
          
          // Calculate conversion factor
          const conversionFactor = calculatePixelToMmConversion(calibrationPointsArray);
          setPIXEL_TO_MM_CONVERSION(conversionFactor);
        } else {
          setCalibrationPoints([]);
          setPIXEL_TO_MM_CONVERSION(0.11);
        }

        // Set result (this will trigger handleLandmarksChange which will call onLandmarksDetected)
        setResult(newResult);
        
        // Mark this analysis index as processed
        lastProcessedAnalysisIndexRef.current = selectedAnalysisIndex;


      } else {
        
        // Clear result if no landmarks found
        setResult(null);
        setCalibrationPoints([]);
        setPIXEL_TO_MM_CONVERSION(0.11);
        lastProcessedAnalysisIndexRef.current = selectedAnalysisIndex;
      }
    } else {
      // Clear selection if invalid index
      if (selectedAnalysisIndex !== null && selectedAnalysisIndex !== undefined) {
        setResult(null);
        setCalibrationPoints([]);
        setPIXEL_TO_MM_CONVERSION(0.11);
        lastProcessedAnalysisIndexRef.current = selectedAnalysisIndex;
      }
    }
  }, [selectedAnalysisIndex, analysisHistory, onLandmarksDetected, calculatePixelToMmConversion, isUploadingImage, lateralImageUrl]);

  // Helper function to convert File to base64 data URL
  const fileToDataURL = (file) => new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  // بارگذاری خودکار تصویر لترال از patient.images.lateral
  useEffect(() => {
    // اگر lateralImageUrl وجود ندارد، کاری نکن
      if (!lateralImageUrl) {
      lastLoadedLateralImageUrlRef.current = null;
      lastLoadedLandmarksImageUrlRef.current = null; // 🔧 FIX: Reset landmarks image URL ref
      isInitialMountRef.current = true;
      // 🔧 FIX: اگر URL خالی شد، همه state های analysis را reset کن
      setResult(null);
      setCalibrationPoints([]);
      setPIXEL_TO_MM_CONVERSION(0.11);
      hasManualLandmarkChangesRef.current = false;
      hasNewAnalysisRef.current = false;
      lastLoadedInitialLandmarksRef.current = null;
      return;
    }

    // بررسی اینکه آیا این URL قبلاً لود شده است یا نه
    const urlChanged = lastLoadedLateralImageUrlRef.current !== lateralImageUrl;
    const isInitialMount = isInitialMountRef.current;
    
    // 🔧 FIX: Check if we have an analysis selected from history
    const hasSelectedAnalysis = selectedAnalysisIndex !== null && selectedAnalysisIndex !== undefined && analysisHistory && analysisHistory.length > 0;


    // 🔧 FIX: اگر URL تغییر کرده، همه state های analysis را reset کن
    // هر آنالیز مختص به یک تصویر است - وقتی تصویر تغییر می‌کند، آنالیز قبلی باید پاک شود
    if (urlChanged) {
      // اگر در حال آپلود تصویر جدید هستیم، حتماً result را reset کن
      // چون تصویر جدید است و آنالیز قبلی مربوط به تصویر قبلی است
      if (isUploadingImage) {
        // 🔧 FIX: Reset result when uploading new image - each analysis is specific to one image
        setResult(null);
        setIsEditMode(false);
        setCalibrationPoints([]);
        setPIXEL_TO_MM_CONVERSION(0.11);
        hasManualLandmarkChangesRef.current = false;
        hasNewAnalysisRef.current = false;
        lastLoadedInitialLandmarksRef.current = null;
        lastLoadedLateralImageUrlRef.current = null;
        lastLoadedLandmarksImageUrlRef.current = null; // 🔧 FIX: Reset landmarks image URL ref for new image
        lastNotifiedResultRef.current = null; // 🔧 FIX: Reset notification ref for new image
      } else if (hasSelectedAnalysis) {
        // Don't reset result if analysis from history is selected (user explicitly selected it)
        lastLoadedLateralImageUrlRef.current = null;
      } else if (isInitialMount && analysisHistory && analysisHistory.length > 0) {
        // Don't reset result on initial mount if there's analysis history - let the selectedAnalysisIndex useEffect handle it
        lastLoadedLateralImageUrlRef.current = null;
      } else {
        
        // Reset all analysis-related state
        setResult(null);
        setIsEditMode(false);
        setCalibrationPoints([]);
        setPIXEL_TO_MM_CONVERSION(0.11);
        hasManualLandmarkChangesRef.current = false;
        hasNewAnalysisRef.current = false;
        lastLoadedInitialLandmarksRef.current = null;
        lastLoadedLateralImageUrlRef.current = null;
        lastLoadedLandmarksImageUrlRef.current = null; // 🔧 FIX: Reset landmarks image URL ref for new image
        lastNotifiedResultRef.current = null; // 🔧 FIX: Reset notification ref for new image
      }
    }

    // اگر URL تغییر نکرده و قبلاً لود شده، کاری نکن
    if (!urlChanged && !isInitialMount && lastLoadedLateralImageUrlRef.current === lateralImageUrl) {
      return; // تصویر قبلاً لود شده، نیازی به fetch مجدد نیست
    }
    
    // اگر در حال لود است، صبر کن
    if (isLoadingImageRef.current) {
      return;
    }
    
    // Set loading flag
    isLoadingImageRef.current = true;
    isInitialMountRef.current = false;
    
    const loadLateralImage = async () => {
      try {
        
        // 🔧 FIX: دیگر نیازی به revoke کردن blob URL نیست چون از base64 استفاده می‌کنیم
        
        // دانلود تصویر از URL
        const response = await fetch(lateralImageUrl);
        if (!response.ok) {
          isLoadingImageRef.current = false;
          return;
        }
        
        const blob = await response.blob();
        const fileName = lateralImageUrl.split('/').pop() || 'lateral-image.jpg';
        const file = new File([blob], fileName, { type: blob.type || 'image/jpeg' });
        
        // 🔧 FIX: استفاده از base64 data URL به جای blob URL
        const previewUrl = await fileToDataURL(file);
        
        setImagePreview(previewUrl);

        setImageFile(file);
        setLateralImageLoaded(true);
        lastLoadedLateralImageUrlRef.current = lateralImageUrl;

        // محاسبه ابعاد تصویر
        const img = new Image();
        img.onload = () => {
          const naturalWidth = img.naturalWidth || img.width;
          const naturalHeight = img.naturalHeight || img.height;
          setImageSize({ width: naturalWidth, height: naturalHeight });
          isLoadingImageRef.current = false;
        };
        img.onerror = (error) => {
          // 🔧 FIX: دیگر نیازی به بررسی blob URL نیست - فقط خطا را log می‌کنیم
          isLoadingImageRef.current = false;
        };
        img.src = previewUrl;
      } catch (err) {
        isLoadingImageRef.current = false;
      }
    };
    
    loadLateralImage();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lateralImageUrl, isUploadingImage]); // Note: selectedAnalysisIndex and analysisHistory checked inside but not in deps to avoid re-triggering

  // بارگذاری لندمارک‌های اولیه از patient data وقتی تصویر و لندمارک‌ها آماده هستند
  // 🔧 FIX: مهم - فقط زمانی لندمارک‌ها را لود کن که واقعاً متعلق به تصویر فعلی باشند
  useEffect(() => {

    // 🔧 FIX: Don't load landmarks if a new image is being uploaded
    // Each analysis is specific to one image - old landmarks should not be loaded for new image
    if (isUploadingImage) {
      return;
    }

    // 🔧 FIX: Don't load landmarks if the image URL has changed
    // This prevents loading old landmarks for a new image
    if (lateralImageUrl && lastLoadedLandmarksImageUrlRef.current && lastLoadedLandmarksImageUrlRef.current !== lateralImageUrl) {
      // Reset the ref for the new image
      lastLoadedLandmarksImageUrlRef.current = null;
      lastLoadedInitialLandmarksRef.current = null;
      return;
    }

    // فقط اگر تصویر لود شده است (imagePreview وجود دارد)
    if (!imagePreview) {
      return;
    }

    // اگر تصویر لود شده و لندمارک‌های اولیه وجود دارند
    if (initialLandmarks && Object.keys(initialLandmarks).length > 0) {
      // بررسی اینکه آیا این لندمارک‌ها قبلاً لود شده‌اند یا نه
      const landmarksString = JSON.stringify(initialLandmarks);
      const landmarksChanged = lastLoadedInitialLandmarksRef.current !== landmarksString;
      
      
      // 🔧 FIX: شرایط سخت‌گیرانه‌تر برای لود کردن لندمارک‌ها
      // لندمارک‌ها فقط در موارد زیر لود شوند:
      // 1. لندمارک‌ها تغییر کرده‌اند AND
      // 2. هیچ تغییر دستی انجام نشده AND
      // 3. یک آنالیز جدید انجام نشده AND
      // 4. result وجود ندارد یا نامعتبر است
      // 5. result از history نیست (اگر از history است، نباید دوباره لود شود)
      const resultIsFromHistory = result?.metadata?.source === 'history';
      const shouldLoad = !hasManualLandmarkChangesRef.current &&
                        !hasNewAnalysisRef.current &&
                        !resultIsFromHistory && // 🔧 FIX: Don't reload if result is already from history
                        (landmarksChanged || !result || !result.success || !result.response?.landmarks || Object.keys(result.response.landmarks).length === 0);
      
      
      if (shouldLoad) {
        
        // 🔧 FIX: Add timestamp to metadata to prevent duplicate notifications
        const timestamp = new Date().toISOString();
        const newResult = {
          success: true,
          response: {
            landmarks: initialLandmarks,
          },
          metadata: {
            source: 'initial',
            timestamp: timestamp,
          },
        };
        
        setResult(newResult);
        lastLoadedInitialLandmarksRef.current = landmarksString;
        // 🔧 FIX: Track the image URL for which landmarks were loaded
        if (lateralImageUrl) {
          lastLoadedLandmarksImageUrlRef.current = lateralImageUrl;
        }
        
        // 🔧 FIX: چک کنیم آیا calibration points از قبل در database ذخیره شده‌اند
        // p1 و p2 نقاط کالیبراسیون هستند که فاصله 1cm دارند
        const hasCalibrationInDb = initialLandmarks.p1 && initialLandmarks.p2;
        
        if (hasCalibrationInDb) {
          // نقاط کالیبراسیون از database لود شده‌اند - بررسی کنیم که آیا درست هستند
          
          // بررسی کنیم که آیا نقاط به صورت عمودی align هستند
          const {p2} = initialLandmarks;
          const {p1} = initialLandmarks;
          const dx = Math.abs(p2.x - p1.x);
          const dy = Math.abs(p2.y - p1.y);
          const isVertical = dx < 30 && dy > 50 && dy < 200;
          
          if (isVertical) {
            // نقاط درست هستند - از آنها استفاده کن
            const calibrationPointsArray = [
              initialLandmarks.p2, // p2 (top)
              initialLandmarks.p1, // p1 (bottom)
            ];
            setCalibrationPoints(calibrationPointsArray);
            
            const conversionFactor = calculatePixelToMmConversion(calibrationPointsArray);
            setPIXEL_TO_MM_CONVERSION(conversionFactor);
          } else {
            // نقاط درست نیستند - حذف کن و دوباره detect کن با ML model
            
            // حذف p1/p2 اشتباه از landmarks
            const { p1, p2, ...landmarksWithoutCalibration } = initialLandmarks;
            const cleanedLandmarks = landmarksWithoutCalibration;
            
            // Update result without incorrect p1/p2
            const cleanedResult = {
              ...newResult,
              response: {
                ...newResult.response,
                landmarks: cleanedLandmarks
              }
            };
            setResult(cleanedResult);
            
            // 🔧 FIX: Disabled auto-detection to prevent infinite loops
            // User can manually detect using Computer Vision button if needed
          }
        } else if (imageFile) {
          // 🔧 FIX: Disabled auto-detection to prevent infinite loops
          // User can manually detect using Computer Vision button if needed
        }
      }
    } else {
      // اگر تصویر لود شده اما لندمارک‌ها وجود ندارند
      // Reset ref if landmarks are cleared
      lastLoadedInitialLandmarksRef.current = null;
      
      // 🔧 FIX: اگر لندمارک‌ها پاک شده‌اند، result را هم reset کن
      if (result) {
        setResult(null);
        setCalibrationPoints([]);
        setPIXEL_TO_MM_CONVERSION(0.11);
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imagePreview, initialLandmarks, isUploadingImage, lateralImageUrl]); // 🔧 FIX: Added isUploadingImage and lateralImageUrl to prevent loading old landmarks

  // بارگذاری نواحی contours وقتی مدل تغییر می‌کند - غیرفعال شده است
  // useEffect(() => {
  //   loadAvailableContourRegions();
  // }, [selectedModel]);

  // بارگذاری لیست نواحی قابل تشخیص برای contours - غیرفعال شده است
  const loadAvailableContourRegions = async () => {
    // Contour detection service has been removed
    // This function is kept for potential future use but is currently disabled
    
    
    // try {
    //   let contourApiUrl = null;
    //   if (selectedModel && selectedModel.startsWith('local/aariz')) {
    //     contourApiUrl = 'http://localhost:5001/contour-regions';
    //   }
    //   
    //   if (!contourApiUrl) {
    //     return;
    //   }
    //   
    //   const response = await fetch(contourApiUrl);
    //   const data = await response.json();
    //   if (data.success) {
    //     setAvailableContourRegions(Object.keys(data.regions || {}));
    //     setSelectedContourRegions(Object.keys(data.regions || {}));
    //   }
    // } catch (err) {
    //   const isLocalModel = selectedModel && selectedModel.startsWith('local/aariz');
    //   if (isLocalModel) {
    //   }
    // }
  };

  const saveTestToDb = async (testResult) => {
    setIsSavingToDb(true);
    try {
      if (!testResult?.model) {
        return;
      }

      let imageUrlForDb = null;
      // 🔧 FIX: base64 data URL را مستقیماً استفاده می‌کنیم
      if (imagePreview && imagePreview.startsWith('data:')) {
        imageUrlForDb = imagePreview;
      }

      const payload = {
        modelId: testResult.model.id || 'unknown',
        modelName: testResult.model.name || 'Unknown Model',
        modelProvider: testResult.model.provider || '',
        imageUrl: imageUrlForDb,
        imageSize: imageSize ? JSON.stringify(imageSize) : null,
        prompt: '',
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
        userId: null,
      };

      // ذخیره آنالیز جدید (بدون حذف تاریخچه قبلی)
      const response = await fetch(getApiUrl('/api/ai-model-tests'), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();
      
      // Handle response if needed
    } catch (err) {
      // Error handling if needed
    } finally {
      setIsSavingToDb(false);
    }
  };

  const handleDrop = useCallback(async (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const newFiles = [];
      
      for (const file of acceptedFiles) {
        try {
          // 🔧 FIX: استفاده از base64 data URL به جای blob URL
          const previewUrl = await fileToDataURL(file);
          
          const fileObj = {
            id: `local-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
            file,
            preview: previewUrl,
            name: file.name,
            size: file.size,
            type: file.type,
            uploadedAt: new Date(),
            isLocal: true, // Mark as local (not yet uploaded to server)
          };
          
          newFiles.push(fileObj);
          
          // Set first file as selected
          if (newFiles.length === 1) {
            setImageFile(file);
            setImagePreview(previewUrl);
            setLateralImageLoaded(false);
            
            const img = new Image();
            img.onload = () => {
              const naturalWidth = img.naturalWidth || img.width;
              const naturalHeight = img.naturalHeight || img.height;
              setImageSize({ width: naturalWidth, height: naturalHeight });
            };
            img.onerror = (error) => {
              setImagePreview(null);
              setError('خطا در بارگذاری تصویر');
            };
            img.src = previewUrl;
          }
        } catch (error) {
          // Error handling if needed
        }
      }
      
      if (newFiles.length > 0) {
        setImageFiles(prev => [...prev, ...newFiles]);
        setError(null);
        setResult(null);
        
        // 🔧 FIX: Upload to server if onImageUpload callback is provided
        if (onImageUpload && patientId) {
          try {
            const filesToUpload = newFiles.map(f => f.file);
            await onImageUpload(filesToUpload, 'lateral');
            // After successful upload, clear local files as they're now on the server
            // The images will be loaded from lateralImages prop
            setImageFiles(prev => prev.filter(f => !newFiles.some(nf => nf.id === f.id)));
            
            // 🔧 FIX: Automatically select the newest image after upload
            // Since images are sorted newest first, the newest uploaded image will be at index 0
            if (onSelectedImageIndexChange) {
              // Use setTimeout to ensure the lateralImages have been updated first
              setTimeout(() => {
                onSelectedImageIndexChange(0);
              }, 100);
            }
            
            // 🔧 FIX: Clear analysis history selection when uploading new images
            if (onSelectedAnalysisIndexChange && selectedAnalysisIndex !== null && selectedAnalysisIndex !== '') {
              onSelectedAnalysisIndexChange('');
            }
            
          } catch (error) {
            // Keep the files in local state even if upload fails
          }
        }
      }
    }
  }, [onImageUpload, patientId, onSelectedAnalysisIndexChange, onSelectedImageIndexChange, selectedAnalysisIndex]);

  const handleRemoveFile = useCallback((fileToRemove) => {
    // Handle both single file removal and array file removal
    if (fileToRemove) {
      // Remove from imageFiles array
      setImageFiles(prev => {
        const updated = prev.filter(f => {
          if (typeof fileToRemove === 'string') {
            return f.id !== fileToRemove;
          }
          return f.id !== fileToRemove.id && f.file !== fileToRemove;
        });
        
        // If removed file was the selected one, select first remaining or clear
        const removedWasSelected = prev.some(f => 
          (typeof fileToRemove === 'string' && f.id === fileToRemove) ||
          (f.id === fileToRemove.id || f.file === fileToRemove)
        ) && imageFile && (
          (typeof fileToRemove === 'string' && prev.find(f => f.id === fileToRemove)?.file === imageFile) ||
          (fileToRemove === imageFile || (fileToRemove.id && prev.find(f => f.id === fileToRemove.id)?.file === imageFile))
        );
        
        if (removedWasSelected) {
          if (updated.length > 0) {
            setImageFile(updated[0].file);
            setImagePreview(updated[0].preview);
          } else {
            setImageFile(null);
            setImagePreview(null);
            setImageSize(null);
          }
        }
        
        return updated;
      });
    } else {
      // Clear all
      setImageFile(null);
      setImageFiles([]);
      setImagePreview(null);
      setImageSize(null);
    }
    
    setResult(null);
    setError(null);
    setLateralImageLoaded(false); // Allow lateral image to load again if URL still exists
  }, [imageFile]);

  const convertImageToBase64 = (file) => 
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const calculateScalingFactor = (landmarks, actualSize) => {
    if (!landmarks || !actualSize) return 1;

    const xCoords = Object.values(landmarks).map(lm => lm.x);
    const yCoords = Object.values(landmarks).map(lm => lm.y);
    
    const maxX = Math.max(...xCoords);
    const maxY = Math.max(...yCoords);
    
    const scaleX = actualSize.width / maxX;
    const scaleY = actualSize.height / maxY;
    
    const scalingFactor = (scaleX + scaleY) / 2;
    
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
      // Use backend API endpoint instead of direct Python server
      const backendUrl = CONFIG?.site?.serverUrl || getApiUrl('');
      const contourApiUrl = `${backendUrl}/api/ai/detect-contours`;
      
        if (!selectedModel || (!selectedModel.startsWith('local/aariz') && selectedModel !== 'local/cldetection2023' && selectedModel !== 'local/fast-cpu-512' && selectedModel !== 'local/p1-p2-fast-cpu-512' && selectedModel !== 'train_p1_p2_heatmap')) {
        throw new Error('Contour detection only available for local models');
      }
      
      const payload = {
        image_base64: base64Image,
        landmarks,
        method: contourMethod,
        selectedRegions: selectedContourRegions,
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
      
      if (data.success && data.contours) {
        let contoursDict = {};
        
        if (data.contours && typeof data.contours === 'object') {
          if (data.contours.contours && typeof data.contours.contours === 'object') {
            contoursDict = data.contours.contours;
          } else if (!data.contours.success && Object.keys(data.contours).length > 0) {
            contoursDict = data.contours;
          } else {
            contoursDict = data.contours;
          }
        }
        
        const validContours = {};
        Object.keys(contoursDict).forEach(region => {
          const regionData = contoursDict[region];
          if (regionData && typeof regionData === 'object' && regionData.contour && Array.isArray(regionData.contour) && regionData.contour.length > 0) {
            validContours[region] = regionData;
          }
        });
        
        if (Object.keys(validContours).length === 0) {
          setContourError('هیچ کانتور معتبری یافت نشد');
          setContours(null);
          return;
        }
        
        const contoursData = {
          success: true,
          num_regions: Object.keys(validContours).length,
          contours: validContours,
        };
        setContours(contoursData);
      } else {
        const errorMsg = data.error || data.contours?.error || 'Contour detection failed';
        throw new Error(errorMsg);
      }
    } catch (err) {
      // خطای detect-contours را به صورت warning نمایش بده، نه error (این خطا بحرانی نیست)
      setContourError(null); // خطا را نمایش نده
      setContours(null);
      // toast نمایش نده - این خطا بحرانی نیست
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
      let contourApiUrl = null;
      if (selectedModel && selectedModel.startsWith('local/aariz')) {
        contourApiUrl = 'http://localhost:5001/detect-contours';
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
      
      if (data.success && data.contours) {
        let contoursDict = {};
        
        if (data.contours && typeof data.contours === 'object') {
          if (data.contours.contours && typeof data.contours.contours === 'object') {
            contoursDict = data.contours.contours;
          } else if (!data.contours.success && Object.keys(data.contours).length > 0) {
            contoursDict = data.contours;
          } else {
            contoursDict = data.contours;
          }
        }
        
        const validContours = {};
        Object.keys(contoursDict).forEach(region => {
          const regionData = contoursDict[region];
          if (regionData && typeof regionData === 'object' && regionData.contour && Array.isArray(regionData.contour) && regionData.contour.length > 0) {
            validContours[region] = regionData;
          }
        });
        
        if (Object.keys(validContours).length === 0) {
          setContourError('هیچ کانتور معتبری یافت نشد');
          setContours(null);
          return;
        }
        
        const contoursData = {
          success: true,
          num_regions: Object.keys(validContours).length,
          contours: validContours,
        };
        setContours(contoursData);
      } else {
        const errorMsg = data.error || data.contours?.error || 'Contour detection failed';
        throw new Error(errorMsg);
      }
    } catch (err) {
      // خطای detect-contours را به صورت warning نمایش بده، نه error (این خطا بحرانی نیست)
      setContourError(null); // خطا را نمایش نده
      setContours(null);
      // toast نمایش نده - این خطا بحرانی نیست
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

  // Detect calibration points in the image
  const detectCalibrationAndCalculateConversion = async (imageData) => {
    setIsDetectingCalibration(true);
    setCalibrationError(null);
    
    try {
      const img = new Image();
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');
      
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
        img.src = imageData;
      });
      
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);
      
      // Detect calibration points in upper right corner
      const detectedPoints = await detectCalibrationPoints(imageData, img.width, img.height);
      
      if (detectedPoints && detectedPoints.length >= 2) {
        setCalibrationPoints(detectedPoints);
        
        // Convert calibration points to landmark format for visualization
        // p1 و p2 نقاط کالیبراسیون هستند که فاصله 1cm دارند
        // فقط p1 و p2 را ایجاد کن (نه p3, p4, ...)
        const calibrationLandmarks = {};
        if (detectedPoints.length >= 2) {
          // p2 بالایی است، p1 پایینی است
          const p2 = detectedPoints[0]; // Upper point
          const p1 = detectedPoints[1]; // Lower point
          calibrationLandmarks.p1 = p1;
          calibrationLandmarks.p2 = p2;
        }
        
        // Calculate conversion factor (even if validation failed, use default)
        let conversionFactor;
        try {
          conversionFactor = calculatePixelToMmConversion(detectedPoints);
        } catch (error) {
          conversionFactor = 0.11; // Default
        }
        
        setPIXEL_TO_MM_CONVERSION(conversionFactor);
        
        
        return {
          success: true,
          calibrationPoints: calibrationLandmarks,
          conversionFactor,
          detectedPointsCount: detectedPoints.length
        };
      } 
        // 🔧 اگر هیچ نقطه‌ای پیدا نشد، خطا برگردان (نه default points)
        const errorMsg = `❌ Computer Vision failed to detect calibration points. Found ${detectedPoints?.length || 0} points. Please adjust CV parameters or use ML detection.`;
        // Don't show error message to user in frontend
        // setCalibrationError(errorMsg);
        setPIXEL_TO_MM_CONVERSION(0.11); // Use default conversion
        
        return {
          success: false,
          error: errorMsg,
          detectedPointsCount: detectedPoints?.length || 0
        };
      
    } catch (error) {
      const errorMsg = `❌ Computer Vision detection failed: ${error.message}`;
      setCalibrationError(errorMsg);
      setPIXEL_TO_MM_CONVERSION(0.11); // Use default conversion
      
      return {
        success: false,
        error: errorMsg,
        detectedPointsCount: 0
      };
    } finally {
      setIsDetectingCalibration(false);
    }
  };

  // Helper function to detect calibration points using ML model (heatmap-based) or CV
  const detectCalibrationPoints = async (imageData, imageWidth, imageHeight) => {
    // 🔧 FIX: Computer Vision برای مدل cldetection غیرفعال است
    const isCldetectionModel = selectedModel === 'local/cldetection2023' || selectedModel === 'local/fast-cpu-512';
    
    // 🔧 FIX: Computer Vision به صورت پیش‌فرض فعال است - ML غیرفعال
    // اما برای cldetection، CV را غیرفعال می‌کنیم
    if (cvParams.enabled && !isCldetectionModel) {
      return detectCalibrationPointsCV(imageData, imageWidth, imageHeight, cvParams);
    }
    if (isCldetectionModel) {
      return null;
    }
    
    // 🔧 DISABLED: ML detection برای p1/p2 غیرفعال شده - فقط از Computer Vision استفاده می‌شود
    // Default: استفاده از ML برای تشخیص p1/p2 (غیرفعال)
    try {
      // Try ML model first (heatmap-based, < 10px accuracy)
      // Use unified AI API server endpoint
      
      const apiUrl = getAiServiceUrl('/detect-p1p2');
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: imageData,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        
        // 🔧 FIX: استفاده از نقاط حتی اگر success=false باشد (validation fail در backend)
        // اما p1 و p2 وجود داشته باشند
        if ((result.success || (result.p1 && result.p2)) && result.p1 && result.p2) {
          const isBackendValidationFailed = !result.success && result.p1 && result.p2;
          
          // Backend validation handled
          
          
          // 🔧 VALIDATION: بررسی صحت نتایج
          const dx = Math.abs(result.p2.x - result.p1.x);
          const dy = Math.abs(result.p2.y - result.p1.y);
          const distance = Math.sqrt(dx * dx + dy * dy);
          
          // 🔧 بهبود validation: اگر dx خیلی بزرگ است، ممکن است p1 و p2 جابجا شده باشند
          // در تصاویر cephalometric، p2 باید بالاتر از p1 باشد (y کوچکتر)
          // و dx باید کوچک باشد (عمودی)
          let p1_final = result.p1;
          let p2_final = result.p2;
          
          // اگر dy منفی است (p1 بالاتر از p2)، جابجا کن
          if (result.p1.y < result.p2.y) {
            [p1_final, p2_final] = [result.p2, result.p1];
          }
          
          // محاسبه مجدد dx و dy با نقاط صحیح
          const dx_corrected = Math.abs(p2_final.x - p1_final.x);
          const dy_corrected = Math.abs(p2_final.y - p1_final.y);
          
          // 🔧 FIX: کاهش سخت‌گیری validation - dy را از 50-200 به 30-250 تغییر دادیم
          // همچنین dx threshold را از 50 به 60 افزایش دادیم
          const isVertical = dx_corrected < 60 && dy_corrected > 30 && dy_corrected < 250;
          const confidence = result.confidence || 0;
          const minConfidence = 0.25; // کاهش از 0.3 به 0.25
          
          // 🔧 FIX: کاهش سخت‌گیری upper right check - margin را بیشتر کردیم
          const isInUpperRight = p1_final.x >= imageWidth * 0.50 && p1_final.x <= imageWidth * 0.99 &&
                                 p1_final.y >= 0 && p1_final.y <= imageHeight * 0.40 &&
                                 p2_final.x >= imageWidth * 0.50 && p2_final.x <= imageWidth * 0.99 &&
                                 p2_final.y >= 0 && p2_final.y <= imageHeight * 0.40;
          
          
          // 🔧 FIX: حتی اگر validation fail کند، p1/p2 را برگردان (برای استفاده)
          // اما با هشدار که ممکن است دقیق نباشند
          if (!isVertical || confidence < minConfidence || !isInUpperRight) {
            
            // بازگرداندن نقاط برای استفاده (با هشدار)
            const points = [
              {
                x: p2_final.x,  // p2 is top point (upper)
                y: p2_final.y,
                brightness: 255,
                contrast: 100,
              },
              {
                x: p1_final.x,  // p1 is bottom point (lower)
                y: p1_final.y,
                brightness: 255,
                contrast: 100,
              },
            ];
            
            
            return points; // بازگرداندن نقاط برای استفاده
          } 
            // Convert to format expected by rest of code (با نقاط تصحیح شده)
            const points = [
              {
                x: p2_final.x,  // p2 is top point (upper)
                y: p2_final.y,
                brightness: 255,
                contrast: 100,
              },
              {
                x: p1_final.x,  // p1 is bottom point (lower)
                y: p1_final.y,
                brightness: 255,
                contrast: 100,
              },
            ];
            
            
            return points;
          
        } 
        
      } else {
        const errorData = await response.json().catch(() => ({}));
      }
    } catch (error) {
      // Error handling if needed
    }
    
    // بدون fallback - فقط ML detection استفاده می‌شود
    return null;
  };
  
  // Helper function to detect calibration points using Computer Vision
  const detectCalibrationPointsCV = async (imageData, imageWidth, imageHeight, params) => new Promise((resolve) => {
      // محاسبه search area از درصدها
      const searchArea = {
        x: Math.floor(imageWidth * params.searchArea.x),
        y: Math.floor(imageHeight * params.searchArea.y),
        width: Math.floor(imageWidth * params.searchArea.width),
        height: Math.floor(imageHeight * params.searchArea.height)
      };
      

      // Create image from base64
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement('canvas');
        const imageCtx = canvas.getContext('2d');
        
        canvas.width = img.width;
        canvas.height = img.height;
        imageCtx.drawImage(img, 0, 0);
        
        // Get image data from search area
        const data = imageCtx.getImageData(searchArea.x, searchArea.y, searchArea.width, searchArea.height);
        const points = findHighContrastPointsCV(data, searchArea, params, img.height); // پاس دادن ارتفاع کل تصویر
        
        // 🔧 همیشه نقاط را برگردان (حتی اگر 0 یا 1 نقطه باشد) - کاربر می‌تواند دستی جابجا کند
        if (points && points.length > 0) {
          // اگر کمتر از 2 نقطه باشد، نقاط پیش‌فرض اضافه کن
          if (points.length === 1) {
            // یک نقطه دیگر در نزدیکی نقطه اول اضافه کن
            const p1 = points[0];
            const defaultP2 = {
              x: p1.x,
              y: p1.y - 100, // 100 pixels above
              brightness: 200,
              contrast: 50
            };
            points.push(defaultP2);
          } else if (points.length === 0) {
            // اگر هیچ نقطه‌ای پیدا نشد، 2 نقطه پیش‌فرض در مرکز search area اضافه کن
            const defaultP1 = {
              x: searchArea.x + searchArea.width / 2,
              y: searchArea.y + searchArea.height / 2 + 50,
              brightness: 200,
              contrast: 50
            };
            const defaultP2 = {
              x: searchArea.x + searchArea.width / 2,
              y: searchArea.y + searchArea.height / 2 - 50,
              brightness: 200,
              contrast: 50
            };
            points.push(defaultP2, defaultP1);
          }
          resolve(points);
        } else {
          // Fallback: 2 نقطه پیش‌فرض در مرکز search area
          const defaultP1 = {
            x: searchArea.x + searchArea.width / 2,
            y: searchArea.y + searchArea.height / 2 + 50,
            brightness: 200,
            contrast: 50
          };
          const defaultP2 = {
            x: searchArea.x + searchArea.width / 2,
            y: searchArea.y + searchArea.height / 2 - 50,
            brightness: 200,
            contrast: 50
          };
          resolve([defaultP2, defaultP1]);
        }
      };
      img.onerror = () => {
        // 🔧 حتی در صورت خطا، نقاط پیش‌فرض برگردان
        const defaultP1 = {
          x: searchArea.x + searchArea.width / 2,
          y: searchArea.y + searchArea.height / 2 + 50,
          brightness: 200,
          contrast: 50
        };
        const defaultP2 = {
          x: searchArea.x + searchArea.width / 2,
          y: searchArea.y + searchArea.height / 2 - 50,
          brightness: 200,
          contrast: 50
        };
        resolve([defaultP2, defaultP1]);
      };
      img.src = imageData;
    });

  // Find high contrast points using Computer Vision with multiple advanced algorithms
  const findHighContrastPointsCV = (imageData, searchArea, params, imageHeight) => {
    const points = [];
    const {data} = imageData;
    const {width} = imageData;
    const {height} = imageData; // ارتفاع search area
    
    // استفاده از پارامترهای قابل تنظیم
    const brightnessThreshold = params.brightness.threshold;
    const {minContrast} = params.brightness;
    const {duplicateThreshold} = params.detection;
    const selectedMethod = params.selectedMethod || 'auto';
    
    // اگر روش خاصی انتخاب شده، فقط آن را اجرا کن
    const shouldRunMethod = (methodName) => selectedMethod === 'auto' || selectedMethod === methodName;
    
    // ============================================
    // Method 1: Local Maxima Detection (Improved)
    // ============================================
    if (shouldRunMethod('local_maxima')) {
      for (let y = 3; y < height - 3; y += 1) {
        for (let x = 3; x < width - 3; x += 1) {
          const pixelIndex = (y * width + x) * 4;
        const pixel = data[pixelIndex]; // Red channel (grayscale)
        
        if (pixel > brightnessThreshold) {
          // Check if this is a LOCAL MAXIMUM
          let isLocalMax = true;
          let neighborSum = 0;
          let neighborCount = 0;
          
          // Check 3x3 neighborhood
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              if (dx === 0 && dy === 0) continue;
              
              const ny = y + dy;
              const nx = x + dx;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const neighborIdx = (ny * width + nx) * 4;
                const neighborVal = data[neighborIdx];
                neighborSum += neighborVal;
                neighborCount++;
                
                // If neighbor is brighter, this is not a local max
                if (neighborVal > pixel) {
                  isLocalMax = false;
                  break;
                }
              }
            }
            if (!isLocalMax) break;
          }
          
          if (isLocalMax && neighborCount > 0) {
            const avgNeighbor = neighborSum / neighborCount;
            
            // استفاده از minContrast از params
            if (pixel - avgNeighbor > minContrast) {
              // Convert to image coordinates
              const imageX = searchArea.x + x;
              const imageY = searchArea.y + y;
              
              // Avoid duplicates (using duplicateThreshold from params)
              const isDuplicate = points.some(p =>
                Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
              );
              
              if (!isDuplicate) {
                points.push({ 
                  x: imageX, 
                  y: imageY, 
                  brightness: pixel,
                  contrast: pixel - avgNeighbor,
                  method: 'local_maxima'
                });
              }
            }
          }
        }
      }
    }
    
    }
    
    // ============================================
    // Method 2: Adaptive Thresholding + Contour Detection
    // ============================================
    if (shouldRunMethod('adaptive_threshold') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // Calculate adaptive threshold based on image statistics
      let pixelSum = 0;
      let pixelCount = 0;
      for (let y = 0; y < height; y += 2) {
        for (let x = 0; x < width; x += 2) {
          const pixelIndex = (y * width + x) * 4;
          pixelSum += data[pixelIndex];
          pixelCount++;
        }
      }
      const meanBrightness = pixelSum / pixelCount;
      const adaptiveThreshold = Math.max(brightnessThreshold, meanBrightness + 30);
      
      // Find bright regions using adaptive threshold
      for (let y = 5; y < height - 5; y += 2) {
        for (let x = 5; x < width - 5; x += 2) {
          const pixelIndex = (y * width + x) * 4;
          const pixel = data[pixelIndex];
          
          if (pixel > adaptiveThreshold) {
            // Check if this is a bright region center
            let regionSum = 0;
            let regionCount = 0;
            let maxInRegion = pixel;
            
            // Check 5x5 region
            for (let dy = -2; dy <= 2; dy++) {
              for (let dx = -2; dx <= 2; dx++) {
                const ny = y + dy;
                const nx = x + dx;
                if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                  const neighborIdx = (ny * width + nx) * 4;
                  const neighborVal = data[neighborIdx];
                  regionSum += neighborVal;
                  regionCount++;
                  if (neighborVal > maxInRegion) maxInRegion = neighborVal;
                }
              }
            }
            
            // If this pixel is the maximum in its region
            if (pixel === maxInRegion && regionCount > 0) {
              const avgRegion = regionSum / regionCount;
              if (pixel - avgRegion > minContrast * 0.8) { // Slightly lower threshold
                const imageX = searchArea.x + x;
                const imageY = searchArea.y + y;
                
                const isDuplicate = points.some(p =>
                  Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
                );
                
                if (!isDuplicate) {
                  points.push({ 
                    x: imageX, 
                    y: imageY, 
                    brightness: pixel,
                    contrast: pixel - avgRegion,
                    method: 'adaptive_threshold'
                  });
                }
              }
            }
          }
        }
      }
      
    }
    
    // ============================================
    // Method 3: Gradient-based Detection (Edge-based)
    // ============================================
    if (shouldRunMethod('gradient') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // Calculate gradients to find edges and bright spots
      for (let y = 3; y < height - 3; y += 1) {
        for (let x = 3; x < width - 3; x += 1) {
          const pixelIndex = (y * width + x) * 4;
          const pixel = data[pixelIndex];
          
          if (pixel > brightnessThreshold * 0.7) { // Lower threshold for gradient method
            // Calculate gradient magnitude
            const gx = data[((y) * width + (x + 1)) * 4] - data[((y) * width + (x - 1)) * 4];
            const gy = data[((y + 1) * width + (x)) * 4] - data[((y - 1) * width + (x)) * 4];
            const gradientMagnitude = Math.sqrt(gx * gx + gy * gy);
            
            // Bright spots with high gradient (edges of bright regions)
            if (gradientMagnitude > 20 && pixel > brightnessThreshold * 0.6) {
              // Check if this is a local maximum in gradient
              let isGradientMax = true;
              let neighborGradSum = 0;
              let neighborGradCount = 0;
              
              for (let dy = -1; dy <= 1; dy++) {
                for (let dx = -1; dx <= 1; dx++) {
                  if (dx === 0 && dy === 0) continue;
                  const ny = y + dy;
                  const nx = x + dx;
                  if (nx >= 1 && nx < width - 1 && ny >= 1 && ny < height - 1) {
                    const nGx = data[((ny) * width + (nx + 1)) * 4] - data[((ny) * width + (nx - 1)) * 4];
                    const nGy = data[((ny + 1) * width + (nx)) * 4] - data[((ny - 1) * width + (nx)) * 4];
                    const nGradient = Math.sqrt(nGx * nGx + nGy * nGy);
                    neighborGradSum += nGradient;
                    neighborGradCount++;
                    if (nGradient > gradientMagnitude) isGradientMax = false;
                  }
                }
              }
              
              if (isGradientMax && neighborGradCount > 0) {
                const avgGradient = neighborGradSum / neighborGradCount;
                if (gradientMagnitude - avgGradient > 10) {
                  const imageX = searchArea.x + x;
                  const imageY = searchArea.y + y;
                  
                  const isDuplicate = points.some(p =>
                    Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
                  );
                  
                  if (!isDuplicate) {
                    points.push({ 
                      x: imageX, 
                      y: imageY, 
                      brightness: pixel,
                      contrast: gradientMagnitude,
                      method: 'gradient'
                    });
                  }
                }
              }
            }
          }
        }
      }
      
    }
    
    // ============================================
    // Method 4: Multi-scale Detection (Different step sizes)
    // ============================================
    if (shouldRunMethod('multiscale') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // Try with different step sizes and thresholds
      const scales = [1, 2, 3];
      const thresholds = [brightnessThreshold, brightnessThreshold * 0.8, brightnessThreshold * 0.6];
      
      for (const scale of scales) {
        for (const threshold of thresholds) {
          for (let y = 3; y < height - 3; y += scale) {
            for (let x = 3; x < width - 3; x += scale) {
              const pixelIndex = (y * width + x) * 4;
              const pixel = data[pixelIndex];
              
              if (pixel > threshold) {
                // Simple local max check
                let isMax = true;
                let localSum = 0;
                let localCount = 0;
                
                for (let dy = -scale; dy <= scale; dy++) {
                  for (let dx = -scale; dx <= scale; dx++) {
                    if (dx === 0 && dy === 0) continue;
                    const ny = y + dy;
                    const nx = x + dx;
                    if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                      const neighborIdx = (ny * width + nx) * 4;
                      const neighborVal = data[neighborIdx];
                      localSum += neighborVal;
                      localCount++;
                      if (neighborVal > pixel) isMax = false;
                    }
                  }
                }
                
                if (isMax && localCount > 0) {
                  const avgLocal = localSum / localCount;
                  if (pixel - avgLocal > minContrast * 0.5) {
                    const imageX = searchArea.x + x;
                    const imageY = searchArea.y + y;
                    
                    const isDuplicate = points.some(p =>
                      Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
                    );
                    
                    if (!isDuplicate) {
                      points.push({ 
                        x: imageX, 
                        y: imageY, 
                        brightness: pixel,
                        contrast: pixel - avgLocal,
                        method: 'multiscale'
                      });
                    }
                  }
                }
              }
            }
          }
        }
      }
      
    }
    
    // ============================================
    // Method 5: Histogram-based Peak Detection
    // ============================================
    if (shouldRunMethod('histogram_peak') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // Create brightness histogram for the search area
      const histogram = new Array(256).fill(0);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const pixelIndex = (y * width + x) * 4;
          const brightness = data[pixelIndex];
          histogram[brightness]++;
        }
      }
      
      // Find peaks in histogram (brightness values that are common)
      const peaks = [];
      for (let i = 1; i < 255; i++) {
        if (histogram[i] > histogram[i-1] && histogram[i] > histogram[i+1] && histogram[i] > 10) {
          peaks.push(i);
        }
      }
      
      // Sort peaks by frequency (most common first)
      peaks.sort((a, b) => histogram[b] - histogram[a]);
      
      // Find points with brightness matching top peaks
      const topPeaks = peaks.slice(0, 3); // Top 3 peaks
      for (const peakBrightness of topPeaks) {
        if (peakBrightness < brightnessThreshold * 0.7) continue;
        
        for (let y = 3; y < height - 3; y += 1) {
          for (let x = 3; x < width - 3; x += 1) {
            const pixelIndex = (y * width + x) * 4;
            const pixel = data[pixelIndex];
            
            // If pixel brightness is close to peak brightness
            if (Math.abs(pixel - peakBrightness) < 10 && pixel > brightnessThreshold * 0.6) {
              // Check if it's a local maximum
              let isLocalMax = true;
              for (let dy = -2; dy <= 2; dy++) {
                for (let dx = -2; dx <= 2; dx++) {
                  if (dx === 0 && dy === 0) continue;
                  const ny = y + dy;
                  const nx = x + dx;
                  if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                    const neighborIdx = (ny * width + nx) * 4;
                    if (data[neighborIdx] > pixel) {
                      isLocalMax = false;
                      break;
                    }
                  }
                }
                if (!isLocalMax) break;
              }
              
              if (isLocalMax) {
                const imageX = searchArea.x + x;
                const imageY = searchArea.y + y;
                
                const isDuplicate = points.some(p =>
                  Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
                );
                
                if (!isDuplicate) {
                  points.push({ 
                    x: imageX, 
                    y: imageY, 
                    brightness: pixel,
                    contrast: 30,
                    method: 'histogram_peak'
                  });
                }
              }
            }
          }
        }
      }
      
    }
    
    
    // Sort by contrast (highest contrast first)
    points.sort((a, b) => b.contrast - a.contrast);
    
    // ============================================
    // Find best pair from detected points
    // ============================================
    const calibrationPairs = [];
    const {maxPoints} = params.detection;
    
    // همه جفت‌های نقاط را بررسی کن با استفاده از پارامترهای verticalAlignment
    const {maxDx} = params.verticalAlignment;
    // 🔧 تبدیل درصد به پیکسل: minDy و maxDy به صورت درصد از ارتفاع کل تصویر هستند
    const minDy = params.verticalAlignment.minDy * imageHeight; // تبدیل درصد به پیکسل (نسبت به ارتفاع کل تصویر)
    const maxDy = params.verticalAlignment.maxDy * imageHeight; // تبدیل درصد به پیکسل (نسبت به ارتفاع کل تصویر)
    
    for (let i = 0; i < Math.min(maxPoints, points.length); i++) {
      for (let j = i + 1; j < Math.min(maxPoints, points.length); j++) {
        const p1 = points[i];
        const p2 = points[j];
        
        const dx = Math.abs(p1.x - p2.x);
        const dy = Math.abs(p1.y - p2.y);
        
        // 🔧 Validation: فقط جفت‌های عمودی را قبول کن
        // dx باید کوچکتر از maxDx باشد (عمودی)
        // dy باید بین minDy و maxDy باشد (فاصله واقعی)
        if (dx > maxDx || dy < minDy || dy > maxDy || dy === 0) {
          continue; // Skip this pair - not vertically aligned or invalid distance
        }
        
        const distance = dy; // استفاده از dy برای فاصله
        
        // Calculate expected mm/pixel if this is 10mm
        const mmPerPixel = 10 / distance;
        
        calibrationPairs.push({
          p1: p1.y < p2.y ? p1 : p2, // Upper point
          p2: p1.y < p2.y ? p2 : p1, // Lower point
          distance,
          dx,
          dy,
          mmPerPixel,
          score: (p1.contrast + p2.contrast) / 2 // Average contrast as score
        });
      }
    }
    
    // Sort by score (highest contrast first)
    calibrationPairs.sort((a, b) => b.score - a.score);
    
    if (calibrationPairs.length > 0) {
      const bestPair = calibrationPairs[0];
      
      return [bestPair.p2, bestPair.p1]; // p2 (top), p1 (bottom)
    }
    
    // If no pairs found, return top 2 brightest points
    if (points.length >= 2) {
      return [points[0], points[1]];
    } if (points.length === 1) {
      // اگر فقط 1 نقطه پیدا شد، یک نقطه دیگر در نزدیکی اضافه کن
      const p1 = points[0];
      const p2 = {
        x: p1.x,
        y: p1.y - 100, // 100 pixels above
        brightness: p1.brightness * 0.9,
        contrast: p1.contrast * 0.9,
        method: 'estimated'
      };
      return [p2, p1];
    }
    
    // 🔧 اگر هیچ نقطه‌ای پیدا نشد، از الگوریتم‌های پیشرفته‌تر استفاده کن
    if (shouldRunMethod('watershed') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // ============================================
      // Method 6: Watershed-like Region Growing
      // ============================================
      
      // 🔧 استفاده از پارامترهای قابل تنظیم برای threshold
      const watershedThreshold = brightnessThreshold * 0.3; // Lower threshold for better detection
      
      // Find seed points (very bright pixels)
      const seeds = [];
      const visited = new Set();
      
      for (let y = 5; y < height - 5; y += 3) {
        for (let x = 5; x < width - 5; x += 3) {
          const pixelIndex = (y * width + x) * 4;
          const pixel = data[pixelIndex];
          
          if (pixel > watershedThreshold) {
          // Check if this is a bright region center
          let regionBrightness = 0;
          let regionCount = 0;
          
          for (let dy = -3; dy <= 3; dy++) {
            for (let dx = -3; dx <= 3; dx++) {
              const ny = y + dy;
              const nx = x + dx;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const neighborIdx = (ny * width + nx) * 4;
                regionBrightness += data[neighborIdx];
                regionCount++;
              }
            }
          }
          
          const avgRegionBrightness = regionBrightness / regionCount;
          if (avgRegionBrightness > watershedThreshold) {
            const key = `${x},${y}`;
            if (!visited.has(key)) {
              seeds.push({ x, y, brightness: avgRegionBrightness });
              visited.add(key);
            }
          }
        }
      }
    }
    
    // Sort seeds by brightness
    seeds.sort((a, b) => b.brightness - a.brightness);
    
    // Take top seeds and find their centers
    const regionCenters = [];
    for (const seed of seeds.slice(0, 10)) {
      // Find center of bright region around seed
      let centerX = seed.x;
      let centerY = seed.y;
      let maxBrightness = 0;
      
      for (let dy = -5; dy <= 5; dy++) {
        for (let dx = -5; dx <= 5; dx++) {
          const ny = seed.y + dy;
          const nx = seed.x + dx;
          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const pixelIndex = (ny * width + nx) * 4;
            const pixel = data[pixelIndex];
            if (pixel > maxBrightness) {
              maxBrightness = pixel;
              centerX = nx;
              centerY = ny;
            }
          }
        }
      }
      
      const imageX = searchArea.x + centerX;
      const imageY = searchArea.y + centerY;
      
      const isDuplicate = regionCenters.some(p =>
        Math.abs(p.x - imageX) < duplicateThreshold * 2 && Math.abs(p.y - imageY) < duplicateThreshold * 2
      );
      
      if (!isDuplicate) {
        regionCenters.push({ 
          x: imageX, 
          y: imageY, 
          brightness: maxBrightness,
          contrast: maxBrightness - watershedThreshold,
          method: 'watershed'
        });
      }
    }
    
      
      // Add region centers to points
      regionCenters.forEach(p => {
        const isDuplicate = points.some(existing =>
          Math.abs(existing.x - p.x) < duplicateThreshold && Math.abs(existing.y - p.y) < duplicateThreshold
        );
        if (!isDuplicate) {
          points.push(p);
        }
      });
      
      // Sort again
      points.sort((a, b) => b.contrast - a.contrast);
      
    }
    
    // ============================================
    // Method 7: Template Matching
    // ============================================
    if (shouldRunMethod('template_matching') && (selectedMethod === 'auto' ? points.length < 2 : true)) {
      
      // 🔧 استفاده از پارامترهای قابل تنظیم برای threshold
      const templateThreshold = brightnessThreshold * 0.3; // Lower threshold for better detection
      
      // Create a simple template (small bright circle)
      const templateSize = 5;
      const template = [];
      for (let ty = -templateSize; ty <= templateSize; ty++) {
        for (let tx = -templateSize; tx <= templateSize; tx++) {
          const dist = Math.sqrt(tx * tx + ty * ty);
          if (dist <= templateSize) {
            template.push({ x: tx, y: ty, weight: 1 - dist / templateSize });
          }
        }
      }
      
      // Search for template matches
      const matches = [];
      for (let y = templateSize; y < height - templateSize; y += 2) {
        for (let x = templateSize; x < width - templateSize; x += 2) {
          let matchScore = 0;
          let totalWeight = 0;
          
          for (const t of template) {
            const nx = x + t.x;
            const ny = y + t.y;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
              const pixelIndex = (ny * width + nx) * 4;
              const pixel = data[pixelIndex];
              matchScore += pixel * t.weight;
              totalWeight += t.weight;
            }
          }
          
          const avgScore = matchScore / totalWeight;
          if (avgScore > templateThreshold) {
            matches.push({ x, y, score: avgScore });
          }
        }
      }
      
      // Sort by score and take top matches
      matches.sort((a, b) => b.score - a.score);
      
      for (const match of matches.slice(0, 5)) {
        const imageX = searchArea.x + match.x;
        const imageY = searchArea.y + match.y;
        
        const isDuplicate = points.some(p =>
          Math.abs(p.x - imageX) < duplicateThreshold && Math.abs(p.y - imageY) < duplicateThreshold
        );
        
        if (!isDuplicate) {
          points.push({ 
            x: imageX, 
            y: imageY, 
            brightness: match.score,
            contrast: match.score - templateThreshold,
            method: 'template_matching'
          });
        }
      }
      
    }
    
    // Final return
    if (points.length >= 2) {
      return [points[0], points[1]];
    } if (points.length === 1) {
      const p1 = points[0];
      const p2 = {
        x: p1.x,
        y: p1.y - 100,
        brightness: p1.brightness * 0.9,
        contrast: p1.contrast * 0.9,
        method: 'estimated'
      };
      return [p2, p1];
    }
    
    // 🔧 اگر هنوز هیچ نقطه‌ای پیدا نشد، از الگوریتم fallback استفاده کن
    // اما این بار با منطق بهتر: پیدا کردن نقاط بر اساس الگوهای تصویر
    
    // Intelligent fallback: Find brightest regions in upper-right area
    const fallbackPoints = [];
    const regionSize = 10;
    
    // Divide search area into grid and find brightest region in each cell
    const gridX = Math.floor(width / regionSize);
    const gridY = Math.floor(height / regionSize);
    
    for (let gy = 0; gy < gridY; gy++) {
      for (let gx = 0; gx < gridX; gx++) {
        let maxBrightness = 0;
        let maxX = gx * regionSize;
        let maxY = gy * regionSize;
        
        for (let y = gy * regionSize; y < Math.min((gy + 1) * regionSize, height); y++) {
          for (let x = gx * regionSize; x < Math.min((gx + 1) * regionSize, width); x++) {
            const pixelIndex = (y * width + x) * 4;
            const pixel = data[pixelIndex];
            if (pixel > maxBrightness) {
              maxBrightness = pixel;
              maxX = x;
              maxY = y;
            }
          }
        }
        
        if (maxBrightness > 50) { // Very low threshold
          fallbackPoints.push({
            x: searchArea.x + maxX,
            y: searchArea.y + maxY,
            brightness: maxBrightness,
            contrast: maxBrightness - 50,
            method: 'grid_fallback'
          });
        }
      }
    }
    
    // Sort and take top 2
    fallbackPoints.sort((a, b) => b.brightness - a.brightness);
    
    if (fallbackPoints.length >= 2) {
      return [fallbackPoints[0], fallbackPoints[1]];
    } if (fallbackPoints.length === 1) {
      const p1 = fallbackPoints[0];
      const p2 = {
        x: p1.x,
        y: p1.y - 100,
        brightness: p1.brightness * 0.9,
        contrast: p1.contrast * 0.9,
        method: 'estimated'
      };
      return [p2, p1];
    }
    
    // Last resort: return points based on image statistics
    
    // Find points based on brightness distribution
    const sortedPixels = [];
    for (let y = 0; y < height; y += 2) {
      for (let x = 0; x < width; x += 2) {
        const pixelIndex = (y * width + x) * 4;
        sortedPixels.push({ x, y, brightness: data[pixelIndex] });
      }
    }
    
    sortedPixels.sort((a, b) => b.brightness - a.brightness);
    
    const finalPoints = [];
    for (const pixel of sortedPixels.slice(0, 10)) {
      const imageX = searchArea.x + pixel.x;
      const imageY = searchArea.y + pixel.y;
      
      const isDuplicate = finalPoints.some(p =>
        Math.abs(p.x - imageX) < duplicateThreshold * 3 && Math.abs(p.y - imageY) < duplicateThreshold * 3
      );
      
      if (!isDuplicate) {
        finalPoints.push({
          x: imageX,
          y: imageY,
          brightness: pixel.brightness,
          contrast: pixel.brightness - 30,
          method: 'statistical'
        });
      }
      
      if (finalPoints.length >= 2) break;
    }
    
    if (finalPoints.length >= 2) {
      return [finalPoints[0], finalPoints[1]];
    }
    
    // If still nothing, throw error instead of returning empty array
    throw new Error('Computer Vision failed to detect any calibration points. Please adjust CV parameters or try ML detection.');
  };

  // Find high contrast points that could be calibration markers (legacy function)
  const findHighContrastPoints = (imageData, searchArea) => {
    const points = [];
    const {data} = imageData;
    const {width} = imageData;
    const {height} = imageData;
    
    // Method 1: Look for LOCAL MAXIMA (bright spots)
    // Lower threshold for better detection
    for (let y = 3; y < height - 3; y += 1) {
      for (let x = 3; x < width - 3; x += 1) {
        const pixelIndex = (y * width + x) * 4;
        const pixel = data[pixelIndex]; // Red channel (grayscale)
        
        // Lower threshold: 150 instead of 180
        if (pixel > 150) {
          // Check if this is a LOCAL MAXIMUM
          let isLocalMax = true;
          let neighborSum = 0;
          let neighborCount = 0;
          
          // Check 3x3 neighborhood
          for (let dy = -2; dy <= 2; dy++) {
            for (let dx = -2; dx <= 2; dx++) {
              if (dx === 0 && dy === 0) continue;
              
              const ny = y + dy;
              const nx = x + dx;
              if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                const neighborIdx = (ny * width + nx) * 4;
                const neighborVal = data[neighborIdx];
                neighborSum += neighborVal;
                neighborCount++;
                
                // If neighbor is brighter, this is not a local max
                if (neighborVal > pixel) {
                  isLocalMax = false;
                  break;
                }
              }
            }
            if (!isLocalMax) break;
          }
          
          if (isLocalMax && neighborCount > 0) {
            const avgNeighbor = neighborSum / neighborCount;
            
            // Require at least 25 units brighter than neighbors (lowered from 30)
            if (pixel - avgNeighbor > 25) {
              // Convert to image coordinates
            const imageX = searchArea.x + x;
            const imageY = searchArea.y + y;
            
              // Avoid duplicates (within 10 pixels)
            const isDuplicate = points.some(p =>
              Math.abs(p.x - imageX) < 10 && Math.abs(p.y - imageY) < 10
            );
            
            if (!isDuplicate) {
                points.push({ 
                  x: imageX, 
                  y: imageY, 
                  brightness: pixel,
                  contrast: pixel - avgNeighbor
                });
              }
            }
          }
        }
      }
    }
    
    
    // Sort by contrast (highest contrast first)
    points.sort((a, b) => b.contrast - a.contrast);
    
    // Method 2: Find best vertical pair
    const calibrationPairs = [];
    
    for (let i = 0; i < Math.min(20, points.length); i++) {
      for (let j = i + 1; j < Math.min(20, points.length); j++) {
        const p1 = points[i];
        const p2 = points[j];
        
        const dx = Math.abs(p1.x - p2.x);
        const dy = Math.abs(p1.y - p2.y);
        // 🔧 ساده‌سازی: چون p1 و p2 عمودی هستند (x یکسان)، فقط از dy استفاده می‌کنیم
        const distance = dy;
        
        // Ruler marks characteristics:
        // - Vertically aligned: dx < 20 pixels
        // - Separated: 15mm to 50mm apart (roughly 50-200 pixels depending on resolution)
        // - Distance should be consistent (typically around 10mm = 90-110 pixels)
        if (dx < 20 && dy > 50 && dy < 200) {
          // Calculate expected mm/pixel if this is 10mm
          const mmPerPixel = 10 / distance;
          
          // Realistic range for cephalometric radiographs
          if (mmPerPixel >= 0.05 && mmPerPixel <= 0.20) {
            calibrationPairs.push(p1.y < p2.y ? p1 : p2); // Upper (p2)
            calibrationPairs.push(p1.y < p2.y ? p2 : p1); // Lower (p1)
            break;
          }
        }
      }
      if (calibrationPairs.length === 2) break;
    }
    
    // Return calibration pair if found, otherwise top 4 brightest points
    if (calibrationPairs.length === 2) {
      return calibrationPairs;
    }
    return points.slice(0, 4);
    
  };
  
  // Get average brightness of surrounding pixels
  const getAverageSurroundingBrightness = (data, centerX, centerY, width, height, radius) => {
    let sum = 0;
    let count = 0;
    
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        // Skip center pixel
        if (dx === 0 && dy === 0) continue;
        
        const x = centerX + dx;
        const y = centerY + dy;
        
        if (x >= 0 && x < width && y >= 0 && y < height) {
          const pixelIndex = (y * width + x) * 4;
          sum += data[pixelIndex];
          count++;
        }
      }
    }
    
    return count > 0 ? sum / count : 0;
  };

  // Check if a group of pixels forms a calibration marker
  const isCalibrationMarker = (surroundingPixels) => {
    // Calibration markers are typically small, dark, and roughly circular
    // or have high contrast edges
    
    const darkCount = surroundingPixels.filter(p => p < 100).length;
    const totalCount = surroundingPixels.length;
    
    // Should have at least 40% dark pixels and be relatively compact
    return darkCount / totalCount > 0.4 && totalCount > 5;
  };

  // Get pixel data from surrounding area
  const getSurroundingPixelData = (data, centerX, centerY, imageWidth, radius) => {
    const pixels = [];
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const x = centerX + dx;
        const y = centerY + dy;
        if (x >= 0 && y >= 0 && x < imageWidth) {
          const index = (y * imageWidth + x) * 4;
          if (index >= 0 && index < data.length) {
            pixels.push(data[index]); // Red channel
          }
        }
      }
    }
    return pixels;
  };

  // Helper functions for calculating measurements from landmarks
  // Helper function to calculate distance from a point to a line (in pixels)
  // Note: This returns distance in pixels. Use calculateDistanceToLineMm for millimeters.
  const calculateDistanceToLine = useCallback((point, lineStart, lineEnd) => {
    const A = point.x - lineStart.x;
    const B = point.y - lineStart.y;
    const C = lineEnd.x - lineStart.x;
    const D = lineEnd.y - lineStart.y;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    if (lenSq === 0) return null;
    
    const param = dot / lenSq;
    
    let xx;
    let yy;
    if (param < 0) {
      xx = lineStart.x;
      yy = lineStart.y;
    } else if (param > 1) {
      xx = lineEnd.x;
      yy = lineEnd.y;
    } else {
      xx = lineStart.x + param * C;
      yy = lineStart.y + param * D;
    }
    
    const dx = point.x - xx;
    const dy = point.y - yy;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  // Helper function to calculate distance from a point to a line (in millimeters)
  const calculateDistanceToLineMm = useCallback((point, lineStart, lineEnd) => {
    const pixelDistance = calculateDistanceToLine(point, lineStart, lineEnd);
    if (pixelDistance === null) return null;
    
    // Convert to millimeters using conversion factor
    const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
    return pixelDistance * conversionFactor;
  }, [PIXEL_TO_MM_CONVERSION, calculateDistanceToLine]);

  const calculateAngle = (p1, vertex, p2) => {
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
    let angle = (angle2 - angle1) * (180 / Math.PI);
    if (angle < 0) angle += 360;
    return angle > 180 ? 360 - angle : angle;
  };

  const calculateLineAngle = (p1, p2) => Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);
  
  // Calculate angle between two lines (for FMA, etc.)
  const calculateAngleBetweenLines = (line1Start, line1End, line2Start, line2End) => {
    // Calculate direction vectors for both lines
    const v1x = line1End.x - line1Start.x;
    const v1y = line1End.y - line1Start.y;
    const v2x = line2End.x - line2Start.x;
    const v2y = line2End.y - line2Start.y;
    
    // Calculate dot product
    const dotProduct = v1x * v2x + v1y * v2y;
    
    // Calculate magnitudes
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    
    // Avoid division by zero
    if (mag1 === 0 || mag2 === 0) {
      return 0;
    }
    
    // Calculate angle in radians
    const cosAngle = dotProduct / (mag1 * mag2);
    // Clamp cosAngle to [-1, 1] to avoid NaN
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    const angleRad = Math.acos(clampedCos);
    
    // Convert to degrees
    const angleDeg = angleRad * (180 / Math.PI);
    
    // Return acute angle (0-90) - always return the smaller angle
    // If angle > 90, return its supplement (180 - angle) to get the acute angle
    return angleDeg > 90 ? 180 - angleDeg : angleDeg;
  };

  // بهینه‌سازی: استفاده از useCallback برای calculateMeasurements
  // توجه: p1 و p2 نقاط کالیبراسیون هستند که فاصله 1cm (10mm) دارند
  // از این نقاط برای تبدیل پیکسل به میلی‌متر استفاده می‌شود (PIXEL_TO_MM_CONVERSION)
  const calculateMeasurements = useCallback((landmarks) => {
    const measures = {};
    
    
    // Normalize landmark names (case-insensitive lookup helper)
    const getLandmark = (names) => {
      for (const name of names) {
        if (landmarks[name]) return landmarks[name];
      }
      return null;
    };
    
    // Helper to find landmark by partial name match (for variations like 'L1', 'lower_incisor', etc.)
    const findLandmarkByPartial = (partialNames) => {
      const landmarkKeys = Object.keys(landmarks);
      for (const partial of partialNames) {
        const found = landmarkKeys.find(key => 
          key.toLowerCase().includes(partial.toLowerCase()) || 
          partial.toLowerCase().includes(key.toLowerCase())
        );
        if (found) return landmarks[found];
      }
      return null;
    };
    
    // Helper function برای normalize کردن زاویه (0-180)
    const normalizeAngle = (angle) => Math.round(Math.max(0, Math.min(180, angle)) * 10) / 10;
    
    try {
      // بهینه‌سازی: پیدا کردن لندمارک‌های رایج یک بار در ابتدای تابع
      const s = getLandmark(['S', 's', 'sella', 'Sella']) || findLandmarkByPartial(['s', 'sella']);
      const n = getLandmark(['N', 'n', 'nasion', 'Nasion']) || findLandmarkByPartial(['n', 'nasion']);
      const a = getLandmark(['A', 'a']);
      const b = getLandmark(['B', 'b']);
      const or = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']) || findLandmarkByPartial(['or', 'orbit']);
      const po = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']) || findLandmarkByPartial(['po', 'porion']);
      const go = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']) || findLandmarkByPartial(['go', 'gonion']);
      const me = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) || findLandmarkByPartial(['me', 'menton']);
      const gn = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']) || findLandmarkByPartial(['gn', 'gnathion']);
      const u1 = getLandmark(['U1', 'u1', 'UIT', 'uit', 'upper_incisor_tip', 'Upper_incisor_tip', 'upper incisor tip']) || 
                 findLandmarkByPartial(['u1', 'uit', 'upper', 'incisor', 'tip']);
      const u1a = getLandmark(['U1A', 'u1a', 'UIA', 'uia', 'upper_incisor_apex', 'Upper_incisor_apex', 'upper incisor apex']) || 
                  findLandmarkByPartial(['u1a', 'uia', 'upper', 'incisor', 'apex']);
      const l1 = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor', 'LIT', 'lit', 'Lit', 'lower_incisor_tip', 'Lower_incisor_tip', 'lower incisor tip']) || 
                 findLandmarkByPartial(['l1', 'lit', 'lower', 'incisor', 'li', 'tip']);
      const l1a = getLandmark(['L1A', 'l1a', 'LIA', 'lia', 'Lia', 'lower_incisor_apex', 'Lower_incisor_apex', 'lower incisor apex']) || 
                  findLandmarkByPartial(['l1a', 'lia', 'lower', 'incisor', 'apex']);
      
      // SNA angle
      if (s && n && a) {
        measures.SNA = calculateAngle(s, n, a);
      }
      
      // SNB angle
      if (s && n && b) {
        measures.SNB = calculateAngle(s, n, b);
      }
      
      // ANB angle
      if (measures.SNA !== undefined && measures.SNB !== undefined) {
        measures.ANB = measures.SNA - measures.SNB;
      }
      
      // FMA (Frankfort-Mandibular Angle)
      // FMA = زاویه بین خط فرانکفورت (Or-Po) و صفحه مندیبولار (Go-Me)
      if (or && po && go && me) {
        measures.FMA = normalizeAngle(calculateAngleBetweenLines(
          or, po,  // خط فرانکفورت
          go, me   // صفحه مندیبولار
        ));
      }
      
      // FMIA (Frankfort-Mandibular Incisor Angle) - زاویه در مثلث Tweed
      // FMIA = زاویه در نقطه تقاطع خط فرانکفورت (Or-Po) و خط incisor (L1A-L1 یا L1-Me)
      // این همان زاویه‌ای است که در visualizer نمایش داده می‌شود (angle2 در مثلث Tweed)
      if (or && po && l1 && me && go) {
        // 🔧 FIX: محاسبه FMIA همانند visualizer (زاویه در مثلث Tweed)
        // تعیین خط incisor: اول L1A-L1، اگر نبود L1-Me
        let incisorStart = null;
        let incisorEnd = null;
        
        if (l1a && l1) {
          incisorStart = l1a;
          incisorEnd = l1;
        } else if (l1 && me) {
          incisorStart = l1;
          incisorEnd = me;
        }
        
        if (incisorStart && incisorEnd) {
          // محاسبه FMIA به روش مثلث Tweed: زاویه در نقطه تقاطع خط فرانکفورت و incisor
          // این همان angle2 در مثلث Tweed است که در visualizer نمایش داده می‌شود
          
          // تابع کمکی برای محاسبه نقطه تقاطع دو خط
          const getLineIntersection = (p1, p2, p3, p4) => {
            const x1 = p1.x; const y1 = p1.y;
            const x2 = p2.x; const y2 = p2.y;
            const x3 = p3.x; const y3 = p3.y;
            const x4 = p4.x; const y4 = p4.y;
            
            const denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
            if (Math.abs(denom) < 1e-10) return null;
            
            const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;
            const intersectionX = x1 + t * (x2 - x1);
            const intersectionY = y1 + t * (y2 - y1);
            
            return { x: intersectionX, y: intersectionY };
          };
          
          // امتداد خطوط برای تشکیل مثلث (استفاده از فاصله بزرگ برای امتداد)
          const extendDistance = 10000; // فاصله بزرگ برای امتداد خطوط
          
          // امتداد خط فرانکفورت (Or-Po)
          const frankfortDx = po.x - or.x;
          const frankfortDy = po.y - or.y;
          const frankfortLength = Math.sqrt(frankfortDx * frankfortDx + frankfortDy * frankfortDy);
          if (frankfortLength > 0) {
            const frankfortDirX = frankfortDx / frankfortLength;
            const frankfortDirY = frankfortDy / frankfortLength;
            const frankfortStart = {
              x: or.x - frankfortDirX * extendDistance,
              y: or.y - frankfortDirY * extendDistance
            };
            const frankfortEnd = {
              x: po.x + frankfortDirX * extendDistance,
              y: po.y + frankfortDirY * extendDistance
            };
            
            // امتداد خط incisor
            const incisorDx = incisorEnd.x - incisorStart.x;
            const incisorDy = incisorEnd.y - incisorStart.y;
            const incisorLength = Math.sqrt(incisorDx * incisorDx + incisorDy * incisorDy);
            if (incisorLength > 0) {
              const incisorDirX = incisorDx / incisorLength;
              const incisorDirY = incisorDy / incisorLength;
              const incisorExtendedStart = {
                x: incisorStart.x - incisorDirX * extendDistance,
                y: incisorStart.y - incisorDirY * extendDistance
              };
              const incisorExtendedEnd = {
                x: incisorEnd.x + incisorDirX * extendDistance,
                y: incisorEnd.y + incisorDirY * extendDistance
              };
              
              // محاسبه نقطه تقاطع (intersection2)
              const intersection2 = getLineIntersection(
                frankfortStart, frankfortEnd,
                incisorExtendedStart, incisorExtendedEnd
              );
              
              // امتداد خط مندیبولار (Go-Me) برای محاسبه زاویه در intersection2
              const mandibularDx = me.x - go.x;
              const mandibularDy = me.y - go.y;
              const mandibularLength = Math.sqrt(mandibularDx * mandibularDx + mandibularDy * mandibularDy);
              if (mandibularLength > 0 && intersection2) {
                const mandibularDirX = mandibularDx / mandibularLength;
                const mandibularDirY = mandibularDy / mandibularLength;
                const mandibularStart = {
                  x: go.x - mandibularDirX * extendDistance,
                  y: go.y - mandibularDirY * extendDistance
                };
                const mandibularEnd = {
                  x: me.x + mandibularDirX * extendDistance,
                  y: me.y + mandibularDirY * extendDistance
                };
                
                // محاسبه نقاط تقاطع دیگر برای مثلث
                const intersection1 = getLineIntersection(frankfortStart, frankfortEnd, mandibularStart, mandibularEnd);
                const intersection3 = getLineIntersection(mandibularStart, mandibularEnd, incisorExtendedStart, incisorExtendedEnd);
                
                // محاسبه زاویه در intersection2 (FMIA) - همانند visualizer
                if (intersection1 && intersection2 && intersection3) {
                  // تابع محاسبه زاویه در رأس مثلث
                  const angleAtVertex = (vertex, point1, point2) => {
                    const v1x = point1.x - vertex.x;
                    const v1y = point1.y - vertex.y;
                    const v2x = point2.x - vertex.x;
                    const v2y = point2.y - vertex.y;
                    
                    const dot = v1x * v2x + v1y * v2y;
                    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
                    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
                    
                    if (mag1 === 0 || mag2 === 0) return 0;
                    
                    const cosAngle = dot / (mag1 * mag2);
                    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
                    const angle = Math.acos(clampedCos) * (180 / Math.PI);
                    
                    return angle;
                  };
                  
                  // FMIA = زاویه در intersection2 (بین intersection1 و intersection3)
                  const angle2 = angleAtVertex(intersection2, intersection1, intersection3);
                  measures.FMIA = normalizeAngle(angle2);
                }
              }
            }
          }
          
          // Fallback: اگر محاسبه مثلث موفق نبود، از روش قبلی استفاده می‌کنیم
          if (measures.FMIA === undefined) {
          measures.FMIA = normalizeAngle(calculateAngleBetweenLines(
            or, po,  // خط فرانکفورت
            incisorStart, incisorEnd  // خط incisor
          ));
          }
        }
      }
      
      // IMPA (Incisor Mandibular Plane Angle) - زاویه بین خط incisor پایین و صفحه مندیبولار (Go-Me)
      if (l1a && l1 && go && me) {
        // استفاده از L1A-L1 برای محاسبه دقیق‌تر (همانند مثلث Tweed)
        measures.IMPA = normalizeAngle(calculateAngleBetweenLines(
          go, me,  // خط اول: صفحه مندیبولار (Go-Me)
          l1a, l1   // خط دوم: incisor (L1A-L1)
        ));
      } else if (l1 && go && me) {
        // Fallback: اگر L1A موجود نبود، از L1-Me استفاده می‌کنیم
        measures.IMPA = normalizeAngle(calculateAngleBetweenLines(
          go, me,  // خط اول: صفحه مندیبولار (Go-Me)
          l1, me   // خط دوم: incisor (L1-Me)
        ));
      }
      
      // GoGn-SN (GoGn to SN angle) - همان GoGnSN
      if (s && n && go && gn) {
        const snAngle = calculateLineAngle(s, n);
        const gognAngle = calculateLineAngle(go, gn);
        const angleDiff = Math.abs(snAngle - gognAngle);
        measures['GoGn-SN'] = angleDiff;
        measures.GoGnSN = angleDiff; // برای سازگاری
      }
      
      // U1-SN (Upper incisor to SN angle) - زاویه بین خط incisor بالا (U1A-U1) و خط SN
      if (u1a && u1 && s && n) {
        // محاسبه زاویه بین خط incisor (U1A-U1) و خط SN (S-N)
        const u1Angle = calculateLineAngle(u1a, u1);
        const snAngle = calculateLineAngle(s, n);
        let angleDiff = Math.abs(u1Angle - snAngle);
        if (angleDiff > 180) angleDiff = 360 - angleDiff;
        // عدد بدست آمده باید از 180 کم شود
        measures['U1-SN'] = normalizeAngle(180 - angleDiff);
      } else if (u1 && s && n) {
        // Fallback: اگر U1A موجود نبود، از نقطه U1 استفاده می‌کنیم
        const u1Angle = calculateLineAngle(u1, n);
        const snAngle = calculateLineAngle(s, n);
        let angleDiff = Math.abs(u1Angle - snAngle);
        if (angleDiff > 180) angleDiff = 360 - angleDiff;
        // عدد بدست آمده باید از 180 کم شود
        measures['U1-SN'] = normalizeAngle(180 - angleDiff);
      }
      
      // L1-MP (Lower incisor to Mandibular Plane angle) - زاویه بین خط incisor پایین (L1A-L1) و صفحه مندیبولار (Go-Me)
      if (l1a && l1 && go && me) {
        // محاسبه زاویه بین خط incisor (L1A-L1) و صفحه مندیبولار (Go-Me)
        const calculatedAngle = calculateAngleBetweenLines(go, me, l1a, l1);
        // عدد بدست آمده باید از 180 کم شود
        measures['L1-MP'] = normalizeAngle(180 - calculatedAngle);
      } else if (go && me && l1) {
        // Fallback: اگر L1A موجود نبود، از نقطه L1 استفاده می‌کنیم
        const l1mpAngle = calculateAngle(l1, me, go);
        measures['L1-MP'] = normalizeAngle(l1mpAngle);
      }
      
      // Interincisal Angle - زاویه بین خط U1-U1A و خط L1-L1A
      if (u1 && u1a && l1 && l1a) {
        // محاسبه زاویه بین دو خط: خط U1-U1A و خط L1-L1A
        const interincisalAngle = calculateAngleBetweenLines(
          u1, u1a,  // خط اول: U1-U1A
          l1, l1a   // خط دوم: L1-L1A
        );
        measures.InterincisalAngle = Math.round(interincisalAngle * 10) / 10;
      }

      // Overbite - فاصله عمودی بین U1 و L1 (تفاوت y)
      const u1LandmarkOverbite = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper_incisor', 'upper incisor']) ||
                                  findLandmarkByPartial(['u1', 'upper', 'incisor', 'ui']);
      const l1LandmarkOverbite = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                                  findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);

      if (u1LandmarkOverbite && l1LandmarkOverbite) {
        // Overbite = yU1 - yL1
        const verticalDistance = u1LandmarkOverbite.y - l1LandmarkOverbite.y;
        
        // محاسبه conversion factor از p1/p2 اگر موجود باشند، در غیر این صورت از مقدار پیش‌فرض
        let conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          // فاصله p1-p2 معمولاً 10mm است (1cm)
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        
        // تبدیل به میلی‌متر با استفاده از conversion factor
        measures.Overbite = Math.round((verticalDistance * conversionFactor) * 10) / 10;
      }

      // Overjet - فاصله افقی بین U1 و L1 (تفاوت x)
      const u1LandmarkOverjet = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper_incisor', 'upper incisor']) ||
                                findLandmarkByPartial(['u1', 'upper', 'incisor', 'ui']);
      const l1LandmarkOverjet = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                                findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);

      if (u1LandmarkOverjet && l1LandmarkOverjet) {
        // Overjet = فاصله افقی (تفاوت x)
        // اگر U1.x > L1.x (U1 جلوتر است) → مقدار مثبت
        // اگر L1.x > U1.x (L1 جلوتر است) → مقدار منفی
        const horizontalDistance = u1LandmarkOverjet.x - l1LandmarkOverjet.x;
        
        // محاسبه conversion factor از p1/p2 اگر موجود باشند، در غیر این صورت از مقدار پیش‌فرض
        let conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          // فاصله p1-p2 معمولاً 10mm است (1cm)
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        
        // تبدیل به میلی‌متر با استفاده از conversion factor (بدون Math.abs برای حفظ علامت)
        measures.Overjet = Math.round((horizontalDistance * conversionFactor) * 10) / 10;
      }
      
      // SN-GoGn (همان GoGn-SN)
      if (landmarks.S && landmarks.N && landmarks.Go && landmarks.Gn && !measures['GoGn-SN']) {
        const snAngle = calculateLineAngle(landmarks.S, landmarks.N);
        const gognAngle = calculateLineAngle(landmarks.Go, landmarks.Gn);
        measures['SN-GoGn'] = Math.abs(snAngle - gognAngle);
      }
      
      // Facial Axis (Ba-Na to Pt-Gn)
      if (landmarks.Ba && landmarks.Na && landmarks.Pt && landmarks.Gn) {
        const baNaAngle = calculateLineAngle(landmarks.Ba, landmarks.Na);
        const ptGnAngle = calculateLineAngle(landmarks.Pt, landmarks.Gn);
        measures.FacialAxis = Math.abs(baNaAngle - ptGnAngle);
      }
      
      // Mandibular Plane Angle
      if (landmarks.Go && landmarks.Me) {
        const mpAngle = calculateLineAngle(landmarks.Go, landmarks.Me);
        measures.MandibularPlane = mpAngle;
      }
      
      // Upper Face Height / Lower Face Height
      if (landmarks.N && landmarks.ANS && landmarks.Me) {
        const upperFaceHeight = Math.sqrt(
          (landmarks.N.x - landmarks.ANS.x)**2 + 
          (landmarks.N.y - landmarks.ANS.y)**2
        );
        const lowerFaceHeight = Math.sqrt(
          (landmarks.ANS.x - landmarks.Me.x)**2 + 
          (landmarks.ANS.y - landmarks.Me.y)**2
        );
        if (lowerFaceHeight > 0) {
          measures.UpperLowerFaceRatio = upperFaceHeight / lowerFaceHeight;
        }
      }
      
      // ========== Ricketts Analysis Parameters ==========
      
      // Helper function to calculate distance between two points
      // Convert pixels to millimeters for medical accuracy using dynamic calibration
      const calculateDistance = (p1, p2) => {
        if (!p1 || !p2) return 0;
        const pixelDistance = Math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2);
        
        // Use dynamic calibration if available, otherwise use fallback
        // The issue: 1 pixel = 0.15 mm was too small.
        // Most radiographs have 1 pixel ≈ 0.11 mm for realistic measurements
        const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const mmDistance = pixelDistance * conversionFactor;
        
        // 🔧 FIX: Removed excessive debug logging - only log in development if needed
        
        return mmDistance;
      };
      
      // Helper function to calculate distance from point to line (in millimeters)
      const distancePointToLine = (point, lineStart, lineEnd) => {
        const A = point.x - lineStart.x;
        const B = point.y - lineStart.y;
        const C = lineEnd.x - lineStart.x;
        const D = lineEnd.y - lineStart.y;
        const dot = A * C + B * D;
        const lenSq = C * C + D * D;
        if (lenSq === 0) return calculateDistance(point, lineStart);
        const param = dot / lenSq;
        let xx; let yy;
        if (param < 0) {
          xx = lineStart.x;
          yy = lineStart.y;
        } else if (param > 1) {
          xx = lineEnd.x;
          yy = lineEnd.y;
        } else {
          xx = lineStart.x + param * C;
          yy = lineStart.y + param * D;
        }
        const dx = point.x - xx;
        const dy = point.y - yy;
        const pixelDistance = Math.sqrt(dx * dx + dy * dy);
        
        // Convert to millimeters using the same conversion factor
        const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const mmDistance = pixelDistance * conversionFactor;
        
        return mmDistance;
      };
      
      // 1. Facial Axis - زاویه بین Gn-Pt-Ba (زاویه در نقطه Pt)
      const baLandmark = getLandmark(['Ba', 'ba', 'BA', 'basion', 'Basion']);
      const ptLandmark = getLandmark(['Pt', 'pt', 'PT', 'pterygoid', 'Pterygoid']);
      const gnLandmarkFacialAxis = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      
      if (baLandmark && ptLandmark && gnLandmarkFacialAxis) {
        // محاسبه زاویه در نقطه Pt بین خطوط Gn-Pt و Ba-Pt
        const facialAxis = calculateAngle(
          gnLandmarkFacialAxis,  // نقطه اول: Gn
          ptLandmark,              // vertex: Pt
          baLandmark               // نقطه دوم: Ba
        );
        measures['Facial Axis'] = Math.round(facialAxis * 10) / 10;
      }
      
      // 2. Facial Depth - زاویه بین امتداد خطوط N-Pog و Or-Po
      const nLandmarkRicketts = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const pogLandmarkRicketts = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      const orLandmarkRicketts = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkRicketts = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      
      if (nLandmarkRicketts && pogLandmarkRicketts && orLandmarkRicketts && poLandmarkRicketts) {
        // محاسبه زاویه بین دو خط N-Pog و Or-Po
        const facialDepth = calculateAngleBetweenLines(
          nLandmarkRicketts, pogLandmarkRicketts,  // خط اول: N-Pog
          orLandmarkRicketts, poLandmarkRicketts   // خط دوم: Or-Po (Frankfort Horizontal)
        );
        measures['Facial Depth'] = Math.round(facialDepth * 10) / 10;
      }
      
      // 3. Lower Face Height - نسبت فاصله عمودی ANS-Me به فاصله عمودی N-Me × 100
      const ansLandmarkRicketts = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const meLandmarkRicketts = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      const nLandmarkLFH = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      
      if (ansLandmarkRicketts && meLandmarkRicketts && nLandmarkLFH) {
        // فقط فاصله عمودی (تفاوت y) - در پیکسل
        const ansMeVerticalPixels = Math.abs(ansLandmarkRicketts.y - meLandmarkRicketts.y);
        const nMeVerticalPixels = Math.abs(nLandmarkLFH.y - meLandmarkRicketts.y);
        
        // بررسی اعتبار: فاصله‌ها باید منطقی باشند
        if (nMeVerticalPixels > 10 && ansMeVerticalPixels > 5) {
          // محاسبه نسبت (درصد) - بدون نیاز به تبدیل به میلی‌متر چون نسبت است
          const ratio = (ansMeVerticalPixels / nMeVerticalPixels) * 100;
          // همیشه مقدار را محاسبه کن (حتی اگر خارج از محدوده باشد) تا کاربر ببیند
          measures['Lower Face Height'] = Math.round(ratio * 10) / 10;
        }
      }
      
      // 4. Mandibular Plane - زاویه Go-Me نسبت به FH (Or-Po)
      const goLandmarkRicketts = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const meLandmarkMP = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      const orLandmarkMP = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkMP = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      
      if (goLandmarkRicketts && meLandmarkMP && orLandmarkMP && poLandmarkMP) {
        const goMeAngle = calculateLineAngle(goLandmarkRicketts, meLandmarkMP);
        const orPoAngle = calculateLineAngle(orLandmarkMP, poLandmarkMP);
        let mandibularPlane = Math.abs(goMeAngle - orPoAngle);
        if (mandibularPlane > 90) mandibularPlane = 180 - mandibularPlane;
        measures['Mandibular Plane'] = Math.round(mandibularPlane * 10) / 10;
      }
      
      // 5. Convexity - فاصله A point از خط N-Pog (عمود بر خط)
      const aLandmarkConvexity = getLandmark(['A', 'a', 'Point A']);
      const nLandmarkConvexity = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const pogLandmarkConvexity = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      
      if (aLandmarkConvexity && nLandmarkConvexity && pogLandmarkConvexity) {
        const convexityDist = distancePointToLine(aLandmarkConvexity, nLandmarkConvexity, pogLandmarkConvexity);
        // تعیین علامت: اگر A در سمت راست خط N-Pog باشد، مثبت است
        const lineVecX = pogLandmarkConvexity.x - nLandmarkConvexity.x;
        const lineVecY = pogLandmarkConvexity.y - nLandmarkConvexity.y;
        const pointVecX = aLandmarkConvexity.x - nLandmarkConvexity.x;
        const pointVecY = aLandmarkConvexity.y - nLandmarkConvexity.y;
        const crossProduct = lineVecX * pointVecY - lineVecY * pointVecX;
        const sign = crossProduct > 0 ? 1 : -1;
        const convexityValue = sign * convexityDist;
        
        // محدود کردن به محدوده منطقی (معمولاً بین -10 تا +10 میلی‌متر)
        // 🔧 FIX: ضرب در -1 برای معکوس کردن علامت
        if (Math.abs(convexityValue) <= 20) {
          measures.Convexity = Math.round(-convexityValue * 10) / 10;
        }
      }
      
      // 6. Upper Incisor - زاویه نسبت به A-Pog
      const u1LandmarkRicketts = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper Incisor']);
      const aLandmarkUI = getLandmark(['A', 'a', 'Point A']);
      const pogLandmarkUI = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      
      if (u1LandmarkRicketts && aLandmarkUI && pogLandmarkUI) {
        // زاویه بین خط U1-A و خط A-Pog
        const upperIncisorAngle = calculateAngle(u1LandmarkRicketts, aLandmarkUI, pogLandmarkUI);
        measures['Upper Incisor'] = Math.round(upperIncisorAngle * 10) / 10;
      }
      
      // 9. Lower Incisor - زاویه نسبت به A-Pog
      const l1LandmarkRicketts = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower Incisor']);
      const aLandmarkLI = getLandmark(['A', 'a', 'Point A']);
      const pogLandmarkLI = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      
      if (l1LandmarkRicketts && aLandmarkLI && pogLandmarkLI) {
        const lowerIncisorAngle = calculateAngle(l1LandmarkRicketts, aLandmarkLI, pogLandmarkLI);
        measures['Lower Incisor'] = Math.round(lowerIncisorAngle * 10) / 10;
      }
      
      // 10. Interincisal Angle - زاویه بین دندان‌های پیشین (قبلاً محاسبه شده)
      if (measures.InterincisalAngle) {
        measures['Interincisal Angle'] = Math.round(measures.InterincisalAngle * 10) / 10;
      }
      
      // 11. Occlusal Plane Angle - زاویه بین Occlusal Plane (L1-LMT) و خط S-N
      const l1LandmarkOcclusal = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower Incisor']);
      const lmtLandmark = getLandmark(['LMT', 'lmt', 'LMT', 'lower_molar', 'Lower Molar', 'L6', 'l6']);
      const sLandmarkOcclusal = getLandmark(['S', 's', 'Sella', 'sella']);
      const nLandmarkOcclusal = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      
      if (l1LandmarkOcclusal && lmtLandmark && sLandmarkOcclusal && nLandmarkOcclusal) {
        let occlusalPlaneAngle = calculateAngleBetweenLines(
          l1LandmarkOcclusal, lmtLandmark,  // خط اول: Occlusal Plane (L1-LMT)
          sLandmarkOcclusal, nLandmarkOcclusal   // خط دوم: S-N
        );
        // اطمینان از اینکه زاویه کوچک‌تر (acute angle) استفاده می‌شود
        // اگر زاویه بیشتر از 90 درجه باشد، از زاویه مکمل استفاده می‌کنیم
        if (occlusalPlaneAngle > 90) {
          occlusalPlaneAngle = 180 - occlusalPlaneAngle;
        }
        measures['Occlusal Plane Angle'] = Math.round(occlusalPlaneAngle * 10) / 10;
      }
      
      // 12. Cranial Deflection - زاویه بین Ba-N-S (زاویه در نقطه N)
      const baLandmarkCranial = getLandmark(['Ba', 'ba', 'BA', 'basion', 'Basion']);
      const nLandmarkCranial = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const sLandmarkCranial = getLandmark(['S', 's', 'Sella', 'sella']);
      
      if (baLandmarkCranial && nLandmarkCranial && sLandmarkCranial) {
        // محاسبه زاویه در نقطه N بین خطوط Ba-N و N-S
        const cranialDeflection = calculateAngle(
          baLandmarkCranial,  // نقطه اول: Ba
          nLandmarkCranial,    // vertex: N
          sLandmarkCranial     // نقطه دوم: S
        );
        measures['Cranial Deflection'] = Math.round(cranialDeflection * 10) / 10;
      }
      
      // 13. Palatal Plane Angle - زاویه بین خط سقف دهان (ANS-PNS) و خط S-N
      const ansLandmarkPalatal = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const pnsLandmarkPalatal = getLandmark(['PNS', 'pns', 'Posterior Nasal Spine']);
      const sLandmarkPalatal = getLandmark(['S', 's', 'Sella', 'sella']);
      const nLandmarkPalatal = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      
      if (ansLandmarkPalatal && pnsLandmarkPalatal && sLandmarkPalatal && nLandmarkPalatal) {
        let palatalPlaneAngle = calculateAngleBetweenLines(
          ansLandmarkPalatal, pnsLandmarkPalatal,  // خط اول: Palatal Plane (ANS-PNS)
          sLandmarkPalatal, nLandmarkPalatal        // خط دوم: S-N
        );
        // اطمینان از اینکه زاویه کوچک‌تر (acute angle) استفاده می‌شود
        if (palatalPlaneAngle > 90) {
          palatalPlaneAngle = 180 - palatalPlaneAngle;
        }
        measures['Palatal Plane Angle'] = Math.round(palatalPlaneAngle * 10) / 10;
      }
      
      // 13. E-line - فاصله از UL و LL تا خط E-line (Prn-Pog')
      // استفاده از همان روش Arnett برای محاسبه دقیق‌تر
      const prnLandmarkRicketts = getLandmark(['Pn', 'pn', 'PN', 'Prn', 'prn', 'PRN', 'Pronasale', 'pronasale']);
      const pgPrimeLandmarkRicketts = getLandmark(['Pog\'', 'Pog\'', 'pog\'', 'PogSoft', 'pogsoft', 'POGSOFT', 'Soft Pogonion', 'Pg\'', 'pg\'', 'PgPrime', 'pgPrime']);
      const lsLandmarkRicketts = getLandmark(['Ls', 'ls', 'LS', 'UL', 'ul', 'Labiale Superius', 'labiale superius', 'Upper Lip', 'upper_lip']);
      const liLandmarkRicketts = getLandmark(['Li', 'li', 'LI', 'LL', 'll', 'Labiale Inferius', 'labiale inferius', 'Lower Lip', 'lower_lip']);
      
      // 13.1. E-line (UL) - فاصله لب بالا تا E-line در میلی‌متر
      if (prnLandmarkRicketts && pgPrimeLandmarkRicketts && lsLandmarkRicketts) {
        // E-line: خط Prn-Pog'
        // فاصله از Ls تا خط Prn-Pog' در میلی‌متر
        const eLineDistanceMm = calculateDistanceToLineMm(lsLandmarkRicketts, prnLandmarkRicketts, pgPrimeLandmarkRicketts);
        if (eLineDistanceMm !== null) {
          // تعیین علامت: اگر Ls در سمت راست خط باشد (جلوتر)، مثبت است
          const sign = ((pgPrimeLandmarkRicketts.x - prnLandmarkRicketts.x) * (lsLandmarkRicketts.y - prnLandmarkRicketts.y) - 
                       (pgPrimeLandmarkRicketts.y - prnLandmarkRicketts.y) * (lsLandmarkRicketts.x - prnLandmarkRicketts.x)) > 0 ? 1 : -1;
          measures['E-line (UL)'] = Math.round(eLineDistanceMm * sign * 10) / 10;
        }
      }
      
      // 13.2. E-line (LL) - فاصله لب پایین تا E-line در میلی‌متر
      if (prnLandmarkRicketts && pgPrimeLandmarkRicketts && liLandmarkRicketts) {
        const eLineDistanceMm = calculateDistanceToLineMm(liLandmarkRicketts, prnLandmarkRicketts, pgPrimeLandmarkRicketts);
        if (eLineDistanceMm !== null) {
          const sign = ((pgPrimeLandmarkRicketts.x - prnLandmarkRicketts.x) * (liLandmarkRicketts.y - prnLandmarkRicketts.y) - 
                       (pgPrimeLandmarkRicketts.y - prnLandmarkRicketts.y) * (liLandmarkRicketts.x - prnLandmarkRicketts.x)) > 0 ? 1 : -1;
          measures['E-line (LL)'] = Math.round(eLineDistanceMm * sign * 10) / 10;
        }
      }
      
      // ========== McNamara Analysis Parameters ==========
      
      // 1. N-A-Pog - زاویه بین دو خط N-A و A-Pog (تحدب صورت)
      const nLandmarkMcNamara = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const aLandmarkMcNamara = getLandmark(['A', 'a', 'Point A']);
      const pogLandmarkMcNamara = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      
      if (nLandmarkMcNamara && aLandmarkMcNamara && pogLandmarkMcNamara) {
        // محاسبه زاویه بین دو خط: خط N-A و خط A-Pog
        // تابع calculateAngleBetweenLines خودش زاویه کوچک‌تر را برمی‌گرداند
        const naPogAngle = calculateAngleBetweenLines(
          nLandmarkMcNamara, aLandmarkMcNamara,  // خط اول: N-A
          aLandmarkMcNamara, pogLandmarkMcNamara // خط دوم: A-Pog
        );
        measures['N-A-Pog'] = Math.round(naPogAngle * 10) / 10;
      }
      
      // 2. Co-A - طول فک بالا (فاصله Co-A) - فقط اگر p1 و p2 وجود داشته باشند
      const p1Landmark = getLandmark(['p1', 'P1']);
      const p2Landmark = getLandmark(['p2', 'P2']);
      const hasCalibrationPoints = p1Landmark && p2Landmark;
      
      if (hasCalibrationPoints) {
      const coLandmark = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                        getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const aLandmarkCoA = getLandmark(['A', 'a', 'Point A']);
      
      if (coLandmark && aLandmarkCoA) {
        const coADist = calculateDistance(coLandmark, aLandmarkCoA);
        if (coADist !== null) {
          measures['Co-A'] = Math.round(coADist * 10) / 10;
          measures['Maxillary Length'] = measures['Co-A']; // همان Co-A است
        }
      }
      
        // 3. Co-Gn - طول فک پایین (فاصله Co-Gn) - فقط اگر p1 و p2 وجود داشته باشند
      const gnLandmarkCoGn = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      
      if (coLandmark && gnLandmarkCoGn) {
        const coGnDist = calculateDistance(coLandmark, gnLandmarkCoGn);
        if (coGnDist !== null) {
          measures['Co-Gn'] = Math.round(coGnDist * 10) / 10;
          measures['Mandibular Length'] = measures['Co-Gn']; // همان Co-Gn است
        }
        }
      } else {
        // اگر p1 و p2 وجود نداشتند، این پارامترها را تعریف نکن
      }
      
      // 4. Wits Appraisal - فاصله AO-BO (نقطه A و B روی صفحه اکلوزال)
      // AO: نقطه A روی صفحه اکلوزال (عمود از A به صفحه اکلوزال)
      // BO: نقطه B روی صفحه اکلوزال (عمود از B به صفحه اکلوزال)
      // Wits = فاصله افقی بین AO و BO (در میلی‌متر)
      // Occlusal Plane: خط عبور کننده از L1 و LMT
      const aLandmarkWits = getLandmark(['A', 'a', 'Point A']);
      const bLandmarkWits = getLandmark(['B', 'b', 'Point B']);
      const l1LandmarkWits = getLandmark(['L1', 'l1', 'lower', 'incisor', 'li', 'lower_incisor']);
      const lmtLandmarkWits = getLandmark(['LMT', 'lmt', 'LMT', 'lower_molar', 'Lower Molar', 'L6', 'l6']);
      
      if (aLandmarkWits && bLandmarkWits && l1LandmarkWits && lmtLandmarkWits) {
        // محاسبه Occlusal Plane (خط L1-LMT)
        const occlusalDx = lmtLandmarkWits.x - l1LandmarkWits.x;
        const occlusalDy = lmtLandmarkWits.y - l1LandmarkWits.y;
        const occlusalLength = Math.sqrt(occlusalDx * occlusalDx + occlusalDy * occlusalDy);
        
        if (occlusalLength > 0) {
          const occlusalDirX = occlusalDx / occlusalLength;
          const occlusalDirY = occlusalDy / occlusalLength;
          
          // محاسبه نقطه تقاطع عمود از A با Occlusal Plane (AO)
          const getPerpendicularIntersection = (point, lineStart, lineDirX, lineDirY) => {
            const dx = point.x - lineStart.x;
            const dy = point.y - lineStart.y;
            const t = (dx * lineDirX + dy * lineDirY) / (lineDirX * lineDirX + lineDirY * lineDirY);
            
            return {
              x: lineStart.x + t * lineDirX,
              y: lineStart.y + t * lineDirY
            };
          };
          
          const aoPoint = getPerpendicularIntersection(aLandmarkWits, l1LandmarkWits, occlusalDirX, occlusalDirY);
          const boPoint = getPerpendicularIntersection(bLandmarkWits, l1LandmarkWits, occlusalDirX, occlusalDirY);
          
          // محاسبه فاصله AO-BO (فاصله افقی روی Occlusal Plane)
          // در Wits Appraisal: فاصله در راستای Occlusal Plane (نه فاصله اقلیدسی)
          // از بردار Occlusal Plane برای محاسبه فاصله افقی استفاده می‌کنیم
          const aoBoVectorX = boPoint.x - aoPoint.x;
          const aoBoVectorY = boPoint.y - aoPoint.y;
          // فاصله در راستای Occlusal Plane (projection)
          const aoBoDistancePixels = aoBoVectorX * occlusalDirX + aoBoVectorY * occlusalDirY;
          
          // تبدیل به میلی‌متر
        const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
          const aoBoDistanceMm = Math.abs(aoBoDistancePixels) * conversionFactor;
          
          // تعیین علامت: در Wits Appraisal
          // اگر AO در سمت راست BO باشد (x بزرگتر)، یعنی A جلوتر از B است، مقدار مثبت (Class II)
          // اگر AO در سمت چپ BO باشد (x کوچکتر)، یعنی A عقب‌تر از B است، مقدار منفی (Class III)
          const sign = aoPoint.x > boPoint.x ? 1 : -1;
          measures['Wits Appraisal'] = Math.round(sign * aoBoDistanceMm * 10) / 10;
        measures['AO-BO'] = measures['Wits Appraisal']; // همان Wits Appraisal است
        }
      }
      
      // 5. Lower Face Height - ارتفاع صورت پایین (فاصله عمودی ANS-Me در میلی‌متر)
      const ansLandmarkLFH = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const meLandmarkLFH = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      
      if (ansLandmarkLFH && meLandmarkLFH) {
        // فقط فاصله عمودی (تفاوت y) - در پیکسل
        const ansMeVerticalPixels = Math.abs(ansLandmarkLFH.y - meLandmarkLFH.y);
        
        // تبدیل به میلی‌متر
        const conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const ansMeVerticalMm = ansMeVerticalPixels * conversionFactor;
        
        // بررسی اعتبار: فاصله باید منطقی باشد (بین 30-100 میلی‌متر معمولاً)
        if (ansMeVerticalMm >= 20 && ansMeVerticalMm <= 120) {
          measures['Lower Face Height'] = Math.round(ansMeVerticalMm * 10) / 10;
        }
      }
      
      // 6. Upper Face Height - ارتفاع صورت بالا (N-ANS)
      const nLandmarkUFH = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const ansLandmarkUFH = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      
      if (nLandmarkUFH && ansLandmarkUFH) {
        const nAnsDist = calculateDistance(nLandmarkUFH, ansLandmarkUFH);
        if (nAnsDist !== null) {
          measures['Upper Face Height'] = Math.round(nAnsDist * 10) / 10;
        }
      }
      
      // 7. Facial Height Ratio - نسبت ارتفاع صورت (ANS-Me/N-Me × 100)
      const nLandmarkFHR = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const ansLandmarkFHR = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const meLandmarkFHR = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      
      if (nLandmarkFHR && ansLandmarkFHR && meLandmarkFHR) {
        const ansMeDist = calculateDistance(ansLandmarkFHR, meLandmarkFHR);
        const nMeDist = calculateDistance(nLandmarkFHR, meLandmarkFHR);
        if (ansMeDist !== null && nMeDist !== null && nMeDist > 0) {
          measures['Facial Height Ratio'] = Math.round((ansMeDist / nMeDist) * 100 * 10) / 10;
        }
      }
      
      // 8. Mandibular Plane Angle - زاویه صفحه مندیبولار (همان FMA یا GoGn-SN)
      if (measures.FMA) {
        measures['Mandibular Plane Angle'] = measures.FMA;
      } else if (measures['GoGn-SN']) {
        measures['Mandibular Plane Angle'] = measures['GoGn-SN'];
      }
      
      // 9. PFH/AFH Ratio - نسبت ارتفاع خلفی به قدامی
      // PFH (Posterior Facial Height): فاصله S → Go
      // AFH (Anterior Facial Height): فاصله N → Me
      const sLandmarkPFH = getLandmark(['S', 's', 'Sella', 'sella']);
      const goLandmarkPFH = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const nLandmarkAFH = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const meLandmarkAFH = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      
      if (sLandmarkPFH && goLandmarkPFH && nLandmarkAFH && meLandmarkAFH) {
        const pfhDist = calculateDistance(sLandmarkPFH, goLandmarkPFH);
        const afhDist = calculateDistance(nLandmarkAFH, meLandmarkAFH);
        
        if (pfhDist !== null && afhDist !== null && afhDist > 0) {
          const pfhAfhRatio = pfhDist / afhDist;
          measures['PFH/AFH Ratio'] = Math.round(pfhAfhRatio * 100 * 10) / 10; // به صورت درصد
          measures.PFH = Math.round(pfhDist * 10) / 10; // ارتفاع خلفی (میلی‌متر)
          measures.AFH = Math.round(afhDist * 10) / 10; // ارتفاع قدامی (میلی‌متر)
        }
      }
      
      // ========== Wits Analysis Parameters ==========
      
      // 1. AO-BO - قبلاً محاسبه شد
      // 2. PP/Go-Gn - زاویه بین صفحه پلاتین (ANS-PNS) و صفحه مندیبولار (Go-Gn)
      const ansLandmarkPP = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const pnsLandmark = getLandmark(['PNS', 'pns', 'PNS', 'posterior nasal spine', 'Posterior Nasal Spine']);
      const goLandmarkPP = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const gnLandmarkPP = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      
      if (ansLandmarkPP && pnsLandmark && goLandmarkPP && gnLandmarkPP) {
        const ppGoGnAngle = calculateAngleBetweenLines(
          ansLandmarkPP, pnsLandmark,  // صفحه پلاتین (ANS-PNS)
          goLandmarkPP, gnLandmarkPP   // صفحه مندیبولار (Go-Gn)
        );
        measures['PP/Go-Gn'] = Math.round(ppGoGnAngle * 10) / 10;
      }
      
      // 3. S-Go - ابعاد عمودی چهره (فاصله S-Go)
      const sLandmarkSGo = getLandmark(['S', 's', 'sella', 'Sella']);
      const goLandmarkSGo = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      
      if (sLandmarkSGo && goLandmarkSGo) {
        const sGoDist = calculateDistance(sLandmarkSGo, goLandmarkSGo);
        if (sGoDist !== null) {
          measures['S-Go'] = Math.round(sGoDist * 10) / 10;
        }
      }
      
      // 4. Sagittal Jaw - زاویه ساژیتال فک (معمولاً همان ANB است)
      if (measures.ANB !== undefined) {
        measures['Sagittal Jaw'] = Math.round(measures.ANB * 10) / 10;
      }
      
      // ========== Bjork Analysis Parameters ==========
      
      // لندمارک‌های مشترک
      const sLandmarkBjork = getLandmark(['S', 's', 'sella', 'Sella']);
      const nLandmarkBjork = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const arLandmarkBjork = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const goLandmarkBjork = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const meLandmarkBjork = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      const gnLandmarkBjork = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      const orLandmarkBjork = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkBjork = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      const ansLandmarkBjork = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const pnsLandmarkBjork = getLandmark(['PNS', 'pns', 'PNS', 'posterior nasal spine', 'Posterior Nasal Spine']);
      
      // 1. Saddle Angle (N-S-Ar) - زاویه سدل
      if (nLandmarkBjork && sLandmarkBjork && arLandmarkBjork) {
        const saddleAngle = calculateAngle(nLandmarkBjork, sLandmarkBjork, arLandmarkBjork);
        measures['Saddle Angle (N-S-Ar)'] = Math.round(saddleAngle * 10) / 10;
      }
      
      // 2. Articular Angle (S-Ar-Go) - زاویه آرتیکولار
      if (sLandmarkBjork && arLandmarkBjork && goLandmarkBjork) {
        const articularAngle = calculateAngle(sLandmarkBjork, arLandmarkBjork, goLandmarkBjork);
        measures['Articular Angle (S-Ar-Go)'] = Math.round(articularAngle * 10) / 10;
      }
      
      // 3. Gonial Angle (Ar-Go-Me) - زاویه گونیال
      if (arLandmarkBjork && goLandmarkBjork && meLandmarkBjork) {
        const gonialAngle = calculateAngle(arLandmarkBjork, goLandmarkBjork, meLandmarkBjork);
        measures['Gonial Angle (Ar-Go-Me)'] = Math.round(gonialAngle * 10) / 10;
      }
      
      // 4. Sum of Angles (Posterior Angle Sum) - مجموع زوایا
      if (measures['Saddle Angle (N-S-Ar)'] && measures['Articular Angle (S-Ar-Go)'] && measures['Gonial Angle (Ar-Go-Me)']) {
        const sumOfAngles = measures['Saddle Angle (N-S-Ar)'] + measures['Articular Angle (S-Ar-Go)'] + measures['Gonial Angle (Ar-Go-Me)'];
        measures['Sum of Angles (Posterior Angle Sum)'] = Math.round(sumOfAngles * 10) / 10;
      }
      
      // 5. Y-Axis (SGn to FH) - محور Y
      if (sLandmarkBjork && gnLandmarkBjork && orLandmarkBjork && poLandmarkBjork) {
        // محاسبه زاویه بین خط S-Gn و خط Or-Po (Frankfort Horizontal)
        const yAxisAngle = calculateAngleBetweenLines(
          sLandmarkBjork, gnLandmarkBjork,  // خط S-Gn
          orLandmarkBjork, poLandmarkBjork   // خط Or-Po (FH)
        );
        measures['Y-Axis (SGn to FH)'] = Math.round(yAxisAngle * 10) / 10;
      }
      
      // 6. Facial Height Ratio (Jarabak Ratio) - نسبت قد صورت
      if (sLandmarkBjork && goLandmarkBjork && nLandmarkBjork && meLandmarkBjork) {
        const posteriorHeight = calculateDistance(sLandmarkBjork, goLandmarkBjork); // S-Go
        const anteriorHeight = calculateDistance(nLandmarkBjork, meLandmarkBjork); // N-Me
        if (posteriorHeight !== null && anteriorHeight !== null && anteriorHeight > 0) {
          const jarabakRatio = (posteriorHeight / anteriorHeight) * 100;
          measures['Facial Height Ratio (Jarabak Ratio)'] = Math.round(jarabakRatio * 10) / 10;
        }
      }
      
      // 7. Anterior Facial Height (N-Me)
      if (nLandmarkBjork && meLandmarkBjork) {
        const anteriorHeight = calculateDistance(nLandmarkBjork, meLandmarkBjork);
        if (anteriorHeight !== null) {
          measures['Anterior Facial Height (N-Me)'] = Math.round(anteriorHeight * 10) / 10;
        }
      }
      
      // 8. Posterior Facial Height (S-Go)
      if (sLandmarkBjork && goLandmarkBjork) {
        const posteriorHeight = calculateDistance(sLandmarkBjork, goLandmarkBjork);
        if (posteriorHeight !== null) {
          measures['Posterior Facial Height (S-Go)'] = Math.round(posteriorHeight * 10) / 10;
        }
      }
      
      // 9. Ramus Height (Ar-Go)
      if (arLandmarkBjork && goLandmarkBjork) {
        const ramusHeight = calculateDistance(arLandmarkBjork, goLandmarkBjork);
        if (ramusHeight !== null) {
          measures['Ramus Height (Ar-Go)'] = Math.round(ramusHeight * 10) / 10;
        }
      }
      
      // 10. Mandibular Plane Angle (FH to MP) - زاویه صفحه مندیبولار
      if (goLandmarkBjork && meLandmarkBjork && orLandmarkBjork && poLandmarkBjork) {
        const mandibularPlaneAngle = calculateAngleBetweenLines(
          orLandmarkBjork, poLandmarkBjork,  // خط Or-Po (FH)
          goLandmarkBjork, meLandmarkBjork   // خط Go-Me (MP)
        );
        measures['Mandibular Plane Angle (FH to MP)'] = Math.round(mandibularPlaneAngle * 10) / 10;
      }
      
      // 11. Basal Plane Angle (PNS-ANS to Go-Me)
      if (pnsLandmarkBjork && ansLandmarkBjork && goLandmarkBjork && meLandmarkBjork) {
        const basalPlaneAngle = calculateAngleBetweenLines(
          pnsLandmarkBjork, ansLandmarkBjork,  // خط PNS-ANS
          goLandmarkBjork, meLandmarkBjork    // خط Go-Me
        );
        measures['Basal Plane Angle (PNS-ANS to Go-Me)'] = Math.round(basalPlaneAngle * 10) / 10;
      }
      
      // 12. SNA, SNB, ANB - این‌ها قبلاً در بخش Steiner محاسبه شده‌اند
      // فقط اگر موجود باشند، در measures نگه می‌داریم
      // مقادیر قبلاً محاسبه شده‌اند
      
      // ========== Jarabak Analysis Parameters ==========
      
      // لندمارک‌های مشترک
      const sLandmarkJarabak = getLandmark(['S', 's', 'Sella', 'sella']);
      const nLandmarkJarabak = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const arLandmarkJarabak = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const goLandmarkJarabak = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const meLandmarkJarabak = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      const gnLandmarkJarabak = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      const orLandmarkJarabak = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkJarabak = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      const ansLandmarkJarabak = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const pnsLandmarkJarabak = getLandmark(['PNS', 'pns', 'PNS', 'posterior nasal spine', 'Posterior Nasal Spine']);
      
      // 1. Jarabak Ratio (Facial Height Ratio) = (S-Go / N-Me) × ۱۰۰
      if (sLandmarkJarabak && goLandmarkJarabak && nLandmarkJarabak && meLandmarkJarabak) {
        const pfhDist = calculateDistance(sLandmarkJarabak, goLandmarkJarabak);
        const afhDist = calculateDistance(nLandmarkJarabak, meLandmarkJarabak);
        if (pfhDist !== null && afhDist !== null && afhDist > 0) {
          const jarabakRatio = (pfhDist / afhDist) * 100;
          measures['Jarabak Ratio (Facial Height Ratio)'] = Math.round(jarabakRatio * 10) / 10;
        }
      }
      
      // 2. Posterior Facial Height (PFH) - S-Go
      if (sLandmarkJarabak && goLandmarkJarabak) {
        const pfhDist = calculateDistance(sLandmarkJarabak, goLandmarkJarabak);
        if (pfhDist !== null) {
          measures['Posterior Facial Height (PFH)'] = Math.round(pfhDist * 10) / 10;
        }
      }
      
      // 3. Anterior Facial Height (AFH) - N-Me
      if (nLandmarkJarabak && meLandmarkJarabak) {
        const afhDist = calculateDistance(nLandmarkJarabak, meLandmarkJarabak);
        if (afhDist !== null) {
          measures['Anterior Facial Height (AFH)'] = Math.round(afhDist * 10) / 10;
        }
      }
      
      // 4. Saddle Angle - ∠N-S-Ar
      if (nLandmarkJarabak && sLandmarkJarabak && arLandmarkJarabak) {
        const saddleAngle = calculateAngle(nLandmarkJarabak, sLandmarkJarabak, arLandmarkJarabak);
        measures['Saddle Angle'] = Math.round(saddleAngle * 10) / 10;
      }
      
      // 5. Articular Angle - ∠S-Ar-Go
      if (sLandmarkJarabak && arLandmarkJarabak && goLandmarkJarabak) {
        const articularAngle = calculateAngle(sLandmarkJarabak, arLandmarkJarabak, goLandmarkJarabak);
        measures['Articular Angle'] = Math.round(articularAngle * 10) / 10;
      }
      
      // 6. Gonial Angle (Total) - ∠Ar-Go-Me
      if (arLandmarkJarabak && goLandmarkJarabak && meLandmarkJarabak) {
        const gonialAngle = calculateAngle(arLandmarkJarabak, goLandmarkJarabak, meLandmarkJarabak);
        measures['Gonial Angle (Total)'] = Math.round(gonialAngle * 10) / 10;
      }
      
      // 7. Upper Gonial Angle - ∠Ar-Go-N
      if (arLandmarkJarabak && goLandmarkJarabak && nLandmarkJarabak) {
        const upperGonialAngle = calculateAngle(arLandmarkJarabak, goLandmarkJarabak, nLandmarkJarabak);
        measures['Upper Gonial Angle'] = Math.round(upperGonialAngle * 10) / 10;
      }
      
      // 8. Lower Gonial Angle - ∠N-Go-Gn
      if (nLandmarkJarabak && goLandmarkJarabak && gnLandmarkJarabak) {
        const lowerGonialAngle = calculateAngle(nLandmarkJarabak, goLandmarkJarabak, gnLandmarkJarabak);
        measures['Lower Gonial Angle'] = Math.round(lowerGonialAngle * 10) / 10;
      }
      
      // 9. Sum of Posterior Angles - Saddle + Articular + Gonial
      if (measures['Saddle Angle'] && measures['Articular Angle'] && measures['Gonial Angle (Total)']) {
        const sumOfAngles = measures['Saddle Angle'] + measures['Articular Angle'] + measures['Gonial Angle (Total)'];
        measures['Sum of Posterior Angles'] = Math.round(sumOfAngles * 10) / 10;
      }
      
      // 10. Y-Axis (Growth Axis) - ∠SGn–FH
      if (sLandmarkJarabak && gnLandmarkJarabak && orLandmarkJarabak && poLandmarkJarabak) {
        const yAxisAngle = calculateAngleBetweenLines(
          sLandmarkJarabak, gnLandmarkJarabak,  // خط S-Gn
          orLandmarkJarabak, poLandmarkJarabak   // خط Or-Po (FH)
        );
        measures['Y-Axis (Growth Axis)'] = Math.round(yAxisAngle * 10) / 10;
      }
      
      // 11. Mandibular Plane Angle - ∠FH–MP (Go-Me یا Go-Gn)
      if (goLandmarkJarabak && meLandmarkJarabak && orLandmarkJarabak && poLandmarkJarabak) {
        const mandibularPlaneAngle = calculateAngleBetweenLines(
          orLandmarkJarabak, poLandmarkJarabak,  // خط Or-Po (FH)
          goLandmarkJarabak, meLandmarkJarabak   // خط Go-Me (MP)
        );
        measures['Mandibular Plane Angle'] = Math.round(mandibularPlaneAngle * 10) / 10;
      }
      
      // 12. Basal Plane Angle - ∠PNS-ANS به Go-Me
      if (pnsLandmarkJarabak && ansLandmarkJarabak && goLandmarkJarabak && meLandmarkJarabak) {
        const basalPlaneAngle = calculateAngleBetweenLines(
          pnsLandmarkJarabak, ansLandmarkJarabak,  // خط PNS-ANS
          goLandmarkJarabak, meLandmarkJarabak    // خط Go-Me
        );
        measures['Basal Plane Angle'] = Math.round(basalPlaneAngle * 10) / 10;
      }
      
      // 13. Ramus Height - Ar-Go
      if (arLandmarkJarabak && goLandmarkJarabak) {
        const ramusHeight = calculateDistance(arLandmarkJarabak, goLandmarkJarabak);
        if (ramusHeight !== null) {
          measures['Ramus Height'] = Math.round(ramusHeight * 10) / 10;
        }
      }
      
      // 14. Mandibular Arc - زاویه بین Corpus Axis و Condylar Axis
      // این زاویه بین خط Go-Me (Corpus) و خط Ar-Go (Condylar) است
      if (arLandmarkJarabak && goLandmarkJarabak && meLandmarkJarabak) {
        const mandibularArc = calculateAngle(arLandmarkJarabak, goLandmarkJarabak, meLandmarkJarabak);
        measures['Mandibular Arc'] = Math.round(mandibularArc * 10) / 10;
      }
      
      // 15. Palatal Plane to FH - ∠ANS-PNS به FH
      if (ansLandmarkJarabak && pnsLandmarkJarabak && orLandmarkJarabak && poLandmarkJarabak) {
        const palatalPlaneAngle = calculateAngleBetweenLines(
          ansLandmarkJarabak, pnsLandmarkJarabak,  // خط ANS-PNS
          orLandmarkJarabak, poLandmarkJarabak      // خط Or-Po (FH)
        );
        measures['Palatal Plane to FH'] = Math.round(palatalPlaneAngle * 10) / 10;
      }
      
      // 16. Occlusal Plane to FH - ∠Occlusal Plane به FH
      // برای محاسبه Occlusal Plane، نیاز به نقاط U1 و L1 داریم
      const u1LandmarkOcclusalJarabak = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper Incisor']);
      const l1LandmarkOcclusalJarabak = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower Incisor']);
      if (u1LandmarkOcclusalJarabak && l1LandmarkOcclusalJarabak && orLandmarkJarabak && poLandmarkJarabak) {
        const occlusalPlaneAngle = calculateAngleBetweenLines(
          u1LandmarkOcclusalJarabak, l1LandmarkOcclusalJarabak,  // خط Occlusal Plane (تقریبی)
          orLandmarkJarabak, poLandmarkJarabak      // خط Or-Po (FH)
        );
        measures['Occlusal Plane to FH'] = Math.round(occlusalPlaneAngle * 10) / 10;
      }
      
      // ========== Sassouni Analysis Parameters ==========
      
      // 1. N-S-Ar - زاویه بین nasion، sella و articulare
      const nLandmarkSassouni = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const sLandmarkSassouni = getLandmark(['S', 's', 'sella', 'Sella']);
      const arLandmarkSassouni = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      
      if (nLandmarkSassouni && sLandmarkSassouni && arLandmarkSassouni) {
        const nsArAngle = calculateAngle(nLandmarkSassouni, sLandmarkSassouni, arLandmarkSassouni);
        measures['N-S-Ar'] = Math.round(nsArAngle * 10) / 10;
      }
      
      // 2. زاویه بین S-N و Gn-Me (امتداد خطوط)
      const gnLandmarkSassouniSN = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      const meLandmarkSassouniSN = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      const goLandmarkSassouniSN = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      
      if (nLandmarkSassouni && sLandmarkSassouni && meLandmarkSassouniSN && goLandmarkSassouniSN) {
        let snMeGoAngle = calculateAngleBetweenLines(
          sLandmarkSassouni, nLandmarkSassouni,  // خط اول: S-N
          meLandmarkSassouniSN, goLandmarkSassouniSN  // خط دوم: Me-Go
        );
        // برای Sassouni: باید زاویه کوچک‌تر (مکمل) را بگیریم
        // اگر زاویه بیشتر از 90 درجه است، زاویه مکمل (کوچک‌تر) را بگیریم
        if (snMeGoAngle > 90) {
          snMeGoAngle = 180 - snMeGoAngle;
        }
        measures['S-N/Me-Go'] = Math.round(snMeGoAngle * 10) / 10;
      }
      
      // 3. زاویه Gn-S-N حذف شده
      
      // 4. زاویه S-Ar-Go (در نقطه Ar)
      const goLandmarkSassouniSAr = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      if (sLandmarkSassouni && arLandmarkSassouni && goLandmarkSassouniSAr) {
        const sArGoAngle = calculateAngle(sLandmarkSassouni, arLandmarkSassouni, goLandmarkSassouniSAr);
        measures['S-Ar-Go'] = Math.round(sArGoAngle * 10) / 10;
      }
      
      // 5. زاویه Ar-Go-Me (در نقطه Go)
      if (arLandmarkSassouni && goLandmarkSassouniSAr && meLandmarkSassouniSN) {
        const arGoMeAngle = calculateAngle(arLandmarkSassouni, goLandmarkSassouniSAr, meLandmarkSassouniSN);
        measures['Ar-Go-Me'] = Math.round(arGoMeAngle * 10) / 10;
      }
      
      // 6. N-Ar-Go - زاویه محدود کننده
      const nLandmarkSassouni2 = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const arLandmarkSassouni2 = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const goLandmarkSassouni = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      
      if (nLandmarkSassouni2 && arLandmarkSassouni2 && goLandmarkSassouni) {
        const nArGoAngle = calculateAngle(nLandmarkSassouni2, arLandmarkSassouni2, goLandmarkSassouni);
        measures['N-Ar-Go'] = Math.round(nArGoAngle * 10) / 10;
      }
      
      // 3. Go-Co-N-S - میزان تمایز ساژیتال (زاویه بین Go-Co و N-S)
      const goLandmarkSassouni3 = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const coLandmarkSassouni = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                                 getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const nLandmarkSassouni3 = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const sLandmarkSassouni3 = getLandmark(['S', 's', 'sella', 'Sella']);
      
      if (goLandmarkSassouni3 && coLandmarkSassouni && nLandmarkSassouni3 && sLandmarkSassouni3) {
        const goCoNSAngle = calculateAngleBetweenLines(
          goLandmarkSassouni3, coLandmarkSassouni,  // Go-Co
          nLandmarkSassouni3, sLandmarkSassouni3    // N-S
        );
        measures['Go-Co-N-S'] = Math.round(goCoNSAngle * 10) / 10;
      }
      
      // 4. Go-Co-Go-Gn - انتخاب اجتماعی (نسبت Go-Co به Go-Gn)
      const goLandmarkSassouni4 = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const coLandmarkSassouni4 = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                                  getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const gnLandmarkSassouni = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      
      if (goLandmarkSassouni4 && coLandmarkSassouni4 && gnLandmarkSassouni) {
        const goCoDist = calculateDistance(goLandmarkSassouni4, coLandmarkSassouni4);
        const goGnDist = calculateDistance(goLandmarkSassouni4, gnLandmarkSassouni);
        if (goCoDist !== null && goGnDist !== null && goGnDist > 0) {
          measures['Go-Co-Go-Gn'] = Math.round((goCoDist / goGnDist) * 100 * 10) / 10;
        }
      }
      
      // 5. N-Co-Go-Co - ایدئال فرهنگی (زاویه بین N-Co و Go-Co)
      const nLandmarkSassouni5 = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const coLandmarkSassouni5 = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                                  getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const goLandmarkSassouni5 = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      
      if (nLandmarkSassouni5 && coLandmarkSassouni5 && goLandmarkSassouni5) {
        const nCoGoCoAngle = calculateAngleBetweenLines(
          nLandmarkSassouni5, coLandmarkSassouni5,  // N-Co
          goLandmarkSassouni5, coLandmarkSassouni5  // Go-Co
        );
        measures['N-Co-Go-Co'] = Math.round(nCoGoCoAngle * 10) / 10;
      }
      
      // 6. Ar-Co-Co-Gn - نخستین sagittal (زاویه بین Ar-Co و Co-Gn)
      const arLandmarkSassouni6 = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const coLandmarkSassouni6 = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                                  getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const gnLandmarkSassouni6 = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
      
      if (arLandmarkSassouni6 && coLandmarkSassouni6 && gnLandmarkSassouni6) {
        const arCoCoGnAngle = calculateAngleBetweenLines(
          arLandmarkSassouni6, coLandmarkSassouni6,  // Ar-Co
          coLandmarkSassouni6, gnLandmarkSassouni6   // Co-Gn
        );
        measures['Ar-Co-Co-Gn'] = Math.round(arCoCoGnAngle * 10) / 10;
      }

      // ========== Legan & Burstone Analysis Parameters (Soft Tissue) ==========
      
      // لندمارک‌های بافت نرم - مطابق با نام‌های مدل CLdetection2023
      const gLandmark = getLandmark(['G', 'g', 'Glabella', 'glabella']); // index 33
      const snLandmark = getLandmark(['Sn', 'sn', 'SN', 'Subnasale', 'subnasale']); // index 14
      const pgPrimeLandmark = getLandmark(['Pog′', 'Pog\'', 'pog′', 'pog\'', 'Pog', 'pog', 'Pg\'', 'Pg', 'pg', 'Soft tissue Pogonion', 'soft tissue pogonion']); // index 15: Pog′
      const cmLandmark = getLandmark(['Cm', 'cm', 'CM', 'Columella', 'columella']); // index 22
      const lsLandmark = getLandmark(['UL', 'ul', 'UL′', 'UL\'', 'ul′', 'ul\'', 'Ls', 'ls', 'LS', 'Labiale Superius', 'labiale superius', 'Upper Lip']); // index 12: UL, index 29: UL′
      const liLandmark = getLandmark(['LL', 'll', 'LL′', 'LL\'', 'll′', 'll\'', 'Li', 'li', 'LI', 'Labiale Inferius', 'labiale inferius', 'Lower Lip']); // index 13: LL, index 30: LL′
      const mePrimeLandmark = getLandmark(['Me′', 'Me\'', 'me′', 'me\'', 'Me', 'me', 'Soft tissue Menton', 'soft tissue menton']); // index 32: Me′
      const prnLandmark = getLandmark(['Pn', 'pn', 'PN', 'Prn', 'prn', 'PRN', 'Pronasale', 'pronasale']); // index 25: Pn (mapped from Prn)
      const gnPrimeLandmark = getLandmark(['Gn′', 'Gn\'', 'gn′', 'gn\'', 'Gn', 'gn', 'GN\'', 'GN', 'Soft tissue Gnathion', 'soft tissue gnathion']); // index 31: Gn′
      
      // 1. Glabella-Sn-Pog' (Facial Convexity) - تحدب صورت
      if (gLandmark && snLandmark && pgPrimeLandmark) {
        const facialConvexity = calculateAngle(gLandmark, snLandmark, pgPrimeLandmark);
        measures['Glabella-Sn-Pog\' (Facial Convexity)'] = Math.round(facialConvexity * 10) / 10;
      }
      
      // 2. Sn-Gn' (Lower Face Height) - ارتفاع صورت پایین
      if (snLandmark && gnPrimeLandmark) {
        const lowerFaceHeight = calculateDistance(snLandmark, gnPrimeLandmark);
        if (lowerFaceHeight !== null) {
          measures['Sn-Gn\' (Lower Face Height)'] = Math.round(lowerFaceHeight * 10) / 10;
        }
      }
      
      // 3. Cm-Sn-UL (Upper Lip Protrusion) - برجستگی لب بالا
      if (cmLandmark && snLandmark && lsLandmark) {
        const upperLipProtrusion = calculateDistance(cmLandmark, lsLandmark);
        if (upperLipProtrusion !== null) {
          measures['Cm-Sn-UL (Upper Lip Protrusion)'] = Math.round(upperLipProtrusion * 10) / 10;
        }
      }
      
      // 7. Sn-Me' (Lower Face Height) - ارتفاع صورت پایین
      if (snLandmark && mePrimeLandmark) {
        const lowerFaceHeight2 = calculateDistance(snLandmark, mePrimeLandmark);
        if (lowerFaceHeight2 !== null) {
          measures['Sn-Me\' (Lower Face Height)'] = Math.round(lowerFaceHeight2 * 10) / 10;
        }
      }
      
      // 4. Glabella-Sn (Midface Length) - طول صورت میانی
      if (gLandmark && snLandmark) {
        const midfaceLength = calculateDistance(gLandmark, snLandmark);
        if (midfaceLength !== null) {
          measures['Glabella-Sn (Midface Length)'] = Math.round(midfaceLength * 10) / 10;
        }
      }
      
      // 5. Sn-Pog' (Lower Face Length) - طول صورت پایین
      if (snLandmark && pgPrimeLandmark) {
        const lowerFaceLength = calculateDistance(snLandmark, pgPrimeLandmark);
        if (lowerFaceLength !== null) {
          measures['Sn-Pog\' (Lower Face Length)'] = Math.round(lowerFaceLength * 10) / 10;
        }
      }
      
      // 6. Nasolabial Angle - زاویه نازولبیال
      if (cmLandmark && snLandmark && lsLandmark) {
        const nasolabialAngle = calculateAngle(cmLandmark, snLandmark, lsLandmark);
        measures['Nasolabial Angle'] = Math.round(nasolabialAngle * 10) / 10;
      }
      
      // 7. Z-Angle - زاویه بین Or-Po (Frankfort Horizontal) و Pog'-UL
      const orLandmarkZ = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkZ = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      if (orLandmarkZ && poLandmarkZ && pgPrimeLandmark && lsLandmark) {
        // محاسبه زاویه بین دو خط: Or-Po و Pog'-UL
        // برای Z-Angle باید زاویه کامل (0-180) را برگردانیم، نه فقط زاویه حاد
        const v1x = poLandmarkZ.x - orLandmarkZ.x;
        const v1y = poLandmarkZ.y - orLandmarkZ.y;
        const v2x = lsLandmark.x - pgPrimeLandmark.x;
        const v2y = lsLandmark.y - pgPrimeLandmark.y;
        
        const dotProduct = v1x * v2x + v1y * v2y;
        const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
        const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
        
        if (mag1 > 0 && mag2 > 0) {
          const cosAngle = dotProduct / (mag1 * mag2);
          const clampedCos = Math.max(-1, Math.min(1, cosAngle));
          let angleDeg = Math.acos(clampedCos) * (180 / Math.PI);
          
          // همیشه زاویه کوچک‌تر را برگردان (بین 0 تا 180)
          if (angleDeg > 180) {
            angleDeg = 360 - angleDeg;
          }
          if (angleDeg > 180) {
            angleDeg = 180 - (angleDeg - 180);
          }
          
          // 🔧 FIX: مقدار Z-Angle باید از 180 کم شود
          measures['Z-Angle'] = Math.round((180 - angleDeg) * 10) / 10;
        }
      }
      
      // 8. Total Soft-Tissue Convexity - زاویه Pog'-Prn-G
      const prnLandmarkTotal = getLandmark(['Pn', 'pn', 'PN', 'Prn', 'prn', 'PRN', 'Pronasale', 'pronasale']); // index 25: Pn (mapped from Prn)
      if (pgPrimeLandmark && prnLandmarkTotal && gLandmark) {
        const totalConvexity = calculateAngle(pgPrimeLandmark, prnLandmarkTotal, gLandmark);
        measures['Total Soft-Tissue Convexity'] = Math.round(totalConvexity * 10) / 10;
      }
      
      // 9. Lower Face Throat Angle - زاویه بین Sn–Gn' و Gn'–C
      const cLandmark = getLandmark(['C', 'c', 'Cervical', 'cervical', 'Cervical Point', 'cervical point', 'C point', 'c point']);
      if (snLandmark && gnPrimeLandmark && cLandmark) {
        // زاویه بین خط Sn–Gn' و خط Gn'–C
        const lowerFaceThroatAngle = calculateAngle(snLandmark, gnPrimeLandmark, cLandmark);
        measures['Lower Face Throat Angle'] = Math.round(lowerFaceThroatAngle * 10) / 10;
      }
      
      // 10. Facial Contour Angle - زاویه N′−Prn−Pog′
      const nPrimeLandmarkContour = getLandmark(['N′', 'N\'', 'n′', 'n\'', 'Nprime', 'nprime', 'Soft Nasion', 'soft nasion', 'N', 'n', 'Nasion', 'nasion']); // N' (Soft Nasion) - اگر موجود نباشد از N استفاده می‌کنیم
      if (nPrimeLandmarkContour && prnLandmark && pgPrimeLandmark) {
        const facialContourAngle = calculateAngle(nPrimeLandmarkContour, prnLandmark, pgPrimeLandmark);
        measures['Facial Contour Angle'] = Math.round(facialContourAngle * 10) / 10;
      }
      
      // 11. نسبت‌های عمودی مهم
      // محاسبه ارتفاع‌های مورد نیاز
      let totalFaceHeight = null; // G-Me'
      let lowerFaceHeight = null; // Sn-Me'
      let midfaceHeight = null; // G-Sn
      let lipHeight = null; // UL-LL
      
      // Total Face Height (G-Me')
      if (gLandmark && mePrimeLandmark) {
        totalFaceHeight = calculateDistance(gLandmark, mePrimeLandmark);
      }
      
      // Lower Face Height (Sn-Me')
      if (snLandmark && mePrimeLandmark) {
        lowerFaceHeight = calculateDistance(snLandmark, mePrimeLandmark);
      }
      
      // Midface Height (G-Sn)
      if (gLandmark && snLandmark) {
        midfaceHeight = calculateDistance(gLandmark, snLandmark);
      }
      
      // Lip Height (UL-LL) - فاصله عمودی بین لب بالا و پایین
      if (lsLandmark && liLandmark) {
        // فاصله عمودی (تفاوت y)
        const verticalDistance = Math.abs(lsLandmark.y - liLandmark.y);
        // تبدیل به میلی‌متر
        let conversionFactor = PIXEL_TO_MM_CONVERSION || 0.11;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        lipHeight = verticalDistance * conversionFactor;
      }
      
      // محاسبه نسبت‌ها
      // 10.1. Lower Face Height / Total Face Height
      if (lowerFaceHeight !== null && totalFaceHeight !== null && totalFaceHeight > 0) {
        const ratio = (lowerFaceHeight / totalFaceHeight) * 100;
        measures['Lower Face Height / Total Face Height'] = Math.round(ratio * 10) / 10;
      }
      
      // 10.2. Midface Height / Total Face Height
      if (midfaceHeight !== null && totalFaceHeight !== null && totalFaceHeight > 0) {
        const ratio = (midfaceHeight / totalFaceHeight) * 100;
        measures['Midface Height / Total Face Height'] = Math.round(ratio * 10) / 10;
      }
      
      // 10.3. Lower Face Height / Midface Height
      if (lowerFaceHeight !== null && midfaceHeight !== null && midfaceHeight > 0) {
        const ratio = lowerFaceHeight / midfaceHeight;
        measures['Lower Face Height / Midface Height'] = Math.round(ratio * 100) / 100;
      }
      
      // 10.4. Lip Height / Lower Face Height
      if (lipHeight !== null && lowerFaceHeight !== null && lowerFaceHeight > 0) {
        const ratio = lipHeight / lowerFaceHeight;
        measures['Lip Height / Lower Face Height'] = Math.round(ratio * 100) / 100;
      }
      
      // 10.5. Upper Lip Length / Lower Lip–Chin Length
      // Upper Lip Length = Sn-UL (فاصله Subnasale تا Upper Lip)
      // Lower Lip–Chin Length = LL-Me' (فاصله Lower Lip تا Soft tissue Menton)
      let upperLipLength = null;
      let lowerLipChinLength = null;
      
      // محاسبه Upper Lip Length (Sn-UL)
      if (snLandmark && lsLandmark) {
        upperLipLength = calculateDistance(snLandmark, lsLandmark);
      }
      
      // محاسبه Lower Lip–Chin Length (LL-Me')
      if (liLandmark && mePrimeLandmark) {
        lowerLipChinLength = calculateDistance(liLandmark, mePrimeLandmark);
      }
      
      // محاسبه نسبت
      if (upperLipLength !== null && lowerLipChinLength !== null && lowerLipChinLength > 0) {
        const ratio = upperLipLength / lowerLipChinLength;
        measures['Upper Lip Length / Lower Lip–Chin Length'] = Math.round(ratio * 100) / 100;
      }
      
      // ========== Arnett & McLaughlin Analysis Parameters (Soft Tissue) ==========
      
      // 1. Upper Lip to E-line - فاصله لب بالا تا E-line
      if (prnLandmark && pgPrimeLandmark && lsLandmark) {
        // E-line: خط Prn-Pog'
        // فاصله عمودی از Ls تا خط Prn-Pog' در میلی‌متر
        const eLineDistanceMm = calculateDistanceToLineMm(lsLandmark, prnLandmark, pgPrimeLandmark);
        if (eLineDistanceMm !== null) {
          // تعیین علامت: اگر Ls در سمت راست خط باشد (جلوتر)، مثبت است
          const sign = ((pgPrimeLandmark.x - prnLandmark.x) * (lsLandmark.y - prnLandmark.y) - 
                       (pgPrimeLandmark.y - prnLandmark.y) * (lsLandmark.x - prnLandmark.x)) > 0 ? 1 : -1;
          measures['Upper Lip to E-line'] = Math.round(eLineDistanceMm * sign * 10) / 10;
        }
      }
      
      // 2. Lower Lip to E-line - فاصله لب پایین تا E-line در میلی‌متر
      if (prnLandmark && pgPrimeLandmark && liLandmark) {
        const eLineDistanceMm = calculateDistanceToLineMm(liLandmark, prnLandmark, pgPrimeLandmark);
        if (eLineDistanceMm !== null) {
          const sign = ((pgPrimeLandmark.x - prnLandmark.x) * (liLandmark.y - prnLandmark.y) - 
                       (pgPrimeLandmark.y - prnLandmark.y) * (liLandmark.x - prnLandmark.x)) > 0 ? 1 : -1;
          measures['Lower Lip to E-line'] = Math.round(eLineDistanceMm * sign * 10) / 10;
        }
      }
      
      // 3. Nasolabial Angle - از Legan & Burstone استفاده می‌کنیم (قبلاً محاسبه شده)
      // (مقدار قبلاً در بخش Legan & Burstone محاسبه شده است)
      
      
      // 6. Chin Prominence - برجستگی چانه (فاصله Pog' تا خط N-Pog) در میلی‌متر
      // در آنالیز Arnett & McLaughlin: فاصله از Pog' تا خط N-Pog (hard tissue)
      const nLandmarkArnett = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const pogLandmarkArnett = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']);
      if (pgPrimeLandmark && nLandmarkArnett && pogLandmarkArnett) {
        // فاصله Pog' (soft tissue) تا خط N-Pog (hard tissue)
        const chinProminenceMm = calculateDistanceToLineMm(pgPrimeLandmark, nLandmarkArnett, pogLandmarkArnett);
        if (chinProminenceMm !== null) {
          // تعیین علامت: اگر Pog' در سمت راست خط N-Pog باشد (جلوتر)، مثبت است
          const sign = ((pogLandmarkArnett.x - nLandmarkArnett.x) * (pgPrimeLandmark.y - nLandmarkArnett.y) - 
                       (pogLandmarkArnett.y - nLandmarkArnett.y) * (pgPrimeLandmark.x - nLandmarkArnett.x)) > 0 ? 1 : -1;
          measures['Chin Prominence'] = Math.round(chinProminenceMm * sign * 10) / 10;
        }
      }
      
      // 7. Soft Facial Convexity - از Legan & Burstone استفاده می‌کنیم
      if (measures['Glabella-Sn-Pog\' (Facial Convexity)'] !== undefined) {
        measures['Soft Facial Convexity (Glabella-Sn-Pog\')'] = measures['Glabella-Sn-Pog\' (Facial Convexity)'];
      }
      
      // 7.5. Chin Prominence Angle - زاویه برجستگی چانه (G–Sn-Pog')
      // این همان زاویه Facial Convexity است
      if (measures['Glabella-Sn-Pog\' (Facial Convexity)'] !== undefined) {
        measures['Chin Prominence Angle'] = measures['Glabella-Sn-Pog\' (Facial Convexity)'];
      } else if (gLandmark && snLandmark && pgPrimeLandmark) {
        // اگر قبلاً محاسبه نشده، محاسبه کن
        const chinProminenceAngle = calculateAngle(gLandmark, snLandmark, pgPrimeLandmark);
        measures['Chin Prominence Angle'] = Math.round(chinProminenceAngle * 10) / 10;
      }
      
      // 8. Lower Face Height - از Legan & Burstone استفاده می‌کنیم
      if (measures['Sn-Me\' (Lower Face Height)'] !== undefined) {
        measures['Lower Face Height (Sn-Me\')'] = measures['Sn-Me\' (Lower Face Height)'];
      }
      
      // 9. Upper Lip Protrusion - از Legan & Burstone استفاده می‌کنیم
      if (measures['Cm-Sn-UL (Upper Lip Protrusion)'] !== undefined) {
        measures['Upper Lip Protrusion'] = measures['Cm-Sn-UL (Upper Lip Protrusion)'];
      }

      // ========== Holdaway Analysis Parameters (Soft Tissue) ==========
      
      // لندمارک‌های مورد نیاز برای Holdaway
      const nLandmarkHoldaway = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const nPrimeLandmarkHoldaway = getLandmark(['N′', 'N\'', 'n′', 'n\'', 'Nprime', 'nprime', 'Soft Nasion', 'soft nasion']); // N' (Soft Nasion)
      const pogLandmarkHoldaway = getLandmark(['Pog', 'pog', 'POG', 'pog', 'pogonion', 'Pogonion']); // Hard tissue Pogonion
      const ulPrimeLandmark = getLandmark(['UL′', 'UL\'', 'ul′', 'ul\'', 'UL\'', 'ul\'']); // index 29
      const llPrimeLandmark = getLandmark(['LL′', 'LL\'', 'll′', 'll\'', 'LL\'', 'll\'']); // index 30
      const orLandmarkHoldaway = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']);
      const poLandmarkHoldaway = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']);
      
      // 🔧 FIX: 1. H-angle - زاویه بین خط N-Pog و H-line (خط از Pog' تا UL)
      // H-line: خط از Pog' (soft tissue pogonion) تا UL (upper lip)
      const ulLandmarkHoldaway = getLandmark(['UL', 'ul', 'UL′', 'UL\'', 'ul′', 'ul\'', 'Ls', 'ls', 'LS', 'Labiale Superius', 'labiale superius', 'Upper Lip']);
      if (nLandmarkHoldaway && pogLandmarkHoldaway && pgPrimeLandmark && ulLandmarkHoldaway) {
        // H-line: خط از Pog' تا UL
        // محاسبه زاویه بین خط N-Pog (hard tissue) و H-line (Pog' تا UL)
        const hAngle = calculateAngleBetweenLines(
          nLandmarkHoldaway, pogLandmarkHoldaway,  // خط N-Pog
          pgPrimeLandmark, ulLandmarkHoldaway  // H-line: Pog' تا UL
        );
        measures['H-angle'] = Math.round(hAngle * 10) / 10;
      }
      
      // 🔧 FIX: 2. Upper Lip to H-line - فاصله لب بالا تا H-line (خط Pog' تا UL) در میلی‌متر
      // در Holdaway: Upper Lip to H-line معمولاً 0 است چون UL خودش روی H-line است
      // اما اگر UL' موجود باشد، فاصله UL' تا H-line محاسبه می‌شود
      if (pgPrimeLandmark && ulLandmarkHoldaway) {
        // H-line: خط از Pog' تا UL
        // اگر UL' موجود باشد، فاصله UL' تا H-line را محاسبه می‌کنیم
        if (ulPrimeLandmark) {
          const hLineDistanceMm = calculateDistanceToLineMm(ulPrimeLandmark, pgPrimeLandmark, ulLandmarkHoldaway);
          if (hLineDistanceMm !== null) {
            const sign = ((ulLandmarkHoldaway.x - pgPrimeLandmark.x) * (ulPrimeLandmark.y - pgPrimeLandmark.y) -
                         (ulLandmarkHoldaway.y - pgPrimeLandmark.y) * (ulPrimeLandmark.x - pgPrimeLandmark.x)) > 0 ? 1 : -1;
            measures['Upper Lip to H-line'] = Math.round(hLineDistanceMm * sign * 10) / 10;
          }
        } else {
          // اگر UL' موجود نباشد، مقدار 0 است (چون UL خودش روی H-line است)
          measures['Upper Lip to H-line'] = 0;
        }
      }
      
      // 🔧 FIX: 3. Lower Lip to H-line - فاصله لب پایین تا H-line (خط Pog' تا UL) در میلی‌متر
      const llLandmarkHoldaway = getLandmark(['LL', 'll', 'LL′', 'LL\'', 'll′', 'll\'', 'Li', 'li', 'LI', 'Labiale Inferius', 'labiale inferius', 'Lower Lip']);
      if (pgPrimeLandmark && ulLandmarkHoldaway && llLandmarkHoldaway) {
        // H-line: خط از Pog' تا UL
        // فاصله LL یا LL' تا H-line
        const lipPoint = llPrimeLandmark || llLandmarkHoldaway;
        const hLineDistanceMm = calculateDistanceToLineMm(lipPoint, pgPrimeLandmark, ulLandmarkHoldaway);
        if (hLineDistanceMm !== null) {
          const sign = ((ulLandmarkHoldaway.x - pgPrimeLandmark.x) * (lipPoint.y - pgPrimeLandmark.y) -
                       (ulLandmarkHoldaway.y - pgPrimeLandmark.y) * (lipPoint.x - pgPrimeLandmark.x)) > 0 ? 1 : -1;
          measures['Lower Lip to H-line'] = Math.round(hLineDistanceMm * sign * 10) / 10;
        }
      }
      
      // 4. Soft Tissue Facial Angle - زاویه بین خط N'-Pog' و Frankfort Horizontal
      if (nPrimeLandmarkHoldaway && pgPrimeLandmark && orLandmarkHoldaway && poLandmarkHoldaway) {
        const softTissueFacialAngle = calculateAngleBetweenLines(
          nPrimeLandmarkHoldaway, pgPrimeLandmark,  // خط N'-Pog' (soft tissue)
          orLandmarkHoldaway, poLandmarkHoldaway  // Frankfort Horizontal
        );
        measures['Soft Tissue Facial Angle'] = Math.round(softTissueFacialAngle * 10) / 10;
      }
      
      
      // 7. Soft Tissue Chin Thickness - فاصله بین Pog و Pog'
      if (pogLandmarkHoldaway && pgPrimeLandmark) {
        const chinThickness = calculateDistance(pogLandmarkHoldaway, pgPrimeLandmark);
        if (chinThickness !== null) {
          measures['Soft Tissue Chin Thickness'] = Math.round(chinThickness * 10) / 10;
        }
      }
      
      // ========== Soft Tissue Angular Analysis Parameters ==========
      
      // 1. Nasolabial Angle - از Legan & Burstone استفاده می‌کنیم (قبلاً محاسبه شده)
      // (مقدار قبلاً در بخش Legan & Burstone محاسبه شده است)
      
      // 2. Mentolabial Angle - زاویه منتولبیال (LL-Pog'-Me')
      if (liLandmark && pgPrimeLandmark && mePrimeLandmark) {
        const mentolabialAngle = calculateAngle(liLandmark, pgPrimeLandmark, mePrimeLandmark);
        measures['Mentolabial Angle'] = Math.round(mentolabialAngle * 10) / 10;
      }
      
      // 3. Soft Tissue Chin Angle - زاویه چانه بافت نرم (Pog'-Sn-N')
      // N' (Soft Nasion) - اگر موجود نباشد از N (hard tissue Nasion) استفاده می‌کنیم
      const nPrimeLandmark = getLandmark(['N′', 'N\'', 'n′', 'n\'', 'Nprime', 'nprime', 'Soft Nasion', 'soft nasion', 'N', 'n', 'Nasion', 'nasion']);
      if (pgPrimeLandmark && snLandmark && nPrimeLandmark) {
        const softTissueChinAngle = calculateAngle(pgPrimeLandmark, snLandmark, nPrimeLandmark);
        measures['Soft Tissue Chin Angle'] = Math.round(softTissueChinAngle * 10) / 10;
      }
      
      // 5. Upper Lip Angle - زاویه لب بالا (Cm-Sn-UL)
      if (cmLandmark && snLandmark && lsLandmark) {
        const upperLipAngle = calculateAngle(cmLandmark, snLandmark, lsLandmark);
        measures['Upper Lip Angle'] = Math.round(upperLipAngle * 10) / 10;
      }
      
      // 6. Lower Lip Angle - زاویه لب پایین (Sn-LL-Pog')
      if (snLandmark && liLandmark && pgPrimeLandmark) {
        const lowerLipAngle = calculateAngle(snLandmark, liLandmark, pgPrimeLandmark);
        measures['Lower Lip Angle'] = Math.round(lowerLipAngle * 10) / 10;
      }
      
      // 7. Total Facial Convexity - مجموع زوایای تحدب صورت
      if (gLandmark && snLandmark && pgPrimeLandmark) {
        // محاسبه مجموع زوایای داخلی مثلث Glabella-Sn-Pog'
        const angle1 = calculateAngle(gLandmark, snLandmark, pgPrimeLandmark);
        const angle2 = calculateAngle(snLandmark, pgPrimeLandmark, gLandmark);
        const angle3 = calculateAngle(pgPrimeLandmark, gLandmark, snLandmark);
        const totalConvexity = angle1 + angle2 + angle3;
        measures['Total Facial Convexity'] = Math.round(totalConvexity * 10) / 10;
      }

      // Overbite - فاصله عمودی بین U1 و L1 (تفاوت y) - محاسبه شده در بالا
      // Overjet - فاصله افقی بین U1 و L1 (تفاوت x) - محاسبه شده در بالا
      // (این مقادیر قبلاً در بخش Interincisal Angle محاسبه شده‌اند)
      
    } catch (err) {
      // Error handling if needed
    }
    
    return measures;
  }, [PIXEL_TO_MM_CONVERSION, calculateDistanceToLineMm]); // Depends on PIXEL_TO_MM_CONVERSION for accurate distance calculations

  // بهینه‌سازی: استفاده از useCallback برای onLandmarksChange
  // باید در سطح بالای کامپوننت باشد، نه داخل JSX
  const handleLandmarksChange = useCallback((updatedLandmarks) => {
    // Mark that landmarks have been manually changed
    hasManualLandmarkChangesRef.current = true;
    
    // Mark that there are unsaved changes
    setHasUnsavedChanges(true);
    
    // 🔧 FIX: محاسبه فوری measurements با landmarks جدید
    const newMeasurements = calculateMeasurements(updatedLandmarks);
    
    // Update result with new landmarks and measurements
    const updatedResult = {
      ...result,
      response: {
        ...result.response,
        landmarks: updatedLandmarks
      },
      measurements: newMeasurements
    };
    setResult(updatedResult);
    
    // 🔧 FIX: به روزرسانی calibrationPoints از landmarks تغییر یافته
    // اگر p1 یا p2 (نقاط کالیبراسیون با فاصله 1cm) تغییر کرده باشند، state را به روز کن
    const calibrationLandmarks = Object.keys(updatedLandmarks)
      .filter(key => key === 'p1' || key === 'p2')
      .map(key => updatedLandmarks[key]);
    
    if (calibrationLandmarks.length >= 2) {
      setCalibrationPoints(calibrationLandmarks);
      
      // محاسبه مجدد ضریب تبدیل با نقاط جدید
      const newConversionFactor = calculatePixelToMmConversion(calibrationLandmarks);
      setPIXEL_TO_MM_CONVERSION(newConversionFactor);
      
    }
    
    // 🔧 FIX: Notify parent immediately with updated measurements
    if (onLandmarksDetected && typeof onLandmarksDetected === 'function') {
      const isFromHistory = result?.metadata?.source === 'history';
      const isFromInitial = result?.metadata?.source === 'initial';
      const displayedOnly = isFromHistory || isFromInitial;
      
      onLandmarksDetected({
        landmarks: updatedLandmarks,
        measurements: newMeasurements,
        metadata: {
          ...result?.metadata,
          displayedOnly: displayedOnly,
          manuallyEdited: true,
        },
        model: selectedModel,
      });
    }
  }, [result, calculatePixelToMmConversion, calculateMeasurements, onLandmarksDetected, selectedModel]);


  // Handle saving changes to database
  const handleSaveChanges = useCallback(async () => {
    if (!result?.response?.landmarks || !onLandmarksDetected) {
      return;
    }

    setIsSavingChanges(true);
    try {
      // Calculate measurements from updated landmarks
      const measurements = calculateMeasurements(result.response.landmarks);
      

      // Call onLandmarksDetected callback (this will save to database)
      // This will trigger the save process in parent component
      await onLandmarksDetected({
        landmarks: result.response.landmarks,
        measurements,
        metadata: {
          ...result.metadata,
          savedAt: new Date().toISOString(),
          manuallyEdited: true,
        },
        model: selectedModel,
      });
      
      // Reset unsaved changes flag
      setHasUnsavedChanges(false);
      
    } catch (error) {
      // Error handling is done in parent component
    } finally {
      setIsSavingChanges(false);
    }
  }, [result, calculateMeasurements, onLandmarksDetected, selectedModel]);
  
  // 🔧 FIX: Separate useEffect for notifying about result changes
  // This runs when landmarks change (including manual edits)
  useEffect(() => {
    if (!result || !result.response?.landmarks) {
      return;
    }
    
    const updatedLandmarks = result.response.landmarks;
    if (!updatedLandmarks || Object.keys(updatedLandmarks).length === 0) {
      return;
    }
    
    // Create a unique ID based on landmarks content (not just result metadata)
    // This ensures recalculation when landmarks are manually edited
    const landmarksHash = JSON.stringify(updatedLandmarks);
    const resultId = result.metadata?.timestamp || result.metadata?.selectedAnalysisIndex || 'unknown';
    const uniqueId = `${resultId}-${landmarksHash.substring(0, 50)}`; // Use first 50 chars of landmarks hash
    
    // Check if we've already processed this exact landmarks state
    const shouldNotify = lastNotifiedResultRef.current !== uniqueId;
    
    if (shouldNotify) {
      lastNotifiedResultRef.current = uniqueId;
      
      requestAnimationFrame(() => {
        const measurements = calculateMeasurements(updatedLandmarks);
        
        if (onLandmarksDetected && typeof onLandmarksDetected === 'function') {
          // 🔧 FIX: If this result is from history, mark as displayedOnly to prevent re-saving
          // If result is from new analysis, set displayedOnly = false to auto-display and save
          const isFromHistory = result.metadata?.source === 'history';
          const isFromInitial = result.metadata?.source === 'initial';
          const displayedOnly = isFromHistory || isFromInitial; // Only display-only if from history or initial load
          
          onLandmarksDetected({
            landmarks: updatedLandmarks,
            measurements,
            metadata: {
              ...result.metadata,
              displayedOnly: displayedOnly, // 🔧 FIX: Auto-display and save new analysis results
            },
            model: selectedModel,
          });
        }
      });
    }
  }, [result, onLandmarksDetected, selectedModel, calculateMeasurements]);

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
    // 🔧 FIX: Don't clear result to prevent layout shift - keep previous result visible during analysis
    // setResult(null);

    const startTime = Date.now();

    try {
      const base64Image = await convertImageToBase64(imageFile);
      
      // Step 1: Detect P1/P2 first using checkpoint_p1_p2_fast_cpu_512_best.pth
      let p1P2Landmarks = {};
      const hasP1P2InInitial = initialLandmarks?.p1 && initialLandmarks?.p2;
      
      if (!hasP1P2InInitial) {
        try {
          const p1P2Endpoint = getAiServiceUrl('/detect-p1-p2');
          const p1P2Response = await fetch(p1P2Endpoint, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              image_base64: base64Image
            }),
          });

          if (p1P2Response.ok) {
            const p1P2Data = await p1P2Response.json();
            if (p1P2Data.success && p1P2Data.landmarks) {
              p1P2Landmarks = p1P2Data.landmarks;

              // Set calibration points for visualization
              if (p1P2Landmarks.p1 && p1P2Landmarks.p2) {
                setCalibrationPoints([
                  { x: p1P2Landmarks.p1.x, y: p1P2Landmarks.p1.y, label: 'P1' },
                  { x: p1P2Landmarks.p2.x, y: p1P2Landmarks.p2.y, label: 'P2' }
                ]);

                // Calculate pixel-to-mm conversion from p1/p2 distance
                const dx = p1P2Landmarks.p2.x - p1P2Landmarks.p1.x;
                const dy = p1P2Landmarks.p2.y - p1P2Landmarks.p1.y;
                const distancePixels = Math.sqrt(dx * dx + dy * dy);
                const conversionFactor = distancePixels > 0 ? 10.0 / distancePixels : 0.11; // 10mm distance
                setPIXEL_TO_MM_CONVERSION(conversionFactor);
              }
              // else block intentionally empty
            }
            // else block intentionally empty
          }
        } catch (p1P2Error) {
          // Error handling if needed
        }
      } else {
        p1P2Landmarks = {
          p1: initialLandmarks.p1,
          p2: initialLandmarks.p2
        };
        
        // Set calibration points from initialLandmarks
        setCalibrationPoints([
          { x: initialLandmarks.p1.x, y: initialLandmarks.p1.y, label: 'P1' },
          { x: initialLandmarks.p2.x, y: initialLandmarks.p2.y, label: 'P2' }
        ]);
        
        // Calculate conversion factor
        const dx = initialLandmarks.p2.x - initialLandmarks.p1.x;
        const dy = initialLandmarks.p2.y - initialLandmarks.p1.y;
        const distancePixels = Math.sqrt(dx * dx + dy * dy);
        const conversionFactor = distancePixels > 0 ? 10.0 / distancePixels : 0.11;
        setPIXEL_TO_MM_CONVERSION(conversionFactor);
      }
      
      // Step 2: Run main analysis
      
      let response; let data; let content; let parsedContent; let processingTime;

      let endpoint;
      if (isLocalModel && selectedModel.startsWith('local/aariz')) {
        endpoint = getAiServiceUrl('/detect');
        if (selectedModel === 'local/aariz-768') {
          endpoint = getAiServiceUrl('/detect-768');
        }
      } else if (isLocalModel && selectedModel === 'local/cldetection2023') {
        endpoint = getAiServiceUrl('/detect-cldetection2023');
      } else if (isLocalModel && selectedModel === 'local/fast-cpu-512') {
        endpoint = getAiServiceUrl('/detect-fast-cpu-512');
      } else if (isLocalModel && selectedModel === 'local/cldetection-optimized-512') {
        endpoint = getAiServiceUrl('/detect-cldetection-optimized-512');
      } else if (isLocalModel && selectedModel === 'local/cldetection-optimized-640') {
        endpoint = getAiServiceUrl('/detect-cldetection-optimized-640');
      } else if (isLocalModel && selectedModel === 'local/cldetection-optimized-1024') {
        endpoint = getAiServiceUrl('/detect-cldetection-optimized-1024');
      } else if (selectedModel === 'train_p1_p2_heatmap') {
        endpoint = getAiServiceUrl('/detect-p1p2-heatmap');
      }
      
      if (endpoint) {
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
          const modelName = selectedModelInfo?.name || 'Model';
          throw new Error(`${modelName} Service Error: ${response.status} - ${errorText}`);
        }

        data = await response.json();
        
        if (!data.success) {
          throw new Error(data.error || 'Detection failed');
        }

        // Check if p1/p2 are in the response
        const hasP1P2 = data.landmarks?.p1 && data.landmarks?.p2;
        
        // 🔧 DEBUG: Log p1/p2 structure for p1p2 model
        // if (selectedModel === 'train_p1_p2_heatmap') {
        //   // Debug logging if needed
        // }

        // 🔧 FIX: لندمارک‌های C و D را برای مدل cldetection نگه دار (حذف نشوند)
        let filteredLandmarks = { ...data.landmarks };
        
        // 🔧 FIX: Ensure p1/p2 are properly formatted for p1p2 model
        if (selectedModel === 'train_p1_p2_heatmap' && filteredLandmarks) {
          // Ensure p1 and p2 have x and y properties
          // Validation is handled by the backend
        }
        
        // 🔧 FIX: Scale landmarks based on backend resize algorithm (for original cldetection2023 only)
        if (selectedModel === 'local/cldetection2023') {
          // Backend resizes to max_dimension=512 for inference, then scales back
          // We need to apply the same scaling logic to ensure landmarks are in correct positions
          if (data.metadata?.image_size && data.metadata?.optimization) {
            const originalWidth = data.metadata.image_size.width;
            const originalHeight = data.metadata.image_size.height;
            const scaleFactor = data.metadata.optimization.scale_factor || 1.0;
            const wasResized = data.metadata.optimization.image_resized || false;
            
            
            // If image was resized for inference, landmarks are already scaled back to processed size
            // But we need to ensure they're correctly positioned relative to the original image
            // The backend already scales landmarks back to original image coordinates,
            // but we need to verify the scaling is correct
            
            // Model was trained with input_size=(1024, 1024) but backend uses max_dimension=512 for speed
            // The backend scales landmarks correctly, but we should verify the coordinates are within image bounds
            const scaledLandmarks = {};
            Object.keys(filteredLandmarks).forEach(key => {
              const landmark = filteredLandmarks[key];
              if (landmark && typeof landmark.x === 'number' && typeof landmark.y === 'number') {
                // Verify landmarks are within image bounds
                const x = Math.max(0, Math.min(originalWidth, landmark.x));
                const y = Math.max(0, Math.min(originalHeight, landmark.y));
                
                scaledLandmarks[key] = {
                  x,
                  y,
                };
                
                // Log if landmark was adjusted
                // Adjustment applied
              } else {
                scaledLandmarks[key] = landmark;
              }
            });
            
            filteredLandmarks = scaledLandmarks;
          }
        }

        // Merge P1/P2 from step 1 with main landmarks from step 2
        // Ensure P1/P2 are always included in the final landmarks
        const mergedLandmarks = {
          ...filteredLandmarks,
          ...p1P2Landmarks  // P1/P2 from step 1 will override if main model also detected them
        };
        
        
        parsedContent = {
          landmarks: mergedLandmarks,
          confidence: data.metadata?.avg_confidence || 0.9,
          notes: `${data.metadata?.model || 'Aariz Model'} - ${data.metadata?.model_architecture || 'trained'} - ${data.metadata?.valid_landmarks || 0}/29 لندمارک`,
          metadata: {
            ...data.metadata,
            frontend_image_size: imageSize,
            p1p2_detected: !!(p1P2Landmarks.p1 && p1P2Landmarks.p2),
          },
        };

        content = JSON.stringify(parsedContent);
      } else {
        // If no endpoint matched, set default parsedContent to prevent undefined error
        parsedContent = {
          landmarks: p1P2Landmarks, // At least include P1/P2 if detected
          confidence: 0.9,
          notes: 'No model selected',
          metadata: {
            frontend_image_size: imageSize,
            p1p2_detected: !!(p1P2Landmarks.p1 && p1P2Landmarks.p2),
          },
        };
        content = JSON.stringify(parsedContent);
      }

      // اعمال auto-scaling
      let scaledContent = parsedContent || { landmarks: p1P2Landmarks };
      let scalingInfo = null;
      
      // 🔧 FIX: Disable auto-scaling for p1p2 model - coordinates are already correct from backend
      // Auto-scaling is only needed for models that return normalized coordinates or incorrect scales
      const shouldApplyAutoScale = autoScale && imageSize && parsedContent && parsedContent.landmarks && selectedModel !== 'train_p1_p2_heatmap';
      
      if (shouldApplyAutoScale) {
        const scalingFactor = calculateScalingFactor(parsedContent.landmarks, imageSize);
        
        if (scalingFactor > 1.5 || scalingFactor < 0.5) {
          scaledContent = {
            ...parsedContent,
            landmarks: scaleLoadmarks(parsedContent.landmarks, scalingFactor),
            original_landmarks: parsedContent.landmarks,
          };
          
          scalingInfo = {
            factor: scalingFactor.toFixed(4),
            imageSize,
            applied: true,
          };
        }
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

      // 🔧 FIX: P1/P2 are already merged in parsedContent.landmarks from pipeline step 1
      // Just ensure they're properly set for visualization
      if (testResult.success && testResult.response?.landmarks) {
        const {landmarks} = testResult.response;
        
        // P1/P2 are already in landmarks from pipeline step 1
        // Just ensure calibration points are set for visualization
        if (landmarks.p1 && landmarks.p2 && 
            typeof landmarks.p1 === 'object' && typeof landmarks.p2 === 'object' &&
            typeof landmarks.p1.x !== 'undefined' && typeof landmarks.p1.y !== 'undefined' &&
            typeof landmarks.p2.x !== 'undefined' && typeof landmarks.p2.y !== 'undefined') {
          setCalibrationPoints([
            { x: landmarks.p1.x, y: landmarks.p1.y, label: 'P1' },
            { x: landmarks.p2.x, y: landmarks.p2.y, label: 'P2' }
          ]);
          
          // Calculate conversion factor if not already set
          if (PIXEL_TO_MM_CONVERSION === 0.11) {
            const dx = landmarks.p2.x - landmarks.p1.x;
            const dy = landmarks.p2.y - landmarks.p1.y;
            const distancePixels = Math.sqrt(dx * dx + dy * dy);
            const conversionFactor = distancePixels > 0 ? 10.0 / distancePixels : 0.11;
            setPIXEL_TO_MM_CONVERSION(conversionFactor);
          }
        }
        
        // All landmarks (including P1/P2) are already in the response
        // No need to merge or process further
        const landmarksWithCalibration = landmarks;
        
        // 🔧 FIX: به‌روزرسانی result state با landmarks + calibration points (فقط یک بار)
        // 🔧 FIX: Add unique timestamp to prevent duplicate notifications
        const analysisTimestamp = new Date().toISOString();
        const updatedResult = {
          ...testResult,
          response: {
            ...testResult.response,
            landmarks: landmarksWithCalibration
          },
          metadata: {
            ...testResult.metadata,
            timestamp: analysisTimestamp, // 🔧 FIX: Ensure unique timestamp for each analysis
            source: 'analysis', // 🔧 FIX: Mark as new analysis (not from history)
          }
        };
        
        // 🔧 FIX: علامت‌گذاری که یک آنالیز جدید انجام شده است
        hasNewAnalysisRef.current = true;
        
        // 🔧 FIX: Reset notification ref to allow notification for this new result
        lastNotifiedResultRef.current = null;
        
        // 🔧 FIX: بعد از 5 ثانیه، flag را reset کن تا اگر initialLandmarks بعداً تغییر کرد، بتواند لود شود
        setTimeout(() => {
          hasNewAnalysisRef.current = false;
        }, 5000);
        
        setResult(updatedResult);
        // Reset unsaved changes flag when new analysis result is received
        setHasUnsavedChanges(false);
        
        // 🔧 FIX: Don't notify here - let the useEffect (line 3684) handle notification
        // This prevents double notification and double saving
      } else {
        // اگر landmarks وجود نداشت، result را بدون p1/p2 set کن
        setResult(testResult);
      }
      
      // Auto-detect contours if enabled - موقتاً غیرفعال شده است
      // if (enableContours && isLocalModel && selectedModel.startsWith('local/aariz') && testResult.success) {
      //   try {
      //     await detectContours(base64Image, scaledContent.landmarks || parsedContent.landmarks);
      //   } catch (contourError) {
      //     // Ignore contour detection errors - they're not critical
      //   }
      // }
      
      // Save to database - موقتاً غیرفعال شده است
      // await saveTestToDb(testResult);

    } catch (err) {
      setError(err.message || 'خطا در تست مدل');
      
      const errorResult = {
        success: false,
        model: MODELS.find(m => m.id === selectedModel),
        error: err.message,
        metadata: {
          timestamp: new Date().toLocaleString('fa-IR'),
        },
      };
      
      // Save to database - موقتاً غیرفعال شده است
      // await saveTestToDb(errorResult);
    } finally {
      setIsLoading(false);
    }
  };

  const selectedModelInfo = MODELS.find(m => m.id === selectedModel);

  return (
    <Stack spacing={3}>
      {/* Main Content */}
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
        {/* Left Panel - Configuration */}
        <Stack spacing={3} sx={{ width: { xs: '100%', md: '400px' }, order: { xs: 2, md: 1 } }}>
          {/* Analysis History */}
          {analysisHistory && analysisHistory.length > 0 && (
            <Card>
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="h6">📋 تاریخچه آنالیز</Typography>
                  <Stack direction="row" spacing={1} alignItems="flex-start">
                    <FormControl fullWidth size="small">
                      <InputLabel>انتخاب آنالیز</InputLabel>
                      <Select
                        value={selectedAnalysisIndex !== null && selectedAnalysisIndex !== undefined ? selectedAnalysisIndex : (analysisHistory.length > 0 ? analysisHistory.length - 1 : '')}
                        onChange={(e) => {
                          const newValue = e.target.value;
                          const newIndex = newValue === '' ? null : parseInt(newValue);
                          if (onSelectedAnalysisIndexChange) {
                            onSelectedAnalysisIndexChange(newIndex);
                          }
                        }}
                        label="انتخاب آنالیز"
                      >
                      {analysisHistory.map((analysis, index) => {
                        const isSelected = selectedAnalysisIndex === index;
                        // 🔧 FIX: Get image index (default to 0 if not available for old analyses)
                        const imageIndex = analysis.imageIndex !== undefined ? analysis.imageIndex : 0;
                        const imageNumber = imageIndex + 1; // Display as 1-based (تصویر 1, تصویر 2, etc.)
                        
                        // 🔧 FIX: Get analysis type without "همه انواع"
                        const analysisType = analysis.analysisType || analysis.currentAnalysisType || 'steiner';
                        
                        return (
                          <MenuItem key={index} value={index}>
                            <Box sx={{ width: '100%' }}>
                              <Box sx={{ flex: 1 }}>
                                <Typography variant="body2">
                                  آنالیز {index + 1} - تصویر {imageNumber}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {analysis.timestamp
                                    ? new Date(analysis.timestamp).toLocaleDateString('fa-IR', {
                                        year: 'numeric',
                                        month: 'long',
                                        day: 'numeric',
                                        hour: '2-digit',
                                        minute: '2-digit',
                                      })
                                    : 'تاریخ نامشخص'}
                                </Typography>
                              </Box>
                            </Box>
                          </MenuItem>
                        );
                      })}
                    </Select>
                  </FormControl>
                  {selectedAnalysisIndex !== null && selectedAnalysisIndex !== undefined && selectedAnalysisIndex >= 0 && selectedAnalysisIndex < analysisHistory.length && (
                    <IconButton
                      color="error"
                      size="small"
                      onClick={() => {
                        const analysis = analysisHistory[selectedAnalysisIndex];
                        if (analysis && onAnalysisToDeleteChange && onDeleteDialogOpenChange) {
                          onAnalysisToDeleteChange(analysis);
                          onDeleteDialogOpenChange(true);
                        }
                      }}
                      sx={{ mt: 1.5 }}
                    >
                      <Iconify icon="solar:trash-bin-trash-bold" width={20} />
                    </IconButton>
                  )}
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          )}

          {/* Upload Image */}
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="h6">📷 آپلود تصویر</Typography>
                
                {/* Image Selector Dropdown - Only show if there are multiple images */}
                {lateralImages && lateralImages.length > 1 && onSelectedImageIndexChange && (
                  <FormControl fullWidth>
                    <InputLabel>انتخاب تصویر برای آنالیز</InputLabel>
                    <Select
                      value={selectedImageIndex}
                      label="انتخاب تصویر برای آنالیز"
                      onChange={(e) => {
                        const newIndex = parseInt(e.target.value);
                        onSelectedImageIndexChange(newIndex);
                        
                        // 🔧 FIX: When changing to a new image, clear analysis results
                        // This ensures the new image starts fresh for analysis
                        if (newIndex !== selectedImageIndex) {
                          setResult(null);
                          setError(null);
                          setIsEditMode(false);
                          setCalibrationPoints([]);
                          setPIXEL_TO_MM_CONVERSION(0.11);
                          hasManualLandmarkChangesRef.current = false;
                          hasNewAnalysisRef.current = false;
                          lastLoadedInitialLandmarksRef.current = null;
                        }
                      }}
                    >
                      {lateralImages.map((image, index) => {
                        const createdAt = image.createdAt ? new Date(image.createdAt) : new Date();
                        const formattedDate = createdAt.toLocaleDateString('fa-IR', {
                          year: 'numeric',
                          month: 'long',
                          day: 'numeric',
                          hour: '2-digit',
                          minute: '2-digit',
                        });
                        
                        // 🔧 FIX: Image numbering (1-based)
                        const imageNumber = lateralImages.length - index;
                        const isSelected = index === selectedImageIndex;
                        
                        return (
                          <MenuItem key={image.id || index} value={index}>
                            <Box sx={{ display: 'flex', flexDirection: 'column', width: '100%' }}>
                              <Typography variant="body2">
                                تصویر {imageNumber} {isSelected ? '✅' : ''} - {image.category || 'lateral'}
                              </Typography>
                              <Typography variant="caption" color="text.secondary">
                                تاریخ: {formattedDate}
                              </Typography>
                            </Box>
                          </MenuItem>
                        );
                      })}
                    </Select>
                  </FormControl>
                )}
                
                <Upload
				  multiple
                  onDrop={handleDrop}
                  onDelete={handleRemoveFile}
                  accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
                />
                
                {/* Uploaded Files List - Show both local files and server images */}
                {(imageFiles.length > 0 || (lateralImages && lateralImages.length > 0)) && (
                  <Box>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      فایل‌های آپلود شده ({imageFiles.length + (lateralImages?.length || 0)})
                    </Typography>
                    <Stack spacing={0}>
                      {/* Show local uploaded files (not yet on server) */}
                      {imageFiles.map((item, index) => {
                        // Truncate file name if longer than 20 characters
                        const fileName = item.name.length > 20
                          ? `${item.name.substring(0, 20)}...`
                          : item.name;
                        
                        return (
                          <Card
                            key={item.id}
                            sx={{
                              p: 1.5,
                              border: 1,
                              borderColor: 'divider',
                              bgcolor: 'background.paper',
                              marginTop: '0 !important',
                            }}
                          >
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Box
                                component="img"
                                src={item.preview}
                                alt={item.name}
                                sx={{
                                  width: 36,
                                  height: 36,
                                  objectFit: 'cover',
                                  borderRadius: 1,
                                }}
                              />
                              <Box sx={{ flex: 1, minWidth: 0 }}>
                                <Typography variant="body2" noWrap>
                                  {fileName}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {(item.size / 1024).toFixed(1)} KB
                                </Typography>
                              </Box>
                              <IconButton
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleRemoveFile(item);
                                }}
                                sx={{
                                  width: 26,
                                  height: 26,
                                  p: 0,
                                }}
                              >
                                <Iconify icon="mingcute:close-line" width={16} />
                              </IconButton>
                            </Stack>
                          </Card>
                        );
                      })}
                      
                      {/* Show server images */}
                      {lateralImages && lateralImages.map((item, index) => {
                        const imageUrl = item.path?.startsWith('http')
                          ? item.path
                          : item.path?.startsWith('/uploads/')
                          ? `${getImageUrl(item.path)}`
                          : `${getImageUrl(item.path)}`;
                        // Truncate file name if longer than 20 characters
                        const fileName = (item.originalName || `تصویر-${item.id}`).length > 20
                          ? `${(item.originalName || `تصویر-${item.id}`).substring(0, 20)}...`
                          : (item.originalName || `تصویر-${item.id}`);
                        
                        return (
                          <Card
                            key={item.id || index}
                            sx={{
                              p: 1.5,
                              border: 1,
                              borderColor: 'divider',
                              bgcolor: 'background.paper',
                              marginTop: '0 !important',
                            }}
                          >
                            <Stack direction="row" spacing={1} alignItems="center">
                              <Box
                                component="img"
                                src={imageUrl}
                                alt={fileName}
                                sx={{
                                  width: 36,
                                  height: 36,
                                  objectFit: 'cover',
                                  borderRadius: 1,
                                }}
                              />
                              <Box sx={{ flex: 1, minWidth: 0 }}>
                                <Typography variant="body2" noWrap>
                                  {fileName}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {item.size ? `${(item.size / 1024).toFixed(1)} KB` : 'حجم نامشخص'}
                                </Typography>
                              </Box>
                              {onDeleteImage && (
                                <IconButton
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    onDeleteImage(item);
                                  }}
                                  sx={{
                                    width: 26,
                                    height: 26,
                                    p: 0,
                                  }}
                                >
                                  <Iconify icon="mingcute:close-line" width={16} />
                                </IconButton>
                              )}
                            </Stack>
                          </Card>
                        );
                      })}
                    </Stack>
                  </Box>
                )}
              </Stack>
			  
            </CardContent>
          </Card>


          {/* Advanced Settings Card */}
          {showAdvancedSettings && (
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
                            {model.provider && (
                              <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                                {model.provider}
                              </Typography>
                            )}
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




              </Stack>
            </CardContent>
          </Card>
          )}

          {/* Test Button and Settings */}
          <Stack direction="row" spacing={1} alignItems="center">
            <Button
              fullWidth
              size="medium"
              variant="contained"
              color="primary"
              onClick={handleTest}
              disabled={isLoading || !imageFile || (!selectedModelInfo?.isLocal && !apiKey)}
              startIcon={isLoading ? <CircularProgress size={16} sx={{ color: 'inherit' }} /> : null}
              sx={{ letterSpacing: 'normal' }}
            >
              {isLoading ? 'در حال پردازش' : 'آنالیز'}
            </Button>
            <IconButton
              onClick={() => setShowAdvancedSettings(!showAdvancedSettings)}
              sx={{
                borderRadius: '50%',
                width: 40,
                height: 40,
                backgroundColor: 'transparent',
                color: 'inherit',
                flexShrink: 0,
                '&:hover': {
                  backgroundColor: 'rgba(0, 0, 0, 0.04)',
                }
              }}
            >
              <SettingsIcon />
            </IconButton>
          </Stack>
        </Stack>

        {/* Delete Confirmation Dialog */}
        <Dialog 
          open={deleteDialogOpen} 
          onClose={() => {
            if (!deleting && onDeleteDialogOpenChange) {
              onDeleteDialogOpenChange(false);
            }
          }}
        >
          <DialogTitle>حذف آنالیز سفالومتری</DialogTitle>
          <DialogContent>
            <Typography 
              variant="body2"
              sx={{ color: 'var(--palette-text-secondary)' }}
            >
              آیا از حذف این آنالیز مطمئن هستید؟ این عمل غیرقابل بازگشت است.
            </Typography>
          </DialogContent>
          <DialogActions>
            <Button 
              onClick={() => {
                if (onDeleteDialogOpenChange) {
                  onDeleteDialogOpenChange(false);
                }
              }} 
              color="inherit" 
              disabled={deleting}
            >
              انصراف
            </Button>
            <Button
              onClick={() => {
                if (onDeleteAnalysis && analysisToDelete) {
                  onDeleteAnalysis(analysisToDelete);
                }
              }}
              color="error"
              variant="contained"
              disabled={deleting}
            >
              {deleting ? 'در حال حذف...' : 'حذف'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Right Panel - Results */}
        <Stack spacing={3} sx={{ flex: 1, order: { xs: 1, md: 2 } }}>
          {/* Error */}
          {error && (
            <Alert severity="error" onClose={() => setError(null)}>
              <Typography variant="subtitle2">خطا</Typography>
              <Typography variant="body2">{error}</Typography>
            </Alert>
          )}

          {/* Visualization - نمایش تصویر و لندمارک‌ها */}
          {(imagePreview || lateralImageUrl) && (
            <>
              <Card sx={{ overflow: 'hidden' }}>
              <CardContent sx={{ '&:last-child': { pb: 4 }, overflow: 'hidden' }}>
                <Stack spacing={2} sx={{ overflow: 'hidden' }}>
                  {imagePreview && result && result.success && result.response?.landmarks && (
                    <>
                      {/* Header and controls - outside fixed container */}
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">📊 لندمارک ها</Typography>
                        <Stack direction="row" spacing={1} alignItems="center">
                            {/* Calibration Status - Hidden from user display */}
                        </Stack>
                      </Stack>
                      
                      {/* View Mode Toggle Buttons and Download - between title and canvas */}
                      <Stack direction="row" spacing={0.5} justifyContent="center" alignItems="center" sx={{ py: 1 }}>
                        <ToggleButtonGroup
                          value={viewMode}
                          exclusive
                          onChange={(event, newViewMode) => {
                            if (newViewMode !== null) {
                              handleViewModeChange(newViewMode);
                            }
                          }}
                          size="small"
                          color="standard"
                        >
                          <ToggleButton value="normal" aria-label="عادی">
                            <Box
                              sx={{ 
                                width: 20, 
                                height: 20,
                                mask: 'url(/assets/icons/images.svg) no-repeat center',
                                maskSize: 'contain',
                                backgroundColor: 'text.secondary',
                                WebkitMask: 'url(/assets/icons/images.svg) no-repeat center',
                                WebkitMaskSize: 'contain',
                              }}
                            />
                          </ToggleButton>
                          <ToggleButton value="coordinate" aria-label="محور مختصات">
                            <Box
                              sx={{ 
                                width: 20, 
                                height: 20,
                                mask: 'url(/assets/icons/coordinate.svg) no-repeat center',
                                maskSize: 'contain',
                                backgroundColor: 'text.secondary',
                                WebkitMask: 'url(/assets/icons/coordinate.svg) no-repeat center',
                                WebkitMaskSize: 'contain',
                              }}
                            />
                          </ToggleButton>
                          <ToggleButton
                          value="edit"
                          selected={isEditMode}
                          onClick={() => setIsEditMode(!isEditMode)}
                          aria-label="ویرایش"
                          size="small"
                        >
                          <Iconify 
                            icon="solar:pen-bold"
                            width={20}
                            sx={{ 
                              color: 'var(--palette-text-secondary)',
                            }}
                          />
                        </ToggleButton>
						                        {/* Download Button */}
                        <Tooltip title="دانلود تصویر">
                          <IconButton 
                            size="small" 
                            onClick={() => {
                              // Trigger download from visualizer
                              const visualizerCanvas = document.querySelector('#cephalometric-canvas');
                              if (visualizerCanvas) {
                                const canvas = visualizerCanvas;
                                const tempCanvas = document.createElement('canvas');
                                tempCanvas.width = canvas.width;
                                tempCanvas.height = canvas.height;
                                const tempCtx = tempCanvas.getContext('2d');
                                tempCtx.drawImage(canvas, 0, 0);
                                tempCanvas.toBlob((blob) => {
                                  if (!blob) return;
                                  const url = URL.createObjectURL(blob);
                                  const link = document.createElement('a');
                                  link.href = url;
                                  link.download = `cephalometric-analysis-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.png`;
                                  document.body.appendChild(link);
                                  link.click();
                                  document.body.removeChild(link);
                                  URL.revokeObjectURL(url);
                                }, 'image/png');
                              }
                            }}
                            sx={{ ml: 0.5 }}
                          >
                            <Iconify icon="solar:download-linear" width={20} />
                          </IconButton>
                        </Tooltip>
                        </ToggleButtonGroup>

                      </Stack>
                    </>
                  )}
                  
                  {/* Container with fixed dimensions to prevent layout shift */}
                  <Box sx={{ 
                    width: '100%',
                    maxWidth: '100%', // 🔧 FIX: اطمینان از اینکه عرض بیش از 100% نشود
                    minWidth: 0, // 🔧 FIX: اجازه می‌دهد که container کوچک شود
                    minHeight: 400,
                    position: 'relative',
                    borderRadius: { xs: 2, sm: 2 },
                    overflow: 'hidden',
                    boxSizing: 'border-box', // 🔧 FIX: اطمینان از اینکه padding در عرض محاسبه شود
                  }}>
                    {!imagePreview && lateralImageUrl ? (
                      <>
                        {/* Skeleton placeholder while image is loading */}
                        <Skeleton 
                          variant="rectangular" 
                          height={400}
                          sx={{ 
                            width: '100%',
                            borderRadius: { xs: 2, sm: 2 } 
                          }}
                        />
                        <Box
                          sx={{
                            position: 'absolute',
                            top: '50%',
                            left: '50%',
                            transform: 'translate(-50%, -50%)',
                            display: 'flex',
                            flexDirection: 'column',
                            alignItems: 'center',
                            gap: 2,
                          }}
                        >
                          <CircularProgress size={40} />
                          <Typography variant="body2" color="text.secondary">
                            در حال بارگذاری تصویر...
                          </Typography>
                        </Box>
                      </>
                    ) : imagePreview && result && result.success && result.response?.landmarks ? (
                      <Box sx={{ 
                        overflow: 'hidden', 
                        borderRadius: { xs: 2, sm: 2 },
                        minHeight: 400, // Reserve space to prevent layout shift
                        width: '100%',
                        maxWidth: '100%', // 🔧 FIX: اطمینان از اینکه عرض بیش از 100% نشود
                        minWidth: 0, // 🔧 FIX: اجازه می‌دهد که container کوچک شود
                        position: 'relative',
                        boxSizing: 'border-box', // 🔧 FIX: اطمینان از اینکه padding در عرض محاسبه شود
                      }}>
                        {/* Analysis Type Dropdown */}
                        {result && result.success && result.response?.landmarks && (
                          <Box sx={{ mt: 1.25, mb: 2 }}>
                            <FormControl fullWidth size="small">
                              <InputLabel>نوع آنالیز</InputLabel>
                              <Select
                                value={displayAnalysisType}
                                onChange={(e) => setDisplayAnalysisType(e.target.value)}
                                label="نوع آنالیز"
                              >
                                <MenuItem value="general">عمومی</MenuItem>
                                <MenuItem value="steiner">Steiner</MenuItem>
                                <MenuItem value="ricketts">Ricketts</MenuItem>
                                <MenuItem value="mcnamara">McNamara</MenuItem>
                                <MenuItem value="wits">Wits</MenuItem>
                                <MenuItem value="tweed">Tweed</MenuItem>
                                <MenuItem value="jarabak">Jarabak</MenuItem>
                                <MenuItem value="sassouni">Sassouni</MenuItem>
                                <MenuItem value="leganBurstone">Legan & Burstone</MenuItem>
                                <MenuItem value="arnettMcLaughlin">Arnett & McLaughlin</MenuItem>
                                <MenuItem value="holdaway">Holdaway</MenuItem>
                                <MenuItem value="softTissueAngular">Soft Tissue Angular</MenuItem>
                                <MenuItem value="all">همه انواع</MenuItem>
                              </Select>
                            </FormControl>
                          </Box>
                        )}
                        <AdvancedCephalometricVisualizer
                          imageUrl={imagePreview}
                          landmarks={result.response.landmarks}
                          imageSize={imageSize}
                          contours={contours?.contours || null}
                          calibrationPoints={calibrationPoints}
                          readOnly={!isEditMode} // اگر در حالت ویرایش نیست، readOnly باشد
                          onLandmarksChange={handleLandmarksChange}
                          showMeasurements
                          showCoordinateSystem={showCoordinateSystem}
                          viewMode={viewMode}
                          analysisType={displayAnalysisType}
                          measurements={result.measurements || result.response?.measurements || {}}
                          pixelToMmConversion={PIXEL_TO_MM_CONVERSION}
                        />
                        {/* Loading overlay when analyzing */}
                        {/* Save Changes and Download PDF Buttons */}
                        {(isEditMode && hasUnsavedChanges) || (result && result.success) ? (
                          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 2, mb: 1.25, width: '100%' }}>
                            {isEditMode && hasUnsavedChanges && (
                              <Button
                                variant="contained"
                                color="primary"
                                fullWidth
                                onClick={handleSaveChanges}
                                disabled={isSavingChanges || saving}
                                startIcon={isSavingChanges || saving ? (
                                  <CircularProgress size={16} sx={{ color: 'inherit' }} />
                                ) : (
                                  <Iconify icon="carbon:save" width={20} />
                                )}
                                sx={{
                                  letterSpacing: 'normal',
                                }}
                              >
                                {isSavingChanges || saving ? 'در حال ذخیره...' : 'ذخیره تغییرات'}
                              </Button>
                            )}
                          </Stack>
                        ) : null}
                      </Box>
                    ) : imagePreview ? (
                      <>
                        <Box
                          component="img"
                          src={imagePreview}
                          alt="Lateral Cephalometric Image"
                          sx={{
                            width: '100%',
                            height: 'auto',
                            maxHeight: 600,
                            minHeight: 400, // Reserve space to prevent layout shift
                            aspectRatio: 'auto', // Preserve image aspect ratio
                            objectFit: 'contain',
                            borderRadius: { xs: 2, sm: 2 }, // border radius در موبایل و دسکتاپ
                            border: '1px solid',
                            borderColor: 'divider',
                            display: 'block', // Prevent inline spacing issues
                          }}
                        />
                        {/* Save Changes Button */}
                            {isEditMode && hasUnsavedChanges && (
                          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mt: 2, mb: 1.25, width: '100%' }}>
                              <Button
                                variant="contained"
                                color="primary"
                                fullWidth
                                onClick={handleSaveChanges}
                                disabled={isSavingChanges || saving}
                                startIcon={isSavingChanges || saving ? (
                                  <CircularProgress size={16} sx={{ color: 'inherit' }} />
                                ) : (
                                  <Iconify icon="carbon:save" width={20} />
                                )}
                                sx={{
                                  letterSpacing: 'normal',
                                }}
                              >
                                {isSavingChanges || saving ? 'در حال ذخیره...' : 'ذخیره تغییرات'}
                              </Button>
                          </Stack>
                        )}
                      </>
                    ) : null}
                  </Box>
                </Stack>
              </CardContent>
            </Card>

              {/* Show Results and Save buttons below image - outside the card */}
              {/* Buttons removed - functionality moved elsewhere */}
            </>
          )}

          {/* Results */}
        </Stack>
      </Stack>

    </Stack>
  );
}

// Memoize component to prevent unnecessary re-renders when props haven't changed
export const MemoizedCephalometricAIAnalysis = memo(CephalometricAIAnalysis, (prevProps, nextProps) => {
  // Custom comparison function for better performance
  // Only re-render if critical props have changed
  const criticalProps = [
    'lateralImageUrl',
    'selectedImageIndex',
    'selectedAnalysisIndex',
    'initialLandmarks',
    'viewMode',
    'showCoordinateSystem',
    'isUploadingImage',
    'deleteDialogOpen',
    'analysisToDelete',
    'deleting',
  ];
  
  // Check if any critical prop has changed
  for (const prop of criticalProps) {
    if (prevProps[prop] !== nextProps[prop]) {
      // Deep comparison for objects
      if (typeof prevProps[prop] === 'object' && prevProps[prop] !== null) {
        if (JSON.stringify(prevProps[prop]) !== JSON.stringify(nextProps[prop])) {
          return false; // Props changed, re-render
        }
      } else {
        return false; // Props changed, re-render
      }
    }
  }
  
  // Props haven't changed significantly, skip re-render
  return true;
});

