import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Menu from '@mui/material/Menu';
import Stack from '@mui/material/Stack';
import Slider from '@mui/material/Slider';
import Divider from '@mui/material/Divider';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';

import { ColorPicker } from 'src/components/color-utils';
import { Iconify, usePreloadIcons } from 'src/components/iconify';
import { extractBoundary, smartObjectSelectionConstrained } from 'src/components/object-selector';

// OPG Detection Classes - باید با backend هماهنگ باشد (نام‌های انگلیسی)
const OPG_CLASSES_EN = {
  0: 'Caries',
  1: 'Crown',
  2: 'Filling',
  3: 'Implant',
  4: 'Malaligned',
  5: 'Mandibular Canal',
  6: 'Missing teeth',
  7: 'Periapical lesion',
  8: 'Retained root',
  9: 'Root Canal Treatment',
  10: 'Root Piece',
  11: 'impacted tooth',
  12: 'maxillary sinus',
  13: 'Bone Loss',
  14: 'Fracture teeth',
  15: 'Permanent Teeth',
  16: 'Supra Eruption',
  17: 'TAD',
  18: 'abutment',
  19: 'attrition',
  20: 'bone defect',
  21: 'gingival former',
  22: 'metal band',
  23: 'orthodontic brackets',
  24: 'permanent retainer',
  25: 'post - core',
  26: 'plating',
  27: 'wire',
  28: 'Cyst',
  29: 'Root resorption',
  30: 'Primary teeth',
};

// ترجمه فارسی کلاس‌ها
const OPG_CLASSES_FA = {
  0: 'پوسیدگی',
  1: 'تاج',
  2: 'پرکردگی',
  3: 'ایمپلنت',
  4: 'نامرتب',
  5: 'کانال مندیبولار',
  6: 'دندان از دست رفته',
  7: 'ضایعه',
  8: 'ریشه باقیمانده',
  9: 'درمان ریشه',
  10: 'تکه ریشه',
  11: 'دندان نهفته',
  12: 'سینوس ماگزیلا',
  13: 'تحلیل استخوان',
  14: 'شکستگی دندان',
  15: 'دندان دائمی',
  16: 'بیرون‌زدگی',
  17: 'TAD',
  18: 'اباتمنت',
  19: 'سایش',
  20: 'نقص استخوان',
  21: 'فرمر لثه',
  22: 'باند فلزی',
  23: 'براکت ارتودنسی',
  24: 'ریتینر دائمی',
  25: 'پست و کور',
  26: 'پلیت',
  27: 'سیم',
  28: 'کیست',
  29: 'جذب ریشه',
  30: 'دندان شیری',
};

// Helper function to get Persian class name
const getClassNameFA = (classId, classNameEN) => {
  // اگر classNameEN ارائه شده و در دیکشنری فارسی هست، از آن استفاده کن
  if (classNameEN && OPG_CLASSES_FA[Object.keys(OPG_CLASSES_EN).find(key => OPG_CLASSES_EN[key] === classNameEN)]) {
    const key = Object.keys(OPG_CLASSES_EN).find(k => OPG_CLASSES_EN[k] === classNameEN);
    return OPG_CLASSES_FA[key] || classNameEN;
  }
  // در غیر این صورت از classId استفاده کن
  return OPG_CLASSES_FA[classId] || OPG_CLASSES_EN[classId] || 'نامشخص';
};

// رنگ‌های سریع برای انتخاب - مدرن و جذاب
const QUICK_COLORS = [
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
];

// رنگ‌ها برای هر کلاس - استفاده از رنگ‌های جذاب مدرن
const getClassColor = (classId, customColor = null) => {
  if (customColor) {
    // اگر رنگ سفارشی وجود دارد، از آن استفاده کن
    return {
      main: customColor,
      light: `${customColor}33`, // alpha 0.2
      dark: customColor,
    };
  }

  // استفاده از رنگ‌های مدرن و جذاب برای هر کلاس
  const modernColors = [
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

  // انتخاب رنگ بر اساس classId
  const colorIndex = classId % modernColors.length;
  const mainColor = modernColors[colorIndex];

  return {
    main: mainColor,
    light: `${mainColor}40`, // alpha 0.25 برای بهتر دیده شدن
    dark: mainColor, // همان رنگ اصلی برای dark
  };
};

// ----------------------------------------------------------------------

export function OPGVisualizer({
  imageUrl,
  detections: initialDetections,
  imageSize,
  onDetectionsChange,
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  // No tabs needed - only detection view with auto-processing

  // State
  const [detections, setDetections] = useState([]);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  
  // Interaction
  const [selectedDetection, setSelectedDetection] = useState(null);
  const [hoveredDetection, setHoveredDetection] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isResizing, setIsResizing] = useState(false);
  const [resizeHandle, setResizeHandle] = useState(null);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [dragStartImage, setDragStartImage] = useState({ x: 0, y: 0 }); // مختصات اولیه در تصویر (برای drag)
  const [dragStartMouseImage, setDragStartMouseImage] = useState({ x: 0, y: 0 }); // موقعیت mouse در تصویر در زمان کلیک
  const [resizeStartBbox, setResizeStartBbox] = useState({ x: 0, y: 0, width: 0, height: 0 }); // bbox اولیه برای resize
  
  // Refs for temporary drag tracking (for smooth dragging without state updates)
  const tempDragPositionRef = useRef(null);
  const rafIdRef = useRef(null);
  
  // Visual settings
  const [showLabels, setShowLabels] = useState(true);
  const [boxThickness, setBoxThickness] = useState(1);
  const [labelFontSize, setLabelFontSize] = useState(14);
  const [displayMode, setDisplayMode] = useState('box'); // 'box' or 'point'
  
  // Preload display mode icons to prevent layout shift
  usePreloadIcons(['carbon:checkbox', 'carbon:circle-dash']);

  // Object selection settings
  const [tolerance, setTolerance] = useState(20);
  
  // Context menu
  const [contextMenu, setContextMenu] = useState(null);
  
  // Custom colors for detections
  const [customColors, setCustomColors] = useState({}); // { detectionIndex: color }

  // Store original detections
  const [originalDetections, setOriginalDetections] = useState([]);

  // Track if auto-processing has been done
  const hasProcessedRef = useRef(false);

  // Auto-process objects in detected regions
  const autoProcessObjectsInDetections = useCallback(async (detectionsList) => {
    if (!imageRef.current || !isImageLoaded) return;

    console.log('Auto-processing objects in detections...');

    // Create temporary canvas to get image data
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = imageRef.current.width;
    tempCanvas.height = imageRef.current.height;
    tempCtx.drawImage(imageRef.current, 0, 0);

    const imageData = tempCtx.getImageData(0, 0, imageRef.current.width, imageRef.current.height);

    const processedDetections = [];

    for (let i = 0; i < detectionsList.length; i++) {
      const det = detectionsList[i];
      const {bbox} = det;
      if (!bbox) {
        processedDetections.push(det);
        continue;
      }

      const bboxX = bbox.x || bbox.x1 || 0;
      const bboxY = bbox.y || bbox.y1 || 0;
      const bboxWidth = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const bboxHeight = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;

      // Find center point of bounding box as starting point
      const centerX = Math.round(bboxX + bboxWidth / 2);
      const centerY = Math.round(bboxY + bboxHeight / 2);

      // Constrain search within bounding box
      const constrainedBbox = {
        x: bboxX,
        y: bboxY,
        width: bboxWidth,
        height: bboxHeight
      };

      try {
        // Use the constrained object selection algorithm
        const selectedPixels = smartObjectSelectionConstrained(
          imageData,
          centerX,
          centerY,
          imageRef.current.width,
          imageRef.current.height,
          tolerance, // tolerance
          constrainedBbox
        );

        // Extract boundary pixels
        const boundaryPixels = extractBoundary(selectedPixels, imageRef.current.width, imageRef.current.height);

        // Calculate refined bounding box from selected pixels
        if (selectedPixels.size > 0) {
          // Use reduce instead of spread operator to avoid stack overflow with large arrays
          let minX = Infinity;
          let minY = Infinity;
          let maxX = -Infinity;
          let maxY = -Infinity;
          
          // Limit processing to avoid stack overflow (max 100000 pixels)
          const maxPixels = 100000;
          let processedCount = 0;
          
          for (const key of selectedPixels) {
            if (processedCount++ >= maxPixels) break;
            const [x, y] = key.split(',').map(Number);
            if (x < minX) minX = x;
            if (y < minY) minY = y;
            if (x > maxX) maxX = x;
            if (y > maxY) maxY = y;
          }

          // Create refined detection with object boundaries
          // Limit boundary pixels to avoid memory issues (max 50000 pixels for rendering)
          const maxBoundaryPixels = 50000;
          const boundaryPixelsArray = [];
          let boundaryCount = 0;
          
          for (const key of boundaryPixels) {
            if (boundaryCount++ >= maxBoundaryPixels) break;
            const [x, y] = key.split(',').map(Number);
            boundaryPixelsArray.push({ x, y });
          }

          const refinedDetection = {
            ...det,
            bbox: {
              x: minX,
              y: minY,
              width: maxX - minX,
              height: maxY - minY,
              x1: minX,
              y1: minY,
              x2: maxX,
              y2: maxY,
            },
            // Store boundary pixels for outline rendering (limited to avoid memory issues)
            boundaryPixels: boundaryPixelsArray,
            // Mark as auto-processed
            autoProcessed: true,
          };

          processedDetections.push(refinedDetection);
        } else {
          // If no object found, keep original detection
          processedDetections.push(det);
        }
      } catch (error) {
        console.error(`Error processing detection ${i}:`, error);
        processedDetections.push(det);
      }
    }

    console.log('Auto-processing complete, processed detections:', processedDetections);
    setDetections(processedDetections);

    if (onDetectionsChange) {
      onDetectionsChange(processedDetections);
    }

    // Clear original detections since we now have processed ones
    setOriginalDetections([]);
  }, [isImageLoaded, onDetectionsChange, tolerance]);

  // Update detections when prop changes
  useEffect(() => {
    if (initialDetections && initialDetections.length > 0) {
      setOriginalDetections([...initialDetections]); // Store a copy of original detections
    } else if (!initialDetections || initialDetections.length === 0) {
      // Reset processing flag when no detections
      hasProcessedRef.current = false;
      setOriginalDetections([]);
    }
  }, [initialDetections]);

  // Auto-trigger object selection 3 seconds after AI detection
  useEffect(() => {
    if (originalDetections.length > 0 && isImageLoaded && !hasProcessedRef.current) {
      console.log('Auto-triggering object selection in 3 seconds...');
      const timer = setTimeout(() => {
        if (originalDetections.length > 0 && isImageLoaded) {
          console.log('Auto object selection triggered after 3 seconds...');
          autoProcessObjectsInDetections(originalDetections);
          hasProcessedRef.current = true;
        }
      }, 3000); // 3 seconds delay

      return () => clearTimeout(timer); // Cleanup timer on unmount
    }
  }, [originalDetections, isImageLoaded, autoProcessObjectsInDetections]);

  // Process detections when both image and detections are ready
  useEffect(() => {
    if (isImageLoaded && originalDetections.length > 0 && !hasProcessedRef.current) {
      hasProcessedRef.current = true;
      console.log('Processing detections after both image and detections are ready...');
      setTimeout(() => {
        autoProcessObjectsInDetections(originalDetections);
      }, 5000); // 5 second delay after AI detection
    }
  }, [isImageLoaded, originalDetections, autoProcessObjectsInDetections]);

  // Convert canvas coordinates to image coordinates
  const getImageCoordinates = useCallback((canvasX, canvasY) => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return null;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return null;

    const scale = Math.min(rect.width / image.width, rect.height / image.height) * zoom;
    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;
    const offsetX = (rect.width - scaledWidth) / 2 + pan.x;
    const offsetY = (rect.height - scaledHeight) / 2 + pan.y;

    const imageX = (canvasX - offsetX) / scale;
    const imageY = (canvasY - offsetY) / scale;

    if (imageX < 0 || imageX >= image.width || imageY < 0 || imageY >= image.height) {
      return null;
    }

    return { x: imageX, y: imageY };
  }, [zoom, pan]);

  // Convert image coordinates to canvas coordinates
  const getCanvasCoordinates = useCallback((imageX, imageY) => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return null;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return null;

    const scale = Math.min(rect.width / image.width, rect.height / image.height) * zoom;
    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;
    const offsetX = (rect.width - scaledWidth) / 2 + pan.x;
    const offsetY = (rect.height - scaledHeight) / 2 + pan.y;

    const canvasX = imageX * scale + offsetX;
    const canvasY = imageY * scale + offsetY;

    return { x: canvasX, y: canvasY };
  }, [zoom, pan]);

  // Draw canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image || !isImageLoaded) return;

    const ctx = canvas.getContext('2d');
    const rect = containerRef.current?.getBoundingClientRect();
    
    if (!rect) return;

    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scale = Math.min(
      canvas.width / image.width,
      canvas.height / image.height
    ) * zoom;

    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;
    const x = (canvas.width - scaledWidth) / 2 + pan.x;
    const y = (canvas.height - scaledHeight) / 2 + pan.y;

    // Draw image
    ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

    // Draw detections (all detections are now auto-processed)
    detections.forEach((det, index) => {
      const {bbox} = det;
      if (!bbox) return;

      const imgX = bbox.x || bbox.x1 || 0;
      const imgY = bbox.y || bbox.y1 || 0;
      const imgWidth = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const imgHeight = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;

      const canvasX = imgX * scale + x;
      const canvasY = imgY * scale + y;
      const canvasWidth = imgWidth * scale;
      const canvasHeight = imgHeight * scale;

      const customColor = customColors[index];
      const color = getClassColor(det.classId || det.class_id || 0, customColor);
      const isSelected = selectedDetection === index;
      const isHovered = hoveredDetection === index;

      // For auto-processed detections, only show boundary outlines without bounding boxes
      // But hide them when displayMode is 'point'
      if (det.autoProcessed && det.boundaryPixels && det.boundaryPixels.length > 0 && displayMode !== 'point') {
        // نمایش فقط مرزهای واقعی شیء (بدون bounding box)
        ctx.save();

        // تنظیمات برای smooth و گرد کردن خطوط
        ctx.strokeStyle = color.main;
        ctx.lineWidth = isSelected ? 4 : 3;
        ctx.globalAlpha = 0.3;
        ctx.lineJoin = 'round'; // گوشه‌های گرد
        ctx.lineCap = 'round'; // انتهای خطوط گرد
        ctx.shadowColor = color.main; // سایه برای smooth تر شدن
        ctx.shadowBlur = 1;

        // Draw boundary pixels as connected outline
        ctx.beginPath();
        det.boundaryPixels.forEach((pixel, index) => {
          const boundaryCanvasX = pixel.x * scale + x;
          const boundaryCanvasY = pixel.y * scale + y;

          if (index === 0) {
            ctx.moveTo(boundaryCanvasX, boundaryCanvasY);
          } else {
            ctx.lineTo(boundaryCanvasX, boundaryCanvasY);
          }
        });
        ctx.closePath();
        ctx.stroke();

        // Fill the selected area with very low opacity overlay
        ctx.fillStyle = color.main;
        ctx.globalAlpha = 0.3; // کاهش opacity از 0.15 به 0.08
        ctx.shadowBlur = 0; // حذف سایه برای fill
        ctx.fill();

        ctx.restore();

        // Draw small indicator dot for interaction (only when selected or hovered)
        if (isSelected || isHovered) {
          const centerX = canvasX + canvasWidth / 2;
          const centerY = canvasY + canvasHeight / 2;
          const pointRadius = isSelected ? 6 : 4;

          ctx.beginPath();
          ctx.arc(centerX, centerY, pointRadius, 0, 2 * Math.PI);
          ctx.fillStyle = color.main;
          ctx.fill();
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      } else {
        // نمایش detections عادی (non-processed) به صورت bounding box یا نقطه
        if (displayMode === 'point') {
          // نمایش به صورت نقطه در مرکز box
          const centerX = canvasX + canvasWidth / 2;
          const centerY = canvasY + canvasHeight / 2;
          const pointRadius = isSelected ? 8 : (isHovered ? 6 : 5);

          // رسم نقطه
          ctx.beginPath();
          ctx.arc(centerX, centerY, pointRadius, 0, 2 * Math.PI);
          ctx.fillStyle = color.main;
          ctx.fill();

          // رسم border سفید برای نقطه
          ctx.strokeStyle = '#FFFFFF';
          ctx.lineWidth = 1;
          ctx.stroke();

          // اگر selected است، یک حلقه بزرگتر هم بکش
          if (isSelected) {
            ctx.beginPath();
            ctx.arc(centerX, centerY, pointRadius + 4, 0, 2 * Math.PI);
            ctx.strokeStyle = color.main;
            ctx.lineWidth = 2;
            ctx.stroke();
          }
        } else {
          // نمایش به صورت bounding box (حالت پیش‌فرض)
          ctx.strokeStyle = isSelected ? color.dark : color.main;
          ctx.fillStyle = isHovered ? color.light : (color.light.replace('0.2', '0.1'));
          ctx.lineWidth = isSelected ? boxThickness + 1 : boxThickness;

          // Fill box
          ctx.fillRect(canvasX, canvasY, canvasWidth, canvasHeight);

          // Draw border
          ctx.strokeRect(canvasX, canvasY, canvasWidth, canvasHeight);

          // Draw resize handles if selected
          if (isSelected) {
            const handleSize = 8;
            ctx.fillStyle = color.main;
            ctx.fillRect(canvasX - handleSize/2, canvasY - handleSize/2, handleSize, handleSize); // NW
            ctx.fillRect(canvasX + canvasWidth - handleSize/2, canvasY - handleSize/2, handleSize, handleSize); // NE
            ctx.fillRect(canvasX - handleSize/2, canvasY + canvasHeight - handleSize/2, handleSize, handleSize); // SW
            ctx.fillRect(canvasX + canvasWidth - handleSize/2, canvasY + canvasHeight - handleSize/2, handleSize, handleSize); // SE
          }
        }
      }

      // Draw label
      if (showLabels) {
        const classId = det.classId || det.class_id || 0;
        const classNameEN = det.className || OPG_CLASSES_EN[classId] || 'Unknown';
        const classNameFA = getClassNameFA(classId, classNameEN);
        // فقط نام کلاس بدون درصد
        const label = classNameFA;
        
        // تنظیم فونت به Yekan Bakh
        ctx.font = `bold ${labelFontSize}px Yekan Bakh, Arial, sans-serif`;
        ctx.textBaseline = 'top';
        ctx.textAlign = 'center'; // وسط‌چین کردن متن
        
        // محاسبه موقعیت مرکز برای label
        let centerX; let labelY;
        if (displayMode === 'point') {
          // در حالت point، label بالای نقطه قرار می‌گیرد
          centerX = canvasX + canvasWidth / 2;
          labelY = canvasY + canvasHeight / 2 - 25; // بالای نقطه (بالاتر)
        } else {
          // در حالت box، label بالای box قرار می‌گیرد
          centerX = canvasX + canvasWidth / 2;
          labelY = canvasY - labelFontSize - 8; // بالای box (بالاتر)
        }
        
        // رسم متن اصلی با رنگ box (text shadow حذف شده)
        ctx.fillStyle = color.main;
        ctx.fillText(label, centerX, labelY);
        
        // بازگشت به حالت چپ‌چین برای سایر عناصر
        ctx.textAlign = 'left';
      }
    });
  }, [isImageLoaded, detections, zoom, pan, showLabels, boxThickness, selectedDetection, hoveredDetection, labelFontSize, displayMode, customColors, imageRef, canvasRef, containerRef]);

  // Load image and auto-process when both image and detections are ready
  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setIsImageLoaded(true);
      drawCanvas();

      // Process detections if we have them and haven't processed yet
      if (originalDetections.length > 0 && !hasProcessedRef.current) {
        hasProcessedRef.current = true;
        setTimeout(() => {
          autoProcessObjectsInDetections(originalDetections);
        }, 5000); // 5 second delay after AI detection
      }
    };
    img.src = imageUrl;
  }, [imageUrl, originalDetections, autoProcessObjectsInDetections, drawCanvas]);

  // Draw canvas whenever state changes (but skip if dragging to avoid lag)
  useEffect(() => {
    if (isImageLoaded && !isDragging && !isResizing) {
      drawCanvas();
    }
  }, [isImageLoaded, detections, zoom, pan, showLabels, boxThickness, hoveredDetection, selectedDetection, customColors, displayMode, labelFontSize, drawCanvas, isDragging, isResizing]);

  // Cleanup animation frames on unmount
  useEffect(() => {
    const cleanup = () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
    return cleanup;
  }, []);

  // Find detection at position
  const findDetectionAtPosition = useCallback((canvasX, canvasY, checkHandles = true) => {
    const imageCoords = getImageCoordinates(canvasX, canvasY);
    if (!imageCoords) return null;

    const image = imageRef.current;
    if (!image) return null;

    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect) return null;

    const scale = Math.min(
      rect.width / image.width,
      rect.height / image.height
    ) * zoom;

    // Check each detection (reverse order to get top-most)
    for (let i = detections.length - 1; i >= 0; i--) {
      const det = detections[i];
      const {bbox} = det;
      
      if (!bbox) continue;

      const x = bbox.x || bbox.x1 || 0;
      const y = bbox.y || bbox.y1 || 0;
      const width = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const height = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;

      // Check if point is inside bounding box
      if (imageCoords.x >= x && imageCoords.x <= x + width &&
          imageCoords.y >= y && imageCoords.y <= y + height) {
        
        // فقط اگر detection selected است و در حالت box هستیم، handle ها را چک کن
        if (checkHandles && selectedDetection === i && displayMode === 'box') {
          // Check if point is near resize handle (فقط برای selected detection)
          // اندازه handle در مختصات تصویر (تقریبا 8 پیکسل در canvas)
          const handleSizeInImage = 12 / scale; // tolerance بیشتر برای راحتی کلیک
          const tolerance = handleSizeInImage * 2; // tolerance بیشتر
          
          // Check corners با distance از گوشه
          const distToNW = Math.sqrt((imageCoords.x - x) ** 2 + (imageCoords.y - y) ** 2);
          const distToNE = Math.sqrt((imageCoords.x - (x + width)) ** 2 + (imageCoords.y - y) ** 2);
          const distToSW = Math.sqrt((imageCoords.x - x) ** 2 + (imageCoords.y - (y + height)) ** 2);
          const distToSE = Math.sqrt((imageCoords.x - (x + width)) ** 2 + (imageCoords.y - (y + height)) ** 2);
          
          if (distToNW < tolerance) {
            return { detection: i, handle: 'nw' };
          }
          if (distToNE < tolerance) {
            return { detection: i, handle: 'ne' };
          }
          if (distToSW < tolerance) {
            return { detection: i, handle: 'sw' };
          }
          if (distToSE < tolerance) {
            return { detection: i, handle: 'se' };
          }
        }
        
        return { detection: i, handle: null };
      }
    }
    
    return null;
  }, [detections, getImageCoordinates, zoom, selectedDetection, displayMode]);

  // Mouse handlers
  const handleMouseDown = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // ابتدا چک کن که آیا روی handle یک detection (حتی غیر-selected) کلیک شده یا نه
    const foundWithHandles = findDetectionAtPosition(canvasX, canvasY, true);
    const found = findDetectionAtPosition(canvasX, canvasY, false); // بدون چک handles
    
    // اگر روی handle کلیک شده (فقط برای selected detection) یا اگر detection جدیدی انتخاب شده و روی handle آن کلیک شده
    if (foundWithHandles && foundWithHandles.handle && displayMode === 'box') {
      // Start resizing (only in box mode)
      setIsResizing(true);
      setResizeHandle(foundWithHandles.handle);
      setSelectedDetection(foundWithHandles.detection);
      setDragStart({ x: canvasX, y: canvasY });
      // ذخیره bbox اولیه برای resize
      const det = detections[foundWithHandles.detection];
      const {bbox} = det;
      const imgX = bbox.x || bbox.x1 || 0;
      const imgY = bbox.y || bbox.y1 || 0;
      const imgWidth = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const imgHeight = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;
      setResizeStartBbox({ x: imgX, y: imgY, width: imgWidth, height: imgHeight });
    } else if (found) {
      // Start dragging
      setIsDragging(true);
      setSelectedDetection(found.detection);
      setDragStart({ x: canvasX, y: canvasY });
      
      // ذخیره موقعیت mouse در تصویر در زمان کلیک
      const imageCoords = getImageCoordinates(canvasX, canvasY);
      if (imageCoords) {
        setDragStartMouseImage({ x: imageCoords.x, y: imageCoords.y });
      }
      
      // ذخیره مختصات اولیه bbox در تصویر
      const det = detections[found.detection];
      const {bbox} = det;
      const imgX = bbox.x || bbox.x1 || 0;
      const imgY = bbox.y || bbox.y1 || 0;
      setDragStartImage({ x: imgX, y: imgY });
    } else {
      // Start panning
      setIsPanning(true);
      setPanStart({ x: canvasX - pan.x, y: canvasY - pan.y });
      setSelectedDetection(null);
    }
  };

  const handleMouseMove = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    if (isResizing && selectedDetection !== null && displayMode === 'box') {
      // استفاده از bbox اولیه برای resize
      const originalX = resizeStartBbox.x;
      const originalY = resizeStartBbox.y;
      const originalWidth = resizeStartBbox.width;
      const originalHeight = resizeStartBbox.height;

      const image = imageRef.current;
      if (!image) return;

      // محاسبه موقعیت فعلی mouse در تصویر
      const currentImageCoords = getImageCoordinates(canvasX, canvasY);
      if (!currentImageCoords) return;
      
      // محاسبه موقعیت handle در زمان شروع drag در تصویر
      // برای resize، باید موقعیت corner مربوطه را در تصویر محاسبه کنیم
      let handleImageX; let handleImageY;
      switch (resizeHandle) {
        case 'nw':
          handleImageX = originalX;
          handleImageY = originalY;
          break;
        case 'ne':
          handleImageX = originalX + originalWidth;
          handleImageY = originalY;
          break;
        case 'sw':
          handleImageX = originalX;
          handleImageY = originalY + originalHeight;
          break;
        case 'se':
          handleImageX = originalX + originalWidth;
          handleImageY = originalY + originalHeight;
          break;
        default:
          handleImageX = originalX;
          handleImageY = originalY;
          break;
      }

      const deltaX = currentImageCoords.x - handleImageX;
      const deltaY = currentImageCoords.y - handleImageY;

      let newX = originalX;
      let newY = originalY;
      let newWidth = originalWidth;
      let newHeight = originalHeight;

      // Resize based on handle
      switch (resizeHandle) {
        case 'nw':
          newX = Math.max(0, originalX + deltaX);
          newY = Math.max(0, originalY + deltaY);
          newWidth = originalWidth - deltaX;
          newHeight = originalHeight - deltaY;
          break;
        case 'ne':
          newY = Math.max(0, originalY + deltaY);
          newWidth = originalWidth + deltaX;
          newHeight = originalHeight - deltaY;
          break;
        case 'sw':
          newX = Math.max(0, originalX + deltaX);
          newWidth = originalWidth - deltaX;
          newHeight = originalHeight + deltaY;
          break;
        case 'se':
          newWidth = originalWidth + deltaX;
          newHeight = originalHeight + deltaY;
          break;
        default:
          break;
      }

      // Ensure minimum size
      if (newWidth < 5) {
        if (resizeHandle === 'nw' || resizeHandle === 'sw') {
          newX = originalX + originalWidth - 5;
        }
        newWidth = 5;
      }
      if (newHeight < 5) {
        if (resizeHandle === 'nw' || resizeHandle === 'ne') {
          newY = originalY + originalHeight - 5;
        }
        newHeight = 5;
      }

      // Ensure within image bounds
      if (newX + newWidth > image.width) {
        newWidth = image.width - newX;
      }
      if (newY + newHeight > image.height) {
        newHeight = image.height - newY;
      }

      // ذخیره موقعیت موقت برای render سریع
      tempDragPositionRef.current = {
        index: selectedDetection,
        bbox: { x: newX, y: newY, width: newWidth, height: newHeight, x1: newX, y1: newY, x2: newX + newWidth, y2: newY + newHeight }
      };

      // Cancel previous animation frame
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }

      // Render immediately using requestAnimationFrame for smooth performance
      rafIdRef.current = requestAnimationFrame(() => {
        // Draw canvas with temporary position
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!canvas || !ctx || !image) return;

        const canvasRect = containerRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        canvas.width = canvasRect.width;
        canvas.height = canvasRect.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const imgScale = Math.min(
          canvas.width / image.width,
          canvas.height / image.height
        ) * zoom;

        const scaledWidth = image.width * imgScale;
        const scaledHeight = image.height * imgScale;
        const x = (canvas.width - scaledWidth) / 2 + pan.x;
        const y = (canvas.height - scaledHeight) / 2 + pan.y;

        // Draw image
        ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

        // Draw all detections with temporary position for resized one
        detections.forEach((detItem, index) => {
          let bboxToUse = detItem.bbox;
          if (index === selectedDetection && tempDragPositionRef.current) {
            bboxToUse = tempDragPositionRef.current.bbox;
          }
          if (!bboxToUse) return;

          const imgX = bboxToUse.x || bboxToUse.x1 || 0;
          const imgY = bboxToUse.y || bboxToUse.y1 || 0;
          const imgW = bboxToUse.width || ((bboxToUse.x2 || 0) - (bboxToUse.x1 || 0)) || 0;
          const imgH = bboxToUse.height || ((bboxToUse.y2 || 0) - (bboxToUse.y1 || 0)) || 0;

          const canvasXPos = imgX * imgScale + x;
          const canvasYPos = imgY * imgScale + y;
          const canvasW = imgW * imgScale;
          const canvasH = imgH * imgScale;

          const customColor = customColors[index];
          const color = getClassColor(detItem.classId || detItem.class_id || 0, customColor);
          const isSelected = selectedDetection === index;
          const isHovered = hoveredDetection === index;

          // Draw detection (simplified for performance)
          if (displayMode === 'point') {
            const centerX = canvasXPos + canvasW / 2;
            const centerY = canvasYPos + canvasH / 2;
            const pointRadius = isSelected ? 8 : (isHovered ? 6 : 5);
            
            ctx.beginPath();
            ctx.arc(centerX, centerY, pointRadius, 0, 2 * Math.PI);
            ctx.fillStyle = color.main;
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            if (isSelected) {
              ctx.beginPath();
              ctx.arc(centerX, centerY, pointRadius + 4, 0, 2 * Math.PI);
              ctx.strokeStyle = color.main;
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          } else {
            ctx.strokeStyle = isSelected ? color.dark : color.main;
            ctx.fillStyle = isHovered ? color.light : (color.light.replace('0.2', '0.1'));
            ctx.lineWidth = isSelected ? boxThickness + 1 : boxThickness;
            ctx.fillRect(canvasXPos, canvasYPos, canvasW, canvasH);
            ctx.strokeRect(canvasXPos, canvasYPos, canvasW, canvasH);

            if (isSelected) {
              const handleSize = 8;
              ctx.fillStyle = color.main;
              ctx.fillRect(canvasXPos - handleSize/2, canvasYPos - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos + canvasW - handleSize/2, canvasYPos - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos - handleSize/2, canvasYPos + canvasH - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos + canvasW - handleSize/2, canvasYPos + canvasH - handleSize/2, handleSize, handleSize);
            }
          }

          // Draw label if needed (simplified)
          if (showLabels) {
            const classId = detItem.classId || detItem.class_id || 0;
            const classNameEN = detItem.className || OPG_CLASSES_EN[classId] || 'Unknown';
            const classNameFA = getClassNameFA(classId, classNameEN);
            const label = classNameFA;
            
            ctx.font = `bold ${labelFontSize}px Yekan Bakh, Arial, sans-serif`;
            ctx.textBaseline = 'top';
            ctx.textAlign = 'center';
            
            let centerX; let labelY;
            if (displayMode === 'point') {
              centerX = canvasXPos + canvasW / 2;
              labelY = canvasYPos + canvasH / 2 - 25;
            } else {
              centerX = canvasXPos + canvasW / 2;
              labelY = canvasYPos - labelFontSize - 8;
            }
            
            // رسم متن اصلی با رنگ box (text shadow حذف شده)
            ctx.fillStyle = color.main;
            ctx.fillText(label, centerX, labelY);
            ctx.textAlign = 'left';
          }
        });

        rafIdRef.current = null;
      });

      // Update state (but don't trigger full re-render during resize)
      const updatedDetections = [...detections];
      const det = detections[selectedDetection];
      updatedDetections[selectedDetection] = {
        ...det,
        bbox: {
          x: newX,
          y: newY,
          width: newWidth,
          height: newHeight,
          x1: newX,
          y1: newY,
          x2: newX + newWidth,
          y2: newY + newHeight,
        },
      };
      setDetections(updatedDetections);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
    } else if (isDragging && selectedDetection !== null) {
      const det = detections[selectedDetection];
      const {bbox} = det;
      const imgWidth = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const imgHeight = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;

      const image = imageRef.current;
      if (!image) return;

      const scale = Math.min(
        rect.width / image.width,
        rect.height / image.height
      ) * zoom;

      // محاسبه delta بر اساس مختصات تصویر از موقعیت mouse اولیه
      const currentImageCoords = getImageCoordinates(canvasX, canvasY);
      if (!currentImageCoords) return;
      
      const deltaX = currentImageCoords.x - dragStartMouseImage.x;
      const deltaY = currentImageCoords.y - dragStartMouseImage.y;

      let newX; let newY; let newWidth; let newHeight;

      if (displayMode === 'point') {
        // در حالت point، مرکز نقطه جابجا می‌شود
        // محاسبه مرکز اولیه
        const originalCenterX = dragStartImage.x + imgWidth / 2;
        const originalCenterY = dragStartImage.y + imgHeight / 2;
        
        // جابجایی مرکز بر اساس حرکت mouse
        const newCenterX = Math.max(imgWidth / 2, Math.min(image.width - imgWidth / 2, originalCenterX + deltaX));
        const newCenterY = Math.max(imgHeight / 2, Math.min(image.height - imgHeight / 2, originalCenterY + deltaY));
        
        newX = newCenterX - imgWidth / 2;
        newY = newCenterY - imgHeight / 2;
        newWidth = imgWidth;
        newHeight = imgHeight;
      } else {
        // در حالت box، box جابجا می‌شود بر اساس موقعیت اولیه
        newX = Math.max(0, Math.min(image.width - imgWidth, dragStartImage.x + deltaX));
        newY = Math.max(0, Math.min(image.height - imgHeight, dragStartImage.y + deltaY));
        newWidth = imgWidth;
        newHeight = imgHeight;
      }

      // ذخیره موقعیت موقت برای render سریع
      tempDragPositionRef.current = {
        index: selectedDetection,
        bbox: { x: newX, y: newY, width: newWidth, height: newHeight, x1: newX, y1: newY, x2: newX + newWidth, y2: newY + newHeight }
      };

      // Cancel previous animation frame
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }

      // Render immediately using requestAnimationFrame for smooth performance
      rafIdRef.current = requestAnimationFrame(() => {
        // Draw canvas with temporary position
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!canvas || !ctx || !image) return;

        const canvasRect = containerRef.current?.getBoundingClientRect();
        if (!canvasRect) return;

        canvas.width = canvasRect.width;
        canvas.height = canvasRect.height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const imgScale = Math.min(
          canvas.width / image.width,
          canvas.height / image.height
        ) * zoom;

        const scaledWidth = image.width * imgScale;
        const scaledHeight = image.height * imgScale;
        const x = (canvas.width - scaledWidth) / 2 + pan.x;
        const y = (canvas.height - scaledHeight) / 2 + pan.y;

        // Draw image
        ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

        // Draw all detections with temporary position for dragged one
        detections.forEach((detItem, index) => {
          let bboxToUse = detItem.bbox;
          if (index === selectedDetection && tempDragPositionRef.current) {
            bboxToUse = tempDragPositionRef.current.bbox;
          }
          if (!bboxToUse) return;

          const imgX = bboxToUse.x || bboxToUse.x1 || 0;
          const imgY = bboxToUse.y || bboxToUse.y1 || 0;
          const imgW = bboxToUse.width || ((bboxToUse.x2 || 0) - (bboxToUse.x1 || 0)) || 0;
          const imgH = bboxToUse.height || ((bboxToUse.y2 || 0) - (bboxToUse.y1 || 0)) || 0;

          const canvasXPos = imgX * imgScale + x;
          const canvasYPos = imgY * imgScale + y;
          const canvasW = imgW * imgScale;
          const canvasH = imgH * imgScale;

          const customColor = customColors[index];
          const color = getClassColor(detItem.classId || detItem.class_id || 0, customColor);
          const isSelected = selectedDetection === index;
          const isHovered = hoveredDetection === index;

          // Draw detection (simplified for performance)
          if (displayMode === 'point') {
            const centerX = canvasXPos + canvasW / 2;
            const centerY = canvasYPos + canvasH / 2;
            const pointRadius = isSelected ? 8 : (isHovered ? 6 : 5);
            
            ctx.beginPath();
            ctx.arc(centerX, centerY, pointRadius, 0, 2 * Math.PI);
            ctx.fillStyle = color.main;
            ctx.fill();
            ctx.strokeStyle = '#FFFFFF';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            if (isSelected) {
              ctx.beginPath();
              ctx.arc(centerX, centerY, pointRadius + 4, 0, 2 * Math.PI);
              ctx.strokeStyle = color.main;
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          } else {
            ctx.strokeStyle = isSelected ? color.dark : color.main;
            ctx.fillStyle = isHovered ? color.light : (color.light.replace('0.2', '0.1'));
            ctx.lineWidth = isSelected ? boxThickness + 1 : boxThickness;
            ctx.fillRect(canvasXPos, canvasYPos, canvasW, canvasH);
            ctx.strokeRect(canvasXPos, canvasYPos, canvasW, canvasH);

            if (isSelected) {
              const handleSize = 8;
              ctx.fillStyle = color.main;
              ctx.fillRect(canvasXPos - handleSize/2, canvasYPos - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos + canvasW - handleSize/2, canvasYPos - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos - handleSize/2, canvasYPos + canvasH - handleSize/2, handleSize, handleSize);
              ctx.fillRect(canvasXPos + canvasW - handleSize/2, canvasYPos + canvasH - handleSize/2, handleSize, handleSize);
            }
          }

          // Draw label if needed (simplified)
          if (showLabels) {
            const classId = detItem.classId || detItem.class_id || 0;
            const classNameEN = detItem.className || OPG_CLASSES_EN[classId] || 'Unknown';
            const classNameFA = getClassNameFA(classId, classNameEN);
            const label = classNameFA;
            
            ctx.font = `bold ${labelFontSize}px Yekan Bakh, Arial, sans-serif`;
            ctx.textBaseline = 'top';
            ctx.textAlign = 'center';
            
            let centerX; let labelY;
            if (displayMode === 'point') {
              centerX = canvasXPos + canvasW / 2;
              labelY = canvasYPos + canvasH / 2 - 25;
            } else {
              centerX = canvasXPos + canvasW / 2;
              labelY = canvasYPos - labelFontSize - 8;
            }
            
            // رسم متن اصلی با رنگ box (text shadow حذف شده)
            ctx.fillStyle = color.main;
            ctx.fillText(label, centerX, labelY);
            ctx.textAlign = 'left';
          }
        });

        rafIdRef.current = null;
      });

      // Update state (but don't trigger full re-render during drag)
      const updatedDetections = [...detections];
      const updatedDetection = {
        ...det,
        bbox: {
          x: newX,
          y: newY,
          width: newWidth,
          height: newHeight,
          x1: newX,
          y1: newY,
          x2: newX + newWidth,
          y2: newY + newHeight,
        },
      };

      // If this is an auto-processed detection with boundary pixels, update them too
      if (det.autoProcessed && det.boundaryPixels && det.boundaryPixels.length > 0) {
        // Calculate the translation delta
        const deltaX = newX - (det.bbox.x || det.bbox.x1 || 0);
        const deltaY = newY - (det.bbox.y || det.bbox.y1 || 0);

        // Update boundary pixels by translating them
        updatedDetection.boundaryPixels = det.boundaryPixels.map(pixel => ({
          x: pixel.x + deltaX,
          y: pixel.y + deltaY,
        }));
      }

      updatedDetections[selectedDetection] = updatedDetection;
      setDetections(updatedDetections);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
    } else if (isPanning) {
      setPan({
        x: canvasX - panStart.x,
        y: canvasY - panStart.y,
      });
    } else {
      // Hover detection
      const found = findDetectionAtPosition(canvasX, canvasY, true);
      setHoveredDetection(found ? found.detection : null);
      
      // Change cursor - فقط اگر detection selected است و روی handle است، resize cursor نشان بده
      if (found && found.handle && found.detection === selectedDetection && displayMode === 'box') {
        canvasRef.current.style.cursor = getResizeCursor(found.handle);
      } else if (found) {
        // اگر روی body detection است، move cursor نشان بده
        canvasRef.current.style.cursor = 'move';
      } else {
        canvasRef.current.style.cursor = 'default';
      }
    }
  };

  const handleMouseUp = () => {
    // Cancel any pending animation frames
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    
    setIsDragging(false);
    setIsResizing(false);
    setIsPanning(false);
    setResizeHandle(null);
    
    // Clear temporary drag position
    tempDragPositionRef.current = null;
    
    // Redraw canvas with final position
    if (isImageLoaded) {
      drawCanvas();
    }
    
    // بعد از پایان drag/resize، به‌روزرسانی موقعیت اولیه برای drag بعدی
    if (selectedDetection !== null && detections[selectedDetection]) {
      const det = detections[selectedDetection];
      const {bbox} = det;
      const imgX = bbox.x || bbox.x1 || 0;
      const imgY = bbox.y || bbox.y1 || 0;
      setDragStartImage({ x: imgX, y: imgY });
    }
  };

  const getResizeCursor = (handle) => {
    switch (handle) {
      case 'nw':
      case 'se':
        return 'nwse-resize';
      case 'ne':
      case 'sw':
        return 'nesw-resize';
      default:
        return 'default';
    }
  };

  // Delete selected detection
  const handleDelete = () => {
    if (selectedDetection !== null) {
      const updatedDetections = detections.filter((_, i) => i !== selectedDetection);
      
      // حذف رنگ سفارشی برای detection حذف شده
      const updatedCustomColors = { ...customColors };
      delete updatedCustomColors[selectedDetection];
      // تنظیم مجدد index ها برای detections باقیمانده
      const newCustomColors = {};
      Object.keys(updatedCustomColors).forEach(oldIndex => {
        const oldIdx = parseInt(oldIndex, 10);
        if (oldIdx > selectedDetection) {
          newCustomColors[oldIdx - 1] = updatedCustomColors[oldIdx];
        } else if (oldIdx < selectedDetection) {
          newCustomColors[oldIdx] = updatedCustomColors[oldIdx];
        }
      });
      setCustomColors(newCustomColors);
      
      setDetections(updatedDetections);
      setSelectedDetection(null);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
    }
  };

  // Context menu handlers
  const handleContextMenu = (e) => {
    e.preventDefault();
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    const found = findDetectionAtPosition(canvasX, canvasY);
    
    if (found) {
      setContextMenu({
        detectionIndex: found.detection,
        mouseX: e.clientX,
        mouseY: e.clientY,
      });
      setSelectedDetection(found.detection);
    } else {
      setContextMenu(null);
    }
  };

  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };

  const handleDeleteFromContextMenu = () => {
    if (contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined) {
      const indexToDelete = contextMenu.detectionIndex;
      const updatedDetections = detections.filter((_, i) => i !== indexToDelete);
      
      // حذف رنگ سفارشی برای detection حذف شده
      const updatedCustomColors = { ...customColors };
      delete updatedCustomColors[indexToDelete];
      // تنظیم مجدد index ها
      const newCustomColors = {};
      Object.keys(updatedCustomColors).forEach(oldIndex => {
        const oldIdx = parseInt(oldIndex, 10);
        if (oldIdx > indexToDelete) {
          newCustomColors[oldIdx - 1] = updatedCustomColors[oldIdx];
        } else if (oldIdx < indexToDelete) {
          newCustomColors[oldIdx] = updatedCustomColors[oldIdx];
        }
      });
      setCustomColors(newCustomColors);
      
      setDetections(updatedDetections);
      setSelectedDetection(null);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
    }
    handleCloseContextMenu();
  };

  const handleSelectColor = (color) => {
    if (contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined) {
      setCustomColors({
        ...customColors,
        [contextMenu.detectionIndex]: color
      });
    }
    handleCloseContextMenu();
  };

  // Zoom handlers
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev * 1.2, 20));
  };

  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev / 1.2, 0.1));
  };

  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };



  return (
    <Stack spacing={2}>
      {/* Toolbar */}
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" flexWrap="wrap">
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Zoom In">
            <IconButton size="small" onClick={handleZoomIn}>
              <Iconify icon="carbon:zoom-in" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <IconButton size="small" onClick={handleZoomOut}>
              <Iconify icon="carbon:zoom-out" />
            </IconButton>
          </Tooltip>
          <Tooltip title="Reset Zoom">
            <IconButton size="small" onClick={handleResetZoom}>
              <Iconify icon="carbon:reset" />
            </IconButton>
          </Tooltip>
          <Box sx={{ width: 2, height: 24, bgcolor: 'divider', mx: 1 }} />
          <Tooltip title={showLabels ? "مخفی کردن برچسب‌ها" : "نمایش برچسب‌ها"}>
            <IconButton size="small" onClick={() => setShowLabels(!showLabels)}>
              <Iconify icon="carbon:label" />
            </IconButton>
          </Tooltip>

          <Tooltip title={displayMode === 'box' ? "نمایش به صورت نقطه" : "نمایش به صورت جعبه"}>
            <IconButton size="small" onClick={() => setDisplayMode(displayMode === 'box' ? 'point' : 'box')}>
              <Iconify icon={displayMode === 'box' ? "carbon:checkbox" : "carbon:circle-dash"} />
            </IconButton>
          </Tooltip>

          <Tooltip title="انتخاب خودکار مرزهای واقعی شیء (3 ثانیه بعد از تشخیص AI)">
            <IconButton
              size="small"
              color="secondary"
              onClick={() => {
                if (originalDetections.length > 0 && isImageLoaded) {
                  console.log('Manual object selection triggered...');
                  autoProcessObjectsInDetections(originalDetections);
                }
              }}
              disabled={!isImageLoaded || originalDetections.length === 0}
            >
              <Iconify icon="carbon:magic-wand" />
            </IconButton>
          </Tooltip>

          <Box sx={{ width: 2, height: 24, bgcolor: 'divider', mx: 1 }} />

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Tooltip title="تنظیم tolerance برای تشخیص مرزهای شیء (مقدار کمتر = مرز دقیق‌تر)">
              <Typography variant="caption" sx={{ color: 'text.secondary', minWidth: 60 }}>
                Tolerance: {tolerance}
              </Typography>
            </Tooltip>
            <Slider
              value={tolerance}
              onChange={(_, newValue) => setTolerance(newValue)}
              min={1}
              max={20}
              step={1}
              size="small"
              sx={{
                width: 80,
                '& .MuiSlider-thumb': {
                  width: 12,
                  height: 12,
                },
                '& .MuiSlider-rail': {
                  height: 2,
                },
                '& .MuiSlider-track': {
                  height: 2,
                },
              }}
            />
          </Box>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          {selectedDetection !== null && (
            <>
              <Tooltip title="تغییر رنگ">
                <IconButton
                  size="small"
                  onClick={(e) => {
                    const rect = canvasRef.current?.getBoundingClientRect();
                    if (rect) {
                      setContextMenu({
                        detectionIndex: selectedDetection,
                        mouseX: e.clientX,
                        mouseY: e.clientY,
                      });
                    }
                  }}
                  sx={{
                    bgcolor: customColors[selectedDetection] || getClassColor(
                      detections[selectedDetection]?.classId || detections[selectedDetection]?.class_id || 0
                    ).main,
                    color: '#FFFFFF',
                    '&:hover': {
                      bgcolor: customColors[selectedDetection] || getClassColor(
                        detections[selectedDetection]?.classId || detections[selectedDetection]?.class_id || 0
                      ).main,
                    },
                  }}
                >
                  <Iconify icon="carbon:color-palette" />
                </IconButton>
              </Tooltip>
              <Tooltip title="حذف">
                <IconButton
                  size="small"
                  color="error"
                  onClick={handleDelete}
                >
                  <Iconify icon="carbon:delete" />
                </IconButton>
              </Tooltip>
            </>
          )}
        </Stack>
      </Stack>

      {/* Canvas Container */}
      <Box
        ref={containerRef}
        sx={{
          position: 'relative',
          width: '100%',
          border: '1px solid',
          borderColor: 'divider',
          borderRadius: 1,
          overflow: 'hidden',
          bgcolor: 'transparent', // Remove gray background
          minHeight: 400,
        }}
      >
        <Box
          component="canvas"
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onContextMenu={handleContextMenu}
          sx={{
            display: 'block',
            width: '100%',
            height: '100%',
            cursor: 'default',
          }}
        />
      </Box>

      {/* Status and Instructions */}
      <Stack spacing={1}>
        {/* Status Indicator */}
        <Typography variant="caption" sx={{ color: 'primary.main', textAlign: 'center', fontWeight: 'bold' }}>
          🤖🔍 حالت خودکار: تشخیص AI + انتخاب مرزهای واقعی شیء
        </Typography>

        {/* Instructions */}
        <Typography variant="caption" sx={{ color: 'text.secondary', textAlign: 'center' }}>
          💡 راهنما: برای جابجایی bounding box، روی آن کلیک و drag کنید | برای تغییر اندازه، گوشه‌های box را بکشید | برای پن (pan)، روی فضای خالی کلیک و drag کنید | راست کلیک برای تغییر رنگ و حذف
        </Typography>
      </Stack>

      {/* Context Menu */}
      <Menu
        open={contextMenu !== null}
        onClose={handleCloseContextMenu}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
        PaperProps={{
          sx: {
            minWidth: 200,
          }
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
            تغییر رنگ:
          </Typography>
          <ColorPicker
            colors={QUICK_COLORS}
            selected={
              contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined && detections[contextMenu.detectionIndex]
                ? (customColors[contextMenu.detectionIndex] || getClassColor(detections[contextMenu.detectionIndex]?.classId || detections[contextMenu.detectionIndex]?.class_id || 0).main)
                : '#000000' // رنگ پیش‌فرض در صورت نبودن
            }
            onSelectColor={handleSelectColor}
          />
        </Box>

        <Divider sx={{ my: 1 }} />

        <MenuItem onClick={handleDeleteFromContextMenu} sx={{ color: 'inherit' }}>
          <Iconify icon="solar:trash-bin-trash-bold" sx={{ mr: 1, color: 'inherit' }} />
          حذف
        </MenuItem>
      </Menu>
    </Stack>
  );
}
