import PropTypes from 'prop-types';
import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Menu from '@mui/material/Menu';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Divider from '@mui/material/Divider';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import { Iconify } from 'src/components/iconify';
import { ColorPicker } from 'src/components/color-utils';

// رنگ‌های مختلف برای کلاس‌های مختلف
const QUICK_COLORS = [
  '#FF6B6B', // قرمز صورتی
  '#4ECDC4', // آبی سبز
  '#45B7D1', // آبی روشن
  '#FFA07A', // نارنجی صورتی
  '#98D8C8', // سبز آبی
  '#F7DC6F', // زرد طلایی
  '#BB8FCE', // بنفش روشن
  '#85C1E9', // آبی آسمانی
  '#F8C471', // زرد نارنجی
  '#82E0AA', // سبز روشن
];

const getClassColor = (className, customColors = {}) => {
  // اگر رنگ سفارشی وجود دارد، از آن استفاده کن
  if (customColors[className]) {
    return customColors[className];
  }
  
  // استفاده از hash برای تعیین رنگ ثابت برای هر کلاس
  let hash = 0;
  for (let i = 0; i < className.length; i++) {
    hash = className.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % QUICK_COLORS.length;
  return QUICK_COLORS[index];
};

// ----------------------------------------------------------------------

export function DetectionVisualizer({ imageUrl, detections = [], onDetectionsChange }) {
  const canvasRef = useRef(null);
  const canvasContainerRef = useRef(null);
  const imageRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [baseCanvasSize, setBaseCanvasSize] = useState({ width: 800, height: 600 });
  
  // Zoom and Pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  // Interaction state
  const [hoveredDetection, setHoveredDetection] = useState(null);
  const [selectedDetection, setSelectedDetection] = useState(null);
  
  // State for custom colors per class
  const [customColors, setCustomColors] = useState({});
  
  // State for show/hide bounding boxes
  const [showBoundingBoxes, setShowBoundingBoxes] = useState(true);

  // State for show/hide landmark names (controls both labels and functionality)
  const [showLandmarkNames, setShowLandmarkNames] = useState(true);
  
  // State for context menu (right click)
  const [contextMenu, setContextMenu] = useState(null);
  
  // State for color picker menu (from legend)
  const [colorMenuAnchor, setColorMenuAnchor] = useState(null);
  const [selectedClassForColor, setSelectedClassForColor] = useState(null);

  // Ref برای requestAnimationFrame
  const rafIdRef = useRef(null);
  
  // Ref برای drawCanvas تا از re-render مداوم جلوگیری کنیم
  const drawCanvasRef = useRef(null);
  
  // State برای canvas size (برای جلوگیری از تغییرات جزئی) - باید قبل از findDetectionAtPosition تعریف شود
  const [canvasSize, setCanvasSize] = useState({ width: 0, height: 0 });

  // محاسبه base scale و canvas size
  const calculateBaseScale = useCallback(() => {
    if (!imageRef.current || !canvasContainerRef.current) return;

    const img = imageRef.current;
    const rect = canvasContainerRef.current.getBoundingClientRect();
    const containerWidth = rect.width;
    
    // محاسبه ارتفاع بر اساس aspect ratio تصویر
    const imageAspectRatio = img.height / img.width;
    const calculatedHeight = containerWidth * imageAspectRatio;
    
    // حداکثر ارتفاع 600px برای جلوگیری از کادر خیلی بلند
    const containerHeight = Math.min(calculatedHeight, 600);

    const scaleX = containerWidth / img.width;
    const scaleY = containerHeight / img.height;
    const baseScale = Math.min(scaleX, scaleY, 1); // حداکثر scale = 1

    setBaseCanvasSize({
      width: img.width * baseScale,
      height: img.height * baseScale,
    });
    
    // به‌روزرسانی canvas size برای container
    setCanvasSize({
      width: containerWidth,
      height: containerHeight,
    });
  }, []);
  
  // Helper: تبدیل مختصات mouse به مختصات canvas
  const getCanvasCoordinates = useCallback((clientX, clientY) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    
    const rect = canvas.getBoundingClientRect();
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }, []);
  
  // Helper: پیدا کردن detection در موقعیت مشخص
  // استفاده از canvas dimensions به جای getBoundingClientRect برای ثبات بیشتر
  const findDetectionAtPosition = useCallback((canvasX, canvasY) => {
    if (!imageRef.current || !canvasRef.current) return null;
    
    const img = imageRef.current;
    const canvas = canvasRef.current;
    
    // استفاده از canvas dimensions به جای getBoundingClientRect برای جلوگیری از تغییرات جزئی
    // اگر canvas size هنوز set نشده، از canvas.width/height استفاده کن
    const canvasWidth = canvas.width || canvasSize.width || 0;
    const canvasHeight = canvas.height || canvasSize.height || 0;

    if (canvasWidth === 0 || canvasHeight === 0) return null;

    const baseScale = Math.min(canvasWidth / img.width, canvasHeight / img.height, 1);
    const currentScale = baseScale * zoom;
    
    const scaledWidth = img.width * currentScale;
    const scaledHeight = img.height * currentScale;
    const imgOffsetX = (canvasWidth - scaledWidth) / 2;
    const imgOffsetY = (canvasHeight - scaledHeight) / 2;
    
    const imgX = (canvasX - pan.x - imgOffsetX) / currentScale;
    const imgY = (canvasY - pan.y - imgOffsetY) / currentScale;
    
    for (let i = detections.length - 1; i >= 0; i--) {
      const det = detections[i];
      const {bbox} = det;
      if (!bbox) continue;
      
      if (imgX >= bbox.x1 && imgX <= bbox.x2 && imgY >= bbox.y1 && imgY <= bbox.y2) {
        return i;
      }
    }
    
    return null;
  }, [detections, zoom, pan, canvasSize]);

  // تابع رسم canvas - باید قبل از useEffect‌ها تعریف شود
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imageRef.current || !canvasContainerRef.current) return;

    const ctx = canvas.getContext('2d');
    
    // بهبود کیفیت تصویر
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    
    // استفاده از canvasSize state به جای getBoundingClientRect برای جلوگیری از تغییرات جزئی
    // اگر canvasSize هنوز set نشده، از getBoundingClientRect استفاده کن
    let width;
    let height;
    if (canvasSize.width > 0 && canvasSize.height > 0) {
      ({ width, height } = canvasSize);
    } else {
      const rect = canvasContainerRef.current.getBoundingClientRect();
      width = Math.round(rect.width);
      height = Math.round(rect.height);
      // فقط یک بار set کن
      if (canvasSize.width === 0 && canvasSize.height === 0) {
        setCanvasSize({ width, height });
      }
    }
    
    // فقط اگر canvas size تغییر کرده یا canvas هنوز set نشده، canvas size را update کن
    if (canvas.width !== width || canvas.height !== height || canvas.width === 0 || canvas.height === 0) {
      // استفاده از devicePixelRatio برای کیفیت بهتر در نمایشگرهای با تراکم بالا
      const dpr = window.devicePixelRatio || 1;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      ctx.scale(dpr, dpr);
      // تنظیم مجدد اندازه CSS برای حفظ اندازه واقعی
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
    }
    
    ctx.clearRect(0, 0, width, height);

    const img = imageRef.current;
    const baseScale = Math.min(width / img.width, height / img.height, 1);
    const currentScale = baseScale * zoom;
    
    const scaledWidth = img.width * currentScale;
    const scaledHeight = img.height * currentScale;
    const x = (width - scaledWidth) / 2 + pan.x;
    const y = (height - scaledHeight) / 2 + pan.y;

    // رسم تصویر
    ctx.drawImage(imageRef.current, x, y, scaledWidth, scaledHeight);

    // رسم bounding boxes
    if (showBoundingBoxes) {
      detections.forEach((detection, index) => {
        const { bbox, class_name } = detection;
        if (!bbox) return;

        const color = getClassColor(class_name, customColors);
        const boxX = bbox.x1 * currentScale + x;
        const boxY = bbox.y1 * currentScale + y;
        const boxWidth = (bbox.x2 - bbox.x1) * currentScale;
        const boxHeight = (bbox.y2 - bbox.y1) * currentScale;

        const isHovered = hoveredDetection === index;
        const isSelected = selectedDetection === index;

        // رسم bounding box با ضخامت ثابت 1px
        ctx.lineWidth = 1;
        ctx.strokeStyle = color;
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(boxX, boxY, boxWidth, boxHeight);

        // پر کردن با رنگ شفاف (پررنگ‌تر برای hover و selected)
        ctx.fillStyle = color;
        ctx.globalAlpha = isSelected ? 0.4 : (isHovered ? 0.3 : 0.2);
        ctx.fillRect(boxX, boxY, boxWidth, boxHeight);
        ctx.globalAlpha = 1.0;
      });
    }

    // رسم برچسب‌ها (در صورت فعال بودن)
    if (showLandmarkNames) {
      detections.forEach((detection, index) => {
        const { bbox, class_name } = detection;
        if (!bbox) return;

        const color = getClassColor(class_name, customColors);
        const boxX = bbox.x1 * currentScale + x;
        const boxY = bbox.y1 * currentScale + y;
        const boxWidth = (bbox.x2 - bbox.x1) * currentScale;
        const centerX = boxX + boxWidth / 2;

        // متن برچسب (بدون درصد، فقط نام کلاس)
        let labelText = class_name;

        // تشخیص موبایل و تنظیم سایز فونت
        const isMobile = window.innerWidth < 768;
        const baseFontSize = isMobile ? 10 : 16;
        const maxFontSize = isMobile ? 14 : 20;
        const minFontSize = isMobile ? 8 : 16;
        
        // تنظیمات فونت - کوچک‌تر برای موبایل
        const fontSize = Math.max(minFontSize, Math.min(maxFontSize, baseFontSize * currentScale));
        ctx.font = `bold ${fontSize}px "Yekan Bakh", Arial, sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'bottom';

        // بررسی عرض متن و کوتاه کردن در صورت نیاز
        const textMetrics = ctx.measureText(labelText);
        const maxTextWidth = boxWidth * 0.9; // حداکثر 90% عرض باکس
        
        if (textMetrics.width > maxTextWidth) {
          // کوتاه کردن متن
          while (textMetrics.width > maxTextWidth && labelText.length > 0) {
            labelText = labelText.slice(0, -1);
            ctx.font = `bold ${fontSize}px "Yekan Bakh", Arial, sans-serif`;
            const newMetrics = ctx.measureText(`${labelText}...`);
            if (newMetrics.width <= maxTextWidth) {
              labelText += '...';
              break;
            }
          }
        }

        // محاسبه موقعیت برچسب (بالای باکس)
        const labelY = boxY - 5;
        
        // بررسی که متن خارج از کادر canvas نرود
        if (labelY >= fontSize) {
          // رسم پس‌زمینه برای خوانایی بهتر
          const padding = 4;
          const textWidth = ctx.measureText(labelText).width;
          const bgX = centerX - textWidth / 2 - padding;
          const bgY = labelY - fontSize - padding;
          const bgWidth = textWidth + padding * 2;
          const bgHeight = fontSize + padding * 2;

          ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
          ctx.fillRect(bgX, bgY, bgWidth, bgHeight);

          // رسم متن
          ctx.fillStyle = '#fff';
          ctx.fillText(labelText, centerX, labelY);
        }
      });
    }
  }, [detections, zoom, pan, customColors, showBoundingBoxes, showLandmarkNames, hoveredDetection, selectedDetection, canvasSize]);

  // بارگذاری تصویر - فقط یک بار برای هر imageUrl
  useEffect(() => {
    if (!imageUrl) return;
    
    // اگر تصویر قبلاً لود شده و همان URL است، فقط canvas را redraw کن
    if (imageRef.current && imageRef.current.src === imageUrl) {
      // فقط canvas را redraw کن، تصویر را دوباره لود نکن
      setTimeout(() => {
        if (canvasRef.current && imageRef.current && drawCanvasRef.current) {
          calculateBaseScale();
          drawCanvasRef.current();
        }
      }, 100);
      return;
    }

    // Cleanup previous image if exists
    if (imageRef.current) {
      imageRef.current.onload = null;
      imageRef.current.onerror = null;
    }

    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    
    let isCancelled = false;
    
    img.onload = () => {
      if (isCancelled) return;
      imageRef.current = img;
      setImageSize({ width: img.width, height: img.height });
      
      // کمی تأخیر برای اطمینان از render شدن container
      setTimeout(() => {
        if (isCancelled) return;
        calculateBaseScale();
        // Force canvas redraw after scale calculation
        if (canvasRef.current && imageRef.current && drawCanvasRef.current) {
          drawCanvasRef.current();
        }
      }, 150);
    };
    img.onerror = () => {
      if (isCancelled) return;
      console.error('Failed to load image:', imageUrl);
    };
    img.src = imageUrl;
    
    return () => {
      isCancelled = true;
      img.onload = null;
      img.onerror = null;
    };
  }, [imageUrl, calculateBaseScale]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      calculateBaseScale();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [calculateBaseScale]);
  
  // استفاده از ResizeObserver برای ردیابی تغییرات اندازه container
  useEffect(() => {
    if (!canvasContainerRef.current) return;
    
    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        // فقط اگر تغییر قابل توجهی (بیش از 2px) وجود دارد، canvas size را update کن
        if (Math.abs(width - canvasSize.width) > 2 || Math.abs(height - canvasSize.height) > 2) {
          setCanvasSize({ width, height });
          // Force canvas redraw after size change
          setTimeout(() => {
            if (canvasRef.current && imageRef.current && drawCanvasRef.current) {
              drawCanvasRef.current();
            }
          }, 50);
        }
      }
    });
    
    resizeObserver.observe(canvasContainerRef.current);
    
    return () => {
      resizeObserver.disconnect();
    };
  }, [canvasSize.width, canvasSize.height]);
  
  // Force canvas redraw when container becomes visible (for tab switching)
  // Use IntersectionObserver to detect when container becomes visible
  useEffect(() => {
    if (!imageUrl || !imageRef.current || !canvasContainerRef.current) return;
    
    const container = canvasContainerRef.current;
    let observer = null;
    
    const redrawCanvas = () => {
      if (canvasRef.current && imageRef.current && drawCanvasRef.current) {
        calculateBaseScale();
        drawCanvasRef.current();
      }
    };
    
    // Use IntersectionObserver to detect visibility
    if ('IntersectionObserver' in window) {
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting && entry.intersectionRatio > 0) {
              // Container is visible, redraw canvas
              setTimeout(redrawCanvas, 100);
            }
          });
        },
        { threshold: 0.1 }
      );
      
      observer.observe(container);
    } else {
      // Fallback for browsers without IntersectionObserver
      const checkAndRedraw = () => {
        const rect = container.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          setTimeout(redrawCanvas, 100);
        }
      };
      
      const timer1 = setTimeout(checkAndRedraw, 200);
      const timer2 = setTimeout(checkAndRedraw, 500);
      
      return () => {
        clearTimeout(timer1);
        clearTimeout(timer2);
      };
    }
    
    return () => {
      if (observer) {
        observer.disconnect();
      }
    };
  }, [imageUrl, imageSize.width, imageSize.height, calculateBaseScale]); // Run when image is loaded
  
  // رسم bounding boxes روی canvas با استفاده از requestAnimationFrame برای بهینه‌سازی
  useEffect(() => {
    // Cancel previous animation frame if exists
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
    }
    
    // Use requestAnimationFrame for smooth rendering
    rafIdRef.current = requestAnimationFrame(() => {
      drawCanvas();
    });
    
    // Cleanup
    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detections, zoom, pan, customColors, showBoundingBoxes, showLandmarkNames, hoveredDetection, selectedDetection, baseCanvasSize]);
  
  // Mouse event handlers
  const handleMouseDown = (e) => {
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    if (!coords) return;
    
    const detectionIndex = findDetectionAtPosition(coords.x, coords.y);
    
    if (detectionIndex !== null) {
      setSelectedDetection(detectionIndex);
    }
    // Pan غیرفعال شده - عکس قابل جابجایی نیست
  };
  
  const handleMouseMove = (e) => {
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    if (!coords) return;
    
    // Pan غیرفعال شده - فقط hover detection
    const detectionIndex = findDetectionAtPosition(coords.x, coords.y);
    setHoveredDetection(detectionIndex);
    
    if (canvasRef.current) {
      canvasRef.current.style.cursor = detectionIndex !== null ? 'pointer' : 'default';
    }
  };
  
  const handleMouseUp = () => {
    setIsPanning(false);
  };
  
  const handleMouseLeave = () => {
    setIsPanning(false);
    setHoveredDetection(null);
  };
  
  const handleContextMenu = (e) => {
    e.preventDefault();
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    if (!coords) return;
    
    const detectionIndex = findDetectionAtPosition(coords.x, coords.y);
    
    if (detectionIndex !== null) {
      setSelectedDetection(detectionIndex);
      setContextMenu({
        mouseX: e.clientX,
        mouseY: e.clientY,
        detectionIndex,
      });
    }
  };
  
  // Context menu handlers
  const handleCloseContextMenu = () => {
    setContextMenu(null);
  };
  
  const handleDeleteDetection = () => {
    if (selectedDetection !== null) {
      const updatedDetections = detections.filter((_, i) => i !== selectedDetection);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
      setSelectedDetection(null);
    }
  };
  
  const handleDeleteFromContextMenu = () => {
    if (contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined) {
      const updatedDetections = detections.filter((_, i) => i !== contextMenu.detectionIndex);
      if (onDetectionsChange) {
        onDetectionsChange(updatedDetections);
      }
      setSelectedDetection(null);
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
  
  // Handlers for color picker (from legend)
  const handleColorMenuOpen = (event, className) => {
    setColorMenuAnchor(event.currentTarget);
    setSelectedClassForColor(className);
  };

  const handleColorMenuClose = () => {
    setColorMenuAnchor(null);
    setSelectedClassForColor(null);
  };

  const handleColorSelect = (color) => {
    if (selectedClassForColor) {
      setCustomColors({
        ...customColors,
        [selectedClassForColor]: color,
      });
    }
    handleColorMenuClose();
  };

  // Group detections by class name for legend
  const detectionsByClass = detections.reduce((acc, det) => {
    if (!acc[det.class_name]) {
      acc[det.class_name] = [];
    }
    acc[det.class_name].push(det);
    return acc;
  }, {});

  return (
    <Box sx={{ width: '100%', maxWidth: '100%' }}>
      <Stack spacing={2}>
        {/* Toolbar */}
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
            
            {/* Show/Hide Bounding Boxes Button */}
            <Tooltip title={showBoundingBoxes ? "مخفی کردن باکس‌ها" : "نمایش باکس‌ها"}>
              <IconButton
                size="small"
                onClick={() => setShowBoundingBoxes(!showBoundingBoxes)}
                color={showBoundingBoxes ? "default" : "default"}
              >
                <Iconify icon={showBoundingBoxes ? "solar:eye-bold" : "solar:eye-closed-bold"} />
              </IconButton>
            </Tooltip>
            {/* Show/Hide Landmark Names Button - integrated with the showLabels functionality */}
            <Tooltip title={showLandmarkNames ? "مخفی کردن نام لندمارک‌ها" : "نمایش نام لندمارک‌ها"}>
              <IconButton
                size="small"
                onClick={() => {
                  setShowLandmarkNames(!showLandmarkNames);
                }}
                color={showLandmarkNames ? "primary" : "default"}
              >
                <Iconify icon={showLandmarkNames ? "solar:tag-bold" : "solar:tag-linear"} />
              </IconButton>
            </Tooltip>
            
          {/* Change Color Button - نمایش فقط وقتی یک detection انتخاب شده */}
          {selectedDetection !== null && detections[selectedDetection] && (
            <>
              <Tooltip title="تغییر رنگ">
                <IconButton
                  size="small"
                  onClick={(e) => {
                    e.stopPropagation(); // جلوگیری از propagation event
                    e.preventDefault(); // جلوگیری از default behavior
                    
                    const detection = detections[selectedDetection];
                    if (!detection) return;
                    
                    // محاسبه موقعیت بهتر برای menu (کنار دکمه)
                    const buttonRect = e.currentTarget.getBoundingClientRect();
                    const mouseX = buttonRect.right + 10; // کنار دکمه
                    const mouseY = buttonRect.top + buttonRect.height / 2; // وسط دکمه
                    
                    // اگر contextMenu قبلاً باز است و همان detection را نشان می‌دهد، فقط ببند
                    if (contextMenu !== null && contextMenu.detectionIndex === selectedDetection) {
                      setContextMenu(null);
                    } else {
                      // باز کردن menu با موقعیت جدید
                      setContextMenu({
                        detectionIndex: selectedDetection,
                        mouseX,
                        mouseY,
                      });
                    }
                  }}
                  sx={{
                    ml: 'auto',
                    bgcolor: getClassColor(detections[selectedDetection]?.class_name, customColors),
                    color: '#FFFFFF',
                    '&:hover': {
                      bgcolor: getClassColor(detections[selectedDetection]?.class_name, customColors),
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
                  onClick={(e) => {
                    e.stopPropagation(); // جلوگیری از propagation event
                    handleDeleteDetection();
                  }}
                >
                  <Iconify icon="humbleicons:times" />
                </IconButton>
              </Tooltip>
            </>
          )}
        </Stack>

        {/* Color Picker Menu (from legend or context menu) */}
        <Menu
          anchorEl={colorMenuAnchor && typeof colorMenuAnchor === 'object' && !colorMenuAnchor.mouseX ? colorMenuAnchor : null}
          open={Boolean(colorMenuAnchor)}
          onClose={handleColorMenuClose}
          anchorReference={
            typeof colorMenuAnchor === 'object' && colorMenuAnchor?.mouseX !== undefined
              ? 'anchorPosition'
              : 'anchorEl'
          }
          anchorPosition={
            typeof colorMenuAnchor === 'object' && colorMenuAnchor?.mouseX !== undefined
              ? { top: colorMenuAnchor.mouseY, left: colorMenuAnchor.mouseX }
              : undefined
          }
          anchorOrigin={{
            vertical: 'center',
            horizontal: 'right',
          }}
          transformOrigin={{
            vertical: 'center',
            horizontal: 'left',
          }}
          disableAutoFocusItem
        >
          <MenuItem>
            <Stack spacing={1}>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                Select color for: {selectedClassForColor}
              </Typography>
              <ColorPicker
                colors={QUICK_COLORS}
                selected={selectedClassForColor ? (customColors[selectedClassForColor] || getClassColor(selectedClassForColor)) : null}
                onSelectColor={handleColorSelect}
              />
            </Stack>
          </MenuItem>
        </Menu>

        {/* Canvas */}
        <Box
          ref={canvasContainerRef}
          sx={{
            position: 'relative',
            width: '100%',
            borderRadius: '12px',
            overflow: 'hidden',
            bgcolor: 'transparent',
            maxHeight: '600px',
            // استفاده از aspect-ratio برای حفظ نسبت تصویر
            ...(imageSize.width > 0 && imageSize.height > 0 && {
              aspectRatio: `${imageSize.width} / ${imageSize.height}`,
              maxHeight: '600px',
            }),
          }}
        >
          <Box
            component="canvas"
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            onContextMenu={handleContextMenu}
            sx={{
              display: 'block',
              width: '100%',
              height: '100%',
              cursor: 'default',
              borderRadius: '12px',
            }}
          />
        </Box>
        
        {/* Context Menu (Right Click or Color Button Click) */}
        <Menu
          open={contextMenu !== null}
          onClose={handleCloseContextMenu}
          anchorReference="anchorPosition"
          anchorPosition={
            contextMenu !== null
              ? { 
                  top: Math.max(10, Math.min(contextMenu.mouseY, window.innerHeight - 300)), // محدود کردن به viewport
                  left: Math.max(10, Math.min(contextMenu.mouseX, window.innerWidth - 250)) // محدود کردن به viewport
                }
              : undefined
          }
          PaperProps={{
            sx: {
              minWidth: 200,
              maxWidth: 250,
            }
          }}
          MenuListProps={{
            'aria-labelledby': 'color-picker-menu',
          }}
          disableAutoFocusItem={false}
          disableEnforceFocus={false}
        >
          <Box sx={{ px: 2, py: 1 }}>
            <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
              تغییر رنگ:
            </Typography>
            <ColorPicker
              colors={QUICK_COLORS}
              selected={
                contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined && detections[contextMenu.detectionIndex]
                  ? (customColors[detections[contextMenu.detectionIndex]?.class_name] || getClassColor(detections[contextMenu.detectionIndex]?.class_name, customColors))
                  : '#000000'
              }
              onSelectColor={(color) => {
                if (contextMenu?.detectionIndex !== null && contextMenu?.detectionIndex !== undefined) {
                  const detection = detections[contextMenu.detectionIndex];
                  if (detection) {
                    setCustomColors({
                      ...customColors,
                      [detection.class_name]: color,
                    });
                  }
                }
                handleCloseContextMenu();
              }}
            />
          </Box>

          <Divider sx={{ my: 1 }} />

          <MenuItem onClick={handleDeleteFromContextMenu} sx={{ color: 'inherit' }}>
            <Iconify icon="solar:trash-bin-trash-bold" width={20} sx={{ mr: 1, color: 'inherit' }} />
            حذف
          </MenuItem>
        </Menu>

        {/* Info */}
        {detections.length === 0 && (
          <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', py: 2 }}>
            No detections to display
          </Typography>
        )}
      </Stack>
    </Box>
  );
}

DetectionVisualizer.propTypes = {
  imageUrl: PropTypes.string,
  detections: PropTypes.arrayOf(
    PropTypes.shape({
      class_name: PropTypes.string.isRequired,
      confidence: PropTypes.number.isRequired,
      bbox: PropTypes.shape({
        x1: PropTypes.number.isRequired,
        y1: PropTypes.number.isRequired,
        x2: PropTypes.number.isRequired,
        y2: PropTypes.number.isRequired,
      }),
    })
  ),
  onDetectionsChange: PropTypes.func,
};
