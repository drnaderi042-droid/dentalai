import PropTypes from 'prop-types';
import { useRef, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Menu from '@mui/material/Menu';
import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import FormControlLabel from '@mui/material/FormControlLabel';

import { Iconify } from 'src/components/iconify';
import { ColorPicker } from 'src/components/color-utils';

// ----------------------------------------------------------------------

// رنگ‌بندی landmarks بر اساس نوع
const LANDMARK_COLORS = {
  // نقاط اسکلتی
  S: '#2196F3',
  N: '#2196F3',
  A: '#2196F3',
  B: '#2196F3',
  Pog: '#2196F3',
  Go: '#2196F3',
  Me: '#2196F3',
  Or: '#2196F3',
  Po: '#2196F3',
  ANS: '#2196F3',
  PNS: '#2196F3',
  
  // نقاط دندانی
  U1: '#F44336',
  L1: '#F44336',
  U1A: '#F44336',
  L1A: '#F44336',
  
  // پیش‌فرض
  default: '#4CAF50',
};

// خطوط ارتباطی بین landmarks
const LANDMARK_LINES = [
  // خط SNA
  ['S', 'N'],
  ['N', 'A'],
  ['S', 'A'],
  
  // خط SNB
  ['S', 'B'],
  ['N', 'B'],
  
  // صفحه اکلوژن
  ['ANS', 'PNS'],
  
  // صفحه فرانکفورت
  ['Or', 'Po'],
  
  // صفحه ماندیبولار
  ['Go', 'Me'],
  
  // Facial profile
  ['N', 'Pog'],
  
  // دندانی
  ['U1', 'L1'],
];

// ----------------------------------------------------------------------

export function LandmarkVisualizer({ imageUrl, landmarks, imageSize, showLabels: initialShowLabels = true }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);
  
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [showLabels, setShowLabels] = useState(initialShowLabels);
  const [showLines, setShowLines] = useState(true);
  const [pointSize, setPointSize] = useState(8);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  
  // State for color changing functionality
  const [selectedLandmark, setSelectedLandmark] = useState(null);
  const [customColors, setCustomColors] = useState({});
  const [colorMenuAnchor, setColorMenuAnchor] = useState(null);

  // Convert landmarks from array format (facial landmark) to object format if needed
  const processedLandmarks = useMemo(() => {
    if (!landmarks || !Object.keys(landmarks).length) {
      if (Array.isArray(landmarks)) {
        // Convert array format to object format
        const landmarksObj = {};
        landmarks.forEach(landmark => {
          if (landmark.name && landmark.x !== undefined && landmark.y !== undefined) {
            landmarksObj[landmark.name] = {
              x: landmark.x,
              y: landmark.y,
              confidence: landmark.confidence
            };
          }
        });
        return landmarksObj;
      }
      return {};
    }
    return landmarks;
  }, [landmarks]);

  // رسم canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image || !isImageLoaded) return;

    const ctx = canvas.getContext('2d');
    const rect = containerRef.current?.getBoundingClientRect();
    
    if (!rect) return;

    // تنظیم اندازه canvas
    canvas.width = rect.width;
    canvas.height = rect.height;

    // پاک کردن canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // محاسبه scale برای fit کردن تصویر
    const scale = Math.min(
      canvas.width / image.width,
      canvas.height / image.height
    ) * zoom;

    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;

    // مرکز کردن تصویر
    const x = (canvas.width - scaledWidth) / 2 + pan.x;
    const y = (canvas.height - scaledHeight) / 2 + pan.y;

    // رسم تصویر
    ctx.save();
    ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

    if (processedLandmarks && Object.keys(processedLandmarks).length > 0) {
      // رسم خطوط بین landmarks
      if (showLines) {
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);

        LANDMARK_LINES.forEach(([start, end]) => {
          if (processedLandmarks[start] && processedLandmarks[end]) {
            const startX = x + (processedLandmarks[start].x * scale);
            const startY = y + (processedLandmarks[start].y * scale);
            const endX = x + (processedLandmarks[end].x * scale);
            const endY = y + (processedLandmarks[end].y * scale);

            ctx.beginPath();
            ctx.moveTo(startX, startY);
            ctx.lineTo(endX, endY);
            ctx.stroke();
          }
        });

        ctx.setLineDash([]);
      }

      // رسم landmarks
      Object.entries(processedLandmarks).forEach(([name, coords]) => {
        const lmX = x + (coords.x * scale);
        const lmY = y + (coords.y * scale);

        // Get color - use custom color if available, otherwise use default
        const color = customColors[name] || LANDMARK_COLORS[name] || LANDMARK_COLORS.default;
        const isSelected = selectedLandmark === name;

        // رسم نقطه
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(lmX, lmY, pointSize, 0, 2 * Math.PI);
        ctx.fill();

        // حاشیه سفید یا زرد برای selected
        ctx.strokeStyle = isSelected ? '#FFD700' : '#FFFFFF';
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.stroke();

        // رسم label
        if (showLabels) {
          ctx.font = 'bold 14px Arial';
          ctx.fillStyle = '#FFFFFF';
          ctx.strokeStyle = '#000000';
          ctx.lineWidth = 3;
          
          // سایه
          ctx.strokeText(name, lmX + pointSize + 5, lmY + 5);
          ctx.fillText(name, lmX + pointSize + 5, lmY + 5);
        }

        // نمایش confidence اگر موجود باشد
        if (coords.confidence && showLabels) {
          const confText = `${Math.round(coords.confidence * 100)}%`;
          ctx.font = '10px Arial';
          ctx.fillStyle = '#FFEB3B';
          ctx.strokeText(confText, lmX + pointSize + 5, lmY + 18);
          ctx.fillText(confText, lmX + pointSize + 5, lmY + 18);
        }
      });
    }

    ctx.restore();
  }, [zoom, pan, processedLandmarks, showLabels, showLines, pointSize, selectedLandmark, customColors, isImageLoaded]);

  // بارگذاری تصویر
  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setIsImageLoaded(true);
      drawCanvas();
    };
    img.src = imageUrl;
  }, [imageUrl, drawCanvas]);

  // رسم مجدد هنگام تغییر
  useEffect(() => {
    if (isImageLoaded) {
      drawCanvas();
    }
  }, [isImageLoaded, drawCanvas]);

  // Pan handlers
  const handleMouseDown = (e) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;

    const canvas = canvasRef.current;
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    // Check if clicking on a landmark
    if (processedLandmarks && Object.keys(processedLandmarks).length > 0) {
      const image = imageRef.current;
      if (!image || !canvas) return;

      const scale = Math.min(
        canvas.width / image.width,
        canvas.height / image.height
      ) * zoom;
      
      const scaledWidth = image.width * scale;
      const scaledHeight = image.height * scale;
      const x = (canvas.width - scaledWidth) / 2 + pan.x;
      const y = (canvas.height - scaledHeight) / 2 + pan.y;
      
      let clickedLandmark = null;
      Object.entries(processedLandmarks).forEach(([name, coords]) => {
        const lmX = x + (coords.x * scale);
        const lmY = y + (coords.y * scale);
        const distance = Math.sqrt((canvasX - lmX) ** 2 + (canvasY - lmY) ** 2);
        if (distance <= pointSize + 5) { // 5px tolerance
          clickedLandmark = name;
        }
      });
      
      if (clickedLandmark) {
        if (selectedLandmark === clickedLandmark) {
          setSelectedLandmark(null); // Deselect if clicking the same landmark
        } else {
          setSelectedLandmark(clickedLandmark);
          setIsPanning(false); // Don't pan when selecting landmark
        }
        return; // Don't start panning when selecting landmark
      }
    }
    
    setIsPanning(true);
    setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
  };

  // Touch handlers
  const handleTouchStart = (e) => {
    if (e.touches.length === 1) {
      setIsPanning(true);
      setPanStart({
        x: e.touches[0].clientX - pan.x,
        y: e.touches[0].clientY - pan.y,
      });
    }
  };

  const handleTouchMove = (e) => {
    if (isPanning && e.touches.length === 1) {
      setPan({
        x: e.touches[0].clientX - panStart.x,
        y: e.touches[0].clientY - panStart.y,
      });
    }
  };

  const handleTouchEnd = () => {
    setIsPanning(false);
  };

  // Zoom handlers
  const handleZoomIn = () => setZoom((prev) => Math.min(prev + 0.2, 5));
  const handleZoomOut = () => setZoom((prev) => Math.max(prev - 0.2, 0.5));
  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Wheel zoom
  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setZoom((prev) => Math.max(0.5, Math.min(5, prev + delta)));
  };

  if (!imageUrl) {
    return (
      <Box
        sx={{
          width: '100%',
          height: 600,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'background.neutral',
          borderRadius: 1,
        }}
      >
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          تصویر آپلود نشده است
        </Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={2}>
      {/* Controls */}
      <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
        <Stack direction="row" spacing={1}>
          <IconButton size="small" onClick={handleZoomIn} disabled={zoom >= 5}>
            <Iconify icon="solar:add-circle-bold" />
          </IconButton>
          <IconButton size="small" onClick={handleZoomOut} disabled={zoom <= 0.5}>
            <Iconify icon="solar:minus-circle-bold" />
          </IconButton>
          <IconButton size="small" onClick={handleResetView}>
            <Iconify icon="solar:refresh-bold" />
          </IconButton>
          
          {/* Add Point Button - borderless with 4px padding */}
          <Tooltip title="افزودن نقطه جدید">
            <IconButton
              size="small"
              onClick={() => {
                // TODO: Implement add point functionality
                console.log('Add point clicked');
              }}
              sx={{
                border: 'none',
                p: '4px',
                '&:hover': {
                  backgroundColor: 'action.hover',
                },
              }}
            >
              <Iconify icon="carbon:add" />
            </IconButton>
          </Tooltip>
        </Stack>

        <Typography variant="caption" sx={{ minWidth: 80 }}>
          Zoom: {Math.round(zoom * 100)}%
        </Typography>

        {/* Show Labels Button - positioned next to zoom controls */}
        <Tooltip title={showLabels ? "مخفی کردن برچسب‌ها" : "نمایش برچسب‌ها"}>
          <IconButton
            size="small"
            onClick={() => setShowLabels(!showLabels)}
            color={showLabels ? "primary" : "default"}
          >
            <Iconify icon={showLabels ? "solar:tag-bold" : "solar:tag-linear"} />
          </IconButton>
        </Tooltip>

        <FormControlLabel
          control={
            <Switch
              size="small"
              checked={showLines}
              onChange={(e) => setShowLines(e.target.checked)}
            />
          }
          label={<Typography variant="caption">خطوط</Typography>}
        />

        <Stack direction="row" spacing={1} alignItems="center" sx={{ minWidth: 150 }}>
          <Typography variant="caption">اندازه:</Typography>
          <Slider
            size="small"
            value={pointSize}
            onChange={(e, value) => setPointSize(value)}
            min={4}
            max={16}
            step={2}
            valueLabelDisplay="auto"
            sx={{ flex: 1 }}
          />
        </Stack>

        {/* Color selection button */}
        {selectedLandmark && (
          <Tooltip title="تغییر رنگ لندمارک انتخاب شده">
            <IconButton
              size="small"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setColorMenuAnchor({
                  mouseX: rect.right + 10,
                  mouseY: rect.top + rect.height / 2,
                });
              }}
              sx={{
                width: 20,
                height: 20,
                border: '2px solid white',
                bgcolor: customColors[selectedLandmark] || LANDMARK_COLORS[selectedLandmark] || LANDMARK_COLORS.default,
                color: '#FFFFFF',
                '&:hover': {
                  bgcolor: customColors[selectedLandmark] || LANDMARK_COLORS[selectedLandmark] || LANDMARK_COLORS.default,
                },
              }}
            >
              <Iconify icon="carbon:color-palette" sx={{ fontSize: 12 }} />
            </IconButton>
          </Tooltip>
        )}
      </Stack>

      {/* Canvas */}
      <Box
        ref={containerRef}
        sx={{
          width: '100%',
          height: 600,
          bgcolor: 'transparent', // Remove gray background
          borderRadius: 1,
          overflow: 'hidden',
          cursor: isPanning ? 'grabbing' : 'grab',
          position: 'relative',
          border: '1px solid',
          borderColor: 'divider',
        }}
      >
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onTouchStart={handleTouchStart}
          onTouchMove={handleTouchMove}
          onTouchEnd={handleTouchEnd}
          onWheel={handleWheel}
          style={{
            width: '100%',
            height: '100%',
            display: 'block',
          }}
        />
      </Box>

      {/* Color Picker Menu */}
      <Menu
        open={colorMenuAnchor !== null}
        onClose={() => setColorMenuAnchor(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          colorMenuAnchor !== null
            ? { top: colorMenuAnchor.mouseY, left: colorMenuAnchor.mouseX }
            : undefined
        }
        PaperProps={{
          sx: {
            minWidth: 200,
            maxWidth: 250,
          }
        }}
      >
        <Box sx={{ px: 2, py: 1 }}>
          <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: 1 }}>
            انتخاب رنگ برای: {selectedLandmark}
          </Typography>
          <ColorPicker
            colors={[
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
              '#F1948A', // صورتی روشن
              '#AED6F1', // آبی خیلی روشن
            ]}
            selected={customColors[selectedLandmark] || LANDMARK_COLORS[selectedLandmark] || LANDMARK_COLORS.default}
            onSelectColor={(color) => {
              setCustomColors({
                ...customColors,
                [selectedLandmark]: color,
              });
              setColorMenuAnchor(null);
            }}
          />
        </Box>
      </Menu>

      {/* Legend */}
      <Stack direction="row" spacing={3} flexWrap="wrap">
        <Stack direction="row" spacing={1} alignItems="center">
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              bgcolor: LANDMARK_COLORS.S,
              border: '2px solid white',
            }}
          />
          <Typography variant="caption">نقاط اسکلتی</Typography>
        </Stack>
        <Stack direction="row" spacing={1} alignItems="center">
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              bgcolor: LANDMARK_COLORS.U1,
              border: '2px solid white',
            }}
          />
          <Typography variant="caption">نقاط دندانی</Typography>
        </Stack>
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          💡 برای جابجایی کلیک کرده و بکشید | Scroll برای zoom
        </Typography>
      </Stack>
    </Stack>
  );
}

LandmarkVisualizer.propTypes = {
  imageUrl: PropTypes.string,
  landmarks: PropTypes.oneOfType([
    PropTypes.arrayOf(PropTypes.object),
    PropTypes.object,
  ]),
  imageSize: PropTypes.shape({
    width: PropTypes.number,
    height: PropTypes.number,
  }),
  showLabels: PropTypes.bool,
};
