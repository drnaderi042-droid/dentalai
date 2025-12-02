import PropTypes from 'prop-types';
import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import { Iconify } from 'src/components/iconify';

// dlib 68 facial landmarks indices mapping
const DLIB_INDICES = {
  glabella: 27,
  subnasale: 33,
  pogonion: 8,
  nose_tip: 30,
  chin: 8,
  menton: 8,
  // Eyes (for pupil calculation and vertical lines)
  right_eye_start: 36,
  right_eye_end: 41,
  right_eye_inner: 39,
  right_eye_outer: 36,
  right_eye_top: 37,
  right_eye_bottom: 41,
  left_eye_start: 42,
  left_eye_end: 47,
  left_eye_inner: 42,
  left_eye_outer: 45,
  left_eye_top: 43,
  left_eye_bottom: 47,
  // Jaw
  jaw_left: 0,
  jaw_right: 16,
  chin_left: 0,
  chin_right: 16,
  // Mouth
  mouth_left: 48,
  mouth_right: 54,
  mouth_top: 51,
  mouth_bottom: 57,
};

// MediaPipe Face Mesh landmark indices mapping (468 landmarks)
const MEDIAPIPE_INDICES = {
  // Left eye
  left_eye_inner: 133,
  left_eye: 33,
  left_eye_outer: 7,
  left_eye_top: 159,
  left_eye_bottom: 145,
  left_eye_corner: 33,
  
  // Right eye
  right_eye_inner: 362,
  right_eye: 263,
  right_eye_outer: 249,
  right_eye_top: 386,
  right_eye_bottom: 374,
  right_eye_corner: 263,
  
  // Nose
  nose_tip: 1,
  nose_bridge_top: 6,
  nose_bridge_1: 6,
  nose_bridge_4: 19,
  nose_left: 131,
  nose_right: 360,
  nose_bottom: 2,
  subnasale: 2, // تقریبی - نزدیک به nose_bottom
  
  // Mouth
  mouth_left: 61,
  mouth_right: 291,
  mouth_top: 13,
  mouth_bottom: 14,
  mouth_center_top: 13,
  mouth_center_bottom: 14,
  mouth_outer_1: 61,
  mouth_outer_3: 13,
  mouth_outer_5: 291,
  mouth_outer_9: 14,
  
  // Chin
  chin: 175,
  chin_center: 175,
  chin_8: 175,
  chin_left: 172,
  chin_right: 397,
  pogonion: 175, // تقریبی
  menton: 175, // تقریبی
  
  // Face outline
  forehead: 10,
  face_left: 172,
  face_right: 397,
  
  // Additional points
  glabella: 9,
  left_cheek: 116,
  right_cheek: 345,
  
  // Jaw (برای MediaPipe از face outline استفاده می‌کنیم)
  jaw_left: 172, // تقریبی - از face_left
  jaw_right: 397, // تقریبی - از face_right
};

// dlib landmark names mapping for better display (English names)
const DLIB_LANDMARK_NAMES = {
  // Jaw line (0-16)
  0: 'Jaw-L',
  1: 'Jaw',
  2: 'Jaw',
  3: 'Jaw',
  4: 'Jaw',
  5: 'Jaw',
  6: 'Jaw',
  7: 'Jaw',
  8: 'Chin',
  9: 'Jaw',
  10: 'Jaw',
  11: 'Jaw',
  12: 'Jaw',
  13: 'Jaw',
  14: 'Jaw',
  15: 'Jaw',
  16: 'Jaw-R',
  // Right eyebrow (17-21)
  17: 'Eyebrow-R-Outer',
  18: 'Eyebrow-R',
  19: 'Eyebrow-R',
  20: 'Eyebrow-R',
  21: 'Eyebrow-R-Inner',
  // Left eyebrow (22-26)
  22: 'Eyebrow-L-Inner',
  23: 'Eyebrow-L',
  24: 'Eyebrow-L',
  25: 'Eyebrow-L',
  26: 'Eyebrow-L-Outer',
  // Nose (27-35)
  27: 'Glabella',
  28: 'Nose-Bridge',
  29: 'Nose-Bridge',
  30: 'Nose-Tip',
  31: 'Nose-L',
  32: 'Nose',
  33: 'Subnasale',
  34: 'Nose',
  35: 'Nose-R',
  // Right eye (36-41)
  36: 'Eye-R-Outer',
  37: 'Eye-R-Top',
  38: 'Eye-R',
  39: 'Eye-R-Inner',
  40: 'Eye-R',
  41: 'Eye-R-Bottom',
  // Left eye (42-47)
  42: 'Eye-L-Inner',
  43: 'Eye-L-Top',
  44: 'Eye-L',
  45: 'Eye-L-Outer',
  46: 'Eye-L',
  47: 'Eye-L-Bottom',
  // Mouth (48-67)
  48: 'Mouth-L',
  49: 'Mouth-Outer',
  50: 'Mouth-Outer',
  51: 'Mouth-Top',
  52: 'Mouth-Outer',
  53: 'Mouth-Outer',
  54: 'Mouth-R',
  55: 'Mouth-Outer',
  56: 'Mouth-Outer',
  57: 'Mouth-Bottom',
  58: 'Mouth-Outer',
  59: 'Mouth-Outer',
  60: 'Mouth-Outer',
  61: 'Mouth-Outer',
  62: 'Upper-Lip',
  63: 'Lower-Lip',
  64: 'Mouth-Outer',
  65: 'Mouth-Outer',
  66: 'Mouth-Outer',
  67: 'Mouth-Outer',
};

// MediaPipe landmark names mapping for better display (English names)
// فقط لندمارک‌های مهم MediaPipe (468 landmarks)
// لیست کامل لندمارک‌های MediaPipe: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
const MEDIAPIPE_LANDMARK_NAMES = {
  // Key facial features - Nose
  1: 'Nose-Tip',
  2: 'Subnasale',
  6: 'Nose-Bridge-Top',
  19: 'Nose-Bridge',
  131: 'Nose-L',
  360: 'Nose-R',
  
  // Forehead
  9: 'Glabella',
  10: 'Forehead',
  
  // Left Eye
  7: 'Eye-L-Outer',
  33: 'Eye-L',
  133: 'Eye-L-Inner',
  145: 'Eye-L-Bottom',
  159: 'Eye-L-Top',
  
  // Right Eye
  249: 'Eye-R-Outer',
  263: 'Eye-R',
  362: 'Eye-R-Inner',
  374: 'Eye-R-Bottom',
  386: 'Eye-R-Top',
  
  // Mouth
  13: 'Mouth-Top',
  14: 'Mouth-Bottom',
  61: 'Mouth-L',
  291: 'Mouth-R',
  
  // Face Outline
  172: 'Face-L',
  175: 'Chin',
  397: 'Face-R',
  
  // Additional important landmarks
  0: 'Face-Outline',
  11: 'Face-Outline',
  12: 'Face-Outline',
  116: 'Cheek-L',
  345: 'Cheek-R',
};

// ----------------------------------------------------------------------

export function LandmarkVisualizer({ 
  imageUrl, 
  landmarks = [],
  showLandmarks: propShowLandmarks = true,
  onShowLandmarksChange,
  showOutlines: propShowOutlines = false,
  onShowOutlinesChange,
  showProfileLines: propShowProfileLines = false,
  onShowProfileLinesChange,
  showFrontalLines: propShowFrontalLines = false,
  onShowFrontalLinesChange,
  showLandmarkNames: propShowLandmarkNames = true,
  onShowLandmarkNamesChange,
  onImageError, // Callback برای اطلاع‌رسانی خطا به parent
  onImageLoad, // Callback برای اطلاع‌رسانی موفقیت بارگذاری به parent
}) {
  const canvasRef = useRef(null);
  const canvasContainerRef = useRef(null);
  const imageRef = useRef(null);
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 });
  const [baseCanvasSize, setBaseCanvasSize] = useState({ width: 800, height: 600 });
  const [imageLoaded, setImageLoaded] = useState(false); // Track if image is loaded
  const [loadError, setLoadError] = useState(null); // Track loading errors
  const [retryCount, setRetryCount] = useState(0); // Track retry attempts
  const [isRetrying, setIsRetrying] = useState(false); // Track retry state
  
  // Zoom and Pan state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [localShowLandmarks, setLocalShowLandmarks] = useState(propShowLandmarks);
  const [localShowOutlines, setLocalShowOutlines] = useState(propShowOutlines);
  const [localShowProfileLines, setLocalShowProfileLines] = useState(propShowProfileLines);
  const [localShowFrontalLines, setLocalShowFrontalLines] = useState(propShowFrontalLines);
  const [localShowLandmarkNames, setLocalShowLandmarkNames] = useState(propShowLandmarkNames);

  // محاسبه base scale و canvas size
  const calculateBaseScale = useCallback(() => {
    if (!imageRef.current || !canvasContainerRef.current) return;

    const img = imageRef.current;
    const rect = canvasContainerRef.current.getBoundingClientRect();
    const containerWidth = rect.width;
    const containerHeight = rect.height;

    const scaleX = containerWidth / img.width;
    const scaleY = containerHeight / img.height;
    const baseScale = Math.min(scaleX, scaleY, 1);

    setBaseCanvasSize({
      width: img.width * baseScale,
      height: img.height * baseScale,
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

  // Manual retry function
  const retryImageLoad = useCallback(() => {
    if (imageUrl) {
      setRetryCount(0);
      setIsRetrying(false);
      setLoadError(null);
      // Trigger a re-render by updating state
      setImageLoaded(false);
      
      // Force a new effect cycle
      setTimeout(() => {
        // Re-trigger the image loading by changing a dummy state
        setRetryCount(prev => prev + 1);
      }, 50);
    }
  }, [imageUrl]);

  // Validate blob URL before attempting to load
  const validateBlobUrl = useCallback((url) => {
    if (!url.startsWith('blob:')) {
      return Promise.resolve(true);
    }
    
    return new Promise((resolve) => {
      const testImg = new Image();
      const timeoutId = setTimeout(() => {
        console.warn('[Landmark Visualizer] Blob URL validation timeout');
        testImg.dispatchEvent(new Event('error'));
      }, 3000); // 3 seconds timeout for validation
      
      testImg.onload = () => {
        clearTimeout(timeoutId);
        console.log('[Landmark Visualizer] Blob URL validation successful');
        resolve(true);
      };
      
      testImg.onerror = () => {
        clearTimeout(timeoutId);
        console.error('[Landmark Visualizer] Blob URL validation failed');
        resolve(false);
      };
      
      testImg.src = url;
    });
  }, []);

  // Enhanced image loading with better error handling
  useEffect(() => {
    if (!imageUrl) {
      console.warn('[Landmark Visualizer] No imageUrl provided');
      imageRef.current = null;
      setImageSize({ width: 0, height: 0 });
      setImageLoaded(false);
      setLoadError('No image URL provided');
      if (onImageError) {
        onImageError('No image URL provided');
      }
      return;
    }
    
    // Reset states for new image load
    setImageLoaded(false);
    setLoadError(null);
    setIsRetrying(false);

    console.log('[Landmark Visualizer] Loading image:', {
      url: imageUrl?.substring(0, 100),
      isBlobUrl: imageUrl.startsWith('blob:'),
      timestamp: new Date().toISOString(),
    });

    // Validate blob URL if applicable
    const loadImageWithValidation = async () => {
      try {
        if (imageUrl.startsWith('blob:')) {
          console.log('[Landmark Visualizer] Validating blob URL before loading...');
          const isValid = await validateBlobUrl(imageUrl);
          if (!isValid) {
            const errorMsg = 'Blob URL is invalid or expired';
            console.error(`[Landmark Visualizer] ${errorMsg}:`, imageUrl);
            // Don't set error immediately - let the image loading attempt handle it
            // The error will be caught in the img.onerror handler below
            console.warn('[Landmark Visualizer] Blob URL validation failed, but proceeding with load attempt');
          }
        }
        
        // Proceed with actual image loading
        const img = new window.Image();
        img.crossOrigin = 'anonymous';
        
        // Enhanced timeout handling for different URL types
        let timeoutId = null;
        if (imageUrl.startsWith('blob:')) {
          // Shorter timeout for blob URLs as they can become invalid quickly
          timeoutId = setTimeout(() => {
            const errorMsg = 'Blob URL loading timeout - URL may be invalid';
            console.warn(`[Landmark Visualizer] ${errorMsg}:`, imageUrl?.substring(0, 100));
            img.onerror(new Error(errorMsg));
          }, 5000); // 5 seconds for blob URLs
        } else {
          // Longer timeout for regular URLs
          timeoutId = setTimeout(() => {
            const errorMsg = 'Image loading timeout';
            console.warn(`[Landmark Visualizer] ${errorMsg}:`, imageUrl?.substring(0, 100));
            img.onerror(new Error(errorMsg));
          }, 15000); // 15 seconds for regular URLs
        }
        
        img.onload = () => {
          if (timeoutId) {
            clearTimeout(timeoutId);
          }
          console.log('[Landmark Visualizer] Image loaded successfully:', {
            width: img.width,
            height: img.height,
            naturalWidth: img.naturalWidth,
            naturalHeight: img.naturalHeight,
            complete: img.complete,
            isBlobUrl: imageUrl.startsWith('blob:'),
            attempt: retryCount + 1,
          });
          
          imageRef.current = img;
          setImageSize({ width: img.width, height: img.height });
          setImageLoaded(true);
          setRetryCount(0);
          setIsRetrying(false);
          setLoadError(null);
          
          // Notify parent of successful load
          if (onImageLoad) {
            onImageLoad({
              success: true,
              width: img.width,
              height: img.height,
              isBlobUrl: imageUrl.startsWith('blob:'),
            });
          }
          
          // Force a re-render of the canvas
          setTimeout(() => {
            calculateBaseScale();
          }, 100);
        };
        
        img.onerror = (error) => {
          if (timeoutId) {
            clearTimeout(timeoutId);
          }
          
          const errorMsg = error?.message || 'Failed to load image';
          console.error('[Landmark Visualizer] Error loading image:', {
            error: errorMsg,
            imageUrl: imageUrl?.substring(0, 100),
            isBlobUrl: imageUrl.startsWith('blob:'),
            attempt: retryCount + 1,
            timestamp: new Date().toISOString(),
          });
          
          imageRef.current = null;
          setImageSize({ width: 0, height: 0 });
          setImageLoaded(false);
          setLoadError(errorMsg);
          setIsRetrying(false);
          
          // Notify parent of error
          if (onImageError) {
            onImageError(errorMsg);
          }
        };
        
        img.src = imageUrl;
      } catch (error) {
        console.error('[Landmark Visualizer] Unexpected error during image load:', error);
        setLoadError('Unexpected error occurred');
        if (onImageError) {
          onImageError('Unexpected error occurred');
        }
      }
    };

    loadImageWithValidation();
  }, [imageUrl, calculateBaseScale, onImageError, onImageLoad, validateBlobUrl, retryCount]);

  // Handle window resize
  useEffect(() => {
    const handleResize = () => {
      calculateBaseScale();
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, [calculateBaseScale]);

  // رسم لندمارک‌ها روی canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !canvasContainerRef.current) {
      console.warn('[Landmark Visualizer] Canvas or container not ready');
      return;
    }
    
    if (!imageRef.current) {
      console.warn('[Landmark Visualizer] Image not loaded yet');
      // Clear canvas and show appropriate message
      const ctx = canvas.getContext('2d');
      const rect = canvasContainerRef.current.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Show loading or error message
      ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'white';
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      
      if (loadError) {
        ctx.fillText(`خطا در بارگذاری تصویر: ${loadError}`, canvas.width / 2, canvas.height / 2);
        if (isRetrying) {
          ctx.font = '14px Arial';
          ctx.fillText('در حال تلاش مجدد...', canvas.width / 2, canvas.height / 2 + 25);
        }
      } else if (isRetrying) {
        ctx.fillText('در حال تلاش مجدد...', canvas.width / 2, canvas.height / 2);
      } else {
        ctx.fillText('در حال بارگذاری تصویر...', canvas.width / 2, canvas.height / 2);
      }
      return;
    }

    const ctx = canvas.getContext('2d');
    const rect = canvasContainerRef.current.getBoundingClientRect();
    
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const img = imageRef.current;
    
    // بررسی اینکه تصویر به درستی load شده است
    if (!img.complete || img.naturalWidth === 0) {
      console.warn('[Landmark Visualizer] Image not fully loaded:', {
        complete: img.complete,
        naturalWidth: img.naturalWidth,
        naturalHeight: img.naturalHeight,
        src: img.src?.substring(0, 100),
      });
      return;
    }
    
    const baseScale = Math.min(rect.width / img.width, rect.height / img.height, 1);
    const currentScale = baseScale * zoom;
    
    const scaledWidth = img.width * currentScale;
    const scaledHeight = img.height * currentScale;
    const x = (canvas.width - scaledWidth) / 2 + pan.x;
    const y = (canvas.height - scaledHeight) / 2 + pan.y;

    // رسم تصویر
    try {
      ctx.drawImage(img, x, y, scaledWidth, scaledHeight);
      console.log('[Landmark Visualizer] Image drawn:', {
        imageSize: { width: img.width, height: img.height },
        canvasSize: { width: canvas.width, height: canvas.height },
        scaledSize: { width: scaledWidth, height: scaledHeight },
        position: { x, y },
        landmarksCount: landmarks.length,
      });
    } catch (error) {
      console.error('[Landmark Visualizer] Error drawing image:', error);
    }

    // Helper function to find landmarks by name or index (supports dlib and MediaPipe)
    const findLandmark = (names) => {
      if (!Array.isArray(names)) names = [names];
      
      // First, try to find by name (for named landmarks)
      for (const name of names) {
        const found = landmarks.find(l => {
          if (!l.name) return false;
          const lName = l.name.toLowerCase();
          const searchName = name.toLowerCase();
          return lName === searchName || 
                 lName.includes(searchName) || 
                 searchName.includes(lName);
        });
        if (found) return found;
      }
      
      // If not found by name, try index mapping for dlib (68 landmarks) or MediaPipe (468 landmarks)
      const landmarkCount = landmarks.length;
      const isDlib = landmarkCount === 68;
      const isMediaPipe = landmarkCount >= 400; // MediaPipe has 468 landmarks
      
      if ((isDlib || isMediaPipe) && landmarks.some(l => l.name && l.name.startsWith('landmark_'))) {
        for (const name of names) {
          const searchName = name.toLowerCase();
          
          // Try direct mapping from DLIB_INDICES (for dlib) or MEDIAPIPE_INDICES (for MediaPipe)
          let targetIndex = null;
          
          if (isDlib) {
            targetIndex = DLIB_INDICES[searchName];
            
            // Try partial matching for eye landmarks
            if (targetIndex === undefined) {
              if (searchName.includes('eye') && searchName.includes('inner')) {
                if (searchName.includes('left') || searchName.includes('l')) {
                  targetIndex = DLIB_INDICES.left_eye_inner;
                } else if (searchName.includes('right') || searchName.includes('r')) {
                  targetIndex = DLIB_INDICES.right_eye_inner;
                }
              }
            }
          } else if (isMediaPipe) {
            targetIndex = MEDIAPIPE_INDICES[searchName];
            
            // Try partial matching for MediaPipe landmarks
            if (targetIndex === undefined) {
              if (searchName.includes('eye') && searchName.includes('inner')) {
                if (searchName.includes('left') || searchName.includes('l')) {
                  targetIndex = MEDIAPIPE_INDICES.left_eye_inner;
                } else if (searchName.includes('right') || searchName.includes('r')) {
                  targetIndex = MEDIAPIPE_INDICES.right_eye_inner;
                }
              }
            }
          }
          
          if (targetIndex !== undefined && targetIndex !== null) {
            const found = landmarks.find(l => {
              const idx = l.index !== undefined ? l.index : 
                         (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx === targetIndex;
            });
            if (found) {
              console.log(`[Landmark Visualizer] Found ${isMediaPipe ? 'MediaPipe' : 'dlib'} landmark "${name}" at index ${targetIndex}`);
              return found;
            }
          }
          
          // Try extracting index from landmark name like "landmark_39" or "landmark_42"
          const indexMatch = searchName.match(/landmark[_\s]?(\d+)/);
          if (indexMatch) {
            const extractedIndex = parseInt(indexMatch[1], 10);
            const found = landmarks.find(l => {
              const idx = l.index !== undefined ? l.index : 
                         (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
              return idx === extractedIndex;
            });
            if (found) return found;
          }
          
          // Special case: calculate pupil positions (for dlib only)
          if (isDlib) {
            if (searchName === 'left_pupil') {
              const leftEyeLandmarks = landmarks.filter(l => {
                const idx = l.index !== undefined ? l.index : 
                           (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
                return idx >= DLIB_INDICES.left_eye_start && idx <= DLIB_INDICES.left_eye_end;
              });
              if (leftEyeLandmarks.length > 0) {
                const avgX = leftEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / leftEyeLandmarks.length;
                const avgY = leftEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / leftEyeLandmarks.length;
                return { x: avgX, y: avgY, index: 69, name: 'left_pupil' };
              }
            } else if (searchName === 'right_pupil') {
              const rightEyeLandmarks = landmarks.filter(l => {
                const idx = l.index !== undefined ? l.index : 
                           (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
                return idx >= DLIB_INDICES.right_eye_start && idx <= DLIB_INDICES.right_eye_end;
              });
              if (rightEyeLandmarks.length > 0) {
                const avgX = rightEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / rightEyeLandmarks.length;
                const avgY = rightEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / rightEyeLandmarks.length;
                return { x: avgX, y: avgY, index: 68, name: 'right_pupil' };
              }
            }
          } else if (isMediaPipe) {
            // برای MediaPipe، از لندمارک‌های چشم برای محاسبه pupil استفاده می‌کنیم
            if (searchName === 'left_pupil') {
              // MediaPipe left eye landmarks: 33, 7, 159, 145, 133, 362 (تقریبی)
              // استفاده از left_eye (33) و left_eye_inner (133) و left_eye_outer (7)
              const leftEyeLandmarks = landmarks.filter(l => {
                const idx = l.index !== undefined ? l.index : 
                           (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
                return idx === 33 || idx === 7 || idx === 133 || idx === 159 || idx === 145;
              });
              if (leftEyeLandmarks.length > 0) {
                const avgX = leftEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / leftEyeLandmarks.length;
                const avgY = leftEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / leftEyeLandmarks.length;
                return { x: avgX, y: avgY, index: 469, name: 'left_pupil' };
              }
            } else if (searchName === 'right_pupil') {
              // MediaPipe right eye landmarks: 263, 249, 386, 374, 362
              const rightEyeLandmarks = landmarks.filter(l => {
                const idx = l.index !== undefined ? l.index : 
                           (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
                return idx === 263 || idx === 249 || idx === 362 || idx === 386 || idx === 374;
              });
              if (rightEyeLandmarks.length > 0) {
                const avgX = rightEyeLandmarks.reduce((sum, l) => sum + l.x, 0) / rightEyeLandmarks.length;
                const avgY = rightEyeLandmarks.reduce((sum, l) => sum + l.y, 0) / rightEyeLandmarks.length;
                return { x: avgX, y: avgY, index: 468, name: 'right_pupil' };
              }
            }
          }
        }
      }
      
      return null;
    };

    // رسم خطوط تقسیم‌بندی پروفایل صورت
    if (localShowProfileLines) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 1;
      
      // پیدا کردن نقاط کلیدی
      const forehead = findLandmark(['nose_bridge_top', 'forehead', 'nose_bridge_1']);
      const noseTip = findLandmark(['nose_tip', 'nose']);
      const chin = findLandmark(['chin_center', 'chin', 'chin_8']);
      const leftFace = findLandmark(['chin_left', 'face_left', 'chin_1']);
      const rightFace = findLandmark(['chin_right', 'face_right', 'chin_17']);
      const leftEyeOuter = findLandmark(['left_eye_outer', 'left_eye_1']);
      const leftEyeInner = findLandmark(['left_eye_inner', 'left_eye_2']);
      const rightEyeInner = findLandmark(['right_eye_inner', 'right_eye_2']);
      const rightEyeOuter = findLandmark(['right_eye_outer', 'right_eye_1']);

      if (forehead && noseTip && chin && leftFace && rightFace) {
        const faceTop = forehead.y * currentScale + y;
        const faceBottom = chin.y * currentScale + y;
        const faceLeft = leftFace.x * currentScale + x;
        const faceRight = rightFace.x * currentScale + x;
        const faceHeight = faceBottom - faceTop;
        const faceWidth = faceRight - faceLeft;

        // خطوط افقی: تقسیم به سه بخش (یک سوم)
        const upperThird = faceTop + faceHeight / 3;
        const middleThird = faceTop + (faceHeight * 2) / 3;
        
        ctx.beginPath();
        ctx.moveTo(faceLeft, upperThird);
        ctx.lineTo(faceRight, upperThird);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(faceLeft, middleThird);
        ctx.lineTo(faceRight, middleThird);
        ctx.stroke();

        // خطوط عمودی: تقسیم به 5 قسمت
        const verticalFifth1 = faceLeft + faceWidth / 5;
        const verticalFifth2 = faceLeft + (faceWidth * 2) / 5;
        const verticalFifth3 = faceLeft + (faceWidth * 3) / 5;
        const verticalFifth4 = faceLeft + (faceWidth * 4) / 5;

        ctx.beginPath();
        ctx.moveTo(verticalFifth1, faceTop);
        ctx.lineTo(verticalFifth1, faceBottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(verticalFifth2, faceTop);
        ctx.lineTo(verticalFifth2, faceBottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(verticalFifth3, faceTop);
        ctx.lineTo(verticalFifth3, faceBottom);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(verticalFifth4, faceTop);
        ctx.lineTo(verticalFifth4, faceBottom);
        ctx.stroke();

        // خطوط عمودی از چشم‌ها
        if (leftEyeOuter && leftEyeInner && rightEyeInner && rightEyeOuter) {
          const leftEyeOuterX = leftEyeOuter.x * currentScale + x;
          const leftEyeInnerX = leftEyeInner.x * currentScale + x;
          const rightEyeInnerX = rightEyeInner.x * currentScale + x;
          const rightEyeOuterX = rightEyeOuter.x * currentScale + x;

          ctx.beginPath();
          ctx.moveTo(leftEyeOuterX, faceTop);
          ctx.lineTo(leftEyeOuterX, faceBottom);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(leftEyeInnerX, faceTop);
          ctx.lineTo(leftEyeInnerX, faceBottom);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(rightEyeInnerX, faceTop);
          ctx.lineTo(rightEyeInnerX, faceBottom);
          ctx.stroke();

          ctx.beginPath();
          ctx.moveTo(rightEyeOuterX, faceTop);
          ctx.lineTo(rightEyeOuterX, faceBottom);
          ctx.stroke();
        }
      }
    }

    // رسم خطوط frontal (برای نمای فرونتال)
    if (localShowFrontalLines) {
      // پیدا کردن نقاط کلیدی برای خطوط عمودی
      const glabella = findLandmark(['glabella', 'nose_bridge_top', 'forehead']);
      const subnasale = findLandmark(['subnasale', 'nose_bottom']);
      const pogonion = findLandmark(['pogonion', 'chin_center', 'chin']);
      const menton = findLandmark(['menton', 'chin']);
      // پیدا کردن Eye-R-Inner و Eye-L-Inner برای رسم خطوط عمودی
      // استفاده از نام‌های مختلف برای سازگاری با همه مدل‌ها
      // تشخیص نوع مدل قبل از جستجو
      const landmarkCountForEye = landmarks.length;
      const isDlibForEye = landmarkCountForEye === 68;
      const isMediaPipeForEye = landmarkCountForEye >= 400;
      
      // برای MediaPipe: از index مستقیم استفاده می‌کنیم
      // برای dlib: از نام‌ها استفاده می‌کنیم
      let leftEyeInner = null;
      let rightEyeInner = null;
      
      if (isMediaPipeForEye) {
        // برای MediaPipe: استفاده از index مستقیم
        // Left Eye Inner = index 133, Right Eye Inner = index 362
        leftEyeInner = landmarks.find(l => {
          const idx = l.index !== undefined ? 
                     (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                     (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
          return idx === 133;
        });
        rightEyeInner = landmarks.find(l => {
          const idx = l.index !== undefined ? 
                     (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                     (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
          return idx === 362;
        });
      } else {
        // برای dlib و سایر مدل‌ها: استفاده از نام‌ها
        leftEyeInner = findLandmark([
          'left_eye_inner', 
          'eye_l_inner', 
          'eye-l-inner',
          'left_eye_2',
          'landmark_42' // dlib index 42
        ]);
        rightEyeInner = findLandmark([
          'right_eye_inner', 
          'eye_r_inner', 
          'eye-r-inner',
          'right_eye_2',
          'landmark_39' // dlib index 39
        ]);
      }
      
      // Debug log برای بررسی پیدا شدن landmarks
      console.log('[Landmark Visualizer] Eye landmarks search result:', {
        leftEyeInner: leftEyeInner ? { 
          found: true, 
          x: leftEyeInner.x, 
          y: leftEyeInner.y, 
          name: leftEyeInner.name, 
          index: leftEyeInner.index 
        } : { found: false },
        rightEyeInner: rightEyeInner ? { 
          found: true, 
          x: rightEyeInner.x, 
          y: rightEyeInner.y, 
          name: rightEyeInner.name, 
          index: rightEyeInner.index 
        } : { found: false },
        landmarksCount: landmarks.length,
        sampleLandmarks: landmarks.slice(0, 10).map(l => ({ 
          name: l.name, 
          index: l.index,
          x: l.x,
          y: l.y
        })),
      });
      
      // برای dlib: Jaw-Left = landmark_0, Jaw-Right = landmark_16
      // برای MediaPipe: Jaw-Left = landmark_172 (Face-L), Jaw-Right = landmark_397 (Face-R)
      // تشخیص نوع مدل
      const landmarkCountForJaw = landmarks.length;
      const isDlibForJaw = landmarkCountForJaw === 68;
      const isMediaPipeForJaw = landmarkCountForJaw >= 400;
      
      let jawLeft = findLandmark(['jaw_left', 'chin_left', 'face_left']);
      let jawRight = findLandmark(['jaw_right', 'chin_right', 'face_right']);
      
      // اگر پیدا نشد، از index استفاده کن
      if (!jawLeft && landmarks.some(l => l.name && l.name.startsWith('landmark_'))) {
        const targetIndex = isMediaPipeForJaw ? 172 : (isDlibForJaw ? 0 : null);
        if (targetIndex !== null) {
          const jawLeftLandmark = landmarks.find(l => {
            const idx = l.index !== undefined ? 
                       (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                       (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
            return idx === targetIndex;
          });
          if (jawLeftLandmark) jawLeft = jawLeftLandmark;
        }
      }
      
      if (!jawRight && landmarks.some(l => l.name && l.name.startsWith('landmark_'))) {
        const targetIndex = isMediaPipeForJaw ? 397 : (isDlibForJaw ? 16 : null);
        if (targetIndex !== null) {
          const jawRightLandmark = landmarks.find(l => {
            const idx = l.index !== undefined ? 
                       (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                       (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
            return idx === targetIndex;
          });
          if (jawRightLandmark) jawRight = jawRightLandmark;
        }
      }
      
      // پیدا کردن بالاترین و پایین‌ترین نقطه صورت برای رسم خطوط عمودی
      let topY = glabella ? glabella.y * currentScale + y : 0;
      let bottomY = menton ? menton.y * currentScale + y : (pogonion ? pogonion.y * currentScale + y : 0);
      
      // اگر glabella یا menton پیدا نشد، از سایر نقاط استفاده کن
      if (!glabella) {
        const forehead = findLandmark(['forehead', 'nose_bridge_top']);
        if (forehead) topY = forehead.y * currentScale + y;
      }
      if (!menton && !pogonion) {
        const chin = findLandmark(['chin']);
        if (chin) bottomY = chin.y * currentScale + y;
      }
      
      // رنگ‌های مدرن و شیک (استفاده از رنگ‌های Tailwind CSS)
      const colors = {
        midline: 'rgba(99, 102, 241, 0.85)',      // Indigo-500 (آبی بنفش)
        eyeR: 'rgba(236, 72, 153, 0.75)',         // Pink-500 (صورتی)
        eyeL: 'rgba(14, 165, 233, 0.75)',         // Sky-500 (آبی آسمانی)
        jawR: 'rgba(34, 197, 94, 0.75)',          // Green-500 (سبز)
        jawL: 'rgba(251, 146, 60, 0.75)',         // Orange-400 (نارنجی)
        cheekR: 'rgba(168, 85, 247, 0.75)',       // Purple-400 (بنفش)
        cheekL: 'rgba(59, 130, 246, 0.75)',       // Blue-500 (آبی)
      };
      
      const lineWidth = 2.5;
      const dashPattern = [10, 5]; // خط چین مدرن با فاصله بیشتر
      
      // 1. خط عمودی میانی (Midfacial Line) - در وسط تصویر
      // محاسبه خط میانی از نقاط کلیدی: Glabella, Subnasale, Pogonion
      if (glabella && subnasale && pogonion) {
        // محاسبه x میانی از سه نقطه کلیدی
        const midlineX = (glabella.x + subnasale.x + pogonion.x) / 3 * currentScale + x;
        
        ctx.strokeStyle = colors.midline;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([]); // خط صاف برای خط میانی
        ctx.beginPath();
        ctx.moveTo(midlineX, topY - 20); // کمی بالاتر از بالای صورت
        ctx.lineTo(midlineX, bottomY + 20); // کمی پایین‌تر از پایین صورت
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors.midline;
        ctx.font = 'bold 12px Arial';
        ctx.fillText('Midline', midlineX + 8, topY - 10);
      } else if (glabella && pogonion) {
        // Fallback: اگر subnasale پیدا نشد
        const midlineX = (glabella.x + pogonion.x) / 2 * currentScale + x;
        ctx.strokeStyle = colors.midline;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(midlineX, topY - 20);
        ctx.lineTo(midlineX, bottomY + 20);
        ctx.stroke();
        
        ctx.fillStyle = colors.midline;
        ctx.font = 'bold 12px Arial';
        ctx.fillText('Midline', midlineX + 8, topY - 10);
      }
      
      // 2. خط عمودی از Eye-R-Inner (باید دقیقاً از این نقطه عبور کند)
      // اگر findLandmark نتوانست پیدا کند، مستقیماً از index استفاده می‌کنیم
      let rightEyeInnerFinal = rightEyeInner;
      
      // تشخیص نوع مدل برای استفاده از index صحیح
      const landmarkCount = landmarks.length;
      const isDlib = landmarkCount === 68;
      const isMediaPipe = landmarkCount >= 400;
      
      if (!rightEyeInnerFinal && landmarks.some(l => l.name && l.name.startsWith('landmark_'))) {
        // برای dlib: index 39
        // برای MediaPipe: index 362
        const targetIndex = isMediaPipe ? 362 : (isDlib ? 39 : null);
        
        if (targetIndex !== null) {
          rightEyeInnerFinal = landmarks.find(l => {
            const idx = l.index !== undefined ? 
                       (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                       (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
            return idx === targetIndex;
          });
          
          if (rightEyeInnerFinal) {
            console.log(`[Landmark Visualizer] Found Right Eye Inner by direct index search (${isMediaPipe ? 'MediaPipe' : 'dlib'} index ${targetIndex}):`, rightEyeInnerFinal);
          } else {
            console.warn(`[Landmark Visualizer] Right Eye Inner not found at index ${targetIndex} (${isMediaPipe ? 'MediaPipe' : 'dlib'})`);
          }
        }
      }
      
      if (rightEyeInnerFinal) {
        // محاسبه مختصات canvas برای Eye-R-Inner
        // مختصات تصویر اصلی را به مختصات canvas تبدیل می‌کنیم
        const eyeRX = rightEyeInnerFinal.x * currentScale + x;
        const eyeRY = rightEyeInnerFinal.y * currentScale + y;
        
        console.log('[Landmark Visualizer] Drawing Eye-R-Inner vertical line:', {
          landmark: { x: rightEyeInnerFinal.x, y: rightEyeInnerFinal.y, name: rightEyeInnerFinal.name, index: rightEyeInnerFinal.index },
          canvas: { x: eyeRX, y: eyeRY },
          scale: currentScale,
          offset: { x, y },
        });
        
        // رسم خط عمودی که دقیقاً از Eye-R-Inner عبور می‌کند
        ctx.strokeStyle = colors.eyeR;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        // خط از بالا تا پایین، دقیقاً از مختصات x مربوط به Eye-R-Inner
        ctx.moveTo(eyeRX, topY - 20);
        ctx.lineTo(eyeRX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // رسم یک دایره کوچک روی landmark برای نشان دادن نقطه دقیق عبور
        ctx.fillStyle = colors.eyeR;
        ctx.beginPath();
        ctx.arc(eyeRX, eyeRY, 5, 0, 2 * Math.PI);
        ctx.fill();
        // Border سفید برای دایره
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors.eyeR;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Eye-R', eyeRX + 8, eyeRY - 15);
      } else {
        console.error('[Landmark Visualizer] Right Eye Inner NOT FOUND!', {
          landmarksCount: landmarks.length,
          sampleLandmarks: landmarks.slice(35, 45).map(l => ({ 
            name: l.name, 
            index: l.index,
            x: l.x,
            y: l.y
          })),
        });
      }
      
      // 3. خط عمودی از Eye-L-Inner (باید دقیقاً از این نقطه عبور کند)
      // اگر findLandmark نتوانست پیدا کند، مستقیماً از index استفاده می‌کنیم
      let leftEyeInnerFinal = leftEyeInner;
      
      if (!leftEyeInnerFinal && landmarks.some(l => l.name && l.name.startsWith('landmark_'))) {
        // برای dlib: index 42
        // برای MediaPipe: index 133
        const targetIndex = isMediaPipe ? 133 : (isDlib ? 42 : null);
        
        if (targetIndex !== null) {
          leftEyeInnerFinal = landmarks.find(l => {
            const idx = l.index !== undefined ? 
                       (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                       (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
            return idx === targetIndex;
          });
          
          if (leftEyeInnerFinal) {
            console.log(`[Landmark Visualizer] Found Left Eye Inner by direct index search (${isMediaPipe ? 'MediaPipe' : 'dlib'} index ${targetIndex}):`, leftEyeInnerFinal);
          } else {
            console.warn(`[Landmark Visualizer] Left Eye Inner not found at index ${targetIndex} (${isMediaPipe ? 'MediaPipe' : 'dlib'})`);
          }
        }
      }
      
      if (leftEyeInnerFinal) {
        // محاسبه مختصات canvas برای Eye-L-Inner
        const eyeLX = leftEyeInnerFinal.x * currentScale + x;
        const eyeLY = leftEyeInnerFinal.y * currentScale + y;
        
        console.log('[Landmark Visualizer] Drawing Eye-L-Inner vertical line:', {
          landmark: { x: leftEyeInnerFinal.x, y: leftEyeInnerFinal.y, name: leftEyeInnerFinal.name, index: leftEyeInnerFinal.index },
          canvas: { x: eyeLX, y: eyeLY },
          scale: currentScale,
          offset: { x, y },
        });
        
        // رسم خط عمودی که دقیقاً از Eye-L-Inner عبور می‌کند
        ctx.strokeStyle = colors.eyeL;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        // خط از بالا تا پایین، دقیقاً از مختصات x مربوط به Eye-L-Inner
        ctx.moveTo(eyeLX, topY - 20);
        ctx.lineTo(eyeLX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // رسم یک دایره کوچک روی landmark برای نشان دادن نقطه دقیق عبور
        ctx.fillStyle = colors.eyeL;
        ctx.beginPath();
        ctx.arc(eyeLX, eyeLY, 5, 0, 2 * Math.PI);
        ctx.fill();
        // Border سفید برای دایره
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors.eyeL;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Eye-L', eyeLX + 8, eyeLY - 15);
      } else {
        console.error('[Landmark Visualizer] Left Eye Inner NOT FOUND!', {
          landmarksCount: landmarks.length,
          sampleLandmarks: landmarks.slice(40, 50).map(l => ({ 
            name: l.name, 
            index: l.index,
            x: l.x,
            y: l.y
          })),
        });
      }
      
      // 4. خط عمودی به موازات Jaw-Right
      if (jawRight) {
        const jawRX = jawRight.x * currentScale + x;
        ctx.strokeStyle = colors.jawR;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        ctx.moveTo(jawRX, topY - 20);
        ctx.lineTo(jawRX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // Label
        ctx.fillStyle = colors.jawR;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Jaw-R', jawRX + 8, bottomY + 15);
      }
      
      // 5. خط عمودی به موازات Jaw-Left
      if (jawLeft) {
        const jawLX = jawLeft.x * currentScale + x;
        ctx.strokeStyle = colors.jawL;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        ctx.moveTo(jawLX, topY - 20);
        ctx.lineTo(jawLX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // Label
        ctx.fillStyle = colors.jawL;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Jaw-L', jawLX + 8, bottomY + 15);
      }
      
      // 6. خط عمودی از Cheek-R (گونه راست) - فقط برای MediaPipe
      // برای MediaPipe: Cheek-R = index 345
      let cheekRight = null;
      if (isMediaPipe) {
        cheekRight = landmarks.find(l => {
          const idx = l.index !== undefined ? 
                     (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                     (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
          return idx === 345;
        });
      } else {
        // برای dlib و سایر مدل‌ها: از findLandmark استفاده می‌کنیم
        cheekRight = findLandmark(['right_cheek', 'cheek_right', 'face_right']);
      }
      
      if (cheekRight) {
        const cheekRX = cheekRight.x * currentScale + x;
        const cheekRY = cheekRight.y * currentScale + y;
        
        console.log('[Landmark Visualizer] Drawing Cheek-R vertical line:', {
          landmark: { x: cheekRight.x, y: cheekRight.y, name: cheekRight.name, index: cheekRight.index },
          canvas: { x: cheekRX, y: cheekRY },
        });
        
        ctx.strokeStyle = colors.cheekR;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        ctx.moveTo(cheekRX, topY - 20);
        ctx.lineTo(cheekRX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // رسم یک دایره کوچک روی landmark
        ctx.fillStyle = colors.cheekR;
        ctx.beginPath();
        ctx.arc(cheekRX, cheekRY, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors.cheekR;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Cheek-R', cheekRX + 8, cheekRY - 15);
      }
      
      // 7. خط عمودی از Cheek-L (گونه چپ) - فقط برای MediaPipe
      // برای MediaPipe: Cheek-L = index 116
      let cheekLeft = null;
      if (isMediaPipe) {
        cheekLeft = landmarks.find(l => {
          const idx = l.index !== undefined ? 
                     (typeof l.index === 'number' ? l.index : parseInt(l.index, 10)) :
                     (l.name ? parseInt(l.name.match(/landmark_(\d+)/)?.[1] || '-1', 10) : -1);
          return idx === 116;
        });
      } else {
        // برای dlib و سایر مدل‌ها: از findLandmark استفاده می‌کنیم
        cheekLeft = findLandmark(['left_cheek', 'cheek_left', 'face_left']);
      }
      
      if (cheekLeft) {
        const cheekLX = cheekLeft.x * currentScale + x;
        const cheekLY = cheekLeft.y * currentScale + y;
        
        console.log('[Landmark Visualizer] Drawing Cheek-L vertical line:', {
          landmark: { x: cheekLeft.x, y: cheekLeft.y, name: cheekLeft.name, index: cheekLeft.index },
          canvas: { x: cheekLX, y: cheekLY },
        });
        
        ctx.strokeStyle = colors.cheekL;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash(dashPattern);
        ctx.beginPath();
        ctx.moveTo(cheekLX, topY - 20);
        ctx.lineTo(cheekLX, bottomY + 20);
        ctx.stroke();
        ctx.setLineDash([]); // Reset
        
        // رسم یک دایره کوچک روی landmark
        ctx.fillStyle = colors.cheekL;
        ctx.beginPath();
        ctx.arc(cheekLX, cheekLY, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.lineWidth = 1.5;
        ctx.stroke();
        
        // Label
        ctx.fillStyle = colors.cheekL;
        ctx.font = 'bold 11px Arial';
        ctx.fillText('Cheek-L', cheekLX + 8, cheekLY - 15);
      }
      
      // خطوط افقی (Interpupillary Line و Commissural Line) - با رنگ‌های مدرن
      const leftPupil = findLandmark(['left_pupil', 'left_eye']);
      const rightPupil = findLandmark(['right_pupil', 'right_eye']);
      const mouthLeft = findLandmark(['mouth_left', 'mouth_outer_1']);
      const mouthRight = findLandmark(['mouth_right', 'mouth_outer_5']);
      
      // Interpupillary Line (خط بین مردمک‌ها)
      if (leftPupil && rightPupil) {
        ctx.strokeStyle = 'rgba(245, 101, 101, 0.8)'; // Red-400 (قرمز) - متفاوت از Cheek
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(leftPupil.x * currentScale + x, leftPupil.y * currentScale + y);
        ctx.lineTo(rightPupil.x * currentScale + x, rightPupil.y * currentScale + y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Commissural Line (خط بین گوشه‌های دهان)
      if (mouthLeft && mouthRight) {
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)'; // Amber-400 (زرد کهربایی) - متفاوت از Cheek
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.moveTo(mouthLeft.x * currentScale + x, mouthLeft.y * currentScale + y);
        ctx.lineTo(mouthRight.x * currentScale + x, mouthRight.y * currentScale + y);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }

    // رسم outline ساختارهای مهم
    if (localShowOutlines) {
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.8)';
      ctx.lineWidth = 2;
      ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';

      // Outline صورت
      const leftFace = findLandmark(['chin_left', 'face_left', 'chin_1']);
      const rightFace = findLandmark(['chin_right', 'face_right', 'chin_17']);
      const forehead = findLandmark(['nose_bridge_top', 'forehead']);
      const chin = findLandmark(['chin_center', 'chin', 'chin_8']);

      if (leftFace && rightFace && forehead && chin) {
        const points = [
          { x: leftFace.x * currentScale + x, y: forehead.y * currentScale + y },
          { x: rightFace.x * currentScale + x, y: forehead.y * currentScale + y },
          { x: rightFace.x * currentScale + x, y: chin.y * currentScale + y },
          { x: leftFace.x * currentScale + x, y: chin.y * currentScale + y },
        ];

        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 1; i < points.length; i++) {
          ctx.lineTo(points[i].x, points[i].y);
        }
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Outline چشم‌ها
      const leftEyeOuter = findLandmark(['left_eye_outer', 'left_eye_1']);
      const leftEyeInner = findLandmark(['left_eye_inner', 'left_eye_2']);
      const leftEyeTop = findLandmark(['left_eye_top']);
      const leftEyeBottom = findLandmark(['left_eye_bottom']);

      if (leftEyeOuter && leftEyeInner && leftEyeTop && leftEyeBottom) {
        ctx.beginPath();
        ctx.moveTo(leftEyeOuter.x * currentScale + x, leftEyeTop.y * currentScale + y);
        ctx.lineTo(leftEyeInner.x * currentScale + x, leftEyeTop.y * currentScale + y);
        ctx.lineTo(leftEyeInner.x * currentScale + x, leftEyeBottom.y * currentScale + y);
        ctx.lineTo(leftEyeOuter.x * currentScale + x, leftEyeBottom.y * currentScale + y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      const rightEyeOuter = findLandmark(['right_eye_outer', 'right_eye_1']);
      const rightEyeInner = findLandmark(['right_eye_inner', 'right_eye_2']);
      const rightEyeTop = findLandmark(['right_eye_top']);
      const rightEyeBottom = findLandmark(['right_eye_bottom']);

      if (rightEyeOuter && rightEyeInner && rightEyeTop && rightEyeBottom) {
        ctx.beginPath();
        ctx.moveTo(rightEyeOuter.x * currentScale + x, rightEyeTop.y * currentScale + y);
        ctx.lineTo(rightEyeInner.x * currentScale + x, rightEyeTop.y * currentScale + y);
        ctx.lineTo(rightEyeInner.x * currentScale + x, rightEyeBottom.y * currentScale + y);
        ctx.lineTo(rightEyeOuter.x * currentScale + x, rightEyeBottom.y * currentScale + y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Outline بینی
      const noseTip = findLandmark(['nose_tip', 'nose']);
      const noseLeft = findLandmark(['nose_left', 'nose_tip_2']);
      const noseRight = findLandmark(['nose_right', 'nose_tip_4']);
      const noseBridgeTop = findLandmark(['nose_bridge_top', 'nose_bridge_1']);

      if (noseTip && noseLeft && noseRight && noseBridgeTop) {
        ctx.beginPath();
        ctx.moveTo(noseBridgeTop.x * currentScale + x, noseBridgeTop.y * currentScale + y);
        ctx.lineTo(noseLeft.x * currentScale + x, noseTip.y * currentScale + y);
        ctx.lineTo(noseRight.x * currentScale + x, noseTip.y * currentScale + y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }

      // Outline دهان
      const mouthLeft = findLandmark(['mouth_left', 'mouth_left_corner']);
      const mouthRight = findLandmark(['mouth_right', 'mouth_right_corner']);
      const mouthTop = findLandmark(['mouth_top', 'mouth_center_top']);
      const mouthBottom = findLandmark(['mouth_bottom', 'mouth_center_bottom']);

      if (mouthLeft && mouthRight && mouthTop && mouthBottom) {
        ctx.beginPath();
        ctx.moveTo(mouthLeft.x * currentScale + x, mouthTop.y * currentScale + y);
        ctx.lineTo(mouthRight.x * currentScale + x, mouthTop.y * currentScale + y);
        ctx.lineTo(mouthRight.x * currentScale + x, mouthBottom.y * currentScale + y);
        ctx.lineTo(mouthLeft.x * currentScale + x, mouthBottom.y * currentScale + y);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
      }
    }

    // رسم لندمارک‌ها
    if (localShowLandmarks) {
      landmarks.forEach((landmark) => {
        const { x: landmarkX, y: landmarkY, name, index } = landmark;
        
        const canvasX = landmarkX * currentScale + x;
        const canvasY = landmarkY * currentScale + y;

        // اندازه نقطه بر اساس تعداد کل لندمارک‌ها (برای تعداد زیاد، نقطه کوچک‌تر)
        const pointSize = landmarks.length > 100 ? 2 : 3;

        // رسم نقطه سفید با opacity 0.8
        ctx.beginPath();
        ctx.arc(canvasX, canvasY, pointSize, 0, 2 * Math.PI);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
        ctx.fill();
        
        // رسم border برای نقطه
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.lineWidth = 1;
        ctx.stroke();

        // نمایش نام لندمارک‌ها
        // برای dlib (68 نقطه): همه نام‌ها نمایش داده می‌شوند
        // برای MediaPipe (468 نقطه): فقط لندمارک‌های مهم نمایش داده می‌شوند
        if (localShowLandmarkNames && name) {
          // تشخیص نوع مدل
          const landmarkCount = landmarks.length;
          const isDlib = landmarkCount === 68;
          const isMediaPipe = landmarkCount >= 400;
          
          // Debug log برای MediaPipe (فقط برای چند لندمارک نمونه)
          if (isMediaPipe && index !== undefined) {
            const indexNum = typeof index === 'number' ? index : parseInt(index, 10);
            if ((indexNum === 1 || indexNum === 9 || indexNum === 175 || indexNum === 133) && !isNaN(indexNum)) {
              console.log('[Landmark Visualizer] MediaPipe landmark check:', {
                originalIndex: index,
                indexType: typeof index,
                parsedIndex: indexNum,
                name,
                hasInMapping: MEDIAPIPE_LANDMARK_NAMES[indexNum] !== undefined,
                mappingValue: MEDIAPIPE_LANDMARK_NAMES[indexNum] || null,
                landmarkCount,
              });
            }
          }
          
          // برای dlib: همه نام‌ها را نمایش بده
          // برای MediaPipe: فقط لندمارک‌های مهم را نمایش بده (موجود در MEDIAPIPE_LANDMARK_NAMES)
          let shouldShowName = false;
          if (isDlib) {
            shouldShowName = true; // همه لندمارک‌های dlib
          } else if (isMediaPipe) {
            // فقط لندمارک‌های مهم MediaPipe
            // استخراج index از index property یا name
            let landmarkIndex = null;
            if (index !== undefined && index !== null) {
              // اگر index موجود است، از آن استفاده کن (ممکن است string یا number باشد)
              landmarkIndex = typeof index === 'number' ? index : parseInt(index, 10);
            } else if (name) {
              // اگر index موجود نیست، از name استخراج کن
              const indexMatch = name.match(/landmark_(\d+)/);
              if (indexMatch) {
                landmarkIndex = parseInt(indexMatch[1], 10);
              }
            }
            
            // بررسی اینکه آیا این index در MEDIAPIPE_LANDMARK_NAMES موجود است
            if (landmarkIndex !== null && !isNaN(landmarkIndex)) {
              shouldShowName = MEDIAPIPE_LANDMARK_NAMES[landmarkIndex] !== undefined;
              
              // Debug log برای لندمارک‌های مهم
              if (shouldShowName) {
                console.log(`[Landmark Visualizer] MediaPipe landmark ${landmarkIndex} (${name}) will be shown as: ${MEDIAPIPE_LANDMARK_NAMES[landmarkIndex]}`);
              }
            }
          } else {
            // برای مدل‌های دیگر (اگر کمتر از 100 لندمارک دارند)
            shouldShowName = landmarkCount <= 100;
          }
          
          if (shouldShowName) {
            // تعیین نام نمایشی
            let displayName = null;
            
            // اگر index موجود است، از mapping استفاده کن
            if (index !== undefined && index !== null) {
              // تبدیل index به number اگر string باشد
              const indexNum = typeof index === 'number' ? index : parseInt(index, 10);
              if (!isNaN(indexNum)) {
                if (isDlib) {
                  displayName = DLIB_LANDMARK_NAMES[indexNum];
                } else if (isMediaPipe) {
                  displayName = MEDIAPIPE_LANDMARK_NAMES[indexNum];
                }
              }
            }
            
            // اگر از mapping چیزی پیدا نشد، از name استفاده کن
            if (!displayName) {
              // اگر نام به صورت landmark_X است، از mapping استفاده کن
              const indexMatch = name.match(/landmark_(\d+)/);
              if (indexMatch) {
                const landmarkIndex = parseInt(indexMatch[1], 10);
                if (isDlib) {
                  displayName = DLIB_LANDMARK_NAMES[landmarkIndex];
                } else if (isMediaPipe) {
                  displayName = MEDIAPIPE_LANDMARK_NAMES[landmarkIndex];
                }
                // اگر در mapping نیست، از index استفاده کن (فقط برای dlib)
                if (!displayName && isDlib) {
                  displayName = `${landmarkIndex}`;
                }
              } else if (name.includes('_')) {
                // تبدیل snake_case به Title Case
                displayName = name
                  .split('_')
                  .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                  .join(' ');
              } else {
                // اگر نام به صورت دیگری است، همان نام را استفاده کن
                displayName = name;
              }
            }

            // نمایش نام
            if (displayName) {
              // تنظیمات فونت - استفاده از فونت کوچک‌تر برای لندمارک‌های زیاد
              const fontSize = landmarks.length > 50 ? 9 : 11;
              ctx.font = `bold ${fontSize}px Arial`;
              ctx.textAlign = 'left';
              ctx.textBaseline = 'middle';
              
              // محاسبه عرض متن
              const textMetrics = ctx.measureText(displayName);
              const textWidth = textMetrics.width;
              const textHeight = fontSize + 2;
              
              // پس‌زمینه برای متن (برای خوانایی بهتر)
              const padding = 3;
              // قرار دادن متن در سمت راست نقطه (برای RTL)
              const bgX = canvasX + pointSize + 4;
              const bgY = canvasY;
              
              // رسم پس‌زمینه با گوشه‌های گرد (تقریبی)
              ctx.fillStyle = 'rgba(0, 0, 0, 0.75)';
              ctx.fillRect(
                bgX - padding,
                bgY - textHeight / 2 - padding,
                textWidth + padding * 2,
                textHeight + padding * 2
              );
              
              // رسم متن
              ctx.fillStyle = 'rgba(255, 255, 255, 1)';
              ctx.fillText(displayName, bgX, bgY);
            }
          }
        }
      });
    }
  }, [landmarks, baseCanvasSize, zoom, pan, localShowLandmarks, localShowOutlines, localShowProfileLines, localShowFrontalLines, localShowLandmarkNames, imageLoaded, imageSize, loadError, isRetrying]);

  // Mouse event handlers
  const handleMouseDown = (e) => {
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    if (!coords) return;
    
    setIsPanning(true);
    setPanStart({ x: coords.x - pan.x, y: coords.y - pan.y });
  };
  
  const handleMouseMove = (e) => {
    const coords = getCanvasCoordinates(e.clientX, e.clientY);
    if (!coords) return;
    
    if (isPanning) {
      setPan({
        x: coords.x - panStart.x,
        y: coords.y - panStart.y,
      });
    }
  };
  
  const handleMouseUp = () => {
    setIsPanning(false);
  };
  
  const handleMouseLeave = () => {
    setIsPanning(false);
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
    <Box sx={{ width: '100%', maxWidth: '100%' }}>
      <Stack spacing={2}>
        {/* Toolbar */}
        <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between" sx={{ flexWrap: 'wrap', gap: 1 }}>
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
            {/* Manual retry button for failed loads */}
            {loadError && (
              <Tooltip title="تلاش مجدد برای بارگذاری تصویر">
                <IconButton 
                  size="small" 
                  onClick={retryImageLoad}
                  disabled={isRetrying}
                  color="error"
                >
                  <Iconify icon="carbon:retry" />
                </IconButton>
              </Tooltip>
            )}
          </Stack>
          <Stack direction="row" spacing={1} alignItems="center">
            <Tooltip title={localShowLandmarkNames ? "مخفی کردن نام لندمارک‌ها" : "نمایش نام لندمارک‌ها"}>
              <IconButton
                size="small"
                onClick={() => {
                  setLocalShowLandmarkNames(!localShowLandmarkNames);
                  if (onShowLandmarkNamesChange) {
                    onShowLandmarkNamesChange(!localShowLandmarkNames);
                  }
                }}
                color={localShowLandmarkNames ? "primary" : "default"}
              >
                <Iconify icon={localShowLandmarkNames ? "solar:tag-bold" : "solar:tag-linear"} />
              </IconButton>
            </Tooltip>
          </Stack>
        </Stack>

        {/* Canvas */}
        <Box
          ref={canvasContainerRef}
          sx={{
            position: 'relative',
            width: '100%',
            overflow: 'hidden',
            minHeight: 400,
          }}
        >
          <Box
            component="canvas"
            ref={canvasRef}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseLeave}
            sx={{
              display: 'block',
              width: '100%',
              height: '100%',
              cursor: isPanning ? 'grabbing' : 'grab',
            }}
          />
        </Box>

        {/* Error Display */}
        {loadError && (
          <Typography variant="body2" sx={{ color: 'error.main', textAlign: 'center', py: 1 }}>
            خطا در بارگذاری تصویر: {loadError}
            {isRetrying && ' (در حال تلاش مجدد...)'}
            {retryCount > 0 && !isRetrying && ` (تعداد تلاش: ${retryCount})`}
          </Typography>
        )}

        {/* Info */}
        {landmarks.length === 0 && !loadError && (
          <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', py: 2 }}>
            No landmarks to display
          </Typography>
        )}
      </Stack>
    </Box>
  );
}

LandmarkVisualizer.propTypes = {
  imageUrl: PropTypes.string,
  landmarks: PropTypes.arrayOf(
    PropTypes.shape({
      x: PropTypes.number.isRequired,
      y: PropTypes.number.isRequired,
      name: PropTypes.string,
    })
  ),
  showLandmarks: PropTypes.bool,
  onShowLandmarksChange: PropTypes.func,
  showOutlines: PropTypes.bool,
  onShowOutlinesChange: PropTypes.func,
  showProfileLines: PropTypes.bool,
  onShowProfileLinesChange: PropTypes.func,
  showFrontalLines: PropTypes.bool,
  onShowFrontalLinesChange: PropTypes.func,
  showLandmarkNames: PropTypes.bool,
  onShowLandmarkNamesChange: PropTypes.func,
  onImageError: PropTypes.func,
  onImageLoad: PropTypes.func,
};
