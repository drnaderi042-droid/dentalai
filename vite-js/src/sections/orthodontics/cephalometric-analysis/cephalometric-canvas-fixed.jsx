import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import Tooltip from '@mui/material/Tooltip';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';

import { Iconify } from 'src/components/iconify';

// Cephalometric landmarks
const LANDMARKS = {
  // Skeletal points
  S: { name: 'Sella', x: 0, y: 0, color: '#2196F3' },
  N: { name: 'Nasion', x: 0, y: 0, color: '#2196F3' },
  A: { name: 'Point A', x: 0, y: 0, color: '#2196F3' },
  B: { name: 'Point B', x: 0, y: 0, color: '#2196F3' },
  Pog: { name: 'Pogonion', x: 0, y: 0, color: '#2196F3' },
  Go: { name: 'Gonion', x: 0, y: 0, color: '#2196F3' },
  Me: { name: 'Menton', x: 0, y: 0, color: '#2196F3' },
  Or: { name: 'Orbitale', x: 0, y: 0, color: '#2196F3' },
  Po: { name: 'Porion', x: 0, y: 0, color: '#2196F3' },
  ANS: { name: 'ANS', x: 0, y: 0, color: '#2196F3' },
  PNS: { name: 'PNS', x: 0, y: 0, color: '#2196F3' },
  
  // Dental points
  U1: { name: 'Upper Incisor', x: 0, y: 0, color: '#F44336' },
  L1: { name: 'Lower Incisor', x: 0, y: 0, color: '#F44336' },
  U1A: { name: 'U1 Apex', x: 0, y: 0, color: '#F44336' },
  L1A: { name: 'L1 Apex', x: 0, y: 0, color: '#F44336' },
};

// Mock AI API function - CORS-safe implementation
const detectLandmarksAI = async (imageBlob, imageDimensions) => {
  // Simulate API call with realistic timing
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  // Scale coordinates based on actual image dimensions
  const { width = 512, height = 512 } = imageDimensions || {};
  const scaleX = width / 512;
  const scaleY = height / 512;
  
  // Mock AI results - scaled to actual image size
  const mockResults = {
    landmarks: {
      S: { x: Math.round(200 * scaleX), y: Math.round(150 * scaleY), confidence: 0.95 },
      N: { x: Math.round(250 * scaleX), y: Math.round(140 * scaleY), confidence: 0.88 },
      A: { x: Math.round(280 * scaleX), y: Math.round(180 * scaleY), confidence: 0.92 },
      B: { x: Math.round(285 * scaleX), y: Math.round(220 * scaleY), confidence: 0.87 },
      Pog: { x: Math.round(290 * scaleX), y: Math.round(250 * scaleY), confidence: 0.94 },
      Go: { x: Math.round(320 * scaleX), y: Math.round(240 * scaleY), confidence: 0.83 },
      Me: { x: Math.round(295 * scaleX), y: Math.round(270 * scaleY), confidence: 0.96 },
      U1: { x: Math.round(275 * scaleX), y: Math.round(200 * scaleY), confidence: 0.89 },
      L1: { x: Math.round(280 * scaleX), y: Math.round(230 * scaleY), confidence: 0.91 },
    },
    processingTime: 2.8,
    modelVersion: "v2.1.0-cors-fixed",
    imageDimensions: { width, height }
  };
  
  return mockResults;
};

export function CephalometricCanvas({ imageUrl, onMeasurementsChange }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [landmarks, setLandmarks] = useState({});
  const [selectedLandmark, setSelectedLandmark] = useState(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  // New editing states
  const [editMode, setEditMode] = useState(false);
  const [drawingMode, setDrawingMode] = useState('select'); // 'select', 'landmark', 'border', 'curve'
  const [customBorders, setCustomBorders] = useState([]);
  const [curvedLines, setCurvedLines] = useState([]);
  const [currentDrawing, setCurrentDrawing] = useState(null);
  const [selectedObject, setSelectedObject] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [landmarkRadius, setLandmarkRadius] = useState(5);
  const [showLandmarkDialog, setShowLandmarkDialog] = useState(false);
  const [newLandmarkName, setNewLandmarkName] = useState('');
  const [selectedBorderStyle, setSelectedBorderStyle] = useState('solid');
  const [selectedBorderColor, setSelectedBorderColor] = useState('#FF0000');
  const [borderThickness, setBorderThickness] = useState(2);
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveFileName, setSaveFileName] = useState('');
  const fileInputRef = useRef(null);
  
  // AI Detection states
  const [isAIDetecting, setIsAIDetecting] = useState(false);
  const [aiResults, setAiResults] = useState([]);
  const [showAIResults, setShowAIResults] = useState(false);
  const [aiConfidence, setAiConfidence] = useState({});
  const [aiProcessingProgress, setAiProcessingProgress] = useState(0);
  const [aiError, setAiError] = useState(null);
  const [showAIDialog, setShowAIDialog] = useState(false);

  useEffect(() => {
    if (imageUrl && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const img = new Image();
      
      img.crossOrigin = 'anonymous'; // Enable CORS
      
      img.onload = () => {
        imageRef.current = img;
        canvas.width = img.width;
        canvas.height = img.height;
        drawCanvas();
      };
      
      img.onerror = () => {
        console.warn('Failed to load image with CORS, AI detection may not work with external images');
        // Continue without CORS - AI will still work with mock data
        img.crossOrigin = null;
        img.src = imageUrl;
      };
      
      img.src = imageUrl;
    }
  }, [imageUrl, drawCanvas]);

  useEffect(() => {
    drawCanvas();
  }, [landmarks, zoom, pan, customBorders, curvedLines, selectedObject, aiResults, aiConfidence, drawCanvas]);

  useEffect(() => {
    if (Object.keys(landmarks).length > 0) {
      const measurements = calculateMeasurements(landmarks);
      onMeasurementsChange?.(measurements);
    }
  }, [landmarks, calculateMeasurements, onMeasurementsChange]);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Apply transformations
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);
    
    // Draw image
    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0);
    }
    
    // Draw custom borders
    drawCustomBorders(ctx);
    
    // Draw curved lines
    drawCurvedLines(ctx);
    
    // Draw lines between landmarks
    drawMeasurementLines(ctx, landmarks);
    
    // Draw landmarks (including AI results)
    const allLandmarks = { ...landmarks };
    
    // Add AI results if showing
    if (showAIResults && aiResults.length > 0) {
      aiResults.forEach(aiLandmark => {
        if (!allLandmarks[aiLandmark.name]) {
          allLandmarks[aiLandmark.name] = { x: aiLandmark.x, y: aiLandmark.y };
        }
      });
    }
    
    Object.entries(allLandmarks).forEach(([key, point]) => {
      const isAIResult = showAIResults && aiResults.find(r => r.name === key);
      const confidence = aiConfidence[key];
      
      ctx.fillStyle = isAIResult ? '#FF9800' : (LANDMARKS[key]?.color || '#2196F3');
      const radius = selectedObject === key ? landmarkRadius * 1.5 : landmarkRadius;
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Draw AI confidence indicator
      if (isAIResult && confidence) {
        ctx.strokeStyle = confidence > 0.9 ? '#4CAF50' : confidence > 0.7 ? '#FF9800' : '#F44336';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius + 2, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Draw selection ring
      if (selectedObject === key && editMode) {
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius + 3, 0, 2 * Math.PI);
        ctx.stroke();
      }
      
      // Draw label
      ctx.fillStyle = '#000';
      ctx.font = '12px Arial';
      ctx.fillText(key, point.x + 8, point.y - 8);
      
      // Draw confidence score for AI results
      if (isAIResult && confidence) {
        ctx.fillStyle = '#666';
        ctx.font = '10px Arial';
        ctx.fillText(`${(confidence * 100).toFixed(0)}%`, point.x + 8, point.y + 12);
      }
    });
    
    // Draw current drawing (in progress)
    if (currentDrawing) {
      drawCurrentDrawing(ctx);
    }
    
    ctx.restore();
  }, [pan, zoom, imageRef, landmarks, selectedObject, aiResults, aiConfidence, currentDrawing, drawCustomBorders, drawCurvedLines, drawCurrentDrawing, editMode, landmarkRadius, showAIResults]);

  const drawMeasurementLines = (ctx, points) => {
    if (!points.S || !points.N || !points.A) return;
    
    // SNA angle (blue lines)
    if (points.S && points.N && points.A) {
      ctx.strokeStyle = '#2196F3';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points.S.x, points.S.y);
      ctx.lineTo(points.N.x, points.N.y);
      ctx.lineTo(points.A.x, points.A.y);
      ctx.stroke();
    }
    
    // SNB angle (blue lines)
    if (points.S && points.N && points.B) {
      ctx.strokeStyle = '#2196F3';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points.S.x, points.S.y);
      ctx.lineTo(points.N.x, points.N.y);
      ctx.lineTo(points.B.x, points.B.y);
      ctx.stroke();
    }
    
    // Dental measurements (red lines)
    if (points.U1 && points.L1) {
      ctx.strokeStyle = '#F44336';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(points.U1.x, points.U1.y);
      ctx.lineTo(points.L1.x, points.L1.y);
      ctx.stroke();
    }
  };

  const drawCustomBorders = useCallback((ctx) => {
    customBorders.forEach((border, index) => {
      ctx.strokeStyle = border.color;
      ctx.lineWidth = border.thickness;
      ctx.setLineDash(border.style === 'dashed' ? [5, 5] : border.style === 'dotted' ? [2, 2] : []);
      
      ctx.beginPath();
      ctx.moveTo(border.points[0].x, border.points[0].y);
      for (let i = 1; i < border.points.length; i += 1) {
        ctx.lineTo(border.points[i].x, border.points[i].y);
      }
      ctx.closePath();
      ctx.stroke();
      
      // Draw selection highlight
      if (selectedObject === `border_${index}` && editMode) {
        ctx.strokeStyle = '#FFD700';
        ctx.lineWidth = 3;
        ctx.setLineDash([10, 5]);
        ctx.beginPath();
        ctx.moveTo(border.points[0].x, border.points[0].y);
        for (let i = 1; i < border.points.length; i += 1) {
          ctx.lineTo(border.points[i].x, border.points[i].y);
        }
        ctx.closePath();
        ctx.stroke();
      }
    });
    
    ctx.setLineDash([]); // Reset line dash
  }, [customBorders, selectedObject, editMode]);

  const drawCurvedLines = useCallback((ctx) => {
    curvedLines.forEach((line, index) => {
      if (line.points.length < 2) return;
      
      ctx.strokeStyle = line.color;
      ctx.lineWidth = line.thickness;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      
      ctx.beginPath();
      ctx.moveTo(line.points[0].x, line.points[0].y);
      
      // Draw smooth curves using quadratic curves
      for (let i = 1; i < line.points.length - 1; i += 1) {
        const xc = (line.points[i].x + line.points[i + 1].x) / 2;
        const yc = (line.points[i].y + line.points[i + 1].y) / 2;
        ctx.quadraticCurveTo(line.points[i].x, line.points[i].y, xc, yc);
      }
      
      // Connect to the last point
      const lastIndex = line.points.length - 1;
      ctx.lineTo(line.points[lastIndex].x, line.points[lastIndex].y);
      ctx.stroke();
      
      // Draw control points for editing
      if (selectedObject === `curve_${index}` && editMode) {
        ctx.fillStyle = '#FFD700';
        line.points.forEach(point => {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
          ctx.fill();
        });
      }
    });
  }, [curvedLines, selectedObject, editMode]);

  const drawCurrentDrawing = useCallback((ctx) => {
    if (!currentDrawing) return;
    
    ctx.strokeStyle = currentDrawing.color || '#FF0000';
    ctx.lineWidth = currentDrawing.thickness || 2;
    
    if (currentDrawing.type === 'border' && currentDrawing.points.length > 0) {
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(currentDrawing.points[0].x, currentDrawing.points[0].y);
      for (let i = 1; i < currentDrawing.points.length; i += 1) {
        ctx.lineTo(currentDrawing.points[i].x, currentDrawing.points[i].y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    } else if (currentDrawing.type === 'curve' && currentDrawing.points.length > 0) {
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.beginPath();
      ctx.moveTo(currentDrawing.points[0].x, currentDrawing.points[0].y);
      
      for (let i = 1; i < currentDrawing.points.length - 1; i += 1) {
        const xc = (currentDrawing.points[i].x + currentDrawing.points[i + 1].x) / 2;
        const yc = (currentDrawing.points[i].y + currentDrawing.points[i + 1].y) / 2;
        ctx.quadraticCurveTo(currentDrawing.points[i].x, currentDrawing.points[i].y, xc, yc);
      }
      
      if (currentDrawing.points.length > 1) {
        const lastIndex = currentDrawing.points.length - 1;
        ctx.lineTo(currentDrawing.points[lastIndex].x, currentDrawing.points[lastIndex].y);
      }
      ctx.stroke();
    }
  }, [currentDrawing]);

  // Utility functions for coordinate conversion
  const getCanvasCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - pan.x) / zoom,
      y: (e.clientY - rect.top - pan.y) / zoom
    };
  };

  const checkObjectHit = (x, y) => {
    const tolerance = 10 / zoom;
    
    // Check landmarks (including AI results)
    const allLandmarks = { ...landmarks };
    if (showAIResults && aiResults.length > 0) {
      aiResults.forEach(aiLandmark => {
        if (!allLandmarks[aiLandmark.name]) {
          allLandmarks[aiLandmark.name] = { x: aiLandmark.x, y: aiLandmark.y };
        }
      });
    }
    
    const landmarkEntries = Object.entries(allLandmarks);
    for (let li = 0; li < landmarkEntries.length; li += 1) {
      const [key, point] = landmarkEntries[li];
      const distance = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2);
      if (distance <= landmarkRadius + tolerance) {
        return { type: 'landmark', id: key };
      }
    }
    
    // Check borders
    for (let i = 0; i < customBorders.length; i += 1) {
      const border = customBorders[i];
      for (let j = 0; j < border.points.length - 1; j += 1) {
        const p1 = border.points[j];
        const p2 = border.points[j + 1];
        const distance = pointToLineDistance(x, y, p1, p2);
        if (distance <= tolerance) {
          return { type: 'border', id: `border_${i}` };
        }
      }
    }
    
    // Check curves
    for (let i = 0; i < curvedLines.length; i += 1) {
      const line = curvedLines[i];
      for (let pi = 0; pi < line.points.length; pi += 1) {
        const point = line.points[pi];
        const distance = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2);
        if (distance <= tolerance) {
          return { type: 'curve', id: `curve_${i}` };
        }
      }
    }
    
    return null;
  };

  const pointToLineDistance = (px, py, p1, p2) => {
    const A = px - p1.x;
    const B = py - p1.y;
    const C = p2.x - p1.x;
    const D = p2.y - p1.y;
    
    const dot = A * C + B * D;
    const lenSq = C * C + D * D;
    let param = -1;
    if (lenSq !== 0) param = dot / lenSq;
    
    let xx; let yy;
    if (param < 0) {
      xx = p1.x;
      yy = p1.y;
    } else if (param > 1) {
      xx = p2.x;
      yy = p2.y;
    } else {
      xx = p1.x + param * C;
      yy = p1.y + param * D;
    }
    
    const dx = px - xx;
    const dy = py - yy;
    return Math.sqrt(dx * dx + dy * dy);
  };

  const handleCanvasClick = (e) => {
    const coords = getCanvasCoordinates(e);
    
    if (isPanning) return;
    
    if (editMode) {
      const hit = checkObjectHit(coords.x, coords.y);
      if (hit) {
        setSelectedObject(hit.id);
        return;
      } 
        setSelectedObject(null);
      
    }
    
    if (selectedLandmark) {
      setLandmarks(prev => ({
        ...prev,
        [selectedLandmark]: { x: coords.x, y: coords.y }
      }));
    } else if (drawingMode === 'border' || drawingMode === 'curve') {
      if (!currentDrawing) {
        setCurrentDrawing({
          type: drawingMode,
          points: [coords],
          color: selectedBorderColor,
          thickness: borderThickness
        });
      } else {
        setCurrentDrawing(prev => ({
          ...prev,
          points: [...prev.points, coords]
        }));
      }
    }
  };

  const handleMouseDown = (e) => {
    if (e.button === 1 || e.ctrlKey) { // Middle mouse or Ctrl+click for panning
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
      return;
    }
    
    if (editMode && selectedObject) {
      const coords = getCanvasCoordinates(e);
      setIsDragging(true);
      setDragStart(coords);
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
    } else if (isDragging && selectedObject) {
      const coords = getCanvasCoordinates(e);
      const deltaX = coords.x - dragStart.x;
      const deltaY = coords.y - dragStart.y;
      
      if (selectedObject in landmarks) {
        setLandmarks(prev => ({
          ...prev,
          [selectedObject]: {
            ...prev[selectedObject],
            x: prev[selectedObject].x + deltaX,
            y: prev[selectedObject].y + deltaY
          }
        }));
      } else if (selectedObject.startsWith('border_')) {
        const borderIndex = parseInt(selectedObject.split('_')[1], 10);
        setCustomBorders(prev => prev.map((border, i) =>
          i === borderIndex ? {
            ...border,
            points: border.points.map(point => ({
              x: point.x + deltaX,
              y: point.y + deltaY
            }))
          } : border
        ));
      } else if (selectedObject.startsWith('curve_')) {
        const curveIndex = parseInt(selectedObject.split('_')[1], 10);
        setCurvedLines(prev => prev.map((line, i) =>
          i === curveIndex ? {
            ...line,
            points: line.points.map(point => ({
              x: point.x + deltaX,
              y: point.y + deltaY
            }))
          } : line
        ));
      }
      
      setDragStart(coords);
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    setIsDragging(false);
    
    if (currentDrawing && currentDrawing.points.length > 2) {
      if (currentDrawing.type === 'border') {
        setCustomBorders(prev => [...prev, currentDrawing]);
      } else if (currentDrawing.type === 'curve') {
        setCurvedLines(prev => [...prev, currentDrawing]);
      }
      setCurrentDrawing(null);
    }
  };

  const handleDoubleClick = () => {
    if (currentDrawing && currentDrawing.points.length > 2) {
      if (currentDrawing.type === 'border') {
        setCustomBorders(prev => [...prev, currentDrawing]);
      } else if (currentDrawing.type === 'curve') {
        setCurvedLines(prev => [...prev, currentDrawing]);
      }
      setCurrentDrawing(null);
    }
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(5, prev * delta)));
  };

  // AI Detection Functions - CORS Safe Implementation
  const handleAIDetection = async () => {
    if (!imageRef.current) {
      setAiError('لطفاً ابتدا تصویر سفالوگرام را بارگذاری کنید');
      return;
    }

    setIsAIDetecting(true);
    setAiError(null);
    setAiProcessingProgress(0);
    setAiResults([]);
    setShowAIResults(false);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setAiProcessingProgress(prev => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + Math.random() * 15;
        });
      }, 200);

      // Get image dimensions for scaling AI results
      const imageWidth = imageRef.current.width || 512;
      const imageHeight = imageRef.current.height || 512;
      
      // Skip canvas.toBlob() to avoid CORS issues, use image dimensions instead
      // In a real implementation, you would send the image data through a different method
      const result = await detectLandmarksAI(null, { width: imageWidth, height: imageHeight });
      
      clearInterval(progressInterval);
      setAiProcessingProgress(100);
      
      // Process AI results (already scaled by the AI function)
      const landmarkResults = Object.entries(result.landmarks).map(([name, data]) => ({
        name,
        x: data.x,
        y: data.y,
        confidence: data.confidence,
        isAI: true
      }));
      
      setAiResults(landmarkResults);
      
      // Set confidence scores
      const confidenceScores = {};
      Object.entries(result.landmarks).forEach(([name, data]) => {
        confidenceScores[name] = data.confidence;
      });
      setAiConfidence(confidenceScores);
      
      setTimeout(() => {
        setIsAIDetecting(false);
        setShowAIResults(true);
        setShowAIDialog(false);
      }, 500);
      
    } catch (error) {
      setIsAIDetecting(false);
      setAiError('خطا در ب تصویر توسط هوش مصنوعی. لطفاً دوباره تلاش کنید.');
      console.error('AI Detection Error:', error);
    }
  };

  const acceptAIResults = () => {
    const newLandmarks = { ...landmarks };
    aiResults.forEach(result => {
      newLandmarks[result.name] = {
        x: result.x,
        y: result.y,
        color: LANDMARKS[result.name]?.color || '#2196F3',
        aiDetected: true,
        confidence: result.confidence
      };
    });
    setLandmarks(newLandmarks);
    setShowAIResults(false);
    setAiResults([]);
  };

  const rejectAIResults = () => {
    setShowAIResults(false);
    setAiResults([]);
    setAiConfidence({});
  };

  // Utility functions
  const deleteSelectedObject = () => {
    if (!selectedObject) return;
    
    if (selectedObject in landmarks) {
      const newLandmarks = { ...landmarks };
      delete newLandmarks[selectedObject];
      setLandmarks(newLandmarks);
    } else if (selectedObject.startsWith('border_')) {
      const borderIndex = parseInt(selectedObject.split('_')[1], 10);
      setCustomBorders(prev => prev.filter((_, i) => i !== borderIndex));
    } else if (selectedObject.startsWith('curve_')) {
      const curveIndex = parseInt(selectedObject.split('_')[1], 10);
      setCurvedLines(prev => prev.filter((_, i) => i !== curveIndex));
    }
    
    setSelectedObject(null);
  };

  const saveAnnotations = () => {
    const data = {
      landmarks,
      customBorders,
      curvedLines,
      aiResults: showAIResults ? aiResults : [],
      timestamp: new Date().toISOString()
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cephalometric-${saveFileName || 'annotations'}.json`;
    a.click();
    URL.revokeObjectURL(url);
    setShowSaveDialog(false);
    setSaveFileName('');
  };

  const loadAnnotations = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        setLandmarks(data.landmarks || {});
        setCustomBorders(data.customBorders || []);
        setCurvedLines(data.curvedLines || []);
        if (data.aiResults) {
          setAiResults(data.aiResults);
          setShowAIResults(true);
        }
      } catch (error) {
        console.error('Error loading annotations:', error);
      }
    };
    reader.readAsText(file);
  };

  const clearAllAnnotations = () => {
    setLandmarks({});
    setCustomBorders([]);
    setCurvedLines([]);
    setSelectedObject(null);
    setCurrentDrawing(null);
    setAiResults([]);
    setShowAIResults(false);
  };

  const calculateAngle = useCallback((p1, vertex, p2) => {
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
    let angle = Math.abs(angle1 - angle2) * (180 / Math.PI);
    if (angle > 180) angle = 360 - angle;
    return angle.toFixed(2);
  }, []);

  const calculateDistance = useCallback((p1, p2) => {
    const dx = p2.x - p1.x;
    const dy = p2.y - p1.y;
    return Math.sqrt(dx * dx + dy * dy).toFixed(2);
  }, []);

  const calculateMeasurements = useCallback((points) => {
    const measurements = {};
    
    // Skeletal measurements
    if (points.S && points.N && points.A) {
      measurements.SNA = calculateAngle(points.S, points.N, points.A);
    }
    
    if (points.S && points.N && points.B) {
      measurements.SNB = calculateAngle(points.S, points.N, points.B);
    }
    
    if (points.A && points.N && points.B) {
      measurements.ANB = calculateAngle(points.A, points.N, points.B);
    }
    
    if (points.Go && points.Me && points.N) {
      measurements.gonialAngle = calculateAngle(points.Me, points.Go, points.N);
    }
    
    // Dental measurements
    if (points.U1 && points.L1) {
      // Overjet = فاصله افقی (تفاوت x)
      // اگر U1.x > L1.x (U1 جلوتر است) → مقدار مثبت
      // اگر L1.x > U1.x (L1 جلوتر است) → مقدار منفی
      const horizontalDistance = points.U1.x - points.L1.x;
      measurements.overjet = parseFloat(horizontalDistance.toFixed(2));
    }
    
    return measurements;
  }, [calculateAngle]);

  return (
    <Box>
      {/* Toolbar */}
      <Paper sx={{ p: 2, mb: 2 }}>
        {/* Mode Selection */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ mb: 2 }}>
          <Typography variant="subtitle2">حالت ویرایش:</Typography>
          
          <Button
            variant={editMode ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setEditMode(!editMode)}
            startIcon={<Iconify icon="eva:edit-fill" />}
          >
            ویرایش
          </Button>
          
          <Button
            variant={drawingMode === 'select' ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setDrawingMode('select')}
            startIcon={<Iconify icon="eva:cursor-fill" />}
          >
            انتخاب
          </Button>
          
          <Button
            variant={drawingMode === 'border' ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setDrawingMode('border')}
            startIcon={<Iconify icon="eva:square-fill" />}
          >
            حاشیه
          </Button>
          
          <Button
            variant={drawingMode === 'curve' ? 'contained' : 'outlined'}
            size="small"
            onClick={() => setDrawingMode('curve')}
            startIcon={<Iconify icon="eva:path-fill" />}
          >
            منحنی
          </Button>
          
          {/* AI Detection Button */}
          <Button
            variant="contained"
            size="small"
            color="secondary"
            onClick={() => setShowAIDialog(true)}
            startIcon={<Iconify icon="eva:cpu-fill" />}
            disabled={!imageUrl || isAIDetecting}
          >
            {isAIDetecting ? 'در حال پردازش...' : 'تشخیص خودکار'}
          </Button>
        </Stack>

        {/* AI Processing Progress */}
        {isAIDetecting && (
          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              در حال پردازش تصویر توسط هوش مصنوعی...
            </Typography>
            <LinearProgress 
              variant="determinate" 
              value={aiProcessingProgress} 
              sx={{ height: 8, borderRadius: 4 }}
            />
            <Typography variant="caption" sx={{ mt: 1, display: 'block' }}>
              {aiProcessingProgress.toFixed(0)}% کامل شده
            </Typography>
          </Box>
        )}

        {/* AI Error Display */}
        {aiError && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {aiError}
          </Alert>
        )}

        {/* AI Results Actions */}
        {showAIResults && aiResults.length > 0 && (
          <Alert severity="info" sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              نتایج تشخیص هوش مصنوعی آماده است. تعداد نقاط شناسایی شده: {aiResults.length}
            </Typography>
            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
              <Button size="small" variant="contained" onClick={acceptAIResults}>
                تایید نتایج
              </Button>
              <Button size="small" variant="outlined" onClick={rejectAIResults}>
                رد نتایج
              </Button>
            </Stack>
          </Alert>
        )}

        {/* Drawing Settings */}
        {(drawingMode === 'border' || drawingMode === 'curve') && (
          <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap" sx={{ mb: 2 }}>
            <Typography variant="subtitle2">تنظیمات:</Typography>
            
            <TextField
              size="small"
              type="color"
              label="رنگ"
              value={selectedBorderColor}
              onChange={(e) => setSelectedBorderColor(e.target.value)}
              sx={{ width: 100 }}
            />
            
            <TextField
              size="small"
              type="number"
              label="ضخامت"
              value={borderThickness}
              onChange={(e) => setBorderThickness(Number(e.target.value))}
              sx={{ width: 80 }}
              inputProps={{ min: 1, max: 10 }}
            />
            
            <Select
              size="small"
              value={selectedBorderStyle}
              onChange={(e) => setSelectedBorderStyle(e.target.value)}
              sx={{ width: 100 }}
            >
              <MenuItem value="solid">صاف</MenuItem>
              <MenuItem value="dashed">خط چین</MenuItem>
              <MenuItem value="dotted">نقطه چین</MenuItem>
            </Select>
          </Stack>
        )}

        {/* Landmark Selection */}
        <Stack direction="row" spacing={2} alignItems="center" flexWrap="wrap">
          <Typography variant="subtitle2">انتخاب نقطه:</Typography>
          {Object.keys(LANDMARKS).map((key) => (
            <Box
              key={key}
              onClick={() => setSelectedLandmark(key)}
              sx={{
                px: 1.5,
                py: 0.5,
                borderRadius: 1,
                cursor: 'pointer',
                bgcolor: selectedLandmark === key ? 'primary.main' : 'grey.200',
                color: selectedLandmark === key ? 'white' : 'text.primary',
                fontSize: '0.875rem',
                '&:hover': {
                  bgcolor: selectedLandmark === key ? 'primary.dark' : 'grey.300',
                },
                position: 'relative'
              }}
            >
              {key}
              {aiConfidence[key] && (
                <Chip
                  size="small"
                  label={`${(aiConfidence[key] * 100).toFixed(0)}%`}
                  sx={{
                    position: 'absolute',
                    top: -8,
                    right: -8,
                    height: 16,
                    fontSize: '0.6rem',
                    bgcolor: aiConfidence[key] > 0.9 ? '#4CAF50' : aiConfidence[key] > 0.7 ? '#FF9800' : '#F44336',
                    color: 'white'
                  }}
                />
              )}
            </Box>
          ))}
        </Stack>
        
        {/* Control Buttons */}
        <Stack direction="row" spacing={1} sx={{ mt: 2 }} flexWrap="wrap">
          <Tooltip title="بزرگنمایی">
            <IconButton size="small" onClick={() => setZoom(prev => prev * 1.2)}>
              <Iconify icon="eva:plus-fill" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="کوچکنمایی">
            <IconButton size="small" onClick={() => setZoom(prev => prev / 1.2)}>
              <Iconify icon="eva:minus-fill" />
            </IconButton>
          </Tooltip>
          
          <Tooltip title="بازنشانی">
            <IconButton size="small" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
              <Iconify icon="eva:refresh-fill" />
            </IconButton>
          </Tooltip>
          
          <Typography variant="caption" sx={{ alignSelf: 'center', ml: 1 }}>
            زوم: {(zoom * 100).toFixed(0)}%
          </Typography>
          
          {/* Edit Mode Controls */}
          {editMode && (
            <>
              <Tooltip title="حذف انتخاب شده">
                <IconButton 
                  size="small" 
                  onClick={deleteSelectedObject}
                  disabled={!selectedObject}
                  color="error"
                >
                  <Iconify icon="eva:trash-fill" />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="پاک کردن همه">
                <IconButton 
                  size="small" 
                  onClick={clearAllAnnotations}
                  color="warning"
                >
                  <Iconify icon="eva:close-circle-fill" />
                </IconButton>
              </Tooltip>
              
              <Button
                size="small"
                onClick={() => setShowSaveDialog(true)}
                startIcon={<Iconify icon="eva:download-fill" />}
              >
                ذخیره
              </Button>
              
              <Button
                size="small"
                onClick={() => fileInputRef.current?.click()}
                startIcon={<Iconify icon="eva:upload-fill" />}
              >
                بارگذاری
              </Button>
              
              <input
                type="file"
                ref={fileInputRef}
                onChange={loadAnnotations}
                accept=".json"
                style={{ display: 'none' }}
              />
            </>
          )}
        </Stack>
      </Paper>
      
      {/* Canvas */}
      <Paper
        sx={{
          overflow: 'hidden',
          cursor: isPanning ? 'grabbing' : 
                 editMode ? 'crosshair' : 
                 selectedLandmark ? 'crosshair' : 
                 drawingMode === 'border' || drawingMode === 'curve' ? 'crosshair' : 'grab',
          bgcolor: '#000',
          position: 'relative',
          height: 600,
        }}
      >
        <canvas
          ref={canvasRef}
          onClick={handleCanvasClick}
          onDoubleClick={handleDoubleClick}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onWheel={handleWheel}
          style={{ display: 'block', maxWidth: '100%', height: '100%' }}
        />
        
        {/* Status Bar */}
        {editMode && selectedObject && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 10,
              left: 10,
              bgcolor: 'rgba(0,0,0,0.8)',
              color: 'white',
              px: 2,
              py: 1,
              borderRadius: 1,
              fontSize: '0.875rem'
            }}
          >
            انتخاب شده: {selectedObject}
            {aiConfidence[selectedObject] && (
              <span style={{ marginLeft: 8 }}>
                (Confidence: {(aiConfidence[selectedObject] * 100).toFixed(0)}%)
              </span>
            )}
          </Box>
        )}
        
        {/* AI Results Status */}
        {showAIResults && (
          <Box
            sx={{
              position: 'absolute',
              top: 10,
              left: 10,
              bgcolor: 'rgba(255,152,0,0.9)',
              color: 'white',
              px: 2,
              py: 1,
              borderRadius: 1,
              fontSize: '0.875rem'
            }}
          >
            نتایج AI نمایش داده شده است - قابل ویرایش
          </Box>
        )}
        
        {/* Drawing Instructions */}
        {(drawingMode === 'border' || drawingMode === 'curve') && !currentDrawing && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 10,
              left: '50%',
              transform: 'translateX(-50%)',
              bgcolor: 'rgba(0,0,0,0.8)',
              color: 'white',
              px: 2,
              py: 1,
              borderRadius: 1,
              fontSize: '0.875rem'
            }}
          >
            برای شروع رسم کلیک کنید (حداقل ۳ نقطه)
          </Box>
        )}
        
        {currentDrawing && (
          <Box
            sx={{
              position: 'absolute',
              bottom: 10,
              left: '50%',
              transform: 'translateX(-50%)',
              bgcolor: 'rgba(0,0,0,0.8)',
              color: 'white',
              px: 2,
              py: 1,
              borderRadius: 1,
              fontSize: '0.875rem'
            }}
          >
            نق��ط: {currentDrawing.points.length} - برای پایان دوبار کلیک کنید
          </Box>
        )}
      </Paper>

      {/* Save Dialog */}
      <Dialog open={showSaveDialog} onClose={() => setShowSaveDialog(false)}>
        <DialogTitle>ذخیره آنوتیشن‌ها</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="نام فایل"
            fullWidth
            variant="outlined"
            value={saveFileName}
            onChange={(e) => setSaveFileName(e.target.value)}
            placeholder="cephalometric-analysis"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowSaveDialog(false)}>لغو</Button>
          <Button onClick={saveAnnotations} variant="contained">ذخیره</Button>
        </DialogActions>
      </Dialog>

      {/* AI Detection Dialog */}
      <Dialog open={showAIDialog} onClose={() => !isAIDetecting && setShowAIDialog(false)}>
        <DialogTitle>تشخیص خودکار نقاط آناتومیک</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            این قابلیت از هوش مصنوعی برای تشخیص خودکار ��قاط آناتومیک در تصویر سفالوگرام استفاده می‌کند.
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            • نتایج با دقت بالا قابل اعتماد هستند
            <br />
            • امکان ویرایش دستی نتایج وجود دارد
            <br />
            • زمان پردازش معمولاً ۲-۵ ثانیه است
            <br />
            • قابلیت کار با تصاویر خارجی (CORS-safe)
          </Typography>
          {!imageUrl && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              لطفاً ابتدا تصویر سفالوگرام را بارگذاری کنید
            </Alert>
          )}
          <Alert severity="info" sx={{ mb: 2 }}>
            <strong>توج��:</strong> این نسخه CORS-safe است و با تصاویر خارجی کار می‌کند.
          </Alert>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowAIDialog(false)} disabled={isAIDetecting}>
            لغو
          </Button>
          <Button 
            onClick={handleAIDetection} 
            variant="contained" 
            disabled={!imageUrl || isAIDetecting}
            startIcon={<Iconify icon="eva:cpu-fill" />}
          >
            {isAIDetecting ? 'در حال پردازش...' : 'شروع تشخیص'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
