import React, { useRef, useState, useEffect, useCallback } from 'react';

import {
  Box,
  Grid,
  Paper,
  Stack,
  Alert,
  Button,
  Dialog,
  Tooltip,
  Typography,
  IconButton,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';

import { Iconify } from 'src/components/iconify';

import AIModelSelector from '../../cephalometric-analysis/ai-model-selector';

// Inline SVG icons to avoid remote loading issues
const LineIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 20L20 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const CurveIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M4 14C7 8 13 8 20 4" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M4 20L20 4" stroke="currentColor" strokeWidth="0" />
  </svg>
);

const RulerIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="3" y="10" width="18" height="4" rx="1" stroke="currentColor" strokeWidth="1.5" />
    <path d="M6 10v4M9 10v4M12 10v4M15 10v4M18 10v4" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
  </svg>
);

const ScissorsIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M14.121 9.879a3 3 0 1 0-4.242 4.242 3 3 0 0 0 4.242-4.242z" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M20 4l-8 8M4 20l8-8" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const EraserIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M3 21l7-7 7-7 3 3-7 7L6 21H3z" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

const RotateIcon = ({ width = 18 }) => (
  <svg width={width} height={width} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M21 12a9 9 0 1 0-3.95 7.05" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
    <path d="M21 3v5h-5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
  </svg>
);

// ToolButton helper wraps disabled children to satisfy MUI Tooltip requirement
function ToolButton({ title, children, disabled, ...props }) {
  return (
    <Tooltip title={title}>
      <span>
        <IconButton {...props} disabled={disabled}>
          {children}
        </IconButton>
      </span>
    </Tooltip>
  );
}

// Common orthodontic landmarks for lateral cephalometric analysis
const CEPHALOMETRIC_LANDMARKS = {
  // Hard tissue landmarks
  S: { name: 'Sella', description: 'Center of sella turcica', color: '#FF6B6B' },
  N: { name: 'Nasion', description: 'Frontonasal suture', color: '#4ECDC4' },
  A: { name: 'Point A', description: 'Deepest point on maxillary alveolus', color: '#45B7D1' },
  B: { name: 'Point B', description: 'Deepest point on mandibular alveolus', color: '#96CEB4' },
  Pg: { name: 'Pogonion', description: 'Most anterior point of chin', color: '#FFEAA7' },
  Gn: { name: 'Gnathion', description: 'Most anterior-inferior point of chin', color: '#DDA0DD' },
  Me: { name: 'Menton', description: 'Most inferior point of mandibular symphysis', color: '#98D8C8' },
  Go: { name: 'Gonion', description: 'Postero-inferior angle of mandible', color: '#F7DC6F' },
  Po: { name: 'Porion', description: 'Superior point of external auditory meatus', color: '#BB8FCE' },
  Or: { name: 'Orbitale', description: 'Lowest point of orbital margin', color: '#85C1E9' },
  ANS: { name: 'Anterior Nasal Spine', description: 'Tip of anterior nasal spine', color: '#F8C471' },
  PNS: { name: 'Posterior Nasal Spine', description: 'Tip of posterior nasal spine', color: '#82E0AA' },

  // Calibration points
  p1: { name: 'P1 (Calibration)', description: 'Lower calibration point (1cm from p2)', color: '#FFD700' },
  p2: { name: 'P2 (Calibration)', description: 'Upper calibration point', color: '#FFA500' },

  // Soft tissue landmarks
  Sn: { name: 'Subnasale', description: 'Point where columella meets upper lip', color: '#F1948A' },
  Ls: { name: 'Labrale Superius', description: 'Most anterior point of upper lip', color: '#AED6F1' },
  Li: { name: 'Labrale Inferius', description: 'Most anterior point of lower lip', color: '#A3E4D7' },
  Sm: { name: 'Supramentale', description: 'Most concave point of mandibular symphysis', color: '#FAD7A0' },
};

// Line drawing modes
const DRAWING_MODES = {
  NONE: 'none',
  STRAIGHT_LINE: 'straight',
  CURVED_LINE: 'curved',
};

// Line structure
const createLine = (start, end, type = 'straight') => ({
  id: `line_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
  start,
  end,
  type,
  color: '#FF5722',
  width: 2,
});

const CephalometricLandmarkViewer = ({
  imageUrl,
  onLandmarkUpdate,
  initialLandmarks = {},
  currentLandmarks = {},
  readOnly = false,
  onLandmarkDetect = null,
  onCropSave = null,
}) => {
  const canvasRef = useRef(null);
  const animationRef = useRef(null);
  const [image, setImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [landmarks, setLandmarks] = useState(initialLandmarks);
  const [selectedLandmark, setSelectedLandmark] = useState(null);
  const [draggedLandmark, setDraggedLandmark] = useState(null);
  const [showLandmarkDialog, setShowLandmarkDialog] = useState(false);
  const [addPointPosition, setAddPointPosition] = useState(null);
  const [transform, setTransform] = useState({ scale: 1, offsetX: 0, offsetY: 0 });
  const [lastMousePos, setLastMousePos] = useState({ x: 0, y: 0 });
  const [isDraggingCanvas, setIsDraggingCanvas] = useState(false);

  // Line drawing state
  const [lines, setLines] = useState([]);
  const [drawingMode, setDrawingMode] = useState(DRAWING_MODES.NONE);
  const [isDrawingLine, setIsDrawingLine] = useState(false);
  const [lineStart, setLineStart] = useState(null);
  const [tempLineEnd, setTempLineEnd] = useState(null);

  // Line editing state
  const [selectedLine, setSelectedLine] = useState(null);
  const [editingPoint, setEditingPoint] = useState(null); // 'start' or 'end'

  // Image manipulation state
  const [showRuler, setShowRuler] = useState(false);
  const [imageRotation, setImageRotation] = useState(0);
  const [isCropping, setIsCropping] = useState(false);
  const [cropArea, setCropArea] = useState(null); // screen coords
  const [cropStart, setCropStart] = useState(null); // screen coords
  const [cropHandle, setCropHandle] = useState(null); // 'tl','tr','bl','br','tc','bc','lc','rc','move'

  // History for undo (store previous image dataUrls)
  const [imageHistory, setImageHistory] = useState([]);

  // AI Model selection
  const [showAIModelDialog, setShowAIModelDialog] = useState(false);
  const [selectedAIModel, setSelectedAIModel] = useState('cephx-v2');
  const [isAIDetecting, setIsAIDetecting] = useState(false);

  // Load image
  useEffect(() => {
    if (!imageUrl) return;

    setImageLoaded(false);
    const img = new window.Image();

    // Handle blob URLs properly - revoke previous blob URLs to prevent memory leaks
    if (imageUrl.startsWith('blob:')) {
      // For blob URLs, we need to handle CORS differently
      img.crossOrigin = null; // Remove crossOrigin for blob URLs
    } else {
      img.crossOrigin = 'anonymous';
    }

    img.onload = () => {
      setImage(img);
      setImageLoaded(true);

      // Calculate initial scale to fit viewer
      const canvas = canvasRef.current;
      const scaleX = canvas.width / img.width;
      const scaleY = canvas.height / img.height;
      const scale = Math.min(scaleX, scaleY, 1);

      setTransform({
        scale,
        offsetX: (canvas.width - img.width * scale) / 2,
        offsetY: (canvas.height - img.height * scale) / 2,
      });
    };

    img.onerror = (error) => {
      console.error('[Landmark Visualizer] Error loading image:', error);
      console.error('[Landmark Visualizer] Failed URL:', imageUrl);
      console.error('[Landmark Visualizer] Image object:', img);
      setImageLoaded(false);

      // If it's a blob URL error, try to reload from original source
      if (imageUrl.startsWith('blob:')) {
        console.warn('[Landmark Visualizer] Blob URL failed, attempting to reload from server...');

        // Try multiple fallback strategies for blob URL recovery
        const fallbackStrategies = [
          // Strategy 1: Try to extract filename and construct server URL
          () => {
            const blobParts = imageUrl.split('/');
            const filename = blobParts[blobParts.length - 1];
            if (filename && filename.length > 10) { // Valid blob filename
              return `http://localhost:5001/uploads/${filename}`;
            }
            return null;
          },
          // Strategy 2: Try common image paths
          () => `http://localhost:5001/uploads/lateral-cephalometric.jpg`,
          () => `http://localhost:5001/uploads/cephalometric.jpg`,
          // Strategy 3: Try patient images endpoint
          () => {
            // Extract patient ID from URL if available
            const urlParams = new URLSearchParams(window.location.search);
            const patientId = urlParams.get('id') || urlParams.get('patientId');
            if (patientId) {
              return `http://localhost:5001/api/patients/${patientId}/images/lateral`;
            }
            return null;
          },
          // Strategy 4: Try fallback to port 7272 (old server)
          () => {
            const blobParts = imageUrl.split('/');
            const filename = blobParts[blobParts.length - 1];
            if (filename && filename.length > 10) {
              return `http://localhost:7272/uploads/${filename}`;
            }
            return null;
          }
        ];

        // Try each fallback strategy
        for (const strategy of fallbackStrategies) {
          const fallbackUrl = strategy();
          if (fallbackUrl) {
            console.log('[Landmark Visualizer] Trying fallback URL:', fallbackUrl);
            // Create new image element to avoid recursion
            const fallbackImg = new window.Image();
            fallbackImg.crossOrigin = 'anonymous';
            fallbackImg.onload = () => {
              console.log('[Landmark Visualizer] Fallback URL succeeded!');
              setImage(fallbackImg);
              setImageLoaded(true);
            };
            fallbackImg.onerror = () => {
              console.warn('[Landmark Visualizer] Fallback URL failed:', fallbackUrl);
            };
            fallbackImg.src = fallbackUrl;
            break; // Only try first successful strategy
          }
        }
        
      }
    };

    img.src = imageUrl;
  }, [imageUrl]);

  // Update landmarks when currentLandmarks prop changes (AI detection results)
  useEffect(() => {
    if (Object.keys(currentLandmarks).length > 0) {
      console.log('🧠 AI landmarks received:', currentLandmarks);
      setLandmarks(currentLandmarks);

      // Notify parent component about the new landmarks
      if (onLandmarkUpdate) {
        onLandmarkUpdate(currentLandmarks);
      }
    }
  }, [currentLandmarks, onLandmarkUpdate]);

  // Optimized render function - Fixed point sizes with smooth performance
  const render = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Save context state
    ctx.save();

    // Apply transformation
    ctx.translate(transform.offsetX + (image.width * transform.scale) / 2, transform.offsetY + (image.height * transform.scale) / 2);
    ctx.rotate((imageRotation * Math.PI) / 180);
    ctx.scale(transform.scale, transform.scale);
    ctx.translate(-(image.width / 2), -(image.height / 2));

    // Draw image with offscreen rendering for smooth dragging
    ctx.drawImage(image, 0, 0, image.width, image.height);

    ctx.restore();

    // Draw lines first (behind landmarks)
    lines.forEach(line => {
      if (!line.start || !line.end) return;

      const startX = transform.offsetX + line.start.x * transform.scale;
      const startY = transform.offsetY + line.start.y * transform.scale;
      const endX = transform.offsetX + line.end.x * transform.scale;
      const endY = transform.offsetY + line.end.y * transform.scale;

      ctx.strokeStyle = line.color;
      ctx.lineWidth = line.width;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();

      if (line.type === 'curved') {
        // Draw curved line using quadratic bezier curve
        const midX = (startX + endX) / 2;
        const midY = (startY + endY) / 2;
        const controlX = midX + (startY - endY) * 0.3;
        const controlY = midY + (endX - startX) * 0.3;

        ctx.moveTo(startX, startY);
        ctx.quadraticCurveTo(controlX, controlY, endX, endY);
      } else {
        // Draw straight line
        ctx.moveTo(startX, startY);
        ctx.lineTo(endX, endY);
      }

      ctx.stroke();
    });

    // Draw temporary line being drawn
    if (isDrawingLine && lineStart && tempLineEnd) {
      const startX = transform.offsetX + lineStart.x * transform.scale;
      const startY = transform.offsetY + lineStart.y * transform.scale;
      const endX = transform.offsetX + tempLineEnd.x * transform.scale;
      const endY = transform.offsetY + tempLineEnd.y * transform.scale;

      ctx.strokeStyle = '#FF5722';
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.lineTo(endX, endY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw landmarks at fixed screen size (not affected by zoom)
    Object.entries(landmarks).forEach(([key, position]) => {
      if (!position) return;

      const landmark = CEPHALOMETRIC_LANDMARKS[key];
      if (!landmark) return;

      // Convert image coordinates to screen coordinates
      const screenX = transform.offsetX + position.x * transform.scale;
      const screenY = transform.offsetY + position.y * transform.scale;

      // Skip rendering dragged landmark during drag for smooth movement
      if (draggedLandmark === key) return;

      // Set opacity to 0.8 for anatomical points
      ctx.globalAlpha = 0.8;

      // Draw landmark point - Professional style with better visibility
      const pointRadius = 8; // Larger for better visibility
      
      // Outer glow effect
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(screenX, screenY, pointRadius + 4, 0, 2 * Math.PI);
      ctx.fillStyle = landmark.color;
      ctx.fill();
      
      ctx.globalAlpha = 0.9;
      
      // Main point circle
      ctx.beginPath();
      ctx.arc(screenX, screenY, pointRadius, 0, 2 * Math.PI);
      ctx.fillStyle = landmark.color;
      ctx.fill();

      // White border for contrast
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Inner darker border
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Center crosshair for precision
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(screenX - 4, screenY);
      ctx.lineTo(screenX + 4, screenY);
      ctx.moveTo(screenX, screenY - 4);
      ctx.lineTo(screenX, screenY + 4);
      ctx.stroke();

      // Reset global alpha
      ctx.globalAlpha = 1.0;

      // Draw label with background for better readability
      ctx.font = 'bold 13px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      
      const labelY = screenY + pointRadius + 4;
      const textMetrics = ctx.measureText(key);
      const textWidth = textMetrics.width;
      
      // Label background
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
      ctx.fillRect(screenX - textWidth / 2 - 3, labelY - 1, textWidth + 6, 16);
      
      // Label border
      ctx.strokeStyle = landmark.color;
      ctx.lineWidth = 1;
      ctx.strokeRect(screenX - textWidth / 2 - 3, labelY - 1, textWidth + 6, 16);
      
      // Label text
      ctx.fillStyle = '#000000';
      ctx.fillText(key, screenX, labelY + 1);

      // Highlight selected landmark with subtle selection
      if (selectedLandmark === key) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(screenX, screenY, 8, 0, 2 * Math.PI);
        ctx.strokeStyle = '#FF0000';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.globalAlpha = 1.0;
      }
    });

    // Draw dragged landmark at mouse position for smooth feedback
    if (draggedLandmark) {
      const pos = landmarks[draggedLandmark];
      const landmark = CEPHALOMETRIC_LANDMARKS[draggedLandmark];
      if (pos && landmark) {
        const screenX = transform.offsetX + pos.x * transform.scale;
        const screenY = transform.offsetY + pos.y * transform.scale;

        ctx.globalAlpha = 0.7;
        ctx.beginPath();
        ctx.arc(screenX, screenY, 8, 0, 2 * Math.PI);
        ctx.fillStyle = landmark.color;
        ctx.fill();
        ctx.globalAlpha = 1.0;
      }
    }

    // Draw rulers if enabled
    if (showRuler && image) {
      ctx.strokeStyle = '#2196F3';
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';

      // Vertical ruler (center line)
      const centerX = transform.offsetX + (image.width * transform.scale) / 2;
      ctx.beginPath();
      ctx.moveTo(centerX, 0);
      ctx.lineTo(centerX, canvas.height);
      ctx.stroke();

      // Horizontal ruler (center line)
      const centerY = transform.offsetY + (image.height * transform.scale) / 2;
      ctx.beginPath();
      ctx.moveTo(0, centerY);
      ctx.lineTo(canvas.width, centerY);
      ctx.stroke();

      // Draw ruler markers every 50 pixels
      ctx.fillStyle = '#2196F3';
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';

      // Vertical markers
      for (let y = 50; y < canvas.height; y += 50) {
        ctx.beginPath();
        ctx.arc(centerX, y, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText(Math.round((y - centerY) / transform.scale), centerX + 20, y + 3);
      }

      // Horizontal markers
      for (let x = 50; x < canvas.width; x += 50) {
        ctx.beginPath();
        ctx.arc(x, centerY, 3, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillText(Math.round((x - centerX) / transform.scale), x, centerY - 15);
      }
    }

    // Draw cropping rectangle if cropping is active
    if (isCropping && cropArea) {
      // Draw dimmed overlay outside crop rect
      ctx.save();
      ctx.fillStyle = 'rgba(0,0,0,0.45)';
      // Top
      ctx.fillRect(0, 0, canvas.width, cropArea.y);
      // Bottom
      ctx.fillRect(0, cropArea.y + cropArea.height, canvas.width, canvas.height - (cropArea.y + cropArea.height));
      // Left
      ctx.fillRect(0, cropArea.y, cropArea.x, cropArea.height);
      // Right
      ctx.fillRect(cropArea.x + cropArea.width, cropArea.y, canvas.width - (cropArea.x + cropArea.width), cropArea.height);

      // Draw crop box border
      ctx.strokeStyle = '#FF5722';
      ctx.lineWidth = 2.5;
      ctx.setLineDash([8, 4]);
      ctx.strokeRect(cropArea.x, cropArea.y, cropArea.width, cropArea.height);
      ctx.setLineDash([]);

      // Draw 3x3 grid inside crop box
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 1;
      const cols = 3;
      const rows = 3;
      for (let i = 1; i < cols; i += 1) {
        const gx = cropArea.x + (cropArea.width * i) / cols;
        ctx.beginPath();
        ctx.moveTo(gx, cropArea.y + 4);
        ctx.lineTo(gx, cropArea.y + cropArea.height - 4);
        ctx.stroke();
      }
      for (let j = 1; j < rows; j += 1) {
        const gy = cropArea.y + (cropArea.height * j) / rows;
        ctx.beginPath();
        ctx.moveTo(cropArea.x + 4, gy);
        ctx.lineTo(cropArea.x + cropArea.width - 4, gy);
        ctx.stroke();
      }

      // Draw resize handles at corners and edges
      const handles = [
        // corners
        { x: cropArea.x, y: cropArea.y }, // top-left
        { x: cropArea.x + cropArea.width, y: cropArea.y }, // top-right
        { x: cropArea.x, y: cropArea.y + cropArea.height }, // bottom-left
        { x: cropArea.x + cropArea.width, y: cropArea.y + cropArea.height }, // bottom-right
        // edges center
        { x: cropArea.x + cropArea.width / 2, y: cropArea.y }, // top-center
        { x: cropArea.x + cropArea.width / 2, y: cropArea.y + cropArea.height }, // bottom-center
        { x: cropArea.x, y: cropArea.y + cropArea.height / 2 }, // left-center
        { x: cropArea.x + cropArea.width, y: cropArea.y + cropArea.height / 2 }, // right-center
      ];

      ctx.fillStyle = '#FF5722';
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      handles.forEach(handle => {
        ctx.beginPath();
        ctx.arc(handle.x, handle.y, 6, 0, 2 * Math.PI);
        ctx.fill();
        ctx.stroke();
      });

      ctx.restore();
    }

    // Draw selected line editing handles
    if (selectedLine) {
      const line = lines.find(l => l.id === selectedLine);
      if (line) {
        // Draw start point handle - green circle
        const startScreenX = transform.offsetX + line.start.x * transform.scale;
        const startScreenY = transform.offsetY + line.start.y * transform.scale;

        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(startScreenX, startScreenY, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#4CAF50'; // Green for start
        ctx.fill();
        ctx.strokeStyle = '#2E7D32';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw white dot in center
        ctx.beginPath();
        ctx.arc(startScreenX, startScreenY, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();

        // Draw end point handle - red circle
        const endScreenX = transform.offsetX + line.end.x * transform.scale;
        const endScreenY = transform.offsetY + line.end.y * transform.scale;

        ctx.beginPath();
        ctx.arc(endScreenX, endScreenY, 8, 0, 2 * Math.PI);
        ctx.fillStyle = '#FF5722'; // Red for end
        ctx.fill();
        ctx.strokeStyle = '#D84315';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Draw white dot in center
        ctx.beginPath();
        ctx.arc(endScreenX, endScreenY, 2, 0, 2 * Math.PI);
        ctx.fillStyle = '#FFFFFF';
        ctx.fill();

        ctx.globalAlpha = 1.0;

        // Highlight the selected line with blue color
        ctx.strokeStyle = '#2196F3';
        ctx.lineWidth = 3;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';

        ctx.beginPath();
        if (line.type === 'curved') {
          const midX = startScreenX;
          const midY = startScreenY;
          const controlX = midX + (startScreenY - endScreenY) * 0.3;
          const controlY = midY + (endScreenX - startScreenX) * 0.3;

          ctx.moveTo(startScreenX, startScreenY);
          ctx.quadraticCurveTo(controlX, controlY, endScreenX, endScreenY);
        } else {
          ctx.moveTo(startScreenX, startScreenY);
          ctx.lineTo(endScreenX, endScreenY);
        }
        ctx.stroke();
      }
    }
  }, [image, landmarks, transform, selectedLandmark, draggedLandmark, lines, isDrawingLine, lineStart, tempLineEnd, imageRotation, showRuler, isCropping, cropArea, selectedLine]);

  // Trigger render on state changes with animation frame optimization
  useEffect(() => {
    if (imageLoaded) {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
      animationRef.current = requestAnimationFrame(render);
    }
  }, [render, imageLoaded]);

  // Enhanced mouse event handlers with better performance
  const handleMouseDown = useCallback((e) => {
    if (!imageLoaded || readOnly) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert screen coordinates to image coordinates
    const imageX = (mouseX - transform.offsetX) / transform.scale;
    const imageY = (mouseY - transform.offsetY) / transform.scale;

    setLastMousePos({ x: mouseX, y: mouseY });

    // If cropping mode is active, start crop rectangle or interact with existing crop
    if (isCropping) {
      if (cropArea) {
        // Check if clicked on any resize handle or inside crop area for moving
        const handles = [
          { key: 'tl', x: cropArea.x, y: cropArea.y },
          { key: 'tr', x: cropArea.x + cropArea.width, y: cropArea.y },
          { key: 'bl', x: cropArea.x, y: cropArea.y + cropArea.height },
          { key: 'br', x: cropArea.x + cropArea.width, y: cropArea.y + cropArea.height },
          { key: 'tc', x: cropArea.x + cropArea.width / 2, y: cropArea.y },
          { key: 'bc', x: cropArea.x + cropArea.width / 2, y: cropArea.y + cropArea.height },
          { key: 'lc', x: cropArea.x, y: cropArea.y + cropArea.height / 2 },
          { key: 'rc', x: cropArea.x + cropArea.width, y: cropArea.y + cropArea.height / 2 },
        ];

        const handleHit = handles.find(h => Math.hypot(h.x - mouseX, h.y - mouseY) < 10);
        if (handleHit) {
          setCropHandle(handleHit.key);
          canvas.style.cursor = 'grabbing';
          return;
        }

        // If clicked inside crop area -> move
        if (mouseX >= cropArea.x && mouseX <= cropArea.x + cropArea.width && mouseY >= cropArea.y && mouseY <= cropArea.y + cropArea.height) {
          setCropHandle('move');
          setLastMousePos({ x: mouseX, y: mouseY });
          canvas.style.cursor = 'grabbing';
          return;
        }

        // Otherwise start a new crop
        setCropStart({ x: mouseX, y: mouseY });
        setCropArea(null);
        canvas.style.cursor = 'crosshair';
        return;
      }

      // No existing crop - start new
      setCropStart({ x: mouseX, y: mouseY });
      setCropArea(null);
      canvas.style.cursor = 'crosshair';
      return;
    }

    // Handle line drawing mode first
    if (drawingMode !== DRAWING_MODES.NONE) {
      if (!isDrawingLine) {
        // Start drawing a new line
        setLineStart({ x: imageX, y: imageY });
        setTempLineEnd({ x: imageX, y: imageY });
        setIsDrawingLine(true);
        canvas.style.cursor = 'crosshair';
        return;
      }
    }

    // Check if clicking on a landmark with optimized collision detection
    let clickedLandmark = null;
    let minDistance = Infinity;

    Object.entries(landmarks).forEach(([key, pos]) => {
      const dx = pos.x - imageX;
      const dy = pos.y - imageY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < 25 && distance < minDistance) { // 25px click tolerance in image space
        clickedLandmark = key;
        minDistance = distance;
      }
    });

    // Check if clicking on line endpoint handle for editing
    if (selectedLine && !readOnly) {
      const line = lines.find(l => l.id === selectedLine);
      if (line) {
        // Check if clicking on start point handle
        const startScreenX = transform.offsetX + line.start.x * transform.scale;
        const startScreenY = transform.offsetY + line.start.y * transform.scale;
        const startDx = startScreenX - mouseX;
        const startDy = startScreenY - mouseY;
        const startDistance = Math.sqrt(startDx * startDx + startDy * startDy);

        // Check if clicking on end point handle
        const endScreenX = transform.offsetX + line.end.x * transform.scale;
        const endScreenY = transform.offsetY + line.end.y * transform.scale;
        const endDx = endScreenX - mouseX;
        const endDy = endScreenY - mouseY;
        const endDistance = Math.sqrt(endDx * endDx + endDy * endDy);

        if (startDistance < 10) {
          setEditingPoint('start');
          canvas.style.cursor = 'grabbing';
          return;
        } if (endDistance < 10) {
          setEditingPoint('end');
          canvas.style.cursor = 'grabbing';
          return;
        } if (startDistance > 15 && endDistance > 15) {
          // Clicked outside handles, check if clicking on the actual line
          const innerCanvas = canvasRef.current;
          const ctx = innerCanvas.getContext('2d');
          ctx.save();

          ctx.translate(transform.offsetX, transform.offsetY);
          ctx.scale(transform.scale, transform.scale);

          // Create path for line click detection
          ctx.beginPath();
          if (line.type === 'curved') {
            const midX = (line.start.x + line.end.x) / 2;
            const midY = (line.start.y + line.end.y) / 2;
            const controlX = midX + (line.start.y - line.end.y) * 0.3;
            const controlY = midY + (line.end.x - line.start.x) * 0.3;

            ctx.moveTo(line.start.x, line.start.y);
            ctx.quadraticCurveTo(controlX, controlY, line.end.x, line.end.y);
          } else {
            ctx.moveTo(line.start.x, line.start.y);
            ctx.lineTo(line.end.x, line.end.y);
          }

          ctx.restore();

          // Check if mouse point is near the line path with some tolerance
          let closeToLine = false;
          const tolerance = 5; // pixels tolerance for line clicking

          // Simple distance-based line hit detection
          if (line.type === 'straight') {
            const dx = line.end.x - line.start.x;
            const dy = line.end.y - line.start.y;
            const length = Math.sqrt(dx * dx + dy * dy);
            const ux = dx / length;
            const uy = dy / length;

            const vx = imageX - line.start.x;
            const vy = imageY - line.start.y;
            const t = Math.max(0, Math.min(length, vx * ux + vy * uy));

            const projX = line.start.x + t * ux;
            const projY = line.start.y + t * uy;

            const dist = Math.sqrt((imageX - projX) ** 2 + (imageY - projY) ** 2);
            if (dist <= tolerance) {
              closeToLine = true;
            }
          } else {
            for (let t = 0; t <= 1; t += 0.1) {
              const midX = (line.start.x + line.end.x) / 2;
              const midY = (line.start.y + line.end.y) / 2;
              const controlX = midX + (line.start.y - line.end.y) * 0.3;
              const controlY = midY + (line.end.x - line.start.x) * 0.3;

              const t2 = t * t;
              const t3 = 1 - t;
              const t4 = t3 * t3;

              const cx = t4 * line.start.x + 2 * t3 * t * controlX + t2 * line.end.x;
              const cy = t4 * line.start.y + 2 * t3 * t * controlY + t2 * line.end.y;

              const dist = Math.sqrt((imageX - cx) ** 2 + (imageY - cy) ** 2);
              if (dist <= tolerance) {
                closeToLine = true;
                break;
              }
            }
          }

          if (!closeToLine) {
            // Clicked outside the selected line, deselect it
            setSelectedLine(null);
            setEditingPoint(null);
          }
        }
      }
      return;
    }

    if (clickedLandmark) {
      setSelectedLandmark(clickedLandmark);
      if (!readOnly) {
        setDraggedLandmark(clickedLandmark);
      }
      canvas.style.cursor = 'grabbing';
    } else {
      setSelectedLandmark(null);
      // Check if clicking on a line for selection
      let clickedLine = null;
      let minLineDistance = Infinity;

      lines.forEach(line => {
        let closestDistance = Infinity;

        if (line.type === 'straight') {
          // Distance from point to line segment algorithm
          const dx = line.end.x - line.start.x;
          const dy = line.end.y - line.start.y;
          const length = Math.sqrt(dx * dx + dy * dy);
          const ux = dx / length;
          const uy = dy / length;

          const vx = imageX - line.start.x;
          const vy = imageY - line.start.y;
          const t = Math.max(0, Math.min(length, vx * ux + vy * uy));

          const projX = line.start.x + t * ux;
          const projY = line.start.y + t * uy;

          const dist = Math.sqrt((imageX - projX) ** 2 + (imageY - projY) ** 2);
          closestDistance = dist;
        } else {
          // For curved lines, sample points along the curve
          for (let t = 0; t <= 1; t += 0.1) {
            const midX = (line.start.x + line.end.x) / 2;
            const midY = (line.start.y + line.end.y) / 2;
            const controlX = midX + (line.start.y - line.end.y) * 0.3;
            const controlY = midY + (line.end.x - line.start.x) * 0.3;

            const t2 = t * t;
            const t3 = 1 - t;
            const t4 = t3 * t3;

            const cx = t4 * line.start.x + 2 * t3 * t * controlX + t2 * line.end.x;
            const cy = t4 * line.start.y + 2 * t3 * t * controlY + t2 * line.end.y;

            const dist = Math.sqrt((imageX - cx) ** 2 + (imageY - cy) ** 2);
            if (dist < closestDistance) closestDistance = dist;
          }
        }

        if (closestDistance < 10 && closestDistance < minLineDistance) { // 10px tolerance for line selection
          clickedLine = line.id;
          minLineDistance = closestDistance;
        }
      });

      if (clickedLine) {
        setSelectedLine(clickedLine);
        setEditingPoint(null);
        canvas.style.cursor = 'move';
      } else {
        setSelectedLine(null);
        setEditingPoint(null);
        // Start canvas panning on any click (not just shift or right click)
        setIsDraggingCanvas(true);
        canvas.style.cursor = 'grabbing';
      }
    }
  }, [imageLoaded, readOnly, landmarks, transform, drawingMode, isDrawingLine, isCropping, lines, selectedLine, cropArea]);

  const handleMouseMove = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Convert screen coordinates to image coordinates
    const imageX = (mouseX - transform.offsetX) / transform.scale;
    const imageY = (mouseY - transform.offsetY) / transform.scale;

    // Handle interactive crop handles
    if (cropHandle && cropArea) {
      const minSize = 20;
      const newArea = { ...cropArea };
      const dx = mouseX;
      const dy = mouseY;

      const left = cropArea.x;
      const top = cropArea.y;
      const right = cropArea.x + cropArea.width;
      const bottom = cropArea.y + cropArea.height;

      switch (cropHandle) {
        case 'tl':
          newArea.x = Math.min(dx, right - minSize);
          newArea.y = Math.min(dy, bottom - minSize);
          newArea.width = Math.max(minSize, right - newArea.x);
          newArea.height = Math.max(minSize, bottom - newArea.y);
          break;
        case 'tr':
          newArea.y = Math.min(dy, bottom - minSize);
          newArea.width = Math.max(minSize, dx - left);
          newArea.x = left;
          newArea.height = Math.max(minSize, bottom - newArea.y);
          break;
        case 'bl':
          newArea.x = Math.min(dx, right - minSize);
          newArea.width = Math.max(minSize, right - newArea.x);
          newArea.height = Math.max(minSize, dy - top);
          newArea.y = top;
          break;
        case 'br':
          newArea.width = Math.max(minSize, dx - left);
          newArea.height = Math.max(minSize, dy - top);
          newArea.x = left;
          newArea.y = top;
          break;
        case 'tc':
          newArea.y = Math.min(dy, bottom - minSize);
          newArea.height = Math.max(minSize, bottom - newArea.y);
          break;
        case 'bc':
          newArea.height = Math.max(minSize, dy - top);
          break;
        case 'lc':
          newArea.x = Math.min(dx, right - minSize);
          newArea.width = Math.max(minSize, right - newArea.x);
          break;
        case 'rc':
          newArea.width = Math.max(minSize, dx - left);
          break;
        case 'move': {
          const movedX = mouseX - lastMousePos.x;
          const movedY = mouseY - lastMousePos.y;
          newArea.x = Math.max(0, Math.min(canvas.width - newArea.width, newArea.x + movedX));
          newArea.y = Math.max(0, Math.min(canvas.height - newArea.height, newArea.y + movedY));
          setLastMousePos({ x: mouseX, y: mouseY });
          break;
        }
        default:
          break;
      }
      setCropArea(newArea);
      return;
    }

    if (draggedLandmark) {
      // Update dragged landmark position smoothly
      setLandmarks(prev => ({
        ...prev,
        [draggedLandmark]: { x: imageX, y: imageY }
      }));
    } else if (editingPoint && selectedLine) {
      // Update line endpoint while dragging
      setLines(prev => prev.map(line => {
        if (line.id === selectedLine) {
          if (editingPoint === 'start') {
            return { ...line, start: { x: imageX, y: imageY } };
          } if (editingPoint === 'end') {
            return { ...line, end: { x: imageX, y: imageY } };
          }
        }
        return line;
      }));
    } else if (isDraggingCanvas) {
      // Pan the canvas smoothly
      const dx = mouseX - lastMousePos.x;
      const dy = mouseY - lastMousePos.y;

      setTransform(prev => ({
        ...prev,
        offsetX: prev.offsetX + dx,
        offsetY: prev.offsetY + dy,
      }));

      setLastMousePos({ x: mouseX, y: mouseY });
    } else if (isDrawingLine && lineStart) {
      // Update the temporary line end point during drawing
      setTempLineEnd({ x: imageX, y: imageY });
    }

    // Update crop area while resizing
    if (cropStart) {
      const x = Math.min(cropStart.x, mouseX);
      const y = Math.min(cropStart.y, mouseY);
      const width = Math.abs(mouseX - cropStart.x);
      const height = Math.abs(mouseY - cropStart.y);
      setCropArea({ x, y, width, height });
    }

    // Update cursor based on what's under the mouse
    let cursor = 'default'; // Default cursor for image

    // Check if mouse is over a landmark
    Object.entries(landmarks).forEach(([key, pos]) => {
      const dx = pos.x - imageX;
      const dy = pos.y - imageY;
      const distance = Math.sqrt(dx * dx + dy * dy);
      if (distance < 25 && distance >= 0) { // 25px click tolerance in image space
        cursor = 'pointer'; // Pointer cursor over landmarks
      }
    });

    // Override cursor for other states
    if (draggedLandmark || isDraggingCanvas) {
      cursor = 'grabbing';
    } else if (drawingMode !== DRAWING_MODES.NONE) {
      cursor = 'crosshair';
    } else if (cropStart) {
      cursor = 'crosshair';
    }

    canvas.style.cursor = cursor;
  }, [draggedLandmark, isDraggingCanvas, transform, lastMousePos, isDrawingLine, lineStart, landmarks, drawingMode, cropStart, cropHandle, cropArea, editingPoint, selectedLine]);

  const handleMouseUp = useCallback(() => {
    if (draggedLandmark) {
      // Finalize landmark position with callback
      if (onLandmarkUpdate) {
        onLandmarkUpdate(landmarks);
      }
      setDraggedLandmark(null);
    }

    if (editingPoint) {
      // Finish editing line endpoint
      setEditingPoint(null);
    }

    if (isDraggingCanvas) {
      setIsDraggingCanvas(false);
    }

    if (isDrawingLine && lineStart && tempLineEnd) {
      // Complete the line drawing
      const newLine = createLine(lineStart, tempLineEnd, drawingMode);
      setLines(prev => [...prev, newLine]);

      // Reset drawing state
      setIsDrawingLine(false);
      setLineStart(null);
      setTempLineEnd(null);

      console.log('Line added:', newLine);
    }

    // Finalize cropping if active
    if (isCropping && cropStart && cropArea && cropArea.width > 5 && cropArea.height > 5) {
      // Save current image for undo
      try {
        const currentSrc = image?.src || null;
        if (currentSrc) setImageHistory(prev => [...prev, currentSrc]);
      } catch (err) {
        // ignore
      }

      // Convert cropArea (screen coords) to image coordinates
      const sx = (cropArea.x - transform.offsetX) / transform.scale;
      const sy = (cropArea.y - transform.offsetY) / transform.scale;
      const sw = cropArea.width / transform.scale;
      const sh = cropArea.height / transform.scale;

      // Ensure within image bounds
      const srcX = Math.max(0, Math.min(image.width, sx));
      const srcY = Math.max(0, Math.min(image.height, sy));
      const srcW = Math.max(1, Math.min(image.width - srcX, sw));
      const srcH = Math.max(1, Math.min(image.height - srcY, sh));

      // Draw cropped image to temp canvas
      const off = document.createElement('canvas');
      off.width = Math.round(srcW);
      off.height = Math.round(srcH);
      const offCtx = off.getContext('2d');
      offCtx.drawImage(image, srcX, srcY, srcW, srcH, 0, 0, off.width, off.height);

      // Convert to blob and call onCropSave if provided
      if (onCropSave && typeof onCropSave === 'function') {
        try {
          off.toBlob(async (blob) => {
            if (blob) {
              try {
                await onCropSave(blob);
              } catch (err) {
                console.error('onCropSave error:', err);
              }

              // Also update local preview with cropped image
              const croppedDataUrl = off.toDataURL('image/png');
              const newImg = new Image();
              newImg.onload = () => {
                setImage(newImg);
                setImageLoaded(true);

                // Fit new image into canvas
                const canvas = canvasRef.current;
                const scaleX = canvas.width / newImg.width;
                const scaleY = canvas.height / newImg.height;
                const scale = Math.min(scaleX, scaleY, 1);
                setTransform({
                  scale,
                  offsetX: (canvas.width - newImg.width * scale) / 2,
                  offsetY: (canvas.height - newImg.height * scale) / 2,
                });
              };
              newImg.src = croppedDataUrl;
            }
          }, 'image/png');
        } catch (err) {
          console.error('Failed to export cropped blob:', err);
        }
      } else {
        // Fallback: just update local preview
        const croppedDataUrl = off.toDataURL('image/png');
        const newImg = new Image();
        newImg.onload = () => {
          setImage(newImg);
          setImageLoaded(true);

          // Fit new image into canvas
          const canvas = canvasRef.current;
          const scaleX = canvas.width / newImg.width;
          const scaleY = canvas.height / newImg.height;
          const scale = Math.min(scaleX, scaleY, 1);
          setTransform({
            scale,
            offsetX: (canvas.width - newImg.width * scale) / 2,
            offsetY: (canvas.height - newImg.height * scale) / 2,
          });
        };
        newImg.src = croppedDataUrl;
      }

      // Reset crop state
      setIsCropping(false);
      setCropStart(null);
      setCropArea(null);
      setCropHandle(null);
    } else {
      // Reset crop start if nothing selected
      setCropStart(null);
      setCropHandle(null);
    }
  }, [draggedLandmark, isDraggingCanvas, landmarks, onLandmarkUpdate, isDrawingLine, lineStart, tempLineEnd, drawingMode, isCropping, cropStart, cropArea, transform, image, editingPoint, onCropSave]);

  const handleDoubleClick = useCallback((e) => {
    if (readOnly) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;

    // Proper coordinate transformation accounting for rotation
    let imageX; let imageY;
    
    if (imageRotation === 0) {
      imageX = (mouseX - transform.offsetX) / transform.scale;
      imageY = (mouseY - transform.offsetY) / transform.scale;
    } else {
      // Account for rotation in coordinate conversion
      const centerX = transform.offsetX + (image.width * transform.scale) / 2;
      const centerY = transform.offsetY + (image.height * transform.scale) / 2;
      
      const dx = mouseX - centerX;
      const dy = mouseY - centerY;
      
      const angle = (-imageRotation * Math.PI) / 180;
      const rotatedX = dx * Math.cos(angle) - dy * Math.sin(angle);
      const rotatedY = dx * Math.sin(angle) + dy * Math.cos(angle);
      
      imageX = (rotatedX / transform.scale) + image.width / 2;
      imageY = (rotatedY / transform.scale) + image.height / 2;
    }

    setAddPointPosition({ x: imageX, y: imageY });
    setShowLandmarkDialog(true);
  }, [readOnly, transform, imageRotation, image]);

  // AI detection integration
  const handleAIDetect = useCallback(async (model) => {
    if (!onLandmarkDetect) return;
    
    setIsAIDetecting(true);
    setShowAIModelDialog(false);
    
    try {
      await onLandmarkDetect(model);
    } finally {
      setIsAIDetecting(false);
    }
  }, [onLandmarkDetect]);

  // Open AI model selector
  const handleOpenAISelector = useCallback(() => {
    setShowAIModelDialog(true);
  }, []);

  // Zoom controls with smooth transitions
  const handleZoomIn = useCallback(() => {
    const canvas = canvasRef.current;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const zoomFactor = 1.2;
    const newScale = Math.min(5, transform.scale * zoomFactor);

    const scaleChange = newScale / transform.scale;
    const newOffsetX = centerX - (centerX - transform.offsetX) * scaleChange;
    const newOffsetY = centerY - (centerY - transform.offsetY) * scaleChange;

    setTransform({
      scale: newScale,
      offsetX: newOffsetX,
      offsetY: newOffsetY,
    });
  }, [transform]);

  const handleZoomOut = useCallback(() => {
    const canvas = canvasRef.current;
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const zoomFactor = 1 / 1.2;
    const newScale = Math.max(0.1, transform.scale * zoomFactor);

    const scaleChange = newScale / transform.scale;
    const newOffsetX = centerX - (centerX - transform.offsetX) * scaleChange;
    const newOffsetY = centerY - (centerY - transform.offsetY) * scaleChange;

    setTransform({
      scale: newScale,
      offsetX: newOffsetX,
      offsetY: newOffsetY,
    });
  }, [transform]);

  const handleResetView = useCallback(() => {
    if (!image) return;

    const canvas = canvasRef.current;
    const scaleX = canvas.width / image.width;
    const scaleY = canvas.height / image.height;
    const scale = Math.min(scaleX, scaleY, 1);

    setTransform({
      scale,
      offsetX: (canvas.width - image.width * scale) / 2,
      offsetY: (canvas.height - image.height * scale) / 2,
    });
  }, [image]);

  // Rotate image handler - compute new rotation and update view using functional updater
  const handleRotateImage = useCallback(() => {
    if (!image) return;

    setImageRotation((prev) => {
      const newRotation = (prev + 90) % 360;

      // When fitting to canvas, consider swapped dimensions if rotated 90/270
      const canvas = canvasRef.current;
      const effectiveW = (newRotation === 90 || newRotation === 270) ? image.height : image.width;
      const effectiveH = (newRotation === 90 || newRotation === 270) ? image.width : image.height;

      const scaleX = canvas.width / effectiveW;
      const scaleY = canvas.height / effectiveH;
      const scale = Math.min(scaleX, scaleY, 1);

      setTransform({
        scale,
        offsetX: (canvas.width - effectiveW * scale) / 2,
        offsetY: (canvas.height - effectiveH * scale) / 2,
      });

      console.log('Image rotated to:', newRotation, 'degrees');
      return newRotation;
    });
  }, [image]);

  // Crop cancel and undo handlers
  const cancelCrop = useCallback(() => {
    setIsCropping(false);
    setCropStart(null);
    setCropArea(null);
  }, []);

  const undoCrop = useCallback(() => {
    if (!imageHistory || imageHistory.length === 0) return;
    const last = imageHistory[imageHistory.length - 1];
    setImageHistory(prev => prev.slice(0, prev.length - 1));
    if (!last) return;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImage(img);
      setImageLoaded(true);

      const canvas = canvasRef.current;
      const scaleX = canvas.width / img.width;
      const scaleY = canvas.height / img.height;
      const scale = Math.min(scaleX, scaleY, 1);
      setTransform({
        scale,
        offsetX: (canvas.width - img.width * scale) / 2,
        offsetY: (canvas.height - img.height * scale) / 2,
      });
    };
    img.src = last;
  }, [imageHistory]);

  // Add landmark dialog handler
  const handleAddLandmark = useCallback((landmarkKey) => {
    if (!landmarkKey || landmarks[landmarkKey] || !addPointPosition) return;

    const newLandmarks = {
      ...landmarks,
      [landmarkKey]: { x: addPointPosition.x, y: addPointPosition.y }
    };

    setLandmarks(newLandmarks);
    setShowLandmarkDialog(false);
    setAddPointPosition(null);

    if (onLandmarkUpdate) {
      onLandmarkUpdate(newLandmarks);
    }
  }, [landmarks, addPointPosition, onLandmarkUpdate]);

  // Cleanup animation frame on unmount
  useEffect(() => () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    }, []);

  return (
    <Box>
      {/* Controls */}
      <Stack
        direction="row"
        spacing={1}
        mb={2}
        alignItems="center"
        sx={{
          flexWrap: 'wrap',
          '& > *': { flexShrink: 0 },
          gap: 1
        }}
      >
        <ToolButton title="زوم به داخل" size="small" onClick={handleZoomIn} disabled={!imageLoaded} sx={{ border: '1px solid', borderColor: 'divider' }}>
          <Iconify icon="solar:magnifer-zoom-in-linear" width={18} />
        </ToolButton>

        <ToolButton title="زوم به بیرون" size="small" onClick={handleZoomOut} disabled={!imageLoaded} sx={{ border: '1px solid', borderColor: 'divider' }}>
          <Iconify icon="solar:magnifer-zoom-out-linear" width={18} />
        </ToolButton>

        <ToolButton title="نمای اولیه" size="small" onClick={handleResetView} disabled={!imageLoaded} sx={{ border: '1px solid', borderColor: 'divider' }}>
          <Iconify icon="solar:refresh-linear" width={18} />
        </ToolButton>

        {!readOnly && (
          <Stack direction="row" spacing={1}>
            {/* Line drawing tools */}
            <ToolButton
              title="خط مستقیم"
              size="small"
              onClick={() => setDrawingMode(drawingMode === DRAWING_MODES.STRAIGHT_LINE ? DRAWING_MODES.NONE : DRAWING_MODES.STRAIGHT_LINE)}
              disabled={!imageLoaded}
              color={drawingMode === DRAWING_MODES.STRAIGHT_LINE ? 'primary' : 'default'}
              sx={{
                border: drawingMode === DRAWING_MODES.STRAIGHT_LINE ? '2px solid' : '1px solid',
                borderColor: drawingMode === DRAWING_MODES.STRAIGHT_LINE ? 'primary.main' : 'divider'
              }}
            >
              <LineIcon />
            </ToolButton>

            <ToolButton
              title="خط منحنی"
              size="small"
              onClick={() => setDrawingMode(drawingMode === DRAWING_MODES.CURVED_LINE ? DRAWING_MODES.NONE : DRAWING_MODES.CURVED_LINE)}
              disabled={!imageLoaded}
              color={drawingMode === DRAWING_MODES.CURVED_LINE ? 'primary' : 'default'}
              sx={{
                border: drawingMode === DRAWING_MODES.CURVED_LINE ? '2px solid' : '1px solid',
                borderColor: drawingMode === DRAWING_MODES.CURVED_LINE ? 'primary.main' : 'divider'
              }}
            >
              <CurveIcon />
            </ToolButton>

            {lines.length === 0 ? (
              <span>
                <IconButton size="small" disabled sx={{ border: '1px solid', borderColor: 'divider' }}>
                  <EraserIcon />
                </IconButton>
              </span>
            ) : (
              <ToolButton title="پاک کردن خطوط" size="small" onClick={() => setLines([])} sx={{ border: '1px solid', borderColor: 'divider' }}>
                <EraserIcon />
              </ToolButton>
            )}

            <ToolButton title="نمایش خطکش" size="small" onClick={() => setShowRuler(!showRuler)} disabled={!imageLoaded} color={showRuler ? 'secondary' : 'default'} sx={{ border: showRuler ? '2px solid' : '1px solid', borderColor: showRuler ? 'secondary.main' : 'divider' }}>
              <RulerIcon />
            </ToolButton>

            <ToolButton title="برش تصویر" size="small" onClick={() => setIsCropping(!isCropping)} disabled={!imageLoaded} color={isCropping ? 'warning' : 'default'} sx={{ border: isCropping ? '2px solid' : '1px solid', borderColor: isCropping ? 'warning.main' : 'divider' }}>
              <ScissorsIcon />
            </ToolButton>

            <ToolButton title="چرخش تصویر" size="small" onClick={handleRotateImage} disabled={!imageLoaded} sx={{ border: '1px solid', borderColor: 'divider' }}>
              <RotateIcon />
            </ToolButton>

            {/* Crop cancel & undo buttons */}
            {isCropping && (
              <ToolButton title="لغو برش" size="small" onClick={cancelCrop} sx={{ border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="caption">انصراف</Typography>
              </ToolButton>
            )}

            {imageHistory.length > 0 && (
              <ToolButton title="بازگشت برش" size="small" onClick={undoCrop} sx={{ border: '1px solid', borderColor: 'divider' }}>
                <Typography variant="caption">بازگشت</Typography>
              </ToolButton>
            )}

            <Box sx={{ width: '2px', bgcolor: 'divider', mx: 0.5, height: 28 }} />

            <ToolButton 
              title="شناسایی خودکار با هوش مصنوعی" 
              size="small" 
              color="primary" 
              onClick={handleOpenAISelector} 
              disabled={!imageLoaded || isAIDetecting} 
              sx={{ 
                border: '1px solid', 
                borderColor: 'primary.main',
                bgcolor: isAIDetecting ? 'action.selected' : 'transparent'
              }}
            >
              {isAIDetecting ? (
                <Iconify icon="eos-icons:loading" width={18} />
              ) : (
                <Iconify icon="solar:cpu-bolt-bold" width={18} />
              )}
            </ToolButton>

            <ToolButton title="افزودن نقطه" size="small" onClick={() => setShowLandmarkDialog(true)} disabled={!imageLoaded} sx={{ border: '1px solid', borderColor: 'divider' }}>
              <Iconify icon="solar:point-on-map-linear" width={18} />
            </ToolButton>
          </Stack>
        )}
      </Stack>

      {/* Canvas viewer */}
      <Paper sx={{ p: 1, display: 'flex', justifyContent: 'center', width: '100%' }}>
        <Box
          sx={{
            position: 'relative',
            width: '100%',
            minHeight: '400px',
            maxHeight: '700px',
            border: '1px solid #ccc',
            borderRadius: 1,
            overflow: 'hidden'
          }}
        >
          <canvas
            ref={canvasRef}
            width={1600}
            height={1200}
            style={{
              width: '100%',
              height: 'auto',
              maxHeight: '700px',
              display: 'block'
            }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onDoubleClick={handleDoubleClick}
          />

          {!imageLoaded && (
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                textAlign: 'center',
                zIndex: 1
              }}
            >
              <Iconify icon="solar:gallery-broken" width={48} />
              <Typography variant="body2" color="text.secondary">
                تصویر در حال بارگذاری...
              </Typography>
            </Box>
          )}
        </Box>
      </Paper>

      {/* Landmark Info */}
      {Object.keys(landmarks).length > 0 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            نقاط علامت‌گذاری شده ({Object.keys(landmarks).length})
          </Typography>

          <Grid container spacing={2}>
            {Object.entries(landmarks).map(([key, position]) => {
              const landmark = CEPHALOMETRIC_LANDMARKS[key];
              if (!landmark || !position) return null;

              return (
                <Grid item xs={12} sm={6} md={4} lg={3} key={key}>
                  <Box
                    sx={{
                      p: 2,
                      border: '1px solid #ddd',
                      borderRadius: 1,
                      backgroundColor: selectedLandmark === key ? '#f5f5f5' : 'transparent',
                      cursor: 'pointer',
                    }}
                    onClick={() => setSelectedLandmark(key)}
                  >
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          borderRadius: '50%',
                          backgroundColor: landmark.color,
                          border: '1px solid #000'
                        }}
                      />
                      <Box>
                        <Typography variant="subtitle2">
                          {key} - {landmark.name}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          ({Math.round(position.x)}, {Math.round(position.y)})
                        </Typography>
                      </Box>
                    </Stack>
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </Paper>
      )}

      {/* Instructions */}
      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2">
          <strong>راهنمایی:</strong>
          {!readOnly ? (
            <>
              • برای رسم خط مستقیم، روی دکمه خط مستقیم کلیک کرده و در تصویر شروع کنید
              • برای رسم خط منحنی، روی دکمه خط منحنی کلیک کرده و در تصویر شروع کنید
              • برای جابجایی نقاط، روی نقطه کلیک کرده و نگه دارید
              • برای افزودن نقطه جدید، دوبار کلیک کنید و نقطه را انتخاب کنید
              • از دکمه‌های پاک کردن برای حذف خطوط استفاده کنید
              • از دکمه‌های زوم برای بزرگنمایی استفاده کنید
              • از دکمه شناسایی AI برای تشخیص خودکار استفاده کنید
              • از Shift + کلیک برای پنینگ تصویر استفاده کنید
            </>
          ) : (
            'این تصویر در حالت فقط خواندنی است'
          )}
        </Typography>
      </Alert>

      {/* AI Model Selector Dialog */}
      <AIModelSelector
        open={showAIModelDialog}
        onClose={() => setShowAIModelDialog(false)}
        onDetect={handleAIDetect}
        currentModel={selectedAIModel}
        isDetecting={isAIDetecting}
      />

      {/* Add Landmark Dialog */}
      <Dialog
        open={showLandmarkDialog}
        onClose={() => {
          setShowLandmarkDialog(false);
          setAddPointPosition(null);
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          {addPointPosition ? 'انتخاب لندمارک برای افزودن' : 'انتخاب لندمارک'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={1}>
            {Object.entries(CEPHALOMETRIC_LANDMARKS).map(([key, landmark]) => (
              <Grid item xs={6} sm={4} key={key}>
                <Button
                  fullWidth
                  variant="outlined"
                  sx={{
                    p: 1,
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center'
                  }}
                  onClick={() => handleAddLandmark(key)}
                  disabled={landmarks[key]}
                >
                  <Typography variant="caption" fontWeight="bold">
                    {key}
                  </Typography>
                  <Typography variant="caption" sx={{ mt: 0.5, textAlign: 'center' }}>
                    {landmark.name}
                  </Typography>
                </Button>
              </Grid>
            ))}
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setShowLandmarkDialog(false);
            setAddPointPosition(null);
          }}>انصراف</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CephalometricLandmarkViewer;
