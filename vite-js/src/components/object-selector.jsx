import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';

import { Iconify } from 'src/components/iconify';

// Smart object selection algorithm - finds complete objects but only renders boundaries
const smartObjectSelection = (imageData, startX, startY, width, height, initialTolerance = 3) => {
  const {data} = imageData;

  // Color distance function - simple RGB Euclidean distance
  const colorDistance = (r1, g1, b1, r2, g2, b2) => {
    const dr = r1 - r2;
    const dg = g1 - g2;
    const db = b1 - b2;
    return Math.sqrt(dr * dr + dg * dg + db * db);
  };

  // Get color at starting point
  const startIndex = (startY * width + startX) * 4;
  const startR = data[startIndex];
  const startG = data[startIndex + 1];
  const startB = data[startIndex + 2];

  // Adaptive tolerance approach to find complete objects
  let currentTolerance = initialTolerance;
  let selectedPixels = new Set();
  const directions = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // 4-directional

  // Try with increasing tolerance until we get a reasonable selection
  while (currentTolerance <= 50 && selectedPixels.size < 50) { // Minimum 50 pixels for a valid object
    const visited = new Set();
    const queue = [];
    selectedPixels = new Set();

    // Start flood fill
    queue.push([startX, startY]);
    visited.add(`${startX},${startY}`);

    // Allow up to 2000 pixels for complete object detection
    while (queue.length > 0 && selectedPixels.size < 2000) {
      const [x, y] = queue.shift();
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];

      // Check if pixel is within current tolerance
      if (colorDistance(startR, startG, startB, r, g, b) <= currentTolerance) {
        selectedPixels.add(`${x},${y}`);

        // Check neighbors
        for (const [dx, dy] of directions) {
          const nx = x + dx;
          const ny = y + dy;

          if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
            const key = `${nx},${ny}`;
            if (!visited.has(key)) {
              visited.add(key);
              queue.push([nx, ny]);
            }
          }
        }
      }
    }

    // If we found enough pixels, break
    if (selectedPixels.size >= 50) {
      break;
    }

    // Increase tolerance for next attempt
    currentTolerance += 2;
  }

  return selectedPixels;
};

// Constrained object selection within bounding box - optimized for dental X-rays
const smartObjectSelectionConstrained = (imageData, startX, startY, width, height, initialTolerance = 3, bbox) => {
  const {data} = imageData;

  // For dental X-rays, use a simpler but more effective approach
  // Instead of complex flood fill, use adaptive thresholding within the bounding box

  // Bounding box constraints
  const bboxX = Math.floor(bbox.x);
  const bboxY = Math.floor(bbox.y);
  const bboxWidth = Math.floor(bbox.width);
  const bboxHeight = Math.floor(bbox.height);
  const bboxX2 = bboxX + bboxWidth;
  const bboxY2 = bboxY + bboxHeight;

  // Collect all pixels within the bounding box
  const bboxPixels = [];
  for (let y = bboxY; y < bboxY2; y++) {
    for (let x = bboxX; x < bboxX2; x++) {
      const index = (y * width + x) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const gray = (r + g + b) / 3;
      bboxPixels.push({ x, y, gray, r, g, b });
    }
  }

  // Calculate statistics within the bounding box
  const grayValues = bboxPixels.map(p => p.gray);
  const mean = grayValues.reduce((a, b) => a + b, 0) / grayValues.length;
  const std = Math.sqrt(grayValues.reduce((a, b) => a + (b - mean)**2, 0) / grayValues.length);

  // For dental structures, we want to find areas that are significantly different from the background
  // This could be either darker (fillings, lesions) or lighter areas
  const selectedPixels = new Set();

  // Adaptive threshold based on local statistics
  const lowerThreshold = mean - std * 0.5; // Areas darker than background
  const upperThreshold = mean + std * 0.5; // Areas lighter than background

  // Also consider the starting point to determine if we're looking for dark or light areas
  const startIndex = (startY * width + startX) * 4;
  const startGray = (data[startIndex] + data[startIndex + 1] + data[startIndex + 2]) / 3;
  const isStartingPointDark = startGray < mean;

  // Select pixels based on adaptive thresholding
  for (const pixel of bboxPixels) {
    let shouldSelect = false;

    if (isStartingPointDark) {
      // If starting point is dark, look for other dark areas (fillings, lesions)
      shouldSelect = pixel.gray < lowerThreshold;
    } else {
      // If starting point is light, look for other light areas
      shouldSelect = pixel.gray > upperThreshold;
    }

    // Also include pixels that are similar to the starting point
    const grayDiff = Math.abs(pixel.gray - startGray);
    if (grayDiff < std * 0.3) { // Within 30% of std deviation
      shouldSelect = true;
    }

    if (shouldSelect) {
      selectedPixels.add(`${pixel.x},${pixel.y}`);
    }
  }

  // If we have very few pixels, fall back to selecting a larger portion of the bounding box
  if (selectedPixels.size < 50) {
    // Select pixels within a smaller region around the center
    const centerX = Math.floor(bboxX + bboxWidth / 2);
    const centerY = Math.floor(bboxY + bboxHeight / 2);
    const regionSize = Math.min(bboxWidth, bboxHeight) * 0.6; // 60% of smaller dimension

    const regionX1 = Math.max(bboxX, centerX - regionSize / 2);
    const regionY1 = Math.max(bboxY, centerY - regionSize / 2);
    const regionX2 = Math.min(bboxX2, centerX + regionSize / 2);
    const regionY2 = Math.min(bboxY2, centerY + regionSize / 2);

    for (let y = regionY1; y < regionY2; y++) {
      for (let x = regionX1; x < regionX2; x++) {
        selectedPixels.add(`${Math.floor(x)},${Math.floor(y)}`);
      }
    }
  }

  return selectedPixels;
};

// Extract boundary from selected pixels
const extractBoundary = (selectedPixels, width, height) => {
  const boundary = new Set();

  for (const pixelKey of selectedPixels) {
    const [x, y] = pixelKey.split(',').map(Number);

    // Check if this pixel is on the boundary (has at least one non-selected neighbor)
    const neighbors = [
      [x-1, y], [x+1, y], [x, y-1], [x, y+1],
      [x-1, y-1], [x-1, y+1], [x+1, y-1], [x+1, y+1]
    ];

    let isBoundary = false;
    for (const [nx, ny] of neighbors) {
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) {
        isBoundary = true;
        break;
      }
      if (!selectedPixels.has(`${nx},${ny}`)) {
        isBoundary = true;
        break;
      }
    }

    if (isBoundary) {
      boundary.add(pixelKey);
    }
  }

  return boundary;
};

// Export the functions for use in other components
export { extractBoundary, smartObjectSelectionConstrained };

// ----------------------------------------------------------------------

export function ObjectSelector({
  imageUrl,
  detections = [], // AI detections to constrain selection within
  onSelectionComplete,
  selectionColor = '#FF6B35',
  outlineColor = '#FFFFFF',
}) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [isImageLoaded, setIsImageLoaded] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isSelecting, setIsSelecting] = useState(false);
  const [markedPoint, setMarkedPoint] = useState(null);
  const [selectedPixels, setSelectedPixels] = useState(new Set());
  const [boundaryPixels, setBoundaryPixels] = useState(new Set());
  const [tolerance, setTolerance] = useState(5);
  const [showSelection, setShowSelection] = useState(true);

  // Load image
  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      imageRef.current = img;
      setIsImageLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Draw canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;

    if (!canvas || !image || !isImageLoaded) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    canvas.width = rect.width;
    canvas.height = rect.height;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const scale = Math.min(rect.width / image.width, rect.height / image.height) * zoom;
    const scaledWidth = image.width * scale;
    const scaledHeight = image.height * scale;
    const x = (rect.width - scaledWidth) / 2 + pan.x;
    const y = (rect.height - scaledHeight) / 2 + pan.y;

    // Draw image
    ctx.drawImage(image, x, y, scaledWidth, scaledHeight);

    // Draw selection outline only (no fill)
    if (showSelection && boundaryPixels.size > 0) {
      ctx.save();
      ctx.strokeStyle = selectionColor;
      ctx.lineWidth = 1;
      ctx.beginPath();

      // Draw boundary pixels as small dots for outline effect
      for (const pixelKey of boundaryPixels) {
        const [imgX, imgY] = pixelKey.split(',').map(Number);
        const canvasX = imgX * scale + x;
        const canvasY = imgY * scale + y;

        // Draw small dot for each boundary pixel
        ctx.rect(canvasX, canvasY, Math.max(1, scale), Math.max(1, scale));
      }

      ctx.stroke();
      ctx.restore();
    }

    // Draw marked point
    if (markedPoint) {
      const canvasX = markedPoint.x * scale + x;
      const canvasY = markedPoint.y * scale + y;

      ctx.save();
      ctx.strokeStyle = '#FF0000';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 8, 0, 2 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = '#FF0000';
      ctx.beginPath();
      ctx.arc(canvasX, canvasY, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.restore();
    }
  }, [isImageLoaded, zoom, pan, markedPoint, boundaryPixels, showSelection, selectionColor]);

  // Draw canvas when image is loaded or canvas is ready
  useEffect(() => {
    if (isImageLoaded && canvasRef.current) {
      drawCanvas();
    }
  }, [isImageLoaded, drawCanvas]);

  // Convert canvas coordinates to image coordinates
  const getImageCoordinates = useCallback((canvasX, canvasY) => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image) return null;

    const rect = canvas.getBoundingClientRect();
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

    return { x: Math.round(imageX), y: Math.round(imageY) };
  }, [zoom, pan]);

  // Check if point is within any detection bounding box
  const findDetectionAtPoint = (imageX, imageY) => {
    if (!detections || detections.length === 0) return null;

    for (let i = 0; i < detections.length; i++) {
      const det = detections[i];
      const {bbox} = det;
      if (!bbox) continue;

      const x = bbox.x || bbox.x1 || 0;
      const y = bbox.y || bbox.y1 || 0;
      const width = bbox.width || ((bbox.x2 || 0) - (bbox.x1 || 0)) || 0;
      const height = bbox.height || ((bbox.y2 || 0) - (bbox.y1 || 0)) || 0;

      // Check if point is inside bounding box
      if (imageX >= x && imageX <= x + width && imageY >= y && imageY <= y + height) {
        return {
          index: i,
          bbox: { x, y, width, height },
          detection: det
        };
      }
    }

    return null;
  };

  // Handle canvas click
  const handleCanvasClick = (e) => {
    if (!isImageLoaded || isSelecting) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const canvasX = e.clientX - rect.left;
    const canvasY = e.clientY - rect.top;

    const imageCoords = getImageCoordinates(canvasX, canvasY);
    if (!imageCoords) return;

    // Check if click is within a detection bounding box
    const detectionAtPoint = findDetectionAtPoint(imageCoords.x, imageCoords.y);

    if (!detectionAtPoint) {
      // Click is outside any detection - show message
      console.log('Click outside detection area. Please click within a detected region.');
      return;
    }

    setMarkedPoint(imageCoords);
    performSelection(imageCoords, detectionAtPoint);
  };

  // Perform object selection
  const performSelection = async (point, detectionAtPoint) => {
    setIsSelecting(true);

    try {
      const canvas = canvasRef.current;
      const image = imageRef.current;
      if (!canvas || !image) return;

      // Create temporary canvas to get image data
      const tempCanvas = document.createElement('canvas');
      const tempCtx = tempCanvas.getContext('2d');
      tempCanvas.width = image.width;
      tempCanvas.height = image.height;
      tempCtx.drawImage(image, 0, 0);

      const imageData = tempCtx.getImageData(0, 0, image.width, image.height);

      // Perform smart object selection constrained within detection bounding box
      const selected = smartObjectSelectionConstrained(
        imageData,
        point.x,
        point.y,
        image.width,
        image.height,
        tolerance,
        detectionAtPoint.bbox
      );

      // Extract boundary
      const boundary = extractBoundary(selected, image.width, image.height);

      setSelectedPixels(selected);
      setBoundaryPixels(boundary);

      // Notify parent component
      if (onSelectionComplete) {
        onSelectionComplete({
          markedPoint: point,
          selectedPixels: Array.from(selected).map(key => {
            const [x, y] = key.split(',').map(Number);
            return { x, y };
          }),
          boundaryPixels: Array.from(boundary).map(key => {
            const [x, y] = key.split(',').map(Number);
            return { x, y };
          }),
          tolerance,
          detectionIndex: detectionAtPoint.index,
          detection: detectionAtPoint.detection,
        });
      }

      drawCanvas();
    } catch (error) {
      console.error('Selection error:', error);
    } finally {
      setIsSelecting(false);
    }
  };

  // Clear selection
  const clearSelection = () => {
    setMarkedPoint(null);
    setSelectedPixels(new Set());
    setBoundaryPixels(new Set());
    drawCanvas();
  };

  // Update selection with new tolerance
  const updateTolerance = (newTolerance) => {
    setTolerance(newTolerance);
    if (markedPoint) {
      performSelection(markedPoint);
    }
  };

  return (
    <Stack spacing={2}>
      {/* Controls */}
      <Stack direction="row" spacing={1} alignItems="center" justifyContent="space-between">
        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title="Zoom In">
            <Button
              size="small"
              onClick={() => setZoom(prev => Math.min(prev * 1.2, 5))}
            >
              <Iconify icon="carbon:zoom-in" />
            </Button>
          </Tooltip>
          <Tooltip title="Zoom Out">
            <Button
              size="small"
              onClick={() => setZoom(prev => Math.max(prev / 1.2, 0.1))}
            >
              <Iconify icon="carbon:zoom-out" />
            </Button>
          </Tooltip>
          <Tooltip title="Reset Zoom">
            <Button
              size="small"
              onClick={() => {
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
            >
              <Iconify icon="carbon:reset" />
            </Button>
          </Tooltip>
        </Stack>

        <Stack direction="row" spacing={1} alignItems="center">
          <Tooltip title={showSelection ? "Hide Selection" : "Show Selection"}>
            <Button
              size="small"
              variant={showSelection ? "contained" : "outlined"}
              onClick={() => setShowSelection(!showSelection)}
            >
              <Iconify icon={showSelection ? "carbon:view" : "carbon:view-off"} />
            </Button>
          </Tooltip>
          <Tooltip title="Clear Selection">
            <Button
              size="small"
              color="error"
              onClick={clearSelection}
              disabled={!markedPoint}
            >
              <Iconify icon="carbon:close" />
            </Button>
          </Tooltip>
        </Stack>
      </Stack>

      {/* Tolerance Slider */}
      <Box sx={{ px: 2 }}>
        <Typography variant="caption" gutterBottom>
          Tolerance: {tolerance}
        </Typography>
        <Slider
          value={tolerance}
          onChange={(e, value) => updateTolerance(value)}
          min={5}
          max={100}
          step={5}
          size="small"
          disabled={isSelecting}
        />
      </Box>

      {/* Canvas */}
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        style={{
          width: '100%',
          height: '400px',
          border: '1px solid #e0e0e0',
          borderRadius: '4px',
          cursor: isSelecting ? 'wait' : 'crosshair',
          backgroundColor: '#f5f5f5',
        }}
      />

      {/* Status */}
      <Stack direction="row" spacing={2} justifyContent="space-between">
        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
          {isSelecting ? '🔄 Selecting object...' :
           markedPoint ? `✅ Selected ${selectedPixels.size} pixels` :
           '👆 Click on an object to select it'}
        </Typography>
        {markedPoint && (
          <Typography variant="caption" sx={{ color: 'text.secondary' }}>
            Point: ({markedPoint.x}, {markedPoint.y})
          </Typography>
        )}
      </Stack>
    </Stack>
  );
}
