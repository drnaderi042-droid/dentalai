/* eslint-disable no-continue */
import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import List from '@mui/material/List';
import Menu from '@mui/material/Menu';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Drawer from '@mui/material/Drawer';
import Dialog from '@mui/material/Dialog';
import Tooltip from '@mui/material/Tooltip';
import Divider from '@mui/material/Divider';
import ListItem from '@mui/material/ListItem';
import MenuItem from '@mui/material/MenuItem';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import ButtonGroup from '@mui/material/ButtonGroup';
import DialogTitle from '@mui/material/DialogTitle';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import ToggleButton from '@mui/material/ToggleButton';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';

import { Iconify } from 'src/components/iconify';

// Annotation types
const ANNOTATION_TYPES = {
  POINT: 'point',
  RECTANGLE: 'rectangle',
  POLYGON: 'polygon',
  POLYLINE: 'polyline',
  LANDMARK: 'landmark'
};

export function AdvancedImageEditor({ imageUrl, onAnnotationsChange }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [annotations, setAnnotations] = useState([]);
  const [history, setHistory] = useState([[]]);
  const [historyIndex, setHistoryIndex] = useState(0);
  
  // UI States
  const [selectedTool, setSelectedTool] = useState(ANNOTATION_TYPES.POINT);
  const [selectedAnnotation, setSelectedAnnotation] = useState(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [currentDrawing, setCurrentDrawing] = useState(null);
  const [showObjectPanel, setShowObjectPanel] = useState(true);
  const [showExportDialog, setShowExportDialog] = useState(false);
  
  // Canvas states
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  
  // Annotation properties
  const [annotationColor, setAnnotationColor] = useState('#FF0000');
  const [annotationThickness, setAnnotationThickness] = useState(2);
  const [annotationOpacity] = useState(1);
  const [pointRadius] = useState(5);
  
  // Context menu
  const [contextMenu, setContextMenu] = useState(null);
  const [labelDialogOpen, setLabelDialogOpen] = useState(false);
  const [labelText, setLabelText] = useState('');
  
  // Load image
  useEffect(() => {
    if (imageUrl && canvasRef.current) {
      const canvas = canvasRef.current;
      const img = new Image();

      img.onload = () => {
        imageRef.current = img;
        canvas.width = img.width;
        canvas.height = img.height;
        drawCanvas();
      };

      img.crossOrigin = 'anonymous';
      img.src = imageUrl;
    }
  }, [imageUrl, drawCanvas]);

  // Draw canvas
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    if (imageRef.current) {
      ctx.drawImage(imageRef.current, 0, 0);
    }

    annotations.forEach((annotation, index) => {
      if (!annotation.visible) return;
      const isSelected = selectedAnnotation === index;
      drawAnnotation(ctx, annotation, isSelected);
    });

    if (currentDrawing) {
      drawAnnotation(ctx, { ...currentDrawing, opacity: 0.7 }, false);
    }

    ctx.restore();
  }, [annotations, zoom, pan, selectedAnnotation, currentDrawing, drawAnnotation]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey && e.key === 'z' && !e.shiftKey) {
        e.preventDefault();
        if (historyIndex > 0) {
          setHistoryIndex((idx) => {
            const newIdx = idx - 1;
            setAnnotations(history[newIdx]);
            return newIdx;
          });
        }
      }
      if ((e.ctrlKey && e.shiftKey && e.key === 'z') || (e.ctrlKey && e.key === 'y')) {
        e.preventDefault();
        if (historyIndex < history.length - 1) {
          setHistoryIndex((idx) => {
            const newIdx = idx + 1;
            setAnnotations(history[newIdx]);
            return newIdx;
          });
        }
      }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotation !== null) {
        e.preventDefault();
        setAnnotations(prev => {
          const newAnnotations = prev.filter((_, i) => i !== selectedAnnotation);
          setSelectedAnnotation(null);
          setHistory(prevHistory => {
            const newHistory = prevHistory.slice(0, historyIndex + 1);
            newHistory.push(newAnnotations);
            setHistoryIndex(newHistory.length - 1);
            return newHistory;
          });
          onAnnotationsChange?.(newAnnotations);
          return newAnnotations;
        });
      }
      if (e.key === 'Escape') {
        setCurrentDrawing(null);
        setSelectedAnnotation(null);
      }
      if (e.key === 'p') setSelectedTool(ANNOTATION_TYPES.POINT);
      if (e.key === 'r') setSelectedTool(ANNOTATION_TYPES.RECTANGLE);
      if (e.key === 'g') setSelectedTool(ANNOTATION_TYPES.POLYGON);
      if (e.key === 'l') setSelectedTool(ANNOTATION_TYPES.POLYLINE);
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [selectedAnnotation, historyIndex, history, onAnnotationsChange]);



  const drawAnnotation = useCallback((ctx, annotation, isSelected) => {
    ctx.save();

    const color = annotation.color || annotationColor;
    const thickness = annotation.thickness || annotationThickness;
    const opacity = annotation.opacity || annotationOpacity;

    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = thickness;
    ctx.globalAlpha = opacity;

    switch (annotation.type) {
      case ANNOTATION_TYPES.POINT: // falls through
      case ANNOTATION_TYPES.LANDMARK: {
        const radius = annotation.radius || pointRadius;
        ctx.beginPath();
        ctx.arc(annotation.x, annotation.y, radius, 0, 2 * Math.PI);
        ctx.fill();

        if (annotation.label) {
          ctx.fillStyle = '#000';
          ctx.font = 'bold 14px Arial';
          ctx.globalAlpha = 1;
          ctx.fillText(annotation.label, annotation.x + 10, annotation.y - 10);
        }
        break;
      }

      case ANNOTATION_TYPES.RECTANGLE: {
        ctx.strokeRect(annotation.x, annotation.y, annotation.width, annotation.height);
        if (annotation.filled) {
          ctx.globalAlpha = opacity * 0.3;
          ctx.fillRect(annotation.x, annotation.y, annotation.width, annotation.height);
        }
        break;
      }

      case ANNOTATION_TYPES.POLYGON:
      case ANNOTATION_TYPES.POLYLINE: {
        if (annotation.points && annotation.points.length > 1) {
          ctx.beginPath();
          ctx.moveTo(annotation.points[0].x, annotation.points[0].y);
          for (let pi = 1; pi < annotation.points.length; pi += 1) {
            const point = annotation.points[pi];
            ctx.lineTo(point.x, point.y);
          }
          if (annotation.type === ANNOTATION_TYPES.POLYGON) {
            ctx.closePath();
            if (annotation.filled) {
              ctx.globalAlpha = opacity * 0.3;
              ctx.fill();
              ctx.globalAlpha = opacity;
            }
          }
          ctx.stroke();

          for (let pi = 0; pi < annotation.points.length; pi += 1) {
            const point = annotation.points[pi];
            ctx.beginPath();
            ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
            ctx.fill();
          }
        }
        break;
      }

      default:
        break;
    }

    if (isSelected) {
      ctx.strokeStyle = '#FFD700';
      ctx.lineWidth = thickness + 2;
      ctx.globalAlpha = 1;
      ctx.setLineDash([5, 5]);

      switch (annotation.type) {
        case ANNOTATION_TYPES.POINT:
        case ANNOTATION_TYPES.LANDMARK: {
          ctx.beginPath();
          ctx.arc(annotation.x, annotation.y, (annotation.radius || pointRadius) + 3, 0, 2 * Math.PI);
          ctx.stroke();
          break;
        }

        case ANNOTATION_TYPES.RECTANGLE: {
          ctx.strokeRect(annotation.x - 2, annotation.y - 2, annotation.width + 4, annotation.height + 4);
          break;
        }

        case ANNOTATION_TYPES.POLYGON:
        case ANNOTATION_TYPES.POLYLINE: {
          if (annotation.points && annotation.points.length > 1) {
            ctx.beginPath();
            ctx.moveTo(annotation.points[0].x, annotation.points[0].y);
            for (let pi = 1; pi < annotation.points.length; pi += 1) {
              const point = annotation.points[pi];
              ctx.lineTo(point.x, point.y);
            }
            if (annotation.type === ANNOTATION_TYPES.POLYGON) ctx.closePath();
            ctx.stroke();
          }
          break;
        }

        default:
          break;
      }

      ctx.setLineDash([]);
    }

    ctx.restore();
  }, [annotationColor, annotationThickness, annotationOpacity, pointRadius]);

  const getCanvasCoordinates = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - pan.x) / zoom,
      y: (e.clientY - rect.top - pan.y) / zoom
    };
  };

  const handleCanvasClick = (e) => {
    if (isPanning) return;
    
    const coords = getCanvasCoordinates(e);
    
    const clickedIndex = findAnnotationAtPoint(coords.x, coords.y);
    if (clickedIndex !== -1 && selectedTool === ANNOTATION_TYPES.POINT) {
      setSelectedAnnotation(clickedIndex);
      return;
    }
    
    switch (selectedTool) {
      case ANNOTATION_TYPES.POINT:
      case ANNOTATION_TYPES.LANDMARK:
        addAnnotation({
          type: selectedTool,
          x: coords.x,
          y: coords.y,
          color: annotationColor,
          thickness: annotationThickness,
          opacity: annotationOpacity,
          radius: pointRadius,
          visible: true,
          label: selectedTool === ANNOTATION_TYPES.LANDMARK ? 'New Point' : ''
        });
        break;
        
      case ANNOTATION_TYPES.RECTANGLE:
        if (!isDrawing) {
          setIsDrawing(true);
          setCurrentDrawing({
            type: ANNOTATION_TYPES.RECTANGLE,
            x: coords.x,
            y: coords.y,
            width: 0,
            height: 0,
            color: annotationColor,
            thickness: annotationThickness,
            opacity: annotationOpacity,
            visible: true
          });
        }
        break;
        
      case ANNOTATION_TYPES.POLYGON:
      case ANNOTATION_TYPES.POLYLINE:
        if (!currentDrawing) {
          setCurrentDrawing({
            type: selectedTool,
            points: [coords],
            color: annotationColor,
            thickness: annotationThickness,
            opacity: annotationOpacity,
            visible: true
          });
        } else {
          setCurrentDrawing(prev => ({
            ...prev,
            points: [...prev.points, coords]
          }));
        }
        break;
      default:
        break;
    }
  };

  const handleMouseMove = (e) => {
    if (isPanning) {
      setPan({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y
      });
      return;
    }
    
    if (isDrawing && currentDrawing && currentDrawing.type === ANNOTATION_TYPES.RECTANGLE) {
      const coords = getCanvasCoordinates(e);
      setCurrentDrawing(prev => ({
        ...prev,
        width: coords.x - prev.x,
        height: coords.y - prev.y
      }));
    }
  };

  const handleMouseDown = (e) => {
    if (e.button === 1 || e.ctrlKey) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseUp = () => {
    setIsPanning(false);
    
    if (isDrawing && currentDrawing && currentDrawing.type === ANNOTATION_TYPES.RECTANGLE) {
      if (Math.abs(currentDrawing.width) > 5 && Math.abs(currentDrawing.height) > 5) {
        addAnnotation(currentDrawing);
      }
      setCurrentDrawing(null);
      setIsDrawing(false);
    }
  };

  const handleDoubleClick = () => {
    if (currentDrawing && (currentDrawing.type === ANNOTATION_TYPES.POLYGON || 
        currentDrawing.type === ANNOTATION_TYPES.POLYLINE)) {
      if (currentDrawing.points.length > 2) {
        addAnnotation(currentDrawing);
      }
      setCurrentDrawing(null);
    }
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom(prev => Math.max(0.1, Math.min(5, prev * delta)));
  };

  const handleContextMenu = (e) => {
    e.preventDefault();
    const coords = getCanvasCoordinates(e);
    const clickedIndex = findAnnotationAtPoint(coords.x, coords.y);
    
    if (clickedIndex !== -1) {
      setSelectedAnnotation(clickedIndex);
      setContextMenu({ mouseX: e.clientX, mouseY: e.clientY });
    }
  };

  const findAnnotationAtPoint = (x, y) => {
    const tolerance = 10 / zoom;

    for (let i = annotations.length - 1; i >= 0; i -= 1) {
      const ann = annotations[i];
      if (!ann.visible) {
        continue;
      }

      switch (ann.type) {
        case ANNOTATION_TYPES.POINT:
        case ANNOTATION_TYPES.LANDMARK: {
          const distance = Math.sqrt((x - ann.x) ** 2 + (y - ann.y) ** 2);
          if (distance <= (ann.radius || pointRadius) + tolerance) return i;
          break;
        }

        case ANNOTATION_TYPES.RECTANGLE: {
          if (x >= ann.x && x <= ann.x + ann.width && y >= ann.y && y <= ann.y + ann.height) return i;
          break;
        }

        case ANNOTATION_TYPES.POLYGON:
        case ANNOTATION_TYPES.POLYLINE: {
          if (ann.points) {
            for (let pi = 0; pi < ann.points.length; pi += 1) {
              const point = ann.points[pi];
              const pointDist = Math.sqrt((x - point.x) ** 2 + (y - point.y) ** 2);
              if (pointDist <= tolerance) return i;
            }
          }
          break;
        }

        default:
          break;
      }
    }
    return -1;
  };

  const addAnnotation = (annotation) => {
    const newAnnotations = [...annotations, annotation];
    setAnnotations(newAnnotations);
    addToHistory(newAnnotations);
    onAnnotationsChange?.(newAnnotations);
  };

  const deleteAnnotation = (index) => {
    const newAnnotations = annotations.filter((_, i) => i !== index);
    setAnnotations(newAnnotations);
    setSelectedAnnotation(null);
    addToHistory(newAnnotations);
    onAnnotationsChange?.(newAnnotations);
  };

  const updateAnnotation = (index, updates) => {
    const newAnnotations = annotations.map((ann, i) => 
      i === index ? { ...ann, ...updates } : ann
    );
    setAnnotations(newAnnotations);
    addToHistory(newAnnotations);
    onAnnotationsChange?.(newAnnotations);
  };

  const addToHistory = (newState) => {
    const newHistory = history.slice(0, historyIndex + 1);
    newHistory.push(newState);
    setHistory(newHistory);
    setHistoryIndex(newHistory.length - 1);
  };

  const undo = () => {
    if (historyIndex > 0) {
      setHistoryIndex(historyIndex - 1);
      setAnnotations(history[historyIndex - 1]);
    }
  };

  const redo = () => {
    if (historyIndex < history.length - 1) {
      setHistoryIndex(historyIndex + 1);
      setAnnotations(history[historyIndex + 1]);
    }
  };

  const exportAnnotations = (format) => {
    let data; let filename; let type;
    
    switch (format) {
      case 'json':
        data = JSON.stringify({ annotations, imageUrl }, null, 2);
        filename = 'annotations.json';
        type = 'application/json';
        break;
      case 'csv':
        data = annotationsToCSV(annotations);
        filename = 'annotations.csv';
        type = 'text/csv';
        break;
      case 'coco':
        data = JSON.stringify(annotationsToCOCO(annotations), null, 2);
        filename = 'annotations_coco.json';
        type = 'application/json';
        break;
      default:
        return;
    }
    
    const blob = new Blob([data], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    setShowExportDialog(false);
  };

  const annotationsToCSV = (anns) => {
    let csv = 'type,x,y,width,height,label,color\n';
    anns.forEach(ann => {
      csv += `${ann.type},${ann.x || ''},${ann.y || ''},${ann.width || ''},${ann.height || ''},${ann.label || ''},${ann.color}\n`;
    });
    return csv;
  };

  const annotationsToCOCO = (anns) => ({
      images: [{
        id: 1,
        file_name: imageUrl,
        width: imageRef.current?.width || 0,
        height: imageRef.current?.height || 0
      }],
      annotations: anns.map((ann, i) => ({
        id: i + 1,
        image_id: 1,
        category_id: 1,
        bbox: ann.type === ANNOTATION_TYPES.RECTANGLE ? 
          [ann.x, ann.y, ann.width, ann.height] : [],
        segmentation: ann.points ? 
          [ann.points.flatMap(p => [p.x, p.y])] : []
      }))
    });

  return (
    <Box sx={{ display: 'flex', height: 'calc(100vh - 100px)' }}>
      <Box sx={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        <Paper sx={{ p: 1, mb: 1 }}>
          <Stack direction="row" spacing={1} alignItems="center" flexWrap="wrap">
            <ToggleButtonGroup
              value={selectedTool}
              exclusive
              onChange={(e, value) => value && setSelectedTool(value)}
              size="small"
            >
              <ToggleButton value={ANNOTATION_TYPES.POINT}>
                <Tooltip title="Point (P)">
                  <Iconify icon="eva:radio-button-on-fill" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value={ANNOTATION_TYPES.RECTANGLE}>
                <Tooltip title="Rectangle (R)">
                  <Iconify icon="eva:square-outline" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value={ANNOTATION_TYPES.POLYGON}>
                <Tooltip title="Polygon (G)">
                  <Iconify icon="eva:grid-outline" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value={ANNOTATION_TYPES.POLYLINE}>
                <Tooltip title="Polyline (L)">
                  <Iconify icon="eva:diagonal-arrow-right-up-outline" />
                </Tooltip>
              </ToggleButton>
              <ToggleButton value={ANNOTATION_TYPES.LANDMARK}>
                <Tooltip title="Landmark">
                  <Iconify icon="eva:pin-fill" />
                </Tooltip>
              </ToggleButton>
            </ToggleButtonGroup>

            <Divider orientation="vertical" flexItem />

            <ButtonGroup size="small">
              <Tooltip title="Undo (Ctrl+Z)">
                <IconButton onClick={undo} disabled={historyIndex <= 0}>
                  <Iconify icon="eva:arrow-back-outline" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Redo (Ctrl+Y)">
                <IconButton onClick={redo} disabled={historyIndex >= history.length - 1}>
                  <Iconify icon="eva:arrow-forward-outline" />
                </IconButton>
              </Tooltip>
            </ButtonGroup>

            <Divider orientation="vertical" flexItem />

            <ButtonGroup size="small">
              <Tooltip title="Zoom In">
                <IconButton onClick={() => setZoom(prev => Math.min(5, prev * 1.2))}>
                  <Iconify icon="eva:plus-outline" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Zoom Out">
                <IconButton onClick={() => setZoom(prev => Math.max(0.1, prev / 1.2))}>
                  <Iconify icon="eva:minus-outline" />
                </IconButton>
              </Tooltip>
              <Tooltip title="Reset">
                <IconButton onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>
                  <Iconify icon="eva:refresh-outline" />
                </IconButton>
              </Tooltip>
            </ButtonGroup>

            <Typography variant="caption" sx={{ mx: 1 }}>
              {(zoom * 100).toFixed(0)}%
            </Typography>

            <Divider orientation="vertical" flexItem />

            <TextField
              size="small"
              type="color"
              value={annotationColor}
              onChange={(e) => setAnnotationColor(e.target.value)}
              sx={{ width: 60 }}
            />
            
            <TextField
              size="small"
              type="number"
              value={annotationThickness}
              onChange={(e) => setAnnotationThickness(Number(e.target.value))}
              label="Thickness"
              sx={{ width: 100 }}
              inputProps={{ min: 1, max: 10 }}
            />

            <Box sx={{ flex: 1 }} />

            <Button
              size="small"
              onClick={() => setShowExportDialog(true)}
              startIcon={<Iconify icon="eva:download-outline" />}
            >
              Export
            </Button>

            <IconButton
              size="small"
              onClick={() => setShowObjectPanel(!showObjectPanel)}
              color={showObjectPanel ? 'primary' : 'default'}
            >
              <Iconify icon="eva:list-outline" />
            </IconButton>
          </Stack>
        </Paper>

        <Box
          sx={{
            flex: 1,
            bgcolor: '#1a1a1a',
            position: 'relative',
            overflow: 'hidden',
            cursor: isPanning ? 'grabbing' : 
                   selectedTool !== ANNOTATION_TYPES.POINT ? 'crosshair' : 'default'
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
            onContextMenu={handleContextMenu}
            style={{ display: 'block', maxWidth: '100%', maxHeight: '100%' }}
          />
          
          {currentDrawing && (
            <Box
              sx={{
                position: 'absolute',
                bottom: 20,
                left: '50%',
                transform: 'translateX(-50%)',
                bgcolor: 'rgba(0,0,0,0.8)',
                color: 'white',
                px: 2,
                py: 1,
                borderRadius: 1
              }}
            >
              {(currentDrawing.type === ANNOTATION_TYPES.POLYGON || 
                currentDrawing.type === ANNOTATION_TYPES.POLYLINE) && 
                `Points: ${currentDrawing.points?.length || 0} - Double click to finish`}
            </Box>
          )}
        </Box>
      </Box>

      <Drawer
        anchor="right"
        open={showObjectPanel}
        variant="persistent"
        sx={{ '& .MuiDrawer-paper': { width: 300, mt: '64px', height: 'calc(100% - 64px)' } }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6" sx={{ mb: 2 }}>
            Objects ({annotations.length})
          </Typography>
          
          <List dense>
            {annotations.map((ann, index) => (
              <ListItem
                key={index}
                selected={selectedAnnotation === index}
                onClick={() => setSelectedAnnotation(index)}
                secondaryAction={
                  <Stack direction="row" spacing={0.5}>
                    <IconButton
                      size="small"
                      onClick={() => updateAnnotation(index, { visible: !ann.visible })}
                    >
                      <Iconify icon={ann.visible ? 'eva:eye-fill' : 'eva:eye-off-fill'} />
                    </IconButton>
                    <IconButton
                      size="small"
                      onClick={() => deleteAnnotation(index)}
                      color="error"
                    >
                      <Iconify icon="eva:trash-2-outline" />
                    </IconButton>
                  </Stack>
                }
                sx={{ 
                  border: 1, 
                  borderColor: 'divider', 
                  borderRadius: 1, 
                  mb: 1,
                  cursor: 'pointer'
                }}
              >
                <ListItemIcon>
                  <Box
                    sx={{
                      width: 20,
                      height: 20,
                      borderRadius: '50%',
                      bgcolor: ann.color
                    }}
                  />
                </ListItemIcon>
                <ListItemText
                  primary={ann.label || `${ann.type} ${index + 1}`}
                  secondary={`Type: ${ann.type}`}
                />
              </ListItem>
            ))}
          </List>
          
          {annotations.length === 0 && (
            <Alert severity="info">
              No annotations yet. Select a tool and start annotating!
            </Alert>
          )}
        </Box>
      </Drawer>

      <Menu
        open={contextMenu !== null}
        onClose={() => setContextMenu(null)}
        anchorReference="anchorPosition"
        anchorPosition={
          contextMenu !== null
            ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
            : undefined
        }
      >
        <MenuItem onClick={() => {
          setLabelDialogOpen(true);
          setContextMenu(null);
        }}>
          <Iconify icon="eva:edit-outline" sx={{ mr: 1 }} />
          Edit Label
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedAnnotation !== null) {
            updateAnnotation(selectedAnnotation, { color: annotationColor });
          }
          setContextMenu(null);
        }}>
          <Iconify icon="eva:color-palette-outline" sx={{ mr: 1 }} />
          Change Color
        </MenuItem>
        <MenuItem onClick={() => {
          if (selectedAnnotation !== null) {
            deleteAnnotation(selectedAnnotation);
          }
          setContextMenu(null);
        }}>
          <Iconify icon="eva:trash-2-outline" sx={{ mr: 1 }} />
          Delete
        </MenuItem>
      </Menu>

      <Dialog open={labelDialogOpen} onClose={() => setLabelDialogOpen(false)}>
        <DialogTitle>Edit Label</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Label"
            fullWidth
            value={labelText}
            onChange={(e) => setLabelText(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setLabelDialogOpen(false)}>Cancel</Button>
          <Button onClick={() => {
            if (selectedAnnotation !== null) {
              updateAnnotation(selectedAnnotation, { label: labelText });
            }
            setLabelDialogOpen(false);
            setLabelText('');
          }} variant="contained">
            Save
          </Button>
        </DialogActions>
      </Dialog>

      <Dialog open={showExportDialog} onClose={() => setShowExportDialog(false)}>
        <DialogTitle>Export Annotations</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            Choose export format:
          </Typography>
          <Stack spacing={1}>
            <Button
              variant="outlined"
              onClick={() => exportAnnotations('json')}
              startIcon={<Iconify icon="eva:file-text-outline" />}
            >
              Export as JSON
            </Button>
            <Button
              variant="outlined"
              onClick={() => exportAnnotations('csv')}
              startIcon={<Iconify icon="eva:file-text-outline" />}
            >
              Export as CSV
            </Button>
            <Button
              variant="outlined"
              onClick={() => exportAnnotations('coco')}
              startIcon={<Iconify icon="eva:file-text-outline" />}
            >
              Export as COCO Format
            </Button>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowExportDialog(false)}>Cancel</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
