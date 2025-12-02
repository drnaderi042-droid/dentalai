import React, { useRef, useState, useEffect, useCallback } from 'react';
/* eslint-disable import/no-extraneous-dependencies */
import { Line, Text, Stage, Layer, Group, Circle, Transformer, Image as KonvaImage } from 'react-konva';
/* eslint-enable import/no-extraneous-dependencies */

import {
  Box,
  Grid,
  Paper,
  Stack,
  Alert,
  Button,
  Dialog,
  Select,
  MenuItem,
  Typography,
  IconButton,
  InputLabel,
  DialogTitle,
  FormControl,
  DialogContent,
  DialogActions,
} from '@mui/material';

import { Iconify } from 'src/components/iconify';

// Cephalometric landmarks for lateral cephalometric analysis
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
};

// Common cephalometric planes
const CEPHALOMETRIC_PLANES = {
  frankfort: {
    name: 'صفحه فرانکفورت (FH)',
    description: 'Porion to Orbitale',
    points: ['Po', 'Or'],
    color: '#FF5722',
  },
  occlusal: {
    name: 'صفحه اکلوزال (OP)',
    description: 'Occlusal plane',
    points: [], // Will be drawn as a line
    color: '#4CAF50',
  },
  mandibular: {
    name: 'صفحه مندیبولار (MP)',
    description: 'Gonion to Menton',
    points: ['Go', 'Me'],
    color: '#2196F3',
  },
  sna: {
    name: 'خط S-NA',
    description: 'Sella to Nasion',
    points: ['S', 'N'],
    color: '#FF9800',
  },
  snb: {
    name: 'خط S-NB',
    description: 'Sella to Point B',
    points: ['S', 'B'],
    color: '#9C27B0',
  },
};

// Drawing modes
const DRAWING_MODES = {
  NONE: 'none',
  LANDMARK: 'landmark',
  LINE: 'line',
  PLANE: 'plane',
  MEASUREMENT: 'measurement',
};

const CephalometricKonvaViewer = ({
  imageUrl,
  onLandmarkUpdate,
  initialLandmarks = {},
  currentLandmarks = {},
  readOnly = false,
  onLandmarkDetect = null,
  onCropSave = null,
}) => {
  const stageRef = useRef(null);
  const layerRef = useRef(null);
  const transformerRef = useRef(null);
  const [image, setImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Drawing state
  const [drawingMode, setDrawingMode] = useState(DRAWING_MODES.NONE);
  const [landmarks, setLandmarks] = useState(initialLandmarks);
  const [planes, setPlanes] = useState({});
  const [measurements, setMeasurements] = useState([]);
  const [lines, setLines] = useState([]);
  const [selectedId, setSelectedId] = useState(null);

  // Temporary drawing
  const [isDrawing, setIsDrawing] = useState(false);
  const [tempPoints, setTempPoints] = useState([]);
  const [stageScale, setStageScale] = useState(1);
  const [stagePos, setStagePos] = useState({ x: 0, y: 0 });

  // Dialog state
  const [addLandmarkDialogOpen, setAddLandmarkDialogOpen] = useState(false);
  const [selectedLandmarkType, setSelectedLandmarkType] = useState('');

  // Measurement state
  const [measurementMode, setMeasurementMode] = useState(false);
  const [firstMeasurementPoint, setFirstMeasurementPoint] = useState(null);

  // Load image
  useEffect(() => {
    if (!imageUrl) return;

    setImageLoaded(false);
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImage(img);
      setImageLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Update landmarks when currentLandmarks prop changes (AI detection results)
  useEffect(() => {
    if (Object.keys(currentLandmarks).length > 0) {
      setLandmarks(currentLandmarks);
      if (onLandmarkUpdate) {
        onLandmarkUpdate(currentLandmarks);
      }
    }
  }, [currentLandmarks, onLandmarkUpdate]);

  // Calculate distance between two points
  const getDistance = (point1, point2) => Math.sqrt((point2.x - point1.x)**2 + (point2.y - point1.y)**2);

  // Convert image coordinates to stage coordinates
  const imageToStage = (imagePoint) => {
    if (!image) return imagePoint;
    // Scale from image coordinates to fit the stage
    const scale = Math.min(800 / image.width, 600 / image.height);
    return {
      x: imagePoint.x * scale,
      y: imagePoint.y * scale,
    };
  };

  // Convert stage coordinates to image coordinates
  const stageToImage = (stagePoint) => {
    if (!image) return stagePoint;
    const scale = Math.min(800 / image.width, 600 / image.height);
    return {
      x: stagePoint.x / scale,
      y: stagePoint.y / scale,
    };
  };

  // Handle stage click for different drawing modes
  const handleStageClick = useCallback((e) => {
    if (readOnly) return;

    const stage = e.target.getStage();
    const pointer = stage.getPointerPosition();
    const clickX = (pointer.x - stagePos.x) / stageScale;
    const clickY = (pointer.y - stagePos.y) / stageScale;

    if (drawingMode === DRAWING_MODES.LANDMARK) {
      setAddLandmarkDialogOpen(true);
    } else if (drawingMode === DRAWING_MODES.LINE && !isDrawing) {
      setTempPoints([clickX, clickY]);
      setIsDrawing(true);
    } else if (drawingMode === DRAWING_MODES.LINE && isDrawing) {
      const newLine = {
        id: `line_${Date.now()}`,
        points: [...tempPoints, clickX, clickY],
        color: '#FF5722',
      };
      setLines((prev) => [...prev, newLine]);
      setIsDrawing(false);
      setTempPoints([]);
    } else if (drawingMode === DRAWING_MODES.MEASUREMENT) {
      if (!firstMeasurementPoint) {
        setFirstMeasurementPoint({ x: clickX, y: clickY });
      } else {
        const distance = getDistance(firstMeasurementPoint, { x: clickX, y: clickY });
        const newMeasurement = {
          id: `measurement_${Date.now()}`,
          point1: firstMeasurementPoint,
          point2: { x: clickX, y: clickY },
          distance,
        };
        setMeasurements((prev) => [...prev, newMeasurement]);
        setFirstMeasurementPoint(null);
        setMeasurementMode(false);
        setDrawingMode(DRAWING_MODES.NONE);
      }
    }
  }, [drawingMode, isDrawing, tempPoints, firstMeasurementPoint, stagePos, stageScale, readOnly]);

  // Handle wheel for zoom
  const handleWheel = useCallback((e) => {
    e.evt.preventDefault();

    const scaleBy = 1.2;
    const stage = stageRef.current;
    const oldScale = stageScale;
    const mousePointTo = {
      x: stage.getPointerPosition().x / oldScale - stagePos.x / oldScale,
      y: stage.getPointerPosition().y / oldScale - stagePos.y / oldScale,
    };

    const newScale = e.evt.deltaY < 0 ? oldScale * scaleBy : oldScale / scaleBy;

    setStageScale(newScale);
    setStagePos({
      x: -(mousePointTo.x - stage.getPointerPosition().x / newScale) * newScale,
      y: -(mousePointTo.y - stage.getPointerPosition().y / newScale) * newScale,
    });
  }, [stageScale, stagePos]);

  // Add landmark
  const handleAddLandmark = useCallback(() => {
    if (!selectedLandmarkType) return;

    const stage = stageRef.current;
    const pointer = stage.getPointerPosition();

    const newLandmarks = {
      ...landmarks,
      [selectedLandmarkType]: {
        x: (pointer.x - stagePos.x) / stageScale,
        y: (pointer.y - stagePos.y) / stageScale,
      },
    };

    setLandmarks(newLandmarks);
    if (onLandmarkUpdate) {
      onLandmarkUpdate(newLandmarks);
    }

    setAddLandmarkDialogOpen(false);
    setSelectedLandmarkType('');
  }, [selectedLandmarkType, landmarks, onLandmarkUpdate, stagePos, stageScale]);

  // Draw common planes
  const drawPlane = useCallback((planeKey) => {
    const plane = CEPHALOMETRIC_PLANES[planeKey];
    if (!plane.points || plane.points.length < 2) return;

    const point1 = landmarks[plane.points[0]];
    const point2 = landmarks[plane.points[1]];

    if (!point1 || !point2) return;

    const newPlane = {
      id: `plane_${planeKey}_${Date.now()}`,
      points: [point1.x, point1.y, point2.x, point2.y],
      color: plane.color,
      name: plane.name,
    };

    setPlanes((prev) => ({
      ...prev,
      [planeKey]: newPlane,
    }));
  }, [landmarks]);

  // AI landmark detection
  const handleAIDetect = useCallback(async () => {
    if (!onLandmarkDetect) return;
    await onLandmarkDetect();
  }, [onLandmarkDetect]);

  // Clear all drawings
  const clearAllDrawings = useCallback(() => {
    setLandmarks({});
    setPlanes({});
    setLines([]);
    setMeasurements([]);
    setSelectedId(null);
    if (onLandmarkUpdate) {
      onLandmarkUpdate({});
    }
  }, [onLandmarkUpdate]);

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
        {/* Drawing Mode Buttons */}
        <Stack direction="row" spacing={1}>
          <Button
            variant={drawingMode === DRAWING_MODES.LANDMARK ? 'contained' : 'outlined'}
            onClick={() => setDrawingMode(DRAWING_MODES.LANDMARK)}
            size="small"
            disabled={readOnly}
          >
            <Iconify icon="solar:point-on-map-linear" width={16} sx={{ mr: 0.5 }} />
            لندمارک
          </Button>

          <Button
            variant={drawingMode === DRAWING_MODES.LINE ? 'contained' : 'outlined'}
            onClick={() => setDrawingMode(DRAWING_MODES.LINE)}
            size="small"
            disabled={readOnly}
          >
            <Iconify icon="solar:minus-line" width={16} sx={{ mr: 0.5 }} />
            خط
          </Button>

          <Button
            variant={drawingMode === DRAWING_MODES.MEASUREMENT ? 'contained' : 'outlined'}
            onClick={() => setDrawingMode(DRAWING_MODES.MEASUREMENT)}
            size="small"
            disabled={readOnly}
          >
            <Iconify icon="solar:ruler-linear" width={16} sx={{ mr: 0.5 }} />
            اندازه‌گیری
          </Button>
        </Stack>

        {/* Plane Drawing Buttons */}
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            onClick={() => drawPlane('frankfort')}
            size="small"
            disabled={readOnly || !landmarks.Po || !landmarks.Or}
            sx={{ fontSize: '0.75rem' }}
          >
            FH
          </Button>

          <Button
            variant="outlined"
            onClick={() => drawPlane('mandibular')}
            size="small"
            disabled={readOnly || !landmarks.Go || !landmarks.Me}
            sx={{ fontSize: '0.75rem' }}
          >
            MP
          </Button>

          <Button
            variant="outlined"
            onClick={() => drawPlane('sna')}
            size="small"
            disabled={readOnly || !landmarks.S || !landmarks.A}
            sx={{ fontSize: '0.75rem' }}
          >
            NA
          </Button>

          <Button
            variant="outlined"
            onClick={() => drawPlane('snb')}
            size="small"
            disabled={readOnly || !landmarks.S || !landmarks.B}
            sx={{ fontSize: '0.75rem' }}
          >
            NB
          </Button>
        </Stack>

        {/* Utility Buttons */}
        <Stack direction="row" spacing={1}>
          <Button
            variant="outlined"
            color="primary"
            onClick={handleAIDetect}
            size="small"
            disabled={!imageLoaded}
          >
            <Iconify icon="solar:robot-2-linear" width={16} sx={{ mr: 0.5 }} />
            تشخیص AI
          </Button>

          <Button
            variant="outlined"
            color="error"
            onClick={clearAllDrawings}
            size="small"
            disabled={readOnly}
          >
            <Iconify icon="solar:trash-bin-2-bold" width={16} sx={{ mr: 0.5 }} />
            پاک کردن همه
          </Button>

          <Button
            variant="outlined"
            onClick={() => {
              setStageScale(1);
              setStagePos({ x: 0, y: 0 });
            }}
            size="small"
          >
            <Iconify icon="solar:refresh-linear" width={16} sx={{ mr: 0.5 }} />
            ریست
          </Button>
        </Stack>
      </Stack>

      {/* Konva Stage Container */}
      <Paper sx={{ p: 1, display: 'flex', justifyContent: 'center', width: '100%' }}>
        <Box
          sx={{
            position: 'relative',
            width: '100%',
            maxWidth: '900px',
            height: '650px',
            border: '1px solid #ccc',
            borderRadius: 1,
            overflow: 'hidden'
          }}
        >
          <Stage
            ref={stageRef}
            width={900}
            height={650}
            scaleX={stageScale}
            scaleY={stageScale}
            x={stagePos.x}
            y={stagePos.y}
            onWheel={handleWheel}
            onClick={handleStageClick}
            style={{ cursor: drawingMode === DRAWING_MODES.LANDMARK ? 'crosshair' : 'default' }}
          >
            <Layer ref={layerRef}>
              {/* Background Image */}
              {image && (
                <KonvaImage
                  image={image}
                  width={image.width}
                  height={image.height}
                  scaleX={Math.min(800 / image.width, 600 / image.height)}
                  scaleY={Math.min(800 / image.width, 600 / image.height)}
                />
              )}

              {/* Planes */}
              {Object.entries(planes).map(([key, plane]) => (
                <Line
                  key={plane.id}
                  points={plane.points}
                  stroke={plane.color}
                  strokeWidth={3}
                  opacity={0.8}
                />
              ))}

              {/* Lines */}
              {lines.map((line) => (
                <Line
                  key={line.id}
                  points={line.points}
                  stroke={line.color}
                  strokeWidth={2}
                />
              ))}

              {/* Measurement Lines */}
              {measurements.map((measurement) => (
                <Group key={measurement.id}>
                  <Line
                    points={[measurement.point1.x, measurement.point1.y, measurement.point2.x, measurement.point2.y]}
                    stroke="#FF5722"
                    strokeWidth={2}
                    dash={[8, 4]}
                  />
                  <Circle
                    x={measurement.point1.x}
                    y={measurement.point1.y}
                    radius={4}
                    fill="#FF5722"
                  />
                  <Circle
                    x={measurement.point2.x}
                    y={measurement.point2.y}
                    radius={4}
                    fill="#FF5722"
                  />
                </Group>
              ))}

              {/* Temporary drawing line */}
              {isDrawing && tempPoints.length === 2 && (
                <Line
                  points={tempPoints}
                  stroke="#FF5722"
                  strokeWidth={2}
                  dash={[5, 5]}
                />
              )}

              {/* Landmarks */}
              {Object.entries(landmarks).map(([key, position]) => {
                if (!position || typeof position !== 'object' || !position.x || !position.y) return null;

                const landmark = CEPHALOMETRIC_LANDMARKS[key];
                if (!landmark) return null;

                const isSelected = selectedId === key;

                return (
                  <Group key={key} id={key}>
                    <Circle
                      x={position.x}
                      y={position.y}
                      radius={8}
                      fill={landmark.color}
                      stroke="#ffffff"
                      strokeWidth={2}
                      draggable={!readOnly}
                      onDragEnd={(e) => {
                        if (readOnly) return;

                        const newPos = { x: e.target.x(), y: e.target.y() };
                        const newLandmarks = {
                          ...landmarks,
                          [key]: newPos,
                        };

                        setLandmarks(newLandmarks);
                        if (onLandmarkUpdate) {
                          onLandmarkUpdate(newLandmarks);
                        }
                      }}
                      onClick={() => setSelectedId(key)}
                    />
                    <Circle
                      x={position.x}
                      y={position.y}
                      radius={3}
                      fill="#000000"
                    />
                    <Text
                      x={position.x + 12}
                      y={position.y - 6}
                      text={key}
                      fontSize={14}
                      fill="#000000"
                      fontStyle="bold"
                    />
                  </Group>
                );
              })}

              {/* Transformer for selected elements */}
              {selectedId && (
                <Transformer
                  ref={transformerRef}
                  anchorSize={6}
                  borderDash={[3, 3]}
                  attachedTo={selectedId ? layerRef.current?.findOne(`#${selectedId}`) : null}
                />
              )}
            </Layer>
          </Stage>

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

      {/* Landmarks List */}
      {Object.keys(landmarks).length > 0 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            لندمارک‌های علامت‌گذاری شده ({Object.keys(landmarks).length})
          </Typography>

          <Grid container spacing={2}>
            {Object.entries(landmarks).map(([key, position]) => {
              const landmark = CEPHALOMETRIC_LANDMARKS[key];
              return (
                <Grid item xs={12} sm={6} md={4} key={key}>
                  <Box
                    sx={{
                      p: 2,
                      border: '1px solid #ddd',
                      borderRadius: 1,
                      backgroundColor: selectedId === key ? '#f5f5f5' : 'transparent'
                    }}
                  >
                    <Stack direction="row" alignItems="center" spacing={1}>
                      <Box
                        sx={{
                          width: 16,
                          height: 16,
                          borderRadius: '50%',
                          backgroundColor: landmark?.color || '#ccc',
                          border: '1px solid #000'
                        }}
                      />
                      <Box>
                        <Typography variant="subtitle2">
                          {key} - {landmark?.name || 'نامشخص'}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          ({Math.round(position.x)}, {Math.round(position.y)})
                        </Typography>
                      </Box>
                      {!readOnly && (
                        <IconButton
                          size="small"
                          color="error"
                          onClick={() => {
                            const newLandmarks = { ...landmarks };
                            delete newLandmarks[key];
                            setLandmarks(newLandmarks);
                            if (onLandmarkUpdate) {
                              onLandmarkUpdate(newLandmarks);
                            }
                          }}
                        >
                          <Iconify icon="solar:trash-bin-2-bold" width={16} />
                        </IconButton>
                      )}
                    </Stack>
                  </Box>
                </Grid>
              );
            })}
          </Grid>
        </Paper>
      )}

      {/* Measurements List */}
      {measurements.length > 0 && (
        <Paper sx={{ p: 2, mt: 2 }}>
          <Typography variant="h6" gutterBottom>
            اندازه‌گیری‌ها ({measurements.length})
          </Typography>

          <Grid container spacing={1}>
            {measurements.map((measurement) => (
              <Grid item xs={12} sm={6} md={4} key={measurement.id}>
                <Box
                  sx={{
                    p: 1,
                    border: '1px solid #ddd',
                    borderRadius: 1,
                  }}
                >
                  <Typography variant="body2">
                    طول: {measurement.distance.toFixed(2)} پیکسل
                  </Typography>
                  <IconButton
                    size="small"
                    color="error"
                    onClick={() => {
                      setMeasurements(prev => prev.filter(m => m.id !== measurement.id));
                    }}
                  >
                    <Iconify icon="solar:trash-bin-2-bold" width={12} />
                  </IconButton>
                </Box>
              </Grid>
            ))}
          </Grid>
        </Paper>
      )}

      {/* Instructions */}
      <Alert severity="info" sx={{ mt: 2 }}>
        <Typography variant="body2">
          <strong>راهنمایی Konva:</strong>
          {!readOnly ? (
            <>
              • روی ابزار مورد نظر کلیک کنید و سپس در تصویر کلیک کنید
              • برای رسم خط مستقیم از ابزار &quot;خط&quot; استفاده کنید
              • برای اندازه‌گیری فاصله از ابزار &quot;اندازه‌گیری&quot; استفاده کنید
              • روی لندمارک‌ها کلیک کرده و آن‌ها را جابجا کنید
              • از چرخ ماوس برای زوم استفاده کنید و کلیک کنید تا حرکت دهید
              • از تشخیص AI برای یافتن خودکار لندمارک‌ها استفاده کنید
            </>
          ) : (
            'این تصویر در حالت فقط خواندنی است'
          )}
        </Typography>
      </Alert>

      {/* Add Landmark Dialog */}
      <Dialog
        open={addLandmarkDialogOpen}
        onClose={() => {
          setAddLandmarkDialogOpen(false);
          setSelectedLandmarkType('');
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>
          انتخاب نوع لندمارک
        </DialogTitle>
        <DialogContent>
          <FormControl fullWidth sx={{ mt: 1 }}>
            <InputLabel>نوع لندمارک</InputLabel>
            <Select
              value={selectedLandmarkType}
              onChange={(e) => setSelectedLandmarkType(e.target.value)}
            >
              {Object.entries(CEPHALOMETRIC_LANDMARKS).map(([key, landmark]) => (
                <MenuItem key={key} value={key}>
                  {key} - {landmark.name}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => {
            setAddLandmarkDialogOpen(false);
            setSelectedLandmarkType('');
          }}>
            انصراف
          </Button>
          <Button variant="contained" onClick={handleAddLandmark}>
            اضافه کردن
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default CephalometricKonvaViewer;
