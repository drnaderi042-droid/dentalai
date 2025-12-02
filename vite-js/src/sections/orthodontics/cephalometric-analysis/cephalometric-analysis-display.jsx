import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';

import { Iconify } from 'src/components/iconify';

// Standard cephalometric lines and planes
const CEPHALOMETRIC_LINES = {
  // Skeletal Lines
  SN: { name: 'خط SN', points: ['S', 'N'], color: '#2196F3', width: 2 },
  NA: { name: 'خط NA', points: ['N', 'A'], color: '#2196F3', width: 2 },
  NB: { name: 'خط NB', points: ['N', 'B'], color: '#F44336', width: 2 },
  FH: { name: 'صفحه فرانکفورت', points: ['Po', 'Or'], color: '#4CAF50', width: 2 },
  GoGn: { name: 'صفحه mandibular', points: ['Go', 'Gn'], color: '#FF9800', width: 2 },
  GoMe: { name: 'خط Go-Me', points: ['Go', 'Me'], color: '#FF9800', width: 2 },

  // Dental Lines
  U1: { name: 'محور incisive فوقانی', points: ['U1', 'U1A'], color: '#E91E63', width: 2 },
  L1: { name: 'محور incisive تحتانی', points: ['L1', 'L1A'], color: '#9C27B0', width: 2 },

  // Reference Lines
  OP: { name: 'صفحه اکلوزال', points: ['ANS', 'PNS'], color: '#00BCD4', width: 1.5 },
  MP: { name: 'صفحه mandibular', points: ['Me', 'Go'], color: '#FF5722', width: 2 },
};

/**
 * Professional Cephalometric Analysis Display Component
 * Displays landmarks, lines, angles, and measurements on cephalometric image
 */
export function CephalometricAnalysisDisplay({
  imageUrl,
  landmarks = {},
  measurements = {},
  showGrid = false,
  onLandmarkEdit,
}) {
  const canvasRef = useRef(null);
  const [image, setImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  // Load image
  useEffect(() => {
    if (!imageUrl) return;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      setImage(img);
      setImageLoaded(true);
    };
    img.src = imageUrl;
  }, [imageUrl]);

  // Draw analysis
  useEffect(() => {
    if (!imageLoaded || !image || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Set canvas size
    canvas.width = image.width;
    canvas.height = image.height;

    // Draw image
    ctx.drawImage(image, 0, 0);

    // Draw grid if enabled
    if (showGrid) {
      drawGrid(ctx, canvas.width, canvas.height);
    }

    // Draw lines
    drawCephalometricLines(ctx, landmarks);

    // Draw landmarks
    drawLandmarks(ctx, landmarks);

    // Draw measurements
    drawMeasurements(ctx, landmarks, measurements);
  }, [imageLoaded, image, landmarks, measurements, showGrid, drawMeasurements]);

  const drawGrid = (ctx, width, height) => {
    ctx.strokeStyle = 'rgba(0, 255, 0, 0.2)';
    ctx.lineWidth = 0.5;

    // Vertical lines
    for (let x = 0; x < width; x += 50) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal lines
    for (let y = 0; y < height; y += 50) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }
  };

  const drawCephalometricLines = (ctx, landmarkData) => {
    Object.entries(CEPHALOMETRIC_LINES).forEach(([key, lineInfo]) => {
      const [point1Name, point2Name] = lineInfo.points;
      const point1 = landmarkData[point1Name];
      const point2 = landmarkData[point2Name];

      if (!point1 || !point2) return;

      ctx.strokeStyle = lineInfo.color;
      ctx.lineWidth = lineInfo.width;
      ctx.setLineDash([]);

      ctx.beginPath();
      ctx.moveTo(point1.x, point1.y);
      ctx.lineTo(point2.x, point2.y);
      ctx.stroke();
    });
  };

  const drawLandmarks = (ctx, landmarkData) => {
    Object.entries(landmarkData).forEach(([key, point]) => {
      if (!point || !point.x || !point.y) return;

      // Outer glow
      ctx.globalAlpha = 0.3;
      ctx.beginPath();
      ctx.arc(point.x, point.y, 10, 0, 2 * Math.PI);
      ctx.fillStyle = '#00FF00';
      ctx.fill();

      ctx.globalAlpha = 1.0;

      // Main point
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = '#00FF00';
      ctx.fill();

      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Crosshair
      ctx.strokeStyle = '#000000';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(point.x - 4, point.y);
      ctx.lineTo(point.x + 4, point.y);
      ctx.moveTo(point.x, point.y - 4);
      ctx.lineTo(point.x, point.y + 4);
      ctx.stroke();

      // Label with background
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      const textMetrics = ctx.measureText(key);
      const labelY = point.y + 18;

      ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
      ctx.fillRect(point.x - textMetrics.width / 2 - 2, labelY - 10, textMetrics.width + 4, 14);

      ctx.fillStyle = '#00FF00';
      ctx.fillText(key, point.x, labelY);
    });
  };

  const drawMeasurements = useCallback((ctx, landmarkData, measurementData) => {
    // Helper function to draw angle measurement
  const drawAngleMeasurement = (ctx, p1, vertex, p2, label, color) => {
    // Draw arc showing the angle
    const radius = 40;
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(vertex.x, vertex.y, radius, angle1, angle2);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw label
    const midAngle = (angle1 + angle2) / 2;
    const labelX = vertex.x + Math.cos(midAngle) * (radius + 20);
    const labelY = vertex.y + Math.sin(midAngle) * (radius + 20);

    ctx.font = 'bold 11px Arial';
    const textMetrics = ctx.measureText(label);

    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(labelX - textMetrics.width / 2 - 3, labelY - 8, textMetrics.width + 6, 16);

    ctx.fillStyle = color;
    ctx.textAlign = 'center';
    ctx.fillText(label, labelX, labelY + 4);
  };

    // Draw key measurements on the image
    if (measurementData.SNA && landmarkData.S && landmarkData.N && landmarkData.A) {
      drawAngleMeasurement(ctx, landmarkData.S, landmarkData.N, landmarkData.A, `SNA: ${measurementData.SNA}°`, '#2196F3');
    }

    if (measurementData.SNB && landmarkData.S && landmarkData.N && landmarkData.B) {
      drawAngleMeasurement(ctx, landmarkData.S, landmarkData.N, landmarkData.B, `SNB: ${measurementData.SNB}°`, '#F44336');
    }
  }, []);

  return (
    <Box>
      {!imageLoaded && (
        <Alert severity="info" sx={{ mb: 2 }}>
          در حال بارگذاری تصویر...
        </Alert>
      )}

      <Grid container spacing={2}>
        {/* Canvas Display */}
        <Grid item xs={12} md={8}>
          <Card sx={{ p: 2 }}>
            <canvas
              ref={canvasRef}
              style={{
                width: '100%',
                height: 'auto',
                maxHeight: '70vh',
                border: '1px solid #ddd',
                borderRadius: '8px',
              }}
            />
          </Card>
        </Grid>

        {/* Measurements Panel */}
        <Grid item xs={12} md={4}>
          <Card sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              نتایج آنالیز سفالومتریک
            </Typography>
            <Divider sx={{ mb: 2 }} />

            {Object.keys(measurements).length > 0 ? (
              <Stack spacing={1.5}>
                {Object.entries(measurements).map(([key, value]) => (
                  <Box
                    key={key}
                    sx={{
                      p: 1.5,
                      borderRadius: 1,
                      bgcolor: 'background.neutral',
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="body2" fontWeight="bold">
                      {key}
                    </Typography>
                    <Typography variant="h6" color="primary.main">
                      {value}
                    </Typography>
                    {getMeasurementInterpretation(key, value) && (
                      <Chip
                        size="small"
                        label={getMeasurementInterpretation(key, value)}
                        color={getMeasurementColor(key, value)}
                        sx={{ mt: 0.5 }}
                      />
                    )}
                  </Box>
                ))}
              </Stack>
            ) : (
              <Alert severity="info">
                هنوز اندازه‌گیری انجام نشده است. لطفاً نقاط آناتومیک را علامت‌گذاری کنید.
              </Alert>
            )}

            <Divider sx={{ my: 2 }} />

            <Typography variant="subtitle2" gutterBottom>
              تعداد نقاط علامت‌گذاری شده
            </Typography>
            <Typography variant="h4" color="success.main">
              {Object.keys(landmarks).length}
            </Typography>

            {onLandmarkEdit && (
              <Button
                fullWidth
                variant="outlined"
                startIcon={<Iconify icon="solar:pen-bold" />}
                onClick={onLandmarkEdit}
                sx={{ mt: 2 }}
              >
                ویرایش لندمارک‌ها
              </Button>
            )}
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}

// Helper functions
function getMeasurementInterpretation(key, value) {
  const numValue = parseFloat(value);
  if (Number.isNaN(numValue)) return null;

  const norms = {
    SNA: { normal: [80, 84], label: 'موقعیت ماگزیلا' },
    SNB: { normal: [78, 82], label: 'موقعیت mandible' },
    ANB: { normal: [0, 4], label: 'رابطه بین فکین' },
  };

  const norm = norms[key];
  if (!norm) return null;

  if (numValue < norm.normal[0]) return 'کمتر از نرمال';
  if (numValue > norm.normal[1]) return 'بیشتر از نرمال';
  return 'نرمال';
}

function getMeasurementColor(key, value) {
  const interpretation = getMeasurementInterpretation(key, value);
  if (interpretation === 'نرمال') return 'success';
  return 'warning';
}

export default CephalometricAnalysisDisplay;

