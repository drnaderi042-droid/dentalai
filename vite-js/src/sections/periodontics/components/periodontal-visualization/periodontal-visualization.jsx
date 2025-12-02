import React, { useMemo, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import { useTheme } from '@mui/material/styles';
import InputLabel from '@mui/material/InputLabel';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';

// ----------------------------------------------------------------------

// Tooth numbering (Quadrant-based: 1-8 per quadrant) - 32 teeth total
const ALL_TEETH = [
  // Upper Right (Universal: 1-8, Quadrant: 1-8)
  { universalNumber: 1, quadrantNumber: 1, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 2, quadrantNumber: 2, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 3, quadrantNumber: 3, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 4, quadrantNumber: 4, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 5, quadrantNumber: 5, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 6, quadrantNumber: 6, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 7, quadrantNumber: 7, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  { universalNumber: 8, quadrantNumber: 8, quadrant: 'Upper Right', jaw: 'Upper', side: 'Right' },
  // Upper Left (Universal: 9-16, Quadrant: 1-8)
  { universalNumber: 9, quadrantNumber: 1, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 10, quadrantNumber: 2, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 11, quadrantNumber: 3, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 12, quadrantNumber: 4, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 13, quadrantNumber: 5, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 14, quadrantNumber: 6, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 15, quadrantNumber: 7, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  { universalNumber: 16, quadrantNumber: 8, quadrant: 'Upper Left', jaw: 'Upper', side: 'Left' },
  // Lower Left (Universal: 17-24, Quadrant: 1-8)
  { universalNumber: 17, quadrantNumber: 1, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 18, quadrantNumber: 2, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 19, quadrantNumber: 3, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 20, quadrantNumber: 4, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 21, quadrantNumber: 5, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 22, quadrantNumber: 6, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 23, quadrantNumber: 7, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  { universalNumber: 24, quadrantNumber: 8, quadrant: 'Lower Left', jaw: 'Lower', side: 'Left' },
  // Lower Right (Universal: 25-32, Quadrant: 1-8)
  { universalNumber: 25, quadrantNumber: 1, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 26, quadrantNumber: 2, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 27, quadrantNumber: 3, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 28, quadrantNumber: 4, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 29, quadrantNumber: 5, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 30, quadrantNumber: 6, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 31, quadrantNumber: 7, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
  { universalNumber: 32, quadrantNumber: 8, quadrant: 'Lower Right', jaw: 'Lower', side: 'Right' },
];

// ----------------------------------------------------------------------

export function PeriodontalVisualization({ chartData }) {
  const theme = useTheme();
  const [selectedSurface, setSelectedSurface] = React.useState('facial'); // 'facial' or 'lingual'

  // Organize teeth for display: Upper jaw (1-16) and Lower jaw (17-32)
  // In dental chart view (as if looking at patient's mouth):
  // Upper: From right to left (patient's perspective): 8,7,6,5,4,3,2,1 (Upper Right reversed) then 1,2,3,4,5,6,7,8 (Upper Left)
  // Lower: From left to right (patient's perspective): 1,2,3,4,5,6,7,8 (Lower Left) then 8,7,6,5,4,3,2,1 (Lower Right reversed)
  // The lower jaw should mirror the upper jaw exactly
  const organizedTeeth = useMemo(() => {
    const upperTeeth = [];
    const lowerTeeth = [];

    // Upper jaw: Display as if looking at patient's mouth
    // Upper Right (universal 1-8): Display in reverse order (8 to 1) - quadrant numbers 8,7,6,5,4,3,2,1
    for (let i = 8; i >= 1; i--) {
      upperTeeth.push(ALL_TEETH.find((t) => t.universalNumber === i));
    }
    // Upper Left (universal 9-16): Display in normal order (9 to 16) - quadrant numbers 1,2,3,4,5,6,7,8
    for (let i = 9; i <= 16; i++) {
      upperTeeth.push(ALL_TEETH.find((t) => t.universalNumber === i));
    }

    // Lower jaw: Display exactly like upper jaw (same order)
    // Lower Right (universal 25-32): Display in reverse order (32 to 25) - quadrant numbers 8,7,6,5,4,3,2,1
    // This matches the upper jaw pattern: right side first (reversed), then left side (normal)
    for (let i = 32; i >= 25; i--) {
      const tooth = ALL_TEETH.find((t) => t.universalNumber === i);
      if (tooth) lowerTeeth.push(tooth);
    }
    // Lower Left (universal 17-24): Display in normal order (17 to 24) - quadrant numbers 1,2,3,4,5,6,7,8
    // This matches the upper jaw pattern: left side second (normal order)
    for (let i = 17; i <= 24; i++) {
      const tooth = ALL_TEETH.find((t) => t.universalNumber === i);
      if (tooth) lowerTeeth.push(tooth);
    }

    // Debug: Log the order to verify
    // console.log('Lower teeth order:', lowerTeeth.map(t => t ? `${t.universalNumber} (q${t.quadrantNumber})` : 'null'));

    return { upperTeeth, lowerTeeth };
  }, []);

  // Configuration
  const TOOTH_WIDTH = 35; // Smaller width to fit 16 teeth
  const TOOTH_HEIGHT = 100;
  const TOOTH_SPACING = 5;
  const CEJ_Y = 50; // Y position of CEJ reference line (in SVG coordinates, top is 0)
  const MM_PER_PIXEL = 2.5; // Scale: 1mm = 2.5 pixels
  const GRID_SPACING = 3 * MM_PER_PIXEL; // 3mm grid spacing
  const ROW_SPACING = 20; // Space between upper and lower rows

  // Calculate continuous paths across all teeth for a jaw (upper or lower)
  const calculateContinuousPaths = useCallback((teethArray, yOffset) => {
    if (!chartData || !teethArray || teethArray.length === 0) {
      return {
        gmLinePath: null,
        pdLinePath: null,
        gmShadedPath: null,
        pdShadedPath: null,
      };
    }

    const allGMPoints = [];
    const allPDPoints = [];
    const sitePositions = [0.2, 0.5, 0.8]; // Relative positions across tooth width: Mesial/Buccal, Buccal/Central, Distal/Buccal

    // Collect all GM and PD points from all teeth with their absolute X positions
    // Process teeth in order to maintain continuity
    let lastGMPoint = null;
    let lastPDPoint = null;

    teethArray.forEach((tooth, toothIndex) => {
      if (!tooth) return;

      const toothData = chartData?.teeth?.[tooth.universalNumber];
      const toothXOffset = toothIndex * (TOOTH_WIDTH + TOOTH_SPACING);

      if (!toothData || toothData.missing) {
        // Reset last points when encountering missing tooth (break in continuity)
        lastGMPoint = null;
        lastPDPoint = null;
        return;
      }

      const surfaceData = toothData[selectedSurface] || {};
      const gmValues = surfaceData.gingivalMargin || [null, null, null];
      const pdValues = surfaceData.pocketDepth || [null, null, null];

      // If this tooth has data and we had a previous tooth, bridge the gap
      if (lastGMPoint !== null) {
        // Calculate the gap position: end of previous tooth to start of this tooth
        // Previous tooth ended at: (toothIndex - 1) * (TOOTH_WIDTH + TOOTH_SPACING) + TOOTH_WIDTH
        // This tooth starts at: toothIndex * (TOOTH_WIDTH + TOOTH_SPACING)
        // Gap is at: (toothIndex - 1) * (TOOTH_WIDTH + TOOTH_SPACING) + TOOTH_WIDTH + TOOTH_SPACING/2
        const prevToothEndX = (toothIndex - 1) * (TOOTH_WIDTH + TOOTH_SPACING) + TOOTH_WIDTH;
        const thisToothStartX = toothXOffset;
        const gapCenterX = (prevToothEndX + thisToothStartX) / 2;
        
        // Add bridging points using the Y values from last point for smooth transition
        allGMPoints.push({ x: gapCenterX, y: lastGMPoint.y, value: lastGMPoint.value, toothIndex, siteIndex: -1, isBridge: true });
        
        if (lastPDPoint !== null) {
          allPDPoints.push({ x: gapCenterX, y: lastPDPoint.y, value: lastPDPoint.value, toothIndex, siteIndex: -1, isBridge: true });
        }
      }

      // Collect points for this tooth in order (from mesial to distal)
      sitePositions.forEach((relativeX, siteIndex) => {
        const gm = gmValues[siteIndex];
        const pd = pdValues[siteIndex];

        // Absolute X position
        const absoluteX = toothXOffset + (relativeX * TOOTH_WIDTH);

        // Treat null/undefined GM as 0
        const gmValue = gm !== null && gm !== undefined ? gm : 0;
        const gmY = CEJ_Y + yOffset - (gmValue * MM_PER_PIXEL);
        const gmPoint = { x: absoluteX, y: gmY, value: gmValue, toothIndex, siteIndex };
        allGMPoints.push(gmPoint);
        lastGMPoint = gmPoint;

        if (pd !== null && pd !== undefined) {
          // Use gmValue (0 if null/undefined) for PD calculation
          const pdY = gmY + (pd * MM_PER_PIXEL);
          const pdPoint = { x: absoluteX, y: pdY, value: pd, toothIndex, siteIndex };
          allPDPoints.push(pdPoint);
          lastPDPoint = pdPoint;
        }
      });
    });

    // Create continuous line paths that connect across all teeth
    const createLinePath = (points) => {
      if (points.length === 0) return null;
      if (points.length === 1) {
        return `M ${points[0].x} ${points[0].y}`;
      }

      const pathData = [];
      pathData.push(`M ${points[0].x} ${points[0].y}`);

      // Connect all points in order
      for (let i = 1; i < points.length; i++) {
        pathData.push(`L ${points[i].x} ${points[i].y}`);
      }

      return pathData.join(' ');
    };

    // Create continuous GM line path
    const gmLinePath = createLinePath(allGMPoints);

    // Create continuous PD line path
    const pdLinePath = createLinePath(allPDPoints);

    // Create continuous GM shaded area (recession) - between CEJ and GM line
    let gmShadedPath = null;
    if (allGMPoints.length > 0) {
      const hasRecession = allGMPoints.some((p) => p.value > 0);
      if (hasRecession) {
        const pathData = [];
        const cejY = CEJ_Y + yOffset;

        // Start from leftmost point at CEJ level
        pathData.push(`M ${allGMPoints[0].x} ${cejY}`);

        // Draw through GM points (use GM Y if recession, otherwise CEJ Y)
        allGMPoints.forEach((point) => {
          const y = point.value > 0 ? point.y : cejY;
          pathData.push(`L ${point.x} ${y}`);
        });

        // Close path: go back to CEJ at rightmost point
        const lastX = allGMPoints[allGMPoints.length - 1].x;
        pathData.push(`L ${lastX} ${cejY}`);
        pathData.push('Z');

        gmShadedPath = pathData.join(' ');
      }
    }

    // Create continuous PD shaded area - between GM line and PD line
    let pdShadedPath = null;
    if (allGMPoints.length > 0 && allPDPoints.length > 0) {
      // Match GM and PD points - they should have the same order and X positions
      // Create pairs by matching toothIndex and siteIndex (including bridge points)
      const matchedPoints = [];
      const pdPointsMap = new Map();
      allPDPoints.forEach((p) => {
        const key = `${p.toothIndex}-${p.siteIndex}`;
        // Allow multiple points with same key (for bridge points), use last one
        pdPointsMap.set(key, p);
      });

      allGMPoints.forEach((gmPoint) => {
        const key = `${gmPoint.toothIndex}-${gmPoint.siteIndex}`;
        const pdPoint = pdPointsMap.get(key);
        // Include bridge points and real points, but only if PD value > 0 for real points
        if (pdPoint) {
          if (gmPoint.isBridge || (pdPoint.value > 0 && !pdPoint.isBridge)) {
            matchedPoints.push({ gm: gmPoint, pd: pdPoint });
          }
        }
      });

      if (matchedPoints.length > 0) {
        const pathData = [];

        // Start from first GM point
        pathData.push(`M ${matchedPoints[0].gm.x} ${matchedPoints[0].gm.y}`);

        // Draw through all GM points in order (continuous line)
        for (let i = 1; i < matchedPoints.length; i++) {
          pathData.push(`L ${matchedPoints[i].gm.x} ${matchedPoints[i].gm.y}`);
        }

        // Continue through PD points in reverse order (to close the polygon)
        for (let i = matchedPoints.length - 1; i >= 0; i--) {
          pathData.push(`L ${matchedPoints[i].pd.x} ${matchedPoints[i].pd.y}`);
        }

        // Close path back to first GM point
        pathData.push('Z');

        pdShadedPath = pathData.join(' ');
      }
    }

    return {
      gmLinePath,
      pdLinePath,
      gmShadedPath,
      pdShadedPath,
    };
  }, [chartData, selectedSurface]);

  // Calculate continuous paths for upper and lower jaws
  const upperJawPaths = useMemo(
    () => calculateContinuousPaths(organizedTeeth.upperTeeth, 0),
    [organizedTeeth.upperTeeth, calculateContinuousPaths]
  );

  const lowerJawPaths = useMemo(
    () => calculateContinuousPaths(organizedTeeth.lowerTeeth, TOOTH_HEIGHT + ROW_SPACING),
    [organizedTeeth.lowerTeeth, calculateContinuousPaths]
  );

  // Calculate points for individual tooth (for backward compatibility if needed)
  const calculatePoints = (toothData, surface) => {
    if (!toothData || toothData.missing) {
      return {
        gmPoints: [],
        pdPoints: [],
      };
    }

    const surfaceData = toothData[surface] || {};
    const gmValues = surfaceData.gingivalMargin || [null, null, null];
    const pdValues = surfaceData.pocketDepth || [null, null, null];

    const gmPoints = [];
    const pdPoints = [];
    const sitePositions = [0.2, 0.5, 0.8];

    sitePositions.forEach((xPos, index) => {
      const gm = gmValues[index];
      const pd = pdValues[index];

      // Treat null/undefined GM as 0
      const gmValue = gm !== null && gm !== undefined ? gm : 0;
      const gmY = CEJ_Y - (gmValue * MM_PER_PIXEL);
      gmPoints.push({ x: xPos, y: gmY, value: gmValue });

      if (pd !== null && pd !== undefined) {
        // Use gmValue (0 if null/undefined) for PD calculation
        const pdY = gmY + (pd * MM_PER_PIXEL);
        pdPoints.push({ x: xPos, y: pdY, value: pd });
      }
    });

    return { gmPoints, pdPoints };
  };

  // Render a single tooth (without paths - paths are rendered separately as continuous)
  const renderTooth = (tooth, index, yOffset = 0) => {
    if (!tooth) return null;
    
    const toothData = chartData?.teeth?.[tooth.universalNumber];
    const xOffset = index * (TOOTH_WIDTH + TOOTH_SPACING);

    // Determine image file name
    let imageFileName;
    if (tooth.jaw === 'Upper') {
      imageFileName = `${tooth.quadrantNumber}.png`;
    } else {
      imageFileName = `${tooth.quadrantNumber + 10}.png`;
    }

    const isImplant = !!(toothData?.implant && !toothData?.missing);
    const imageSrc = isImplant ? '/teeth/implant.png' : `/teeth/${imageFileName}`;

    return (
      <g key={tooth.universalNumber} transform={`translate(${xOffset}, ${yOffset})`}>
        {/* Tooth image/background - using SVG image element for proper z-index */}
        <image
          x={0}
          y={0}
          width={TOOTH_WIDTH}
          height={TOOTH_HEIGHT}
          href={imageSrc}
          opacity={!toothData || toothData.missing ? 0.3 : 1}
          style={{
            objectFit: 'contain',
            filter: !toothData || toothData.missing ? 'grayscale(100%)' : 'none',
          }}
          onError={(e) => {
            e.target.style.display = 'none';
          }}
        />

        {/* Tooth number label */}
        <text
          x={TOOTH_WIDTH / 2}
          y={TOOTH_HEIGHT + 15}
          textAnchor="middle"
          fontSize={10}
          fill={theme.palette.text.primary}
          fontWeight="bold"
        >
          {tooth.quadrantNumber}
        </text>
      </g>
    );
  };

  if (!chartData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body2" color="text.secondary" textAlign="center">
            چارت پریودونتال ثبت نشده است
          </Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Box
      sx={{
        border: `1px solid ${theme.palette.divider}`,
        borderRadius: 1,
        bgcolor: 'background.paper',
        overflow: 'hidden',
      }}
    >
      <Box sx={{ p: 2.5, borderBottom: `1px solid ${theme.palette.divider}` }}>
        <Typography variant="subtitle1" fontWeight={600} gutterBottom>
          نمایش بصری چارت پریودونتال
        </Typography>
        <Typography variant="caption" color="text.secondary">
          نواحی آبی: پسروی لثه (Gingival Recession) | نواحی قرمز: عمق پاکت (Pocket Depth)
        </Typography>
      </Box>
      <Box sx={{ p: 2.5 }}>
        <Stack spacing={3}>
          {/* Controls */}
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControl size="small" sx={{ minWidth: 150 }}>
              <InputLabel>سطح</InputLabel>
              <Select
                value={selectedSurface}
                label="سطح"
                onChange={(e) => setSelectedSurface(e.target.value)}
              >
                <MenuItem value="facial">Facial (باکال)</MenuItem>
                <MenuItem value="lingual">Lingual (پالاتال)</MenuItem>
              </Select>
            </FormControl>
          </Stack>

          {/* Legend */}
          <Box
            sx={{
              display: 'flex',
              gap: 3,
              p: 2,
              bgcolor: 'background.neutral',
              borderRadius: 1,
            }}
          >
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 20,
                  height: 3,
                  bgcolor: theme.palette.error.main,
                }}
              />
              <Typography variant="caption">خط مرجع CEJ</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 20,
                  height: 3,
                  bgcolor: theme.palette.info.main,
                }}
              />
              <Typography variant="caption">حاشیه لثه (Gingival Margin)</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 20,
                  height: 12,
                  bgcolor: theme.palette.info.main,
                  opacity: 0.3,
                }}
              />
              <Typography variant="caption">پسروی لثه (Recession)</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 20,
                  height: 3,
                  bgcolor: theme.palette.error.main,
                }}
              />
              <Typography variant="caption">عمق پاکت (Pocket Depth)</Typography>
            </Stack>
            <Stack direction="row" spacing={1} alignItems="center">
              <Box
                sx={{
                  width: 20,
                  height: 12,
                  bgcolor: theme.palette.error.main,
                  opacity: 0.4,
                }}
              />
              <Typography variant="caption">پاکت پریودونتال</Typography>
            </Stack>
          </Box>

          {/* SVG Visualization */}
          <Box
            sx={{
              overflowX: 'auto',
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <svg
              width={16 * (TOOTH_WIDTH + TOOTH_SPACING)}
              height={TOOTH_HEIGHT * 2 + ROW_SPACING + 30}
              style={{ display: 'block' }}
            >
              {/* Background */}
              <rect
                width="100%"
                height="100%"
                fill={theme.palette.background.paper}
              />

              {/* Grid lines across entire chart */}
              {Array.from({ length: Math.floor((TOOTH_HEIGHT * 2 + ROW_SPACING) / GRID_SPACING) + 1 }).map((_, i) => {
                const y = i * GRID_SPACING;
                return (
                  <g key={`grid-${i}`}>
                    <line
                      x1={0}
                      y1={y}
                      x2={16 * (TOOTH_WIDTH + TOOTH_SPACING)}
                      y2={y}
                      stroke={theme.palette.divider}
                      strokeWidth={0.5}
                      opacity={0.2}
                    />
                    {y < TOOTH_HEIGHT && (
                      <line
                        x1={0}
                        y1={TOOTH_HEIGHT + ROW_SPACING + y}
                        x2={16 * (TOOTH_WIDTH + TOOTH_SPACING)}
                        y2={TOOTH_HEIGHT + ROW_SPACING + y}
                        stroke={theme.palette.divider}
                        strokeWidth={0.5}
                        opacity={0.2}
                      />
                    )}
                  </g>
                );
              })}

              {/* CEJ Reference Line (Red horizontal line across all teeth) */}
              <line
                x1={0}
                y1={CEJ_Y}
                x2={16 * (TOOTH_WIDTH + TOOTH_SPACING)}
                y2={CEJ_Y}
                stroke={theme.palette.error.main}
                strokeWidth={2}
              />
              <line
                x1={0}
                y1={TOOTH_HEIGHT + ROW_SPACING + CEJ_Y}
                x2={16 * (TOOTH_WIDTH + TOOTH_SPACING)}
                y2={TOOTH_HEIGHT + ROW_SPACING + CEJ_Y}
                stroke={theme.palette.error.main}
                strokeWidth={2}
              />

              {/* Render upper jaw teeth (1-16) - render first so paths appear on top */}
              {organizedTeeth.upperTeeth.map((tooth, index) => renderTooth(tooth, index, 0))}

              {/* Render lower jaw teeth (17-32) - render first so paths appear on top */}
              {organizedTeeth.lowerTeeth.map((tooth, index) => renderTooth(tooth, index, TOOTH_HEIGHT + ROW_SPACING))}

              {/* Continuous shaded areas and lines for upper jaw - rendered after teeth */}
              {/* Blue shaded area (Gingival Recession) - Upper */}
              {upperJawPaths.gmShadedPath && (
                <path
                  d={upperJawPaths.gmShadedPath}
                  fill={theme.palette.info.main}
                  fillOpacity={0.3}
                  stroke={theme.palette.info.main}
                  strokeWidth={1.5}
                />
              )}

              {/* Blue line (Gingival Margin) - Upper */}
              {upperJawPaths.gmLinePath && (
                <path
                  d={upperJawPaths.gmLinePath}
                  fill="none"
                  stroke={theme.palette.info.main}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Red shaded area (Pocket Depth) - Upper */}
              {upperJawPaths.pdShadedPath && (
                <path
                  d={upperJawPaths.pdShadedPath}
                  fill={theme.palette.error.main}
                  fillOpacity={0.4}
                  stroke={theme.palette.error.main}
                  strokeWidth={1.5}
                />
              )}

              {/* Red line (Pocket Base) - Upper */}
              {upperJawPaths.pdLinePath && (
                <path
                  d={upperJawPaths.pdLinePath}
                  fill="none"
                  stroke={theme.palette.error.main}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Continuous shaded areas and lines for lower jaw - rendered after teeth */}
              {/* Blue shaded area (Gingival Recession) - Lower */}
              {lowerJawPaths.gmShadedPath && (
                <path
                  d={lowerJawPaths.gmShadedPath}
                  fill={theme.palette.info.main}
                  fillOpacity={0.3}
                  stroke={theme.palette.info.main}
                  strokeWidth={1.5}
                />
              )}

              {/* Blue line (Gingival Margin) - Lower */}
              {lowerJawPaths.gmLinePath && (
                <path
                  d={lowerJawPaths.gmLinePath}
                  fill="none"
                  stroke={theme.palette.info.main}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Red shaded area (Pocket Depth) - Lower */}
              {lowerJawPaths.pdShadedPath && (
                <path
                  d={lowerJawPaths.pdShadedPath}
                  fill={theme.palette.error.main}
                  fillOpacity={0.4}
                  stroke={theme.palette.error.main}
                  strokeWidth={1.5}
                />
              )}

              {/* Red line (Pocket Base) - Lower */}
              {lowerJawPaths.pdLinePath && (
                <path
                  d={lowerJawPaths.pdLinePath}
                  fill="none"
                  stroke={theme.palette.error.main}
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Labels for jaws */}
              <text
                x={-40}
                y={TOOTH_HEIGHT / 2}
                fontSize={14}
                fill={theme.palette.text.primary}
                textAnchor="middle"
                fontWeight="bold"
                transform="rotate(-90, -40, 50)"
              >
                {selectedSurface === 'facial' ? 'Buccal' : 'Palatal'} (فک بالا)
              </text>
              <text
                x={-40}
                y={TOOTH_HEIGHT + ROW_SPACING + TOOTH_HEIGHT / 2}
                fontSize={14}
                fill={theme.palette.text.primary}
                textAnchor="middle"
                fontWeight="bold"
                transform="rotate(-90, -40, 150)"
              >
                {selectedSurface === 'facial' ? 'Buccal' : 'Palatal'} (فک پایین)
              </text>
            </svg>
          </Box>
        </Stack>
      </Box>
    </Box>
  );
}

