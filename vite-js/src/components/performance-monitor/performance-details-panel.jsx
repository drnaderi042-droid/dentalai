import { useState } from 'react';

import {
  Box,
  Card,
  Chip,
  Grid,
  Stack,
  Table,
  Paper,
  Divider,
  Collapse,
  TableRow,
  TableBody,
  TableCell,
  Typography,
  IconButton,
  CardContent,
  LinearProgress,
  TableContainer,
} from '@mui/material';

import { usePerformanceContext } from 'src/contexts/performance-context';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

/**
 * Performance Details Panel - نمایش جزئیات کامل یک کامپوننت
 */
export function PerformanceDetailsPanel({ componentName, position = 'bottom-right' }) {
  const { getComponentData, getProfilerData } = usePerformanceContext();
  const [expanded, setExpanded] = useState(true);

  const componentData = componentName ? getComponentData(componentName) : null;
  const profilerData = componentName ? getProfilerData(componentName) : null;

  const positionStyles = {
    'top-left': { top: 16, left: 16 },
    'top-right': { top: 16, right: 16 },
    'bottom-left': { bottom: 16, left: 16 },
    'bottom-right': { bottom: 16, right: 16 },
  };

  const getMemoryColor = (mb) => {
    if (mb < 5) return 'success';
    if (mb < 15) return 'warning';
    return 'error';
  };

  const getCPUColor = (usage) => {
    if (usage < 5) return 'success';
    if (usage < 15) return 'warning';
    return 'error';
  };

  if (!componentName || !componentData) {
    return (
      <Box
        sx={{
          position: 'fixed',
          zIndex: 9997,
          ...positionStyles[position],
          maxWidth: 400,
          width: '100%',
        }}
      >
        <Card sx={{ boxShadow: 6, borderRadius: 2 }}>
          <CardContent>
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 2 }}>
              کامپوننتی انتخاب نشده است
            </Typography>
          </CardContent>
        </Card>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        zIndex: 9997,
        ...positionStyles[position],
        maxWidth: 450,
        width: '100%',
        maxHeight: '80vh',
      }}
    >
      <Card
        sx={{
          boxShadow: 6,
          borderRadius: 2,
          display: 'flex',
          flexDirection: 'column',
          maxHeight: '80vh',
        }}
      >
        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            p: 1.5,
            bgcolor: 'background.neutral',
            cursor: 'pointer',
          }}
          onClick={() => setExpanded(!expanded)}
        >
          <Stack direction="row" spacing={1} alignItems="center" sx={{ flex: 1, minWidth: 0 }}>
            <Iconify icon="mdi:information-outline" width={20} />
            <Typography variant="subtitle2" noWrap>
              {componentName}
            </Typography>
          </Stack>
          <IconButton size="small" onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}>
            <Iconify icon={expanded ? 'eva:arrow-up-fill' : 'eva:arrow-down-fill'} />
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          <CardContent sx={{ pt: 2, overflow: 'auto' }}>
            {/* Memory Details */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  حافظه (RAM)
                </Typography>
                <Chip
                  label={`${(componentData.memory?.usedMB || 0).toFixed(2)} MB`}
                  size="small"
                  color={getMemoryColor(componentData.memory?.usedMB || 0)}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, ((componentData.memory?.usedMB || 0) / 50) * 100)}
                color={getMemoryColor(componentData.memory?.usedMB || 0)}
                sx={{ height: 8, borderRadius: 1, mb: 1.5 }}
              />
              <Grid container spacing={1}>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">
                    مصرف تخمینی
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(componentData.memory?.usedMB || 0).toFixed(2)} MB
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">
                    بر اساس زمان render
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(componentData.cpu?.renderTime || 0).toFixed(2)} ms
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* CPU Details */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="subtitle2" sx={{ fontWeight: 600 }}>
                  پردازنده (CPU)
                </Typography>
                <Chip
                  label={`${(componentData.cpu?.usagePercent || 0).toFixed(1)}%`}
                  size="small"
                  color={getCPUColor(componentData.cpu?.usagePercent || 0)}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, (componentData.cpu?.usagePercent || 0) * 5)}
                color={getCPUColor(componentData.cpu?.usagePercent || 0)}
                sx={{ height: 8, borderRadius: 1, mb: 1.5 }}
              />
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    مصرف نسبی
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(componentData.cpu?.usagePercent || 0).toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    زمان render
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {(componentData.cpu?.renderTime || 0).toFixed(2)} ms
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            <Divider sx={{ my: 2 }} />

            {/* Render Details */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                اطلاعات رندر
              </Typography>
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    زمان رندر
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {componentData.renderTime.toFixed(2)} ms
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    تعداد رندر
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {componentData.renderCount}
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Profiler Details */}
            {profilerData && (
              <>
                <Divider sx={{ my: 2 }} />
                <Box>
                  <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                    جزئیات Profiler
                  </Typography>
                  <TableContainer component={Paper} variant="outlined" sx={{ mt: 1 }}>
                    <Table size="small">
                      <TableBody>
                        <TableRow>
                          <TableCell>
                            <Typography variant="caption">Phase</Typography>
                          </TableCell>
                          <TableCell>
                            <Chip
                              label={profilerData.phase}
                              size="small"
                              color={profilerData.phase === 'mount' ? 'primary' : 'default'}
                            />
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>
                            <Typography variant="caption">Actual Duration</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {profilerData.actualDuration.toFixed(2)} ms
                            </Typography>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>
                            <Typography variant="caption">Base Duration</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {profilerData.baseDuration.toFixed(2)} ms
                            </Typography>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>
                            <Typography variant="caption">Start Time</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {profilerData.startTime.toFixed(2)} ms
                            </Typography>
                          </TableCell>
                        </TableRow>
                        <TableRow>
                          <TableCell>
                            <Typography variant="caption">Commit Time</Typography>
                          </TableCell>
                          <TableCell>
                            <Typography variant="body2" sx={{ fontWeight: 500 }}>
                              {profilerData.commitTime.toFixed(2)} ms
                            </Typography>
                          </TableCell>
                        </TableRow>
                      </TableBody>
                    </Table>
                  </TableContainer>
                </Box>
              </>
            )}
          </CardContent>
        </Collapse>
      </Card>
    </Box>
  );
}

