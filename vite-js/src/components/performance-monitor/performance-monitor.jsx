import { useState } from 'react';

import { Box, Card, Chip, Stack, Collapse, Typography, IconButton, CardContent, LinearProgress } from '@mui/material';

import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

/**
 * کامپوننت PerformanceMonitor
 * نمایش مصرف CPU و RAM برای یک کامپوننت خاص
 */
export function PerformanceMonitor({ componentName, interval = 1000, position = 'bottom-right', showOnMount = false }) {
  const [expanded, setExpanded] = useState(showOnMount);
  const metrics = usePerformanceMonitor(componentName, { interval });

  const positionStyles = {
    'top-left': { top: 16, left: 16 },
    'top-right': { top: 16, right: 16 },
    'bottom-left': { bottom: 16, left: 16 },
    'bottom-right': { bottom: 16, right: 16 },
  };

  const getMemoryColor = (percentage) => {
    if (percentage < 50) return 'success';
    if (percentage < 80) return 'warning';
    return 'error';
  };

  const getCPUColor = (usage) => {
    if (usage < 30) return 'success';
    if (usage < 70) return 'warning';
    return 'error';
  };

  return (
    <Box
      sx={{
        position: 'fixed',
        zIndex: 9999,
        ...positionStyles[position],
        maxWidth: 320,
        width: '100%',
      }}
    >
      <Card
        sx={{
          boxShadow: 6,
          borderRadius: 2,
          overflow: 'visible',
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
          <Stack direction="row" spacing={1} alignItems="center">
            <Iconify icon="mdi:chart-line" width={20} />
            <Typography variant="subtitle2">Performance Monitor</Typography>
            <Chip
              label={componentName}
              size="small"
              color="primary"
              sx={{ height: 20, fontSize: '0.65rem' }}
            />
          </Stack>
          <IconButton size="small" onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}>
            <Iconify icon={expanded ? 'eva:arrow-up-fill' : 'eva:arrow-down-fill'} />
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          <CardContent sx={{ pt: 2 }}>
            {/* Memory Metrics */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  حافظه (RAM)
                </Typography>
                <Chip
                  label={`${metrics.memory.percentage.toFixed(1)}%`}
                  size="small"
                  color={getMemoryColor(metrics.memory.percentage)}
                  sx={{ height: 20 }}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, metrics.memory.percentage)}
                color={getMemoryColor(metrics.memory.percentage)}
                sx={{ height: 8, borderRadius: 1, mb: 1 }}
              />
              <Stack direction="row" justifyContent="space-between" spacing={2}>
                <Typography variant="caption" color="text.secondary">
                  استفاده شده: {metrics.memory.used.toFixed(2)} MB
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  کل: {metrics.memory.total.toFixed(2)} MB
                </Typography>
              </Stack>
              {metrics.memory.limit && (
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 0.5 }}>
                  محدودیت: {metrics.memory.limit.toFixed(2)} MB
                </Typography>
              )}
            </Box>

            {/* CPU Metrics */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  پردازنده (CPU)
                </Typography>
                <Chip
                  label={`${metrics.cpu.usage.toFixed(1)}%`}
                  size="small"
                  color={getCPUColor(metrics.cpu.usage)}
                  sx={{ height: 20 }}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, metrics.cpu.usage)}
                color={getCPUColor(metrics.cpu.usage)}
                sx={{ height: 8, borderRadius: 1, mb: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                بار پردازشی: {metrics.cpu.load.toFixed(1)}%
              </Typography>
            </Box>

            {/* Render Metrics */}
            <Box>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  رندر
                </Typography>
              </Stack>
              <Stack direction="row" justifyContent="space-between" spacing={2}>
                <Typography variant="caption" color="text.secondary">
                  زمان: {metrics.renderTime.toFixed(2)} ms
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  تعداد: {metrics.renderCount}
                </Typography>
              </Stack>
            </Box>
          </CardContent>
        </Collapse>
      </Card>
    </Box>
  );
}


