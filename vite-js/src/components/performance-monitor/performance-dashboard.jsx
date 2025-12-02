import { useState } from 'react';

import {
  Box,
  Card,
  Chip,
  Stack,
  Table,
  Paper,
  Switch,
  Collapse,
  TableRow,
  TableBody,
  TableCell,
  TableHead,
  Typography,
  IconButton,
  CardContent,
  TableContainer,
  FormControlLabel,
} from '@mui/material';

import { usePerformanceMonitor } from 'src/hooks/use-performance-monitor';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

/**
 * Performance Dashboard - نمایش عملکرد چندین کامپوننت به صورت همزمان
 */
export function PerformanceDashboard({ components = [], position = 'bottom-right' }) {
  const [expanded, setExpanded] = useState(false);
  const [enabled, setEnabled] = useState(true);

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

  if (!enabled || components.length === 0) {
    return null;
  }

  return (
    <Box
      sx={{
        position: 'fixed',
        zIndex: 9999,
        ...positionStyles[position],
        maxWidth: 600,
        width: '100%',
        maxHeight: '80vh',
        overflow: 'auto',
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
            <Iconify icon="mdi:chart-multiple" width={20} />
            <Typography variant="subtitle2">Performance Dashboard</Typography>
            <Chip label={`${components.length} کامپوننت`} size="small" color="primary" sx={{ height: 20, fontSize: '0.65rem' }} />
          </Stack>
          <Stack direction="row" spacing={1}>
            <FormControlLabel
              control={<Switch size="small" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} />}
              label="فعال"
              sx={{ m: 0 }}
              onClick={(e) => e.stopPropagation()}
            />
            <IconButton size="small" onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}>
              <Iconify icon={expanded ? 'eva:arrow-up-fill' : 'eva:arrow-down-fill'} />
            </IconButton>
          </Stack>
        </Box>

        <Collapse in={expanded}>
          <CardContent sx={{ pt: 2, p: 0 }}>
            <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>کامپوننت</TableCell>
                    <TableCell align="center">RAM</TableCell>
                    <TableCell align="center">CPU</TableCell>
                    <TableCell align="center">زمان رندر</TableCell>
                    <TableCell align="center">تعداد رندر</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {components.map((componentName) => (
                    <ComponentRow key={componentName} componentName={componentName} />
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </CardContent>
        </Collapse>
      </Card>
    </Box>
  );
}

// کامپوننت ردیف جدول
function ComponentRow({ componentName }) {
  const metrics = usePerformanceMonitor(componentName, { 
    interval: 2000,
    trackMemory: true,
    trackCPU: true,
  });

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
    <TableRow hover>
      <TableCell>
        <Typography variant="caption" sx={{ fontWeight: 600 }}>
          {componentName}
        </Typography>
      </TableCell>
      <TableCell align="center">
        <Stack spacing={0.5} alignItems="center">
          <Chip
            label={`${(metrics.memory?.usedMB || 0).toFixed(1)} MB`}
            size="small"
            color={getMemoryColor(metrics.memory?.usedMB || 0)}
            sx={{ height: 20, fontSize: '0.65rem' }}
          />
        </Stack>
      </TableCell>
      <TableCell align="center">
        <Chip
          label={`${(metrics.cpu?.usagePercent || 0).toFixed(1)}%`}
          size="small"
          color={getCPUColor(metrics.cpu?.usagePercent || 0)}
          sx={{ height: 20, fontSize: '0.65rem' }}
        />
      </TableCell>
      <TableCell align="center">
        <Typography variant="caption" color="text.secondary">
          {metrics.renderTime.toFixed(2)} ms
        </Typography>
      </TableCell>
      <TableCell align="center">
        <Typography variant="caption" color="text.secondary">
          {metrics.renderCount}
        </Typography>
      </TableCell>
    </TableRow>
  );
}

