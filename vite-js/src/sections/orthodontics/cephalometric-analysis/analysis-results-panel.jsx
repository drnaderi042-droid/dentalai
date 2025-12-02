import { useState } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Tabs from '@mui/material/Tabs';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import { alpha } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import LinearProgress from '@mui/material/LinearProgress';

import { Iconify } from 'src/components/iconify';

// Normal values and ranges for cephalometric measurements
const NORMAL_RANGES = {
  SNA: { mean: 82, sd: 3.5, min: 78.5, max: 85.5 },
  SNB: { mean: 80, sd: 3.5, min: 76.5, max: 83.5 },
  ANB: { mean: 2, sd: 2, min: 0, max: 4 },
  gonialAngle: { mean: 120, sd: 5, min: 115, max: 125 },
  FMA: { mean: 25, sd: 4, min: 21, max: 29 },
  IMPA: { mean: 90, sd: 3, min: 87, max: 93 },
  interincisalAngle: { mean: 130, sd: 6, min: 124, max: 136 },
  overjet: { mean: 3, sd: 1, min: 2, max: 4 },
  overbite: { mean: 3, sd: 1, min: 2, max: 4 },
  combinationFactor: { mean: 157.9, sd: 6.5, min: 151.4, max: 164.4 },
};

const SKELETAL_MEASUREMENTS = [
  { key: 'SNA', label: 'SNA', unit: '°', description: 'زاویه SNA' },
  { key: 'SNB', label: 'SNB', unit: '°', description: 'زاویه SNB' },
  { key: 'ANB', label: 'ANB', unit: '°', description: 'زاویه ANB' },
  { key: 'gonialAngle', label: 'Gonial Angle', unit: '°', description: 'زاویه گونیال' },
  { key: 'FMA', label: 'FMA', unit: '°', description: 'زاویه فرانکفورت-مندیبولار' },
];

const DENTAL_MEASUREMENTS = [
  { key: 'overjet', label: 'Overjet', unit: 'mm', description: 'اورجت' },
  { key: 'overbite', label: 'Overbite', unit: 'mm', description: 'اوربایت' },
  { key: 'IMPA', label: 'IMPA', unit: '°', description: 'زاویه اینسایزور مندیبولار' },
  { key: 'interincisalAngle', label: 'Interincisal Angle', unit: '°', description: 'زاویه بین اینسایزورها' },
];

export function AnalysisResultsPanel({ measurements }) {
  const [tabValue, setTabValue] = useState(0);

  const getStatus = (value, normalRange) => {
    if (!value || !normalRange) return null;
    const numValue = parseFloat(value);
    if (numValue < normalRange.min) return 'low';
    if (numValue > normalRange.max) return 'high';
    return 'normal';
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'low': return 'error';
      case 'high': return 'error';
      case 'normal': return 'success';
      default: return 'grey';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'low': return 'کمتر از نرمال';
      case 'high': return 'بیشتر از نرمال';
      case 'normal': return 'نرمال';
      default: return 'نامشخص';
    }
  };

  const calculateCombinationFactor = () => {
    if (measurements.SNA && measurements.SNB) {
      const cf = parseFloat(measurements.SNA) + parseFloat(measurements.SNB);
      return cf.toFixed(2);
    }
    return null;
  };

  const combinationFactor = calculateCombinationFactor();
  const cfStatus = getStatus(combinationFactor, NORMAL_RANGES.combinationFactor);

  const renderMeasurementItem = (measurement) => {
    const value = measurements[measurement.key];
    const normalRange = NORMAL_RANGES[measurement.key];
    const status = getStatus(value, normalRange);
    const statusColor = getStatusColor(status);

    if (!value) return null;

    return (
      <Box key={measurement.key} sx={{ mb: 3 }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
          <Typography variant="subtitle2">{measurement.label}</Typography>
          <Stack direction="row" alignItems="center" spacing={1}>
            <Typography variant="h6" color={`${statusColor}.main`}>
              {value}{measurement.unit}
            </Typography>
            {status && (
              <Box
                sx={{
                  px: 1,
                  py: 0.5,
                  borderRadius: 0.75,
                  bgcolor: (theme) => alpha(theme.palette[statusColor].main, 0.16),
                  color: `${statusColor}.main`,
                  fontSize: '0.75rem',
                }}
              >
                {getStatusText(status)}
              </Box>
            )}
          </Stack>
        </Stack>
        
        {normalRange && (
          <>
            <Stack direction="row" justifyContent="space-between" sx={{ mb: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                میانگین: {normalRange.mean}{measurement.unit}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                SD: ±{normalRange.sd}
              </Typography>
            </Stack>
            
            <Box sx={{ position: 'relative' }}>
              <LinearProgress
                variant="determinate"
                value={100}
                sx={{
                  height: 8,
                  borderRadius: 1,
                  bgcolor: (theme) => alpha(theme.palette.grey[500], 0.12),
                  '& .MuiLinearProgress-bar': {
                    bgcolor: (theme) => alpha(theme.palette.success.main, 0.24),
                  }
                }}
              />
              {value && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: -2,
                    left: `${((parseFloat(value) - normalRange.min) / (normalRange.max - normalRange.min)) * 100}%`,
                    transform: 'translateX(-50%)',
                  }}
                >
                  <Box
                    sx={{
                      width: 12,
                      height: 12,
                      borderRadius: '50%',
                      bgcolor: `${statusColor}.main`,
                      border: 2,
                      borderColor: 'background.paper',
                    }}
                  />
                </Box>
              )}
            </Box>
            
            <Stack direction="row" justifyContent="space-between" sx={{ mt: 0.5 }}>
              <Typography variant="caption" color="text.secondary">
                {normalRange.min}{measurement.unit}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                {normalRange.max}{measurement.unit}
              </Typography>
            </Stack>
          </>
        )}
      </Box>
    );
  };

  const renderCombinationFactor = () => {
    if (!combinationFactor) return null;

    const normalRange = NORMAL_RANGES.combinationFactor;
    const statusColor = getStatusColor(cfStatus);

    return (
      <Card sx={{ p: 3, mb: 3, bgcolor: (theme) => alpha(theme.palette.primary.main, 0.08) }}>
        <Typography variant="h6" gutterBottom>
          Combination Factor
        </Typography>
        
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
          <Box>
            <Typography variant="h3" color={`${statusColor}.main`}>
              {combinationFactor}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              میانگین: {normalRange.mean} | SD: ±{normalRange.sd}
            </Typography>
          </Box>
          
          <Box
            sx={{
              px: 2,
              py: 1,
              borderRadius: 1,
              bgcolor: (theme) => alpha(theme.palette[statusColor].main, 0.16),
              color: `${statusColor}.main`,
            }}
          >
            <Typography variant="subtitle2">
              {getStatusText(cfStatus)}
            </Typography>
          </Box>
        </Stack>
        
        <Box sx={{ position: 'relative', mb: 1 }}>
          <LinearProgress
            variant="determinate"
            value={100}
            sx={{
              height: 12,
              borderRadius: 1,
              bgcolor: (theme) => alpha(theme.palette.grey[500], 0.12),
              '& .MuiLinearProgress-bar': {
                bgcolor: (theme) => alpha(theme.palette.success.main, 0.24),
              }
            }}
          />
          <Box
            sx={{
              position: 'absolute',
              top: -2,
              left: `${((parseFloat(combinationFactor) - normalRange.min) / (normalRange.max - normalRange.min)) * 100}%`,
              transform: 'translateX(-50%)',
            }}
          >
            <Box
              sx={{
                width: 16,
                height: 16,
                borderRadius: '50%',
                bgcolor: `${statusColor}.main`,
                border: 2,
                borderColor: 'background.paper',
              }}
            />
          </Box>
        </Box>
        
        <Stack direction="row" justifyContent="space-between">
          <Typography variant="caption" color="text.secondary">
            {normalRange.min}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {normalRange.mean}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {normalRange.max}
          </Typography>
        </Stack>
      </Card>
    );
  };

  return (
    <Box>
      <Card sx={{ mb: 3 }}>
        <Tabs
          value={tabValue}
          onChange={(e, newValue) => setTabValue(newValue)}
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          <Tab label="تحلیل خطی" />
          <Tab label="پروفیلوگرام" />
          <Tab label="نمودار" />
        </Tabs>

        <Box sx={{ p: 3 }}>
          {tabValue === 0 && (
            <Box>
              <Typography variant="h6" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <Iconify icon="solar:chart-2-bold" />
                نتایج تحلیل سفالومتری
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              {renderCombinationFactor()}
              
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3, mb: 2 }}>
                اندازه‌گیری‌های اسکلتی
              </Typography>
              {SKELETAL_MEASUREMENTS.map(renderMeasurementItem)}
              
              <Typography variant="subtitle1" gutterBottom sx={{ mt: 3, mb: 2 }}>
                اندازه‌گیری‌های دندانی
              </Typography>
              {DENTAL_MEASUREMENTS.map(renderMeasurementItem)}
              
              <Stack direction="row" spacing={2} sx={{ mt: 4 }}>
                <Button
                  variant="contained"
                  startIcon={<Iconify icon="solar:printer-bold" />}
                  fullWidth
                >
                  چاپ گزارش
                </Button>
                <Button
                  variant="outlined"
                  startIcon={<Iconify icon="solar:export-bold" />}
                  fullWidth
                >
                  خروجی Excel
                </Button>
              </Stack>
            </Box>
          )}

          {tabValue === 1 && (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Iconify icon="solar:graph-bold" width={64} sx={{ color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                پروفیلوگرام
              </Typography>
              <Typography variant="body2" color="text.disabled">
                نمایش گرافیکی نتایج تحلیل
              </Typography>
            </Box>
          )}

          {tabValue === 2 && (
            <Box sx={{ textAlign: 'center', py: 8 }}>
              <Iconify icon="solar:chart-bold" width={64} sx={{ color: 'text.disabled', mb: 2 }} />
              <Typography variant="h6" color="text.secondary">
                نمودار مقایسه‌ای
              </Typography>
              <Typography variant="body2" color="text.disabled">
                مقایسه اندازه‌گیری‌ها با مقادیر نرمال
              </Typography>
            </Box>
          )}
        </Box>
      </Card>
    </Box>
  );
}
