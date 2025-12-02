import { useMemo, useState, useEffect } from 'react';

import {
  Box,
  Card,
  Chip,
  Grid,
  Stack,
  Collapse,
  Typography,
  IconButton,
  CardContent,
  LinearProgress,
} from '@mui/material';

import { usePerformanceContext } from 'src/contexts/performance-context';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

/**
 * Total Performance Monitor - نمایش مصرف کل RAM و CPU صفحه
 */
export function TotalPerformanceMonitor({ position = 'top-right' }) {
  const { getAllComponents } = usePerformanceContext();
  const [expanded, setExpanded] = useState(true);
  const [totalMemory, setTotalMemory] = useState({ used: 0, total: 0, limit: 0 });
  const [totalCPU, setTotalCPU] = useState(0);

  const components = getAllComponents();

  // محاسبه مصرف کل RAM از کامپوننت‌ها
  const totalMemoryFromComponents = useMemo(() => components.reduce((sum, comp) => sum + (comp.memory?.usedMB || 0), 0), [components]);

  // محاسبه مصرف کل CPU از کامپوننت‌ها
  const totalCPUFromComponents = useMemo(() => components.reduce((sum, comp) => sum + (comp.cpu?.usagePercent || 0), 0), [components]);

  // ردیابی مصرف کل RAM از performance.memory
  useEffect(() => {
    const updateTotalMemory = () => {
      if (typeof performance !== 'undefined' && performance.memory) {
        const memoryInfo = performance.memory;
        setTotalMemory({
          used: memoryInfo.usedJSHeapSize / 1024 / 1024, // تبدیل به MB
          total: memoryInfo.totalJSHeapSize / 1024 / 1024,
          limit: memoryInfo.jsHeapSizeLimit / 1024 / 1024,
        });
      }
    };

    updateTotalMemory();
    const interval = setInterval(updateTotalMemory, 1000);

    return () => clearInterval(interval);
  }, []);

  // ردیابی مصرف کل CPU (تخمینی)
  // نکته: در مرورگر نمی‌توانیم مصرف دقیق CPU را اندازه بگیریم
  // این یک تخمین بر اساس FPS است
  // مهم: این درصد برای کل CPU سیستم است، نه یک هسته
  useEffect(() => {
    let lastTime = performance.now();
    let frameCount = 0;
    let fps = 60;

    const measureFPS = () => {
      const now = performance.now();
      const delta = now - lastTime;
      frameCount++;

      if (delta >= 1000) {
        fps = frameCount;
        frameCount = 0;
        lastTime = now;

        // تخمین CPU usage بر اساس FPS
        // اگر FPS کمتر از 60 باشد، CPU بیشتری مصرف می‌شود
        // فرمول ساده: اگر FPS = 60 باشد، CPU = 0%
        // اگر FPS = 30 باشد، CPU = 50%
        // اگر FPS = 0 باشد، CPU = 100%
        // اما در عمل، FPS معمولاً بین 50-60 است
        const cpuEstimate = Math.max(0, Math.min(100, ((60 - fps) / 60) * 100));
        
        setTotalCPU(cpuEstimate);
      }

      requestAnimationFrame(measureFPS);
    };

    const frameId = requestAnimationFrame(measureFPS);

    return () => {
      cancelAnimationFrame(frameId);
    };
  }, []);

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

  const memoryPercentage = totalMemory.limit > 0 
    ? (totalMemory.used / totalMemory.limit) * 100 
    : 0;

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
            <Iconify icon="mdi:chart-box" width={20} />
            <Typography variant="subtitle2">مصرف کل صفحه</Typography>
          </Stack>
          <IconButton size="small" onClick={(e) => { e.stopPropagation(); setExpanded(!expanded); }}>
            <Iconify icon={expanded ? 'eva:arrow-up-fill' : 'eva:arrow-down-fill'} />
          </IconButton>
        </Box>

        <Collapse in={expanded}>
          <CardContent sx={{ pt: 2 }}>
            {/* Total Memory */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  RAM کل
                </Typography>
                <Chip
                  label={`${memoryPercentage.toFixed(1)}%`}
                  size="small"
                  color={getMemoryColor(memoryPercentage)}
                  sx={{ height: 20 }}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, memoryPercentage)}
                color={getMemoryColor(memoryPercentage)}
                sx={{ height: 8, borderRadius: 1, mb: 1 }}
              />
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    استفاده شده
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {totalMemory.used.toFixed(2)} MB
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    محدودیت
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {totalMemory.limit.toFixed(2)} MB
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary">
                    مجموع از کامپوننت‌ها
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {totalMemoryFromComponents.toFixed(2)} MB
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Total CPU */}
            <Box sx={{ mb: 3 }}>
              <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 1 }}>
                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                  CPU کل
                </Typography>
                <Chip
                  label={`${totalCPU.toFixed(1)}%`}
                  size="small"
                  color={getCPUColor(totalCPU)}
                  sx={{ height: 20 }}
                />
              </Stack>
              <LinearProgress
                variant="determinate"
                value={Math.min(100, totalCPU)}
                color={getCPUColor(totalCPU)}
                sx={{ height: 8, borderRadius: 1, mb: 1 }}
              />
              <Grid container spacing={1}>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    مصرف تخمینی کل
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {totalCPU.toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={6}>
                  <Typography variant="caption" color="text.secondary">
                    مجموع از کامپوننت‌ها
                  </Typography>
                  <Typography variant="body2" sx={{ fontWeight: 500 }}>
                    {totalCPUFromComponents.toFixed(2)}%
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                    ⚠️ مصرف CPU شامل اسکریپت‌های دیگر (extensions, analytics) هم می‌شود
                  </Typography>
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem' }}>
                    📊 این درصد برای کل CPU سیستم است، نه یک هسته
                  </Typography>
                </Grid>
              </Grid>
            </Box>

            {/* Component Count */}
            <Box sx={{ mb: 2 }}>
              <Typography variant="caption" color="text.secondary">
                تعداد کامپوننت‌های ردیابی شده
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 500 }}>
                {components.length}
              </Typography>
            </Box>

            {/* Info Box */}
            <Box sx={{ p: 1.5, bgcolor: 'background.neutral', borderRadius: 1 }}>
              <Typography variant="caption" sx={{ fontWeight: 600, display: 'block', mb: 0.5 }}>
                📌 نکات مهم:
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', display: 'block' }}>
                • مصرف CPU کل شامل تمام اسکریپت‌های صفحه است
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', display: 'block' }}>
                • درصد CPU برای کل سیستم است (نه یک هسته)
              </Typography>
              <Typography variant="caption" color="text.secondary" sx={{ fontSize: '0.7rem', display: 'block' }}>
                • محاسبات تخمینی هستند و دقیق نیستند
              </Typography>
            </Box>
          </CardContent>
        </Collapse>
      </Card>
    </Box>
  );
}

