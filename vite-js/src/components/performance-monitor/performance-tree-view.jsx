import { useMemo, useState } from 'react';

import {
  Box,
  Card,
  Chip,
  List,
  Stack,
  Button,
  Tooltip,
  Divider,
  ListItem,
  Typography,
  CardContent,
  ListItemText,
  ListItemButton,
  ListItemSecondaryAction,
} from '@mui/material';

import { usePerformanceContext } from 'src/contexts/performance-context';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

/**
 * Performance Tree View - نمایش سلسله مراتب کامپوننت‌ها با مصرف CPU و RAM
 */
export function PerformanceTreeView({ position = 'bottom-left', onComponentSelect }) {
  const { getAllComponents, getProfilerData } = usePerformanceContext();
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [sortBy, setSortBy] = useState('name'); // 'name', 'memory', 'cpu', 'renderTime'

  const components = getAllComponents();

  // مرتب‌سازی کامپوننت‌ها
  const sortedComponents = useMemo(() => {
    const sorted = [...components];
    switch (sortBy) {
      case 'memory':
        return sorted.sort((a, b) => (b.memory?.usedMB || 0) - (a.memory?.usedMB || 0));
      case 'cpu':
        return sorted.sort((a, b) => (b.cpu?.usagePercent || 0) - (a.cpu?.usagePercent || 0));
      case 'renderTime':
        return sorted.sort((a, b) => b.renderTime - a.renderTime);
      default:
        return sorted.sort((a, b) => a.componentName.localeCompare(b.componentName));
    }
  }, [components, sortBy]);

  const handleSelect = (componentName) => {
    setSelectedComponent(componentName);
    if (onComponentSelect) {
      onComponentSelect(componentName);
    }
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

  const positionStyles = {
    'top-left': { top: 16, left: 16 },
    'top-right': { top: 16, right: 16 },
    'bottom-left': { bottom: 16, left: 16 },
    'bottom-right': { bottom: 16, right: 16 },
  };

  // ساختار لیست کامپوننت‌ها
  const buildComponentList = (comps) => comps.map((comp) => {
      const profilerData = getProfilerData(comp.componentName);
      const isSelected = selectedComponent === comp.componentName;

      return (
        <ListItem
          key={comp.componentName}
          disablePadding
          sx={{
            bgcolor: isSelected ? 'action.selected' : 'transparent',
            '&:hover': {
              bgcolor: 'action.hover',
            },
          }}
        >
          <ListItemButton onClick={() => handleSelect(comp.componentName)}>
            <ListItemText
              primary={
                <Typography variant="body2" sx={{ fontWeight: isSelected ? 600 : 400 }}>
                  {comp.componentName}
                </Typography>
              }
            />
            <ListItemSecondaryAction>
              <Stack direction="row" spacing={0.5} alignItems="center">
                <Tooltip title={`RAM: ${comp.memory?.usedMB?.toFixed(2) || 0} MB`}>
                  <Chip
                    label={`${comp.memory?.usedMB?.toFixed(1) || 0} MB`}
                    size="small"
                    color={getMemoryColor(comp.memory?.usedMB || 0)}
                    sx={{ height: 18, fontSize: '0.65rem', minWidth: 50 }}
                  />
                </Tooltip>
                <Tooltip title={`CPU: ${comp.cpu?.usagePercent?.toFixed(1) || 0}%`}>
                  <Chip
                    label={`${comp.cpu?.usagePercent?.toFixed(1) || 0}%`}
                    size="small"
                    color={getCPUColor(comp.cpu?.usagePercent || 0)}
                    sx={{ height: 18, fontSize: '0.65rem', minWidth: 45 }}
                  />
                </Tooltip>
                {profilerData && (
                  <Tooltip title={`Render: ${profilerData.actualDuration.toFixed(2)}ms`}>
                    <Chip
                      label={`${profilerData.actualDuration.toFixed(0)}ms`}
                      size="small"
                      variant="outlined"
                      sx={{ height: 18, fontSize: '0.65rem', minWidth: 40 }}
                    />
                  </Tooltip>
                )}
              </Stack>
            </ListItemSecondaryAction>
          </ListItemButton>
        </ListItem>
      );
    });

  return (
    <Box
      sx={{
        position: 'fixed',
        zIndex: 9998,
        ...positionStyles[position],
        maxWidth: 500,
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
          }}
        >
          <Stack direction="row" spacing={1} alignItems="center">
            <Iconify icon="mdi:file-tree" width={20} />
            <Typography variant="subtitle2">Component Tree</Typography>
            <Chip label={`${components.length} کامپوننت`} size="small" color="primary" sx={{ height: 20, fontSize: '0.65rem' }} />
          </Stack>
        </Box>

        <Divider />

        <Box sx={{ p: 1, display: 'flex', gap: 1, flexWrap: 'wrap' }}>
          <Button
            size="small"
            variant={sortBy === 'name' ? 'contained' : 'outlined'}
            onClick={() => setSortBy('name')}
          >
            نام
          </Button>
          <Button
            size="small"
            variant={sortBy === 'memory' ? 'contained' : 'outlined'}
            onClick={() => setSortBy('memory')}
          >
            RAM
          </Button>
          <Button
            size="small"
            variant={sortBy === 'cpu' ? 'contained' : 'outlined'}
            onClick={() => setSortBy('cpu')}
          >
            CPU
          </Button>
          <Button
            size="small"
            variant={sortBy === 'renderTime' ? 'contained' : 'outlined'}
            onClick={() => setSortBy('renderTime')}
          >
            زمان رندر
          </Button>
        </Box>

        <CardContent sx={{ flex: 1, overflow: 'auto', p: 0 }}>
          {sortedComponents.length === 0 ? (
            <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center', py: 4, px: 2 }}>
              هیچ کامپوننتی ردیابی نشده است
              <br />
              <Typography variant="caption" component="span">
                از TrackedComponent برای wrap کردن کامپوننت‌ها استفاده کنید
              </Typography>
            </Typography>
          ) : (
            <List dense sx={{ py: 0 }}>
              {buildComponentList(sortedComponents)}
            </List>
          )}
        </CardContent>
      </Card>
    </Box>
  );
}

