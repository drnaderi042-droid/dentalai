import { useState } from 'react';

import { Box, Stack, Button } from '@mui/material';

import { PerformanceTreeView } from './performance-tree-view';
import { PerformanceDashboard } from './performance-dashboard';
import { PerformanceDetailsPanel } from './performance-details-panel';
import { TotalPerformanceMonitor } from './total-performance-monitor';

// ----------------------------------------------------------------------

/**
 * Advanced Performance Monitor - نمایش کامل عملکرد تمام کامپوننت‌ها
 * شامل Tree View و Details Panel
 */
export function AdvancedPerformanceMonitor({ 
  components = [],
  showTreeView = true,
  showDetailsPanel = true,
  showDashboard = false,
  showTotalMonitor = true,
  treeViewPosition = 'bottom-left',
  detailsPanelPosition = 'bottom-right',
  dashboardPosition = 'top-right',
  totalMonitorPosition = 'top-right',
}) {
  const [selectedComponent, setSelectedComponent] = useState(null);
  const [viewMode, setViewMode] = useState('tree'); // 'tree', 'dashboard', 'both'

  const handleComponentSelect = (componentName) => {
    setSelectedComponent(componentName);
  };

  return (
    <>
      {/* Total Performance Monitor */}
      {showTotalMonitor && (
        <TotalPerformanceMonitor position={totalMonitorPosition} />
      )}

      {/* Tree View */}
      {showTreeView && (viewMode === 'tree' || viewMode === 'both') && (
        <PerformanceTreeView
          position={treeViewPosition}
          onComponentSelect={handleComponentSelect}
        />
      )}

      {/* Details Panel */}
      {showDetailsPanel && (
        <PerformanceDetailsPanel
          componentName={selectedComponent}
          position={detailsPanelPosition}
        />
      )}

      {/* Dashboard */}
      {showDashboard && (viewMode === 'dashboard' || viewMode === 'both') && components.length > 0 && (
        <PerformanceDashboard
          components={components}
          position={dashboardPosition}
        />
      )}

      {/* Mode Switcher */}
      {import.meta.env.DEV && (
        <Box
          sx={{
            position: 'fixed',
            bottom: 16,
            left: '50%',
            transform: 'translateX(-50%)',
            zIndex: 10000,
            bgcolor: 'background.paper',
            borderRadius: 2,
            boxShadow: 6,
            p: 1,
          }}
        >
          <Stack direction="row" spacing={1}>
            <Button
              size="small"
              variant={viewMode === 'tree' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('tree')}
            >
              Tree
            </Button>
            {components.length > 0 && (
              <Button
                size="small"
                variant={viewMode === 'dashboard' ? 'contained' : 'outlined'}
                onClick={() => setViewMode('dashboard')}
              >
                Dashboard
              </Button>
            )}
            <Button
              size="small"
              variant={viewMode === 'both' ? 'contained' : 'outlined'}
              onClick={() => setViewMode('both')}
            >
              هر دو
            </Button>
          </Stack>
        </Box>
      )}
    </>
  );
}

