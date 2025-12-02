import React, { memo, useRef, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Switch from '@mui/material/Switch';
import Checkbox from '@mui/material/Checkbox';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import TextField from '@mui/material/TextField';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import { alpha, useTheme } from '@mui/material/styles';
import TableContainer from '@mui/material/TableContainer';
import FormControlLabel from '@mui/material/FormControlLabel';

import { TablePaginationCustom } from 'src/components/table';

// ----------------------------------------------------------------------

// Tooth numbering (Quadrant-based: 1-8 per quadrant) - 32 teeth total
// Universal numbers are kept for data storage, but display uses quadrant-based numbering (1-8)
// Number 1 = Central Incisor, Number 8 = Third Molar
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

// Site labels for 6-point probing
const FACIAL_SITES = ['MB', 'B', 'DB'];
const LINGUAL_SITES = ['ML', 'L', 'DL'];

// ----------------------------------------------------------------------

// Memoized TableRow component for better performance
const PeriodontalChartRow = memo(({
  tooth,
  toothData,
  facialPD,
  lingualPD,
  facialGM,
  lingualGM,
  mesialBOP,
  distalBOP,
  showGingivalMargin,
  showMobility,
  theme,
  renderToothIcon,
  handleInputChange,
  handleUpdateBOP,
  handleUpdateField,
  hasFurcation,
  paginatedTeeth,
  handleToggleMissing,
}) => {
  // Local state for input values to reduce re-renders
  const [localPDValues, setLocalPDValues] = useState({
    facial: [...facialPD],
    lingual: [...lingualPD],
  });
  const [localGMValues, setLocalGMValues] = useState({
    facial: [...facialGM],
    lingual: [...lingualGM],
  });

  // Sync local state with props when they change externally
  useEffect(() => {
    setLocalPDValues({
      facial: [...facialPD],
      lingual: [...lingualPD],
    });
  }, [facialPD, lingualPD]);

  useEffect(() => {
    setLocalGMValues({
      facial: [...facialGM],
      lingual: [...lingualGM],
    });
  }, [facialGM, lingualGM]);

  const handlePDChange = useCallback((surface, index, value) => {
    // Update local state immediately for responsive UI
    if (surface === 'facial') {
      setLocalPDValues(prev => ({
        ...prev,
        facial: prev.facial.map((v, i) => i === index ? value : v),
      }));
    } else {
      setLocalPDValues(prev => ({
        ...prev,
        lingual: prev.lingual.map((v, i) => i === index ? value : v),
      }));
    }
    // Debounced update to parent
    handleInputChange(tooth.universalNumber, surface, index, value, paginatedTeeth);
  }, [tooth.universalNumber, handleInputChange, paginatedTeeth]);

  const handleGMChange = useCallback((surface, index, value) => {
    // Update local state immediately for responsive UI
    if (surface === 'facial') {
      setLocalGMValues(prev => ({
        ...prev,
        facial: prev.facial.map((v, i) => i === index ? value : v),
      }));
    } else {
      setLocalGMValues(prev => ({
        ...prev,
        lingual: prev.lingual.map((v, i) => i === index ? value : v),
      }));
    }
    // Debounced update to parent
    handleUpdateField(tooth.universalNumber, surface, 'gingivalMargin', index, value, false);
  }, [tooth.universalNumber, handleUpdateField]);

  return (
    <TableRow
      key={tooth.universalNumber}
      sx={{
        ...(toothData.missing && {
          opacity: 0.6,
        }),
      }}
    >
      {/* Tooth Icon & Number - First */}
      <TableCell
        align="center"
        sx={{
          width: { xs: 45, md: 55 },
          minWidth: { xs: 45, md: 55 },
          maxWidth: { xs: 45, md: 55 },
          position: 'sticky',
          left: 0,
          zIndex: 1,
          bgcolor: 'var(--palette-background-neutral)',
          borderRight: `2px solid ${theme.palette.divider}`,
          padding: { xs: '4px 2px', md: '4px 4px' },
        }}
      >
        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: { xs: 0.2, md: 0.3 } }}>
          {renderToothIcon(tooth.universalNumber, tooth.quadrantNumber, tooth.jaw)}
          <Typography variant="caption" fontWeight="bold" sx={{ fontSize: { xs: '0.7rem', md: '0.75rem' } }}>
            {tooth.quadrantNumber}
          </Typography>
        </Box>
      </TableCell>

      {/* Combined Probing Sites: Facial above, Lingual below */}
      {FACIAL_SITES.map((pos, idx) => (
        <TableCell key={`site-${idx}`} align="center" sx={{ px: 0.05, py: 0.25, width: 32, maxWidth: 32 }}>
          <Stack direction="column" spacing={0.1} alignItems="center">
            {/* Facial site */}
            <TextField
              id={`facial-${tooth.universalNumber}-${idx}`}
              size="small"
              type="number"
              value={localPDValues.facial[idx] ?? ''}
              onChange={(e) => {
                const value = e.target.value === '' ? null : parseInt(e.target.value, 10) || null;
                handlePDChange('facial', idx, value);
              }}
              inputProps={{ 
                min: 0, 
                max: 9, 
                style: { textAlign: 'center', padding: '0px 1px', fontSize: '0.875rem' },
                maxLength: 1
              }}
              sx={{ 
                width: 24,
                '& .MuiOutlinedInput-root': { 
                  padding: '0px 1px',
                  height: '28px',
                  '& input': {
                    padding: '4px 2px',
                  }
                } 
              }}
              disabled={toothData.missing}
            />
            {/* Lingual site */}
            <TextField
              id={`lingual-${tooth.universalNumber}-${idx}`}
              size="small"
              type="number"
              value={localPDValues.lingual[idx] ?? ''}
              onChange={(e) => {
                const value = e.target.value === '' ? null : parseInt(e.target.value, 10) || null;
                handlePDChange('lingual', idx, value);
              }}
              inputProps={{ 
                min: 0, 
                max: 9, 
                style: { textAlign: 'center', padding: '0px 1px', fontSize: '0.875rem' },
                maxLength: 1
              }}
              sx={{ 
                width: 24,
                '& .MuiOutlinedInput-root': { 
                  padding: '0px 1px',
                  height: '28px',
                  '& input': {
                    padding: '4px 2px',
                  }
                } 
              }}
              disabled={toothData.missing}
            />
          </Stack>
        </TableCell>
      ))}

      {/* BOP */}
      <TableCell align="center" sx={{ px: { xs: 0.25, md: 0.5 }, width: { xs: 45, md: 50 }, minWidth: { xs: 45, md: 50 }, maxWidth: { xs: 45, md: 50 } }}>
        <Stack direction="column" spacing={0.2} alignItems="center">
          <FormControlLabel
            control={
              <Checkbox
                size="small"
                checked={mesialBOP}
                onChange={(e) => {
                  handleUpdateBOP(tooth.universalNumber, 'mesial', e.target.checked);
                }}
                disabled={toothData.missing}
                sx={{ padding: { xs: '2px', md: '4px' } }}
              />
            }
            label={<Typography variant="caption" sx={{ fontSize: { xs: '0.65rem', md: '0.7rem' }, ml: 0.5 }}>M</Typography>}
            sx={{ margin: 0 }}
          />
          <FormControlLabel
            control={
              <Checkbox
                size="small"
                checked={distalBOP}
                onChange={(e) => {
                  handleUpdateBOP(tooth.universalNumber, 'distal', e.target.checked);
                }}
                disabled={toothData.missing}
                sx={{ padding: { xs: '2px', md: '4px' } }}
              />
            }
            label={<Typography variant="caption" sx={{ fontSize: { xs: '0.65rem', md: '0.7rem' }, ml: 0.5 }}>D</Typography>}
            sx={{ margin: 0 }}
          />
        </Stack>
      </TableCell>

      {/* Gingival Margin - Conditional */}
      {showGingivalMargin && FACIAL_SITES.map((pos, idx) => (
        <TableCell key={`gm-${idx}`} align="center" sx={{ px: 0.05, py: 0.25, width: 32, maxWidth: 32 }}>
          <Stack direction="column" spacing={0.1} alignItems="center">
            {/* Facial site */}
            <TextField
              id={`facial-gm-${tooth.universalNumber}-${idx}`}
              size="small"
              type="number"
              value={localGMValues.facial[idx] ?? ''}
              onChange={(e) => {
                const inputValue = e.target.value;
                const value = inputValue === '' ? null : parseInt(inputValue, 10) || null;
                handleGMChange('facial', idx, value);
                // Auto-focus to next field if single digit entered (including negative)
                if (value !== null && value >= -9 && value <= 9 && inputValue.length <= 2) {
                  setTimeout(() => {
                    let nextId = null;
                    if (idx < 2) {
                      nextId = `facial-gm-${tooth.universalNumber}-${idx + 1}`;
                    } else {
                      nextId = `lingual-gm-${tooth.universalNumber}-0`;
                    }
                    if (nextId) {
                      const nextInput = document.getElementById(nextId);
                      if (nextInput) {
                        nextInput.focus();
                        nextInput.select();
                      }
                    }
                  }, 10);
                }
              }}
              inputProps={{ 
                min: -10, 
                max: 10, 
                style: { textAlign: 'center', padding: '0px 1px', fontSize: '0.875rem' },
                maxLength: 2
              }}
              sx={{ 
                width: 24,
                '& .MuiOutlinedInput-root': { 
                  padding: '0px 1px',
                  height: '28px',
                  '& input': {
                    padding: '4px 2px',
                  }
                } 
              }}
              disabled={toothData.missing}
            />
            {/* Lingual site */}
            <TextField
              id={`lingual-gm-${tooth.universalNumber}-${idx}`}
              size="small"
              type="number"
              value={localGMValues.lingual[idx] ?? ''}
              onChange={(e) => {
                const inputValue = e.target.value;
                const value = inputValue === '' ? null : parseInt(inputValue, 10) || null;
                handleGMChange('lingual', idx, value);
                // Auto-focus to next field if single digit entered (including negative)
                if (value !== null && value >= -9 && value <= 9 && inputValue.length <= 2) {
                  setTimeout(() => {
                    let nextId = null;
                    if (idx < 2) {
                      nextId = `lingual-gm-${tooth.universalNumber}-${idx + 1}`;
                    } else {
                      const currentIndex = paginatedTeeth.findIndex(t => t.universalNumber === tooth.universalNumber);
                      if (currentIndex < paginatedTeeth.length - 1) {
                        const nextTooth = paginatedTeeth[currentIndex + 1];
                        nextId = `facial-${nextTooth.universalNumber}-0`;
                      }
                    }
                    if (nextId) {
                      const nextInput = document.getElementById(nextId);
                      if (nextInput) {
                        nextInput.focus();
                        nextInput.select();
                      }
                    }
                  }, 10);
                }
              }}
              inputProps={{ 
                min: -10, 
                max: 10, 
                style: { textAlign: 'center', padding: '0px 1px', fontSize: '0.875rem' },
                maxLength: 2
              }}
              sx={{ 
                width: 24,
                '& .MuiOutlinedInput-root': { 
                  padding: '0px 1px',
                  height: '28px',
                  '& input': {
                    padding: '4px 2px',
                  }
                } 
              }}
              disabled={toothData.missing}
            />
          </Stack>
        </TableCell>
      ))}

      {/* Plaque */}
      <TableCell align="center" sx={{ px: { xs: 0.25, md: 0.5 }, width: { xs: 35, md: 40 }, minWidth: { xs: 35, md: 40 }, maxWidth: { xs: 35, md: 40 } }}>
        <Checkbox
          size="small"
          checked={toothData.facial?.plaque || false}
          onChange={(e) => handleUpdateField(tooth.universalNumber, 'facial', 'plaque', null, e.target.checked, true)}
          disabled={toothData.missing}
          sx={{ p: 0, padding: { xs: '2px', md: '4px' } }}
        />
      </TableCell>

      {/* Furcation */}
      <TableCell align="center" sx={{ px: { xs: 0.25, md: 0.5 }, width: { xs: 50, md: 55 }, minWidth: { xs: 50, md: 55 }, maxWidth: { xs: 50, md: 55 } }}>
        {hasFurcation(tooth.universalNumber) ? (
          <IconButton
            size="small"
            onClick={() => {
              const current = toothData.facial?.furcation || null;
              let next = null;
              if (!current) next = 'I';
              else if (current === 'I') next = 'II';
              else if (current === 'II') next = 'III';
              else next = null;
              handleUpdateField(tooth.universalNumber, 'facial', 'furcation', null, next, true);
            }}
            disabled={toothData.missing}
            sx={{
              width: { xs: 32, md: 36 },
              height: { xs: 32, md: 36 },
              border: `2px solid ${toothData.facial?.furcation ? theme.palette.warning.main : theme.palette.divider}`,
              borderRadius: '50%',
              bgcolor: toothData.facial?.furcation ? alpha(theme.palette.warning.main, 0.1) : 'inherit',
              padding: 0,
            }}
          >
            <Typography variant="caption" fontWeight="bold" sx={{ fontSize: { xs: '0.7rem', md: '0.75rem' } }}>
              {toothData.facial?.furcation || '-'}
            </Typography>
          </IconButton>
        ) : (
          <Typography variant="caption" color="text.disabled" sx={{ fontSize: { xs: '0.7rem', md: '0.75rem' } }}>
            -
          </Typography>
        )}
      </TableCell>

      {/* Mobility - Conditional */}
      {showMobility && (
        <TableCell align="center" sx={{ px: { xs: 0.25, md: 0.5 }, width: { xs: 50, md: 55 }, minWidth: { xs: 50, md: 55 }, maxWidth: { xs: 50, md: 55 } }}>
          <IconButton
            size="small"
            onClick={() => {
              const current = toothData.facial?.mobility || 0;
              const next = (current + 1) % 4;
              handleUpdateField(tooth.universalNumber, 'facial', 'mobility', null, next, true);
            }}
            disabled={toothData.missing}
            sx={{
              width: { xs: 32, md: 36 },
              height: { xs: 32, md: 36 },
              border: `2px solid ${toothData.facial?.mobility > 0 ? theme.palette.error.main : theme.palette.divider}`,
              borderRadius: '50%',
              bgcolor: toothData.facial?.mobility > 0 ? alpha(theme.palette.error.main, 0.1) : 'inherit',
              padding: 0,
            }}
          >
            <Typography variant="caption" fontWeight="bold" sx={{ fontSize: { xs: '0.7rem', md: '0.75rem' } }}>
              M{toothData.facial?.mobility || 0}
            </Typography>
          </IconButton>
        </TableCell>
      )}
    </TableRow>
  );
}, (prevProps, nextProps) => 
  // Custom comparison function for memo
   (
    prevProps.tooth.universalNumber === nextProps.tooth.universalNumber &&
    JSON.stringify(prevProps.facialPD) === JSON.stringify(nextProps.facialPD) &&
    JSON.stringify(prevProps.lingualPD) === JSON.stringify(nextProps.lingualPD) &&
    JSON.stringify(prevProps.facialGM) === JSON.stringify(nextProps.facialGM) &&
    JSON.stringify(prevProps.lingualGM) === JSON.stringify(nextProps.lingualGM) &&
    prevProps.mesialBOP === nextProps.mesialBOP &&
    prevProps.distalBOP === nextProps.distalBOP &&
    prevProps.toothData.missing === nextProps.toothData.missing &&
    prevProps.toothData.implant === nextProps.toothData.implant &&
    prevProps.toothData.facial?.plaque === nextProps.toothData.facial?.plaque &&
    prevProps.toothData.facial?.furcation === nextProps.toothData.facial?.furcation &&
    prevProps.toothData.facial?.mobility === nextProps.toothData.facial?.mobility &&
    prevProps.showGingivalMargin === nextProps.showGingivalMargin &&
    prevProps.showMobility === nextProps.showMobility
  )
);

PeriodontalChartRow.displayName = 'PeriodontalChartRow';

// ----------------------------------------------------------------------

export function PeriodontalChart({ chartData, onUpdateTooth, onUpdateToothFields, selectedQuadrant, onQuadrantChange }) {
  const theme = useTheme();

  // Show/hide gingival margin column
  const [showGingivalMargin, setShowGingivalMargin] = useState(false);
  // Show/hide mobility column
  const [showMobility, setShowMobility] = useState(false);

  // Detect mobile device and adjust rowsPerPage accordingly
  const [page, setPage] = useState(0);
  const defaultRowsPerPage = useMemo(() => {
    // Check if mobile device
    if (typeof window !== 'undefined') {
      const isMobile = window.innerWidth < 768;
      return isMobile ? 4 : 8;
    }
    return 8;
  }, []);
  const [rowsPerPage, setRowsPerPage] = useState(defaultRowsPerPage);

  const handleToggleMissing = useCallback((toothNumber) => {
    const toothData = chartData.teeth[toothNumber] || { missing: false, implant: false };
    const isMissing = toothData.missing || false;
    const isImplant = toothData.implant || false;
    
    // Cycle through: normal -> missing -> implant -> normal
    if (onUpdateToothFields) {
      // Use the helper function to update multiple fields at once
      if (!isMissing && !isImplant) {
        // Normal -> Missing
        onUpdateToothFields(toothNumber, { missing: true, implant: false });
      } else if (isMissing && !isImplant) {
        // Missing -> Implant
        onUpdateToothFields(toothNumber, { missing: false, implant: true });
      } else {
        // Implant -> Normal (or any other state -> Normal)
        onUpdateToothFields(toothNumber, { missing: false, implant: false });
      }
    } else if (!isMissing && !isImplant) {
      // Fallback to individual updates if helper is not available
      // Normal -> Missing
      onUpdateTooth(toothNumber, 'missing', true);
      onUpdateTooth(toothNumber, 'implant', false);
    } else if (isMissing && !isImplant) {
      // Missing -> Implant
      onUpdateTooth(toothNumber, 'missing', false);
      onUpdateTooth(toothNumber, 'implant', true);
    } else {
      // Implant -> Normal (or any other state -> Normal)
      onUpdateTooth(toothNumber, 'missing', false);
      onUpdateTooth(toothNumber, 'implant', false);
    }
  }, [onUpdateTooth, onUpdateToothFields, chartData]);

  const handleToggleImplant = useCallback((toothNumber) => {
    const toothData = chartData.teeth[toothNumber] || { implant: false };
    onUpdateTooth(toothNumber, 'implant', !toothData.implant);
  }, [chartData, onUpdateTooth]);

  // Debounce timer ref for input fields
  const debounceTimerRef = useRef({});

  // Cleanup debounce timers on unmount
  useEffect(() => () => {
      Object.values(debounceTimerRef.current).forEach(timer => {
        if (timer) clearTimeout(timer);
      });
      debounceTimerRef.current = {};
    }, []);

  // Debounced update function with shorter delay for better UX
  const debouncedUpdate = useCallback((toothNumber, surface, field, index, value, delay = 100) => {
    const key = `${toothNumber}-${surface}-${field}-${index}`;
    
    // Clear existing timer
    if (debounceTimerRef.current[key]) {
      clearTimeout(debounceTimerRef.current[key]);
    }

    // Set new timer
    debounceTimerRef.current[key] = setTimeout(() => {
    const toothData = chartData.teeth[toothNumber] || { [surface]: { [field]: Array(3).fill(0) } };
    const currentSurface = toothData[surface] || { [field]: Array(3).fill(0) };
    const newSurfaceData = { ...currentSurface };

    if (Array.isArray(newSurfaceData[field])) {
      if (index !== null && index !== undefined) {
        const newArray = [...newSurfaceData[field]];
        newArray[index] = value;
        newSurfaceData[field] = newArray;
      } else {
        newSurfaceData[field] = value;
      }
    } else {
      newSurfaceData[field] = value;
    }

    onUpdateTooth(toothNumber, surface, newSurfaceData);
      delete debounceTimerRef.current[key];
    }, delay);
  }, [chartData, onUpdateTooth]);

  const handleUpdateField = useCallback((toothNumber, surface, field, index, value, immediate = false) => {
    if (immediate) {
      // For non-input fields (checkboxes, buttons), update immediately
      const toothData = chartData.teeth[toothNumber] || { [surface]: { [field]: Array(3).fill(0) } };
      const currentSurface = toothData[surface] || { [field]: Array(3).fill(0) };
      const newSurfaceData = { ...currentSurface };

      if (Array.isArray(newSurfaceData[field])) {
        if (index !== null && index !== undefined) {
          const newArray = [...newSurfaceData[field]];
          newArray[index] = value;
          newSurfaceData[field] = newArray;
        } else {
          newSurfaceData[field] = value;
        }
      } else {
        newSurfaceData[field] = value;
      }

      onUpdateTooth(toothNumber, surface, newSurfaceData);
    } else {
      // For input fields, use debounce
      debouncedUpdate(toothNumber, surface, field, index, value);
    }
  }, [chartData, onUpdateTooth, debouncedUpdate]);

  const handleUpdateBOP = useCallback((toothNumber, site, checked) => {
    // site is either 'mesial' (index 0) or 'distal' (index 2)
    // M (mesial) = only facial side (right side when viewing)
    // D (distal) = only facial side (left side when viewing)
    const index = site === 'mesial' ? 0 : 2;
    const toothData = chartData.teeth[toothNumber] || {};
    
    const facialSurface = toothData.facial || { bleeding: [false, false, false] };
    
    const newFacialBleeding = [...(facialSurface.bleeding || [false, false, false])];
    
    // Only update facial side for BOP (M = right side, D = left side)
    newFacialBleeding[index] = checked;
    
    const newFacialSurface = { ...facialSurface, bleeding: newFacialBleeding };
    
    onUpdateTooth(toothNumber, 'facial', newFacialSurface);
  }, [chartData, onUpdateTooth]);

  const handleChangePage = useCallback((event, newPage) => {
    setPage(newPage);
  }, []);

  const handleChangeRowsPerPage = useCallback((event) => {
    setPage(0);
    setRowsPerPage(parseInt(event.target.value, 10));
  }, []);

  const handleQuadrantChangeInternal = useCallback((event) => {
    if (onQuadrantChange) {
      onQuadrantChange(event);
    }
    setPage(0); // Reset to first page when quadrant changes
  }, [onQuadrantChange]);

  // Filter teeth by selected quadrant (use prop if provided, otherwise default to 'Upper Right')
  const currentQuadrant = selectedQuadrant || 'Upper Right';
  const TEETH = useMemo(() => 
    ALL_TEETH.filter((tooth) => tooth.quadrant === currentQuadrant),
    [currentQuadrant]
  );

  // Paginate teeth
  const paginatedTeeth = useMemo(() => 
    TEETH.slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage),
    [TEETH, page, rowsPerPage]
  );

  // Teeth that don't have furcation (single-rooted)
  const singleRootedTeeth = useMemo(() => [1, 2, 3, 11, 12, 13], []);
  const hasFurcation = useCallback((toothNumber) => !singleRootedTeeth.includes(toothNumber), [singleRootedTeeth]);



  const renderToothIcon = useCallback((universalNumber, quadrantNumber, jaw) => {
    const toothData = chartData.teeth[universalNumber] || {};
    
    // Determine image file name based on jaw and quadrant number
    // Upper jaw: 1.png to 8.png (based on quadrantNumber)
    // Lower jaw: 11.png to 18.png (quadrantNumber + 10)
    let imageFileName;
    if (jaw === 'Upper') {
      imageFileName = `${quadrantNumber}.png`;
    } else {
      // Lower jaw
      imageFileName = `${quadrantNumber + 10}.png`;
    }
    
    // Determine image source: use implant image if implant is true and not missing
    const isImplant = !!(toothData.implant && !toothData.missing);
    const imageSrc = isImplant ? '/teeth/implant.png' : `/teeth/${imageFileName}`;
    
    // Always try to use image (for upper and lower jaws, images exist)
    return (
        <Box
          sx={{
            position: 'relative',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            height: { xs: 32, md: 38 },
            width: { xs: 24, md: 28 },
            cursor: 'pointer',
          }}
          onClick={() => handleToggleMissing(universalNumber)}
        >
          <Box
            component="img"
            src={imageSrc}
            alt={isImplant ? `Implant ${universalNumber}` : `Tooth ${universalNumber}`}
            sx={{
              maxWidth: '100%',
              maxHeight: '100%',
              objectFit: 'contain',
              opacity: toothData.missing ? 0.4 : 1,
              filter: toothData.missing ? 'grayscale(100%)' : 'none',
            }}
            onError={(e) => {
              // If image fails, fallback to SVG (will be handled by parent component)
              console.error(`Failed to load image: ${imageSrc}`);
              e.target.style.display = 'none';
            }}
          />
          
          {/* Missing overlay */}
          {toothData.missing && (
            <Box
              sx={{
                position: 'absolute',
                top: '50%',
                left: 0,
                right: 0,
                height: { xs: 2, md: 2.5 },
                bgcolor: theme.palette.error.main,
                transform: 'rotate(-20deg) translateY(-50%)',
                zIndex: 1,
              }}
            />
          )}
          
          {/* BOP indicator - M (mesial, index 0) on right side */}
          {!toothData.missing && toothData.facial?.bleeding?.[0] && (
            <Box
              sx={{
                position: 'absolute',
                top: 1,
                right: 1,
                width: { xs: 4, md: 5 },
                height: { xs: 4, md: 5 },
                borderRadius: '50%',
                bgcolor: theme.palette.error.main,
                zIndex: 1,
              }}
            />
          )}
          
          {/* BOP indicator - D (distal, index 2) on left side */}
          {!toothData.missing && toothData.facial?.bleeding?.[2] && (
            <Box
              sx={{
                position: 'absolute',
                top: 1,
                left: 1,
                width: { xs: 4, md: 5 },
                height: { xs: 4, md: 5 },
                borderRadius: '50%',
                bgcolor: theme.palette.error.main,
                zIndex: 1,
              }}
            />
          )}
        </Box>
      );
  }, [chartData, handleToggleMissing, theme]);

  // Memoized handleInputChange for pocket depth
  const handleInputChange = useCallback((universalNum, surface, index, value, paginatedTeeth) => {
    // Update immediately for auto-focus to work, but debounce the actual state update
    handleUpdateField(universalNum, surface, 'pocketDepth', index, value, false);
    // Auto-focus to next field if single digit entered
    if (value !== null && value >= 0 && value <= 9) {
      setTimeout(() => {
        let nextId = null;
        if (surface === 'facial') {
          if (index < 2) {
            // Next facial site
            nextId = `facial-${universalNum}-${index + 1}`;
          } else {
            // First lingual site of same tooth
            nextId = `lingual-${universalNum}-0`;
          }
        } else if (index < 2) {
          // surface === 'lingual'
          // Next lingual site
          nextId = `lingual-${universalNum}-${index + 1}`;
        } else {
          // Last lingual site (index === 2) - go to first facial PD site of next tooth
          const currentIndex = paginatedTeeth.findIndex(t => t.universalNumber === universalNum);
          if (currentIndex < paginatedTeeth.length - 1) {
            const nextTooth = paginatedTeeth[currentIndex + 1];
            nextId = `facial-${nextTooth.universalNumber}-0`;
          }
          // If no next tooth, don't auto-focus (stay in current field or let user navigate manually)
        }
        if (nextId) {
          const nextInput = document.getElementById(nextId);
          if (nextInput) {
            nextInput.focus();
            nextInput.select();
          }
        }
      }, 10);
    }
  }, [handleUpdateField]);

  return (
    <Box>
      {/* Toggle for Gingival Margin and Mobility */}
      <Card sx={{ mb: 2 }}>
        <CardContent sx={{ py: 1.5 }}>
          <Stack direction="row" spacing={3} flexWrap="wrap">
            <FormControlLabel
              control={
                <Switch
                  checked={showGingivalMargin}
                  onChange={(e) => setShowGingivalMargin(e.target.checked)}
                  size="small"
                />
              }
              label={
                <Typography variant="body2">
                  نمایش ستون Gingival Margin
                </Typography>
              }
            />
            <FormControlLabel
              control={
                <Switch
                  checked={showMobility}
                  onChange={(e) => setShowMobility(e.target.checked)}
                  size="small"
                />
              }
              label={
                <Typography variant="body2">
                  نمایش ستون Mobility
                </Typography>
              }
            />
          </Stack>
        </CardContent>
      </Card>

              {/* Facial Surface Table - Combined sites (Facial above, Lingual below) */}
        <TableContainer 
          component={Card}
          sx={{
            maxHeight: { xs: '60vh', md: 'none' },
            overflow: 'auto',
            '& .MuiTable-root': {
              minWidth: { xs: 800, md: 'auto' },
            },
          }}
        >
          <Table 
            size="small"
            sx={{
              '& .MuiTableCell-root': {
                fontSize: { xs: '0.75rem', md: '0.875rem' },
                padding: { xs: '4px 1px', md: '6px 2px' },
              },
            }}
          >
            <TableHead>
                              <TableRow>
                  {/* Tooth Column - First */}
                  <TableCell align="center" rowSpan={2} sx={{ 
                    width: { xs: 45, md: 55 },
                    minWidth: { xs: 45, md: 55 },
                    maxWidth: { xs: 45, md: 55 },
                    fontWeight: 'bold', 
                    position: 'sticky', 
                    left: 0, 
                    zIndex: 2,
                    bgcolor: 'var(--palette-background-neutral)',
                    padding: { xs: '4px 2px', md: '4px 4px' },
                    fontSize: { xs: '0.7rem', md: '0.875rem' },
                  }}>
                    Tooth
                  </TableCell>
                                  {/* Probing Sites */}
                  <TableCell align="center" colSpan={3} sx={{ fontWeight: 'bold', padding: '4px 0px' }}>
                    Probing Sites
                  </TableCell>
                  <TableCell align="center" rowSpan={2} sx={{ fontWeight: 'bold', width: { xs: 45, md: 50 }, minWidth: { xs: 45, md: 50 }, maxWidth: { xs: 45, md: 50 }, whiteSpace: 'nowrap', fontSize: { xs: '0.7rem', md: '0.875rem' }, padding: { xs: '4px 2px', md: '4px 4px' } }}>
                    BOP
                  </TableCell>
                  {/* Gingival Margin - Conditional */}
                  {showGingivalMargin && (
                    <TableCell align="center" colSpan={3} sx={{ fontWeight: 'bold', padding: '4px 0px' }}>
                    Gingival margin
                  </TableCell>
                  )}
                  <TableCell align="center" rowSpan={2} sx={{ fontWeight: 'bold', width: { xs: 35, md: 40 }, minWidth: { xs: 35, md: 40 }, maxWidth: { xs: 35, md: 40 }, whiteSpace: 'nowrap', fontSize: { xs: '0.7rem', md: '0.875rem' }, padding: { xs: '4px 2px', md: '4px 4px' } }}>
                    Plaque
                  </TableCell>
                  <TableCell align="center" rowSpan={2} sx={{ fontWeight: 'bold', width: { xs: 50, md: 55 }, minWidth: { xs: 50, md: 55 }, maxWidth: { xs: 50, md: 55 }, whiteSpace: 'nowrap', fontSize: { xs: '0.7rem', md: '0.875rem' }, padding: { xs: '4px 2px', md: '4px 4px' } }}>
                    Furcation
                  </TableCell>
                  {/* Mobility - Conditional */}
                  {showMobility && (
                    <TableCell align="center" rowSpan={2} sx={{ fontWeight: 'bold', width: { xs: 50, md: 55 }, minWidth: { xs: 50, md: 55 }, maxWidth: { xs: 50, md: 55 }, whiteSpace: 'nowrap', fontSize: { xs: '0.7rem', md: '0.875rem' }, padding: { xs: '4px 2px', md: '4px 4px' } }}>
                      Mobility
                    </TableCell>
                  )}
                </TableRow>
                <TableRow>
                  {FACIAL_SITES.map((site, idx) => (
                    <TableCell key={site} align="center" sx={{ fontWeight: 'bold', padding: '2px 0px', width: 32, maxWidth: 32 }}>
                      <Stack direction="column" spacing={0.1} alignItems="center">
                        <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.65rem', lineHeight: 1 }}>{site}</Typography>
                        <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.65rem', color: 'text.secondary', lineHeight: 1 }}>{LINGUAL_SITES[idx]}</Typography>
                      </Stack>
                    </TableCell>
                  ))}
                  {showGingivalMargin && FACIAL_SITES.map((site, idx) => (
                    <TableCell key={`gm-${site}`} align="center" sx={{ fontWeight: 'bold', padding: '2px 1px', minWidth: 40, maxWidth: 45 }}>
                      <Stack direction="column" spacing={0.2} alignItems="center">
                        <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.7rem' }}>{site}</Typography>
                        <Typography variant="caption" fontWeight="bold" sx={{ fontSize: '0.7rem', color: 'text.secondary' }}>{LINGUAL_SITES[idx]}</Typography>
                      </Stack>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
                 <TableBody>
                   {paginatedTeeth.map((tooth) => {
                     const toothData = chartData.teeth[tooth.universalNumber] || {};
                const facialPD = toothData.facial?.pocketDepth || [null, null, null];
                const lingualPD = toothData.lingual?.pocketDepth || [null, null, null];
                const facialGM = toothData.facial?.gingivalMargin || [null, null, null];
                const lingualGM = toothData.lingual?.gingivalMargin || [null, null, null];
                const facialBleeding = toothData.facial?.bleeding || [false, false, false];
                
                // Check BOP for mesial (index 0) and distal (index 2) - only check facial side
                const mesialBOP = facialBleeding[0] || false;
                const distalBOP = facialBleeding[2] || false;

                                  return (
                  <PeriodontalChartRow
                         key={tooth.universalNumber}
                    tooth={tooth}
                    toothData={toothData}
                    facialPD={facialPD}
                    lingualPD={lingualPD}
                    facialGM={facialGM}
                    lingualGM={lingualGM}
                    mesialBOP={mesialBOP}
                    distalBOP={distalBOP}
                    showGingivalMargin={showGingivalMargin}
                    showMobility={showMobility}
                    theme={theme}
                    renderToothIcon={renderToothIcon}
                    handleInputChange={handleInputChange}
                    handleUpdateBOP={handleUpdateBOP}
                    handleUpdateField={handleUpdateField}
                    hasFurcation={hasFurcation}
                    paginatedTeeth={paginatedTeeth}
                    handleToggleMissing={handleToggleMissing}
                  />
                  );
                })}
          </TableBody>
          </Table>
        </TableContainer>

            <TablePaginationCustom
              count={TEETH.length}
              page={page}
              rowsPerPage={rowsPerPage}
              onPageChange={handleChangePage}
              onRowsPerPageChange={handleChangeRowsPerPage}
              rowsPerPageOptions={[defaultRowsPerPage]}
            />

    </Box>
  );
}
