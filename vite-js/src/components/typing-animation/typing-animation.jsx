import { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Table from '@mui/material/Table';
import { alpha } from '@mui/material/styles';
import TableRow from '@mui/material/TableRow';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';

// ----------------------------------------------------------------------

export function TypingAnimation({ text, speed = 30, onComplete, sx, ...other }) {
  const [displayedText, setDisplayedText] = useState('');
  const [isComplete, setIsComplete] = useState(false);
  const timeoutRef = useRef(null);

  useEffect(() => {
    if (!text) {
      setDisplayedText('');
      setIsComplete(false);
      return;
    }

    setDisplayedText('');
    setIsComplete(false);
    let currentIndex = 0;

    const typeNextChar = () => {
      if (currentIndex < text.length) {
        setDisplayedText(text.substring(0, currentIndex + 1));
        currentIndex += 1;
        timeoutRef.current = setTimeout(typeNextChar, speed);
      } else {
        setIsComplete(true);
        if (onComplete) {
          onComplete();
        }
      }
    };

    // Start typing after a small delay
    timeoutRef.current = setTimeout(typeNextChar, 100);

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [text, speed, onComplete]);

  return (
    <Box
      sx={{
        position: 'relative',
        ...sx,
      }}
      {...other}
    >
      <Typography
        component="div"
        sx={{
          whiteSpace: 'pre-wrap',
          wordBreak: 'break-word',
          lineHeight: 1.8,
          fontSize: '0.8125rem', // Smaller size during typing and after completion (same size)
          position: 'relative',
          '&::before': !isComplete ? {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: (theme) => `linear-gradient(90deg, transparent 0%, ${alpha(theme.palette.primary.main, 0.2)} 50%, transparent 100%)`,
            backgroundSize: '200% 100%',
            pointerEvents: 'none',
            mixBlendMode: 'overlay',
            animation: 'shining 2s linear infinite',
            '@keyframes shining': {
              '0%': { backgroundPosition: '-200% 0' },
              '100%': { backgroundPosition: '200% 0' },
            },
          } : {},
          '&::after': {
            content: isComplete ? '""' : '"▊"',
            display: 'inline-block',
            width: '2px',
            height: '1.2em',
            ml: 0.5,
            bgcolor: 'primary.main',
            animation: isComplete ? 'none' : 'blink 1s infinite',
            '@keyframes blink': {
              '0%, 50%': { opacity: 1 },
              '51%, 100%': { opacity: 0 },
            },
          },
        }}
      >
        {displayedText}
      </Typography>
    </Box>
  );
}

// ----------------------------------------------------------------------

// Cephalometric Parameters Table Component
export function CephalometricTable({ measurements = {} }) {
  const normalRanges = {
    SNA: { min: 80, max: 84, description: 'موقعیت ماگزیلا نسبت به جمجمه' },
    SNB: { min: 78, max: 82, description: 'موقعیت مندیبل نسبت به جمجمه' },
    ANB: { min: 2, max: 4, description: 'رابطه اسکلتی فک بالا و پایین' },
    FMA: { min: 22, max: 28, description: 'الگوی رشد عمودی صورت' },
    FMIA: { min: 65, max: 75, description: 'زاویه دندان قدامی پایین با صفحه فرانکفورت' },
    IMPA: { min: 90, max: 95, description: 'زاویه دندان قدامی پایین با صفحه مندیبل' },
    'U1-SN': { min: 102, max: 108, description: 'زاویه دندان قدامی بالا با صفحه SN' },
    'L1-MP': { min: 90, max: 95, description: 'زاویه دندان قدامی پایین با صفحه مندیبل' },
    GoGnSN: { min: 30, max: 38, description: 'زاویه صفحه مندیبل با صفحه SN' },
  };

  const getStatusColor = (value, min, max) => {
    if (value === undefined || value === null) return 'text.secondary';
    if (value >= min && value <= max) return 'success.main';
    if (value < min - 2 || value > max + 2) return 'error.main';
    return 'warning.main';
  };

  const getStatusText = (value, min, max) => {
    if (value === undefined || value === null) return 'نامشخص';
    if (value >= min && value <= max) return 'نرمال';
    if (value < min) return 'کمتر از نرمال';
    return 'بیشتر از نرمال';
  };

  const parameters = Object.keys(normalRanges).filter(key => measurements[key] !== undefined);

  if (parameters.length === 0) {
    return (
      <Typography variant="body2" color="text.secondary">
        داده‌ای برای نمایش وجود ندارد
      </Typography>
    );
  }

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 600 }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>پارامتر</TableCell>
            <TableCell align="center" sx={{ fontWeight: 600 }}>مقدار بیمار</TableCell>
            <TableCell align="center" sx={{ fontWeight: 600 }}>مقدار نرمال</TableCell>
            <TableCell align="center" sx={{ fontWeight: 600 }}>وضعیت</TableCell>
            <TableCell sx={{ fontWeight: 600 }}>توضیحات</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {parameters.map((param) => {
            const value = measurements[param];
            const range = normalRanges[param];
            const statusColor = getStatusColor(value, range.min, range.max);
            const statusText = getStatusText(value, range.min, range.max);

            return (
              <TableRow key={param} hover>
                <TableCell sx={{ fontWeight: 500 }}>{param}</TableCell>
                <TableCell align="center">
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {value !== undefined && value !== null ? value.toFixed(1) : '-'}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" color="text.secondary">
                    {range.min} - {range.max}
                  </Typography>
                </TableCell>
                <TableCell align="center">
                  <Typography variant="body2" sx={{ color: statusColor, fontWeight: 500 }}>
                    {statusText}
                  </Typography>
                </TableCell>
                <TableCell>
                  <Typography variant="caption" color="text.secondary">
                    {range.description}
                  </Typography>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </Box>
  );
}

// ----------------------------------------------------------------------

// Occlusion Table Component
function OcclusionTable({ occlusionData = [] }) {
  // Fixed structure: always show 4 rows
  const fixedOcclusionTypes = ['کانین چپ', 'مولر چپ', 'کانین راست', 'مولر راست'];
  
  // Create a map from occlusionData for quick lookup
  const positionMap = {};
  occlusionData.forEach((item) => {
    if (item.occlusion && item.position) {
      positionMap[item.occlusion] = item.position;
    }
  });

  return (
    <Box sx={{ overflowX: 'auto' }}>
      <Table size="small" sx={{ minWidth: 400 }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>اکلوژن دندان</TableCell>
            <TableCell align="center" sx={{ fontWeight: 600 }}>موقعیت</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {fixedOcclusionTypes.map((occlusionType) => (
            <TableRow key={occlusionType} hover>
              <TableCell sx={{ fontWeight: 500 }}>
                {occlusionType}
              </TableCell>
              <TableCell align="center">
                <Typography variant="body2" sx={{ fontWeight: 500 }}>
                  {positionMap[occlusionType] || '-'}
                </Typography>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

// ----------------------------------------------------------------------

export function TypingReport({ sections = [], onComplete, sx, ...other }) {
  const [currentSectionIndex, setCurrentSectionIndex] = useState(0);
  const [completedSections, setCompletedSections] = useState([]);
  const sectionCompleteRef = useRef(false);
  const sectionsRef = useRef(null);
  const hasInitializedRef = useRef(false);
  const timeoutRef = useRef(null);
  const onCompleteCalledRef = useRef(false);

  // Debug: Log sections when they change
  useEffect(() => {
    console.log('[TypingReport] Sections received:', {
      count: sections.length,
      sections: sections.map(s => ({ title: s.title, contentLength: s.content?.length || 0, type: s.type }))
    });
  }, [sections]);

  useEffect(() => {
    if (sections.length === 0) {
      setCurrentSectionIndex(0);
      setCompletedSections([]);
      sectionCompleteRef.current = false;
      hasInitializedRef.current = false;
      onCompleteCalledRef.current = false;
      return;
    }

    // Compare sections by content to avoid unnecessary resets
    const sectionsKey = JSON.stringify(sections.map(s => ({ title: s.title, content: s.content })));
    
    // Only reset if sections actually changed (not just reference)
    if (sectionsRef.current !== sectionsKey) {
      // Clear any pending timeouts
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
      
      sectionsRef.current = sectionsKey;
      onCompleteCalledRef.current = false; // Reset onComplete flag when sections change
      
      // Only reset if we've already initialized (avoid double initialization in Strict Mode)
      if (hasInitializedRef.current) {
        setCurrentSectionIndex(0);
        setCompletedSections([]);
        sectionCompleteRef.current = false;
      } else {
        hasInitializedRef.current = true;
      }
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, [sections]);
  
  const handleSectionComplete = useCallback(() => {
    if (sectionCompleteRef.current) return;
    sectionCompleteRef.current = true;

    setCurrentSectionIndex((currentIdx) => {
      const currentSection = sections[currentIdx];
      if (currentSection) {
        setCompletedSections((prev) => {
          // Prevent duplicate sections
          const isDuplicate = prev.some(
            s => s.title === currentSection.title && s.type === currentSection.type
          );
          if (isDuplicate) {
            return prev;
          }
          return [...prev, currentSection];
        });
      }

      // Move to next section after a delay
      timeoutRef.current = setTimeout(() => {
        setCurrentSectionIndex((prev) => {
          if (prev < sections.length - 1) {
            sectionCompleteRef.current = false;
            return prev + 1;
          }
            // All sections complete - only call onComplete once
            if (onComplete && !onCompleteCalledRef.current) {
              onCompleteCalledRef.current = true;
              onComplete();
            }
            return prev;
        });
        timeoutRef.current = null;
      }, 500);

      return currentIdx;
    });
  }, [sections, onComplete]);

  const currentSection = sections[currentSectionIndex];

  // Auto-complete table sections immediately
  useEffect(() => {
    if (currentSection && (currentSection.type === 'table' || currentSection.type === 'occlusion-table') && !sectionCompleteRef.current) {
      // Check if this section is already in completedSections to prevent duplicate completion
      const isAlreadyCompleted = completedSections.some(
        s => s.title === currentSection.title && s.type === currentSection.type
      );
      
      // Also check if currentSectionIndex matches the section we're trying to complete
      const currentSectionMatches = sections[currentSectionIndex] && 
        sections[currentSectionIndex].title === currentSection.title &&
        sections[currentSectionIndex].type === currentSection.type;
      
      if (!isAlreadyCompleted && currentSectionMatches) {
        const timer = setTimeout(() => {
          handleSectionComplete();
        }, 100);
        return () => clearTimeout(timer);
      }
    }
  }, [currentSection, currentSectionIndex, sections, handleSectionComplete, completedSections]);

  return (
    <Card
      sx={{
        pt: 1,
        bgcolor: (theme) => alpha(theme.palette.background.paper, 0.8),
        backdropFilter: 'blur(10px)',
        border: (theme) => `1px solid ${alpha(theme.palette.divider, 0.1)}`,
        ...sx,
      }}
      {...other}
    >
      <Box sx={{ mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          آنالیز کامل
        </Typography>
        <Typography variant="body2" color="text.secondary">
          در حال انجام آنالیزهای مختلف...
        </Typography>
      </Box>

      <Box 
        sx={{ 
          minHeight: '400px',
          maxHeight: 'none',
          overflowY: 'visible',
          py: 2,
        }}
      >
        {/* Completed sections */}
        {completedSections.map((section, index) => {
          // Use unique key based on title, type, and index to prevent duplicate rendering
          const sectionKey = `${section.title}-${section.type || 'text'}-${index}`;
          return (
            <Box key={sectionKey} sx={{ mb: 3 }}>
              <Typography
                variant="subtitle1"
                sx={{
                  mb: 1,
                  color: 'text.primary',
                  fontWeight: 600,
                  display: 'flex',
                  alignItems: 'center',
                  gap: 1,
                }}
              >
                <Box
                  component="span"
                  sx={{
                    width: 8,
                    height: 8,
                    borderRadius: '50%',
                    bgcolor: 'text.secondary',
                    display: 'inline-block',
                  }}
                />
                {section.title}
              </Typography>
              <Box
                sx={{
                  pl: 2,
                }}
              >
                {section.type === 'table' && section.tableData ? (
                  <CephalometricTable measurements={section.tableData.measurements} />
                ) : section.type === 'occlusion-table' && section.tableData ? (
                  <OcclusionTable occlusionData={section.tableData.occlusionData} />
                ) : (
                  <Typography
                    variant="body2"
                    sx={{
                      whiteSpace: 'pre-wrap',
                      wordBreak: 'break-word',
                      lineHeight: 1.8,
                      fontSize: '0.8125rem', // Same size as typing text
                    }}
                  >
                    {section.content}
                  </Typography>
                )}
              </Box>
            </Box>
          );
        })}

        {/* Current typing section - only show if not already in completedSections */}
        {currentSection && !completedSections.some(s => s.title === currentSection.title && s.type === currentSection.type) && (
          <Box sx={{ mb: 3 }}>
            <Typography
              variant="subtitle1"
              sx={{
                mb: 1,
                color: 'text.primary',
                fontWeight: 600,
                display: 'flex',
                alignItems: 'center',
                gap: 1,
              }}
            >
              <Box
                component="span"
                sx={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  bgcolor: 'text.secondary',
                  display: 'inline-block',
                  animation: currentSection.type === 'table' ? 'none' : 'pulse 1.5s infinite',
                  '@keyframes pulse': {
                    '0%, 100%': { opacity: 1 },
                    '50%': { opacity: 0.5 },
                  },
                }}
              />
              {currentSection.title}
            </Typography>
            <Box
              sx={{
                pl: 2,
              }}
            >
              {currentSection.type === 'table' && currentSection.tableData ? (
                <CephalometricTable measurements={currentSection.tableData.measurements} />
              ) : currentSection.type === 'occlusion-table' && currentSection.tableData ? (
                <OcclusionTable occlusionData={currentSection.tableData.occlusionData} />
              ) : (
                <TypingAnimation
                  text={currentSection.content}
                  speed={20}
                  onComplete={handleSectionComplete}
                />
              )}
            </Box>
          </Box>
        )}

        {/* Progress indicator */}
        {sections.length > 0 && (
          <Box sx={{ mt: 3, pt: 2, borderTop: (theme) => `1px solid ${theme.palette.divider}` }}>
            <Typography variant="caption" color="text.secondary">
              پیشرفت: {completedSections.length + (currentSection ? 1 : 0)} / {sections.length}
            </Typography>
          </Box>
        )}
      </Box>
    </Card>
  );
}
