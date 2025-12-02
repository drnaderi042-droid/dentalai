import PropTypes from 'prop-types';
import { useState, useEffect } from 'react';

import {
  Box,
  Card,
  Chip,
  Grid,
  Alert,
  Stack,
  Button,
  Dialog,
  Divider,
  TextField,
  IconButton,
  Typography,
  DialogTitle,
  DialogActions,
  DialogContent,
} from '@mui/material';

import { Iconify } from 'src/components/iconify';

// Parse AI diagnosis text into structured sections
const parseAIDiagnosis = (diagnosisText) => {
  if (!diagnosisText) {
    return {
      summary: '',
      intraoral: '',
      extraoral: '',
      radiographic: '',
      cephalometric: '',
      recommendations: '',
      raw: '',
    };
  }

  const sections = {
    summary: '',
    intraoral: '',
    extraoral: '',
    radiographic: '',
    cephalometric: '',
    recommendations: '',
    raw: diagnosisText,
  };

  // Split by headers and clean up
  const lines = diagnosisText.split('\n');
  let currentSection = 'summary';
  let currentContent = [];

  lines.forEach(line => {
    const trimmedLine = line.trim();
    
    // Check for section headers
    if (trimmedLine.includes('داخل دهانی') || trimmedLine.includes('تحلیل داخل دهانی')) {
      if (currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }
      currentSection = 'intraoral';
      currentContent = [];
    } else if (trimmedLine.includes('خارج دهانی') || trimmedLine.includes('تحلیل خارج دهانی')) {
      if (currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }
      currentSection = 'extraoral';
      currentContent = [];
    } else if (trimmedLine.includes('رادیولوژی') || trimmedLine.includes('تحلیل رادیولوژی')) {
      if (currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }
      currentSection = 'radiographic';
      currentContent = [];
    } else if (trimmedLine.includes('سفالومتری') || trimmedLine.includes('آنالیز سفالومتری')) {
      if (currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }
      currentSection = 'cephalometric';
      currentContent = [];
    } else if (trimmedLine.includes('توصیه') || trimmedLine.includes('طرح درمان')) {
      if (currentContent.length > 0) {
        sections[currentSection] = currentContent.join('\n').trim();
      }
      currentSection = 'recommendations';
      currentContent = [];
    } else if (trimmedLine.length > 0 && !trimmedLine.match(/^[*#-]+$/)) {
      // Add non-empty lines that aren't just separators
      currentContent.push(trimmedLine);
    }
  });

  // Save last section
  if (currentContent.length > 0) {
    sections[currentSection] = currentContent.join('\n').trim();
  }

  // If no structured sections found, use the whole text as summary
  if (!sections.intraoral && !sections.extraoral && !sections.radiographic && !sections.cephalometric) {
    sections.summary = diagnosisText;
  }

  // Clean up sections - remove very short or meaningless content
  Object.keys(sections).forEach(key => {
    if (key !== 'raw') {
      const content = sections[key];
      // Remove if too short or only contains punctuation/symbols
      if (content.length < 3 || content.match(/^[:\s\-•و]+$/)) {
        sections[key] = '';
      }
    }
  });

  return sections;
};

// Combine sections back to text
const combineSections = (sections) => {
  const parts = [];

  if (sections.summary) parts.push(`**خلاصه:**\n${sections.summary}`);
  if (sections.intraoral) parts.push(`\n**تحلیل داخل دهانی:**\n${sections.intraoral}`);
  if (sections.extraoral) parts.push(`\n**تحلیل خارج دهانی:**\n${sections.extraoral}`);
  if (sections.radiographic) parts.push(`\n**تحلیل رادیولوژی:**\n${sections.radiographic}`);
  if (sections.cephalometric) parts.push(`\n**تحلیل سفالومتری:**\n${sections.cephalometric}`);
  if (sections.recommendations) parts.push(`\n**توصیه‌های درمانی:**\n${sections.recommendations}`);

  return parts.join('\n');
};

export default function AIDiagnosisDisplay({ diagnosis, onUpdate, onSave, readOnly = false }) {
  const [sections, setSections] = useState(() => parseAIDiagnosis(diagnosis));
  const [editDialogOpen, setEditDialogOpen] = useState(false);
  const [currentEditSection, setCurrentEditSection] = useState('');
  const [editValue, setEditValue] = useState('');

  useEffect(() => {
    setSections(parseAIDiagnosis(diagnosis));
  }, [diagnosis]);

  const handleEditSection = (sectionKey, currentValue) => {
    setCurrentEditSection(sectionKey);
    setEditValue(currentValue);
    setEditDialogOpen(true);
  };

  const handleSaveSection = () => {
    const updatedSections = {
      ...sections,
      [currentEditSection]: editValue,
    };
    setSections(updatedSections);
    
    // Update parent component
    const combinedText = combineSections(updatedSections);
    onUpdate(combinedText);
    
    setEditDialogOpen(false);
    setCurrentEditSection('');
    setEditValue('');
  };

  const getSeverityColor = (text) => {
    if (!text) return 'default';
    const lowerText = text.toLowerCase();
    if (lowerText.includes('شدید') || lowerText.includes('severe')) return 'error';
    if (lowerText.includes('متوسط') || lowerText.includes('moderate')) return 'warning';
    return 'success';
  };

  const extractKeyFindings = (text) => {
    if (!text) return [];
    
    const findings = [];
    const lines = text.split('\n');
    
    lines.forEach(line => {
      if (line.match(/^[•\-\d.]/)) {
        findings.push(line.replace(/^[•\-\d.]\s*/, '').trim());
      }
    });
    
    return findings.slice(0, 5); // Top 5 findings
  };

  const sectionConfig = [
    {
      key: 'summary',
      title: 'خلاصه تشخیص',
      icon: 'solar:document-text-bold',
      color: 'primary',
    },
    {
      key: 'intraoral',
      title: 'تحلیل داخل دهانی',
      icon: 'solar:tooth-bold',
      color: 'info',
    },
    {
      key: 'extraoral',
      title: 'تحلیل خارج دهانی',
      icon: 'solar:user-bold',
      color: 'success',
    },
    {
      key: 'radiographic',
      title: 'تحلیل رادیولوژی',
      icon: 'solar:camera-bold',
      color: 'warning',
    },
    {
      key: 'cephalometric',
      title: 'تحلیل سفالومتری',
      icon: 'solar:ruler-bold',
      color: 'secondary',
    },
    {
      key: 'recommendations',
      title: 'توصیه‌های درمانی',
      icon: 'solar:health-bold',
      color: 'error',
    },
  ];

  if (!diagnosis || diagnosis.trim() === '') {
    return (
      <Card sx={{ p: 4, textAlign: 'center', bgcolor: 'background.neutral' }}>
        <Iconify icon="solar:robot-outline" width={64} color="text.disabled" />
        <Typography variant="h6" sx={{ mt: 2, color: 'text.secondary' }}>
          هنوز تحلیل AI انجام نشده است
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          روی دکمه "تحلیل کامل با AI" کلیک کنید تا تشخیص شروع شود
        </Typography>
      </Card>
    );
  }

  return (
    <Stack spacing={3}>
      {/* Summary Card */}
      {sections.summary && (
        <Card 
          sx={{ 
            p: 3, 
            border: 2, 
            borderColor: 'primary.main',
            bgcolor: 'primary.lighter',
          }}
        >
          <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 2 }}>
            <Stack direction="row" alignItems="center" spacing={2}>
              <Iconify icon="solar:star-bold" width={28} color="primary.main" />
              <Typography variant="h5" color="primary.main" fontWeight={700}>
                خلاصه تشخیص AI
              </Typography>
            </Stack>
            {!readOnly && (
              <IconButton
                size="small"
                onClick={() => handleEditSection('summary', sections.summary)}
                sx={{ color: 'primary.main' }}
              >
                <Iconify icon="solar:pen-bold" width={20} />
              </IconButton>
            )}
          </Stack>
          <Typography variant="body1" sx={{ lineHeight: 2, fontSize: '1.1rem' }}>
            {sections.summary}
          </Typography>
        </Card>
      )}

      {/* Key Findings */}
      {sections.intraoral && extractKeyFindings(sections.intraoral).length > 0 && (
        <Card sx={{ p: 3, bgcolor: 'background.neutral' }}>
          <Stack direction="row" alignItems="center" spacing={1.5} sx={{ mb: 2 }}>
            <Iconify icon="solar:flag-bold" width={24} color="warning.main" />
            <Typography variant="h6" fontWeight={600}>
              یافته‌های کلیدی
            </Typography>
          </Stack>
          <Stack direction="row" flexWrap="wrap" gap={1}>
            {extractKeyFindings(sections.intraoral).map((finding, index) => (
              <Chip
                key={index}
                label={finding}
                size="medium"
                color={getSeverityColor(finding)}
                variant="outlined"
                sx={{ fontSize: '0.9rem', py: 2.5 }}
              />
            ))}
          </Stack>
        </Card>
      )}

      {/* Detailed Sections - Grid Layout */}
      <Grid container spacing={3}>
        {sectionConfig.map((config) => {
          const sectionContent = sections[config.key];
          
          // Skip if empty or is summary (already shown above)
          if (!sectionContent || config.key === 'summary') return null;

          return (
            <Grid item xs={12} md={6} key={config.key}>
              <Card
                sx={{
                  p: 3,
                  height: '100%',
                  border: 1,
                  borderColor: `${config.color}.main`,
                  borderRadius: 2,
                  transition: 'all 0.3s',
                  '&:hover': {
                    boxShadow: `0 8px 24px ${config.color === 'primary' ? 'rgba(0, 123, 255, 0.2)' : config.color === 'error' ? 'rgba(255, 0, 0, 0.2)' : 'rgba(0, 0, 0, 0.1)'}`,
                    transform: 'translateY(-4px)',
                  },
                }}
              >
                <Stack spacing={2}>
                  {/* Header */}
                  <Stack direction="row" alignItems="center" justifyContent="space-between">
                    <Stack direction="row" alignItems="center" spacing={1.5}>
                      <Box
                        sx={{
                          width: 48,
                          height: 48,
                          borderRadius: 2,
                          bgcolor: `${config.color}.lighter`,
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                        }}
                      >
                        <Iconify icon={config.icon} width={28} color={`${config.color}.main`} />
                      </Box>
                      <Typography variant="h6" fontWeight={600} color={`${config.color}.darker`}>
                        {config.title}
                      </Typography>
                    </Stack>
                    {!readOnly && (
                      <IconButton
                        size="small"
                        onClick={() => handleEditSection(config.key, sectionContent)}
                        sx={{ color: `${config.color}.main` }}
                      >
                        <Iconify icon="solar:pen-bold" width={18} />
                      </IconButton>
                    )}
                  </Stack>

                  <Divider />

                  {/* Content */}
                  <Typography
                    variant="body2"
                    sx={{
                      lineHeight: 2,
                      whiteSpace: 'pre-wrap',
                      color: 'text.secondary',
                      minHeight: 100,
                    }}
                  >
                    {sectionContent}
                  </Typography>

                  {/* Actions */}
                  {!readOnly && (
                    <Stack direction="row" spacing={1} sx={{ pt: 1 }}>
                      <Button
                        size="small"
                        variant="text"
                        startIcon={<Iconify icon="solar:copy-bold" />}
                        onClick={() => navigator.clipboard.writeText(sectionContent)}
                        sx={{ color: 'text.secondary' }}
                      >
                        کپی
                      </Button>
                    </Stack>
                  )}
                </Stack>
              </Card>
            </Grid>
          );
        })}
      </Grid>

      {/* Edit Dialog */}
      <Dialog 
        open={editDialogOpen} 
        onClose={() => setEditDialogOpen(false)}
        maxWidth="md"
        fullWidth
      >
        <DialogTitle>
          <Stack direction="row" alignItems="center" justifyContent="space-between">
            <Typography variant="h6">
              ویرایش {sectionConfig.find(s => s.key === currentEditSection)?.title}
            </Typography>
            <IconButton onClick={() => setEditDialogOpen(false)}>
              <Iconify icon="solar:close-circle-bold" />
            </IconButton>
          </Stack>
        </DialogTitle>
        
        <DialogContent>
          <Alert severity="info" sx={{ mb: 2 }}>
            شما می‌توانید نتایج AI را ویرایش کنید. تغییرات شما بعد از ذخیره اعمال خواهد شد.
          </Alert>
          
          <TextField
            fullWidth
            multiline
            rows={12}
            value={editValue}
            onChange={(e) => setEditValue(e.target.value)}
            placeholder="متن خود را وارد کنید..."
            sx={{ mt: 2 }}
          />
          
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            تعداد کلمات: {editValue.split(/\s+/).filter(w => w.length > 0).length}
          </Typography>
        </DialogContent>
        
        <DialogActions>
          <Button onClick={() => setEditDialogOpen(false)}>
            انصراف
          </Button>
          <Button 
            variant="contained" 
            onClick={handleSaveSection}
            startIcon={<Iconify icon="solar:check-circle-bold" />}
          >
            ذخیره تغییرات
          </Button>
        </DialogActions>
      </Dialog>

      {/* Action Buttons */}
      {!readOnly && (
        <Card sx={{ p: 2, bgcolor: 'background.neutral' }}>
          <Stack direction="row" spacing={2} flexWrap="wrap">
            <Button
              variant="outlined"
              startIcon={<Iconify icon="solar:printer-bold" />}
              onClick={() => window.print()}
            >
              پرینت تحلیل
            </Button>
            <Button
              variant="outlined"
              startIcon={<Iconify icon="solar:export-bold" />}
              onClick={() => {
                const text = combineSections(sections);
                const blob = new Blob([text], { type: 'text/plain' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'ai-diagnosis.txt';
                a.click();
              }}
            >
              دانلود متن
            </Button>
            <Button
              variant="outlined"
              startIcon={<Iconify icon="solar:copy-bold" />}
              onClick={() => {
                const text = combineSections(sections);
                navigator.clipboard.writeText(text);
              }}
            >
              کپی همه
            </Button>
            {onSave && (
              <Button
                variant="contained"
                startIcon={<Iconify icon="solar:diskette-bold" />}
                onClick={onSave}
                sx={{ ml: 'auto !important' }}
              >
                ذخیره نهایی
              </Button>
            )}
          </Stack>
        </Card>
      )}
    </Stack>
  );
}

AIDiagnosisDisplay.propTypes = {
  diagnosis: PropTypes.string,
  onUpdate: PropTypes.func,
  onSave: PropTypes.func,
  readOnly: PropTypes.bool,
};

