/**
 * Clinical RAG Analysis Component
 * کامپوننت React برای نمایش تحلیل RAG بالینی
 */

import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Accordion from '@mui/material/Accordion';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import AccordionDetails from '@mui/material/AccordionDetails';
import AccordionSummary from '@mui/material/AccordionSummary';
import CircularProgress from '@mui/material/CircularProgress';

import { RealClinicalRAGService } from 'src/utils/rag/real-rag-service.ts';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export function ClinicalRAGAnalysis({ patientData, onAnalysisComplete }) {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [ragService] = useState(() => new RealClinicalRAGService());
  const [isInitialized, setIsInitialized] = useState(false);
  const [isInitializing, setIsInitializing] = useState(false);

  // Initialize RAG service once
  useEffect(() => {
    let mounted = true;
    
    async function initializeRAG() {
      setIsInitializing(true);
      try {
        // مسیر پوشه PDF‌ها - می‌تواند از environment variable یا config بیاید
        const pdfDirectory = import.meta.env.VITE_RAG_PDF_DIRECTORY || './knowledge-base/books';
        
        await ragService.initialize(pdfDirectory, {
          useEmbeddings: false, // بدون Embeddings (رایگان و سریع)
        });
        
        if (mounted) {
          setIsInitialized(true);
          console.log('✅ RAG Service initialized:', ragService.getStats());
        }
      } catch (err) {
        console.error('❌ Error initializing RAG service:', err);
        if (mounted) {
          setError('خطا در راه‌اندازی سرویس RAG. لطفاً مطمئن شوید که PDF‌ها در مسیر صحیح قرار دارند.');
        }
      } finally {
        if (mounted) {
          setIsInitializing(false);
        }
      }
    }

    initializeRAG();

    return () => {
      mounted = false;
    };
  }, [ragService]);

  const analyzePatient = useCallback(async () => {
    if (!patientData || !patientData.cephalometricMeasurements) {
      return;
    }

    if (!isInitialized) {
      setError('سرویس RAG در حال راه‌اندازی است. لطفاً صبر کنید...');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const result = await ragService.analyzePatient(patientData);
      setAnalysis(result);
      if (onAnalysisComplete) {
        onAnalysisComplete(result);
      }
    } catch (err) {
      setError(err.message || 'خطا در تحلیل بیمار');
    } finally {
      setLoading(false);
    }
  }, [patientData, onAnalysisComplete, ragService, isInitialized]);

  useEffect(() => {
    if (patientData && isInitialized) {
      analyzePatient();
    }
  }, [patientData, analyzePatient, isInitialized]);

  if (isInitializing) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>
              در حال راه‌اندازی سرویس RAG...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
            <CircularProgress />
            <Typography variant="body2" sx={{ ml: 2 }}>
              در حال تحلیل بیمار...
            </Typography>
          </Box>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">{error}</Alert>
          <Button onClick={analyzePatient} sx={{ mt: 2 }}>
            تلاش مجدد
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!analysis) {
    return (
      <Card>
        <CardContent>
          <Alert severity="info">
            برای شروع تحلیل، اطلاعات بیمار را وارد کنید.
          </Alert>
        </CardContent>
      </Card>
    );
  }

  return (
    <Stack spacing={3}>
      {/* تشخیص */}
      <Card>
        <CardContent>
          <Stack direction="row" spacing={2} alignItems="center" sx={{ mb: 2 }}>
            <Typography variant="h5">تشخیص</Typography>
            <Chip
              label={analysis.severity === 'severe' ? 'شدید' : analysis.severity === 'moderate' ? 'متوسط' : 'خفیف'}
              color={analysis.severity === 'severe' ? 'error' : analysis.severity === 'moderate' ? 'warning' : 'success'}
            />
          </Stack>
          <Typography variant="h6" color="primary">
            {analysis.diagnosis}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            {analysis.prognosis}
          </Typography>
        </CardContent>
      </Card>

      {/* مشکلات شناسایی شده */}
      {analysis.issues.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              مشکلات شناسایی شده
            </Typography>
            <Stack spacing={2}>
              {analysis.issues.map((issue, index) => (
                <Box key={index}>
                  <Stack direction="row" spacing={2} alignItems="center">
                    <Typography variant="subtitle1" fontWeight="bold">
                      {issue.parameter}
                    </Typography>
                    <Chip
                      label={`${issue.value.toFixed(1)}°`}
                      size="small"
                      color={issue.deviation > 5 ? 'error' : issue.deviation > 2.5 ? 'warning' : 'default'}
                    />
                    <Typography variant="caption" color="text.secondary">
                      (نرمال: {issue.normalRange.min}-{issue.normalRange.max}°)
                    </Typography>
                  </Stack>
                  <Typography variant="body2" sx={{ mt: 0.5 }}>
                    {issue.description}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {issue.clinicalSignificance}
                  </Typography>
                  {index < analysis.issues.length - 1 && <Divider sx={{ mt: 2 }} />}
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* طرح درمان */}
      {analysis.treatmentPlan.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              طرح درمان مبتنی بر شواهد
            </Typography>
            <Stack spacing={2}>
              {analysis.treatmentPlan.map((phase, index) => (
                <Accordion key={index}>
                  <AccordionSummary expandIcon={<Iconify icon="eva:arrow-ios-down-fill" />}>
                    <Stack direction="row" spacing={2} alignItems="center" sx={{ width: '100%' }}>
                      <Typography variant="subtitle1" fontWeight="bold">
                        فاز {index + 1}: {phase.phase}
                      </Typography>
                      <Chip label={phase.duration} size="small" />
                    </Stack>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Stack spacing={2}>
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                          روش‌های درمانی:
                        </Typography>
                        <Stack spacing={0.5}>
                          {phase.procedures.map((proc, i) => (
                            <Typography key={i} variant="body2">
                              • {proc}
                            </Typography>
                          ))}
                        </Stack>
                      </Box>
                      <Box>
                        <Typography variant="subtitle2" sx={{ mb: 1 }}>
                          اهداف:
                        </Typography>
                        <Stack spacing={0.5}>
                          {phase.goals.map((goal, i) => (
                            <Typography key={i} variant="body2">
                              • {goal}
                            </Typography>
                          ))}
                        </Stack>
                      </Box>
                      {phase.evidence.length > 0 && (
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            شواهد علمی:
                          </Typography>
                          <Stack spacing={0.5}>
                            {phase.evidence.map((ref, i) => (
                              <Typography key={i} variant="caption" color="text.secondary">
                                {i + 1}. {ref.authors} ({ref.year}): {ref.title}
                              </Typography>
                            ))}
                          </Stack>
                        </Box>
                      )}
                      <Typography variant="body2" color="text.secondary" sx={{ fontStyle: 'italic' }}>
                        {phase.rationale}
                      </Typography>
                    </Stack>
                  </AccordionDetails>
                </Accordion>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* توصیه‌ها */}
      {analysis.recommendations.length > 0 && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 2 }}>
              توصیه‌ها
            </Typography>
            <Stack spacing={2}>
              {analysis.recommendations.map((rec, index) => (
                <Box key={index}>
                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 1 }}>
                    <Chip
                      label={rec.priority === 'high' ? 'اولویت بالا' : rec.priority === 'medium' ? 'اولویت متوسط' : 'اولویت پایین'}
                      size="small"
                      color={rec.priority === 'high' ? 'error' : rec.priority === 'medium' ? 'warning' : 'default'}
                    />
                  </Stack>
                  <Typography variant="body2">{rec.recommendation}</Typography>
                  {rec.evidence.length > 0 && (
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5, display: 'block' }}>
                      بر اساس: {rec.evidence.map((e) => `${e.authors} (${e.year})`).join(', ')}
                    </Typography>
                  )}
                  {index < analysis.recommendations.length - 1 && <Divider sx={{ mt: 2 }} />}
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* رفرنس‌ها */}
      {analysis.references.length > 0 && (
        <Card>
          <CardContent>
            <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
              <Typography variant="h6">
                منابع و رفرنس‌های علمی
              </Typography>
              {analysis.references.some(ref => ref.isReal) && (
                <Alert severity="success" sx={{ py: 0.5, px: 1 }}>
                  <Typography variant="caption">
                    ✅ رفرنس‌های واقعی از PDF‌ها استخراج شده‌اند
                  </Typography>
                </Alert>
              )}
            </Stack>
            <Stack spacing={1}>
              {analysis.references.slice(0, 5).map((ref, index) => (
                <Box key={index}>
                  <Typography variant="body2" fontWeight="medium">
                    {index + 1}. {ref.authors} ({ref.year})
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {ref.title}
                  </Typography>
                  {ref.journal && (
                    <Typography variant="caption" color="text.secondary">
                      {ref.journal}
                    </Typography>
                  )}
                  {ref.chapter && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      {ref.chapter}
                    </Typography>
                  )}
                  {ref.page && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      صفحه: {ref.page}
                    </Typography>
                  )}
                  {ref.volume && (
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                      {ref.volume}
                    </Typography>
                  )}
                </Box>
              ))}
            </Stack>
          </CardContent>
        </Card>
      )}

      {/* توضیحات کامل */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2 }}>
            توضیحات کامل
          </Typography>
          <Box
            sx={{
              p: 2,
              bgcolor: 'background.neutral',
              borderRadius: 1,
              whiteSpace: 'pre-wrap',
            }}
          >
            <Typography variant="body2" component="div">
              {analysis.explanation.split('\n').map((line, i) => (
                <span key={i}>
                  {line}
                  <br />
                </span>
              ))}
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Stack>
  );
}

