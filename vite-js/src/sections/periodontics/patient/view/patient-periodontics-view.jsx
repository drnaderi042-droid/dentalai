import { useParams, useNavigate } from 'react-router-dom';
import { lazy, memo, useState, Suspense, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import CircularProgress from '@mui/material/CircularProgress';

import axios, { endpoints } from 'src/utils/axios';

import { DashboardContent } from 'src/layouts/dashboard';

import { Iconify } from 'src/components/iconify';
import { CustomTabs } from 'src/components/custom-tabs';

import { useAuthContext } from 'src/auth/hooks';

// Lazy load tab components for better performance
const PatientInfoTab = lazy(() => import('../components/patient-info-tab').then(module => ({ default: module.PatientInfoTab })));
const PeriodontalChartTab = lazy(() => import('../components/periodontal-chart-tab').then(module => ({ default: module.PeriodontalChartTab })));
const AnalysisTab = lazy(() => import('../components/analysis-tab').then(module => ({ default: module.AnalysisTab })));

// Loading fallback component
const TabLoadingFallback = memo(() => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400, py: 4 }}>
    <CircularProgress size={40} />
  </Box>
));
TabLoadingFallback.displayName = 'TabLoadingFallback';

// ----------------------------------------------------------------------

export function PatientPeriodonticsView() {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user } = useAuthContext();

  const [currentTab, setCurrentTab] = useState('info');
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Fetch patient data
  const fetchPatient = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${endpoints.patients}/${id}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setPatient(response.data.patient);
      setError(null);
    } catch (err) {
      console.error('Error fetching patient:', err);
      setError('خطا در بارگیری اطلاعات بیمار');
    } finally {
      setLoading(false);
    }
  }, [id, user]);

  useEffect(() => {
    if (user) {
      fetchPatient();
    }
  }, [user, fetchPatient]);

  const handleChangeTab = useCallback((event, newValue) => {
    setCurrentTab(newValue);
  }, []);

  const handleBackToList = () => {
    navigate('/dashboard/periodontics');
  };

  if (loading) {
    return (
      <DashboardContent>
        <Container>
          <Typography variant="h4" sx={{ textAlign: 'center', mt: 4 }}>
            در حال بارگیری...
          </Typography>
        </Container>
      </DashboardContent>
    );
  }

  if (error || !patient) {
    return (
      <DashboardContent>
        <Container>
          <Typography variant="h4" color="error" sx={{ textAlign: 'center', mt: 4 }}>
            {error || 'بیمار یافت نشد'}
          </Typography>
          <Box sx={{ textAlign: 'center', mt: 2 }}>
            <Button variant="contained" onClick={handleBackToList}>
              بازگشت به لیست
            </Button>
          </Box>
        </Container>
      </DashboardContent>
    );
  }

  return (
    <DashboardContent>
      <Container maxWidth="xl">
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Button
            startIcon={<Iconify icon="solar:arrow-left-bold-duotone" />}
            onClick={handleBackToList}
            sx={{ mb: 2 }}
          >
            بازگشت به لیست بیماران
          </Button>

          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <Box>
              <Typography variant="h4">
                {patient.firstName} {patient.lastName}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {patient.age} سال | {patient.phone || 'بدون شماره تماس'}
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* Tabs */}
        <Box
          sx={{
            mb: 3,
            display: 'flex',
            justifyContent: 'center',
          }}
        >
          <CustomTabs
            value={currentTab}
            onChange={handleChangeTab}
            variant="fullWidth"
            sx={{
              width: '100%',
              minWidth: 0,
              borderRadius: 1,
              mx: 'auto',
              maxWidth: 'none',
              '& .MuiTab-root': {
                flex: 1,
                minWidth: 0,
                maxWidth: 'none',
                zIndex: 2,
                position: 'relative',
              },
            }}
          >
            <Tab label="اطلاعات کلی" value="info" />
            <Tab label="چارت پریودونتال" value="chart" />
            <Tab label="آنالیز و طرح درمان" value="analysis" />
          </CustomTabs>
        </Box>

        {/* Tab Content - Using CSS display to keep components mounted for faster switching */}
        <Box>
          <Box
            sx={{
              display: currentTab === 'info' ? 'block' : 'none',
            }}
          >
            <Suspense fallback={<TabLoadingFallback />}>
              <PatientInfoTab patient={patient} onUpdate={fetchPatient} />
            </Suspense>
          </Box>

          <Box
            sx={{
              display: currentTab === 'chart' ? 'block' : 'none',
            }}
          >
            <Suspense fallback={<TabLoadingFallback />}>
              <PeriodontalChartTab 
                patient={patient} 
                onUpdate={fetchPatient}
                onNavigateToAnalysis={() => setCurrentTab('analysis')}
              />
            </Suspense>
          </Box>

          <Box
            sx={{
              display: currentTab === 'analysis' ? 'block' : 'none',
            }}
          >
            <Suspense fallback={<TabLoadingFallback />}>
              <AnalysisTab patient={patient} onUpdate={fetchPatient} />
            </Suspense>
          </Box>
        </Box>
      </Container>
    </DashboardContent>
  );
}

