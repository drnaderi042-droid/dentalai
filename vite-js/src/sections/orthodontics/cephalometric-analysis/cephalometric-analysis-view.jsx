import { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import { alpha } from '@mui/material/styles';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';

import axios, { endpoints } from 'src/utils/axios';

import { CONFIG } from 'src/config-global';
import { DashboardContent } from 'src/layouts/dashboard';

import { Iconify } from 'src/components/iconify';

import { useAuthContext } from 'src/auth/hooks';

// import { CephalometricCanvas } from './cephalometric-canvas'; // File is currently empty
import { AnalysisResultsPanel } from './analysis-results-panel';

export function CephalometricAnalysisView() {
  const { patientId } = useParams();
  const { user } = useAuthContext();
  const [patient, setPatient] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [measurements, setMeasurements] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPatient = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${endpoints.patients}/${patientId}`, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        setPatient(response.data.patient);
        
        // Try to fetch patient images separately
        try {
          const imagesResponse = await axios.get(`${endpoints.patients}/${patientId}/images`, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });
          
          const images = imagesResponse.data.images || [];
          console.log('📸 Patient images loaded:', images.length);
          
          // Find lateral ceph image with multiple possible category names
          const cephImage = images.find(
            img => img.category === 'lateral' ||
                   img.category === 'cephalometric' ||
                   img.category === 'cephalometry'
          );
          
          if (cephImage) {
            setSelectedImage(`${CONFIG.site.serverUrl}${cephImage.path}`);
            console.log('✅ Lateral ceph image found:', cephImage.originalName, 'Category:', cephImage.category);
          } else {
            console.log('⚠️ No lateral ceph image found in categories:', images.map(i => i.category).join(', '));
          }
        } catch (imgErr) {
          console.warn('Could not fetch patient images:', imgErr);
          
          // Fallback: try to get from patient.radiologyImages
          const cephImage = response.data.patient?.radiologyImages?.find(
            img => img.category === 'lateral' ||
                   img.category === 'cephalometric' ||
                   img.category === 'cephalometry'
          );
          if (cephImage) {
            setSelectedImage(`${CONFIG.site.serverUrl}${cephImage.path}`);
            console.log('✅ Lateral ceph image found from patient data');
          }
        }
      } catch (err) {
        console.error('Error fetching patient:', err);
      } finally {
        setLoading(false);
      }
    };

    if (user && patientId) {
      fetchPatient();
    }
  }, [user, patientId]);

  const handleMeasurementsChange = (newMeasurements) => {
    setMeasurements(newMeasurements);
  };

  const handleSaveAnalysis = async () => {
    try {
      await axios.post(`${endpoints.patients}/${patientId}/cephalometric-analysis`, {
        measurements,
        timestamp: new Date().toISOString(),
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      alert('تحلیل با موفقیت ذخیره شد');
    } catch (err) {
      console.error('Error saving analysis:', err);
      alert('خطا در ذخیره تحلیل');
    }
  };

  const handleImageUpload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('image', file);
    formData.append('category', 'cephalometric');

    try {
      const response = await axios.post(
        `${endpoints.patients}/${patientId}/radiology-images`,
        formData,
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
            'Content-Type': 'multipart/form-data',
          },
        }
      );
      
      const imagePath = response.data.image?.path;
      if (imagePath) {
        setSelectedImage(`${CONFIG.site.serverUrl}${imagePath}`);
      }
    } catch (err) {
      console.error('Error uploading image:', err);
      alert('خطا در آپلود تصویر');
    }
  };

  if (loading) {
    return (
      <DashboardContent>
        <Container>
          <Typography variant="h4" sx={{ textAlign: 'center', mt: 4 }}>
            بارگیری...
          </Typography>
        </Container>
      </DashboardContent>
    );
  }

  return (
    <DashboardContent>
      <Container maxWidth="xl">
        <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 3 }}>
          <Box>
            <Typography variant="h4" gutterBottom>
              تحلیل سفالومتری
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {patient?.firstName} {patient?.lastName} - {patient?.age} سال
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={2}>
            <Button
              variant="outlined"
              startIcon={<Iconify icon="eva:arrow-back-fill" />}
              onClick={() => window.history.back()}
            >
              بازگشت
            </Button>
            <Button
              fullWidth
              size="large"
              variant="contained"
              startIcon={<Iconify icon="eva:save-fill" />}
              onClick={handleSaveAnalysis}
              disabled={Object.keys(measurements).length === 0}
            >
              ذخیره تحلیل
            </Button>
          </Stack>
        </Stack>

        {!selectedImage ? (
          <Card
            sx={{
              p: 8,
              textAlign: 'center',
              borderStyle: 'dashed',
              borderWidth: 2,
              borderColor: 'divider',
              bgcolor: (theme) => alpha(theme.palette.grey[500], 0.04),
            }}
          >
            <Iconify icon="solar:upload-bold" width={64} sx={{ color: 'text.disabled', mb: 2 }} />
            <Typography variant="h6" gutterBottom>
              آپلود تصویر سفالومتری
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              لطفاً تصویر رادیوگرافی جانبی جمجمه را آپلود کنید
            </Typography>
            <label htmlFor="ceph-image-upload">
              <input
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                id="ceph-image-upload"
                onChange={handleImageUpload}
              />
              <Button variant="contained" component="span" startIcon={<Iconify icon="eva:cloud-upload-fill" />}>
                انتخاب تصویر
              </Button>
            </label>
          </Card>
        ) : (
          <Grid container spacing={3}>
            <Grid item xs={12} lg={8}>
              <Card sx={{ p: 2 }}>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                  <Typography variant="h6">
                    کانوس تحلیل
                  </Typography>
                  <label htmlFor="change-ceph-image">
                    <input
                      type="file"
                      accept="image/*"
                      style={{ display: 'none' }}
                      id="change-ceph-image"
                      onChange={handleImageUpload}
                    />
                    <Button
                      component="span"
                      size="small"
                      startIcon={<Iconify icon="eva:image-outline" />}
                    >
                      تغییر تصویر
                    </Button>
                  </label>
                </Stack>
                {/* <CephalometricCanvas
                  imageUrl={selectedImage}
                  onMeasurementsChange={handleMeasurementsChange}
                /> */}
                <Box sx={{ p: 3, textAlign: 'center', color: 'text.secondary' }}>
                  CephalometricCanvas component is currently being refactored
                </Box>
              </Card>
            </Grid>

            <Grid item xs={12} lg={4}>
              <AnalysisResultsPanel measurements={measurements} />
            </Grid>
          </Grid>
        )}

        {selectedImage && (
          <Card sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              راهنمای استفاده
            </Typography>
            <Stack spacing={1.5}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Iconify icon="eva:checkmark-circle-2-fill" color="success.main" />
                <Typography variant="body2">
                  نقاط آناتومیکی را با کلیک روی تصویر مشخص کنید
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} alignItems="center">
                <Iconify icon="eva:checkmark-circle-2-fill" color="success.main" />
                <Typography variant="body2">
                  برای بزرگنمایی از چرخ ماوس استفاده کنید
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} alignItems="center">
                <Iconify icon="eva:checkmark-circle-2-fill" color="success.main" />
                <Typography variant="body2">
                  برای حرکت تصویر، کلید Ctrl را نگه دارید و ماوس را حرکت دهید
                </Typography>
              </Stack>
              <Stack direction="row" spacing={1} alignItems="center">
                <Iconify icon="eva:checkmark-circle-2-fill" color="success.main" />
                <Typography variant="body2">
                  نتایج تحلیل به صورت خودکار محاسبه و نمایش داده می‌شوند
                </Typography>
              </Stack>
            </Stack>
          </Card>
        )}
      </Container>
    </DashboardContent>
  );
}
