import { Helmet } from 'react-helmet-async';
import { useState, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import LinearProgress from '@mui/material/LinearProgress';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { OPGVisualizer } from 'src/components/opg-visualizer';

// OPG Detection Classes - انگلیسی
const OPG_CLASSES_EN = {
  0: 'Caries',
  1: 'Crown',
  2: 'Filling',
  3: 'Implant',
  4: 'Malaligned',
  5: 'Mandibular Canal',
  6: 'Missing teeth',
  7: 'Periapical lesion',
  8: 'Retained root',
  9: 'Root Canal Treatment',
  10: 'Root Piece',
  11: 'impacted tooth',
  12: 'maxillary sinus',
  13: 'Bone Loss',
  14: 'Fracture teeth',
  15: 'Permanent Teeth',
  16: 'Supra Eruption',
  17: 'TAD',
  18: 'abutment',
  19: 'attrition',
  20: 'bone defect',
  21: 'gingival former',
  22: 'metal band',
  23: 'orthodontic brackets',
  24: 'permanent retainer',
  25: 'post - core',
  26: 'plating',
  27: 'wire',
  28: 'Cyst',
  29: 'Root resorption',
  30: 'Primary teeth',
};

// ترجمه فارسی کلاس‌ها
const OPG_CLASSES_FA = {
  0: 'پوسیدگی',
  1: 'تاج',
  2: 'پرکردگی',
  3: 'ایمپلنت',
  4: 'نامرتب',
  5: 'کانال مندیبولار',
  6: 'دندان از دست رفته',
  7: 'ضایعه پری‌اپیکال',
  8: 'ریشه باقیمانده',
  9: 'درمان ریشه',
  10: 'تکه ریشه',
  11: 'دندان نهفته',
  12: 'سینوس ماگزیلا',
  13: 'تحلیل استخوان',
  14: 'شکستگی دندان',
  15: 'دندان دائمی',
  16: 'بیرون‌زدگی',
  17: 'TAD',
  18: 'اباتمنت',
  19: 'سایش',
  20: 'نقص استخوان',
  21: 'فرمر لثه',
  22: 'باند فلزی',
  23: 'براکت ارتودنسی',
  24: 'ریتینر دائمی',
  25: 'پست و کور',
  26: 'پلیت',
  27: 'سیم',
  28: 'کیست',
  29: 'جذب ریشه',
  30: 'دندان شیری',
};

// Helper function to get Persian class name
const getClassNameFA = (classId, classNameEN) => {
  if (classNameEN && OPG_CLASSES_FA[Object.keys(OPG_CLASSES_EN).find(key => OPG_CLASSES_EN[key] === classNameEN)]) {
    const key = Object.keys(OPG_CLASSES_EN).find(k => OPG_CLASSES_EN[k] === classNameEN);
    return OPG_CLASSES_FA[key] || classNameEN;
  }
  return OPG_CLASSES_FA[classId] || OPG_CLASSES_EN[classId] || 'نامشخص';
};

// ----------------------------------------------------------------------

export default function RadiologyPage() {
  const [imageFile, setImageFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [imageSize, setImageSize] = useState(null);
  const [detections, setDetections] = useState([]);

  const handleDrop = useCallback((acceptedFiles) => {
    const file = acceptedFiles[0];
    if (file) {
      setImageFile(file);
      const previewUrl = URL.createObjectURL(file);
      setImagePreview(previewUrl);
      setError(null);
      setResult(null);
      setDetections([]);
      
      // دریافت ابعاد واقعی تصویر
      const img = new Image();
      img.onload = () => {
        const naturalWidth = img.naturalWidth || img.width;
        const naturalHeight = img.naturalHeight || img.height;
        setImageSize({ width: naturalWidth, height: naturalHeight });
        console.log(`📏 ابعاد تصویر: ${naturalWidth} × ${naturalHeight}`);
      };
      img.src = previewUrl;
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setImageFile(null);
    setImagePreview(null);
    setResult(null);
    setDetections([]);
  }, []);

  const convertImageToBase64 = (file) => 
    new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });

  const handleAnalyze = async () => {
    if (!imageFile) {
      setError('لطفاً ابتدا یک تصویر OPG آپلود کنید');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);
    setDetections([]);

    const startTime = Date.now();

    try {
      // تبدیل تصویر به base64
      const base64Image = await convertImageToBase64(imageFile);
      
      console.log('🖼️ شروع آنالیز OPG...');
      console.log('   File name:', imageFile.name);
      console.log('   File size:', imageFile.size, 'bytes');
      console.log('   Image size:', imageSize);

      // ارسال درخواست به unified AI API server
      // استفاده از endpoint /detect-opg در unified_ai_api_server (پورت 5001)
      const response = await fetch('http://localhost:5001/detect-opg', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Image,
          conf_threshold: 0.25,
          iou_threshold: 0.45,
        }),
      });

      const endTime = Date.now();
      const processingTime = (endTime - startTime) / 1000;

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`OPG Service Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Detection failed');
      }

      console.log('✅ نتایج آنالیز OPG:', data);

      // پردازش نتایج
      const detectionResults = data.detections || [];
      
      // تبدیل detections به فرمت قابل نمایش
      const processedDetections = detectionResults.map((det, index) => ({
        id: index,
        classId: det.class_id || det.class,
        className: OPG_CLASSES_EN[det.class_id || det.class] || 'Unknown',
        confidence: det.confidence || det.conf,
        bbox: det.bbox || det.box || {
          x: det.x || det.x1 || 0,
          y: det.y || det.y1 || 0,
          width: (det.width || (det.x2 - det.x1) || 0),
          height: (det.height || (det.y2 - det.y1) || 0),
        },
        // برای segmentation
        segmentation: det.segmentation || det.mask || null,
      }));

      setDetections(processedDetections);

      const analysisResult = {
        success: true,
        detections: processedDetections,
        totalDetections: processedDetections.length,
        metadata: {
          processingTime: processingTime.toFixed(2),
          timestamp: new Date().toLocaleString('fa-IR'),
          model: data.metadata?.model || 'OPG YOLO Model',
          imageSize,
        },
        rawResponse: data,
      };

      setResult(analysisResult);
      
      console.log(`✅ ${processedDetections.length} مورد تشخیص داده شد`);

    } catch (err) {
      console.error('خطا در آنالیز OPG:', err);
      setError(err.message || 'خطا در آنالیز تصویر OPG');
      
      const errorResult = {
        success: false,
        error: err.message,
        metadata: {
          timestamp: new Date().toLocaleString('fa-IR'),
        },
      };
      
      setResult(errorResult);
    } finally {
      setIsLoading(false);
    }
  };


  return (
    <>
      <Helmet>
        <title>رادیولوژی - آنالیز OPG | DentalAI</title>
      </Helmet>

      <Container maxWidth="xl">
        <Stack spacing={3}>
          {/* Header */}
          <Stack direction="row" alignItems="center" spacing={2}>
            <Iconify icon="mdi:xray" width={40} />
            <Box>
              <Typography variant="h4">رادیولوژی - آنالیز OPG</Typography>

            </Box>
          </Stack>

          {/* Warning */}


          {/* Main Content */}
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
            {/* Left Panel - Upload */}
            <Stack spacing={3} sx={{ width: { xs: '100%', md: '400px' } }}>
              {/* Image Upload */}
              <Card>
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h6">📷 آپلود تصویر OPG</Typography>

                    <Upload
                      file={imageFile}
                      onDrop={handleDrop}
                      onDelete={handleRemoveFile}
                      accept={{ 'image/*': [] }}
                    />
                  </Stack>
                </CardContent>
              </Card>

              {/* Analyze Button */}
              <Button
                fullWidth
                size="large"
                variant="contained"
                color="primary"
                onClick={handleAnalyze}
                disabled={isLoading || !imageFile}
                startIcon={
                  isLoading ? (
                    <Iconify icon="line-md:loading-loop" />
                  ) : (
                    <Iconify icon="carbon:ai-status" />
                  )
                }
              >
                {isLoading ? 'در حال آنالیز...' : 'آنالیز با AI'}
              </Button>

            </Stack>

            {/* Right Panel - Results */}
            <Stack spacing={3} sx={{ flex: 1 }}>
              {/* Loading */}
              {isLoading && (
                <Card>
                  <CardContent>
                    <Stack spacing={2}>
                      <Stack direction="row" justifyContent="space-between">
                        <Typography variant="h6">⏳ در حال پردازش...</Typography>
                      </Stack>
                      <LinearProgress />
                      <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center' }}>
                        در حال تشخیص لندمارک‌ها و ناهنجاری‌ها در تصویر OPG...
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>
              )}

              {/* Error */}
              {error && (
                <Alert severity="error" onClose={() => setError(null)}>
                  <Typography variant="subtitle2">خطا</Typography>
                  <Typography variant="body2">{error}</Typography>
                </Alert>
              )}

              {/* Visualization */}
              {imagePreview && (
                <Card>
                  <CardContent>
                    <Stack spacing={2}>
                      {result && result.success && detections.length > 0 ? (
                        <>
                          <Typography variant="h6">📊 نمایش بصری و ویرایش نتایج تشخیص</Typography>
                          
                          <OPGVisualizer
                            imageUrl={imagePreview}
                            detections={detections}
                            imageSize={imageSize}
                            onDetectionsChange={(updatedDetections) => {
                              setDetections(updatedDetections);
                              setResult({
                                ...result,
                                detections: updatedDetections,
                                totalDetections: updatedDetections.length,
                              });
                            }}
                          />

                          {/* Summary */}
                          <Alert severity="success">
                            <Typography variant="body2">
                              ✅ {result.totalDetections} مورد تشخیص داده شد
                              {result.metadata?.processingTime && (
                                <> • زمان پردازش: {result.metadata.processingTime} ثانیه</>
                              )}
                            </Typography>
                          </Alert>
                        </>
                      ) : (
                        <>
                          <Typography variant="h6">📊 تصویر OPG</Typography>
                          
                          <OPGVisualizer
                            imageUrl={imagePreview}
                            detections={[]}
                            imageSize={imageSize}
                            onDetectionsChange={() => {}}
                          />
                        </>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
              )}


            </Stack>
          </Stack>
        </Stack>
      </Container>
    </>
  );
}

