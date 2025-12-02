import { toast } from 'sonner';
import React, { useRef, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';
import CircularProgress from '@mui/material/CircularProgress';

import axios, { endpoints } from 'src/utils/axios';
import { getImageUrl, getAiServiceUrl } from 'src/utils/url-helpers';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { OPGVisualizer } from 'src/components/opg-visualizer';

import { useAuthContext } from 'src/auth/hooks';

import ImageListItem from './image-list-item';

// ----------------------------------------------------------------------

// حداقل confidence برای نمایش detections (1%)
const MIN_CONFIDENCE_THRESHOLD = 0.01;

export function OPGView({
  initialImages = [],
  onEditCategory,
  onDeleteImage,
  patientId = null,
}) {
  const { user } = useAuthContext();
  const [imageFiles, setImageFiles] = useState([]);
  const [imageResults, setImageResults] = useState({}); // { imageId: { preview, detections, result } }
  const [isLoading, setIsLoading] = useState(false);
  const [loadingImages, setLoadingImages] = useState(new Set());
  const [error, setError] = useState(null);
  const [localInitialImages, setLocalInitialImages] = useState(initialImages);
  const imageFilesRef = useRef([]);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [deleteImageDialogOpen, setDeleteImageDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);

  // Helper function to get category label in Persian
  const getCategoryLabel = (category) => {
    const categoryLabels = {
      opg: 'OPG',
      panoramic: 'پانورامیک',
      general: 'کلی',
    };
    return categoryLabels[category] || (category || 'نامشخص');
  };

  const handleDropMultiFile = useCallback(async (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const newFiles = acceptedFiles.map((file) => {
        const previewUrl = URL.createObjectURL(file);
        return {
          file,
          preview: previewUrl,
          id: `temp-${Date.now()}-${Math.random()}`,
        };
      });

      setImageFiles((prev) => [...prev, ...newFiles]);
      imageFilesRef.current = [...imageFilesRef.current, ...newFiles];

      // Upload files to server if patientId exists
      if (patientId && user?.accessToken) {
        try {
          const formData = new FormData();
          newFiles.forEach(({ file }) => {
            formData.append('images', file);
          });
          formData.append('category', 'opg');

          await axios.post(`${endpoints.patients}/${patientId}/images`, formData, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });

          // Refresh images list
          const imagesResponse = await axios.get(`${endpoints.patients}/${patientId}/images`, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });

          const newImages = imagesResponse.data.images || [];
          const opgImages = newImages.filter(img => img.category === 'opg' || img.category === 'panoramic');
          setLocalInitialImages(opgImages);
        } catch (uploadError) {
          console.error('Error uploading images:', uploadError);
          toast.error('خطا در آپلود تصاویر');
        }
      }
    }
  }, [patientId, user?.accessToken]);

  // Update localInitialImages when initialImages prop changes
  useEffect(() => {
    setLocalInitialImages(initialImages);
  }, [initialImages]);

  // Auto-analyze when new images are added
  useEffect(() => {
    if (localInitialImages.length > 0 && selectedImageIndex < localInitialImages.length) {
      const currentImage = localInitialImages[selectedImageIndex];
      if (currentImage && !imageResults[currentImage.id]) {
        handleAnalyzeImage(currentImage);
      }
    }
  }, [localInitialImages, selectedImageIndex, handleAnalyzeImage, imageResults]);

  const convertImageToBase64 = (imageUrl) => 
    new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);
        canvas.toBlob((blob) => {
          const reader = new FileReader();
          reader.onloadend = () => resolve(reader.result);
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        });
      };
      img.onerror = reject;
      img.src = imageUrl;
    });

  const handleAnalyzeImage = useCallback(async (image) => {
    if (!image) return;

    setIsLoading(true);
    setLoadingImages((prev) => new Set([...prev, image.id]));
    setError(null);

    try {
      const imageUrl = getImageUrl(image.path);
      const base64Image = await convertImageToBase64(imageUrl);
      
      const unifiedAiServiceUrl = getAiServiceUrl(); // Port 5001
      const response = await fetch(`${unifiedAiServiceUrl}/detect-opg`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64Image,
          conf_threshold: confidenceThreshold,
        }),
      });

      if (!response.ok) {
        throw new Error(`OPG Service Error: ${response.status}`);
      }

      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Detection failed');
      }

      // Process detections
      const detectionResults = data.detections || [];
      const processedDetections = detectionResults
        .filter(det => det.confidence >= MIN_CONFIDENCE_THRESHOLD)
        .map((det, index) => ({
          id: index,
          classId: det.class_id,
          className: det.class_name,
          confidence: det.confidence,
          bbox: det.bbox,
        }));

      // Get image size
      const img = new Image();
      img.src = imageUrl;
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      const imageSize = {
        width: img.naturalWidth || img.width,
        height: img.naturalHeight || img.height,
      };

      setImageResults((prev) => ({
        ...prev,
        [image.id]: {
          preview: imageUrl,
          detections: processedDetections,
          result: data,
          imageSize,
        },
      }));

      toast.success(`${processedDetections.length} مورد در تصویر OPG تشخیص داده شد`);
    } catch (err) {
      console.error('Error analyzing OPG image:', err);
      setError(err.message || 'خطا در آنالیز تصویر OPG');
      toast.error(err.message || 'خطا در آنالیز تصویر OPG');
    } finally {
      setIsLoading(false);
      setLoadingImages((prev) => {
        const newSet = new Set(prev);
        newSet.delete(image.id);
        return newSet;
      });
    }
  }, [confidenceThreshold]);

  const handleDeleteImageClick = (image) => {
    setImageToDelete(image);
    setDeleteImageDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!imageToDelete || !onDeleteImage) return;

    try {
      await onDeleteImage(imageToDelete);
      setLocalInitialImages((prev) => prev.filter(img => img.id !== imageToDelete.id));
      setImageResults((prev) => {
        const newResults = { ...prev };
        delete newResults[imageToDelete.id];
        return newResults;
      });
      toast.success('تصویر با موفقیت حذف شد');
    } catch (err) {
      console.error('Error deleting image:', err);
      toast.error('خطا در حذف تصویر');
    } finally {
      setDeleteImageDialogOpen(false);
      setImageToDelete(null);
    }
  };

  const currentImage = localInitialImages[selectedImageIndex];
  const currentResult = currentImage ? imageResults[currentImage.id] : null;

  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Header */}
        <Stack direction="row" alignItems="center" justifyContent="space-between">
          <Stack direction="row" alignItems="center" spacing={2}>
            <Iconify icon="mdi:xray" width={40} />
            <Box>
              <Typography variant="h4">آنالیز OPG</Typography>
              <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                تشخیص لندمارک‌ها و ناهنجاری‌ها در تصاویر Panoramic OPG
              </Typography>
            </Box>
          </Stack>

          {/* Confidence Threshold */}
          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>حد آستانه اطمینان</InputLabel>
            <Select
              value={confidenceThreshold}
              label="حد آستانه اطمینان"
              onChange={(e) => setConfidenceThreshold(e.target.value)}
            >
              <MenuItem value={0.1}>10%</MenuItem>
              <MenuItem value={0.25}>25%</MenuItem>
              <MenuItem value={0.5}>50%</MenuItem>
              <MenuItem value={0.75}>75%</MenuItem>
            </Select>
          </FormControl>
        </Stack>

        {/* Error Alert */}
        {error && (
          <Alert severity="error" onClose={() => setError(null)}>
            {error}
          </Alert>
        )}

        {/* Image List */}
        <Card>
          <CardContent>
            <Stack spacing={2}>
              <Typography variant="h6">📷 تصاویر OPG</Typography>
              
              <Upload
                file={null}
                onDrop={handleDropMultiFile}
                accept={{ 'image/*': [] }}
                multiple
              />

              {localInitialImages.length > 0 && (
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                  {localInitialImages.map((image, index) => (
                    <ImageListItem
                      key={image.id}
                      item={image}
                      onEdit={onEditCategory}
                      onDelete={() => handleDeleteImageClick(image)}
                      getCategoryLabel={getCategoryLabel}
                    />
                  ))}
                </Box>
              )}
            </Stack>
          </CardContent>
        </Card>

        {/* Analysis Section */}
        {currentImage && (
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Stack direction="row" alignItems="center" justifyContent="space-between">
                  <Typography variant="h6">📊 نتایج آنالیز</Typography>
                  <Button
                    variant="contained"
                    onClick={() => handleAnalyzeImage(currentImage)}
                    disabled={isLoading || loadingImages.has(currentImage.id)}
                    startIcon={
                      isLoading && loadingImages.has(currentImage.id) ? (
                        <CircularProgress size={20} />
                      ) : (
                        <Iconify icon="carbon:ai-status" />
                      )
                    }
                  >
                    {isLoading && loadingImages.has(currentImage.id) ? 'در حال آنالیز...' : 'آنالیز با AI'}
                  </Button>
                </Stack>

                {isLoading && loadingImages.has(currentImage.id) && (
                  <LinearProgress />
                )}

                {currentResult ? (
                  <OPGVisualizer
                    imageUrl={currentResult.preview}
                    detections={currentResult.detections}
                    imageSize={currentResult.imageSize}
                    onDetectionsChange={(updatedDetections) => {
                      setImageResults((prev) => ({
                        ...prev,
                        [currentImage.id]: {
                          ...prev[currentImage.id],
                          detections: updatedDetections,
                        },
                      }));
                    }}
                  />
                ) : (
                  <Box sx={{ textAlign: 'center', py: 4 }}>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      برای نمایش نتایج، ابتدا آنالیز را انجام دهید
                    </Typography>
                  </Box>
                )}
              </Stack>
            </CardContent>
          </Card>
        )}
      </Stack>

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteImageDialogOpen} onClose={() => setDeleteImageDialogOpen(false)}>
        <DialogTitle>حذف تصویر</DialogTitle>
        <DialogContent>
          <Typography>آیا از حذف این تصویر مطمئن هستید؟</Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteImageDialogOpen(false)}>انصراف</Button>
          <Button onClick={handleConfirmDelete} color="error" variant="contained">
            حذف
          </Button>
        </DialogActions>
      </Dialog>
    </Container>
  );
}

