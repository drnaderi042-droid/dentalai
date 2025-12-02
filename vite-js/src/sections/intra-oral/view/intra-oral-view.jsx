import { toast } from 'sonner';
import React, { useRef, useMemo, useState, useEffect, useCallback } from 'react';

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
import IconButton from '@mui/material/IconButton';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import CircularProgress from '@mui/material/CircularProgress';

import axios, { endpoints } from 'src/utils/axios';

import { CONFIG } from 'src/config-global';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { DetectionVisualizer } from 'src/components/detection-visualizer/detection-visualizer';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

// حداقل confidence برای نمایش detections (1%)
const MIN_CONFIDENCE_THRESHOLD = 0.01;

export function IntraOralView({
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
  // Local state for initialImages to handle deletions
  const [localInitialImages, setLocalInitialImages] = useState(initialImages);
  // Ref to access current imageFiles in callbacks
  const imageFilesRef = useRef([]);
  const [selectedModel, setSelectedModel] = useState('fyp2');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [showConfidenceSettings, setShowConfidenceSettings] = useState(false);
  const [initialImagesLoaded, setInitialImagesLoaded] = useState(false);
  const [lastSavedAnalysis, setLastSavedAnalysis] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [analysisToDelete, setAnalysisToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  // State for delete image confirmation dialog
  const [deleteImageDialogOpen, setDeleteImageDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);

  // Helper function to get category label in Persian
  const getCategoryLabel = (category) => {
    const categoryLabels = {
      profile: 'پروفایل',
      frontal: 'فرونتال',
      panoramic: 'پانورامیک',
      lateral: 'لترال سفالومتری',
      occlusal: 'اکلوزال',
      'lateral-intraoral': 'لترال راست دهان',
      'lateral-intraoral-left': 'لترال چپ دهان',
      'frontal-intraoral': 'فرونتال داخل دهان',
      // Legacy categories for backward compatibility
      intraoral: 'داخل دهانی',
      general: 'کلی',
      cephalometric: 'سفالومتری',
      cephalometry: 'سفالومتری',
      intra: 'داخل دهانی',
      opg: 'OPG',
    };
    return categoryLabels[category] || (category || 'نامشخص');
  };

  // Available models for selection - فقط بهترین مدل‌ها
  const availableModels = [
    { value: 'fyp2', label: 'FYP2', description: 'بهترین مدل FYP2 - تشخیص دقیق canine/molar Class I/II/III با subdivisions' },
    { value: 'lateral', label: 'Lateral', description: 'بهترین مدل Lateral - تشخیص Class I/II/III + مشکلات دندانی' },
  ];

  const handleDropMultiFile = useCallback(async (acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      // Add files to local state first for immediate UI feedback
      const newFiles = acceptedFiles.map((file) => {
        const imageId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const previewUrl = URL.createObjectURL(file);
        return {
          id: imageId,
          file,
          preview: previewUrl,
          uploading: true,
        };
      });
      
      setImageFiles((prev) => [...prev, ...newFiles]);
      
      // Initialize results for new images
      const newResults = {};
      newFiles.forEach((img) => {
        newResults[img.id] = {
          preview: img.preview,
          detections: [],
          result: null,
        };
      });
      setImageResults((prev) => ({ ...prev, ...newResults }));
      setError(null);

      // Upload files to server if patientId exists
      if (patientId && user?.accessToken) {
        try {
          const formData = new FormData();
          acceptedFiles.forEach((file) => {
            formData.append('images', file);
          });
          // Use 'intraoral' as default category, can be changed via onEditCategory
          formData.append('category', 'intraoral');

          const uploadResponse = await axios.post(`${endpoints.patients}/${patientId}/images`, formData, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
              'Content-Type': 'multipart/form-data',
            },
          });

          console.log('✅ Files uploaded successfully:', uploadResponse.data);

          // Refresh images from server
          const imagesResponse = await axios.get(`${endpoints.patients}/${patientId}/images`, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });

          const serverImages = imagesResponse.data.images || [];
          const uploadedImages = serverImages.filter(img => 
            img.category === 'intraoral' || 
            img.category === 'lateral-intraoral' ||
            img.category === 'lateral-intraoral-left' || 
            img.category === 'frontal-intraoral' ||
            img.category === 'intra'
          );

          // Update localInitialImages with new server images
          setLocalInitialImages((prev) => {
            // Merge with existing, avoiding duplicates
            const existingIds = new Set(prev.map(img => img.id));
            const newServerImages = uploadedImages.filter(img => !existingIds.has(img.id));
            return [...prev, ...newServerImages];
          });

          // Update imageFiles with server IDs
          setImageFiles((prev) => prev.map((localFile) => {
              if (localFile.uploading) {
                // Find matching server image by name/size
                const serverImage = uploadedImages.find((serverImg) => {
                  const localFileName = localFile.file.name.toLowerCase();
                  const serverFileName = (serverImg.originalName || serverImg.name || '').toLowerCase();
                  return serverFileName.includes(localFileName.substring(0, 10)) || 
                         serverFileName === localFileName;
                });

                if (serverImage) {
                  return {
                    ...localFile,
                    serverId: serverImage.id,
                    uploading: false,
                  };
                }
              }
              return localFile;
          }));

          toast.success(`${acceptedFiles.length} تصویر با موفقیت آپلود شد`);
        } catch (error) {
          console.error('❌ Error uploading files:', error);
          const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
          toast.error(`خطا در آپلود تصاویر: ${errorMsg}`);
          
          // Mark files as failed
          setImageFiles((prev) => prev.map((file) => {
              if (file.uploading) {
                return { ...file, uploading: false, uploadFailed: true };
              }
              return file;
          }));
        }
      }
    }
  }, [patientId, user?.accessToken]);

  // Update localInitialImages when initialImages prop changes
  useEffect(() => {
    setLocalInitialImages(initialImages);
  }, [initialImages]);

  // Keep imageFilesRef in sync with imageFiles
  useEffect(() => {
    imageFilesRef.current = imageFiles;
  }, [imageFiles]);

  // Create a key from initialImages to detect changes
  const imagesKey = useMemo(() => initialImages?.map(img => img.path || img.id).join(',') || '', [initialImages]);

  // Load analysis history for the patient
  const loadAnalysisHistory = useCallback(async () => {
    console.log('📚 [IntraOralView] loadAnalysisHistory called for patient:', patientId);

    if (!patientId) return;
    
    setIsLoadingHistory(true);
    try {
      console.log('🔍 [IntraOralView] Fetching patient data from API...');
      const res = await axios.get(`${endpoints.patients}/${patientId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      // Parse analysis history from intraOralAnalysis field
      const patientData = res.data?.patient || res.data;
      console.log('📊 [IntraOralView] Patient data received:', {
        hasIntraOralAnalysis: !!patientData.intraOralAnalysis,
        intraOralAnalysisType: typeof patientData.intraOralAnalysis,
        intraOralAnalysisLength: patientData.intraOralAnalysis?.length || 0
      });

      let analyses = [];

      if (patientData.intraOralAnalysis) {
        try {
          const data = patientData.intraOralAnalysis;
          
          // Handle both object and string formats
          if (typeof data === 'object') {
            if (Array.isArray(data)) {
              analyses = data;
            } else {
              const { analyses: dataAnalyses } = data;
              if (dataAnalyses && Array.isArray(dataAnalyses)) {
                analyses = dataAnalyses;
              }
            }
          } else if (typeof data === 'string') {
            const trimmedData = data.trim();
            if (trimmedData.startsWith('{') || trimmedData.startsWith('[')) {
              const parsed = JSON.parse(trimmedData);
              if (Array.isArray(parsed)) {
                analyses = parsed;
              } else {
                const { analyses: parsedAnalyses } = parsed;
                if (parsedAnalyses && Array.isArray(parsedAnalyses)) {
                  analyses = parsedAnalyses;
                }
              }
            }
          }
        } catch (parseError) {
          console.error('❌ [IntraOralView] Failed to parse intraoral analysis:', parseError);
        }
      } else {
        console.log('⚠️ [IntraOralView] No intraOralAnalysis field in patient data');
      }

      console.log('📋 [IntraOralView] Final analysis history:', analyses.length, 'entries');
      setAnalysisHistory(analyses);
      
      // Auto-select the latest analysis only if not already selected
      if (analyses.length > 0 && selectedAnalysisIndex === null) {
        setSelectedAnalysisIndex(analyses.length - 1);
      }
    } catch (err) {
      console.error('Failed to load analysis history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [patientId, user?.accessToken]); // Remove selectedAnalysisIndex from dependencies to prevent infinite loop

  // Reset loaded flag when initialImages change (e.g., different patient or new images added)
  useEffect(() => {
    if (imagesKey) {
      setInitialImagesLoaded(false);
    }
  }, [imagesKey]);

  // Load initial images from props
  useEffect(() => {
    if (initialImages && initialImages.length > 0 && !initialImagesLoaded) {
      const loadInitialImages = () => {
        const loadedImages = [];
        
        for (const image of initialImages) {
          try {
            // Get image URL - ensure it's a complete URL
            let imageUrl = image.path;
            if (!imageUrl?.startsWith('http')) {
              // If path doesn't start with http, prepend server URL
              const baseUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
              // Ensure path starts with /
              const path = image.path?.startsWith('/') ? image.path : `/${image.path}`;
              imageUrl = `${baseUrl}${path}`;
            }
            
            const imageId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            
            // For server images, use file-like object with path and preview
            // This allows FileThumbnail to display the image correctly after page refresh
            const fileLikeObject = {
              name: image.originalName || image.name || `image-${Date.now()}.jpg`,
              path: image.path,
              preview: imageUrl, // Complete URL for preview
              type: 'image/jpeg',
              size: image.size || 0,
              lastModified: image.createdAt ? new Date(image.createdAt).getTime() : Date.now(),
            };
            
            loadedImages.push({
              id: imageId,
              serverId: image.id || null,
              file: fileLikeObject, // Use file-like object instead of File object
              preview: imageUrl, // Complete URL for thumbnail display
              originalName: image.originalName || image.name || `image-${Date.now()}.jpg`,
            });
          } catch (err) {
            console.error(`Error loading image ${image.path}:`, err);
          }
        }
        
        if (loadedImages.length > 0) {
          setImageFiles(loadedImages);

          // Initialize results for loaded images
          const newResults = {};
          loadedImages.forEach((img) => {
            newResults[img.id] = {
              preview: img.preview,
              detections: [],
              result: null,
            };
          });
          setImageResults(newResults);
          setInitialImagesLoaded(true);
        } else {
          // If no images loaded, still mark as loaded to prevent retry
          setInitialImagesLoaded(true);
        }
      };
      
      loadInitialImages();
    } else if (!initialImages || initialImages.length === 0) {
      // Reset when initialImages is empty
      setInitialImagesLoaded(false);
      setImageFiles([]);
      setImageResults({});
    }
  }, [initialImages, initialImagesLoaded]);

  // Load analysis history when patientId changes
  // Use ref to prevent multiple calls
  const loadHistoryCalledRef = useRef(null);
  useEffect(() => {
    const key = `${patientId}-${user?.accessToken}`;
    if (patientId && user?.accessToken && loadHistoryCalledRef.current !== key) {
      loadHistoryCalledRef.current = key;
      loadAnalysisHistory();
    }
  }, [patientId, user?.accessToken, loadAnalysisHistory]);

  // Load selected analysis data when selectedAnalysisIndex changes
  useEffect(() => {
    if (analysisHistory.length > 0 && selectedAnalysisIndex !== null && selectedAnalysisIndex < analysisHistory.length) {
      const selectedAnalysis = analysisHistory[selectedAnalysisIndex];
      
      if (selectedAnalysis && selectedAnalysis.analyses && Array.isArray(selectedAnalysis.analyses)) {
        // Map saved analyses to loaded images by serverId
        const mappedResults = {};
        selectedAnalysis.analyses.forEach((item) => {
          if (!item || !item.serverImageId) return;
          // find local image id by serverId
          const localImg = imageFiles.find(li => li.serverId && String(li.serverId) === String(item.serverImageId));
          if (localImg && item.result) {
            mappedResults[localImg.id] = {
              preview: localImg.preview,
              detections: item.result?.detections || [],
              result: item.result || null,
            };
          }
        });

        // Update imageResults with selected analysis data
        if (Object.keys(mappedResults).length > 0) {
          setImageResults((prev) => ({ ...prev, ...mappedResults }));
        }
      }
    }
  }, [selectedAnalysisIndex, analysisHistory, imageFiles]);

  // Handle delete image from server
  const handleDeleteImageFromServer = useCallback(async (image) => {
    console.log('🗑️ [IntraOralView] handleDeleteImageFromServer called:', {
      patientId,
      imageId: image?.id,
      image: image,
    });

    if (!patientId) {
      console.warn('⚠️ [IntraOralView] Cannot delete image: missing patientId');
      toast.error('شناسه بیمار یافت نشد');
      return;
    }

    if (!image?.id) {
      console.warn('⚠️ [IntraOralView] Cannot delete image: missing image.id', image);
      toast.error('شناسه تصویر یافت نشد');
      return;
    }

    try {
      const response = await axios.delete(`${endpoints.patients}/${patientId}/images`, {
        data: { imageId: image.id },
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
      });
      
      console.log('✅ [IntraOralView] Image deleted successfully:', response.data);
      toast.success('تصویر با موفقیت حذف شد');
      
      // Update local initial images list
      setLocalInitialImages((prev) => {
        const filtered = prev.filter((img) => img.id !== image.id);
        console.log('📝 [IntraOralView] Updated localInitialImages:', {
          before: prev.length,
          after: filtered.length,
          removedId: image.id,
        });
        return filtered;
      });
      
      // Also remove from imageFiles if exists
      setImageFiles((prev) => {
        const fileToRemove = prev.find(f => f.serverId === image.id);
        if (fileToRemove) {
          if (fileToRemove.preview && fileToRemove.preview.startsWith('blob:')) {
            try {
              URL.revokeObjectURL(fileToRemove.preview);
            } catch (error) {
              console.warn('⚠️ [IntraOralView] Error revoking blob URL:', error);
            }
          }
          return prev.filter(f => f.serverId !== image.id);
        }
        return prev;
      });
      
      // Remove from imageResults - find and remove results for files matching the deleted image
      setImageResults((prev) => {
        const updated = { ...prev };
        // Find all result keys that belong to files with this serverId
        const currentFiles = imageFilesRef.current;
        Object.keys(updated).forEach((key) => {
          const file = currentFiles.find(f => f.id === key);
          if (file && file.serverId === image.id) {
            delete updated[key];
          }
        });
        return updated;
      });
      
      // If onDeleteImage prop is provided, call it
      if (onDeleteImage) {
        onDeleteImage(image);
      }
    } catch (error) {
      console.error('❌ [IntraOralView] Error deleting image:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
      toast.error(`خطا در حذف تصویر: ${errorMsg}`);
    }
  }, [patientId, user?.accessToken, onDeleteImage]);

  const handleRemoveFile = useCallback((fileToRemove) => {
    // Remove by file object - find by file reference, name, path, or size
    if (!fileToRemove) return;
    
    // Use imageFilesRef to get current state
    const currentFiles = imageFilesRef.current;
    const imageToRemove = currentFiles.find((img) => {
      // Try exact match first (same object reference)
      if (img.file === fileToRemove) return true;
      
      // Match by path (for server images)
      if (
        img.file?.path && 
        fileToRemove?.path && 
        img.file.path === fileToRemove.path
      ) return true;
      
      // Match by preview URL
      if (
        img.preview && 
        fileToRemove?.preview && 
        img.preview === fileToRemove.preview
      ) return true;
      
      // Fallback: match by name and size (for new File objects)
      if (
        img.file && 
        fileToRemove && 
        img.file.name === fileToRemove.name && 
        img.file.size === fileToRemove.size
      ) return true;
      
      return false;
    });
    
    if (imageToRemove) {
      // Cleanup preview URL
      if (imageToRemove.preview && imageToRemove.preview.startsWith('blob:')) {
        URL.revokeObjectURL(imageToRemove.preview);
      }
      
      const imageIdToRemove = imageToRemove.id;
      
      // Remove from files
      setImageFiles((prev) => prev.filter((img) => img.id !== imageIdToRemove));
      
      // Remove from results
      setImageResults((prev) => {
        const updated = { ...prev };
        delete updated[imageIdToRemove];
        return updated;
      });

      // Also remove from server if serverId exists
      if (imageToRemove.serverId && patientId && user?.accessToken) {
        // Call handleDeleteImageFromServer directly without adding as dependency
        // to avoid circular dependency
        axios.delete(`${endpoints.patients}/${patientId}/images`, {
          data: { imageId: imageToRemove.serverId },
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
            'Content-Type': 'application/json',
          },
        }).then(() => {
          // Update localInitialImages
          setLocalInitialImages((prev) => prev.filter((img) => img.id !== imageToRemove.serverId));
          toast.success('تصویر با موفقیت حذف شد');
        }).catch((error) => {
          console.error('Error deleting image from server:', error);
          const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
          toast.error(`خطا در حذف تصویر از سرور: ${errorMsg}`);
        });
      }
    }
  }, [patientId, user?.accessToken]);

  const handleRemoveAllFiles = useCallback(() => {
    // Cleanup all preview URLs
    setImageFiles((prevFiles) => {
      prevFiles.forEach((img) => {
        if (img.preview && img.preview.startsWith('blob:')) {
          URL.revokeObjectURL(img.preview);
        }
      });
      return [];
    });
    setImageResults({});
  }, []);


  // محاسبه IoU (Intersection over Union) بین دو bounding box
  // detections from backend have x1, y1, x2, y2 directly (not nested in bbox object)
  const calculateIoU = (det1, det2) => {
    // Handle both formats: direct properties or nested bbox
    const bbox1 = det1.bbox || det1;
    const bbox2 = det2.bbox || det2;
    
    const x1 = Math.max(bbox1.x1, bbox2.x1);
    const y1 = Math.max(bbox1.y1, bbox2.y1);
    const x2 = Math.min(bbox1.x2, bbox2.x2);
    const y2 = Math.min(bbox1.y2, bbox2.y2);

    if (x2 < x1 || y2 < y1) return 0;

    const intersection = (x2 - x1) * (y2 - y1);
    const area1 = (bbox1.x2 - bbox1.x1) * (bbox1.y2 - bbox1.y1);
    const area2 = (bbox2.x2 - bbox2.x1) * (bbox2.y2 - bbox2.y1);
    const union = area1 + area2 - intersection;

    return union > 0 ? intersection / union : 0;
  };

  // Non-Maximum Suppression برای حذف bounding box های تکراری با همان class_name
  const applyNMS = (detections, iouThreshold = 0.5) => {
    if (!detections || detections.length === 0) return [];

    console.log(`[NMS] Starting NMS with threshold: ${iouThreshold}, total detections: ${detections.length}`);

    // Group by class name
    const detectionsByClass = {};
    detections.forEach((det) => {
      if (!detectionsByClass[det.class_name]) {
        detectionsByClass[det.class_name] = [];
      }
      detectionsByClass[det.class_name].push(det);
    });

    console.log(`[NMS] Grouped into ${Object.keys(detectionsByClass).length} classes:`, Object.keys(detectionsByClass));

    const selected = [];

    // Apply NMS for each class separately
    Object.keys(detectionsByClass).forEach((className) => {
      const classDetections = detectionsByClass[className];
      
      // Sort by confidence (highest first) - باکس با confidence بالاتر نگه داشته می‌شود
      const sortedDetections = [...classDetections].sort((a, b) => b.confidence - a.confidence);
      
      const suppressed = new Set();

      for (let i = 0; i < sortedDetections.length; i++) {
        if (suppressed.has(i)) continue;

        const current = sortedDetections[i];
        selected.push(current);

        // Suppress overlapping detections of the same class
        for (let j = i + 1; j < sortedDetections.length; j++) {
          if (suppressed.has(j)) continue;

          const other = sortedDetections[j];
          
          // Get bbox from detection (handle both formats: direct properties or nested bbox)
          const currentBbox = current.bbox || current;
          const otherBbox = other.bbox || other;
          
          // Calculate IoU
          const iou = calculateIoU(current, other);
          
          // بررسی اینکه آیا یکی در داخل دیگری است
          const isOneInsideOther = 
            (currentBbox.x1 >= otherBbox.x1 && currentBbox.y1 >= otherBbox.y1 && 
             currentBbox.x2 <= otherBbox.x2 && currentBbox.y2 <= otherBbox.y2) ||
            (otherBbox.x1 >= currentBbox.x1 && otherBbox.y1 >= currentBbox.y1 && 
             otherBbox.x2 <= currentBbox.x2 && otherBbox.y2 <= currentBbox.y2);
          
          // باکس فاصله مرکز دو باکس
          const center1X = (currentBbox.x1 + currentBbox.x2) / 2;
          const center1Y = (currentBbox.y1 + currentBbox.y2) / 2;
          const center2X = (otherBbox.x1 + otherBbox.x2) / 2;
          const center2Y = (otherBbox.y1 + otherBbox.y2) / 2;
          
          const distance = Math.sqrt(
            (center2X - center1X)**2 + (center2Y - center1Y)**2
          );
          
          // محاسبه میانگین اندازه باکس
          const avgWidth = ((currentBbox.x2 - currentBbox.x1) + (otherBbox.x2 - otherBbox.x1)) / 2;
          const avgHeight = ((currentBbox.y2 - currentBbox.y1) + (otherBbox.y2 - otherBbox.y1)) / 2;
          const avgSize = Math.sqrt(avgWidth * avgHeight);
          
          // حذف اگر:
          // 1. IoU بیشتر از threshold باشد (همپوشانی زیاد) - این اصلی‌ترین شرط است
          // 2. یکی کاملاً در داخل دیگری باشد (nested boxes) و IoU بالا باشد
          // برای Class I/II/III، باید threshold بالاتری استفاده کنیم تا detections ضعیف حذف نشوند
          const isClassDetection = current.class_name && (
            current.class_name.toLowerCase().includes('class i') ||
            current.class_name.toLowerCase().includes('class ii') ||
            current.class_name.toLowerCase().includes('class iii')
          );
          
          // برای Class I/II/III، threshold بالاتری استفاده می‌کنیم (0.7 به جای 0.5)
          const effectiveThreshold = isClassDetection ? Math.max(iouThreshold, 0.7) : iouThreshold;
          
          // فقط اگر IoU خیلی بالا باشد یا یکی کاملاً داخل دیگری باشد
          const shouldSuppress = iou > effectiveThreshold || (isOneInsideOther && iou > 0.5);
          
          if (shouldSuppress) {
            console.log(`[NMS] Suppressing ${other.class_name} (conf: ${(other.confidence * 100).toFixed(1)}%) - IoU: ${iou.toFixed(3)}, effectiveThreshold: ${effectiveThreshold.toFixed(2)}, nested: ${isOneInsideOther}`);
            suppressed.add(j);
          }
        }
      }
      
      console.log(`[NMS] Class '${className}': ${classDetections.length} -> ${selected.filter(d => d.class_name === className).length} (removed ${classDetections.length - selected.filter(d => d.class_name === className).length})`);
    });

    console.log(`[NMS] Final result: ${detections.length} -> ${selected.length} (removed ${detections.length - selected.length})`);
    return selected;
  };

  // تابع کمکی برای فیلتر detections
  const filterDetections = useCallback((rawDetections, modelType) => {
    // فیلتر detections با confidence کمتر از 1%
    const confidenceFilteredDetections = rawDetections.filter(det => {
      const conf = typeof det.confidence === 'number' ? det.confidence : parseFloat(det.confidence || 0);
      return conf >= MIN_CONFIDENCE_THRESHOLD;
    });
    
    // فیلتر کلاس‌های تکراری: برای FYP2 فقط یک canine و یک molar، برای Lateral فقط یک کلاس
    let filteredDetections = [];
    
    if (modelType === 'fyp2') {
      // برای FYP2: فقط یک canine و یک molar با بیشترین confidence
      const canineDetections = confidenceFilteredDetections.filter(det => 
        det.class_name.toLowerCase().includes('canine')
      );
      const molarDetections = confidenceFilteredDetections.filter(det => 
        det.class_name.toLowerCase().includes('molar')
      );
      const otherDetections = confidenceFilteredDetections.filter(det => 
        !det.class_name.toLowerCase().includes('canine') && 
        !det.class_name.toLowerCase().includes('molar')
      );
      
      // انتخاب canine با بیشترین confidence
      if (canineDetections.length > 0) {
        const bestCanine = canineDetections.reduce((best, current) => 
          current.confidence > best.confidence ? current : best
        );
        filteredDetections.push(bestCanine);
      }
      
      // انتخاب molar با بیشترین confidence
      if (molarDetections.length > 0) {
        const bestMolar = molarDetections.reduce((best, current) => 
          current.confidence > best.confidence ? current : best
        );
        filteredDetections.push(bestMolar);
      }
      
      // اضافه کردن سایر کلاس‌ها (غیر canine و molar)
      filteredDetections = filteredDetections.concat(otherDetections);
    } else if (modelType === 'lateral') {
      // برای Lateral: فقط یک کلاس با بیشترین confidence
      if (confidenceFilteredDetections.length > 0) {
        const bestDetection = confidenceFilteredDetections.reduce((best, current) => 
          current.confidence > best.confidence ? current : best
        );
        filteredDetections = [bestDetection];
      }
    } else {
      // برای سایر مدل‌ها: بدون فیلتر خاص
      filteredDetections = confidenceFilteredDetections;
    }
    
    // Transform detections to match DetectionVisualizer format
    const transformedDetections = filteredDetections.map((det) => {
      if (det.bbox) {
        return det;
      }
      return {
        ...det,
        bbox: {
          x1: det.x1,
          y1: det.y1,
          x2: det.x2,
          y2: det.y2,
        },
      };
    });
    
    return transformedDetections;
  }, []);

  // آنالیز یک عکس
  const analyzeSingleImage = async (imageFile, imageId) => {
    try {
      setLoadingImages((prev) => new Set(prev).add(imageId));
      
      // Check if imageFile is a File object or file-like object
      let fileToSend = imageFile;
      
      // If it's a file-like object (has preview/path but not a File instance), convert it to File
      if (!(imageFile instanceof File) && !(imageFile instanceof Blob)) {
        // Get the image URL from preview or path
        const imageUrl = imageFile.preview || imageFile.path || '';
        
        if (imageUrl) {
          // Fetch the image and convert to File
          const response = await fetch(imageUrl);
          const blob = await response.blob();
          const fileName = imageFile.name || `image-${Date.now()}.jpg`;
          fileToSend = new File([blob], fileName, { type: blob.type || 'image/jpeg' });
        } else {
          throw new Error('تصویر معتبر یافت نشد');
        }
      }
      
      const formData = new FormData();
      formData.append('file', fileToSend);
      formData.append('model', selectedModel);
      formData.append('conf', confidenceThreshold.toString());

      const response = await fetch(`${CONFIG.site.serverUrl?.replace(':7272', ':5001') || 'http://localhost:5001'}/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `خطای HTTP: ${response.status}`);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || 'خطا در تشخیص');
      }

      // اعمال فیلتر detections
      const filteredDetections = filterDetections(data.detections || [], selectedModel);
      
      // به‌روزرسانی summary
      const newSummary = {};
      filteredDetections.forEach((det) => {
        if (!newSummary[det.class_name]) {
          newSummary[det.class_name] = {
            count: 0,
            max_confidence: 0,
          };
        }
        newSummary[det.class_name].count += 1;
        newSummary[det.class_name].max_confidence = Math.max(
          newSummary[det.class_name].max_confidence,
          det.confidence
        );
      });

      const filteredResult = {
        ...data,
        detections: filteredDetections,
        total_detections: filteredDetections.length,
        summary: newSummary,
      };

      // به‌روزرسانی results برای این عکس
      setImageResults((prev) => ({
        ...prev,
        [imageId]: {
          ...prev[imageId],
          detections: filteredDetections,
          result: filteredResult,
        },
      }));

      // Return result for caller
      return filteredResult;
    } catch (err) {
      console.error(`خطا در تشخیص برای عکس ${imageId}:`, err);
      setError(err.message || `خطا در ارتباط با سرور AI. لطفاً اطمینان حاصل کنید unified_ai_api_server.py روی port 5001 در حال اجراست.`);
      return null;
    } finally {
      setLoadingImages((prev) => {
        const updated = new Set(prev);
        updated.delete(imageId);
        return updated;
      });
    }
  };

  const saveAnalysis = useCallback(async (resultsToSave = null, filesToUse = null) => {
    console.log('🔄 [IntraOralView] saveAnalysis called with:', {
      patientId,
      hasResults: !!resultsToSave,
      resultsCount: resultsToSave ? Object.keys(resultsToSave).length : 0,
      filesCount: filesToUse ? filesToUse.length : (imageFiles ? imageFiles.length : 0)
    });

    if (!patientId) {
      console.warn('Cannot save analysis: patientId is missing');
      return;
    }
    
    // Use provided results or current state
    const currentResults = resultsToSave || imageResults;
    const currentFiles = filesToUse || imageFiles;
    
    try {
      // Get existing history
      const existingRes = await axios.get(`${endpoints.patients}/${patientId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      const patientData = existingRes.data?.patient || existingRes.data;
      let existingHistory = [];
      
      if (patientData.intraOralAnalysis) {
        try {
          const data = patientData.intraOralAnalysis;
          
          // Handle both object and string formats
          if (typeof data === 'object') {
            if (Array.isArray(data)) {
              existingHistory = data;
            } else if (data.analyses && Array.isArray(data.analyses)) {
              existingHistory = data.analyses;
            }
          } else if (typeof data === 'string') {
            const trimmedData = data.trim();
            if (trimmedData.startsWith('{') || trimmedData.startsWith('[')) {
              const parsed = JSON.parse(trimmedData);
              if (Array.isArray(parsed)) {
                existingHistory = parsed;
              } else if (parsed.analyses && Array.isArray(parsed.analyses)) {
                existingHistory = parsed.analyses;
              }
            }
          }
        } catch (parseError) {
          console.error('Failed to parse existing history:', parseError);
        }
      }

      // Prepare new analysis data
      const analyses = [];
      Object.keys(currentResults).forEach((localId) => {
        const r = currentResults[localId]?.result;
        const file = currentFiles.find(f => f.id === localId);
        if (r && file) {
          analyses.push({ serverImageId: file.serverId || null, result: r });
        }
      });

      if (analyses.length === 0) {
        console.warn('No analysis results to save');
        return;
      }

      // Add new analysis to history
      const newAnalysis = {
        id: `analysis_${Date.now()}`,
        timestamp: new Date().toISOString(),
        analyses,
      };
      
      const updatedHistory = [...existingHistory, newAnalysis];

      // Save to database using PUT endpoint
      console.log('📤 [IntraOralView] Sending data to API:', {
        patientId,
        dataSize: JSON.stringify(updatedHistory).length,
        analysisCount: updatedHistory.length,
        sampleAnalysis: updatedHistory[0] ? {
          id: updatedHistory[0].id,
          timestamp: updatedHistory[0].timestamp,
          analysesCount: updatedHistory[0].analyses?.length || 0
        } : null
      });

      await axios.put(
        `${endpoints.patients}/${patientId}`,
        { intraOralAnalysis: JSON.stringify(updatedHistory) },
        {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
        }
      );

      console.log('✅ Intraoral analysis saved to history');
      toast.success(`✅ نتایج آنالیز ذخیره شد (${analyses.length} تصویر)`);

      // Reload history but don't auto-select (keep current analysis visible)
        await loadAnalysisHistory();

      // Reset selectedAnalysisIndex to prevent loading historical data over current results
      setSelectedAnalysisIndex(null);
    } catch (err) {
      console.error('❌ Failed to save intraoral analysis:', err);
      const errorMsg = err.response?.data?.error || err.response?.data?.message || err.message || 'خطای نامشخص';
      toast.error(`خطا در ذخیره نتایج آنالیز: ${errorMsg}`);
    }
  }, [patientId, imageResults, imageFiles, user?.accessToken, loadAnalysisHistory]);

  const handleDetect = async () => {
    console.log('🚀 [IntraOralView] handleDetect called - starting analysis');

    if (imageFiles.length === 0) {
      setError('لطفاً ابتدا یک یا چند تصویر آپلود کنید');
      return;
    }

    console.log('📊 [IntraOralView] Starting analysis for', imageFiles.length, 'images');
    setIsLoading(true);
    setError(null);

    try {
      // آنالیز همه عکس‌ها به صورت موازی
      const results = await Promise.all(
        imageFiles.map((img) => analyzeSingleImage(img.file, img.id))
      );

      // Ensure state is up-to-date, but also merge returned results defensively
      const toMerge = {};
      results.forEach((res, idx) => {
        if (res) {
          const localId = imageFiles[idx]?.id;
          if (localId) {
            toMerge[localId] = {
              preview: imageFiles[idx].preview,
              detections: res.detections || [],
              result: res,
            };
          }
        }
      });
      
      if (Object.keys(toMerge).length > 0) {
        // Create updated results object
        const updatedResults = { ...imageResults, ...toMerge };
        
        // Update state
        setImageResults(updatedResults);

        // Persist analysis results to backend using the updated results and current files
        await saveAnalysis(updatedResults, imageFiles);
      }
    } catch (err) {
      console.error('خطا در تشخیص:', err);
      setError(err.message || 'خطا در پردازش تصاویر');
    } finally {
      setIsLoading(false);
    }
  };
  
  const handleDetectionsChange = useCallback(async (imageId, updatedDetections) => {
    // اعمال فیلتر detections
    const filteredDetections = filterDetections(updatedDetections, selectedModel);
    
    // به‌روزرسانی results برای این عکس
    setImageResults((prev) => {
      if (!prev[imageId]) return prev;
      return {
        ...prev,
        [imageId]: {
          ...prev[imageId],
          detections: filteredDetections,
          result: prev[imageId].result ? {
            ...prev[imageId].result,
            detections: filteredDetections,
            total_detections: filteredDetections.length,
          } : null,
        },
      };
    });
    
    // ذخیره تغییرات در آنالیز اگر آنالیز انتخاب شده باشد
    if (selectedAnalysisIndex !== null && analysisHistory.length > 0 && patientId) {
      try {
        // به‌روزرسانی آنالیز در history
        const updatedHistory = [...analysisHistory];
        const currentAnalysis = updatedHistory[selectedAnalysisIndex];
        
        if (currentAnalysis && currentAnalysis.analyses) {
          // پیدا کردن آنالیز مربوط به این تصویر
          const imageAnalysisIndex = currentAnalysis.analyses.findIndex(
            (analysis) => {
              const fileId = imageFiles.find(f => f.id === imageId)?.serverId || imageId;
              return analysis.imageId === fileId || analysis.imageId === imageId;
            }
          );
          
          if (imageAnalysisIndex !== -1) {
            // به‌روزرسانی detections در آنالیز
            updatedHistory[selectedAnalysisIndex] = {
              ...currentAnalysis,
              analyses: currentAnalysis.analyses.map((analysis, idx) => {
                if (idx === imageAnalysisIndex) {
                  return {
                    ...analysis,
                    result: {
                      ...analysis.result,
                      detections: filteredDetections,
                      total_detections: filteredDetections.length,
                    },
                  };
                }
                return analysis;
              }),
            };
            
            // ذخیره در دیتابیس
            await axios.put(
              `${endpoints.patients}/${patientId}`,
              { intraOralAnalysis: JSON.stringify(updatedHistory) },
              {
                headers: {
                  Authorization: `Bearer ${user?.accessToken}`,
                  'Content-Type': 'application/json',
                },
              }
            );
            
            // به‌روزرسانی state
            setAnalysisHistory(updatedHistory);
            toast.success('تغییرات لندمارک‌ها ذخیره شد');
          }
        }
      } catch (error) {
        console.error('❌ [IntraOralView] Error saving detection changes:', error);
        toast.error('خطا در ذخیره تغییرات لندمارک‌ها');
      }
    }
  }, [selectedModel, filterDetections, selectedAnalysisIndex, analysisHistory, patientId, imageFiles, user?.accessToken]);


  return (
    <Container maxWidth="xl">
      <Stack spacing={3}>
        {/* Header */}
        <Stack direction="row" alignItems="center" spacing={2}>
          <Iconify icon="solar:teeth-bold" width={40} />
          <Box>
            <Typography variant="h6">آنالیز عکس‌های داخل دهانی</Typography>

          </Box>
        </Stack>


        {/* Delete Image Confirmation Dialog */}
        <Dialog open={deleteImageDialogOpen} onClose={() => !deleting && setDeleteImageDialogOpen(false)}>
          <DialogTitle>حذف تصویر</DialogTitle>
          <DialogContent>
            <Typography>
              آیا از حذف این تصویر مطمئن هستید؟ این عمل غیرقابل بازگشت است.
            </Typography>
            {imageToDelete && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                نام فایل: {imageToDelete.originalName || `تصویر-${imageToDelete.id}`}
              </Typography>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteImageDialogOpen(false)} color="inherit" disabled={deleting}>
              انصراف
            </Button>
            <Button
              onClick={async () => {
                if (!imageToDelete) {
                  console.warn('⚠️ [IntraOralView] No image to delete');
                  return;
                }
                
                console.log('🗑️ [IntraOralView] Delete button clicked:', {
                  imageToDelete,
                  patientId,
                  hasOnDeleteImage: !!onDeleteImage,
                });
                
                try {
                  setDeleting(true);
                  
                  if (patientId) {
                    // If we have patientId, delete directly from server
                    console.log('📞 [IntraOralView] Deleting image with patientId');
                    await handleDeleteImageFromServer(imageToDelete);
                    // Also call onDeleteImage if provided to update parent component
                    if (onDeleteImage) {
                      onDeleteImage(imageToDelete);
                    }
                  } else if (onDeleteImage) {
                    // If no patientId but onDeleteImage exists, use it
                    console.log('📞 [IntraOralView] Calling onDeleteImage prop (no patientId)');
                    onDeleteImage(imageToDelete);
                    // Update local state
                    setLocalInitialImages((prev) => prev.filter((img) => img.id !== imageToDelete.id));
                  } else {
                    console.warn('⚠️ [IntraOralView] Cannot delete: no patientId and no onDeleteImage');
                    toast.error('امکان حذف تصویر وجود ندارد. شناسه بیمار یافت نشد.');
                  }
                  
                  setDeleteImageDialogOpen(false);
                  setImageToDelete(null);
                } catch (error) {
                  console.error('❌ [IntraOralView] Error deleting image:', error);
                  const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
                  toast.error(`خطا در حذف تصویر: ${errorMsg}`);
                } finally {
                  setDeleting(false);
                }
              }}
              color="error"
              variant="contained"
              disabled={deleting}
              startIcon={deleting ? <Iconify icon="eva:loader-fill" /> : <Iconify icon="solar:trash-bin-trash-bold" />}
            >
              {deleting ? 'در حال حذف...' : 'حذف'}
            </Button>
          </DialogActions>
        </Dialog>

        {/* Main Content */}
        <Stack direction={{ xs: 'column', md: 'row' }} spacing={3}>
          {/* Right Panel - Results - Mobile: 1st, Desktop: 1st */}
          <Stack spacing={3} sx={{ flex: 1, order: { xs: 1, md: 1 } }}>
            {/* Error */}
            {error && (
              <Alert 
                severity="error" 
                onClose={() => setError(null)}
                icon={<Iconify icon="solar:danger-triangle-bold" />}
              >
                <Typography variant="subtitle2">خطا در تشخیص</Typography>
                <Typography variant="body2" sx={{ mt: 0.5 }}>{error}</Typography>
              </Alert>
            )}

            {/* Results for each image */}
            {imageFiles.map((img) => {
              const imageResult = imageResults[img.id];
              const detections = imageResult?.detections || [];
              const result = imageResult?.result;

              return (
                <Card key={img.id} sx={{ mb: 2 }}>
                  <CardContent>
                    <Stack spacing={2}>
                      {/* Image Header */}
                      <Stack direction="row" justifyContent="space-between" alignItems="center">
                        <Typography variant="h6">تصویر {imageFiles.indexOf(img) + 1}</Typography>
                      </Stack>

                      {/* Visualization - نمایش تصویر حتی اگر آنالیز نشده باشد */}
                      {img.preview && (
                        <DetectionVisualizer
                          imageUrl={img.preview}
                          detections={detections}
                          onDetectionsChange={(updatedDetections) => handleDetectionsChange(img.id, updatedDetections)}
                        />
                      )}

                    </Stack>
                  </CardContent>
                </Card>
              );
            })}

            {/* Empty State */}
            {imageFiles.length === 0 && !error && !isLoading && initialImagesLoaded && (
              <Card>
                <CardContent>
                  <Stack spacing={2} alignItems="center" sx={{ py: 4 }}>
                    <Iconify 
                      icon="solar:scan-bold" 
                      width={64} 
                      sx={{ color: 'text.disabled', mb: 2 }} 
                    />
                    <Typography variant="h6" sx={{ color: 'text.secondary' }}>
                      آماده تشخیص
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.disabled', textAlign: 'center' }}>
                      یک یا چند تصویر داخل دهانی را آپلود کرده و روی دکمه "آنالیز" کلیک کنید
                    </Typography>
                  </Stack>
                </CardContent>
              </Card>
            )}
          </Stack>

          {/* Left Panel - Upload - Mobile: 2nd, Desktop: 2nd */}
          <Stack spacing={3} sx={{ width: { xs: '100%', md: '400px' }, order: { xs: 2, md: 2 } }}>
            
                    {/* Analysis Selection Dropdown */}
        {patientId && (
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="h6">📋 تاریخچه آنالیز</Typography>
                <Stack direction="row" spacing={1} alignItems="flex-start">
                <FormControl fullWidth size="small">
                    <InputLabel>انتخاب آنالیز از تاریخچه</InputLabel>
                  <Select
                      value={selectedAnalysisIndex !== null ? selectedAnalysisIndex : ''}
                      onChange={(e) => setSelectedAnalysisIndex(e.target.value !== '' ? Number(e.target.value) : null)}
                      label="انتخاب آنالیز از تاریخچه"
                      disabled={analysisHistory.length === 0}
                  >
                      {analysisHistory.length > 0 ? (
                        analysisHistory.map((analysis, index) => (
                      <MenuItem key={index} value={index}>
                            <Box sx={{ width: '100%' }}>
                          <Typography variant="body2">
                                آنالیز {index + 1} - {analysis.analyses?.length || 0} تصویر
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                                {analysis.timestamp 
                                  ? new Date(analysis.timestamp).toLocaleDateString('fa-IR', {
                                  year: 'numeric',
                                  month: 'long',
                                  day: 'numeric',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                })
                              : 'تاریخ نامشخص'}
                          </Typography>
                        </Box>
                      </MenuItem>
                        ))
                      ) : (
                        <MenuItem value="" disabled>
                          <Typography variant="body2" color="text.secondary">
                            هیچ آنالیز ذخیره شده‌ای وجود ندارد
                          </Typography>
                        </MenuItem>
                      )}
                    </Select>
                  </FormControl>
                  {selectedAnalysisIndex !== null && analysisHistory.length > 0 && (
                    <IconButton
                      color="error"
                      size="small"
                      onClick={() => {
                          const analysis = analysisHistory[selectedAnalysisIndex];
                          if (analysis) {
                            setAnalysisToDelete({ index: selectedAnalysisIndex, analysis });
                            setDeleteDialogOpen(true);
                          }
                        }}
                      sx={{ mt: 1.5 }}
                      >
                          <Iconify icon="solar:trash-bin-trash-bold" width={20} />
                    </IconButton>
                    )}
                </Stack>
              </Stack>
            </CardContent>
          </Card>
        )}

        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onClose={() => !deleting && setDeleteDialogOpen(false)}>
          <DialogTitle>حذف آنالیز داخل دهان</DialogTitle>
          <DialogContent>
            <Typography>
              آیا از حذف این آنالیز مطمئن هستید؟ این عمل غیرقابل بازگشت است.
            </Typography>
            {analysisToDelete && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                تعداد تصاویر: {analysisToDelete.analysis?.analyses?.length || 0}
              </Typography>
            )}
          </DialogContent>
          <DialogActions>
            <Button onClick={() => setDeleteDialogOpen(false)} color="inherit" disabled={deleting}>
              انصراف
            </Button>
            <Button
              onClick={async () => {
                if (!analysisToDelete || !patientId) return;
                
                try {
                  setDeleting(true);
                  
                  // Remove the analysis from history
                  const newHistory = analysisHistory.filter((_, idx) => idx !== analysisToDelete.index);
                  
                  // Update database using PUT endpoint
                  await axios.put(
                    `${endpoints.patients}/${patientId}`,
                    { intraOralAnalysis: newHistory.length > 0 ? JSON.stringify(newHistory) : null },
                    {
                      headers: {
                        Authorization: `Bearer ${user?.accessToken}`,
                        'Content-Type': 'application/json',
                      },
                    }
                  );
                  
                  // Update state
                  setAnalysisHistory(newHistory);
                  
                  // Select the first analysis if available
                  if (newHistory.length > 0) {
                    setSelectedAnalysisIndex(0);
                  } else {
                    setSelectedAnalysisIndex(null);
                    setLastSavedAnalysis(null);
                  }
                  
                  toast.success('آنالیز با موفقیت حذف شد');
                  setDeleteDialogOpen(false);
                  setAnalysisToDelete(null);
                } catch (error) {
                  console.error('Error deleting analysis:', error);
                  toast.error('خطا در حذف آنالیز');
                } finally {
                  setDeleting(false);
                }
              }}
              color="error"
              variant="contained"
              disabled={deleting}
              startIcon={deleting ? <Iconify icon="eva:loader-fill" /> : <Iconify icon="solar:trash-bin-trash-bold" />}
            >
              {deleting ? 'در حال حذف...' : 'حذف'}
            </Button>
          </DialogActions>
        </Dialog>


<Card>
            
              <CardContent>
                <Stack spacing={2}>
                  <Typography variant="h6">📷 تصاویر</Typography>

                  <Upload
                    multiple
                    thumbnail={true}
                    hideUploadButton={true} // مخفی کردن دکمه آپلود و نمایش فقط thumbnail ها
                    value={imageFiles.map(img => img.file)}
                    onDrop={handleDropMultiFile}
                    onRemove={handleRemoveFile}
                    onRemoveAll={handleRemoveAllFiles}
                  />

                </Stack>
              </CardContent>
              

</Card>
            {/* Detect Button */}
            <Button
              fullWidth
              size="medium"
              variant="contained"
              color="primary"
              onClick={handleDetect}
              disabled={isLoading || imageFiles.length === 0}
              sx={{ mb: '10px' }}
              startIcon={
                isLoading ? (
                  <CircularProgress size={16} sx={{ color: 'inherit' }} />
                ) : (
                  <Iconify icon="solar:scan-bold" width={20} />
                )
              }
            >
              {isLoading ? 'در حال پردازش' : `آنالیز`}
            </Button>
          </Stack>
        </Stack>
      </Stack>
    </Container>
  );
}
