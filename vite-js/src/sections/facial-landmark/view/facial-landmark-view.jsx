import { toast } from 'sonner';
import { useRef, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Table from '@mui/material/Table';
import Paper from '@mui/material/Paper';
import Dialog from '@mui/material/Dialog';
import Button from '@mui/material/Button';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import Container from '@mui/material/Container';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';

import axios, { endpoints } from 'src/utils/axios';
import { getApiUrl, getImageUrl } from 'src/utils/url-helpers';
import { analyzeFacialBeauty } from 'src/utils/facial-beauty-analysis';
import { beautyAnalysisTableData } from 'src/utils/beauty-analysis-table-data';

import { CONFIG } from 'src/config-global';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { LandmarkVisualizer } from 'src/components/landmark-visualizer/landmark-visualizer';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

export function FacialLandmarkView({ initialImages = [], patientId = null }) {
  const { user } = useAuthContext();
  const [imageFiles, setImageFiles] = useState([]);
  const [selectedFileIndex, setSelectedFileIndex] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [landmarks, setLandmarks] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [availableModels, setAvailableModels] = useState([]);
  const [beautyAnalysis, setBeautyAnalysis] = useState(null);
  const [initialImagesLoaded, setInitialImagesLoaded] = useState(false);

  // Track blob URL failures for recovery
  const [blobUrlFailures, setBlobUrlFailures] = useState(new Set());

  // Analysis history states
  const [lastSavedAnalysis, setLastSavedAnalysis] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [analysisToDelete, setAnalysisToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  
  // Track if this is the first time loading history (after page refresh)
  const isFirstHistoryLoad = useRef(true);
  
  // Track if we've already calculated beauty analysis for current landmarks
  const beautyAnalysisCalculatedRef = useRef(false);
  
  // Track current beauty analysis to avoid dependency issues
  const beautyAnalysisRef = useRef(beautyAnalysis);
  
  // Update ref when beautyAnalysis changes
  useEffect(() => {
    beautyAnalysisRef.current = beautyAnalysis;
  }, [beautyAnalysis]);
  
  // Table pagination state
  const [tablePage, setTablePage] = useState(0);
  const [tableRowsPerPage, setTableRowsPerPage] = useState(5);

  // Get currently selected file
  const selectedFile = selectedFileIndex !== null && selectedFileIndex < imageFiles.length ? imageFiles[selectedFileIndex] : null;
  const imagePreview = selectedFile?.preview || null;

  const handleDrop = useCallback((acceptedFiles) => {
    if (acceptedFiles && acceptedFiles.length > 0) {
      const newFiles = acceptedFiles.map((file) => {
        const preview = URL.createObjectURL(file);
        return {
          file,
          preview,
          id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
          name: file.name,
          size: file.size,
          type: file.type,
        };
      });
      
      setImageFiles((prev) => {
        const updated = [...prev, ...newFiles];
        // If no file is selected, select the first new file
        if (selectedFileIndex === null && newFiles.length > 0) {
          setSelectedFileIndex(prev.length);
        }
        return updated;
      });
      
      setError(null);
      // Don't clear result when adding new files
    }
  }, [selectedFileIndex]);

  // Handle delete image from server
  const handleDeleteImageFromServer = useCallback(async (image) => {
    if (!patientId || !image?.id) {
      console.warn('⚠️ [FacialLandmarkView] Cannot delete image: missing patientId or image.id');
      return;
    }

    try {
      await axios.delete(`${endpoints.patients}/${patientId}/images`, {
        data: { imageId: image.id },
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
      });
      
      toast.success('تصویر با موفقیت حذف شد');
    } catch (error) {
      console.error('❌ [FacialLandmarkView] Error deleting image:', error);
      const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
      toast.error(`خطا در حذف تصویر: ${errorMsg}`);
    }
  }, [patientId, user?.accessToken]);

  const handleRemoveFile = useCallback((indexOrFile) => {
    setImageFiles((prev) => {
      // Handle both index (number) and file object
      let indexToRemove = -1;
      if (typeof indexOrFile === 'number') {
        indexToRemove = indexOrFile;
      } else {
        // Find by file object
        indexToRemove = prev.findIndex((item) => item.file === indexOrFile);
        // Fallback: match by name and size
        if (indexToRemove === -1) {
          indexToRemove = prev.findIndex((item) => {
            if (item.file && item.file.name === indexOrFile.name && item.file.size === indexOrFile.size) {
              return true;
            }
            return false;
          });
        }
      }
      
      if (indexToRemove === -1) {
        return prev; // File not found
      }
      
      const fileToRemove = prev[indexToRemove];
      
      // If file has serverId, delete from server
      if (fileToRemove?.serverId && patientId) {
        handleDeleteImageFromServer({ id: fileToRemove.serverId });
      }
      
      const newFiles = prev.filter((_, index) => index !== indexToRemove);
      
      // Clean up preview URL with delay to prevent race conditions
      // Only revoke if this is not the currently selected file (to avoid revoking active image)
      if (fileToRemove?.preview && selectedFileIndex !== indexToRemove) {
        // Use setTimeout to delay revocation and prevent race conditions
        setTimeout(() => {
          try {
            if (fileToRemove.preview.startsWith('blob:')) {
              URL.revokeObjectURL(fileToRemove.preview);
              console.log('[Facial Landmark] Revoked blob URL for removed file:', fileToRemove.name);
            }
          } catch (error) {
            console.warn('[Facial Landmark] Error revoking blob URL:', error);
          }
        }, 100);
      }
      
      // Adjust selected index
      if (selectedFileIndex === indexToRemove) {
        // If removed file was selected, select first available or null
        if (newFiles.length > 0) {
          setSelectedFileIndex(0);
        } else {
          setSelectedFileIndex(null);
          setResult(null);
          setLandmarks([]);
          setBeautyAnalysis(null);
        }
      } else if (selectedFileIndex > indexToRemove) {
        // Adjust index if selected file is after removed file
        setSelectedFileIndex(selectedFileIndex - 1);
      }
      
      return newFiles;
    });
  }, [selectedFileIndex, patientId, handleDeleteImageFromServer]);

  const handleSelectFile = useCallback((index) => {
    setSelectedFileIndex(index);
    setResult(null);
    setLandmarks([]);
    setBeautyAnalysis(null);
    setError(null);
  }, []);

  // Function to refresh blob URL when it fails
  const refreshBlobUrl = useCallback((fileIndex) => {
    setImageFiles((prev) => {
      const updated = [...prev];
      const fileItem = updated[fileIndex];
      
      if (fileItem && fileItem.file) {
        // Revoke old blob URL if it exists
        if (fileItem.preview && fileItem.preview.startsWith('blob:')) {
          try {
            URL.revokeObjectURL(fileItem.preview);
          } catch (error) {
            console.warn('[Facial Landmark] Error revoking old blob URL:', error);
          }
        }
        
        // Create new blob URL
        const newPreview = URL.createObjectURL(fileItem.file);
        updated[fileIndex] = {
          ...fileItem,
          preview: newPreview,
        };
        
        console.log('[Facial Landmark] Refreshed blob URL for file:', fileItem.name);
        
        // Remove from failures set
        setBlobUrlFailures((prev) => {
          const newSet = new Set(prev);
          newSet.delete(fileItem.id);
          return newSet;
        });
      }
      
      return updated;
    });
  }, []);

  // Handle image load error from LandmarkVisualizer
  const handleImageLoadError = useCallback((errorInfo) => {
    console.log('[Facial Landmark] Image load error from LandmarkVisualizer:', errorInfo);
    
    if (errorInfo.isBlobUrl && selectedFileIndex !== null && imageFiles[selectedFileIndex]) {
      // Mark this file as having a failed blob URL
      setBlobUrlFailures((prev) => new Set([...prev, imageFiles[selectedFileIndex].id]));
      
      // Optionally auto-refresh the blob URL for better UX
      console.log('[Facial Landmark] Auto-refreshing blob URL due to error...');
      setTimeout(() => {
        refreshBlobUrl(selectedFileIndex);
      }, 1000);
    }
    
    // Set error state for user feedback
    setError(`خطا در بارگذاری تصویر: ${errorInfo.error}`);
  }, [selectedFileIndex, imageFiles, refreshBlobUrl]);

  // لیست مدل‌های موجود (hardcoded در frontend برای جلوگیری از layout shift)
  const AVAILABLE_MODELS = useMemo(() => [
    'mediapipe',  // مدل پیش فرض - MediaPipe معمولاً بهتر کار می‌کند
    'face_alignment',
    'dlib',
  ], []);

  // تنظیم اولیه مدل‌ها
  useEffect(() => {
    setAvailableModels(AVAILABLE_MODELS);
    
    // Auto-select first model if none selected
    if (!selectedModel || !AVAILABLE_MODELS.includes(selectedModel)) {
      setSelectedModel(AVAILABLE_MODELS[0]);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  // Create a key from initialImages to detect changes
  const imagesKey = useMemo(() => initialImages?.map(img => img.path || img.id).join(',') || '', [initialImages]);

  // Reset loaded flag when initialImages change (e.g., different patient or new images added)
  useEffect(() => {
    if (imagesKey) {
      setInitialImagesLoaded(false);
    }
  }, [imagesKey]);

  // Load initial images from props
  useEffect(() => {
    if (initialImages && initialImages.length > 0 && !initialImagesLoaded) {
      const loadInitialImages = async () => {
        const loadedImages = [];
        
        for (const image of initialImages) {
          try {
            // Get image URL
            const imageUrl = image.path?.startsWith('http') 
              ? image.path 
              : getImageUrl(image.path);
            
            console.log('[Facial Landmark] Loading initial image:', {
              path: image.path,
              url: imageUrl,
              originalName: image.originalName || image.name,
            });
            
            // Fetch image and convert to File
            const response = await fetch(imageUrl);
            if (!response.ok) {
              console.error(`Failed to fetch image ${imageUrl}:`, response.status, response.statusText);
              continue;
            }
            
            const blob = await response.blob();
            
            // Verify blob is valid
            if (!blob || blob.size === 0) {
              console.error(`Blob is empty for image ${imageUrl}`);
              continue;
            }
            
            // Determine file extension from blob type or URL
            let extension = 'jpg';
            if (blob.type) {
              if (blob.type.includes('png')) extension = 'png';
              else if (blob.type.includes('jpeg') || blob.type.includes('jpg')) extension = 'jpg';
            } else if (imageUrl) {
              // Try to get extension from URL
              const urlExtension = imageUrl.match(/\.(jpg|jpeg|png)$/i)?.[1]?.toLowerCase();
              if (urlExtension) extension = urlExtension === 'jpeg' ? 'jpg' : urlExtension;
            }
            
            // Ensure we have a valid file name with extension
            let fileName = image.originalName || image.name || `image.${extension}`;
            // Remove any existing extension and add the correct one
            fileName = `${fileName.replace(/\.[^/.]+$/, '')  }.${extension}`;
            
            // Ensure file type is set correctly
            const fileType = blob.type || (extension === 'png' ? 'image/png' : 'image/jpeg');
            
            // Create File object with proper name and type
            let file = new File([blob], fileName, { type: fileType });
            
            // Verify file was created correctly
            if (!file || file.size === 0) {
              console.error(`Failed to create file for image ${imageUrl}: file is empty`);
              continue;
            }
            
            // Double-check file name is valid
            if (!file.name || file.name.trim() === '' || !file.name.includes('.')) {
              console.error(`Invalid file name for image ${imageUrl}:`, file.name);
              // Create a new file with a valid name
              const validFileName = `image-${Date.now()}.${extension}`;
              const newFile = new File([blob], validFileName, { type: fileType });
              if (newFile && newFile.size > 0) {
                file = newFile;
                fileName = validFileName;
              } else {
                console.error(`Failed to create file with valid name for image ${imageUrl}`);
                continue;
              }
            }
            
            const imageId = image.id || `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
            const previewUrl = URL.createObjectURL(file);
            
            loadedImages.push({
              id: imageId,
              file,
              preview: previewUrl,
              name: file.name, // Use file.name to ensure it's correct
              size: file.size,
              type: file.type,
            });
            
            console.log('[Facial Landmark] Successfully loaded image:', {
              id: imageId,
              fileName: file.name,
              size: file.size,
              type: file.type,
              blobSize: blob.size,
              blobType: blob.type,
            });
          } catch (err) {
            console.error(`Error loading image ${image.path}:`, err);
          }
        }
        
        if (loadedImages.length > 0) {
          console.log('[Facial Landmark] Loaded', loadedImages.length, 'initial images');
          setImageFiles(loadedImages);
          // Select first image automatically
          setSelectedFileIndex(0);
          setInitialImagesLoaded(true);
        } else {
          console.warn('[Facial Landmark] No images were loaded from initialImages');
          // If no images loaded, still mark as loaded to prevent retry
          setInitialImagesLoaded(true);
        }
      };
      
      loadInitialImages();
    } else if (!initialImages || initialImages.length === 0) {
      // Reset when initialImages is empty
      setInitialImagesLoaded(false);
    }
  }, [initialImages, initialImagesLoaded]);

  // Cleanup preview URLs on unmount
  useEffect(() => () => {
      // Only clean up blob URLs when component is actually unmounting
      // Use a small delay to prevent race conditions with ongoing image loads
      setTimeout(() => {
        imageFiles.forEach((item) => {
          if (item.preview && item.preview.startsWith('blob:')) {
            try {
              URL.revokeObjectURL(item.preview);
              console.log('[Facial Landmark] Cleaned up blob URL on unmount:', item.name);
            } catch (error) {
              console.warn('[Facial Landmark] Error cleaning up blob URL:', error);
            }
          }
        });
      }, 500); // Delay to allow any ongoing operations to complete
    }, [imageFiles]);

  // Load analysis history
  const loadAnalysisHistory = useCallback(async () => {
    console.log('📚 [FacialLandmarkView] loadAnalysisHistory called for patient:', patientId);

    if (!patientId) return;
    
    setIsLoadingHistory(true);
    try {
      console.log('🔍 [FacialLandmarkView] Fetching patient data from API...');
      const res = await axios.get(`${endpoints.patients}/${patientId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      // Parse analysis history from facialLandmarkAnalysis field
      const patientData = res.data?.patient || res.data;
      console.log('📊 [FacialLandmarkView] Patient data received:', {
        hasFacialLandmarkAnalysis: !!patientData.facialLandmarkAnalysis,
        facialLandmarkAnalysisType: typeof patientData.facialLandmarkAnalysis,
        facialLandmarkAnalysisLength: patientData.facialLandmarkAnalysis?.length || 0
      });

      let analyses = [];

      if (patientData.facialLandmarkAnalysis) {
        try {
          const data = patientData.facialLandmarkAnalysis;
          
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
          console.error('❌ [FacialLandmarkView] Failed to parse facial landmark analysis:', parseError);
        }
      } else {
        console.log('⚠️ [FacialLandmarkView] No facialLandmarkAnalysis field in patient data');
      }

      console.log('📋 [FacialLandmarkView] Final analysis history:', analyses.length, 'entries');
      setAnalysisHistory(analyses);
      
      // Auto-select the latest analysis on first load (after page refresh)
      // This ensures the latest analysis is shown when user refreshes the page
      // Only auto-select if this is the first load and no analysis is currently selected
      if (isFirstHistoryLoad.current && analyses.length > 0 && selectedAnalysisIndex === null) {
        const latestIndex = analyses.length - 1;
        console.log('🔄 [FacialLandmarkView] Auto-selecting latest analysis on first load (index:', latestIndex, ')');
        setSelectedAnalysisIndex(latestIndex);
        isFirstHistoryLoad.current = false;
      } else if (isFirstHistoryLoad.current) {
        // Mark as not first load even if no analyses found
        isFirstHistoryLoad.current = false;
      }
    } catch (err) {
      console.error('Failed to load analysis history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [patientId, user?.accessToken, selectedAnalysisIndex]);

  // Reset first load flag when patientId changes
  useEffect(() => {
    isFirstHistoryLoad.current = true;
  }, [patientId]);

  // Load history when component mounts
  useEffect(() => {
    if (patientId && user?.accessToken) {
      loadAnalysisHistory();
    }
  }, [patientId, user?.accessToken, loadAnalysisHistory]);

  // Load selected analysis data when selectedAnalysisIndex changes
  useEffect(() => {
    // Only load from history if user explicitly selected an analysis
    // Don't auto-load when selectedAnalysisIndex is set by loadAnalysisHistory
    if (analysisHistory.length > 0 && selectedAnalysisIndex !== null && selectedAnalysisIndex >= 0 && selectedAnalysisIndex < analysisHistory.length) {
      const selectedAnalysis = analysisHistory[selectedAnalysisIndex];
      const firstAnalysis = selectedAnalysis?.analyses?.[0];

      if (firstAnalysis) {
        console.log('📥 Loading analysis from history (index:', selectedAnalysisIndex, ')');
        console.log('📦 [FacialLandmarkView] Analysis data:', {
          hasResult: !!firstAnalysis.result,
          hasLandmarks: !!firstAnalysis.landmarks,
          landmarksInResult: !!firstAnalysis.result?.landmarks,
          landmarksCount: firstAnalysis.landmarks?.length || firstAnalysis.result?.landmarks?.length || 0,
          serverImageId: firstAnalysis.serverImageId,
          modelId: firstAnalysis.modelId,
        });
        
        // Restore result, landmarks, and beauty analysis
        if (firstAnalysis.result) {
          setResult(firstAnalysis.result);
          
          // Extract landmarks from result if not stored separately
          if (firstAnalysis.result.landmarks && (!firstAnalysis.landmarks || firstAnalysis.landmarks.length === 0)) {
            console.log('📊 [FacialLandmarkView] Loading landmarks from result');
            setLandmarks(firstAnalysis.result.landmarks || []);
          }
        }
        
        // Use landmarks from firstAnalysis if available, otherwise use from result
        let finalLandmarks = [];
        if (firstAnalysis.landmarks && firstAnalysis.landmarks.length > 0) {
          console.log('📊 [FacialLandmarkView] Loading landmarks from firstAnalysis.landmarks');
          finalLandmarks = firstAnalysis.landmarks;
          setLandmarks(firstAnalysis.landmarks);
        } else if (firstAnalysis.result?.landmarks && firstAnalysis.result.landmarks.length > 0) {
          console.log('📊 [FacialLandmarkView] Loading landmarks from firstAnalysis.result.landmarks');
          finalLandmarks = firstAnalysis.result.landmarks;
          setLandmarks(firstAnalysis.result.landmarks);
        }
        
        // Load or recalculate beauty analysis
        if (firstAnalysis.beautyAnalysis && firstAnalysis.beautyAnalysis.success) {
          // Use saved beauty analysis if it exists and is valid
          console.log('✅ [FacialLandmarkView] Using saved beauty analysis');
          setBeautyAnalysis(firstAnalysis.beautyAnalysis);
          // Mark as calculated to prevent recalculation in useEffect
          if (Array.isArray(finalLandmarks) && finalLandmarks.length > 0) {
            try {
              const sample = finalLandmarks.slice(0, 5).map(l => {
                if (l && typeof l === 'object' && ('x' in l || 'y' in l)) {
                  return { x: l.x || 0, y: l.y || 0 };
                }
                return null;
              }).filter(Boolean);
              if (sample.length > 0) {
                const landmarksKey = JSON.stringify(sample);
                beautyAnalysisCalculatedRef.current = landmarksKey;
              }
            } catch (err) {
              console.warn('⚠️ [FacialLandmarkView] Error creating landmarks key:', err);
            }
          }
        } else if (Array.isArray(finalLandmarks) && finalLandmarks.length > 0) {
          // Recalculate beauty analysis from landmarks if not saved or invalid
          console.log('🔄 [FacialLandmarkView] Recalculating beauty analysis from landmarks (count:', finalLandmarks.length, ')');
          try {
            const analysis = analyzeFacialBeauty(finalLandmarks);
            if (analysis && analysis.success) {
              console.log('✅ [FacialLandmarkView] Beauty analysis recalculated:', {
                success: analysis.success,
                overallScore: analysis.overallScore,
              });
              setBeautyAnalysis(analysis);
              // Mark as calculated to prevent recalculation in useEffect
              try {
                const sample = finalLandmarks.slice(0, 5).map(l => {
                  if (l && typeof l === 'object' && ('x' in l || 'y' in l)) {
                    return { x: l.x || 0, y: l.y || 0 };
                  }
                  return null;
                }).filter(Boolean);
                if (sample.length > 0) {
                  const landmarksKey = JSON.stringify(sample);
                  beautyAnalysisCalculatedRef.current = landmarksKey;
                }
              } catch (err) {
                console.warn('⚠️ [FacialLandmarkView] Error creating landmarks key:', err);
              }
            } else {
              console.warn('⚠️ [FacialLandmarkView] Beauty analysis calculation returned invalid result:', analysis);
              setBeautyAnalysis(null);
            }
          } catch (analysisErr) {
            console.error('❌ [FacialLandmarkView] Error recalculating beauty analysis:', analysisErr);
            setBeautyAnalysis(null);
          }
        } else {
          console.warn('⚠️ [FacialLandmarkView] No landmarks found, cannot calculate beauty analysis');
          setBeautyAnalysis(null);
          beautyAnalysisCalculatedRef.current = '';
        }
        
        if (firstAnalysis.modelId) setSelectedModel(firstAnalysis.modelId);
        
        // Try to load image from serverImageId or from result
        const imageIdToLoad = firstAnalysis.serverImageId || firstAnalysis.result?.imageId || firstAnalysis.result?.serverImageId;
        
        console.log('🔍 [FacialLandmarkView] Looking for image:', {
          imageIdToLoad,
          initialImagesLength: initialImages.length,
          imageFilesLength: imageFiles.length,
        });
        
        let imageFound = false;
        
        // First, try to find in imageFiles (already loaded images)
        if (imageIdToLoad && imageFiles.length > 0) {
          const existingFile = imageFiles.find(f => 
            (f.serverId && String(f.serverId) === String(imageIdToLoad)) ||
            (f.id && String(f.id) === String(imageIdToLoad)) ||
            (f.id && f.id.includes(String(imageIdToLoad)))
          );
          
          if (existingFile) {
            console.log('✅ [FacialLandmarkView] Found image in imageFiles:', existingFile);
            const index = imageFiles.findIndex(f => 
              (f.serverId && String(f.serverId) === String(imageIdToLoad)) ||
              (f.id && String(f.id) === String(imageIdToLoad)) ||
              (f.id && f.id.includes(String(imageIdToLoad)))
            );
            if (index >= 0) {
              setSelectedFileIndex(index);
              imageFound = true;
            }
          }
        }
        
        // If not found in imageFiles, try to find in initialImages
        if (!imageFound && imageIdToLoad && initialImages.length > 0) {
          const serverImage = initialImages.find(img => 
            String(img.id) === String(imageIdToLoad) || 
            String(img.serverId) === String(imageIdToLoad)
          );
          
          if (serverImage) {
            console.log('✅ [FacialLandmarkView] Found image in initialImages:', serverImage);
            const imageUrl = serverImage.path?.startsWith('http')
              ? serverImage.path
              : serverImage.path?.startsWith('/uploads/')
              ? `${getImageUrl(serverImage.path)}`
              : `${getImageUrl(serverImage.path)}`;
            
            // Add to imageFiles if not already there
            const existingFile = imageFiles.find(f => 
              f.serverId === serverImage.id || 
              (f.serverId && String(f.serverId) === String(serverImage.id))
            );
            
            if (!existingFile) {
              const newFile = {
                file: null,
                preview: imageUrl,
                id: `server-${serverImage.id}`,
                serverId: serverImage.id,
                name: serverImage.originalName || serverImage.name,
                size: serverImage.size,
                type: serverImage.mimeType,
              };
              setImageFiles(prev => {
                const updated = [...prev, newFile];
                // Select this file
                setSelectedFileIndex(updated.length - 1);
                return updated;
              });
              imageFound = true;
            } else {
              // Update existing file if it has an expired blob URL
              if (existingFile.preview && existingFile.preview.startsWith('blob:')) {
                // Revoke old blob URL
                try {
                  URL.revokeObjectURL(existingFile.preview);
                } catch (error) {
                  console.warn('⚠️ [FacialLandmarkView] Failed to revoke old blob URL:', error);
                }
                
                // Update with server URL
                setImageFiles(prev => {
                  const updated = [...prev];
                  const fileIndex = updated.findIndex(f => 
                    f.serverId === serverImage.id || 
                    (f.serverId && String(f.serverId) === String(serverImage.id))
                  );
                  if (fileIndex >= 0) {
                    updated[fileIndex] = {
                      ...updated[fileIndex],
                      preview: imageUrl,
                    };
                  }
                  return updated;
                });
              }
              
              // Select existing file
              const index = imageFiles.findIndex(f => 
                f.serverId === serverImage.id || 
                (f.serverId && String(f.serverId) === String(serverImage.id))
              );
              if (index >= 0) {
                setSelectedFileIndex(index);
                imageFound = true;
              }
            }
          } else {
            console.warn('⚠️ [FacialLandmarkView] Image not found in initialImages:', imageIdToLoad);
          }
        }
        
        // If still not found and we have imageFiles, use the first available
        if (!imageFound && imageFiles.length > 0) {
          console.log('⚠️ [FacialLandmarkView] Image not found by serverImageId, using first available image');
          setSelectedFileIndex(0);
        } else if (!imageFound && !imageIdToLoad) {
          console.log('⚠️ [FacialLandmarkView] No serverImageId found in analysis');
        }
        
        // Log final landmarks count after setting
        console.log('✅ Loaded analysis from history:', {
          hasResult: !!firstAnalysis.result,
          landmarksCount: finalLandmarks.length,
          hasBeauty: !!firstAnalysis.beautyAnalysis,
          model: firstAnalysis.modelId,
          serverImageId: imageIdToLoad,
        });
      }
    }
  }, [selectedAnalysisIndex, analysisHistory, initialImages, imageFiles]);

  // Recalculate beauty analysis when landmarks change (if beauty analysis is missing or invalid)
  // This ensures beauty analysis is calculated even when loading from history or after page refresh
  useEffect(() => {
    // Safety check: ensure landmarks is an array
    if (!Array.isArray(landmarks)) {
      return;
    }
    
    // Create a key from landmarks to detect changes (safely)
    let landmarksKey = '';
    if (landmarks.length > 0) {
      try {
        const sample = landmarks.slice(0, 5).map(l => {
          if (l && typeof l === 'object' && ('x' in l || 'y' in l)) {
            return { x: l.x || 0, y: l.y || 0 };
          }
          return null;
        }).filter(Boolean);
        if (sample.length > 0) {
          landmarksKey = JSON.stringify(sample);
        }
      } catch (err) {
        console.warn('⚠️ [FacialLandmarkView] Error creating landmarks key:', err);
      }
    }
    
    // Only recalculate if:
    // 1. We have landmarks
    // 2. We don't have a valid beauty analysis (missing or invalid)
    // 3. We have a result (meaning analysis was done, not just loading)
    // 4. We haven't already calculated for these landmarks
    const currentBeautyAnalysis = beautyAnalysisRef.current;
    if (landmarks.length > 0 && (!currentBeautyAnalysis || !currentBeautyAnalysis.success) && result) {
      // Check if we've already calculated for these landmarks
      const lastCalculatedKey = beautyAnalysisCalculatedRef.current;
      if (lastCalculatedKey === landmarksKey && lastCalculatedKey !== '' && currentBeautyAnalysis) {
        // Already calculated for these landmarks, skip
        return;
      }
      
      console.log('🔄 [FacialLandmarkView] Recalculating beauty analysis from landmarks (useEffect):', {
        landmarksCount: landmarks.length,
        hasBeautyAnalysis: !!currentBeautyAnalysis,
        beautyAnalysisSuccess: currentBeautyAnalysis?.success,
        hasResult: !!result,
      });
      
      try {
        const analysis = analyzeFacialBeauty(landmarks);
        if (analysis && analysis.success) {
          console.log('✅ [FacialLandmarkView] Beauty analysis recalculated (useEffect):', {
            success: analysis.success,
            overallScore: analysis.overallScore,
          });
          setBeautyAnalysis(analysis);
          beautyAnalysisCalculatedRef.current = landmarksKey;
        } else {
          console.warn('⚠️ [FacialLandmarkView] Beauty analysis calculation returned invalid result:', analysis);
        }
      } catch (analysisErr) {
        console.error('❌ [FacialLandmarkView] Error recalculating beauty analysis (useEffect):', analysisErr);
        // Don't set to null here, keep existing value if any
      }
    } else if (landmarks.length === 0) {
      // Reset ref when landmarks are cleared
      beautyAnalysisCalculatedRef.current = '';
    }
  }, [landmarks, result]); // Only depend on landmarks and result to avoid infinite loop

  // Save analysis function
  const saveAnalysis = useCallback(async (resultsToSave = null) => {
    console.log('🔄 [FacialLandmarkView] saveAnalysis called with:', {
      patientId,
      hasResults: !!resultsToSave,
      hasLandmarks: !!(resultsToSave?.landmarks || result?.landmarks),
      landmarkCount: (resultsToSave?.landmarks || result?.landmarks) ? Object.keys(resultsToSave?.landmarks || result?.landmarks).length : 0
    });

    if (!patientId) {
      console.warn('Cannot save analysis: patientId is missing');
      return;
    }
    
    const currentResult = resultsToSave || result;
    const currentLandmarks = landmarks;
    const currentBeauty = beautyAnalysis;
    
    try {
      // Get existing history
      const existingRes = await axios.get(`${endpoints.patients}/${patientId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      const patientData = existingRes.data?.patient || existingRes.data;
      let existingHistory = [];
      
      if (patientData.facialLandmarkAnalysis) {
        try {
          const data = patientData.facialLandmarkAnalysis;
          
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

      if (!currentResult) {
        console.warn('No analysis results to save');
        return;
      }

      // Get serverImageId from selectedFile or from imageFiles
      let serverImageIdToSave = selectedFile?.serverId || null;
      
      // If serverId is not in selectedFile, try to find it from imageFiles
      if (!serverImageIdToSave && selectedFileIndex !== null && imageFiles[selectedFileIndex]) {
        serverImageIdToSave = imageFiles[selectedFileIndex].serverId || null;
      }
      
      // If still not found, try to match with initialImages by preview URL or name
      if (!serverImageIdToSave && selectedFile && initialImages.length > 0) {
        const matchedImage = initialImages.find(img => {
          // Try to match by preview URL
          if (selectedFile.preview && img.path) {
            const imgUrl = img.path.startsWith('http') ? img.path : getImageUrl(img.path);
            if (selectedFile.preview === imgUrl || selectedFile.preview.includes(img.path)) {
              return true;
            }
          }
          // Try to match by name
          if (selectedFile.name && img.originalName) {
            if (selectedFile.name === img.originalName || selectedFile.name === img.name) {
              return true;
            }
          }
          return false;
        });
        
        if (matchedImage) {
          serverImageIdToSave = matchedImage.id;
        }
      }
      
      console.log('💾 [FacialLandmarkView] Saving analysis with serverImageId:', serverImageIdToSave);
      
      // Add new analysis to history
      const newAnalysis = {
        id: `analysis_${Date.now()}`,
        timestamp: new Date().toISOString(),
        analyses: [{
          serverImageId: serverImageIdToSave,
          modelId: selectedModel,
          result: currentResult,
          landmarks: currentLandmarks,
          beautyAnalysis: currentBeauty,
        }]
      };

      const updatedHistory = [...existingHistory, newAnalysis];

      // Save to database using PUT endpoint
      await axios.put(
        `${endpoints.patients}/${patientId}`,
        { facialLandmarkAnalysis: JSON.stringify(updatedHistory) },
        {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
        }
      );

      console.log('✅ Facial landmark analysis saved to history');
      toast.success('✅ نتایج آنالیز ذخیره شد');

      // Don't reload history or change selectedAnalysisIndex after saving
      // This prevents the newly detected landmarks from being overwritten
      // Just reload the history data without triggering the selection effect
      try {
        const res = await axios.get(`${endpoints.patients}/${patientId}`, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        
        const patientData = res.data?.patient || res.data;
        let analyses = [];
        
        if (patientData.facialLandmarkAnalysis) {
          try {
            const data = patientData.facialLandmarkAnalysis;
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
            console.error('Failed to parse existing history:', parseError);
          }
        }
        
        // Update history without changing selectedAnalysisIndex
        setAnalysisHistory(analyses);
      } catch (err) {
        console.error('Failed to reload history:', err);
      }
    } catch (err) {
      console.error('❌ Failed to save facial landmark analysis:', err);
      const errorMsg = err.response?.data?.error || err.response?.data?.message || err.message || 'خطای نامشخص';
      toast.error(`خطا در ذخیره نتایج آنالیز: ${errorMsg}`);
    }
  }, [patientId, result, landmarks, beautyAnalysis, selectedModel, selectedFile, selectedFileIndex, imageFiles, initialImages, user?.accessToken]);

  const handleDetect = async () => {
    console.log('🚀 [FacialLandmarkView] handleDetect called - starting analysis');

    // اگر فایلی انتخاب نشده باشد، اولین فایل را به صورت خودکار انتخاب کن
    let fileToUse = selectedFile;
    if (!fileToUse && imageFiles.length > 0) {
      fileToUse = imageFiles[0];
      setSelectedFileIndex(0);
    }
    
    if (!fileToUse) {
      setError('لطفاً ابتدا یک تصویر آپلود کنید');
      return;
    }
    
    // بررسی اینکه فایل واقعاً موجود است
    if (!fileToUse.file) {
      setError('فایل انتخاب شده معتبر نیست. لطفاً دوباره فایل را آپلود کنید');
      return;
    }
    
    if (!selectedModel || availableModels.length === 0) {
      setError('لطفاً ابتدا یک مدل انتخاب کنید یا منتظر بمانید تا مدل‌ها بارگذاری شوند');
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      // Use backend API endpoint instead of direct Python server
      const backendUrl = CONFIG.site.serverUrl || getApiUrl('');
      
      // ایجاد FormData برای ارسال فایل
      const formData = new FormData();
      
      // اطمینان از اینکه فایل به درستی اضافه شده است
      let {file} = fileToUse;
      
      console.log('[Facial Landmark] Preparing file for upload:', {
        fileToUse,
        file,
        isFile: file instanceof File,
        isBlob: file instanceof Blob,
        fileName: file?.name,
        fileSize: file?.size,
        fileType: file?.type,
      });
      
      // اگر file یک File object نیست، سعی کن آن را بسازی
      if (!(file instanceof File)) {
        // اگر file یک Blob است، آن را به File تبدیل کن
        if (file instanceof Blob) {
          // ساخت نام فایل معتبر
          let fileName = fileToUse.name && fileToUse.name.trim() !== '' 
            ? fileToUse.name 
            : 'image.jpg';
          
          // اگر fileName extension ندارد، اضافه کن
          if (!fileName.includes('.')) {
            const extension = file.type?.includes('png') ? 'png' : 
                             file.type?.includes('jpeg') || file.type?.includes('jpg') ? 'jpg' : 'jpg';
            fileName = `${fileName}.${extension}`;
          }
          
          const fileType = file.type || 'image/jpeg';
          file = new File([file], fileName, { type: fileType });
          console.log('[Facial Landmark] Converted Blob to File:', {
            fileName,
            fileType,
            size: file.size,
          });
        } else {
          console.error('[Facial Landmark] File is not a File or Blob:', file);
          throw new Error('فایل انتخاب شده معتبر نیست. لطفاً فایل را دوباره آپلود کنید');
        }
      }
      
      // بررسی اینکه فایل خالی نیست
      if (!file || file.size === 0) {
        console.error('[Facial Landmark] File is empty or invalid:', {
          file,
          size: file?.size,
        });
        throw new Error('فایل انتخاب شده خالی است. لطفاً فایل را دوباره آپلود کنید');
      }
      
      // اطمینان از اینکه نام فایل معتبر است (باید extension داشته باشد)
      if (!file.name || file.name.trim() === '' || !file.name.includes('.')) {
        // ساخت نام فایل معتبر با extension
        const extension = file.type?.includes('png') ? 'png' : 
                         file.type?.includes('jpeg') || file.type?.includes('jpg') ? 'jpg' : 'jpg';
        const newFileName = fileToUse.name && fileToUse.name.trim() !== '' 
          ? `${fileToUse.name}.${extension}`
          : `image.${extension}`;
        file = new File([file], newFileName, { type: file.type || 'image/jpeg' });
        console.log('[Facial Landmark] Fixed file name:', newFileName);
      }
      
      // اضافه کردن فایل به FormData (مشابه IntraOralView - بدون پارامتر سوم)
      formData.append('file', file);
      
      // Debug: بررسی اینکه فایل واقعاً آماده است
      console.log('[Facial Landmark Client] File prepared for upload:', {
        fileName: file.name,
        fileSize: file.size,
        fileType: file.type,
        isFile: file instanceof File,
        isBlob: file instanceof Blob,
        fileObject: {
          name: file.name,
          size: file.size,
          type: file.type,
          lastModified: file.lastModified,
        },
      });
      
      // بررسی نهایی: اطمینان از اینکه فایل معتبر است
      if (!file || file.size === 0 || !file.name || file.name.trim() === '') {
        console.error('[Facial Landmark] File validation failed:', {
          file,
          size: file?.size,
          name: file?.name,
        });
        throw new Error('خطا در آماده‌سازی فایل برای ارسال: فایل خالی یا نام معتبر ندارد');
      }

      // Verify FormData before sending
      // Note: We can't directly check FormData contents in browser, but we can verify the file
      console.log('[Facial Landmark] Sending request to backend:', {
        url: `${backendUrl}/api/ai/facial-landmark?model=${selectedModel}`,
        fileSize: file.size,
        fileName: file.name,
        fileType: file.type,
        model: selectedModel,
      });

      // ارسال درخواست به backend API که به Python server proxy می‌کند
      const response = await fetch(`${backendUrl}/api/ai/facial-landmark?model=${selectedModel}`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it automatically with boundary for FormData
        // This is crucial for multipart/form-data
      });

      if (!response.ok) {
        let errorMessage = `خطای HTTP: ${response.status}`;
        try {
          const errorData = await response.json();
          errorMessage = errorData.error || errorData.message || errorMessage;
        } catch (e) {
          // If response is not JSON, try to get text
          try {
            const errorText = await response.text();
            if (errorText) errorMessage = errorText;
          } catch (e2) {
            // Ignore
          }
        }
        throw new Error(errorMessage);
      }

      const data = await response.json();

      if (!data.success) {
        throw new Error(data.error || data.message || 'خطا در تشخیص');
      }

      console.log('✅ [FacialLandmarkView] Detection successful:', {
        success: data.success,
        landmarksCount: data.landmarks?.length || 0,
        totalLandmarks: data.total_landmarks,
        hasLandmarks: !!data.landmarks,
        landmarksType: typeof data.landmarks,
        sampleLandmark: data.landmarks?.[0],
      });
      
      setResult(data);
      const detectedLandmarks = data.landmarks || [];
      
      console.log('📊 [FacialLandmarkView] Setting landmarks:', {
        count: detectedLandmarks.length,
        sample: detectedLandmarks.slice(0, 3),
      });
      
      setLandmarks(detectedLandmarks);
      
      // انجام آنالیز زیبایی صورت
      if (detectedLandmarks.length > 0) {
        try {
          console.log('[Facial Beauty] Analyzing landmarks:', {
            count: detectedLandmarks.length,
            sample: detectedLandmarks.slice(0, 5),
            firstLandmark: detectedLandmarks[0],
            hasIndex: detectedLandmarks[0]?.index !== undefined,
            hasName: detectedLandmarks[0]?.name !== undefined,
          });
          
          const analysis = analyzeFacialBeauty(detectedLandmarks);
          
          console.log('[Facial Beauty] Analysis result:', {
            success: analysis?.success,
            overallScore: analysis?.overallScore,
            hasSymmetry: !!analysis?.symmetry,
            hasGoldenRatio: !!analysis?.goldenRatio,
            hasEyes: !!analysis?.eyes,
            hasNose: !!analysis?.nose,
            hasMouth: !!analysis?.mouth,
          });
          
          setBeautyAnalysis(analysis);
        } catch (analysisErr) {
          console.error('خطا در آنالیز زیبایی:', analysisErr);
          console.error('Error stack:', analysisErr.stack);
          setBeautyAnalysis(null);
        }
      } else {
        console.warn('[Facial Beauty] No landmarks detected, skipping analysis');
        setBeautyAnalysis(null);
      }

      // Save analysis results to backend if patientId is available
      // Don't await - let it run in background to avoid blocking UI
      if (patientId && data) {
        console.log('💾 [FacialLandmarkView] Saving analysis in background...');
        saveAnalysis(data).catch(err => {
          console.error('❌ Background save failed:', err);
        });
      }
    } catch (err) {
      console.error('خطا در تشخیص:', err);
      
      let errorMessage = 'خطا در ارتباط با سرور AI';
      const errMessage = err?.message || String(err || '');
      
      // Check for specific error types
      if (errMessage.includes('fetch failed') || errMessage.includes('Failed to fetch') || errMessage.includes('NetworkError')) {
        errorMessage = 'خطا در اتصال به سرور. لطفاً اطمینان حاصل کنید که backend server در حال اجراست.';
      } else if (errMessage.includes('503') || errMessage.includes('not available')) {
        errorMessage = 'سرور AI در دسترس نیست. لطفاً اطمینان حاصل کنید که unified_ai_api_server.py روی پورت 5001 در حال اجراست.';
      } else if (errMessage.includes('CORS') || errMessage.includes('cors')) {
        errorMessage = 'خطای CORS. لطفاً با مدیر سیستم تماس بگیرید.';
      } else if (errMessage) {
        errorMessage = errMessage;
      }
      
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <Container maxWidth="xl" sx={{ width: '100%', maxWidth: '100%', px: { xs: 1, sm: 2, md: 3 } }}>
      <Stack spacing={3} sx={{ width: '100%' }}>
        {/* Header */}
        <Stack direction={{ xs: 'column', sm: 'row' }} alignItems={{ xs: 'flex-start', sm: 'center' }} spacing={2}>
          <Iconify icon="solar:face-smile-bold" width={40} />
          <Box>
            <Typography variant="h6" sx={{ fontSize: { xs: '1.25rem', sm: '1.25rem' } }}>آنالیز صورت</Typography>

          </Box>
        </Stack>


        {/* Delete Confirmation Dialog */}
        <Dialog open={deleteDialogOpen} onClose={() => !deleting && setDeleteDialogOpen(false)}>
          <DialogTitle>حذف آنالیز لندمارک صورت</DialogTitle>
          <DialogContent>
            <Typography>
              آیا از حذف این آنالیز مطمئن هستید؟ این عمل غیرقابل بازگشت است.
            </Typography>
            {analysisToDelete && (
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                مدل: {analysisToDelete.analysis?.analyses?.[0]?.modelId || 'نامشخص'}
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
                  
                  console.log('🗑️ [FacialLandmarkView] Deleting analysis:', {
                    index: analysisToDelete.index,
                    historyLength: analysisHistory.length,
                    analysisId: analysisToDelete.analysis?.id,
                  });
                  
                  // Remove the analysis from history
                  const newHistory = analysisHistory.filter((_, idx) => idx !== analysisToDelete.index);
                  
                  console.log('📝 [FacialLandmarkView] New history after deletion:', {
                    oldLength: analysisHistory.length,
                    newLength: newHistory.length,
                    removedIndex: analysisToDelete.index,
                  });
                  
                  // Update database using PUT endpoint
                  await axios.put(
                    `${endpoints.patients}/${patientId}`,
                    { facialLandmarkAnalysis: newHistory.length > 0 ? JSON.stringify(newHistory) : null },
                    {
                      headers: {
                        Authorization: `Bearer ${user?.accessToken}`,
                        'Content-Type': 'application/json',
                      },
                    }
                  );
                  
                  console.log('✅ [FacialLandmarkView] Analysis deleted from database');
                  
                  // Update local state immediately for better UX
                  setAnalysisHistory(newHistory);
                  
                  // Reload history from server to ensure consistency
                  await loadAnalysisHistory();
                  console.log('✅ [FacialLandmarkView] History reloaded from server');
                  
                  // After reload, select the first analysis if available
                  // Use setTimeout to ensure state is updated after loadAnalysisHistory
                  setTimeout(() => {
                    setAnalysisHistory((currentHistory) => {
                      if (currentHistory.length > 0) {
                        setSelectedAnalysisIndex(0);
                      } else {
                        setSelectedAnalysisIndex(null);
                        setLastSavedAnalysis(null);
                        // Clear current result and landmarks if no history left
                        setResult(null);
                        setLandmarks([]);
                        setBeautyAnalysis(null);
                      }
                      return currentHistory;
                    });
                  }, 50);
                  
                  toast.success('آنالیز با موفقیت حذف شد');
                  setDeleteDialogOpen(false);
                  setAnalysisToDelete(null);
                } catch (error) {
                  console.error('❌ [FacialLandmarkView] Error deleting analysis:', error);
                  const errorMsg = error.response?.data?.error || error.response?.data?.message || error.message || 'خطای نامشخص';
                  toast.error(`خطا در حذف آنالیز: ${errorMsg}`);
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
        <Stack direction={{ xs: 'column', lg: 'row' }} spacing={3} sx={{ width: '100%' }}>
          {/* Right Panel - Results - Mobile: 1st, Desktop: 1st */}
          <Stack spacing={3} sx={{ flex: 1, minWidth: 0, width: { xs: '100%', lg: 'auto' }, order: { xs: 1, lg: 1 } }}>
            {/* Visualization - Always show if image is available */}
            {imagePreview && (
              <Card>
                <CardContent>
                  <Stack spacing={2}>
                    <Typography variant="h6">🎨 نمایش لندمارک‌ها</Typography>
                    {landmarks.length > 0 ? (
                      <LandmarkVisualizer
                        imageUrl={imagePreview}
                        landmarks={landmarks}
                        showLandmarks
                        showOutlines={false}
                        showProfileLines={false}
                        showFrontalLines={true}
                        showLandmarkNames={true}
                        onImageLoadError={handleImageLoadError}
                        retryFailedBlob={true}
                      />
                    ) : (
                      <Box sx={{ position: 'relative' }}>
                        {/* Show image even without landmarks */}
                        <Box
                          component="img"
                          src={imagePreview}
                          alt="Facial image"
                          sx={{
                            width: '100%',
                            height: 'auto',
                            maxHeight: 600,
                            objectFit: 'contain',
                            borderRadius: 1,
                            border: '1px solid',
                            borderColor: 'divider',
                          }}
                          onError={(e) => {
                            console.error('[Facial Landmark] Image load error:', e);
                            handleImageLoadError({
                              error: 'Failed to load image',
                              isBlobUrl: imagePreview.startsWith('blob:'),
                            });
                          }}
                        />
                        {!result && (
                          <Box sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary">
                              برای شناسایی لندمارک‌ها، دکمه "تشخیص لندمارک‌ها" را بزنید
                            </Typography>
                          </Box>
                        )}
                        {result && landmarks.length === 0 && (
                          <Box sx={{ p: 2, textAlign: 'center' }}>
                            <Typography variant="body2" color="text.secondary">
                              لندمارک‌ها شناسایی نشدند
                            </Typography>
                            <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                              تعداد لندمارک‌های شناسایی شده: {result.total_landmarks || 0}
                            </Typography>
                          </Box>
                        )}
                      </Box>
                    )}
                  </Stack>
                </CardContent>
              </Card>
            )}

            {result && (
              <>

                {/* Beauty Analysis */}
                {beautyAnalysis && beautyAnalysis.success && (
                  <Card>
                    <CardContent>
                      <Stack spacing={2}>
                        <Stack direction="row" alignItems="center" spacing={1}>
                          <Iconify icon="solar:star-bold" width={24} sx={{ color: 'primary.main' }} />
                          <Typography variant="h6">آنالیز زیبایی صورت</Typography>
                        </Stack>

                        {/* Overall Score */}
                        <Box>
                          <Typography variant="subtitle2" sx={{ mb: 1 }}>
                            امتیاز کلی زیبایی
                          </Typography>
                          <Stack direction="row" alignItems="center" spacing={2}>
                            <Box sx={{ flex: 1 }}>
                              <LinearProgress
                                variant="determinate"
                                value={beautyAnalysis.overallScore || 0}
                            sx={{
                                  height: 8,
                                  borderRadius: 4,
                                  bgcolor: 'grey.200',
                                  '& .MuiLinearProgress-bar': {
                                    bgcolor: 'primary.main',
                                    borderRadius: 4,
                                  },
                            }}
                              />
                            </Box>
                            <Typography variant="h6" sx={{ minWidth: 50, textAlign: 'right' }}>
                              {beautyAnalysis.overallScore?.toFixed(1) || 0}%
                            </Typography>
                          </Stack>
                          </Box>

                        {/* Symmetry Analysis */}
                        {beautyAnalysis.symmetry && (
                          <Box>
                            <Typography variant="subtitle2" sx={{ mb: 1 }}>
                              تقارن صورت
                            </Typography>
                            <Stack spacing={1}>
                              <Stack direction="row" justifyContent="space-between">
                                <Typography variant="body2">تقارن کلی:</Typography>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  {beautyAnalysis.symmetry.overall.toFixed(1)}%
                                </Typography>
                              </Stack>
                              <Stack direction="row" justifyContent="space-between">
                                <Typography variant="body2">تقارن چشم‌ها:</Typography>
                                <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                  {beautyAnalysis.symmetry.eyes.toFixed(1)}%
                                </Typography>
                              </Stack>
                                <Stack direction="row" justifyContent="space-between">
                                  <Typography variant="body2">تقارن دهان:</Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {beautyAnalysis.symmetry.mouth.toFixed(1)}%
                                  </Typography>
                                </Stack>
                            </Stack>
                          </Box>
                        )}

                        {/* Golden Ratio Analysis */}
                        {beautyAnalysis.goldenRatio && Object.keys(beautyAnalysis.goldenRatio).length > 0 && (
                          <Box>
                            <Typography variant="subtitle2" sx={{ mb: 1 }}>
                              نسبت طلایی (1:1.618)
                            </Typography>
                            <Stack spacing={1}>
                              {beautyAnalysis.goldenRatio.verticalRatio && (
                                <Stack direction="row" justifyContent="space-between">
                                  <Typography variant="body2">نسبت عمودی:</Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {beautyAnalysis.goldenRatio.verticalRatio.ratio} ({beautyAnalysis.goldenRatio.verticalRatio.score}%)
                                  </Typography>
                                </Stack>
                              )}
                              {beautyAnalysis.goldenRatio.horizontalRatio && (
                                <Stack direction="row" justifyContent="space-between">
                                  <Typography variant="body2">نسبت افقی:</Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {beautyAnalysis.goldenRatio.horizontalRatio.ratio} ({beautyAnalysis.goldenRatio.horizontalRatio.score}%)
                                  </Typography>
                                </Stack>
                              )}
                              {beautyAnalysis.goldenRatio.eyeToNoseRatio && (
                                <Stack direction="row" justifyContent="space-between">
                                  <Typography variant="body2">نسبت چشم به بینی:</Typography>
                                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                                    {beautyAnalysis.goldenRatio.eyeToNoseRatio.ratio} ({beautyAnalysis.goldenRatio.eyeToNoseRatio.score}%)
                                  </Typography>
                                </Stack>
                              )}
                            </Stack>
                          </Box>
                        )}

                      </Stack>
                    </CardContent>
                  </Card>
                )}

                {/* Analysis Table */}
                        {beautyAnalysis && beautyAnalysis.success && (
                  <Box sx={{ width: '100%' }}>
                            <Typography variant="h6" sx={{ mb: 2, color: 'primary.main' }}>
                              جدول آنالیز زیبایی صورت
                            </Typography>

                    {/* Calculate paginated data */}
                    {(() => {
                      const tableData = beautyAnalysisTableData.filter((row) => {
                        const currentValue = row.getValue(beautyAnalysis);
                        return currentValue;
                      });
                      
                      const paginatedData = tableData.slice(
                        tablePage * tableRowsPerPage,
                        tablePage * tableRowsPerPage + tableRowsPerPage
                      );

                      return (
                        <>
                            <TableContainer 
                              component={Paper} 
                              sx={{ 
                                maxHeight: 600,
                                overflowX: 'auto',
                                overflowY: 'auto',
                        borderRadius: '16px',
                                '& .MuiTable-root': {
                          minWidth: 800,
                                }
                              }}
                            >
                              <Table stickyHeader size="small">
                                <TableHead>
                                  <TableRow>
                            <TableCell sx={{ fontWeight: 'bold', whiteSpace: 'nowrap', width: '8%', px: 1.5, py: 1 }}>دسته‌بندی</TableCell>
                            <TableCell sx={{ fontWeight: 'bold', width: '15%', px: 1.5, py: 1 }}>پارامتر</TableCell>
                            <TableCell sx={{ fontWeight: 'bold', whiteSpace: 'nowrap', width: '10%', px: 1.5, py: 1 }}>مقدار فعلی</TableCell>
                            <TableCell sx={{ fontWeight: 'bold', whiteSpace: 'nowrap', width: '12%', px: 1.5, py: 1 }}>مقدار ایده‌آل</TableCell>
                            <TableCell sx={{ fontWeight: 'bold', whiteSpace: 'nowrap', width: '10%', px: 1.5, py: 1 }}>وضعیت</TableCell>
                            <TableCell sx={{ fontWeight: 'bold', width: '45%', px: 1.5, py: 1 }}>توضیحات</TableCell>
                                  </TableRow>
                                </TableHead>
                                <TableBody>
                                {paginatedData.map((row, index) => {
                                    const currentValue = row.getValue(beautyAnalysis);
                                    const score = row.getScore?.(beautyAnalysis);
                                    const grade = row.getGrade?.(beautyAnalysis);
                                    
                                  let statusText = 'نامشخص';
                                    let statusColor = 'default';
                                    
                                  if (grade) {
                                    if (grade >= 90) {
                                        statusText = 'عالی';
                                        statusColor = 'success';
                                    } else if (grade >= 70) {
                                      statusText = 'خوب';
                                      statusColor = 'info';
                                    } else if (grade >= 50) {
                                      statusText = 'متوسط';
                                        statusColor = 'warning';
                                      } else {
                                      statusText = 'نیاز به بهبود';
                                        statusColor = 'error';
                                      }
                                    }

                                    return (
                                      <TableRow key={index} hover>
                                        <TableCell sx={{ whiteSpace: 'nowrap', px: 1.5, py: 0.75, width: '8%' }}>{row.category}</TableCell>
                                        <TableCell sx={{ px: 1.5, py: 0.75, width: '15%' }}>{row.parameter}</TableCell>
                                        <TableCell sx={{ fontFamily: 'monospace', whiteSpace: 'nowrap', px: 1.5, py: 0.75, width: '10%' }}>
                                          {currentValue}
                                          {score && ` (${score}%)`}
                                        </TableCell>
                                        <TableCell sx={{ fontFamily: 'monospace', color: 'text.secondary', whiteSpace: 'nowrap', px: 1.5, py: 0.75, width: '12%' }}>
                                          {row.idealValue}
                                        </TableCell>
                                        <TableCell sx={{ whiteSpace: 'nowrap', px: 1.5, py: 0.75, width: '10%' }}>
                                          <Chip 
                                            label={statusText} 
                                            size="small" 
                                            color={statusColor}
                                            sx={{ height: 20, fontSize: '0.7rem' }}
                                          />
                                        </TableCell>
                                        <TableCell sx={{ fontSize: '0.85rem', color: 'text.secondary', px: 1.5, py: 0.75, width: '45%' }}>
                                          {row.description}
                                        </TableCell>
                                      </TableRow>
                                    );
                                  })}
                                </TableBody>
                              </Table>
                            </TableContainer>
                          
                          {/* Pagination */}
                          <TablePagination
                            component="div"
                            count={tableData.length}
                            page={tablePage}
                            rowsPerPage={tableRowsPerPage}
                            onPageChange={(event, newPage) => setTablePage(newPage)}
                            rowsPerPageOptions={[5]}
                            onRowsPerPageChange={(event) => {
                              setTableRowsPerPage(parseInt(event.target.value, 10));
                              setTablePage(0);
                            }}
                            labelRowsPerPage="تعداد سطر در هر صفحه:"
                            labelDisplayedRows={({ from, to, count }) => `${from}-${to} از ${count}`}
                          />
                        </>
                      );
                    })()}
                                      </Box>
                                    )}
              </>
            )}

            {!result && !isLoading && (
              <Card>
                <CardContent>
                  <Typography variant="body2" sx={{ color: 'text.secondary', textAlign: 'center', py: 4 }}>
                    لطفاً یک تصویر صورت آپلود کرده و دکمه تشخیص را بزنید.
                                        </Typography>
                </CardContent>
              </Card>
            )}
                                      </Stack>

          {/* Left Panel - History & Upload */}
          <Stack spacing={3} sx={{ width: { xs: '100%', lg: '400px' }, flexShrink: 0, order: { xs: 2, lg: 2 } }}>
            {/* Analysis History - Mobile: 2nd, Desktop: 1st */}
            {patientId && (
              <Card sx={{ order: { xs: 1, lg: 1 } }}>
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
                                    آنالیز {index + 1} - {analysis.analyses?.[0]?.modelId || 'مدل نامشخص'}
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

            {/* Upload Card - Mobile: 3rd, Desktop: 2nd */}
            <Card sx={{ order: { xs: 3, lg: 2 } }}>
              <CardContent>
                          <Stack spacing={2}>
                  <Typography variant="h6">📷 آپلود تصویر صورت</Typography>

                  <Upload
                    multiple
                    onDrop={handleDrop}
                    accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
                                        />

                  {/* Model Selection */}
                  <FormControl fullWidth size="small">
                    <InputLabel>انتخاب مدل</InputLabel>
                    <Select
                      value={selectedModel || ''}
                      label="انتخاب مدل"
                      onChange={(e) => setSelectedModel(e.target.value)}
                      disabled={availableModels.length === 0}
                    >
                      {availableModels.length === 0 ? (
                        <MenuItem value="" disabled>
                          در حال بارگذاری مدل‌ها...
                        </MenuItem>
                      ) : (
                        availableModels.map((model) => (
                          <MenuItem key={model} value={model}>
                            {model === 'mediapipe' && 'MediaPipe (468 points - سریع)'}
                            {model === 'dlib' && 'dlib (68 points - کلاسیک)'}
                            {model === 'face_alignment' && 'face-alignment (68 points - دقیق)'}
                            {model === 'retinaface' && 'RetinaFace (5 points - کلیدی)'}
                            {model === 'lab' && 'LAB - Look at Boundary (68 points - دقت بالا)'}
                            {model === '3ddfa' && '3DDFA - 3D Dense Face Alignment (68 points - 3D)'}
                            {!['mediapipe', 'dlib', 'face_alignment', 'retinaface', 'lab', '3ddfa'].includes(model) && model}
                          </MenuItem>
                        ))
                                    )}
                    </Select>
                  </FormControl>

                  {/* File Selection List */}
                  {imageFiles.length > 0 && (
                            <Box>
                              <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        فایل‌های آپلود شده ({imageFiles.length})
                              </Typography>
                      <Stack spacing={1}>
                        {imageFiles.map((item, index) => {
                          // Truncate file name if longer than 20 characters
                          const fileName = item.name.length > 20
                            ? `${item.name.substring(0, 20)}...`
                            : item.name;
                          
                          return (
                            <Card
                              key={item.id}
                              sx={{
                                p: 1.5,
                                border: 1,
                                borderColor: 'divider',
                                bgcolor: 'background.paper',
                                marginTop: '0 !important',
                              }}
                            >
                              <Stack direction="row" spacing={1} alignItems="center">
                                <Box
                                  component="img"
                                  src={item.preview}
                                  alt={item.name}
                                  sx={{
                                    width: 36,
                                    height: 36,
                                    objectFit: 'cover',
                                    borderRadius: 1,
                                  }}
                                />
                                <Box sx={{ flex: 1, minWidth: 0 }}>
                                  <Typography variant="body2" noWrap>
                                    {fileName}
                              </Typography>
                                  <Typography variant="caption" color="text.secondary">
                                    {(item.size / 1024).toFixed(1)} KB
                                  </Typography>
                            </Box>
                                <IconButton
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleRemoveFile(index);
                                  }}
                                  sx={{
                                    width: 26,
                                    height: 26,
                                    p: 0,
                                  }}
                                >
                                  <Iconify icon="mingcute:close-line" width={16} />
                                </IconButton>
                              </Stack>
                            </Card>
                          );
                        })}
                              </Stack>
                            </Box>
                          )}

                  {/* Detect Button */}
                  <Button
                    fullWidth
                    size="medium"
                    variant="contained"
                    color="primary"
                    onClick={handleDetect}
                    disabled={imageFiles.length === 0 || isLoading || !selectedModel || availableModels.length === 0}
                    startIcon={
                      isLoading ? (
                        <CircularProgress size={16} sx={{ color: 'inherit' }} />
                      ) : (
                        <Iconify icon="solar:face-recognition-bold" width={20} />
                      )
                    }
                  sx={{
                      transition: 'all 0.1s ease-in-out !important',
                      '&:active': {
                        transform: 'scale(0.98)'
                      }
                  }}
                >
                    {isLoading ? 'در حال پردازش' : 'تشخیص لندمارک‌ها'}
                  </Button>

                  {error && (
                    <Alert severity="error">
                      {error}
                    </Alert>
                      )}
                    </Stack>
                  </CardContent>
                </Card>
          </Stack>
        </Stack>
      </Stack>
    </Container>
  );
}
