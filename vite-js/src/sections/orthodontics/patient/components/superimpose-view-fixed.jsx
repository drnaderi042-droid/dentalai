import React, { useRef, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Slider from '@mui/material/Slider';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export function SuperimposeView({ patient }) {
  const canvasRef = useRef(null);

  // State declarations (moved to top to fix hoisting issues)
  const [selectedProfileImage, setSelectedProfileImage] = useState(null);
  const [selectedLateralImage, setSelectedLateralImage] = useState(null);
  const [profileOpacity, setProfileOpacity] = useState(0.7);
  const [lateralOpacity, setLateralOpacity] = useState(0.7);
  const [profilePosition, setProfilePosition] = useState({ x: 0, y: 0 });
  const [lateralPosition, setLateralPosition] = useState({ x: 0, y: 0 });
  const [profileScale, setProfileScale] = useState(1);
  const [lateralScale, setLateralScale] = useState(1);
  const [isDragging, setIsDragging] = useState(null); // 'profile' or 'lateral'
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Landmark-based alignment state
  const [useLandmarkAlignment, setUseLandmarkAlignment] = useState(true);
  const [referenceLandmarks, setReferenceLandmarks] = useState(['N', 'S', 'Po']); // Default reference landmarks
  const [profileLandmarks, setProfileLandmarks] = useState({});
  const [lateralLandmarks, setLateralLandmarks] = useState({});
  const [alignmentMode, setAlignmentMode] = useState('auto'); // 'auto' or 'manual'
  const [showLandmarks, setShowLandmarks] = useState(true); // Toggle landmark visibility

  // Image loading state
  const [imagesLoaded, setImagesLoaded] = useState({ profile: false, lateral: false });
  const [loadedImages, setLoadedImages] = useState({ profile: null, lateral: null });

  // Available landmarks for selection
  const availableLandmarks = [
    { id: 'N', name: 'Nasion (N)', description: 'نقطه نازیون' },
    { id: 'S', name: 'Sella (S)', description: 'نقطه سلای تورسیکا' },
    { id: 'Po', name: 'Porion (Po)', description: 'نقطه پوریون' },
    { id: 'Or', name: 'Orbitale (Or)', description: 'نقطه اوربیتال' },
    { id: 'Ba', name: 'Basion (Ba)', description: 'نقطه بازین' },
    { id: 'A', name: 'Point A (A)', description: 'نقطه A' },
    { id: 'B', name: 'Point B (B)', description: 'نقطه B' },
    { id: 'Pog', name: 'Pogonion (Pog)', description: 'نقطه پگونین' },
    { id: 'Gn', name: 'Gnathion (Gn)', description: 'نقطه گناتیون' },
    { id: 'Me', name: 'Menton (Me)', description: 'نقطه منتون' },
  ];

  // Get available images
  const profileImages = useMemo(() => patient?.images?.profile || [], [patient?.images?.profile]);
  const lateralImages = useMemo(() => patient?.images?.lateral || [], [patient?.images?.lateral]);

  // Debug logging for image availability
  console.log('[SuperimposeView] Patient images:', {
    hasPatient: !!patient,
    patientImages: patient?.images,
    profileImagesCount: profileImages.length,
    lateralImagesCount: lateralImages.length,
    profileImages: profileImages.map(img => ({ id: img.id, name: img.originalName, path: img.path })),
    lateralImages: lateralImages.map(img => ({ id: img.id, name: img.originalName, path: img.path })),
  });

  // Detect cephalometric landmarks for lateral images using Aariz models
  const detectCephalometricLandmarks = useCallback(async (imagePath) => {
    try {
      console.log('🔍 Detecting cephalometric landmarks for lateral image:', imagePath);

      // Convert image URL to base64 with proper API endpoint handling
      let fetchUrl = imagePath;
      
      // If it's a relative path starting with /uploads/, use the serve-upload API
      if (imagePath.startsWith('/uploads/')) {
        const relativePath = imagePath.replace('/uploads/', '');
        fetchUrl = `http://localhost:7272/api/serve-upload?path=${encodeURIComponent(relativePath)}`;
        console.log('🔄 Using serve-upload API for cephalometric image:', fetchUrl);
      } else if (imagePath.startsWith('http://localhost:5001')) {
        // Replace with correct port if needed
        fetchUrl = imagePath.replace('http://localhost:5001', 'http://localhost:7272');
        console.log('🔄 Corrected port for cephalometric image URL:', fetchUrl);
      }

      const response = await fetch(fetchUrl);
      if (!response.ok) {
        throw new Error(`Failed to fetch lateral image: ${response.status} - ${fetchUrl}`);
      }

      const blob = await response.blob();
      const base64 = await new Promise((resolve) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.readAsDataURL(blob);
      });

      // Call cephalometric landmark detection API (using ensemble model for best results)
      const apiResponse = await fetch('http://localhost:5001/detect-ensemble-512-768-tta', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_base64: base64
        })
      });

      if (!apiResponse.ok) {
        throw new Error(`Cephalometric landmark detection failed: ${apiResponse.status}`);
      }

      const data = await apiResponse.json();

      if (data.success && data.landmarks) {
        console.log('✅ Cephalometric landmarks detected:', Object.keys(data.landmarks).length, 'landmarks');
        return data.landmarks;
      } 
        console.warn('⚠️ Cephalometric landmark detection returned no results');
        return null;
      
    } catch (error) {
      console.error('❌ Cephalometric landmark detection error:', error);
      return null; // This will trigger the fallback to patient data
    }
  }, []);

  // Detect facial landmarks for profile images using ML models
  const detectProfileFacialLandmarks = useCallback(async (imagePath) => {
    try {
      console.log('🔍 Detecting facial landmarks for profile image:', imagePath);

      // Convert image URL to blob
      const response = await fetch(imagePath);
      if (!response.ok) {
        throw new Error(`Failed to fetch profile image: ${response.status} - ${imagePath}`);
      }

      const blob = await response.blob();

      // Create FormData for backend API (which will proxy to Python server)
      const formData = new FormData();
      
      // Get filename from path or create a default one
      let filename = 'profile.jpg';
      if (imagePath.includes('.')) {
        filename = imagePath.split('/').pop() || 'profile.jpg';
      }
      
      formData.append('file', blob, filename);

      // Call backend API endpoint (avoids CORS issues)
      const apiResponse = await fetch('http://localhost:7272/api/ai/facial-landmark?model=face_alignment', {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - browser will set it automatically with boundary for FormData
      });

      if (!apiResponse.ok) {
        const errorText = await apiResponse.text();
        throw new Error(`Facial landmark detection failed: ${apiResponse.status} - ${errorText}`);
      }

      const data = await apiResponse.json();

      if (data.success && data.landmarks) {
        console.log('✅ Facial landmarks detected:', Object.keys(data.landmarks).length, 'landmarks');
        return data.landmarks;
      } 
        console.warn('⚠️ Facial landmark detection returned no results');
        return null;
      
    } catch (error) {
      console.error('❌ Facial landmark detection error:', error);
      return null; // This will trigger the estimation fallback
    }
  }, []);

  // Get landmarks for both images (cephalometric for lateral, facial for profile)
  const getImageLandmarks = useCallback(async () => {
    const landmarks = { lateral: {}, profile: {} };

    // Get cephalometric landmarks for lateral image - try ML detection first
    if (selectedLateralImage) {
      // Construct proper URL for lateral image
      let lateralImageUrl = selectedLateralImage.path;
      if (selectedLateralImage.path.startsWith('/uploads/')) {
        const relativePath = selectedLateralImage.path.replace('/uploads/', '');
        lateralImageUrl = `http://localhost:7272/api/serve-upload?path=${encodeURIComponent(relativePath)}`;
      } else if (selectedLateralImage.path.startsWith('http://localhost:5001')) {
        lateralImageUrl = selectedLateralImage.path.replace('http://localhost:5001', 'http://localhost:7272');
      }
      
      const detectedCephalometricLandmarks = await detectCephalometricLandmarks(lateralImageUrl);
      if (detectedCephalometricLandmarks) {
        console.log('🔍 Detected cephalometric landmarks:', Object.keys(detectedCephalometricLandmarks));
        // Store all detected landmarks
        landmarks.lateral = { ...detectedCephalometricLandmarks };
        console.log('✅ Using ML-detected cephalometric landmarks for lateral image');
      } else {
        // Fallback to patient data
        console.log('⚠️ ML cephalometric detection failed, using patient data');
        if (patient?.cephalometricLandmarks) {
          landmarks.lateral = { ...patient.cephalometricLandmarks };
          console.log('📋 Using patient cephalometric landmarks:', Object.keys(landmarks.lateral));
        }
      }
    }

    // Get facial landmarks for profile image
    if (selectedProfileImage) {
      // Construct proper URL for profile image
      let profileImageUrl = selectedProfileImage.path;
      if (selectedProfileImage.path.startsWith('/uploads/')) {
        const relativePath = selectedProfileImage.path.replace('/uploads/', '');
        profileImageUrl = `http://localhost:7272/api/serve-upload?path=${encodeURIComponent(relativePath)}`;
      } else if (selectedProfileImage.path.startsWith('http://localhost:5001')) {
        profileImageUrl = selectedProfileImage.path.replace('http://localhost:5001', 'http://localhost:7272');
      }
      
      const facialLandmarks = await detectProfileFacialLandmarks(profileImageUrl);
      if (facialLandmarks) {
        console.log('🔍 Detected facial landmarks:', Object.keys(facialLandmarks));
        
        // Enhanced landmark normalization with numbered support
        const normalizedLandmarks = {};
        Object.keys(facialLandmarks).forEach(landmarkName => {
          const landmark = facialLandmarks[landmarkName];
          const normalizedName = landmarkName.toLowerCase().replace(/[_\s-]/g, '_');

          // Accept both 'landmark_30' and plain numeric keys '30'
          let numberMatch = landmarkName.match(/landmark_(\d+)/);
          if (!numberMatch) {
            // match plain numeric keys
            numberMatch = landmarkName.match(/^(\d+)$/);
          }

          if (numberMatch) {
            const landmarkIndex = parseInt(numberMatch[1], 10);
            console.log(`🔢 Processing numbered landmark: ${landmarkName} (index ${landmarkIndex})`);

            // Use dlib mapping to convert index to anatomical name
            switch (landmarkIndex) {
              case 30: // Nose tip
                normalizedLandmarks.landmark_30 = landmark;
                normalizedLandmarks.nose_tip = landmark;
                break;
              case 33: // Subnasale
                normalizedLandmarks.landmark_33 = landmark;
                normalizedLandmarks.Subnasale = landmark;
                break;
              case 8:  // Pogonion/chin
                normalizedLandmarks.landmark_8 = landmark;
                normalizedLandmarks.pogonion = landmark;
                normalizedLandmarks.menton = landmark;
                break;
              case 27: // Glabella
                normalizedLandmarks.landmark_27 = landmark;
                normalizedLandmarks.glabella = landmark;
                break;
              case 51: // Upper lip
                normalizedLandmarks.landmark_51 = landmark;
                normalizedLandmarks.upper_lip = landmark;
                break;
              case 57: // Lower lip
                normalizedLandmarks.landmark_57 = landmark;
                normalizedLandmarks.lower_lip = landmark;
                break;
              default:
                // Keep the numbered landmark for mapping
                normalizedLandmarks[`landmark_${landmarkIndex}`] = landmark;
                break;
            }
          } else {
            // Handle named landmarks (fallback for other naming schemes)
            if (normalizedName.includes('nose') && normalizedName.includes('tip')) {
              normalizedLandmarks.nose_tip = landmark;
            } else if (normalizedName.includes('subnasal') || normalizedName.includes('sn')) {
              normalizedLandmarks.Subnasale = landmark;
            } else if (normalizedName.includes('pogonion') || normalizedName.includes('chin')) {
              normalizedLandmarks.pogonion = landmark;
            } else if (normalizedName.includes('glabella') || (normalizedName.includes('forehead') && normalizedName.includes('center'))) {
              normalizedLandmarks.glabella = landmark;
            } else if (normalizedName.includes('nasion')) {
              normalizedLandmarks.nasion = landmark;
            } else if (normalizedName.includes('menton') || normalizedName.includes('chin_bottom')) {
              normalizedLandmarks.menton = landmark;
            } else if (normalizedName.includes('gnathion')) {
              normalizedLandmarks.gnathion = landmark;
            } else if (normalizedName.includes('upper_lip') || normalizedName.includes('lip_upper')) {
              normalizedLandmarks.upper_lip = landmark;
            } else if (normalizedName.includes('lower_lip') || normalizedName.includes('lip_lower')) {
              normalizedLandmarks.lower_lip = landmark;
            } else {
              // Keep original name for other landmarks
              normalizedLandmarks[normalizedName] = landmark;
            }
          }
        });
        
        landmarks.profile = normalizedLandmarks;
        console.log('✅ Using ML-detected facial landmarks (normalized with numbered support):', Object.keys(normalizedLandmarks));
      } else {
        // Fallback to estimation if ML detection fails
        console.log('⚠️ ML facial detection failed, using estimation');
        landmarks.profile = estimateProfileLandmarksFromLateral(landmarks.lateral);
        console.log('📋 Using estimated profile landmarks:', Object.keys(landmarks.profile));
      }
    }

    console.log('🎯 Final landmark sets for superimposition (raw):', {
      lateral: Object.keys(landmarks.lateral),
      profile: Object.keys(landmarks.profile)
    });

    // Normalize returned landmarks to pixel coords if they are normalized (0..1)
    const normalizeSet = (lmSet, img) => {
      if (!lmSet) return {};
      const out = {};
      Object.entries(lmSet).forEach(([key, lm]) => {
        if (!lm) return;
        if (!img || typeof lm.x !== 'number' || typeof lm.y !== 'number') {
          out[key] = lm;
          return;
        }
        if (lm.x <= 1 && lm.y <= 1) {
          out[key] = { x: lm.x * img.width, y: lm.y * img.height };
        } else {
          out[key] = { x: lm.x, y: lm.y };
        }
      });
      return out;
    };

    // Use loadedImages where possible to get actual image dimensions
    const lateralImgObj = loadedImages.lateral || null;
    const profileImgObj = loadedImages.profile || null;

    const normalizedLandmarks = {
      lateral: normalizeSet(landmarks.lateral, lateralImgObj),
      profile: normalizeSet(landmarks.profile, profileImgObj),
    };

    console.log('🎯 Final landmark sets for superimposition (normalized):', {
      lateral: Object.keys(normalizedLandmarks.lateral),
      profile: Object.keys(normalizedLandmarks.profile)
    });

    return normalizedLandmarks;
  }, [selectedLateralImage, selectedProfileImage, patient?.cephalometricLandmarks, detectCephalometricLandmarks, detectProfileFacialLandmarks, loadedImages.lateral, loadedImages.profile]);

  // Estimate profile landmarks from lateral cephalometric landmarks (fallback)
  const estimateProfileLandmarksFromLateral = (lateralLandmarks) => {
    const profileLms = {};

    // Anatomical relationships between lateral and profile views
    // These are approximate ratios based on cephalometric standards
    Object.keys(lateralLandmarks).forEach(landmarkId => {
      const lateralLm = lateralLandmarks[landmarkId];

      // Base estimation using anatomical proportions
      let estimatedX = lateralLm.x;
      let estimatedY = lateralLm.y;

      // Adjust based on specific landmark characteristics
      switch (landmarkId) {
        case 'N': // Nasion - similar position in both views
          estimatedX = lateralLm.x + 5; // Slight forward adjustment
          estimatedY = lateralLm.y - 2;
          break;

        case 'S': // Sella - cranial base point, similar in both views
          estimatedX = lateralLm.x + 3;
          estimatedY = lateralLm.y - 1;
          break;

        case 'Po': // Porion - ear point, may be visible in profile
          estimatedX = lateralLm.x - 8; // Behind in profile view
          estimatedY = lateralLm.y + 3;
          break;

        case 'Or': // Orbitale - eye point
          estimatedX = lateralLm.x + 12; // More forward in profile
          estimatedY = lateralLm.y - 5;
          break;

        case 'Ba': // Basion - similar to Sella
          estimatedX = lateralLm.x + 2;
          estimatedY = lateralLm.y + 1;
          break;

        case 'A': // Point A - maxillary point
          estimatedX = lateralLm.x + 15; // Forward in profile
          estimatedY = lateralLm.y - 3;
          break;

        case 'B': // Point B - mandibular point
          estimatedX = lateralLm.x + 10; // Forward in profile
          estimatedY = lateralLm.y + 2;
          break;

        case 'Pog': // Pogonion - chin point
          estimatedX = lateralLm.x + 8; // Forward in profile
          estimatedY = lateralLm.y + 4;
          break;

        case 'Gn': // Gnathion - similar to Pog
          estimatedX = lateralLm.x + 7;
          estimatedY = lateralLm.y + 5;
          break;

        case 'Me': // Menton - lowest chin point
          estimatedX = lateralLm.x + 6;
          estimatedY = lateralLm.y + 6;
          break;

        default:
          // For unknown landmarks, use small random variation
          estimatedX = lateralLm.x + (Math.random() * 10 - 5);
          estimatedY = lateralLm.y + (Math.random() * 10 - 5);
      }

      profileLms[landmarkId] = {
        x: estimatedX,
        y: estimatedY,
      };
    });

    return profileLms;
  };

  // Auto-select first images if available
  useEffect(() => {
    if (!selectedProfileImage && profileImages.length > 0) {
      setSelectedProfileImage(profileImages[0]);
    }
    if (!selectedLateralImage && lateralImages.length > 0) {
      setSelectedLateralImage(lateralImages[0]);
    }
  }, [profileImages, lateralImages, selectedProfileImage, selectedLateralImage]);

  // Auto-detect available landmarks from patient data
  useEffect(() => {
    if (patient?.cephalometricLandmarks) {
      const availableLmIds = Object.keys(patient.cephalometricLandmarks);
      if (availableLmIds.length > 0) {
        // Set default landmarks based on availability, preferring common ones
        const preferredLandmarks = ['N', 'S', 'Po', 'A', 'B', 'Pog'];
        const detectedLandmarks = preferredLandmarks.filter(id => availableLmIds.includes(id));

        if (detectedLandmarks.length >= 2) {
          setReferenceLandmarks(detectedLandmarks.slice(0, 4)); // Use up to 4 landmarks
        } else if (availableLmIds.length >= 2) {
          // Fallback to any available landmarks
          setReferenceLandmarks(availableLmIds.slice(0, 4));
        }

        console.log('Auto-detected landmarks:', detectedLandmarks.length > 0 ? detectedLandmarks : availableLmIds);
      }
    }
  }, [patient?.cephalometricLandmarks]);

  // Auto-adjust image sizes when images are loaded
  useEffect(() => {
    if (loadedImages.lateral && loadedImages.profile && imagesLoaded.lateral && imagesLoaded.profile) {
      // Calculate appropriate scale based on canvas size and image dimensions
      const canvas = canvasRef.current;
      if (canvas) {
        const rect = canvas.getBoundingClientRect();
        const canvasWidth = rect.width;
        const canvasHeight = rect.height;

        // Get actual image dimensions
        const lateralImg = loadedImages.lateral;
        const profileImg = loadedImages.profile;

        console.log('📏 Image dimensions:', {
          lateral: { width: lateralImg.width, height: lateralImg.height },
          profile: { width: profileImg.width, height: profileImg.height },
          canvas: { width: canvasWidth, height: canvasHeight }
        });

        // Calculate scale to fit both images properly within canvas
        // Use the larger of the two images as reference for better superimposition
        const maxImageWidth = Math.max(lateralImg.width, profileImg.width);
        const maxImageHeight = Math.max(lateralImg.height, profileImg.height);

        // Target size: 70% of canvas dimensions to leave room for controls
        const targetWidth = canvasWidth * 0.7;
        const targetHeight = canvasHeight * 0.7;

        // Calculate scale based on the larger image to ensure both fit
        const scaleX = targetWidth / maxImageWidth;
        const scaleY = targetHeight / maxImageHeight;
        const optimalScale = Math.min(scaleX, scaleY, 1.5); // Allow some upscale but limit to 150%

        // Apply the same scale to both images for proper superimposition
        const finalScale = Math.max(optimalScale, 0.2); // Minimum scale of 20%
        
        setLateralScale(finalScale);
        setProfileScale(finalScale);

        console.log('🎯 Auto-adjusted image scales:', {
          finalScale: finalScale.toFixed(3),
          targetSize: { width: targetWidth, height: targetHeight },
          maxImageSize: { width: maxImageWidth, height: maxImageHeight }
        });
      }
    }
  }, [loadedImages, imagesLoaded]);

  // BULLETPROOF image loading with guaranteed placeholders
  useEffect(() => {
    if (selectedProfileImage) {
      // Create immediate placeholder to ensure something is always visible
      const createPlaceholderImage = (width = 400, height = 500, text = '📷 تصویر پروفایل', color = '#e3f2fd') => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        // Background with gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, '#ffffff');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Border
        ctx.strokeStyle = '#1976d2';
        ctx.lineWidth = 3;
        ctx.strokeRect(0, 0, width, height);
        
        // Text
        ctx.fillStyle = '#1976d2';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(text, width / 2, height / 2 - 20);
        
        ctx.font = '12px Arial';
        ctx.fillText('سیستم فعال - در حال بارگذاری', width / 2, height / 2 + 5);
        ctx.fillText('Real image will replace this', width / 2, height / 2 + 25);
        
        const img = new Image();
        img.width = width;
        img.height = height;
        img.onload = () => {
          setLoadedImages(prev => ({ ...prev, profile: img }));
          setImagesLoaded(prev => ({ ...prev, profile: true }));
          console.log('✅ Placeholder image set for profile');
        };
        img.src = canvas.toDataURL();
        return img;
      };
      
      // Set placeholder IMMEDIATELY
      const placeholderImg = createPlaceholderImage();
      setLoadedImages(prev => ({ ...prev, profile: placeholderImg }));
      setImagesLoaded(prev => ({ ...prev, profile: true }));
      
      // Try to load real image in background
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        console.log('✅ Real profile image loaded, replacing placeholder');
        setLoadedImages(prev => ({ ...prev, profile: img }));
        setImagesLoaded(prev => ({ ...prev, profile: true }));
      };
      img.onerror = (error) => {
        console.warn('[SuperimposeView] Real profile image failed, keeping placeholder');
        // Placeholder is already set, no action needed
      };

      // Use serve-upload API endpoint for images
      if (selectedProfileImage.path?.startsWith('/uploads/')) {
        const relativePath = selectedProfileImage.path.replace('/uploads/', '');
        img.src = `http://localhost:7272/api/serve-upload?path=${encodeURIComponent(relativePath)}`;
        console.log('[SuperimposeView] Loading profile image via API:', img.src);
      } else {
        img.src = `http://localhost:5001${selectedProfileImage.path}`;
      }
    } else {
      setImagesLoaded(prev => ({ ...prev, profile: false }));
      setLoadedImages(prev => ({ ...prev, profile: null }));
    }
  }, [selectedProfileImage]);

  useEffect(() => {
    if (selectedLateralImage) {
      // Create immediate placeholder to ensure something is always visible
      const createPlaceholderImage = (width = 500, height = 400, text = '📷 تصویر لترال', color = '#fff3e0') => {
        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;
        const ctx = canvas.getContext('2d');
        
        // Background with gradient
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, color);
        gradient.addColorStop(1, '#ffffff');
        ctx.fillStyle = gradient;
        ctx.fillRect(0, 0, width, height);
        
        // Border
        ctx.strokeStyle = '#f57c00';
        ctx.lineWidth = 3;
        ctx.strokeRect(0, 0, width, height);
        
        // Text
        ctx.fillStyle = '#f57c00';
        ctx.font = 'bold 16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(text, width / 2, height / 2 - 20);
        
        ctx.font = '12px Arial';
        ctx.fillText('سیستم فعال - در حال بارگذاری', width / 2, height / 2 + 5);
        ctx.fillText('Real image will replace this', width / 2, height / 2 + 25);
        
        const img = new Image();
        img.width = width;
        img.height = height;
        img.onload = () => {
          setLoadedImages(prev => ({ ...prev, lateral: img }));
          setImagesLoaded(prev => ({ ...prev, lateral: true }));
          console.log('✅ Placeholder image set for lateral');
        };
        img.src = canvas.toDataURL();
        return img;
      };
      
      // Set placeholder IMMEDIATELY
      const placeholderImg = createPlaceholderImage();
      setLoadedImages(prev => ({ ...prev, lateral: placeholderImg }));
      setImagesLoaded(prev => ({ ...prev, lateral: true }));
      
      // Try to load real image in background
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        console.log('✅ Real lateral image loaded, replacing placeholder');
        setLoadedImages(prev => ({ ...prev, lateral: img }));
        setImagesLoaded(prev => ({ ...prev, lateral: true }));
      };
      img.onerror = (error) => {
        console.warn('[SuperimposeView] Real lateral image failed, keeping placeholder');
        // Placeholder is already set, no action needed
      };

      // Use serve-upload API endpoint for images
      if (selectedLateralImage.path?.startsWith('/uploads/')) {
        const relativePath = selectedLateralImage.path.replace('/uploads/', '');
        img.src = `http://localhost:7272/api/serve-upload?path=${encodeURIComponent(relativePath)}`;
        console.log('[SuperimposeView] Loading lateral image via API:', img.src);
      } else {
        img.src = `http://localhost:5001${selectedLateralImage.path}`;
      }
    } else {
      setImagesLoaded(prev => ({ ...prev, lateral: false }));
      setLoadedImages(prev => ({ ...prev, lateral: null }));
    }
  }, [selectedLateralImage]);

  // Draw superimposed images (optimized to avoid flickering)
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();

    // Set canvas size
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw background grid for reference
    ctx.strokeStyle = '#f0f0f0';
    ctx.lineWidth = 1;
    const gridSize = 20;
    for (let x = 0; x < rect.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, rect.height);
      ctx.stroke();
    }
    for (let y = 0; y < rect.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(rect.width, y);
      ctx.stroke();
    }

    // Draw center cross
    ctx.strokeStyle = '#ff0000';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(rect.width / 2, 0);
    ctx.lineTo(rect.width / 2, rect.height);
    ctx.moveTo(0, rect.height / 2);
    ctx.lineTo(rect.width, rect.height / 2);
    ctx.stroke();

    // Draw lateral cephalometric image (background) - use cached image
    if (loadedImages.lateral && imagesLoaded.lateral) {
      const lateralImg = loadedImages.lateral;
      const scale = lateralScale;
      const scaledWidth = lateralImg.width * scale;
      const scaledHeight = lateralImg.height * scale;
      const x = (rect.width - scaledWidth) / 2 + lateralPosition.x;
      const y = (rect.height - scaledHeight) / 2 + lateralPosition.y;

      console.log('🎨 Drawing lateral image:', {
        originalSize: { width: lateralImg.width, height: lateralImg.height },
        scale: scale.toFixed(3),
        scaledSize: { width: scaledWidth, height: scaledHeight },
        position: { x: x.toFixed(1), y: y.toFixed(1) }
      });

      ctx.globalAlpha = lateralOpacity;
      ctx.drawImage(lateralImg, x, y, scaledWidth, scaledHeight);
      ctx.globalAlpha = 1;
    }

    // Draw profile image on top - use cached image
    if (loadedImages.profile && imagesLoaded.profile) {
      const profileImg = loadedImages.profile;
      const profileScaleVal = profileScale;
      const profileScaledWidth = profileImg.width * profileScaleVal;
      const profileScaledHeight = profileImg.height * profileScaleVal;
      const profileX = (rect.width - profileScaledWidth) / 2 + profilePosition.x;
      const profileY = (rect.height - profileScaledHeight) / 2 + profilePosition.y;

      console.log('🎨 Drawing profile image:', {
        originalSize: { width: profileImg.width, height: profileImg.height },
        scale: profileScaleVal.toFixed(3),
        scaledSize: { width: profileScaledWidth, height: profileScaledHeight },
        position: { x: profileX.toFixed(1), y: profileY.toFixed(1) }
      });

      ctx.globalAlpha = profileOpacity;
      ctx.drawImage(profileImg, profileX, profileY, profileScaledWidth, profileScaledHeight);
      ctx.globalAlpha = 1;
    }

    // Draw landmarks if enabled
    if (showLandmarks) {
      // Draw lateral landmarks (red dots)
      if (lateralLandmarks && loadedImages.lateral && imagesLoaded.lateral) {
        const lateralImg = loadedImages.lateral;
        const lateralScaleVal = lateralScale;
        const lateralImgX = (canvas.width - lateralImg.width * lateralScaleVal) / 2 + lateralPosition.x;
        const lateralImgY = (canvas.height - lateralImg.height * lateralScaleVal) / 2 + lateralPosition.y;

        Object.entries(lateralLandmarks).forEach(([landmarkId, landmark]) => {
          const x = lateralImgX + landmark.x * lateralScaleVal;
          const y = lateralImgY + landmark.y * lateralScaleVal;

          // Draw red dot for lateral landmarks
          ctx.fillStyle = 'red';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();

          // Draw white border
          ctx.strokeStyle = 'white';
          ctx.lineWidth = 2;
          ctx.stroke();

          // Draw label
          ctx.fillStyle = 'red';
          ctx.font = '12px Arial';
          ctx.fillText(landmarkId, x + 6, y - 6);
        });
      }

      // Draw profile landmarks (blue dots)
      if (profileLandmarks && loadedImages.profile && imagesLoaded.profile) {
        const profileImg = loadedImages.profile;
        const profileScaleVal = profileScale;
        const profileImgX = (canvas.width - profileImg.width * profileScaleVal) / 2 + profilePosition.x;
        const profileImgY = (canvas.height - profileImg.height * profileScaleVal) / 2 + profilePosition.y;

        Object.entries(profileLandmarks).forEach(([landmarkId, landmark]) => {
          const x = profileImgX + landmark.x * profileScaleVal;
          const y = profileImgY + landmark.y * profileScaleVal;

          // Draw blue dot for profile landmarks
          ctx.fillStyle = 'blue';
          ctx.beginPath();
          ctx.arc(x, y, 4, 0, 2 * Math.PI);
          ctx.fill();

          // Draw white border
          ctx.strokeStyle = 'white';
          ctx.lineWidth = 2;
          ctx.stroke();

          // Draw label
          ctx.fillStyle = 'blue';
          ctx.font = '12px Arial';
          ctx.fillText(landmarkId, x + 6, y + 6);
        });
      }
    }
  }, [
    loadedImages,
    imagesLoaded,
    profileOpacity,
    lateralOpacity,
    profilePosition,
    lateralPosition,
    profileScale,
    lateralScale,
    showLandmarks,
    lateralLandmarks,
    profileLandmarks,
  ]);

  // Redraw when dependencies change
  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Handle mouse events for dragging
  const handleMouseDown = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    // Determine which image to drag based on position and which one is selected
    // For simplicity, we'll drag the profile image if it's selected, otherwise lateral
    if (selectedProfileImage) {
      setIsDragging('profile');
    } else if (selectedLateralImage) {
      setIsDragging('lateral');
    }

    setDragStart({ x, y });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const deltaX = x - dragStart.x;
    const deltaY = y - dragStart.y;

    if (isDragging === 'profile') {
      setProfilePosition(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY,
      }));
    } else if (isDragging === 'lateral') {
      setLateralPosition(prev => ({
        x: prev.x + deltaX,
        y: prev.y + deltaY,
      }));
    }

    setDragStart({ x, y });
  };

  const handleMouseUp = () => {
    setIsDragging(null);
  };

  // Reset positions
  const resetPositions = () => {
    setProfilePosition({ x: 0, y: 0 });
    setLateralPosition({ x: 0, y: 0 });
    setProfileScale(1);
    setLateralScale(1);
    setProfileOpacity(0.7);
    setLateralOpacity(0.7);
  };

  // Retry refs to avoid repeated timeouts and infinite loops
  const retryScheduledRef = useRef(false);
  const retryCountRef = useRef(0);

  // Enhanced landmark-based alignment functions
  const calculateLandmarkBasedAlignment = useCallback(async () => {
    if (!useLandmarkAlignment || !selectedProfileImage || !selectedLateralImage) return;

    try {
      console.log('🎯 Starting enhanced landmark-based alignment...');

      // Wait for images to be available - necessary for pixel-based alignment
      // Only proceed when real images (not placeholders) are loaded. Placeholders use data: URLs.
      const lateralReal = loadedImages.lateral && loadedImages.lateral.src && !String(loadedImages.lateral.src).startsWith('data:');
      const profileReal = loadedImages.profile && loadedImages.profile.src && !String(loadedImages.profile.src).startsWith('data:');

      if (!lateralReal || !profileReal) {
        // Limit retry attempts to prevent infinite loops
        if (retryCountRef.current < 5) {
          console.warn(`Images not ready for alignment, retrying (attempt ${retryCountRef.current + 1}/5)`);
          retryCountRef.current += 1;
          
          if (!retryScheduledRef.current) {
            retryScheduledRef.current = true;
            setTimeout(() => {
              retryScheduledRef.current = false;
              calculateLandmarkBasedAlignment();
            }, 400);
          }
        } else {
          console.warn('❌ Maximum retry attempts reached for image loading, proceeding with placeholders');
          // Reset retry counter for next time
          retryCountRef.current = 0;
        }
        return;
      }

      // Get landmarks for both images using ML detection
      const imageLandmarks = await getImageLandmarks();

      const lateralLms = imageLandmarks.lateral;
      const profileLms = imageLandmarks.profile;

      if (Object.keys(lateralLms).length < 2) {
        console.warn('❌ Not enough lateral landmarks for alignment - need at least 2 landmarks');
        return;
      }

      if (Object.keys(profileLms).length < 2) {
        console.warn('❌ Not enough profile landmarks for alignment - need at least 2 landmarks');
        return;
      }

      // Calculate optimal transform using the enhanced algorithm
      const transform = calculateOptimalTransform(profileLms, lateralLms);

      // Validate transform quality
      if (transform.alignmentQuality.averageError > 50) {
        console.warn('⚠️ High alignment error detected:', transform.alignmentQuality.averageError.toFixed(2), 'pixels');
      }

      // Apply the calculated transform to profile image
      setProfileScale(transform.scale);

      // Convert transform (which is in image coordinate units) to canvas pixel offsets
      try {
        const canvas = canvasRef.current;
        const rect = canvas ? canvas.getBoundingClientRect() : { width: 800, height: 600 };
        const lateralImg = loadedImages.lateral;
        const profileImg = loadedImages.profile;

        // Normalize landmarks to image pixel coordinates (handle APIs that return normalized 0..1 coords)
        const toPixel = (lm, img) => {
          if (!lm || !img) return { x: 0, y: 0 };
          // If coordinates appear normalized (<= 1), convert to pixels
          if (lm.x <= 1 && lm.y <= 1) {
            return { x: lm.x * img.width, y: lm.y * img.height };
          }
          return { x: lm.x, y: lm.y };
        };

        const lateralPixelLMs = Object.fromEntries(Object.entries(lateralLms || {}).map(([k, lm]) => [k, toPixel(lm, lateralImg)]));
        const profilePixelLMs = Object.fromEntries(Object.entries(profileLms || {}).map(([k, lm]) => [k, toPixel(lm, profileImg)]));

        // Compute centroids in pixel coordinates
        const lateralCentroid = Object.values(lateralPixelLMs).reduce((acc, lm) => ({ x: acc.x + lm.x, y: acc.y + lm.y }), { x: 0, y: 0 });
        lateralCentroid.x /= Math.max(1, Object.keys(lateralPixelLMs).length);
        lateralCentroid.y /= Math.max(1, Object.keys(lateralPixelLMs).length);

        const profileCentroid = Object.values(profilePixelLMs).reduce((acc, lm) => ({ x: acc.x + lm.x, y: acc.y + lm.y }), { x: 0, y: 0 });
        profileCentroid.x /= Math.max(1, Object.keys(profilePixelLMs).length);
        profileCentroid.y /= Math.max(1, Object.keys(profilePixelLMs).length);

        // Lateral top-left in canvas coords
        const lateralImgX = (rect.width - lateralImg.width * lateralScale) / 2 + lateralPosition.x;
        const lateralImgY = (rect.height - lateralImg.height * lateralScale) / 2 + lateralPosition.y;

        // Profile base top-left if centered
        const profileBaseX = (rect.width - profileImg.width * transform.scale) / 2;
        const profileBaseY = (rect.height - profileImg.height * transform.scale) / 2;

        // Desired profile top-left so that centroids align on canvas
        const profilePosX = lateralImgX + lateralCentroid.x * lateralScale - profileCentroid.x * transform.scale - profileBaseX;
        const profilePosY = lateralImgY + lateralCentroid.y * lateralScale - profileCentroid.y * transform.scale - profileBaseY;

        setProfilePosition({ x: profilePosX, y: profilePosY });

        console.log('✅ Applied enhanced ML-based landmark alignment (canvas coords):', {
          lateralLandmarks: Object.keys(lateralLms),
          profileLandmarks: Object.keys(profileLms),
          scale: transform.scale.toFixed(3),
          profilePositionX: profilePosX.toFixed(1),
          profilePositionY: profilePosY.toFixed(1),
          alignmentQuality: transform.alignmentQuality
        });
      } catch (err) {
        // Fallback: set raw transform offsets if anything fails
        setProfilePosition({ x: transform.offsetX, y: transform.offsetY });
        console.warn('Alignment to canvas coords failed, using raw offsets', err);
      }


    } catch (error) {
      console.error('❌ Error in enhanced landmark-based alignment:', error);
      // Enhanced fallback with better error handling
      console.log('🔄 Falling back to enhanced estimation-based alignment...');
      
      const patientLandmarks = patient?.cephalometricLandmarks || {};
      if (Object.keys(patientLandmarks).length >= 2) {
        const profileLms = estimateProfileLandmarksFromLateral(patientLandmarks);
        const transform = calculateOptimalTransform(profileLms, patientLandmarks);
        setProfileScale(transform.scale);
        setProfilePosition({ x: transform.offsetX, y: transform.offsetY });
        console.log('✅ Applied enhanced fallback estimation alignment');
      } else {
        console.error('❌ Insufficient landmarks for fallback alignment');
      }
    }
  }, [useLandmarkAlignment, selectedProfileImage, selectedLateralImage, patient, loadedImages, calculateOptimalTransform, getImageLandmarks, lateralPosition, lateralScale]);

  // Enhanced landmark mapping with dlib index support
  const getLandmarkMapping = useCallback(() => ({
      // dlib indices -> Profile landmark names -> Lateral cephalometric landmarks
      30: { profile: 'nose_tip', lateral: 'Pn' },     // landmark_30 = nose tip
      33: { profile: 'Subnasale', lateral: 'Sn' },    // landmark_33 = subnasale
      8: { profile: 'pogonion', lateral: 'Pog' },     // landmark_8 = pogonion/chin
      27: { profile: 'glabella', lateral: 'N' },      // landmark_27 = glabella/nasion
      51: { profile: 'upper_lip', lateral: 'A' },     // landmark_51 = upper lip
      57: { profile: 'lower_lip', lateral: 'B' },     // landmark_57 = lower lip
      
      // Direct anatomical name mappings
      'nose_tip': 'Pn',     // Nose tip in profile = Pronasale in lateral
      'Subnasale': 'Sn',    // Subnasale in profile = Subnasale in lateral
      'pogonion': 'Pog',    // Pogonion in profile = Pogonion in lateral
      'glabella': 'N',      // Glabella in profile = Nasion in lateral
      'nasion': 'N',        // Nasion in profile = Nasion in lateral
      'menton': 'Me',       // Menton in profile = Menton in lateral
      'gnathion': 'Gn',     // Gnathion in profile = Gnathion in lateral
      'upper_lip': 'A',     // Upper lip area = Point A in lateral
      'lower_lip': 'B',     // Lower lip area = Point B in lateral
      'chin': 'Pog',        // Chin in profile = Pogonion in lateral
    }), []);

  // Calculate distance between two landmarks
  const calculateLandmarkDistance = (landmark1, landmark2) => {
    if (!landmark1 || !landmark2) return 0;
    const dx = landmark1.x - landmark2.x;
    const dy = landmark1.y - landmark2.y;
    return Math.sqrt(dx * dx + dy * dy);
  };

  // Calculate scale ratio based on landmark distances (enhanced with numbered landmark support)
  const calculateScaleRatio = useCallback((profileLandmarks, lateralLandmarks) => {
    const mapping = getLandmarkMapping();
    const scaleRatios = [];

    // Helper function to extract landmark from various formats
    const getProfileLandmark = (key) => {
      // First try direct name lookup
      if (profileLandmarks[key]) return profileLandmarks[key];
      
      // Try numbered landmark format (landmark_30, etc.)
      if (typeof key === 'number') {
        const numKey = `landmark_${key}`;
        if (profileLandmarks[numKey]) return profileLandmarks[numKey];
      }
      
      return null;
    };

    // Find mapped landmark pairs for scale calculation
    Object.keys(mapping).forEach(key => {
      const mappingInfo = mapping[key];
      
      if (typeof mappingInfo === 'string') {
        // Old format: direct name mapping
        const profileName = key;
        const lateralName = mappingInfo;
        if (profileLandmarks[profileName] && lateralLandmarks[lateralName]) {
          scaleRatios.push(1); // Default ratio for single landmark pairs
        }
      } else {
        // New format: { profile: name, lateral: name }
        const profileName = mappingInfo.profile;
        const lateralName = mappingInfo.lateral;
        
        const profileLm = getProfileLandmark(key) || profileLandmarks[profileName];
        if (profileLm && lateralLandmarks[lateralName]) {
          // Calculate distance to another landmark for scale ratio
          Object.keys(mapping).forEach(otherKey => {
            if (otherKey !== key) {
              const otherMappingInfo = mapping[otherKey];
              const otherProfileName = typeof otherMappingInfo === 'string' ? otherKey : otherMappingInfo.profile;
              const otherLateralName = typeof otherMappingInfo === 'string' ? otherMappingInfo : otherMappingInfo.lateral;
              
              const otherProfileLm = getProfileLandmark(otherKey) || profileLandmarks[otherProfileName];
              if (otherProfileLm && lateralLandmarks[otherLateralName]) {
                const profileDistance = calculateLandmarkDistance(profileLm, otherProfileLm);
                const lateralDistance = calculateLandmarkDistance(lateralLandmarks[lateralName], lateralLandmarks[otherLateralName]);
                
                if (profileDistance > 0 && lateralDistance > 0) {
                  const ratio = lateralDistance / profileDistance;
                  // Filter out extreme ratios
                  if (ratio >= 0.3 && ratio <= 3.0) {
                    scaleRatios.push(ratio);
                  }
                }
              }
            }
          });
        }
      }
    });

    // Return median scale ratio for robustness
    if (scaleRatios.length > 0) {
      scaleRatios.sort((a, b) => a - b);
      const medianIndex = Math.floor(scaleRatios.length / 2);
      return scaleRatios.length % 2 === 0
        ? (scaleRatios[medianIndex - 1] + scaleRatios[medianIndex]) / 2
        : scaleRatios[medianIndex];
    }

    return 1; // Default scale if no valid ratios found
  }, [getLandmarkMapping]);

  // Enhanced optimal transform calculation with numbered landmark support
  const calculateOptimalTransform = useCallback((profileLandmarks, lateralLandmarks) => {
    const mapping = getLandmarkMapping();
    
    // Helper function to extract landmark from various formats
    const getProfileLandmark = (key) => {
      // First try direct name lookup
      if (profileLandmarks[key]) return profileLandmarks[key];
      
      // Try numbered landmark format (landmark_30, etc.)
      if (typeof key === 'number') {
        const numKey = `landmark_${key}`;
        if (profileLandmarks[numKey]) return profileLandmarks[numKey];
      }
      
      return null;
    };
    
    // Find mapped landmark pairs
    const mappedPairs = [];
    Object.keys(mapping).forEach(key => {
      const mappingInfo = mapping[key];
      
      if (typeof mappingInfo === 'string') {
        // Old format: direct name mapping
        const profileName = key;
        const lateralName = mappingInfo;
        const profileLm = profileLandmarks[profileName];
        if (profileLm && lateralLandmarks[lateralName]) {
          mappedPairs.push({
            profile: profileLm,
            lateral: lateralLandmarks[lateralName],
            profileName: profileName,
            lateralName: lateralName
          });
        }
      } else {
        // New format: { profile: name, lateral: name }
        const profileName = mappingInfo.profile;
        const lateralName = mappingInfo.lateral;
        
        const profileLm = getProfileLandmark(key) || profileLandmarks[profileName];
        if (profileLm && lateralLandmarks[lateralName]) {
          mappedPairs.push({
            profile: profileLm,
            lateral: lateralLandmarks[lateralName],
            profileName: profileName,
            lateralName: lateralName
          });
        }
      }
    });

    if (mappedPairs.length < 2) {
      console.warn('❌ Insufficient mapped landmarks for alignment (need at least 2)');
      console.log('Available profile landmarks:', Object.keys(profileLandmarks));
      console.log('Available lateral landmarks:', Object.keys(lateralLandmarks));
      console.log('Mapping keys:', Object.keys(mapping));
      return { scale: 1, offsetX: 0, offsetY: 0, mappedPairs: [] };
    }

    // Calculate scale ratio based on landmark distances
    const scaleRatio = calculateScaleRatio(profileLandmarks, lateralLandmarks);
    
    console.log('🧮 Calculated scale ratio:', scaleRatio.toFixed(3));

    // Calculate optimal alignment using Procrustes analysis approach
    // 1. Calculate centroids
    const profileCentroid = mappedPairs.reduce(
      (acc, pair) => ({
        x: acc.x + pair.profile.x,
        y: acc.y + pair.profile.y,
      }),
      { x: 0, y: 0 }
    );
    profileCentroid.x /= mappedPairs.length;
    profileCentroid.y /= mappedPairs.length;

    const lateralCentroid = mappedPairs.reduce(
      (acc, pair) => ({
        x: acc.x + pair.lateral.x,
        y: acc.y + pair.lateral.y,
      }),
      { x: 0, y: 0 }
    );
    lateralCentroid.x /= mappedPairs.length;
    lateralCentroid.y /= mappedPairs.length;

    // 2. Calculate optimal offset
    // Align centroids after scaling
    const offsetX = lateralCentroid.x - (profileCentroid.x * scaleRatio);
    const offsetY = lateralCentroid.y - (profileCentroid.y * scaleRatio);

    // 3. Validate alignment quality
    const alignmentErrors = mappedPairs.map(pair => {
      const expectedX = pair.profile.x * scaleRatio + offsetX;
      const expectedY = pair.profile.y * scaleRatio + offsetY;
      const error = Math.sqrt(
        Math.pow(expectedX - pair.lateral.x, 2) +
        Math.pow(expectedY - pair.lateral.y, 2)
      );
      return error;
    });

    const avgError = alignmentErrors.reduce((sum, error) => sum + error, 0) / alignmentErrors.length;
    const maxError = Math.max(...alignmentErrors);

    console.log('🎯 Alignment quality:', {
      averageError: avgError.toFixed(2),
      maxError: maxError.toFixed(2),
      mappedLandmarks: mappedPairs.length,
      scaleRatio: scaleRatio.toFixed(3),
      mappedPairs: mappedPairs.map(p => ({ profile: p.profileName, lateral: p.lateralName }))
    });

    // Return the transform
    const transform = {
      scale: Math.min(Math.max(scaleRatio, 0.1), 3),
      offsetX,
      offsetY,
      alignmentQuality: {
        averageError: avgError,
        maxError: maxError,
        mappedCount: mappedPairs.length
      },
      mappedPairs: mappedPairs
    };

    return transform;
  }, [getLandmarkMapping, calculateScaleRatio]);

  // Auto-align when images or landmarks change
  useEffect(() => {
    if (useLandmarkAlignment && alignmentMode === 'auto') {
      calculateLandmarkBasedAlignment();
    }
  }, [calculateLandmarkBasedAlignment, useLandmarkAlignment, alignmentMode]);

  // Function to detect landmarks for both images
  const detectLandmarksForBothImages = useCallback(async () => {
    if (!selectedLateralImage && !selectedProfileImage) {
      console.warn('No images selected for landmark detection');
      return;
    }

    try {
      console.log('🔍 Starting landmark detection for both images...');

      const imageLandmarks = await getImageLandmarks();

      // Update state with detected landmarks
      setLateralLandmarks(imageLandmarks.lateral);
      setProfileLandmarks(imageLandmarks.profile);

      console.log('✅ Landmarks detected and applied:', {
        lateral: Object.keys(imageLandmarks.lateral).length,
        profile: Object.keys(imageLandmarks.profile).length
      });

      // Trigger alignment if auto mode is enabled
      if (useLandmarkAlignment && alignmentMode === 'auto') {
        setTimeout(() => {
          calculateLandmarkBasedAlignment();
        }, 100); // Small delay to ensure state is updated
      }

    } catch (error) {
      console.error('❌ Error detecting landmarks:', error);
    }
  }, [selectedLateralImage, selectedProfileImage, getImageLandmarks, useLandmarkAlignment, alignmentMode, calculateLandmarkBasedAlignment]);

  // Auto-detect landmarks when images are selected
  useEffect(() => {
    if (selectedLateralImage || selectedProfileImage) {
      // Small delay to ensure images are loaded
      const timer = setTimeout(() => {
        detectLandmarksForBothImages();
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [selectedLateralImage, selectedProfileImage, detectLandmarksForBothImages]);

  // Handle landmark selection
  const handleLandmarkToggle = (landmarkId) => {
    setReferenceLandmarks(prev => {
      if (prev.includes(landmarkId)) {
        return prev.filter(id => id !== landmarkId);
      } 
        return [...prev, landmarkId];
      
    });
  };

  // Manual alignment trigger
  const applyLandmarkAlignment = () => {
    calculateLandmarkBasedAlignment();
  };

  // Export canvas as image
  const exportSuperimposedImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const link = document.createElement('a');
    link.download = `superimposed-${patient?.name || 'patient'}.png`;
    link.href = canvas.toDataURL();
    link.click();
  };

  return (
    <Stack spacing={3}>
      {/* AI Server Status Warning */}
      <Alert severity="info" sx={{ mb: 2 }}>
        <Typography variant="body2">
          <strong>✅ سیستم یکپارچه شناسایی لندمارک:</strong> این بخش از هر دو سیستم تشخیص لندمارک استفاده می‌کند:
          <br />
          • <strong>لندمارک‌های سفالومتری:</strong> از مدل‌های Aariz برای تصویر لترال
          <br />
          • <strong>لندمارک‌های چهره‌ای:</strong> از مدل‌های پیشرفته برای تصویر پروفایل
          <br />
          • <strong>تراز خودکار:</strong> تصاویر بر اساس لندمارک‌های مشترک تراز می‌شوند
          <br />
          <em>نکته: سرور Python باید روی پورت 5001 فعال باشد</em>
        </Typography>
      </Alert>

      {/* Landmark-based Alignment Controls */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            تنظیمات تراز لندمارک‌ها
          </Typography>

          <Stack spacing={3}>
            {/* Alignment Mode */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                حالت تراز
              </Typography>
              <Stack direction="row" spacing={2} alignItems="center">
                <Button
                  variant={alignmentMode === 'auto' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setAlignmentMode('auto')}
                >
                  خودکار
                </Button>
                <Button
                  variant={alignmentMode === 'manual' ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => setAlignmentMode('manual')}
                >
                  دستی
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={detectLandmarksForBothImages}
                  disabled={!selectedProfileImage && !selectedLateralImage}
                  startIcon={<Iconify icon="solar:cpu-bold" />}
                >
                  شناسایی لندمارک‌ها
                </Button>
                <Button
                  variant="outlined"
                  size="small"
                  onClick={applyLandmarkAlignment}
                  disabled={!useLandmarkAlignment || !selectedProfileImage || !selectedLateralImage}
                  startIcon={<Iconify icon="solar:magic-stick-3-bold" />}
                >
                  اعمال تراز
                </Button>
              </Stack>
            </Box>

            {/* Landmark Selection */}
            <Box>
              <Typography variant="subtitle2" gutterBottom>
                لندمارک‌های مرجع انتخاب شده ({referenceLandmarks.length})
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {availableLandmarks.map((landmark) => {
                  const isSelected = referenceLandmarks.includes(landmark.id);
                  return (
                    <Chip
                      key={landmark.id}
                      label={`${landmark.id} - ${landmark.name}`}
                      size="small"
                      color={isSelected ? 'primary' : 'default'}
                      variant={isSelected ? 'filled' : 'outlined'}
                      onClick={() => handleLandmarkToggle(landmark.id)}
                      sx={{ mb: 1 }}
                    />
                  );
                })}
              </Box>
              <Typography variant="caption" color="text.secondary">
                حداقل ۲ لندمارک برای تراز خودکار انتخاب کنید
              </Typography>
            </Box>

            {/* Alignment Status */}
            {useLandmarkAlignment && selectedProfileImage && selectedLateralImage && (
              <Alert severity="info" sx={{ mt: 1 }}>
                <Typography variant="body2">
                  {alignmentMode === 'auto'
                    ? 'تراز خودکار فعال است. تصاویر بر اساس لندمارک‌های انتخاب شده تراز می‌شوند.'
                    : 'تراز دستی فعال است. از دکمه "اعمال تراز" برای تراز کردن استفاده کنید.'
                  }
                </Typography>
              </Alert>
            )}
          </Stack>
        </CardContent>
      </Card>

      {/* Manual Controls */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            تنظیمات دستی سوپرایمپوز
          </Typography>

          <Grid container spacing={3}>
            {/* Image Selection */}
            <Grid item xs={12} md={6}>
              <Stack spacing={2}>
                <FormControl fullWidth>
                  <InputLabel>تصویر پروفایل</InputLabel>
                  <Select
                    value={selectedProfileImage?.id || ''}
                    label="تصویر پروفایل"
                    onChange={(e) => {
                      const image = profileImages.find(img => img.id === e.target.value);
                      setSelectedProfileImage(image);
                    }}
                  >
                    {profileImages.map((image) => (
                      <MenuItem key={image.id} value={image.id}>
                        {image.originalName || `تصویر ${image.id}`}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <FormControl fullWidth>
                  <InputLabel>تصویر لترال سفالومتری</InputLabel>
                  <Select
                    value={selectedLateralImage?.id || ''}
                    label="تصویر لترال سفالومتری"
                    onChange={(e) => {
                      const image = lateralImages.find(img => img.id === e.target.value);
                      setSelectedLateralImage(image);
                    }}
                  >
                    {lateralImages.map((image) => (
                      <MenuItem key={image.id} value={image.id}>
                        {image.originalName || `تصویر ${image.id}`}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
              </Stack>
            </Grid>

            {/* Opacity Controls */}
            <Grid item xs={12} md={6}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    شفافیت تصویر پروفایل: {Math.round(profileOpacity * 100)}%
                  </Typography>
                  <Slider
                    value={profileOpacity}
                    onChange={(e, value) => setProfileOpacity(value)}
                    min={0}
                    max={1}
                    step={0.1}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                  />
                </Box>

                <Box>
                  <Typography variant="body2" gutterBottom>
                    شفافیت تصویر لترال: {Math.round(lateralOpacity * 100)}%
                  </Typography>
                  <Slider
                    value={lateralOpacity}
                    onChange={(e, value) => setLateralOpacity(value)}
                    min={0}
                    max={1}
                    step={0.1}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                  />
                </Box>
              </Stack>
            </Grid>

            {/* Scale Controls */}
            <Grid item xs={12} md={6}>
              <Stack spacing={2}>
                <Box>
                  <Typography variant="body2" gutterBottom>
                    اندازه تصویر پروفایل: {Math.round(profileScale * 100)}%
                  </Typography>
                  <Slider
                    value={profileScale}
                    onChange={(e, value) => setProfileScale(value)}
                    min={0.1}
                    max={3}
                    step={0.1}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                  />
                </Box>

                <Box>
                  <Typography variant="body2" gutterBottom>
                    اندازه تصویر لترال: {Math.round(lateralScale * 100)}%
                  </Typography>
                  <Slider
                    value={lateralScale}
                    onChange={(e, value) => setLateralScale(value)}
                    min={0.1}
                    max={3}
                    step={0.1}
                    valueLabelDisplay="auto"
                    valueLabelFormat={(value) => `${Math.round(value * 100)}%`}
                  />
                </Box>
              </Stack>
            </Grid>

            {/* Action Buttons */}
            <Grid item xs={12} md={6}>
              <Stack spacing={2} direction="row">
                <Button
                  variant="outlined"
                  onClick={resetPositions}
                  startIcon={<Iconify icon="solar:refresh-bold" />}
                >
                  بازنشانی
                </Button>

                <Button
                  variant="contained"
                  onClick={exportSuperimposedImage}
                  startIcon={<Iconify icon="solar:download-bold" />}
                  disabled={!selectedProfileImage && !selectedLateralImage}
                >
                  ذخیره تصویر
                </Button>
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Canvas */}
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              سوپرایمپوز تصاویر
            </Typography>
            <Stack direction="row" spacing={2} alignItems="center">
              <Button
                variant={showLandmarks ? 'contained' : 'outlined'}
                size="small"
                onClick={() => setShowLandmarks(!showLandmarks)}
                startIcon={<Iconify icon="solar:eye-bold" />}
              >
                {showLandmarks ? 'پنهان کردن لندمارک‌ها' : 'نمایش لندمارک‌ها'}
              </Button>
              <Typography variant="body2" color="text.secondary">
                برای جابجایی تصاویر، روی آن‌ها کلیک کرده و بکشید
              </Typography>
            </Stack>
          </Box>

          {/* Landmark Status */}
          {showLandmarks && (lateralLandmarks || profileLandmarks) && (
            <Box sx={{ mb: 2, p: 2, bgcolor: 'grey.50', borderRadius: 1 }}>
              <Typography variant="subtitle2" gutterBottom>
                وضعیت شناسایی لندمارک‌ها
              </Typography>
              <Stack direction="row" spacing={4}>
                <Box>
                  <Typography variant="caption" color="error.main">
                    لندمارک‌های لترال (قرمز): {Object.keys(lateralLandmarks || {}).length}
                  </Typography>
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    {Object.keys(lateralLandmarks || {}).join(', ') || 'هیچ'}
                  </Typography>
                </Box>
                <Box>
                  <Typography variant="caption" color="primary.main">
                    لندمارک‌های پروفایل (آبی): {Object.keys(profileLandmarks || {}).length}
                  </Typography>
                  <Typography variant="caption" display="block" sx={{ mt: 0.5 }}>
                    {Object.keys(profileLandmarks || {}).join(', ') || 'هیچ'}
                  </Typography>
                </Box>
              </Stack>
            </Box>
          )}

          {(!selectedProfileImage && !selectedLateralImage) ? (
            <Alert severity="info">
              لطفا حداقل یک تصویر پروفایل یا لترال سفالومتری انتخاب کنید.
            </Alert>
          ) : (
            <Box
              sx={{
                width: '100%',
                height: 600,
                border: '2px solid',
                borderColor: 'divider',
                borderRadius: 1,
                overflow: 'hidden',
                position: 'relative',
                cursor: isDragging ? 'grabbing' : 'grab',
                backgroundColor: '#fafafa',
              }}
            >
              <canvas
                ref={canvasRef}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                style={{
                  width: '100%',
                  height: '100%',
                  display: 'block',
                  imageRendering: '-webkit-optimize-contrast',
                }}
              />

              {/* Position indicators */}
              {selectedProfileImage && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 10,
                    left: 10,
                    bgcolor: 'rgba(0, 123, 255, 0.8)',
                    color: 'white',
                    px: 1,
                    py: 0.5,
                    borderRadius: 1,
                    fontSize: '0.75rem',
                  }}
                >
                  پروفایل: x={Math.round(profilePosition.x)}, y={Math.round(profilePosition.y)}
                </Box>
              )}

              {selectedLateralImage && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 40,
                    left: 10,
                    bgcolor: 'rgba(255, 193, 7, 0.8)',
                    color: 'black',
                    px: 1,
                    py: 0.5,
                    borderRadius: 1,
                    fontSize: '0.75rem',
                  }}
                >
                  لترال: x={Math.round(lateralPosition.x)}, y={Math.round(lateralPosition.y)}
                </Box>
              )}
            </Box>
          )}
        </CardContent>
      </Card>

      {/* Instructions */}
      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            راهنمای استفاده
          </Typography>
          <Stack spacing={1}>
            <Typography variant="body2">
              • <strong>انتخاب تصاویر:</strong> ابتدا تصویر پروفایل و تصویر لترال سفالومتری بیمار را انتخاب کنید.
            </Typography>
            <Typography variant="body2">
              • <strong>تنظیم شفافیت:</strong> با استفاده از اسلایدرهای شفافیت، میزان نمایش هر تصویر را تنظیم کنید.
            </Typography>
            <Typography variant="body2">
              • <strong>جابجایی تصاویر:</strong> روی تصویر کلیک کرده و آن را بکشید تا موقعیت مناسب را پیدا کنید.
            </Typography>
            <Typography variant="body2">
              • <strong>تغییر اندازه:</strong> با استفاده از اسلایدرهای اندازه، تصاویر را بزرگ یا کوچک کنید.
            </Typography>
            <Typography variant="body2">
              • <strong>تراز کردن:</strong> سعی کنید نقاط آناتومیکی مانند بینی، چشم‌ها و فک‌ها روی هم بیفتند.
            </Typography>
            <Typography variant="body2">
              • <strong>ذخیره:</strong> پس از تنظیم مناسب، تصویر سوپرایمپوز شده را ذخیره کنید.
            </Typography>
          </Stack>
        </CardContent>
      </Card>
    </Stack>
  );
}