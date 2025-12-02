import axios from 'axios';
import { toast } from 'sonner';
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
import IconButton from '@mui/material/IconButton';
import CardContent from '@mui/material/CardContent';
import FormControl from '@mui/material/FormControl';

import { endpoints } from 'src/utils/axios';

import { Iconify } from 'src/components/iconify';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

export function SuperimposeView({ patient }) {
  const { user } = useAuthContext();
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
  const [referenceLandmarks, setReferenceLandmarks] = useState(['N', 'S', 'Sn', 'Pog']); // Default reference landmarks (using available landmarks)
  const [profileLandmarks, setProfileLandmarks] = useState({});
  const [lateralLandmarks, setLateralLandmarks] = useState({});
  const [alignmentMode, setAlignmentMode] = useState('auto'); // 'auto' or 'manual'
  const [showLandmarks, setShowLandmarks] = useState(true); // Toggle landmark visibility

  // Image loading state
  const [imagesLoaded, setImagesLoaded] = useState({ profile: false, lateral: false });
  const [loadedImages, setLoadedImages] = useState({ profile: null, lateral: null });

  // Available landmarks for selection (based on actual cephalometric model)
  const availableLandmarks = [
    { id: 'N', name: 'Nasion (N)', description: 'نقطه نازیون' },
    { id: 'S', name: 'Sella (S)', description: 'نقطه سلای تورسیکا' },
    { id: 'Po', name: 'Porion (Po)', description: 'نقطه پوریون' },
    { id: 'Sn', name: 'Subnasale (Sn)', description: 'نقطه زیر بینی' },
    { id: 'Pn', name: 'Pronasale (Pn)', description: 'نقطه نوک بینی' },
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
  const detectCephalometricLandmarks = async (imagePath) => {
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
  };

  // Detect facial landmarks for profile images using ML models
  const detectProfileFacialLandmarks = async (imagePath) => {
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
  };

  // Get landmarks for both images - READ FROM DATABASE (cephalometricLandmarks)
  const getImageLandmarks = useCallback(async () => {
    const landmarks = { lateral: {}, profile: {} };

    // Get cephalometric landmarks for lateral image - ALWAYS USE DATABASE
    if (selectedLateralImage && patient?.cephalometricLandmarks) {
      console.log('📋 Using cephalometric landmarks from database');
      landmarks.lateral = { ...patient.cephalometricLandmarks };
      console.log('✅ Loaded', Object.keys(landmarks.lateral).length, 'lateral landmarks from database');
    } else if (selectedLateralImage) {
      console.warn('⚠️ No cephalometric landmarks found in database');
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
        
        // Enhanced landmark normalization with numbered support and better error handling
        const normalizedLandmarks = {};
        Object.keys(facialLandmarks).forEach(landmarkName => {
          const landmark = facialLandmarks[landmarkName];
          
          // Validate landmark has x and y coordinates
          if (!landmark || typeof landmark.x !== 'number' || typeof landmark.y !== 'number') {
            console.warn(`⚠️ Invalid landmark detected: ${landmarkName}`, landmark);
            return;
          }
          
          const normalizedName = landmarkName.toLowerCase().replace(/[_\s-]/g, '_');

          // Accept both 'landmark_30' and plain numeric keys '30'
          let numberMatch = landmarkName.match(/landmark_(\d+)/);
          if (!numberMatch) {
            // match plain numeric keys
            numberMatch = landmarkName.match(/^(\d+)$/);
          }

          if (numberMatch) {
            const landmarkIndex = parseInt(numberMatch[1], 10);
            console.log(`🔢 Processing numbered landmark: ${landmarkName} (index ${landmarkIndex}) at (${landmark.x.toFixed(1)}, ${landmark.y.toFixed(1)})`);

            // Use dlib mapping to convert index to anatomical name
            switch (landmarkIndex) {
              case 30: // Nose tip
                normalizedLandmarks.landmark_30 = landmark;
                normalizedLandmarks.nose_tip = landmark;
                console.log('  ✅ Mapped to: nose_tip');
                break;
              case 33: // Subnasale
                normalizedLandmarks.landmark_33 = landmark;
                normalizedLandmarks.Subnasale = landmark;
                normalizedLandmarks.subnasale = landmark; // Add lowercase variant
                console.log('  ✅ Mapped to: Subnasale');
                break;
              case 8:  // Pogonion/chin
                normalizedLandmarks.landmark_8 = landmark;
                normalizedLandmarks.pogonion = landmark;
                normalizedLandmarks.menton = landmark;
                normalizedLandmarks.chin = landmark;
                console.log('  ✅ Mapped to: pogonion, menton, chin');
                break;
              case 27: // Glabella/Nasion approximation
                normalizedLandmarks.landmark_27 = landmark;
                normalizedLandmarks.glabella = landmark;
                normalizedLandmarks.nasion = landmark; // Close approximation
                console.log('  ✅ Mapped to: glabella, nasion');
                break;
              case 51: // Upper lip
                normalizedLandmarks.landmark_51 = landmark;
                normalizedLandmarks.upper_lip = landmark;
                console.log('  ✅ Mapped to: upper_lip');
                break;
              case 57: // Lower lip
                normalizedLandmarks.landmark_57 = landmark;
                normalizedLandmarks.lower_lip = landmark;
                console.log('  ✅ Mapped to: lower_lip');
                break;
              case 36: // Left eye outer corner
              case 45: // Right eye outer corner
                normalizedLandmarks[`landmark_${landmarkIndex}`] = landmark;
                normalizedLandmarks.eye_corner = landmark;
                break;
              default:
                // Keep the numbered landmark for potential future mapping
                normalizedLandmarks[`landmark_${landmarkIndex}`] = landmark;
                console.log(`  ℹ️ Kept as landmark_${landmarkIndex}`);
                break;
            }
          } else {
            // Handle named landmarks (fallback for other naming schemes)
            if (normalizedName.includes('nose') && normalizedName.includes('tip')) {
              normalizedLandmarks.nose_tip = landmark;
              normalizedLandmarks[landmarkName] = landmark; // Keep original
            } else if (normalizedName.includes('subnasal') || normalizedName.includes('sn')) {
              normalizedLandmarks.Subnasale = landmark;
              normalizedLandmarks.subnasale = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('pogonion') || (normalizedName.includes('chin') && !normalizedName.includes('below'))) {
              normalizedLandmarks.pogonion = landmark;
              normalizedLandmarks.chin = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('glabella') || (normalizedName.includes('forehead') && normalizedName.includes('center'))) {
              normalizedLandmarks.glabella = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('nasion')) {
              normalizedLandmarks.nasion = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('menton') || normalizedName.includes('chin_bottom')) {
              normalizedLandmarks.menton = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('gnathion')) {
              normalizedLandmarks.gnathion = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('upper') && normalizedName.includes('lip')) {
              normalizedLandmarks.upper_lip = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else if (normalizedName.includes('lower') && normalizedName.includes('lip')) {
              normalizedLandmarks.lower_lip = landmark;
              normalizedLandmarks[landmarkName] = landmark;
            } else {
              // Keep original name for other landmarks
              normalizedLandmarks[normalizedName] = landmark;
              normalizedLandmarks[landmarkName] = landmark; // Also keep exact original
            }
          }
        });
        
        console.log('✅ Normalized profile landmarks:', Object.keys(normalizedLandmarks).filter(k => !k.startsWith('landmark_')));
        
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
  }, [selectedLateralImage, selectedProfileImage, patient?.cephalometricLandmarks, loadedImages.lateral, loadedImages.profile]);

  // Estimate profile landmarks from lateral cephalometric landmarks (improved fallback)
  const estimateProfileLandmarksFromLateral = (lateralLandmarks) => {
    const profileLms = {};

    console.log('📐 Estimating profile landmarks from lateral (fallback mode)');
    console.log('  Available lateral landmarks:', Object.keys(lateralLandmarks));

    // Anatomical relationships between lateral and profile views
    // Enhanced with better approximations
    Object.keys(lateralLandmarks).forEach(landmarkId => {
      const lateralLm = lateralLandmarks[landmarkId];

      // Validate landmark
      if (!lateralLm || typeof lateralLm.x !== 'number' || typeof lateralLm.y !== 'number') {
        console.warn(`  ⚠️ Invalid lateral landmark: ${landmarkId}`);
        return;
      }

      // Base estimation using anatomical proportions
      let estimatedX = lateralLm.x;
      let estimatedY = lateralLm.y;
      let mappedName = landmarkId; // Keep original name by default

      // Adjust based on specific landmark characteristics
      // Use case-insensitive matching
      const lmIdUpper = landmarkId.toUpperCase();
      
      if (lmIdUpper === 'N' || lmIdUpper === 'NASION') {
        // Nasion - similar position in both views
        estimatedX = lateralLm.x + 5;
        estimatedY = lateralLm.y - 2;
        mappedName = 'nasion';
      } else if (lmIdUpper === 'S' || lmIdUpper === 'SELLA') {
        // Sella - cranial base point
        estimatedX = lateralLm.x + 3;
        estimatedY = lateralLm.y - 1;
      } else if (lmIdUpper === 'PO' || lmIdUpper === 'PORION') {
        // Porion - ear point
        estimatedX = lateralLm.x - 8;
        estimatedY = lateralLm.y + 3;
      } else if (lmIdUpper === 'BA' || lmIdUpper === 'BASION') {
        // Basion
        estimatedX = lateralLm.x + 2;
        estimatedY = lateralLm.y + 1;
      } else if (lmIdUpper === 'POG' || lmIdUpper === 'POGONION') {
        // Pogonion - chin point
        estimatedX = lateralLm.x + 8;
        estimatedY = lateralLm.y + 4;
        mappedName = 'pogonion';
      } else if (lmIdUpper === 'GN' || lmIdUpper === 'GNATHION') {
        // Gnathion
        estimatedX = lateralLm.x + 7;
        estimatedY = lateralLm.y + 5;
        mappedName = 'gnathion';
      } else if (lmIdUpper === 'ME' || lmIdUpper === 'MENTON') {
        // Menton - lowest chin point
        estimatedX = lateralLm.x + 6;
        estimatedY = lateralLm.y + 6;
        mappedName = 'menton';
      } else if (lmIdUpper === 'PN' || lmIdUpper === 'PRONASALE') {
        // Pronasale - nose tip
        estimatedX = lateralLm.x + 18;
        estimatedY = lateralLm.y - 2;
        mappedName = 'nose_tip';
      } else if (lmIdUpper === 'SN' || lmIdUpper === 'SUBNASALE') {
        // Subnasale
        estimatedX = lateralLm.x + 16;
        estimatedY = lateralLm.y - 1;
        mappedName = 'Subnasale';
      } else {
        // For unknown landmarks, use conservative estimation
        estimatedX = lateralLm.x + 5;
        estimatedY = lateralLm.y;
        console.log(`  ℹ️ Unknown landmark ${landmarkId}, using conservative estimation`);
      }

      // Store with both original and mapped names
      profileLms[landmarkId] = {
        x: estimatedX,
        y: estimatedY,
      };
      
      if (mappedName !== landmarkId) {
        profileLms[mappedName] = {
          x: estimatedX,
          y: estimatedY,
        };
        console.log(`  📍 Estimated ${landmarkId} -> ${mappedName} at (${estimatedX.toFixed(1)}, ${estimatedY.toFixed(1)})`);
      }
    });

    console.log('✅ Estimated', Object.keys(profileLms).length, 'profile landmarks from lateral');
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
        // Set default landmarks based on availability, preferring common ones that exist in the model
        const preferredLandmarks = ['N', 'S', 'Sn', 'Pn', 'Po', 'Or', 'A', 'B', 'Pog', 'Me'];
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
  // این scale فقط برای lateral استفاده می‌شود، profile scale از landmark alignment می‌آید
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

        // محاسبه scale فقط برای تصویر lateral (reference image)
        // تصویر profile از landmark-based scaling استفاده می‌کند
        
        // Target size: 70% of canvas dimensions
        const targetWidth = canvasWidth * 0.7;
        const targetHeight = canvasHeight * 0.7;

        // Calculate scale for lateral image to fit in canvas
        const scaleX = targetWidth / lateralImg.width;
        const scaleY = targetHeight / lateralImg.height;
        const lateralOptimalScale = Math.min(scaleX, scaleY, 1.0); // Don't upscale (max 100%)

        // Set lateral scale only - profile will be scaled relative to lateral via landmark alignment
        const finalLateralScale = Math.max(lateralOptimalScale, 0.1); // Minimum scale of 10%
        
        setLateralScale(finalLateralScale);
        // Don't set profile scale here - it will be calculated by landmark alignment

        console.log('🎯 Auto-adjusted lateral scale only:', {
          lateralScale: finalLateralScale.toFixed(3),
          targetSize: { width: targetWidth, height: targetHeight },
          lateralImageSize: { width: lateralImg.width, height: lateralImg.height },
          note: 'Profile scale will be calculated via landmark alignment'
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

          // Draw label (remove 'landmark_' prefix for cleaner display)
          ctx.fillStyle = 'blue';
          ctx.font = '12px Arial';
          const displayLabel = landmarkId.startsWith('landmark_') ? landmarkId.replace('landmark_', '') : landmarkId;
          ctx.fillText(displayLabel, x + 6, y + 6);
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

  // Enhanced landmark-based alignment functions with retry limit
  const alignmentRetryCountRef = useRef(0);
  const alignmentInProgressRef = useRef(false);
  const maxAlignmentRetries = 10; // Maximum 10 retries (3 seconds total)

  const calculateLandmarkBasedAlignment = useCallback(async () => {
    if (!useLandmarkAlignment || !selectedProfileImage || !selectedLateralImage) {
      alignmentRetryCountRef.current = 0; // Reset counter when conditions not met
      alignmentInProgressRef.current = false;
      return;
    }

    // Prevent concurrent alignment attempts
    if (alignmentInProgressRef.current) {
      console.log('⏸️ Alignment already in progress, skipping...');
      return;
    }

    try {
      alignmentInProgressRef.current = true;
      console.log('🎯 Starting enhanced landmark-based alignment...');

      // Wait for images to be available - necessary for pixel-based alignment
      if (!loadedImages.lateral || !loadedImages.profile || !imagesLoaded.lateral || !imagesLoaded.profile) {
        if (alignmentRetryCountRef.current < maxAlignmentRetries) {
          console.warn(`⏳ Images not ready for alignment, retry ${alignmentRetryCountRef.current + 1}/${maxAlignmentRetries}`);
          alignmentRetryCountRef.current += 1;
          alignmentInProgressRef.current = false; // Allow retry
          setTimeout(() => {
            calculateLandmarkBasedAlignment();
          }, 300);
          return;
        } 
          console.error('❌ Images failed to load after maximum retries');
          alignmentRetryCountRef.current = 0; // Reset for next attempt
          alignmentInProgressRef.current = false;
          return;
        
      }

      // Reset retry counter once images are loaded
      alignmentRetryCountRef.current = 0;

      // Get landmarks for both images using ML detection
      const imageLandmarks = await getImageLandmarks();

      const lateralLms = imageLandmarks.lateral;
      const profileLms = imageLandmarks.profile;

      console.log('📊 Landmarks retrieved:', {
        lateral: Object.keys(lateralLms).length,
        profile: Object.keys(profileLms).length
      });

      if (Object.keys(lateralLms).length < 2) {
        console.warn('❌ Not enough lateral landmarks for alignment - need at least 2 landmarks');
        alignmentInProgressRef.current = false;
        return;
      }

      if (Object.keys(profileLms).length < 2) {
        console.warn('❌ Not enough profile landmarks for alignment - need at least 2 landmarks');
        alignmentInProgressRef.current = false;
        return;
      }

      // Update state with landmarks for visualization
      setLateralLandmarks(lateralLms);
      setProfileLandmarks(profileLms);

      // Calculate optimal transform using the enhanced algorithm
      const transform = calculateOptimalTransform(profileLms, lateralLms);

      // Check if we have valid mapped pairs
      if (transform.alignmentQuality.mappedCount < 2) {
        console.warn('❌ Not enough mapped landmark pairs for alignment');
        alignmentInProgressRef.current = false;
        return;
      }

      // Validate transform quality using normalized error
      const normalizedError = transform.alignmentQuality.normalizedError || 0;
      if (normalizedError > 5) { // More than 5% normalized error
        console.warn('⚠️ High alignment error detected:',
          transform.alignmentQuality.averageError.toFixed(2), 'pixels',
          `(${normalizedError.toFixed(2)}% normalized) - alignment may not be accurate`);
      } else if (normalizedError > 2) {
        console.log('ℹ️ Moderate alignment quality:', transform.alignmentQuality.averageError.toFixed(2), 'pixels', `(${normalizedError.toFixed(2)}% normalized)`);
      } else {
        console.log('✅ Good alignment quality:', transform.alignmentQuality.averageError.toFixed(2), 'pixels', `(${normalizedError.toFixed(2)}% normalized - excellent!)`);
      }

      // Apply the calculated transform to profile image
      // Profile scale باید relative به lateral scale باشد
      const profileScaleRelativeToLateral = transform.scale * lateralScale;
      setProfileScale(profileScaleRelativeToLateral);
      
      console.log('📐 Scale application:', {
        lateralScale: lateralScale.toFixed(3),
        landmarkScaleRatio: transform.scale.toFixed(3),
        finalProfileScale: profileScaleRelativeToLateral.toFixed(3),
        explanation: `profile = lateral(${lateralScale.toFixed(3)}) × ratio(${transform.scale.toFixed(3)}) = ${profileScaleRelativeToLateral.toFixed(3)}`
      });

      // محاسبه position برای تراز دقیق لندمارک‌ها
      // استراتژی: تصویر lateral مرجع است، profile باید طوری قرار بگیرد که Sn روی Subnasale بیفتد
      try {
        const canvas = canvasRef.current;
        const rect = canvas ? canvas.getBoundingClientRect() : { width: 800, height: 600 };
        const lateralImg = loadedImages.lateral;
        const profileImg = loadedImages.profile;

        // مرکز canvas
        const canvasCenterX = rect.width / 2;
        const canvasCenterY = rect.height / 2;

        // موقعیت lateral image در canvas (reference)
        const lateralCanvasX = canvasCenterX - (lateralImg.width * lateralScale) / 2 + lateralPosition.x;
        const lateralCanvasY = canvasCenterY - (lateralImg.height * lateralScale) / 2 + lateralPosition.y;

        // استفاده از اولین جفت لندمارک mapped برای alignment
        let profilePosX; let profilePosY;

        if (transform.mappedPairs.length > 0) {
          const firstPair = transform.mappedPairs[0];
          
          // پیدا کردن لندمارک‌ها در landmark sets
          const lateralLm = lateralLms[firstPair.lateralName];
          let profileLm = profileLms[firstPair.profileName];
          
          // If not found, try with landmark_ prefix removed
          if (!profileLm && firstPair.profileName.startsWith('landmark_')) {
            const numericName = firstPair.profileName.replace('landmark_', '');
            profileLm = profileLms[numericName] || profileLms[firstPair.profileName];
          }

          if (lateralLm && profileLm) {
            // موقعیت لندمارک lateral در canvas
            const lateralLmCanvasX = lateralCanvasX + lateralLm.x * lateralScale;
            const lateralLmCanvasY = lateralCanvasY + lateralLm.y * lateralScale;

            // موقعیت لندمارک profile در image (scaled)
            const profileLmOffsetX = profileLm.x * profileScaleRelativeToLateral;
            const profileLmOffsetY = profileLm.y * profileScaleRelativeToLateral;

            // Profile position طوری که لندمارک profile روی لندمارک lateral بیفتد
            profilePosX = lateralLmCanvasX - profileLmOffsetX - (canvasCenterX - (profileImg.width * profileScaleRelativeToLateral) / 2);
            profilePosY = lateralLmCanvasY - profileLmOffsetY - (canvasCenterY - (profileImg.height * profileScaleRelativeToLateral) / 2);

            console.log(`✅ Aligned ${firstPair.profileName} → ${firstPair.lateralName}:`, {
              lateralLmCanvas: { x: lateralLmCanvasX.toFixed(1), y: lateralLmCanvasY.toFixed(1) },
              profileLmOffset: { x: profileLmOffsetX.toFixed(1), y: profileLmOffsetY.toFixed(1) },
              profilePosition: { x: profilePosX.toFixed(1), y: profilePosY.toFixed(1) }
            });
          } else {
            // Fallback: use scaled offset from transform
            profilePosX = transform.offsetX * lateralScale;
            profilePosY = transform.offsetY * lateralScale;
            console.warn('⚠️ Landmarks not found in image data, using calculated offset');
          }
        } else {
          // Ultimate fallback
          profilePosX = 0;
          profilePosY = 0;
          console.warn('⚠️ No mapped pairs, using zero offset');
        }

        setProfilePosition({ x: profilePosX, y: profilePosY });

        console.log('✅ Applied landmark-based alignment:', {
          lateralScale: lateralScale.toFixed(3),
          profileScale: profileScaleRelativeToLateral.toFixed(3),
          scaleRatio: transform.scale.toFixed(3),
          profilePosition: { x: profilePosX.toFixed(1), y: profilePosY.toFixed(1) },
          alignmentQuality: {
            avgError: `${transform.alignmentQuality.averageError.toFixed(2)  }px`,
            normalizedError: `${transform.alignmentQuality.normalizedError.toFixed(2)  }%`
          }
        });
      } catch (err) {
        console.error('⚠️ Alignment to canvas coords failed:', err);
        alignmentInProgressRef.current = false;
        
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
    } finally {
      alignmentInProgressRef.current = false;
    }
  }, [useLandmarkAlignment, selectedProfileImage, selectedLateralImage, patient, loadedImages, calculateOptimalTransform, getImageLandmarks, lateralScale, lateralPosition, imagesLoaded.lateral, imagesLoaded.profile]);

  // Enhanced landmark mapping with dlib index support (only using landmarks available in the model)
  // REMOVED: A, B, OR, Li, Ls as per user request
  const getLandmarkMapping = useCallback(() => ({
      // دlib indices -> Profile landmark names -> Lateral cephalometric landmarks
      // فقط لندمارک‌هایی که در مدل سفالومتری موجود هستند
      30: { profile: 'nose_tip', lateral: 'Pn' },     // landmark_30 = nose tip -> Pronasale ✓
      33: { profile: 'Subnasale', lateral: 'Sn' },    // landmark_33 = subnasale ✓
      8: { profile: 'pogonion', lateral: 'Pog' },     // landmark_8 = pogonion/chin ✓
      27: { profile: 'glabella', lateral: 'N' },      // landmark_27 = glabella/nasion ✓
      
      // Direct anatomical name mappings with alternatives (فقط موجود در مدل)
      'nose_tip': 'Pn',          // Nose tip = Pronasale ✓
      'landmark_30': 'Pn',       // Direct dlib mapping ✓
      'Subnasale': 'Sn',         // Subnasale ✓
      'subnasale': 'Sn',         // Lowercase variant ✓
      'landmark_33': 'Sn',       // Direct dlib mapping ✓
      'pogonion': 'Pog',         // Pogonion ✓
      'landmark_8': 'Pog',       // Direct dlib mapping ✓
      'glabella': 'N',           // Glabella = Nasion approximation ✓
      'landmark_27': 'N',        // Direct dlib mapping ✓
      'nasion': 'N',             // Nasion ✓
      'menton': 'Me',            // Menton ✓
      'gnathion': 'Gn',          // Gnathion ✓
      'chin': 'Pog',             // Chin = Pogonion ✓
      
      // Additional common cephalometric landmarks (فقط موجود در مدل)
      'S': 'S',                  // Sella ✓
      'Po': 'Po',                // Porion ✓
      'Ar': 'Ar',                // Articulare ✓
      'Co': 'Co',                // Condylion ✓
      'Go': 'Go',                // Gonion ✓
      'ANS': 'ANS',              // Anterior Nasal Spine ✓
      'PNS': 'PNS',              // Posterior Nasal Spine ✓
    }), []);

  // Calculate distance between two landmarks
  const calculateLandmarkDistance = useCallback((landmark1, landmark2) => {
    if (!landmark1 || !landmark2) return 0;
    const dx = landmark1.x - landmark2.x;
    const dy = landmark1.y - landmark2.y;
    return Math.sqrt(dx * dx + dy * dy);
  }, []);

  // Calculate scale ratio based on landmark distances (enhanced with numbered landmark support)
  const calculateScaleRatio = (profileLandmarks, lateralLandmarks) => {
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

    const getLateralLandmark = (lateralName) => {
      // Try direct match first
      if (lateralLandmarks[lateralName]) return lateralLandmarks[lateralName];
      
      // Try case-insensitive match
      const lowerName = lateralName.toLowerCase();
      for (const key of Object.keys(lateralLandmarks)) {
        if (key.toLowerCase() === lowerName) {
          return lateralLandmarks[key];
        }
      }
      
      return null;
    };

    // Find mapped landmark pairs for scale calculation
    const mappedPairs = [];
    Object.keys(mapping).forEach(key => {
      const mappingInfo = mapping[key];
      
      if (typeof mappingInfo === 'string') {
        // Old format: direct name mapping
        const profileName = key;
        const lateralName = mappingInfo;
        const profileLm = profileLandmarks[profileName];
        const lateralLm = getLateralLandmark(lateralName);
        
        if (profileLm && lateralLm) {
          mappedPairs.push({
            profile: profileLm,
            lateral: lateralLm,
            profileName,
            lateralName
          });
        }
      } else if (typeof mappingInfo === 'object') {
        // New format: { profile: name, lateral: name }
        const profileName = mappingInfo.profile;
        const lateralName = mappingInfo.lateral;
        
        const profileLm = getProfileLandmark(key) || profileLandmarks[profileName];
        const lateralLm = getLateralLandmark(lateralName);
        
        if (profileLm && lateralLm) {
          mappedPairs.push({
            profile: profileLm,
            lateral: lateralLm,
            profileName,
            lateralName
          });
        }
      }
    });

    console.log('🔍 Mapped pairs for scale calculation:', mappedPairs.length);

    // Calculate distances between all pairs of mapped landmarks
    for (let i = 0; i < mappedPairs.length; i++) {
      for (let j = i + 1; j < mappedPairs.length; j++) {
        const pair1 = mappedPairs[i];
        const pair2 = mappedPairs[j];
        
        const profileDistance = calculateLandmarkDistance(pair1.profile, pair2.profile);
        const lateralDistance = calculateLandmarkDistance(pair1.lateral, pair2.lateral);
        
        if (profileDistance > 5 && lateralDistance > 5) { // Minimum distance threshold (lowered)
          const ratio = lateralDistance / profileDistance;
          // Filter out extreme ratios - expanded threshold to handle different image sizes
          if (ratio >= 0.1 && ratio <= 10.0) {
            scaleRatios.push(ratio);
            console.log(`  Scale ratio ${pair1.profileName}-${pair2.profileName}: ${ratio.toFixed(3)} (dist: profile=${profileDistance.toFixed(1)}, lateral=${lateralDistance.toFixed(1)})`);
          } else {
            console.log(`  ⚠️ Rejected ratio ${ratio.toFixed(3)} for ${pair1.profileName}-${pair2.profileName} (out of range 0.1-10.0)`);
          }
        }
      }
    }

    // Return median scale ratio for robustness
    if (scaleRatios.length > 0) {
      scaleRatios.sort((a, b) => a - b);
      const medianIndex = Math.floor(scaleRatios.length / 2);
      const medianRatio = scaleRatios.length % 2 === 0
        ? (scaleRatios[medianIndex - 1] + scaleRatios[medianIndex]) / 2
        : scaleRatios[medianIndex];
      
      console.log('✅ Calculated median scale ratio:', medianRatio.toFixed(3), 'from', scaleRatios.length, 'ratios');
      return medianRatio;
    }

    console.warn('⚠️ No valid scale ratios found, using default 1.0');
    return 1; // Default scale if no valid ratios found
  };

  // الگوریتم تراز بر اساس لندمارک‌های مشترک (استاندارد superimposition)
  const calculateOptimalTransform = useCallback((profileLandmarks, lateralLandmarks) => {
    console.log('🎯 Starting standard superimposition algorithm...');
    
    // Helper functions to find landmarks with case-insensitive search
    const getProfileLandmark = (names) => {
      for (const name of names) {
        if (profileLandmarks[name]) return profileLandmarks[name];
        // Try case-insensitive
        const found = Object.keys(profileLandmarks).find(k => k.toLowerCase() === name.toLowerCase());
        if (found) return profileLandmarks[found];
      }
      return null;
    };

    const getLateralLandmark = (names) => {
      for (const name of names) {
        if (lateralLandmarks[name]) return lateralLandmarks[name];
        // Try case-insensitive
        const found = Object.keys(lateralLandmarks).find(k => k.toLowerCase() === name.toLowerCase());
        if (found) return lateralLandmarks[found];
      }
      return null;
    };

    // مرحله 1: محاسبه scale بر اساس لندمارک‌های انتخابی کاربر
    let scaleRatio = 1.0;
    let referenceProfileDistance = 0;
    let referenceLateralDistance = 0;
    let usedLandmarks = [];

    // پیدا کردن اولین دو لندمارک مشترک که هم در profile و هم در lateral موجود هستند
    const mapping = getLandmarkMapping();
    const availablePairs = [];

    // پیدا کردن جفت‌های لندمارک موجود
    Object.keys(mapping).forEach(key => {
      const mappingInfo = mapping[key];
      let profileName; let lateralName;
      
      if (typeof mappingInfo === 'string') {
        profileName = key;
        lateralName = mappingInfo;
      } else if (typeof mappingInfo === 'object') {
        profileName = mappingInfo.profile;
        lateralName = mappingInfo.lateral;
      } else {
        return;
      }
      
      let profileLm = null;
      if (typeof key === 'number') {
        profileLm = getProfileLandmark([`landmark_${key}`, profileName]);
      } else {
        profileLm = getProfileLandmark([key, profileName]);
      }
      
      const lateralLm = getLateralLandmark([lateralName]);
      
      if (profileLm && lateralLm) {
        availablePairs.push({
          profile: profileLm,
          lateral: lateralLm,
          profileName,
          lateralName
        });
      }
    });

    // استفاده از اولین دو جفت لندمارک موجود برای محاسبه scale
    if (availablePairs.length >= 2) {
      const pair1 = availablePairs[0];
      const pair2 = availablePairs[1];
      
      referenceProfileDistance = calculateLandmarkDistance(pair1.profile, pair2.profile);
      referenceLateralDistance = calculateLandmarkDistance(pair1.lateral, pair2.lateral);
      
      if (referenceProfileDistance > 0) {
        scaleRatio = referenceLateralDistance / referenceProfileDistance;
        usedLandmarks = [pair1.lateralName, pair2.lateralName];
        console.log(`✅ Scale calculated from ${pair1.lateralName}-${pair2.lateralName} distance:`, {
          profileDistance: referenceProfileDistance.toFixed(2),
          lateralDistance: referenceLateralDistance.toFixed(2),
          scaleRatio: scaleRatio.toFixed(3)
        });
      }
    } else if (availablePairs.length === 1) {
      console.warn('⚠️ Only 1 landmark pair found, using default scale 1.0');
    } else {
      console.warn('⚠️ No landmark pairs found, using default scale 1.0');
    }

    // مرحله 2: استفاده از همان جفت‌های پیدا شده
    const mappedPairs = availablePairs;

    if (mappedPairs.length < 2) {
      console.warn('❌ Insufficient mapped landmarks for alignment (need at least 2)');
      return {
        scale: 1,
        offsetX: 0,
        offsetY: 0,
        mappedPairs: [],
        alignmentQuality: { averageError: 999, normalizedError: 100, maxError: 999, mappedCount: 0 }
      };
    }

    console.log('✅ Found', mappedPairs.length, 'unique mapped landmark pairs');

    // مرحله 3: محاسبه offset برای تراز اولین لندمارک mapped
    let offsetX = 0;
    let offsetY = 0;
    let alignmentLandmark = 'centroid';
    
    // استفاده از اولین جفت لندمارک موجود برای alignment
    if (mappedPairs.length > 0) {
      const firstPair = mappedPairs[0];
      offsetX = firstPair.lateral.x - (firstPair.profile.x * scaleRatio);
      offsetY = firstPair.lateral.y - (firstPair.profile.y * scaleRatio);
      alignmentLandmark = `${firstPair.profileName}-${firstPair.lateralName}`;
      console.log(`✅ Using ${firstPair.lateralName} (${firstPair.profileName}) for offset calculation`);
    } else {
      console.warn('⚠️ No mapped pairs found, using centroid');
    }

    console.log('📍 Calculated transform:', {
      scale: scaleRatio.toFixed(3),
      offset: { x: offsetX.toFixed(1), y: offsetY.toFixed(1) }
    });

    // مرحله 4: ارزیابی کیفیت تراز
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
    
    // محاسبه normalized error به عنوان درصد از diagonal تصویر
    const lateralDiagonal = Math.sqrt(
      Math.pow(Math.max(...mappedPairs.map(p => p.lateral.x)) - Math.min(...mappedPairs.map(p => p.lateral.x)), 2) +
      Math.pow(Math.max(...mappedPairs.map(p => p.lateral.y)) - Math.min(...mappedPairs.map(p => p.lateral.y)), 2)
    );
    const normalizedError = lateralDiagonal > 0 ? (avgError / lateralDiagonal) * 100 : 0;

    console.log('🎯 Alignment quality:', {
      averageError: `${avgError.toFixed(2)  } pixels`,
      normalizedError: `${normalizedError.toFixed(2)  }% of image diagonal`,
      maxError: `${maxError.toFixed(2)  } pixels`,
      mappedLandmarks: mappedPairs.length,
      scaleMethod: usedLandmarks.length > 0 ? `${usedLandmarks[0]}-${usedLandmarks[1]} distance` : 'fallback',
      alignmentLandmark: alignmentLandmark
    });

    // Return the transform
    const transform = {
      scale: Math.min(Math.max(scaleRatio, 0.1), 10), // Clamp between 0.1 and 10 (wider range for different image sizes)
      offsetX,
      offsetY,
      alignmentQuality: {
        averageError: avgError,
        maxError: maxError,
        normalizedError: normalizedError, // نسبت خطا به اندازه تصویر (%)
        mappedCount: mappedPairs.length
      },
      mappedPairs: mappedPairs
    };

    return transform;
  }, [getLandmarkMapping, calculateLandmarkDistance]);

  // Auto-align when images or landmarks change (with proper dependency management)
  const hasTriggeredAlignmentRef = useRef(false);
  
  useEffect(() => {
    // Only trigger if we haven't already triggered for these specific images
    const imageKey = `${selectedProfileImage?.id || 'none'}-${selectedLateralImage?.id || 'none'}`;
    const lastTriggeredKey = hasTriggeredAlignmentRef.current;
    
    if (useLandmarkAlignment && alignmentMode === 'auto' && selectedProfileImage && selectedLateralImage) {
      // Only trigger if images changed or first time
      if (lastTriggeredKey !== imageKey) {
        hasTriggeredAlignmentRef.current = imageKey;
        // Small delay to ensure images are loaded
        const timer = setTimeout(() => {
          calculateLandmarkBasedAlignment();
        }, 800);
        return () => clearTimeout(timer);
      }
    }
  }, [useLandmarkAlignment, alignmentMode, selectedProfileImage, selectedLateralImage, calculateLandmarkBasedAlignment]);

  // Function to detect landmarks for both images
  const landmarkDetectionInProgressRef = useRef(false);
  
  const detectLandmarksForBothImages = useCallback(async () => {
    if (!selectedLateralImage && !selectedProfileImage) {
      console.warn('No images selected for landmark detection');
      return;
    }

    // Prevent concurrent detection
    if (landmarkDetectionInProgressRef.current) {
      console.log('⏸️ Landmark detection already in progress, skipping...');
      return;
    }

    try {
      landmarkDetectionInProgressRef.current = true;
      console.log('🔍 Starting landmark detection for both images...');

      const imageLandmarks = await getImageLandmarks();

      // Update state with detected landmarks
      setLateralLandmarks(imageLandmarks.lateral);
      setProfileLandmarks(imageLandmarks.profile);

      console.log('✅ Landmarks detected and applied:', {
        lateral: Object.keys(imageLandmarks.lateral).length,
        profile: Object.keys(imageLandmarks.profile).length
      });

      // Note: Alignment will be triggered by the main useEffect, not here

    } catch (error) {
      console.error('❌ Error detecting landmarks:', error);
    } finally {
      landmarkDetectionInProgressRef.current = false;
    }
  }, [selectedLateralImage, selectedProfileImage, getImageLandmarks]);

  // Auto-detect landmarks when images are selected (only once per image)
  const lastDetectedImagesRef = useRef({ profile: null, lateral: null });
  
  useEffect(() => {
    const profileId = selectedProfileImage?.id;
    const lateralId = selectedLateralImage?.id;
    
    // Only detect if images changed
    if ((profileId && profileId !== lastDetectedImagesRef.current.profile) ||
        (lateralId && lateralId !== lastDetectedImagesRef.current.lateral)) {
      
      lastDetectedImagesRef.current = {
        profile: profileId,
        lateral: lateralId
      };
      
      // Wait for images to load before detecting landmarks
      const timer = setTimeout(() => {
        if (imagesLoaded.profile || imagesLoaded.lateral) {
          detectLandmarksForBothImages();
        }
      }, 1000); // Longer delay to ensure images are fully loaded
      
      return () => clearTimeout(timer);
    }
  }, [selectedLateralImage?.id, selectedProfileImage?.id, imagesLoaded.profile, imagesLoaded.lateral, detectLandmarksForBothImages]);

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

  // Save superimpose settings to database
  const saveSuperimposeSettings = useCallback(async () => {
    if (!patient?.id || !user?.accessToken) {
      console.warn('Missing patient ID or user token');
      return;
    }

    try {
      const settings = {
        selectedProfileImageId: selectedProfileImage?.id,
        selectedLateralImageId: selectedLateralImage?.id,
        profileOpacity,
        lateralOpacity,
        profilePosition,
        lateralPosition,
        profileScale,
        lateralScale,
        referenceLandmarks,
        alignmentMode,
        useLandmarkAlignment,
        timestamp: new Date().toISOString(),
      };

      await axios.put(
        `${endpoints.patients}/${patient.id}`,
        { superimposeSettings: JSON.stringify(settings) },
        {
          headers: {
            Authorization: `Bearer ${user.accessToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      toast.success('تنظیمات سوپرایمپوز ذخیره شد');
      console.log('✅ Superimpose settings saved');
    } catch (error) {
      console.error('❌ Failed to save superimpose settings:', error);
      toast.error('خطا در ذخیره تنظیمات');
    }
  }, [
    patient?.id,
    user?.accessToken,
    selectedProfileImage?.id,
    selectedLateralImage?.id,
    profileOpacity,
    lateralOpacity,
    profilePosition,
    lateralPosition,
    profileScale,
    lateralScale,
    referenceLandmarks,
    alignmentMode,
    useLandmarkAlignment,
  ]);

  // Load superimpose settings from database
  const loadSuperimposeSettings = useCallback(async () => {
    if (!patient?.superimposeSettings) {
      console.log('No saved superimpose settings found');
      return;
    }

    try {
      const settings = JSON.parse(patient.superimposeSettings);
      console.log('📥 Loading superimpose settings:', settings);

      // Restore image selections
      if (settings.selectedProfileImageId) {
        const profileImg = profileImages.find(img => img.id === settings.selectedProfileImageId);
        if (profileImg) setSelectedProfileImage(profileImg);
      }
      if (settings.selectedLateralImageId) {
        const lateralImg = lateralImages.find(img => img.id === settings.selectedLateralImageId);
        if (lateralImg) setSelectedLateralImage(lateralImg);
      }

      // Restore positions and scales
      if (settings.profileOpacity !== undefined) setProfileOpacity(settings.profileOpacity);
      if (settings.lateralOpacity !== undefined) setLateralOpacity(settings.lateralOpacity);
      if (settings.profilePosition) setProfilePosition(settings.profilePosition);
      if (settings.lateralPosition) setLateralPosition(settings.lateralPosition);
      if (settings.profileScale !== undefined) setProfileScale(settings.profileScale);
      if (settings.lateralScale !== undefined) setLateralScale(settings.lateralScale);

      // Restore alignment settings
      if (settings.referenceLandmarks) setReferenceLandmarks(settings.referenceLandmarks);
      if (settings.alignmentMode) setAlignmentMode(settings.alignmentMode);
      if (settings.useLandmarkAlignment !== undefined) setUseLandmarkAlignment(settings.useLandmarkAlignment);

      toast.success('تنظیمات قبلی بازیابی شد');
      console.log('✅ Superimpose settings restored');
    } catch (error) {
      console.error('❌ Failed to load superimpose settings:', error);
    }
  }, [patient?.superimposeSettings, profileImages, lateralImages]);

  // Auto-load settings on mount
  useEffect(() => {
    if (patient?.superimposeSettings) {
      loadSuperimposeSettings();
    }
  }, [patient?.id, patient?.superimposeSettings, loadSuperimposeSettings]); // Only trigger on patient change, not on settings change

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
                  color="success"
                  onClick={saveSuperimposeSettings}
                  startIcon={<Iconify icon="solar:diskette-bold" />}
                  disabled={!patient?.id}
                >
                  ذخیره تنظیمات
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
              <IconButton
                onClick={() => setShowLandmarks(!showLandmarks)}
                color={showLandmarks ? 'primary' : 'default'}
                sx={{
                  bgcolor: showLandmarks ? 'primary.light' : 'transparent',
                  '&:hover': {
                    bgcolor: showLandmarks ? 'primary.main' : 'action.hover',
                  },
                }}
              >
                <Iconify
                  icon={showLandmarks ? "solar:eye-closed-bold" : "solar:eye-bold"}
                  width={20}
                />
              </IconButton>
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
                cursor: 'default',
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
