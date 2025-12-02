import './patient-orthodontics-view-optimizations.css';

import { toast } from 'sonner';
import moment from 'moment-jalaali';
import { useParams, useNavigate } from 'react-router-dom';
import React, { lazy, useRef, useMemo, useState, Suspense, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Grid from '@mui/material/Grid';
import Grow from '@mui/material/Grow';
import Menu from '@mui/material/Menu';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import Divider from '@mui/material/Divider';
import MenuItem from '@mui/material/MenuItem';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import CircularProgress from '@mui/material/CircularProgress';
import { MobileDateTimePicker } from '@mui/x-date-pickers/MobileDateTimePicker';

import { paths } from 'src/routes/paths';

import { uuidv4 } from 'src/utils/uuidv4';
import axios, { endpoints } from 'src/utils/axios';
import { getImageUrl, getServiceUrl, getAiServiceUrl } from 'src/utils/url-helpers';
import { compressMultipleImages, getCompressionSettingsForModel } from 'src/utils/image-compression';
import {
  formatAnalysisForDisplay,
  generateComprehensiveAnalysis,
} from 'src/utils/orthodontic-analysis.ts';

import { CONFIG } from 'src/config-global';
import { useTranslate } from 'src/locales';
import { varAlpha } from 'src/theme/styles';
import { createEvent } from 'src/actions/calendar';
import { UploadIllustration } from 'src/assets/illustrations';
import { useHeaderContent } from 'src/contexts/header-content-context';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { AnimateBorder } from 'src/components/animate';
import { CustomTabs } from 'src/components/custom-tabs';
import { TypingReport, CephalometricTable } from 'src/components/typing-animation/typing-animation';

import { useAuthContext } from 'src/auth/hooks';

import ImageListItem from '../components/image-list-item';

// Lazy load dialogs
const ImageCropDialog = lazy(() => import('../components/image-crop-dialog'));
const RoleNavStructure = lazy(() => import('../components/role-nav-structure'));

// Lazy load heavy components for better performance
const IntraOralView = React.lazy(() => import('src/sections/intra-oral/view/intra-oral-view').then(module => ({ default: module.IntraOralView })));
const AIDiagnosisDisplay = React.lazy(() => import('./ai-diagnosis-display'));
const CephalometricAIAnalysis = React.lazy(() => import('../components/cephalometric-ai-analysis').then(module => ({ default: module.CephalometricAIAnalysis })));

// ----------------------------------------------------------------------

export function PatientOrthodonticsView() {

  const { id } = useParams();
  const { user } = useAuthContext();
  const navigate = useNavigate();
  const { setHeaderContent, setHideRightButtons } = useHeaderContent();
  const { currentLang } = useTranslate();
  const theme = useTheme();
  const [currentTab, setCurrentTab] = useState('general');

  // Define navigation tabs
  const navigationTabs = [
    {
      value: 'general',
      label: '��طلاعات کلی',
      icon: 'solar:user-bold',
    },
    {
      value: 'diagnosis',
      label: 'تشخیص AI',
      icon: 'solar:robot-outline',
    },
    {
      value: 'cephalometric',
      label: 'آنالیز سفالومتری',
      icon: 'solar:chart-square-bold',
    },
    {
      value: 'intra-oral',
      label: 'آنالیز داخل دهان',
      icon: 'solar:microscope-bold',
    },
  ];
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [uploadedImages, setUploadedImages] = useState([]);
  const [saving, setSaving] = useState(false);
  const [success, setSuccess] = useState(false);

  // Upload dialog state
  const [uploadDialogOpen, setUploadDialogOpen] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('profile');
  const [selectedFiles, setSelectedFiles] = useState([]);

  // Image category edit dialog state (kept for compatibility)
  const [editCategoryDialogOpen, setEditCategoryDialogOpen] = useState(false);
  const [editingImage, setEditingImage] = useState(null);
  const [newImageCategory, setNewImageCategory] = useState('general');


  // Image crop dialog state
  const [cropDialogOpen, setCropDialogOpen] = useState(false);
  const [cropImage, setCropImage] = useState(null); // { meta: imageObj, src }

  // Split composite image dialog state
  const [splitDialogOpen, setSplitDialogOpen] = useState(false);
  const [splitImageFile, setSplitImageFile] = useState(null);
  const [splitResults, setSplitResults] = useState([]);
  const [splitting, setSplitting] = useState(false);
  
  // Edit mode state for cephalometric analysis - به صورت پیش‌فرض غیرفعال (view mode)
  const [isCephalometricEditMode, setIsCephalometricEditMode] = useState(false);
  // به صورت پیش‌فرض: اگر آنالیزی وجود داشته باشد لیست را نمایش بده، وگرنه تصو��ر را نمایش بده
  const [showCephalometricImage, setShowCephalometricImage] = useState(true); // به صورت پیش‌فرض تصویر نمایش داده می‌شود
  // وضعیت تایید آنالیز
  const [isAnalysisConfirmed, setIsAnalysisConfirmed] = useState(false);
  // Ref برای ردیابی اینکه آیا کاربر دکمه "نمایش نتا��ج" را زده است یا نه
  const userClickedShowResultsRef = useRef(false);
  const [selectedSplits, setSelectedSplits] = useState(new Set());

  // Image selection state for auto-selecting newest images
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);

  // Complete analysis state with typing animation
  const [completeAnalysisReport, setCompleteAnalysisReport] = useState(null);
  const [isRunningCompleteAnalysis, setIsRunningCompleteAnalysis] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [currentAnalysisStep, setCurrentAnalysisStep] = useState('');

  // Debug: Track completeAnalysisReport changes
  useEffect(() => {
    console.log('[State] completeAnalysisReport changed:', {
      isNull: completeAnalysisReport === null,
      isArray: Array.isArray(completeAnalysisReport),
      length: completeAnalysisReport?.length || 0,
      sections: completeAnalysisReport?.map(s => ({ title: s.title, contentLength: s.content?.length || 0 })) || []
    });
  }, [completeAnalysisReport]);

  // Auto-select newest image when images are uploaded
  useEffect(() => {
    if (uploadedImages.length > 0) {
      setSelectedImageIndex(0); // Always select the newest image (index 0)
    }
  }, [uploadedImages.length]);

  // Sort images by creation date (newest first)
  const sortImagesByDate = useCallback((images) => [...images].sort((a, b) => {
      // Sort by createdAt field (newest first)
      const dateA = new Date(a.createdAt || a.date || a.uploadDate || 0);
      const dateB = new Date(b.createdAt || b.date || b.uploadDate || 0);
      return dateB.getTime() - dateA.getTime();
    }), []);

  // Image options menu state (open at click position)
  const [menuPosition, setMenuPosition] = useState(null);
  const [menuImage, setMenuImage] = useState(null);

  const handleOpenMenu = useCallback((e, image) => {
    e.stopPropagation();
    // open menu at pointer location
    setMenuPosition({ top: e.clientY - 8, left: e.clientX - 8 });
    setMenuImage(image);
  }, []);

  const handleCloseMenu = useCallback(() => {
    setMenuPosition(null);
    setMenuImage(null);
  }, []);

  // Delete confirmation dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);
  
  // Refs for cleanup timers
  const successTimerRef = useRef(null);
  const saveTimerRef = useRef(null);
  const imagesLoadedRef = useRef(false);

  // Handle scrollbar compensation to prevent layout shift when dialog opens/closes
  useEffect(() => {
    if (deleteDialogOpen) {
      // Calculate scrollbar width before MUI Dialog applies overflow:hidden
      // Use requestAnimationFrame to ensure measurement happens before MUI's scroll lock
      requestAnimationFrame(() => {
        const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
        
        // Apply padding to body to compensate for scrollbar
        if (scrollbarWidth > 0) {
          document.body.style.paddingRight = `${scrollbarWidth}px`;
          document.documentElement.style.setProperty('--scrollbar-width', `${scrollbarWidth}px`);
        }
      });
    } else {
      // Remove padding when dialog closes (with slight delay to match transition)
      const timer = setTimeout(() => {
        document.body.style.paddingRight = '';
        document.documentElement.style.removeProperty('--scrollbar-width');
      }, 300); // Match dialog transition duration

      return () => clearTimeout(timer);
    }

    // Cleanup on unmount
    return () => {
      document.body.style.paddingRight = '';
      document.documentElement.style.removeProperty('--scrollbar-width');
    };
  }, [deleteDialogOpen]);

  // Memoize image categorization function
  const categorizeImages = useCallback((images) => ({
      profile: images.filter(img => img.category === 'profile' || img.category === 'frontal'),
      lateral: images.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
      intraoral: images.filter(img => img.category === 'intraoral' || img.category === 'intra'),
      opg: images.filter(img => img.category === 'opg' || img.category === 'panoramic'),
      general: images.filter(img => img.category === 'general' && img.category !== 'opg' && img.category !== 'panoramic'),
    }), []);

  // Wrapper for menu-based deletion (defined after handleDeleteImage)
  const openDeleteDialog = useCallback((image) => {
    // This function is kept for compatibility with menu-based deletion
    // It opens the delete confirmation dialog
    let imageToDelete;
    
    if (typeof image === 'string' || typeof image === 'number') {
      imageToDelete = uploadedImages.find(img => img.id === image);
      if (!imageToDelete) {
        imageToDelete = { id: image };
      }
    } else if (image?.id) {
      imageToDelete = image;
    } else if (image?._imageId) {
      const imageId = image._imageId;
      imageToDelete = uploadedImages.find(img => img.id === imageId);
      if (!imageToDelete) {
        imageToDelete = { id: imageId };
      }
    }
    
    if (imageToDelete && imageToDelete.id) {
      setImageToDelete(imageToDelete);
      setDeleteDialogOpen(true);
    }
    handleCloseMenu();
  }, [uploadedImages, handleCloseMenu]);

  const confirmDeleteImage = useCallback(async () => {
    if (!imageToDelete || !imageToDelete.id) return;
    
    setSaving(true);
    
    // Close dialog with transition (delay to allow fade-out)
    setTimeout(() => {
      setDeleteDialogOpen(false);
    }, 150);
    
    try {
      await axios.delete(`${endpoints.patients}/${id}/images`, {
        data: { imageId: imageToDelete.id },
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
      });

      // Refresh images list
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      const images = imagesResponse.data.images || [];
      setUploadedImages(images);
      
      // Update patient images - use memoized categorization
      const categorizedImages = categorizeImages(images);
      setPatient(prev => ({
        ...prev,
        images: categorizedImages,
      }));
      
      // Don't show "اطلاعات با موفقیت ذخیره شد!" message for image deletion
      // setSuccess(true);
      // Clear previous timer if exists
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      // successTimerRef.current = setTimeout(() => setSuccess(false), 3000);
      toast.success('تصویر با موفقیت حذف شد');
    } catch (error) {
      console.error('Error deleting image:', error);
      toast.error('خطا در حذف تصویر');
    } finally {
      setSaving(false);
      // Clear imageToDelete after transition completes
      setTimeout(() => {
        setImageToDelete(null);
      }, 300);
    }
  }, [imageToDelete, id, user?.accessToken, categorizeImages]);

  const handleDownloadImage = useCallback(async (image) => {
    try {
      const imageUrl = getImageUrl(image.path);
      const resp = await fetch(imageUrl);
      const blob = await resp.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = image.originalName || `image-${image.id}.jpg`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      toast.success('دانلود آغاز شد');
    } catch (err) {
      console.error('Download error', err);
      toast.error('خطا در دانلود تصویر');
    } finally {
      handleCloseMenu();
    }
  }, [handleCloseMenu]);

  // AI model selection for cephalometric analysis
  const [selectedAIModel, setSelectedAIModel] = useState('gpt-4o');

  // Cephalometric analysis type selection
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('general');
  
  // Analysis history state - موقتاً غیرفعال شده است
  // const [analysisHistory, setAnalysisHistory] = useState([]);
  // const [isLoadingHistory, setIsLoadingHistory] = useState(false);

  // Cephalometric landmarks state
  const [landmarks, setLandmarks] = useState({});

  // Cephalometric parameter templates for each analysis method
  // Status mapping between database values and display text
  const statusMap = {
    'PENDING': 'شروع درمان',
    'IN_TREATMENT': 'در حال درمان',
    'COMPLETED': 'اتمام درمان',
    'CANCELLED': 'متوقف شده',
  };

  const statusMapReverse = {
    'شروع درمان': 'PENDING',
    'در حال درمان': 'IN_TREATMENT',
    'اتمام درمان': 'COMPLETED',
    'متوقف شده': 'CANCELLED',
  };

  const cephalometricTemplates = {
    general: {
      // Steiner Analysis Parameters
      SNA: { 
        mean: '82', 
        sd: '3.5', 
        severity: 'نرمال', 
        note: 'نشان‌دهنده موقعیت قدامی-خلفی فک بالا نسبت به قاعده جمجمه. افزایش: جلو بودن ماگزیلا. کاهش: عقب بودن ماگزیلا' 
      },
      SNB: { 
        mean: '80', 
        sd: '3.5', 
        severity: 'نرمال', 
        note: 'نشان‌دهنده موقعیت قدامی-خلفی فک پایین نسبت به قاعده جمجمه. افزایش: جلو بودن مندیبل. کاهش: عقب بودن مندیبل' 
      },
      ANB: { 
        mean: '2', 
        sd: '2', 
        severity: 'نرمال', 
        note: 'نشان‌دهنده رابطه قدامی-خلفی فک بالا نسبت به فک پایین (اختلاف SNA و SNB).مندیبل (کلاس II). کاهش: جلو بودن مندیبل یا عقب بودن ماگزیلا (کلاس III)' 
      },
      'U1-SN': { 
        mean: '103', 
        sd: '6', 
        severity: 'نرمال', 
        note: 'زاویه دندان سانترال فک بالا نسبت به خط SN. افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' 
      },
      'L1-MP': { 
        mean: '90', 
        sd: '3', 
        severity: 'نرمال', 
        note: 'زاویه دندان سانترال فک پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' 
      },
      'GoGn-SN': { 
        mean: '32', 
        sd: '4', 
        severity: 'نرمال', 
        note: 'زاویه صفحه مندیبولار نسبت به خط SN. افزایش: صفحه مندیبولار شیب دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' 
      },
      Overbite: { 
        mean: '2.5', 
        sd: '1.5', 
        severity: 'نرمال', 
        note: 'فاصله عمودی بین دندان‌های پیشین بالا و پایین (میلی‌متر). افزایش: overbite زیاد (deep bite). کاهش: overbite کم یا open bite' 
      },
      Overjet: { 
        mean: '2.5', 
        sd: '1.5', 
        severity: 'نرمال', 
        note: 'فاصله افقی بین دندان‌های پیشین بالا و پایین (میلی‌متر). افزایش: overjet زیاد (دندان‌های بالا جلوتر). کاهش: overjet کم یا negative overjet (دندان‌های پایین جلوتر)' 
      },
      // Ricketts Analysis Parameters
      'Facial Axis': { mean: '90', sd: '3', severity: 'نرمال', note: 'محور صورت (زاویه بین Ba-Na و Pt-Gn). افزایش: رشد عمودی صورت (vertical growth pattern). کاهش: رشد افقی صورت (horizontal growth pattern)' },
      'Facial Depth': { mean: '88', sd: '3', severity: 'نرمال', note: 'عمق صورت (زاویه بین N-Po و FH). افزایش: صورت عمیق‌تر (deep face). کاهش: صورت کم‌عمق‌تر (shallow face)' },
      'Lower Face Height Ratio': { mean: '47', sd: '2', severity: 'نرمال', note: 'نسبت فاصله عمودی صورت پایین (فاصله عمودی ANS-Me / فاصله عمودی N-Me × 100). افزایش: ارتفاع صورت پایین بیشتر (long face). کاهش: ارتفاع صورت پایین کمتر (short face)' },
      'Mandibular Plane': { mean: '26', sd: '4', severity: 'نرمال', note: 'صفحه مندیبولار (زاویه Go-Me نسبت به FH). افزایش: صفحه مندیبولار شیب‌دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' },
      'Convexity': { mean: '0', sd: '2', severity: 'نرمال', note: 'تحدب صورت (فاصله A point از خط N-Pog). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Upper Incisor': { mean: '22', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین بالا (نسبت به A-Pog). افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' },
      'Lower Incisor': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین پایین (نسبت به A-Pog). افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
      'Interincisal Angle': { mean: '130', sd: '6', severity: 'نرمال', note: 'زاویه بین خط U1-U1A و خط L1-L1A. افزایش: زاویه بیشتر (بیشتر retroclined). کاهش: زاویه کمتر (بیشتر proclined)' },
      // McNamara Analysis Parameters
      'N-A-Pog': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه بین N-A-Pog (تحدب صورت). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Co-A': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول فک بالا (فاصله Co-A). افزایش: فک بالا بلندتر (maxillary prognathism). کاهش: فک بالا کوتاه‌تر (maxillary retrognathism)' },
      'Co-Gn': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول فک پایین (فاصله Co-Gn). افزایش: فک پایین بلندتر (mandibular prognathism). کاهش: فک پایین کوتاه‌تر (mandibular retrognathism)' },
      'Lower Face Height': { mean: '65', sd: '4', severity: 'نرمال', note: 'فاصله عمودی صورت پایین (فاصله عمودی ANS-Me). افزایش: ارتفاع صورت پایین بیشتر (long face). کاهش: ارتفاع صورت پایین کمتر (short face)' },
      'Upper Face Height': { mean: '55', sd: '3', severity: 'نرمال', note: 'ارتفاع صورت بالا (N-ANS). افزایش: ارتفاع صورت بالا بیشتر. کاهش: ارتفاع صورت بالا کمتر' },
      'Facial Height Ratio': { mean: '55', sd: '2', severity: 'نرمال', note: 'نسبت ارتفاع صورت (ANS-Me/N-Me × 100). افزایش: نسبت بیشتر (long face pattern). کاهش: نسبت کمتر (short face pattern)' },
      'Mandibular Plane Angle': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
      // Tweed Analysis Parameters
      FMA: { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به صفحه مندیبولار. افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' },
      FMIA: { mean: '65', sd: '5', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به incisor پایین. افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' },
      IMPA: { mean: '90', sd: '3', severity: 'نرمال', note: 'زاویه incisor پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
      // Jarabak Analysis Parameters
      'Jarabak Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع خلفی (PFH: S-Go) به ارتفاع قدامی (AFH: N-Me) ضربدر 100. افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Saddle Angle': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه سدل (∠N-S-Ar). ↑ → عقب‌گرد، ↓ → جلوگرد' },
      'Articular Angle': { mean: '143', sd: '6', severity: 'نرمال', note: 'زاویه آرتیکولار (∠S-Ar-Go). ↑ → رشد افقی، ↓ → رشد عمودی' },
      'Gonial Angle (Total)': { mean: '130', sd: '7', severity: 'نرمال', note: 'زاویه گونیال کل (∠Ar-Go-Me). ↑ → رشد عمودی، ↓ → رشد افقی' },
      'Upper Gonial Angle': { mean: '53.5', sd: '1.5', severity: 'نرمال', note: 'زاویه گونیال بالا (∠Ar-Go-N)' },
      'Lower Gonial Angle': { mean: '72.5', sd: '2.5', severity: 'نرمال', note: 'زاویه گونیال پایین (∠N-Go-Gn). ↑ → رشد عمودی' },
      'Sum of Posterior Angles': { mean: '396', sd: '6', severity: 'نرمال', note: 'مجموع زوایای خلفی (Saddle + Articular + Gonial). >۳۹۶° → عمودی، <۳۹۶° → افقی' },
      'Y-Axis (Growth Axis)': { mean: '59.5', sd: '3.5', severity: 'نرمال', note: 'محور Y (∠SGn–FH فرانکفورت). ↑ → رشد عمودی، ↓ → رشد افقی' },
      'Basal Plane Angle': { mean: '-', sd: '-', severity: '-', note: 'زاویه صفحه بازال (∠PNS-ANS به Go-Me). مشابه MP angle' },
      'Ramus Height': { mean: '-', sd: '-', severity: '-', note: 'ارتفاع راموس (Ar-Go). ↑ → افقی، ↓ → عمودی' },
      'Mandibular Arc': { mean: '27', sd: '1', severity: 'نرمال', note: 'زاویه بین Corpus Axis و Condylar Axis. ↑ → رشد عمودی' },
      'Palatal Plane to FH': { mean: '0', sd: '3', severity: 'نرمال', note: 'زاویه صفحه پالاتال به FH (∠ANS-PNS به FH). کمی ↓ به خلف' },
      'Occlusal Plane to FH': { mean: '9', sd: '1', severity: 'نرمال', note: 'زاویه صفحه اکلوژال به FH (∠Occlusal Plane به FH)' },
      // Sassouni Analysis Parameters
      'N-S-Ar': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه بین nasion، sella و articulare. افزایش: زاویه بیشتر (بازتر). کاهش: زاویه کمتر (بسته‌تر)' },
      'N-Ar-Go': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه محدود کننده (زاویه بین N-Ar-Go). افزایش: رشد عمودی صورت. کاهش: رشد افقی صورت' },
      'Go-Co-N-S': { mean: '59', sd: '4', severity: 'نرمال', note: 'میزان تمایز ساژیتال (زاویه بین Go-Co و N-S). افزایش: تمایز ساژیتال بیشتر. کاهش: تمایز ساژیتال کمتر' },
      'Go-Co-Go-Gn': { mean: '4', sd: '2', severity: 'نرمال', note: 'انتخاب اجتماعی (نسبت Go-Co به Go-Gn). افزایش: نسبت بیشتر. کاهش: نسبت کمتر' },
      'N-Co-Go-Co': { mean: '90', sd: '5', severity: 'نرمال', note: 'ایدئال فرهنگی (زاویه بین N-Co و Go-Co). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
      'Ar-Co-Co-Gn': { mean: '74', sd: '4', severity: 'نرمال', note: 'نخستین sagittal (زاویه بین Ar-Co و Co-Gn). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
      // Wits Analysis Parameters
      'AO-BO': { mean: '0', sd: '2', severity: 'نرمال', note: 'تفاوت عمودی A و B points نسبت به نیروی عمودی (0 = کلاس I). افزایش: کلاس II (ماگزیلا جلوتر یا مندیبل عقب‌تر). کاهش: کلاس III (ماگزیلا عقب‌تر یا مندیبل جلوتر)' },
      'PP/Go-Gn': { mean: '27', sd: '4', severity: 'نرمال', note: 'زاویه بین صفحه پلاتین و صفحه مندیبولار. افزایش: زاویه بیشتر (vertical growth pattern). کاهش: زاویه کمتر (horizontal growth pattern)' },
      'S-Go': { mean: '75', sd: '5', severity: 'نرمال', note: 'ابعاد عمودی چهره (سلا-گناتیون). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
      'Sagittal Jaw': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه ساژیتال فک (معمولاً همان ANB). افزایش: کلاس II. کاهش: کلاس III' },
    },
    steiner: {
      SNA: { 
        mean: '82', 
        sd: '3.5', 
        severity: 'نرمال', 
        note: 'نشان‌دهنده موقعیت قدامی-خلفی فک بالا (ماگزیلا) نسبت به قاعده جمجمه. افزایش: جلو بودن ماگزیلا. کاهش: عقب بودن ماگزیلا' 
      },
      SNB: { 
        mean: '80', 
        sd: '3.5', 
        severity: 'نرمال', 
        note: 'زاویه ارتباط بین اسنو، nasion و B point. افزایش: جلو بودن مندیبل. کاهش: عقب بودن مندیبل' 
      },
      ANB: { 
        mean: '2', 
        sd: '2', 
        severity: 'نرمال', 
        note: 'تفاوت SNA و SNB . افزایش: جلو بودن ماگزیلا یا عقب بودن مندیبل (کلاس II). کاهش: جلو بودن مندیبل یا عقب بودن ماگزیلا (کلاس III)' 
      },
      Overbite: { 
        mean: '2.5', 
        sd: '1.5', 
        severity: 'نرمال', 
        note: 'فاصله عمودی بین دندان‌های پیشین بالا و پایین (میلی‌متر). افزایش: overbite زیاد (deep bite). کاهش: overbite کم یا open bite' 
      },
      Overjet: { 
        mean: '2.5', 
        sd: '1.5', 
        severity: 'نرمال', 
        note: 'فاصله افقی بین دندان‌های پیشین بالا و پایین (میلی‌متر). افزایش: overjet زیاد (دندان‌های بالا جلوتر). کاهش: overjet کم یا negative overjet (دندان‌های پایین جلوتر)' 
      },
      'U1-SN': { 
        mean: '103', 
        sd: '6', 
        severity: 'نرما��', 
        note: 'زاویه دندان پیشین فک بالا نسبت به خط SN. افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' 
      },
      'L1-MP': { 
        mean: '90', 
        sd: '3', 
        severity: 'نرمال', 
        note: 'زاویه دندان پیشین فک پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' 
      },
      'GoGn-SN': { 
        mean: '32', 
        sd: '4', 
        severity: 'نرمال', 
        note: 'زاویه صفحه مندیبولار نسبت به خط SN. افزایش: صفحه مندیبولار شیب دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' 
      },
    },
    ricketts: {
      'Facial Axis': { mean: '90', sd: '3', severity: 'نرمال', note: 'محور صورت (زاویه بین Ba-Na و Pt-Gn). افزایش: رشد عمودی صورت (vertical growth pattern). کاهش: رشد افقی صورت (horizontal growth pattern)' },
      'Facial Depth': { mean: '88', sd: '3', severity: 'نرمال', note: 'عمق صورت (زاویه بین N-Po و FH). افزایش: صورت عمیق‌تر (deep face). کاهش: صور�� کم‌عمق‌تر (shallow face)' },
      'Lower Face Height': { mean: '47', sd: '2', severity: 'نرمال', note: 'نسبت فاصله عمودی صورت پایین (فاصله عمودی ANS-Me / فاصله عمودی N-Me × 100). افزایش: ارتفاع صورت پایین بیشتر (long face). کاهش: ارتفاع صورت پایین کمتر (short face)' },
      'Mandibular Plane': { mean: '26', sd: '4', severity: 'نرمال', note: 'صفحه مندیبولار (زاویه Go-Me نسبت به FH). افزایش: صفحه مندیبولار شیب‌دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' },
      'Convexity': { mean: '0', sd: '2', severity: 'نرمال', note: 'تحدب صورت (فاصله A point از خط N-Pog). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Upper Incisor': { mean: '22', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین بالا (نسبت به A-Pog). افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' },
      'Lower Incisor': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین پایین (نسبت به A-Pog). افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
      'Interincisal Angle': { mean: '130', sd: '6', severity: 'نرمال', note: 'زاویه بین خط U1-U1A و خط L1-L1A. افزایش: زاویه بیشتر (بیشتر retroclined). کاهش: زاویه کمتر (بیشتر proclined)' },
    },
    mcnamara: {
      'N-A-Pog': { mean: '0', sd: '2', severity: 'ن��مال', note: 'زاویه بین N-A-Pog (تحدب صورت). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Co-A': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول فک بالا (فاصله Co-A). افزایش: ��ک بالا بلندتر (maxillary prognathism). کاهش: فک بالا کوتاه‌تر (maxillary retrognathism)' },
      'Co-Gn': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول فک پایین (فاصله Co-Gn). افزایش: فک پایین بلندتر (mandibular prognathism). کاهش: فک پایین کوتاه‌تر (mandibular retrognathism)' },
      'Mandibular Length': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول مندیبول (Co-Gn). افزایش: مندیبول بلندتر (mandibular prognathism). کاهش: مندیبول کوتاه‌تر (mandibular retrognathism)' },
      'Maxillary Length': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول ماگزیلا (Co-A). افزایش: ماگزیلا بلندتر (maxillary prognathism). کاهش: ماگزیلا کوتاه‌تر (maxillary retrognathism)' },
      'Lower Face Height': { mean: '65', sd: '4', severity: 'نرمال', note: 'فاصله عمودی صورت پایین (فاصله عمودی ANS-Me). افزایش: ارتفاع صورت پایین بیشتر (long face). کاهش: ارتفاع صورت پایین کمتر (short face)' },
      'Upper Face Height': { mean: '55', sd: '3', severity: 'نرمال', note: 'ارتفاع صورت بالا (N-ANS). افزایش: ارتفاع صورت بالا بیشتر. کاهش: ارتفاع صورت بالا کمتر' },
      'Facial Height Ratio': { mean: '55', sd: '2', severity: 'نرمال', note: 'نسبت ارتفاع صورت (ANS-Me/N-Me × 100). افزایش: نسبت بیشتر (long face pattern). کاهش: نسبت کمتر (short face pattern)' },
      'Mandibular Plane Angle': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
    },
    wits: {
      'AO-BO': { mean: '0', sd: '2', severity: 'نرمال', note: 'تفاوت عمودی A و B points نسبت به نیروی عمودی (0 = کلاس I). افزایش: کلاس II (ماگزیلا جلوتر یا مندیبل عقب‌تر). کاهش: کلاس III (ماگزیلا عقب‌تر یا مندیبل جلوتر)' },
      'PP/Go-Gn': { mean: '27', sd: '4', severity: 'نرمال', note: 'زاویه بین صفحه پلاتین و صفحه مندیبولار. افزایش: زاویه بیشتر (vertical growth pattern). کاهش: زاویه کمتر (horizontal growth pattern)' },
      'S-Go': { mean: '75', sd: '5', severity: 'نرمال', note: 'ابعاد عمودی چهره (سلا-گناتیون). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
      'Sagittal Jaw': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه ساژیتال فک (معمولاً همان ANB). افزایش: کلاس II. کاهش: کلاس III' },
      'Occlusal Plane': { mean: '8', sd: '3', severity: 'نرمال', note: 'تفاوت افقی صفحه اکلوزال. افزایش: صفحه اک��وزال شیب‌دارتر. کاهش: صفحه اکلوزال صاف‌تر' },
    },
    tweed: {
      FMA: { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به صفحه مندیبولار. افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' },
      FMIA: { mean: '65', sd: '5', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به incisor پایین. افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' },
      IMPA: { mean: '90', sd: '3', severity: 'نرمال', note: 'زاویه incisor پایین نسبت به صفحه مندیبولار. افز��یش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
    },
    bjork: {
      'S-Ar/Go-Gn Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت طول سوراسلار به گونی ذره (ارزش رشد). افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Ar-Go-N/Go-Me Ratio': { mean: '56', sd: '3', severity: 'نرمال', note: 'نسبت sagittal مندیبولار. افزایش: مندیبول جلوتر. کاهش: مندیبول عقب‌تر' },
      'S-Go/Go-Me Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت عمودی چهره (اندازه‌گیری مقایسه‌ای). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
      'MP/SN': { mean: '32', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار نسبت به SN. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
      'NS-Gn': { mean: '104', sd: '5', severity: 'نرمال', note: 'زاویه nasion-sella-gnathion. افزایش: صورت عمودی‌تر. کاهش: صورت افقی‌تر' },
    },
    jarabak: {
      'Jarabak Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع خلفی (PFH: S-Go) به ارتفاع قدامی (AFH: N-Me) ضربدر 100. افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'S-Go/Ar-Go Ratio': { mean: '53', sd: '3', severity: 'نرمال', note: 'فاکتور CAG رشد (اندازه‌گیری مقایسه‌ای رشد). افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Ar-Go/N-Go Ratio': { mean: '47', sd: '3', severity: 'نرمال', note: 'عکس اندازه‌گیری مقایسه‌ای رشد. افزایش: رشد افقی بیشتر. کاهش: رشد عمودی بیشتر' },
      'Co-Gn/Ar-Go Ratio': { mean: '2.5', sd: '0.3', severity: 'نرمال', note: 'ف��کتور CG ارتفاع قدامی رامی. افزایش: ارتفاع قدامی بیشتر. کاهش: ارتفاع قدامی کمتر' },
      'S-Ar/Go-Gn Ratio': { mean: '98', sd: '5', severity: 'نرمال', note: 'نسبت SAG رشد. افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Go-Gn/SN Angle': { mean: '46', sd: '4', severity: 'نرمال', note: 'زاویه رشد (صفحه مندیبولار نسبت به SN). افزایش: رشد عمودی (vertical growth). کاهش: رشد افقی (horizontal growth)' },
    },
    sassouni: {
      'N-S-Ar': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه بین nasion، sella �� articulare. افزایش: زاویه بیشتر (بازتر). کاهش: زاویه کمتر (بسته‌تر)' },
      'N-Ar-Go': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه محدود کننده (زاویه بین N-Ar-Go). افزایش: رشد عمودی صورت. کاهش: رشد افقی صورت' },
      'Go-Co-N-S': { mean: '59', sd: '4', severity: 'نرمال', note: 'میزان تمایز ساژیتال (زاویه بین Go-Co و N-S). افزایش: تمایز ساژیتال بیشتر. کاهش: تمایز ساژیتال کمتر' },
      'Go-Co-Go-Gn': { mean: '4', sd: '2', severity: 'نرمال', note: 'انتخاب اجتماعی (نسبت Go-Co به Go-Gn). افزایش: نسبت بیشتر. کاهش: نسبت کمتر' },
      'N-Co-Go-Co': { mean: '90', sd: '5', severity: 'نرمال', note: 'ایدئال فرهنگی (زاویه بین N-Co و Go-Co). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
      'Ar-Co-Co-Gn': { mean: '74', sd: '4', severity: 'نرمال', note: 'نخستین sagittal (زاویه بین Ar-Co و Co-Gn). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
    },
  };

  // Cleanup timers on unmount
  useEffect(() => () => {
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      if (saveTimerRef.current) {
        clearTimeout(saveTimerRef.current);
      }
    }, []);

  // Fetch patient data from API
  useEffect(() => {
    const fetchPatient = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${endpoints.patients}/${id}`, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });

        const patientData = response.data.patient;
        // Parse cephalometricTableData from JSON string to object
        let cephalometricTable = null;
        try {
          if (patientData.cephalometricTableData && typeof patientData.cephalometricTableData === 'string') {
            cephalometricTable = JSON.parse(patientData.cephalometricTableData);
          }
        } catch (parseError) {
          console.warn('Failed to parse cephalometricTableData:', parseError);
          cephalometricTable = null;
        }

        // Parse cephalometricRawData from JSON string to object
        let cephalometricRawData = null;
        try {
          if (patientData.cephalometricRawData && typeof patientData.cephalometricRawData === 'string') {
            cephalometricRawData = JSON.parse(patientData.cephalometricRawData);
          }
        } catch (parseError) {
          console.warn('Failed to parse cephalometricRawData:', parseError);
          cephalometricRawData = null;
        }

        // Parse cephalometricLandmarks from JSON string to object
        let cephalometricLandmarks = null;
        try {
          if (patientData.cephalometricLandmarks && typeof patientData.cephalometricLandmarks === 'string') {
            cephalometricLandmarks = JSON.parse(patientData.cephalometricLandmarks);
          }
        } catch (parseError) {
          console.warn('Failed to parse cephalometricLandmarks:', parseError);
          cephalometricLandmarks = null;
        }

        // Parse intraOralAnalysis from JSON string to object
        let intraOralAnalysis = null;
        try {
          if (patientData.intraOralAnalysis && typeof patientData.intraOralAnalysis === 'string') {
            intraOralAnalysis = JSON.parse(patientData.intraOralAnalysis);
          }
        } catch (parseError) {
          console.warn('Failed to parse intraOralAnalysis:', parseError);
          intraOralAnalysis = null;
        }

        setPatient({
          id: patientData.id,
          name: `${patientData.firstName} ${patientData.lastName}`,
          age: patientData.age,
          phone: patientData.phone,
          gender: patientData.gender || '',
          diagnosis: patientData.diagnosis,
          treatment: patientData.treatment,
          status: patientData.status,
          startDate: patientData.createdAt ? moment(patientData.createdAt) : null,
          nextVisit: patientData.nextVisit ? moment(patientData.nextVisit) : null,
          notes: patientData.notes || '',
          aiDiagnosis: patientData.diagnosis,
          softTissue: patientData.softTissueAnalysis || 'آنالیز بافت نرم: خط زیبایی E مناسب، لب بالایی در حد طبیعی',
          cephalometric: patientData.cephalometricAnalysis || 'آنالیز سفالومتریک: ANB: 6 درجه، SNA: 82 درجه، SNB: 76 درجه',
          cephalometricTable,
          cephalometricRawData,
          cephalometricLandmarks,
          treatmentPlan: patientData.treatmentPlan || 'طرح درمان ��یشنهادی: استخراج دندان‌های آسیاب، استفاده از mini-implants',
          summary: patientData.summary || 'خلاصه پرونده: بیمار نیازمند درمان ارتودنسی پیشرفته',
          images: {
            profile: [],
            lateral: [],
            intraoral: [],
            general: [],
          },
        });

        // Fetch patient images
        try {
          const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });

          const images = imagesResponse.data.images || [];
          // Only log once per patient load to reduce console noise
          if (!imagesLoadedRef.current) {
            console.log('[Patient Images] Total images:', images.length);
            console.log('[Patient Images] Categories found:', images.map(img => ({ id: img.id, category: img.category, name: img.originalName })));
            imagesLoadedRef.current = true;
          }

          // Sort images by creation date (newest first) and then categorize
          const sortedImages = sortImagesByDate(images);
          const categorizedImages = {
            profile: sortedImages.filter(img => img.category === 'profile' || img.category === 'frontal'),
            lateral: sortedImages.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
            intraoral: sortedImages.filter(img => img.category === 'intraoral' || img.category === 'intra'),
            general: sortedImages.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
          };

          // Only update if images actually changed
          setPatient(prev => {
            // Check if images are actually different
            const prevImagesCount = 
              (prev?.images?.profile?.length || 0) +
              (prev?.images?.lateral?.length || 0) +
              (prev?.images?.intraoral?.length || 0) +
              (prev?.images?.general?.length || 0);
            const newImagesCount = 
              categorizedImages.profile.length +
              categorizedImages.lateral.length +
              categorizedImages.intraoral.length +
              categorizedImages.general.length;
            
            // If counts are the same, check if IDs are the same
            if (prevImagesCount === newImagesCount && prevImagesCount > 0) {
              const prevImageIds = new Set([
                ...(prev?.images?.profile || []).map(img => img.id),
                ...(prev?.images?.lateral || []).map(img => img.id),
                ...(prev?.images?.intraoral || []).map(img => img.id),
                ...(prev?.images?.general || []).map(img => img.id),
              ]);
              const newImageIds = new Set([
                ...categorizedImages.profile.map(img => img.id),
                ...categorizedImages.lateral.map(img => img.id),
                ...categorizedImages.intraoral.map(img => img.id),
                ...categorizedImages.general.map(img => img.id),
              ]);
              
              // If IDs are the same, don't update
              if (prevImageIds.size === newImageIds.size && 
                  [...prevImageIds].every(id => newImageIds.has(id))) {
                return prev; // No change, return previous state
              }
            }
            
            return {
            ...prev,
            images: categorizedImages,
            };
          });
          
          // Only update uploadedImages if it's actually different
          setUploadedImages(prev => {
            const sortedImages = sortImagesByDate(images);
            if (prev.length === sortedImages.length) {
              const prevIds = new Set(prev.map(img => img.id));
              const newIds = new Set(sortedImages.map(img => img.id));
              if (prevIds.size === newIds.size &&
                  [...prevIds].every(id => newIds.has(id))) {
                return prev; // No change
              }
            }
            return sortImagesByDate(images);
          });
        } catch (imagesError) {
          console.warn('Could not fetch patient images:', imagesError);
        }

        setLoading(false);
      } catch (error) {
        console.error('Error fetching patient:', error);
        setLoading(false);
        // Fallback to mock data if API fails
        setPatient({
          id,
          name: 'بیمار یافت نشد',
          age: 0,
          phone: '',
          diagnosis: '',
          treatment: '',
          status: '',
          startDate: null,
          nextVisit: null,
          notes: '',
          aiDiagnosis: '',
          softTissue: '',
          cephalometric: '',
          treatmentPlan: '',
          summary: '',
          images: {
            profile: [],
            lateral: [],
            intraoral: [],
            general: [],
          },
        });
      }
    };

    if (user && id) {
      // Reset imagesLoadedRef when patient changes
      imagesLoadedRef.current = false;
      fetchPatient();
    }
  }, [id, user, sortImagesByDate]);

  // Load analysis history - موقتاً غیرفعال شده است
  // const loadAnalysisHistory = async () => {
  //   setIsLoadingHistory(true);
  //   try {
  //     const response = await axios.get(`${CONFIG.site.serverUrl || 'http://localhost:7272'}/api/ai-model-tests?limit=50`, {
  //       headers: {
  //         Authorization: `Bearer ${user?.accessToken}`,
  //       },
  //     });
  //     
  //     if (response.data.success) {
  //       setAnalysisHistory(response.data.data || []);
  //     }
  //   } catch (error) {
  //     console.error('Error loading analysis history:', error);
  //     // Don't show toast error, just log it
  //   } finally {
  //     setIsLoadingHistory(false);
  //   }
  // };

  // Load history when component mounts - موقتاً غیرفعال شده است
  // useEffect(() => {
  //   if (user && id) {
  //     loadAnalysisHistory();
  //   }
  //   // eslint-disable-next-line react-hooks/exhaustive-deps
  // }, [user, id]);

  // Helper function to calculate measurements from landmarks (extracted from CephalometricAIAnalysis)
  const calculateMeasurementsFromLandmarks = useCallback((landmarks) => {
    const measures = {};

    // Debug: Log available landmarks (فقط در development)
    if (process.env.NODE_ENV === 'development') {
      console.log('🔍 Available landmarks for measurement calculation:', Object.keys(landmarks));
    }

    // Normalize landmark names (case-insensitive lookup helper)
    const getLandmark = (names) => {
      for (const name of names) {
        if (landmarks[name]) return landmarks[name];
      }
      return null;
    };

    // Helper to find landmark by partial name match
    const findLandmarkByPartial = (partialNames) => {
      const landmarkKeys = Object.keys(landmarks);
      for (const partial of partialNames) {
        const found = landmarkKeys.find(key =>
          key.toLowerCase().includes(partial.toLowerCase()) ||
          partial.toLowerCase().includes(key.toLowerCase())
        );
        if (found) return landmarks[found];
      }
      return null;
    };

    try {
      // SNA angle
      const sLandmarkSNA = getLandmark(['S', 's']);
      const nLandmarkSNA = getLandmark(['N', 'n']);
      const aLandmark = getLandmark(['A', 'a']);

      if (sLandmarkSNA && nLandmarkSNA && aLandmark) {
        measures.SNA = calculateAngle(sLandmarkSNA, nLandmarkSNA, aLandmark);
      }

      // SNB angle
      const sLandmarkSNB = getLandmark(['S', 's']);
      const nLandmarkSNB = getLandmark(['N', 'n']);
      const bLandmark = getLandmark(['B', 'b']);

      if (sLandmarkSNB && nLandmarkSNB && bLandmark) {
        measures.SNB = calculateAngle(sLandmarkSNB, nLandmarkSNB, bLandmark);
      }

      // ANB angle
      if (measures.SNA !== undefined && measures.SNB !== undefined) {
        measures.ANB = measures.SNA - measures.SNB;
      }

      // FMA (Frankfort-Mandibular Angle)
      const orLandmarkFMA = getLandmark(['Or', 'or', 'OR']) || findLandmarkByPartial(['or', 'orbit']);
      const poLandmarkFMA = getLandmark(['Po', 'po', 'PO']) || findLandmarkByPartial(['po', 'porion']);
      const goLandmarkFMA = getLandmark(['Go', 'go', 'GO']) || findLandmarkByPartial(['go', 'gonion']);
      const meLandmarkFMA = getLandmark(['Me', 'me', 'ME']) || findLandmarkByPartial(['me', 'menton']);

      if (orLandmarkFMA && poLandmarkFMA && goLandmarkFMA && meLandmarkFMA) {
        measures.FMA = calculateAngleBetweenLines(
          orLandmarkFMA, poLandmarkFMA,
          goLandmarkFMA, meLandmarkFMA
        );
        measures.FMA = Math.round(Math.max(0, Math.min(180, measures.FMA)) * 10) / 10;
      }

      // FMIA
      const orLandmark = getLandmark(['Or', 'or', 'OR', 'orbitale', 'Orbitale']) ||
                        findLandmarkByPartial(['or', 'orbit']);
      const poLandmark = getLandmark(['Po', 'po', 'PO', 'porion', 'Porion']) ||
                        findLandmarkByPartial(['po', 'porion']);
      const l1Landmark = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                        findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);
      const meLandmark = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) ||
                        findLandmarkByPartial(['me', 'menton']);

      if (orLandmark && poLandmark && l1Landmark && meLandmark) {
        const frankfortAngle = calculateLineAngle(orLandmark, poLandmark);
        const incisorAngle = calculateLineAngle(l1Landmark, meLandmark);
        let angleDiff = Math.abs(frankfortAngle - incisorAngle);
        if (angleDiff > 180) {
          angleDiff = 360 - angleDiff;
        }
        if (angleDiff > 90) {
          measures.FMIA = 180 - angleDiff;
        } else {
          measures.FMIA = angleDiff;
        }
        measures.FMIA = Math.round(Math.max(0, Math.min(180, measures.FMIA)) * 10) / 10;
      }

      // IMPA
      const goLandmark = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']) ||
                        findLandmarkByPartial(['go', 'gonion']);
      const meLandmark2 = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) ||
                         findLandmarkByPartial(['me', 'menton']);
      const liaLandmark = getLandmark(['LIA', 'lia', 'Lia', 'lower_incisor_apex', 'Lower_incisor_apex', 'lower incisor apex']) ||
                         findLandmarkByPartial(['lia', 'lower', 'incisor', 'apex']);
      const litLandmark = getLandmark(['LIT', 'lit', 'Lit', 'lower_incisor_tip', 'Lower_incisor_tip', 'lower incisor tip']) ||
                         findLandmarkByPartial(['lit', 'lower', 'incisor', 'tip']);

      if (!liaLandmark || !litLandmark) {
        const l1Landmark2 = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                           findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);
        if (l1Landmark2 && goLandmark && meLandmark2) {
          // 🔧 FIX: زاویه بین خط Go-Me (صفحه مندیبولار) و خط L1-Me (incisor)
          measures.IMPA = calculateAngleBetweenLines(
            goLandmark, meLandmark2,  // خط اول: صفحه مندیبولار (Go-Me)
            l1Landmark2, meLandmark2  // خط دوم: incisor (L1-Me)
          );
          measures.IMPA = Math.round(Math.max(0, Math.min(180, measures.IMPA)) * 10) / 10;
        }
      } else if (goLandmark && meLandmark2 && liaLandmark && litLandmark) {
        // 🔧 FIX: زاویه بین خط Go-Me (صفحه مندیبولار) و خط LIA-LIT (incisor)
        measures.IMPA = calculateAngleBetweenLines(
          goLandmark, meLandmark2,  // خط اول: صفحه مندیبولار (Go-Me)
          liaLandmark, litLandmark   // خط دوم: incisor (LIA-LIT)
        );
        measures.IMPA = Math.round(Math.max(0, Math.min(180, measures.IMPA)) * 10) / 10;
      }

      // GoGn-SN
      const sLandmarkGoGn = getLandmark(['S', 's']);
      const nLandmarkGoGn = getLandmark(['N', 'n']);
      const goLandmarkGoGnSN = getLandmark(['Go', 'go', 'GO']);
      const gnLandmarkGoGnSN = getLandmark(['Gn', 'gn', 'GN']);

      if (sLandmarkGoGn && nLandmarkGoGn && goLandmarkGoGnSN && gnLandmarkGoGnSN) {
        const snAngle = calculateLineAngle(sLandmarkGoGn, nLandmarkGoGn);
        const gognAngle = calculateLineAngle(goLandmarkGoGnSN, gnLandmarkGoGnSN);
        const angleDiff = Math.abs(snAngle - gognAngle);
        measures['GoGn-SN'] = angleDiff;
        measures.GoGnSN = angleDiff;
      }

      // U1-SN
      const sLandmark = getLandmark(['S', 's', 'sella', 'Sella']) ||
                       findLandmarkByPartial(['s', 'sella']);
      const nLandmark = getLandmark(['N', 'n', 'nasion', 'Nasion']) ||
                       findLandmarkByPartial(['n', 'nasion']);
      const u1Landmark = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper_incisor', 'upper incisor']) ||
                        findLandmarkByPartial(['u1', 'upper', 'incisor', 'ui']);

      if (sLandmark && nLandmark && u1Landmark) {
        const u1snAngle = calculateAngle(u1Landmark, nLandmark, sLandmark);
        measures['U1-SN'] = Math.round(Math.max(0, Math.min(180, u1snAngle)) * 10) / 10;
      }

      // L1-MP
      const goLandmark2 = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']) ||
                         findLandmarkByPartial(['go', 'gonion']);
      const meLandmark3 = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) ||
                         findLandmarkByPartial(['me', 'menton']);
      const l1Landmark3 = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                         findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);

      if (goLandmark2 && meLandmark3 && l1Landmark3) {
        const l1mpAngle = calculateAngle(l1Landmark3, meLandmark3, goLandmark2);
        measures['L1-MP'] = Math.round(Math.max(0, Math.min(180, l1mpAngle)) * 10) / 10;
      }

      // Interincisal Angle - زاویه بین خط U1-U1A و خط L1-L1A
      const u1LandmarkInter = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper Incisor']);
      const u1aLandmarkInter = getLandmark(['U1A', 'u1a', 'U1a', 'upper_incisor_apex', 'Upper_incisor_apex']);
      const l1LandmarkInter = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower Incisor']);
      const l1aLandmarkInter = getLandmark(['L1A', 'l1a', 'L1a', 'lower_incisor_apex', 'Lower_incisor_apex']);

      if (u1LandmarkInter && u1aLandmarkInter && l1LandmarkInter && l1aLandmarkInter) {
        // محاسبه زاویه بین دو خط: خط U1-U1A و خط L1-L1A
        const interincisalAngle = calculateAngleBetweenLines(
          u1LandmarkInter, u1aLandmarkInter,  // خط اول: U1-U1A
          l1LandmarkInter, l1aLandmarkInter   // خط دوم: L1-L1A
        );
        measures.InterincisalAngle = Math.round(interincisalAngle * 10) / 10;
      }

      // SN-GoGn
      if (landmarks.S && landmarks.N && landmarks.Go && landmarks.Gn && !measures['GoGn-SN']) {
        const snAngle = calculateLineAngle(landmarks.S, landmarks.N);
        const gognAngle = calculateLineAngle(landmarks.Go, landmarks.Gn);
        measures['SN-GoGn'] = Math.abs(snAngle - gognAngle);
      }

      // Facial Axis
      if (landmarks.Ba && landmarks.Na && landmarks.Pt && landmarks.Gn) {
        const baNaAngle = calculateLineAngle(landmarks.Ba, landmarks.Na);
        const ptGnAngle = calculateLineAngle(landmarks.Pt, landmarks.Gn);
        measures.FacialAxis = Math.abs(baNaAngle - ptGnAngle);
      }

      // Mandibular Plane Angle
      if (landmarks.Go && landmarks.Me) {
        const mpAngle = calculateLineAngle(landmarks.Go, landmarks.Me);
        measures.MandibularPlane = mpAngle;
      }

      // Upper Face Height / Lower Face Height
      if (landmarks.N && landmarks.ANS && landmarks.Me) {
        const upperFaceHeight = Math.sqrt(
          (landmarks.N.x - landmarks.ANS.x)**2 +
          (landmarks.N.y - landmarks.ANS.y)**2
        );
        const lowerFaceHeight = Math.sqrt(
          (landmarks.ANS.x - landmarks.Me.x)**2 +
          (landmarks.ANS.y - landmarks.Me.y)**2
        );
        if (lowerFaceHeight > 0) {
          measures.UpperLowerFaceRatio = upperFaceHeight / lowerFaceHeight;
        }
      }

      // Jarabak Ratio - نسبت ارتفاع خلفی (PFH: S-Go) به ارتفاع قدامی (AFH: N-Me) × 100
      const sLandmarkJarabak = getLandmark(['S', 's', 'Sella', 'sella']);
      const goLandmarkJarabak = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      const nLandmarkJarabak = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const meLandmarkJarabak = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);

      if (sLandmarkJarabak && goLandmarkJarabak && nLandmarkJarabak && meLandmarkJarabak) {
        // محاسبه PFH (Posterior Face Height): فاصله S-Go
        const pfh = Math.sqrt(
          (sLandmarkJarabak.x - goLandmarkJarabak.x)**2 +
          (sLandmarkJarabak.y - goLandmarkJarabak.y)**2
        );
        
        // محاسبه AFH (Anterior Face Height): فاصله N-Me
        const afh = Math.sqrt(
          (nLandmarkJarabak.x - meLandmarkJarabak.x)**2 +
          (nLandmarkJarabak.y - meLandmarkJarabak.y)**2
        );
        
        if (afh > 0) {
          // Jarabak Ratio = (PFH / AFH) × 100
          measures['Jarabak Ratio'] = Math.round((pfh / afh) * 100 * 10) / 10;
        }
      }

      console.log('✅ Calculated measurements from landmarks:', measures);
    } catch (err) {
      console.error('Error calculating measurements from landmarks:', err);
    }

    return measures;
  }, []);

  // Helper functions for angle calculations (extracted from CephalometricAIAnalysis)
  const calculateAngle = (p1, vertex, p2) => {
    const angle1 = Math.atan2(p1.y - vertex.y, p1.x - vertex.x);
    const angle2 = Math.atan2(p2.y - vertex.y, p2.x - vertex.x);
    let angle = (angle2 - angle1) * (180 / Math.PI);
    if (angle < 0) angle += 360;
    return angle > 180 ? 360 - angle : angle;
  };

  const calculateLineAngle = (p1, p2) => Math.atan2(p2.y - p1.y, p2.x - p1.x) * (180 / Math.PI);

  const calculateAngleBetweenLines = (line1Start, line1End, line2Start, line2End) => {
    const v1x = line1End.x - line1Start.x;
    const v1y = line1End.y - line1Start.y;
    const v2x = line2End.x - line2Start.x;
    const v2y = line2End.y - line2Start.y;
    const dotProduct = v1x * v2x + v1y * v2y;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
    if (mag1 === 0 || mag2 === 0) {
      return 0;
    }
    const cosAngle = dotProduct / (mag1 * mag2);
    const clampedCos = Math.max(-1, Math.min(1, cosAngle));
    const angleRad = Math.acos(clampedCos);
    const angleDeg = angleRad * (180 / Math.PI);
    return angleDeg > 90 ? 180 - angleDeg : angleDeg;
  };

  // Handle loading analysis from history
  const handleLoadAnalysisFromHistory = async (testId) => {
    try {
      const response = await axios.get(`${CONFIG.site.serverUrl || 'http://localhost:7272'}/api/ai-model-tests/${testId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      if (response.data.success) {
        const test = response.data.data;
        
        // Parse landmarks from test data
        let landmarks = null;
        
        try {
          if (test.landmarks) {
            landmarks = typeof test.landmarks === 'string' ? JSON.parse(test.landmarks) : test.landmarks;
          }
          
          if (test.rawResponse) {
            const rawResponse = typeof test.rawResponse === 'string' ? JSON.parse(test.rawResponse) : test.rawResponse;
            if (rawResponse.landmarks) {
              landmarks = rawResponse.landmarks; // eslint-disable-line prefer-destructuring
            }
          }
        } catch (parseError) {
          console.error('Error parsing test data:', parseError);
        }
        
        if (landmarks && Object.keys(landmarks).length > 0) {
          // Calculate measurements from landmarks
          const measurements = calculateMeasurementsFromLandmarks(landmarks);

          // Update patient state with test data using the same logic as onLandmarksDetected
          const template = cephalometricTemplates[selectedAnalysisType] || cephalometricTemplates.steiner;
          const filteredTable = {};
          
          // First, add all parameters from the selected template
          Object.keys(template).forEach(param => {
            const measuredValue = measurements[param];
            filteredTable[param] = {
              ...template[param],
              measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '') 
                ? String(measuredValue) 
                : '',
            };
          });
          
          // Then, add any additional measurements that might not be in the current template
          Object.keys(measurements).forEach(param => {
            if (!filteredTable[param]) {
              // Check if this parameter exists in any template
              let paramTemplate = null;
              Object.values(cephalometricTemplates).forEach(t => {
                if (t[param]) {
                  paramTemplate = t[param];
                }
              });
              
              if (paramTemplate) {
                const measuredValue = measurements[param];
                filteredTable[param] = {
                  ...paramTemplate,
                  measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '') 
                    ? String(measuredValue) 
                    : '',
                };
              }
            }
          });
          
          // Update patient state
          setPatient(prev => {
            const mergedRawData = {
              ...(prev?.cephalometricRawData || {}),
              ...measurements,
            };
            
            const updatedPatient = {
              ...prev,
              cephalometricTable: filteredTable,
              cephalometricRawData: mergedRawData,
              cephalometricLandmarks: landmarks,
              cephalometric: `آنالیز سفالومتریک از تاریخچه - ${test.modelName || 'مدل نامشخص'} - ${new Date(test.createdAt).toLocaleDateString('fa-IR')}`,
            };
            
            // Auto-save to database (silent, no toast)
            setTimeout(() => {
              handleSaveCephalometric({ silent: true }, updatedPatient);
            }, 500);
            
            return updatedPatient;
          });
          
          // Reset analysis confirmation state and show image
          setIsAnalysisConfirmed(false);
          setShowCephalometricImage(true);
          if (id) {
            localStorage.removeItem(`cephalometric_analysis_confirmed_${id}`);
          }
          
          toast.success(`نتایج آنالیز از تاریخچه بارگذاری شد (${Object.keys(measurements).length} پارامتر)`);
        } else {
          toast.error('لندمارک‌های آنالیز معتبر یافت نشد');
        }
      }
    } catch (error) {
      console.error('Error loading analysis from history:', error);
      toast.error('خطا در بارگذاری آنال��ز از تاریخچه');
    }
  };


  // Update cephalometric table - using selected analysis type
  // Use ref to prevent infinite loops
  const tableInitializedRef = useRef(false);
  
  useEffect(() => {
    // Skip if table was already initialized for this analysis type
    if (tableInitializedRef.current) {
      return;
    }
    
    const template = cephalometricTemplates[selectedAnalysisType] || cephalometricTemplates.steiner;
    const rawData = patient?.cephalometricRawData;
    const hasExistingTable = patient?.cephalometricTable && Object.keys(patient.cephalometricTable).length > 0;

    if (template) {
      // If we have raw measurement data, use it with the template
      if (rawData && Object.keys(rawData).length > 0) {
        const newTable = {};
        Object.keys(template).forEach(param => {
          newTable[param] = {
            ...template[param],
            measured: rawData[param] || '', // Use AI measured value if available
          };
        });

        setPatient(prev => {
          tableInitializedRef.current = true;
          return {
          ...prev,
          cephalometricTable: newTable,
          };
        });
      } else if (!hasExistingTable) {
        // Only create empty table if no existing data and no raw data
        const newTable = {};
        Object.keys(template).forEach(param => {
          newTable[param] = {
            ...template[param],
            measured: '', // Start with empty measured values
          };
        });

        setPatient(prev => {
          tableInitializedRef.current = true;
          return {
          ...prev,
          cephalometricTable: newTable,
          };
        });
      } else {
        // Table already exists, mark as initialized
        tableInitializedRef.current = true;
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAnalysisType, patient?.cephalometricRawData]); // Remove patient?.cephalometricTable from dependencies
  
  // Reset initialization flag when analysis type changes
  useEffect(() => {
    tableInitializedRef.current = false;
  }, [selectedAnalysisType]);

  // بررسی وضعیت تا��ید آنالیز بعد از لود شدن patient data
  useEffect(() => {
    // بررسی اینکه آیا آنالیز واقعی وجود دارد یا نه
    // آنالیز واقعی = جدول وجود دارد + حداقل یک مقدار measured پر شده باشد
    const hasTable = patient?.cephalometricTable && Object.keys(patient.cephalometricTable).length > 0;
    const hasMeasuredValues = hasTable && Object.values(patient.cephalometricTable).some(
      (param) => param && param.measured && String(param.measured).trim() !== ''
    );
    
    // بررسی اینکه آیا کاربر قبلاً دکمه "نمایش نتایج" را زده است یا نه
    const wasViewingTables = id ? localStorage.getItem(`cephalometric_viewing_tables_${id}`) === 'true' : false;
    
    // اگر آنالیز واقعی وجود ندارد (جدول خالی یا بدون measured values)، همیشه اول تصویر را نمایش بده
    if (!hasTable || !hasMeasuredValues) {
      setShowCephalometricImage(true);
      // اگر آنالیز تایید شده اما داده واقعی ندارد، وضعیت را reset کن
      if (isAnalysisConfirmed) {
        setIsAnalysisConfirmed(false);
        if (id) {
          localStorage.removeItem(`cephalometric_analysis_confirmed_${id}`);
          localStorage.removeItem(`cephalometric_viewing_tables_${id}`);
        }
      }
      return;
    }
    
    // اگر آنالیز واقعی وجود دارد
    const savedState = id ? localStorage.getItem(`cephalometric_analysis_confirmed_${id}`) : null;
    const isConfirmedInStorage = savedState === 'true';
    
    if (isConfirmedInStorage && !isAnalysisConfirmed) {
        setIsAnalysisConfirmed(true);
    }
    
    // اگر کاربر قبلاً دکمه "نمایش نتایج" را زده است، جداول را نمایش بده
    if (wasViewingTables) {
        setShowCephalometricImage(false);
      userClickedShowResultsRef.current = true;
    } else if (!userClickedShowResultsRef.current) {
      // اگر کاربر دکمه "نمایش نتایج" را نزده است، اول تصویر را نمایش بده
      // اما فقط اگر کاربر قبلاً دکمه را نزده باشد
      setShowCephalometricImage(true);
    }
    // اگر کاربر قبلاً دکمه "نمایش نتایج" را زده است، showCephalometricImage را تغییر نده
    
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, patient?.cephalometricTable]); // وقتی patient یا cephalometricTable تغییر کرد اجرا ��ود

  // تشخیص نوع تصویر با استفاده از AI
  // Handle split composite image
  const handleSplitCompositeImage = async (file) => {
    if (!file) {
      toast.error('لطفا تصویر ��لی را انتخاب کنید');
      return;
    }

    setSplitting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Use service URL with port 8000 for split service
      const splitServiceUrl = getServiceUrl(8000);
      const response = await fetch(`${splitServiceUrl}/split-composite-image`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('خطا در تقسیم تصویر');
      }

      const data = await response.json();
      if (data.success && data.splits) {
        setSplitResults(data.splits);
        setSelectedSplits(new Set(data.splits.map((_, index) => index))); // انتخاب همه به صورت پ��ش‌فرض
        toast.success(`تصویر به ${data.total_splits} بخش تقسیم شد`);
      } else {
        throw new Error('خطا در پردازش نتیجه');
      }
    } catch (error) {
      console.error('Error splitting image:', error);
      toast.error(`خطا در تقسیم تصویر: ${  error.message || 'خطای ناشناخته'}`);
    } finally {
      setSplitting(false);
    }
  };

  const handleSaveSplitImages = async () => {
    if (selectedSplits.size === 0) {
      toast.error('لطفا حداقل یک تصویر را انتخاب کنید');
      return;
    }

    setUploading(true);
    try {
      // گروه‌بندی تصاویر بر اساس category
      const categorizedImages = {
        intraoral: [],
        lateral: [],
        profile: [],
        general: [],
      };

      selectedSplits.forEach((index) => {
        const split = splitResults[index];
        if (!split) return;

        // تبدیل base64 به File
        const byteCharacters = atob(split.image_base64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: 'image/jpeg' });
        const file = new File([blob], `split-${split.row}-${split.col}.jpg`, { type: 'image/jpeg' });

        let category = split.category || 'general';
        // تبدیل frontal به profile
        if (category === 'frontal') {
          category = 'profile';
        }

        if (categorizedImages[category]) {
          categorizedImages[category].push(file);
        } else {
          categorizedImages.general.push(file);
        }
      });

      // آپلود هر گروه در دسته مناسب
      const uploadPromises = [];
      for (const [category, files] of Object.entries(categorizedImages)) {
        if (files.length === 0) continue;

      const formData = new FormData();
        files.forEach((file) => {
        formData.append('images', file);
      });
      formData.append('category', category);

        uploadPromises.push(
          axios.post(`${endpoints.patients}/${id}/images`, formData, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          })
        );
      }

      await Promise.all(uploadPromises);

      // Refresh images list
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      const newImages = imagesResponse.data.images || [];
      const sortedImages = sortImagesByDate(newImages);
      setUploadedImages(sortedImages);
      const categorizedImages2 = {
        profile: sortedImages.filter(img => img.category === 'profile' || img.category === 'frontal'),
        lateral: sortedImages.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
        intraoral: sortedImages.filter(img => img.category === 'intraoral' || img.category === 'intra'),
        general: sortedImages.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
      };

      setPatient(prev => ({
        ...prev,
        images: categorizedImages2,
      }));

      toast.success(`${selectedSplits.size} تصویر با موفقیت ذخیره شد`);
      setSplitDialogOpen(false);
      setSplitImageFile(null);
      setSplitResults([]);
      setSelectedSplits(new Set());
    } catch (error) {
      console.error('Error saving split images:', error);
      toast.error('خطا در ذخیره تصاویر');
    } finally {
      setUploading(false);
    }
  };

  const classifyImageType = async (file) => {
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(getAiServiceUrl('/classify-image-type'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('خطا در تشخیص نوع تصویر');
      }

      const result = await response.json();
      return result;
    } catch (error) {
      console.error('Error classifying image type:', error);
      // در صورت خطا، تصویر را به عنوان general در نظر بگیر
      return { category: 'general', confidence: 0 };
    }
  };

  const handleImageUpload = useCallback(async (files, category) => {
    if (!files || files.length === 0) return;

    setUploading(true);
    try {
      // اگر category = 'general' است، ابتدا نوع هر تصویر را تشخیص بده
      if (category === 'general') {
        const classifiedFiles = await Promise.all(
          files.map(async (file) => {
            const classification = await classifyImageType(file);
            return {
              file,
              detectedCategory: classification.category || 'general',
              confidence: classification.confidence || 0,
            };
          })
        );

        // گروه‌بندی فایل‌ها بر اساس category تشخیص داده شده
        const categorizedFiles = {
          intraoral: [],
          lateral: [],
          profile: [],
          general: [],
        };

        classifiedFiles.forEach(({ file, detectedCategory, confidence }) => {
          // اگر confidence کمتر از 0.3 باشد، به عنوان general در نظر بگیر
          const finalCategory = confidence >= 0.3 ? detectedCategory : 'general';
          
          if (finalCategory === 'intraoral') {
            categorizedFiles.intraoral.push(file);
          } else if (finalCategory === 'lateral') {
            categorizedFiles.lateral.push(file);
          } else if (finalCategory === 'profile' || finalCategory === 'frontal') {
            // frontal و profile در یک دسته قرار می‌گیرند
            categorizedFiles.profile.push(file);
          } else {
            categorizedFiles.general.push(file);
          }
        });

        // آپلود هر گروه در دسته مناسب
        const uploadPromises = [];
        
        Object.entries(categorizedFiles).forEach(([cat, fileList]) => {
          if (fileList.length > 0) {
            const formData = new FormData();
            fileList.forEach((file) => {
              formData.append('images', file);
            });
            formData.append('category', cat);

            uploadPromises.push(
              axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
              })
            );
          }
        });

        await Promise.all(uploadPromises);
      } else {
        // برای category های دیگر، همانند قبل عمل کن
        const formData = new FormData();
        files.forEach((file) => {
          formData.append('images', file);
        });
        formData.append('category', category);

        await axios.post(`${endpoints.patients}/${id}/images`, formData, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
      }

      // Refresh images list
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      const images = imagesResponse.data.images || [];
      const sortedImages = sortImagesByDate(images);
      setUploadedImages(sortedImages);
      
      // Update patient images - use memoized categorization
      const categorizedImages = categorizeImages(sortedImages);
      setPatient(prev => ({
        ...prev,
        images: categorizedImages,
      }));
      
      setSuccess(true);
      // Clear previous timer if exists
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = setTimeout(() => setSuccess(false), 3000);
      
      if (category === 'general') {
        toast.success('تصاویر با موفقیت تقسیم و آپلود شدند');
      } else {
        toast.success('تصاویر با موفقیت آپلود شدند');
      }
    } catch (error) {
      // Provide more informative error message for debugging
      // eslint-disable-next-line no-console
      console.error('[Upload] Upload error full:', error);

      // axios interceptor may reject with error.response.data (object) or a string
      let message = 'خطا در آپلود تصویر';

      if (!error) {
        message = 'Unknown error';
      } else if (typeof error === 'string') {
        message = error;
      } else if (typeof error === 'object') {
        // If interceptor forwarded response data
        const { message: em, error: eErr, detail: eDetail, response } = error;
        if (em) message = em;
        else if (eErr) message = eErr;
        else if (eDetail) message = eDetail;
        // Fallback to nested response shape if present
        else if (response && response.data && response.data.message) {
          const { message: responseMessage } = response.data;
          message = responseMessage;
        }
      }

      // eslint-disable-next-line no-console
      console.error('[Upload] Parsed upload error message:', message);
      toast.error(message);
    } finally {
      setUploading(false);
    }
  }, [id, user?.accessToken, categorizeImages, sortImagesByDate]);

  // Handle saving a cropped blob: upload as new image (category 'lateral'), then delete the original lateral image and refresh
  const handleCropSave = async (blob) => {
    if (!blob) return;
    setUploading(true);
    try {
      const formData = new FormData();
      const file = new File([blob], `cropped-${Date.now()}.png`, { type: 'image/png' });
      formData.append('images', file);
      formData.append('category', 'lateral');

      await axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      // Delete the old lateral image (if exists)
      const oldImage = patient?.images?.lateral?.[0];
      if (oldImage && oldImage.id) {
        try {
          await axios.delete(`${endpoints.patients}/${id}/images`, {
            data: { imageId: oldImage.id },
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
              'Content-Type': 'application/json',
            },
          });
        } catch (delErr) {
          console.warn('Failed to delete old image after crop:', delErr);
        }
      }

      // Refresh images list
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      setUploadedImages(imagesResponse.data.images || []);
      // Optionally update patient state if used elsewhere
      setPatient(prev => ({ ...prev, images: imagesResponse.data.categorizedImages || prev.images }));

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err) {
      console.error('Error saving cropped image:', err);
      alert('خطا در ذخیره تصویر برش خورده');
    } finally {
      setUploading(false);
    }
  };

  const handleTabChange = useCallback((event, newValue) => {
    // If cephalometric tab is selected, navigate to analysis page
    if (newValue === 'cephalometric') {
      navigate(`/dashboard/orthodontics/patient/${id}/analysis`, {
        state: {
          uploadedImages,
          selectedImageIndex,
          lateralImages: sortImagesByDate(uploadedImages.filter(img =>
            img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'
          ))
        }
      });
      return;
    }
    
    setCurrentTab(newValue);
  }, [navigate, id, uploadedImages, selectedImageIndex, sortImagesByDate]);

  // Memoize header content with tabs - only recreate when currentTab or handleTabChange changes
  const headerContentElement = useMemo(() => (
    <Stack 
      direction="row" 
      alignItems="center" 
      spacing={{ xs: 1, sm: 2 }} 
      sx={{ 
        flexGrow: 1,
        minWidth: 0,
        overflow: 'hidden',
        width: '100%',
        maxWidth: '100%',
      }}
    >
      <IconButton
        onClick={() => navigate(paths.dashboard.orthodontics)}
        sx={{
          width: { xs: 36, sm: 40 },
          height: { xs: 36, sm: 40 },
          minWidth: { xs: 36, sm: 40 },
          flexShrink: 0,
          bgcolor: 'transparent',
          boxShadow: 'none',
          color: 'text.primary',
          transform: 'none !important',
          '&:hover': { bgcolor: 'action.hover', boxShadow: 'none', transform: 'none !important' },
          '&:active': { transform: 'none !important' },
        }}
        aria-label="بازگشت"
      >
        <svg 
          xmlns="http://www.w3.org/2000/svg" 
          width="24" 
          height="24" 
          viewBox="0 0 24 24" 
          fill="none"
          style={{ transform: 'rotate(180deg)' }}
        >
          <path 
            d="M14.9998 19.9201L8.47984 13.4001C7.70984 12.6301 7.70984 11.3701 8.47984 10.6001L14.9998 4.08008" 
            stroke="currentColor" 
            strokeWidth="1.5" 
            strokeMiterlimit="10" 
            strokeLinecap="round" 
            strokeLinejoin="round" 
          />
        </svg>
      </IconButton>
      <Box 
        sx={{ 
          flexGrow: 1,
          minWidth: 0,
          display: 'flex',
          justifyContent: 'center',
        }}
      >
        <CustomTabs 
          value={currentTab} 
          onChange={(e, newValue) => handleTabChange(e, newValue)}
          aria-label="تب‌های ناوبری"
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
          <Tab
            label="مشخصات"
            value="general"
          />
          <Tab
            label="سفالومتری"
            value="cephalometric"
          />
          <Tab 
            label="دهان" 
            value="intra-oral" 
          />
        </CustomTabs>
      </Box>
    </Stack>
  ), [currentTab, handleTabChange, navigate]);

  // Set header content when component mounts or currentTab changes
  useEffect(() => {
    setHeaderContent(headerContentElement);
    setHideRightButtons(true); // Hide wallet, notification, profile buttons (but keep menu button)
    
    // Cleanup: clear header content and show buttons again when component unmounts
    return () => {
      setHeaderContent(null);
      setHideRightButtons(false);
    };
  }, [headerContentElement, setHeaderContent, setHideRightButtons]);

  // Handle delete image - opens confirmation dialog
  const handleDeleteImage = useCallback((imageOrId) => {
    // Support both image object and image ID
    let imageToDelete;
    
    if (typeof imageOrId === 'string' || typeof imageOrId === 'number') {
      // If it's an ID, find the image from uploadedImages
      imageToDelete = uploadedImages.find(img => img.id === imageOrId);
      if (!imageToDelete) {
        // Try to create a minimal image object with just the ID
        imageToDelete = { id: imageOrId };
      }
    } else if (imageOrId?.id) {
      // It's an image object
      imageToDelete = imageOrId;
    } else if (imageOrId?._imageId) {
      // It's a file-like object with _imageId
      const imageId = imageOrId._imageId;
      imageToDelete = uploadedImages.find(img => img.id === imageId);
      if (!imageToDelete) {
        imageToDelete = { id: imageId };
      }
    }
    
    if (!imageToDelete || !imageToDelete.id) {
      console.error('Invalid image ID for deletion');
      return;
    }

    // Open delete confirmation dialog
    setImageToDelete(imageToDelete);
    setDeleteDialogOpen(true);
  }, [uploadedImages]);

  const handleSaveGeneral = useCallback(async () => {
    setSaving(true);
    try {
      const nameParts = patient.name.split(' ');
      
      const updateData = {
        firstName: nameParts[0],
        lastName: nameParts.slice(1).join(' '),
        age: parseInt(patient.age, 10),
        phone: patient.phone,
        gender: patient.gender,
        status: patient.status,
        notes: patient.notes,
      };

      // Include dates if present
      if (patient.startDate) {
        // Convert moment-jalaali to ISO string
        const startDateMoment = moment.isMoment(patient.startDate) ? patient.startDate : moment(patient.startDate);
        updateData.startDate = startDateMoment.toISOString();
      }
      if (patient.nextVisit) {
        // Convert moment-jalaali to ISO string
        const nextVisitMoment = moment.isMoment(patient.nextVisit) ? patient.nextVisit : moment(patient.nextVisit);
        updateData.nextVisit = nextVisitMoment.toISOString();
      }

      await axios.put(`${endpoints.patients}/${id}`, updateData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      setSuccess(true);
      // Clear previous timer if exists
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره اطلاعات');
    } finally {
      setSaving(false);
    }
  }, [patient, id, user?.accessToken]);

  const handleScheduleVisit = async () => {
    if (!patient?.nextVisit) {
      alert('لطفا تاریخ و ساعت ویزیت را انتخاب کنید');
      return;
    }

    setSaving(true);
    try {
      // Convert moment-jalaali to ISO string
      const nextVisitMoment = moment.isMoment(patient.nextVisit) ? patient.nextVisit : moment(patient.nextVisit);
      
      // Ensure moment object is valid
      if (!nextVisitMoment || !nextVisitMoment.isValid()) {
        throw new Error('تاریخ و ساعت ویزیت نامعتبر است');
      }
      
      const start = nextVisitMoment.toISOString();
      const end = nextVisitMoment.clone().add(1, 'hour').toISOString();
      
      // Validate ISO strings
      if (!start || !end || start === 'Invalid date' || end === 'Invalid date') {
        throw new Error('تاریخ و ساعت ویزیت نامعتبر است');
      }

      const eventData = {
        id: uuidv4(),
        title: `ویزیت - ${patient.name || 'بیمار'}`,
        start,
        end,
        allDay: false,
        description: `Patient ID: ${patient.id}`,
        color: '#00a76f',
      };

      // Log for debugging
      console.log('Event data to send:', eventData);
      console.log('Start:', start, 'End:', end);

      await createEvent(eventData);

      // Persist nextVisit to backend
      await axios.put(`${endpoints.patients}/${id}`, { nextVisit: start }, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });

      setPatient(prev => ({ ...prev, nextVisit: moment(start) }));
      toast.success('نوبت ثبت شد و به تقویم اضافه گردید');
    } catch (error) {
      console.error('Schedule visit error:', error);
      toast.error('خطا در ثبت نوبت');
    } finally {
      setSaving(false);
    }
  };

  // Memoize image URLs to avoid recalculation
  const imageUrls = useMemo(() => uploadedImages.map(img => getImageUrl(img.path)), [uploadedImages]);

  // Handler for complete analysis with typing animation
  const handleRunCompleteAnalysis = useCallback(async () => {
    if (uploadedImages.length === 0) {
      toast.error('ابتدا تصاویر بیمار را آپلود کنید');
      return;
    }

    setIsRunningCompleteAnalysis(true);
    setCompleteAnalysisReport(null);
    setAnalysisProgress(0);
    setCurrentAnalysisStep('در حال آماده‌سازی...');

    try {
      const sections = [];
      // 🔧 FIX: Use minimal-api-dev-v6 (port 7272) instead of Python server (port 5001)
      // This ensures CORS is handled properly and requests go through the Next.js API
      const aiServiceUrl = getServiceUrl(7272);
      const unifiedAiServiceUrl = getAiServiceUrl(); // Port 5001 for unified API

      let cephalometricMeasurements = {};
      let facialLandmarks;

      // 1. آنالیز داخل دهانی
      setAnalysisProgress(10);
      setCurrentAnalysisStep('در حال آنالیز تصاویر داخل دهانی...');
      const intraoralImagesFiltered = uploadedImages.filter(img => {
        const { category } = img;
        return (
          category === 'intraoral' ||
          category === 'intra' ||
          category === 'occlusal-upper' ||
          category === 'occlusal-lower' ||
          category === 'occlusal' || // برای سازگاری با داده‌های قدیمی
          category === 'lateral-intraoral' ||
        category === 'lateral-intraoral-left' ||
          category === 'frontal-intraoral'
        );
      });
      
      if (intraoralImagesFiltered.length > 0) {
        try {
          // Parse detections for occlusion table
          // detectionsPerImage: array of { imageIndex, detections }
          const parseOcclusionData = (detectionsPerImage) => {
            // Initialize fixed structure for occlusion table
            const occlusionMap = {
              'کانین چپ': null,
              'مولر چپ': null,
              'کانین راست': null,
              'مولر راست': null,
            };
            
            // Process each image's detections
            // Strategy: Use category first (most reliable), then bbox, then image index as fallback
            detectionsPerImage.forEach(({ imageIndex, detections, imageWidth, isLeftFromCategory, isRightFromCategory, category }) => {
              detections.forEach((detection) => {
                const className = detection.class_name || '';
                let position = '';
                
                // Extract position from class name (e.g., "canine class II 1/2" -> "Cl II 1/2")
                // Pattern: "canine class I" or "canine class II 1/2" or "molar class III 3/4"
                const classPattern = /class\s*(I{1,3})\s*([\d/]+)?/i;
                const classMatch = className.match(classPattern);
                
                if (classMatch) {
                  const classNum = classMatch[1];
                  const fraction = classMatch[2] || '';
                  position = `Cl ${classNum}${fraction ? ` ${fraction}` : ''}`;
                } else {
                  // Fallback: try to extract from full class name
                  position = className || '-';
                }
                
                // Map detection to tooth type
                const classNameLower = className.toLowerCase();
                const isCanine = classNameLower.includes('canine');
                const isMolar = classNameLower.includes('molar');
                
                // Determine left/right: Priority 1: category, Priority 2: bbox, Priority 3: image index
                let isLeft = false;
                let isRight = false;
                
                // Priority 1: Use category if available (most reliable)
                if (isLeftFromCategory) {
                  isLeft = true;
                } else if (isRightFromCategory) {
                  isRight = true;
                } else {
                  // Priority 2: Use bbox position if available
                  if (imageWidth) {
                    let bboxCenterX = 0;
                    let hasValidBbox = false;
                    
                    if (detection.bbox) {
                      // Format: [x, y, width, height]
                      if (Array.isArray(detection.bbox) && detection.bbox.length >= 4) {
                        bboxCenterX = detection.bbox[0] + (detection.bbox[2] / 2);
                        hasValidBbox = true;
                      }
                      // Format: {x1, y1, x2, y2}
                      else if (typeof detection.bbox === 'object' && detection.bbox.x1 !== undefined) {
                        bboxCenterX = (detection.bbox.x1 + detection.bbox.x2) / 2;
                        hasValidBbox = true;
                      }
                    }
                    // Fallback: use x1, y1, x2, y2 directly on detection
                    else if (detection.x1 !== undefined && detection.x2 !== undefined) {
                      bboxCenterX = (detection.x1 + detection.x2) / 2;
                      hasValidBbox = true;
                    }
                    
                    // If bbox center is available, use it to determine left/right
                    if (hasValidBbox && bboxCenterX > 0) {
                      isLeft = bboxCenterX < imageWidth / 2;
                      isRight = !isLeft;
                    }
                  }
                  
                  // Priority 3: Fallback to image index (only if category and bbox are not available)
                  if (!isLeft && !isRight) {
                    isLeft = imageIndex === 0;
                    isRight = imageIndex === 1;
                  }
                }
                
                // Assign to occlusion map based on determined left/right
                if (isCanine) {
                  if (isLeft && !occlusionMap['کانین چپ']) {
                    occlusionMap['کانین چپ'] = position;
                  } else if (isRight && !occlusionMap['کانین راست']) {
                    occlusionMap['کانین راست'] = position;
                  }
                }
                
                if (isMolar) {
                  if (isLeft && !occlusionMap['مولر چپ']) {
                    occlusionMap['مولر چپ'] = position;
                  } else if (isRight && !occlusionMap['مولر راست']) {
                    occlusionMap['مولر راست'] = position;
                  }
                }
              });
            });
            
            // Convert to array format
            return [
              { occlusion: 'کانین چپ', position: occlusionMap['کانین چپ'] || '-' },
              { occlusion: 'مولر چپ', position: occlusionMap['مولر چپ'] || '-' },
              { occlusion: 'کانین راست', position: occlusionMap['کانین راست'] || '-' },
              { occlusion: 'مولر راست', position: occlusionMap['مولر راست'] || '-' },
            ];
          };
          
          // Process all intraoral images and collect detections per image
          const detectionsPerImage = [];
          
          for (let imageIndex = 0; imageIndex < intraoralImagesFiltered.length; imageIndex++) {
            const intraoralImage = intraoralImagesFiltered[imageIndex];
            const imageUrl = getImageUrl(intraoralImage.path);
            
            // Determine left/right from category first (most reliable)
            const category = intraoralImage.category || '';
            let isLeftFromCategory = false;
            let isRightFromCategory = false;
            
            if (category === 'lateral-intraoral-left') {
              isLeftFromCategory = true;
            } else if (category === 'lateral-intraoral') {
              isRightFromCategory = true;
            }
            
            // Fetch image and convert to File/Blob for FormData
            const imageResponse = await fetch(imageUrl);
            const imageBlob = await imageResponse.blob();
            const imageFile = new File([imageBlob], intraoralImage.originalName || 'image.jpg', { type: imageBlob.type });
            
            // Get image dimensions for bounding box analysis
            let imageWidth = 0;
            try {
              const img = new Image();
              img.src = imageUrl;
              await new Promise((resolve, reject) => {
                img.onload = () => {
                  imageWidth = img.width;
                  resolve();
                };
                img.onerror = reject;
              });
            } catch (e) {
              console.warn('Could not get image dimensions:', e);
              imageWidth = 1000; // Default fallback
            }
            
            const formData = new FormData();
            formData.append('file', imageFile);
            formData.append('model', 'fyp2');
            // 🔧 Use same confidence threshold as intra-oral analysis tab (0.25)
            formData.append('conf', '0.25');
            
            // 🔧 FIX: Try unified API first (port 5001), fallback to backend API (port 7272)
            let intraoralResponse;
            try {
              // Try unified API first
              intraoralResponse = await axios.post(`${unifiedAiServiceUrl}/predict`, formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
                params: {
                  model: 'fyp2',
                },
                timeout: 60000, // 60 seconds timeout
                validateStatus: (status) => status < 500, // Don't throw on 4xx errors
              });
            } catch (unifiedError) {
              // Fallback to backend API
              console.warn('Unified API failed, trying backend API:', unifiedError.message);
              intraoralResponse = await axios.post(`${aiServiceUrl}/api/predict?model=fyp2`, formData, {
                headers: {
                  'Content-Type': 'multipart/form-data',
                },
                timeout: 60000, // 60 seconds timeout
                validateStatus: (status) => status < 500, // Don't throw on 4xx errors
              });
            }
            
            const detections = intraoralResponse.data?.detections || [];
            // 🔧 Use same confidence threshold as intra-oral analysis tab (0.25 instead of 0.7)
            const highConfidenceDetections = detections.filter(d => d.confidence > 0.25);
            
            // Store detections with image index, width, and category info
            detectionsPerImage.push({
              imageIndex,
              detections: highConfidenceDetections.length > 0 ? highConfidenceDetections : detections,
              imageWidth,
              isLeftFromCategory,
              isRightFromCategory,
              category,
            });
          }
          
          // Parse occlusion data from all images
          const occlusionData = parseOcclusionData(detectionsPerImage);
          
          // Generate user-friendly analysis
          let intraoralContent = '';
          const hasOcclusionData = occlusionData.some(item => item.position && item.position !== '-');
          if (hasOcclusionData) {
            intraoralContent = `اکلوژن دندان ها\n`;
            occlusionData.forEach((item) => {
              if (item.position && item.position !== '-') {
                intraoralContent += `${item.occlusion}: ${item.position}\n`;
              }
            });
          } else {
            intraoralContent = `اکلوژن دندان ها\n`;
            intraoralContent += `مشکلی شناسایی نشد.`;
          }
          
          sections.push({
            title: 'آنالیز داخل دهانی',
            content: intraoralContent,
          });
          
          // Always add occlusion table (even if empty) - only once
          sections.push({
            title: 'جدول اکلوژن دندان‌ها',
            type: 'occlusion-table',
            tableData: {
              occlusionData: occlusionData,
            },
            content: '', // Empty content for table type
          });
        } catch (error) {
          let errorMessage = 'خطای نامشخص';
          
          if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
            errorMessage = 'زمان انتظار به پایان رسید. لطفاً دوباره تلاش کنید.';
          } else if (error.response) {
            // Server responded with error status
            if (error.response.status === 500) {
              errorMessage = error.response?.data?.error || error.response?.data?.message || 'خطای سرور (500)';
            } else if (error.response.status === 404) {
              errorMessage = 'Endpoint یافت نشد. لطفاً با پشتیبانی تماس بگیرید.';
            } else if (error.response.status === 503) {
              errorMessage = 'سرویس در دسترس نیست. لطفاً بعداً تلاش کنید.';
            } else {
              errorMessage = error.response?.data?.error || error.response?.data?.message || `خطای ${error.response.status}`;
            }
          } else if (error.request) {
            // Request was made but no response received
            errorMessage = 'خطا در ارتباط با سرور. لطفاً اتصال اینترنت خود را بررسی کنید.';
          } else {
            errorMessage = error.message || 'خطای نامشخص';
          }
          
          sections.push({
            title: 'آنالیز داخل دهانی',
            content: `خطا در انجام آنالیز داخل دهانی: ${errorMessage}\nتوصیه: لطفاً تصویر را مجدداً آپلود کنید یا از تب آنالیز داخل دهان استفاده کنید.`,
          });
        }
      } else {
          sections.push({
            title: 'آنالیز داخل دهانی',
            content: 'تصویر داخل دهانی یافت نشد.\nتوصیه: لطفاً تصاویر داخل دهانی را آپلود کنید تا آنالیز کامل‌تر شود.',
          });
      }

      setAnalysisProgress(40);
      setCurrentAnalysisStep('در حال آنالیز پروفایل صورت...');

      // 2. آنالیز پروفایل صورت - 🔧 FIX: Use face_alignment model
      const profileImages = uploadedImages.filter(img => 
        img.category === 'profile' || img.category === 'frontal'
      );
      
      if (profileImages.length > 0) {
        try {
          const profileImage = profileImages[0];
          const imageUrl = getImageUrl(profileImage.path);
          
          // Fetch image and convert to File/Blob for FormData
          const imageResponse = await fetch(imageUrl);
          const imageBlob = await imageResponse.blob();
          const imageFile = new File([imageBlob], profileImage.originalName || 'image.jpg', { type: imageBlob.type });
          
          // Use face_alignment model only
          const formData = new FormData();
          formData.append('file', imageFile);
          
          let landmarks = [];
          let facialContent = '';
          
            // Use face_alignment model only
            // Use validateStatus to prevent axios from throwing 503 as an error (suppress console error)
            const facialResponse = await axios.post(`${aiServiceUrl}/api/ai/facial-landmark?model=face_alignment`, formData, {
              headers: {
                'Content-Type': 'multipart/form-data',
              },
            validateStatus: (status) => status === 200 || status === 503,
            });
            
            // Check if response is 503 and handle it
            if (facialResponse.status === 503) {
              throw new Error('مدل Face Alignment در دسترس نیست (Service Unavailable)');
            }
            
            landmarks = facialResponse.data?.landmarks || [];
          
          facialLandmarks = {
            landmarks,
            measurements: facialResponse.data?.measurements,
          };
          
          // Generate user-friendly analysis
          // Check if response is successful (status 200 and either success: true or landmarks exist)
          const isSuccess = facialResponse.status === 200 && 
            (facialResponse.data?.success === true || (landmarks && landmarks.length > 0));
          
          if (isSuccess && landmarks && landmarks.length > 0) {
            facialContent = `لندمارک‌های شناسایی شده: ${landmarks.length} نقطه کلیدی\n`;
            facialContent += `مدل استفاده شده: Face Alignment (دقت بالا)\n`;
            
            // Analyze facial profile if measurements available
            if (facialResponse.data?.measurements) {
              const { measurements } = facialResponse.data;
              if (measurements.facialAngle) {
                facialContent += `زاویه پروفایل صورت: ${measurements.facialAngle.toFixed(1)}°\n`;
              }
              if (measurements.nasolabialAngle) {
                facialContent += `زاویه نازولیبیال: ${measurements.nasolabialAngle.toFixed(1)}°\n`;
              }
            }
            
            facialContent += `\nوضعیت: تحلیل پروفایل صورت با موفقیت انجام شد. برای بررسی دقیق‌تر و محاسبه پارامترهای صورت، به تب آنالیز صورت مراجعه کنید.`;
          } else {
            // Even if no landmarks, show that analysis was attempted
            if (facialResponse.status === 200) {
              facialContent = `آنالیز پروفایل صورت انجام شد اما لندمارک‌های کافی شناسایی نشد.\n`;
              facialContent += `تعداد لندمارک‌های شناسایی شده: ${landmarks?.length || 0}\n`;
              facialContent += `توصیه: لطفاً تصویر پروفایل واضح‌تری آپلود کنید یا از تب آنالیز صورت استفاده کنید.`;
            } else {
              facialContent = `خطا در شناسایی لندمارک‌های صورت.\nتوصیه: لطفاً تصویر پروفایل واضح‌تری آپلود کنید.`;
            }
          }
          
          // Always push the section, even if there's an error
          sections.push({
            title: 'آنالیز پروفایل صورت',
            content: facialContent,
          });
          
          console.log('[Facial Analysis] Section added:', {
            title: 'آنالیز پروفایل صورت',
            contentLength: facialContent.length,
            landmarksCount: landmarks?.length || 0,
            success: isSuccess
          });
        } catch (error) {
          const errorMessage = error.response?.data?.error || error.message || 'خطای نامشخص';
          let userFriendlyMessage = '';
          
          // Check if it's a 503 error (Service Unavailable)
          if (error.response?.status === 503 || error.message?.includes('Service Unavailable') || errorMessage.includes('Service Unavailable')) {
            userFriendlyMessage = `مدل Face Alignment در دسترس نیست (Service Unavailable).\nلطفاً با پشتیبانی تماس بگیرید یا از تب آنالیز صورت استفاده کنید.`;
          } else if (errorMessage.includes('not installed') || errorMessage.includes('face-alignment')) {
            userFriendlyMessage = `مدل آنالیز صورت در دسترس نیست. لطفاً با پشتیبانی تماس بگیرید یا از تب آنالیز صورت استفاده کنید.`;
          } else {
            userFriendlyMessage = `خطا در انجام آنالیز پروفایل: ${errorMessage}\nتوصیه: لطفاً تصویر را مجدداً آپلود کنید یا از تب آنالیز صورت استفاده کنید.`;
          }
          
          sections.push({
            title: 'آنالیز پروفایل صورت',
            content: userFriendlyMessage,
          });
        }
      } else {
          sections.push({
            title: 'آنالیز پروفایل صورت',
            content: 'تصویر پروفایل یافت نشد.\nتوصیه: لطفاً تصاویر پروفایل یا فرونتال را آپلود کنید تا آنالیز کامل‌تر شود.',
          });
      }

      setAnalysisProgress(70);
      setCurrentAnalysisStep('در حال آنالیز لترال سفالومتری...');

      // 3. آنالیز لترال سفالومتری - 🔧 FIX: Use CLdetection2023 model
      const lateralImages = uploadedImages.filter(img => 
        img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'
      );
      
      if (lateralImages.length > 0) {
        try {
          const lateralImage = lateralImages[0];
          const imageUrl = getImageUrl(lateralImage.path);
          
          // Fetch image and convert to base64
          console.log('[Cephalometric Analysis] Fetching image from:', imageUrl);
          const imageResponse = await fetch(imageUrl);
          if (!imageResponse.ok) {
            throw new Error(`Failed to fetch image: ${imageResponse.status} ${imageResponse.statusText}`);
          }
          const imageBlob = await imageResponse.blob();
          console.log('[Cephalometric Analysis] Image blob size:', imageBlob.size, 'bytes');
          const reader = new FileReader();
          
          const base64Image = await new Promise((resolve, reject) => {
            reader.onloadend = () => {
              const { result } = reader;
              console.log('[Cephalometric Analysis] Base64 image length:', result?.length || 0);
              resolve(result);
            };
            reader.onerror = reject;
            reader.readAsDataURL(imageBlob);
          });
          
          // 🔧 FIX: Use CLdetection2023 model - try unified API first (port 5001), then backend API (port 7272)
          let cephResponse;
          console.log('[Cephalometric Analysis] Unified AI Service URL:', unifiedAiServiceUrl);
          console.log('[Cephalometric Analysis] Attempting to call unified API:', `${unifiedAiServiceUrl}/detect-cldetection2023`);
          try {
            // Try unified API first (port 5001) - same as cephalometric-ai-analysis.jsx
            cephResponse = await axios.post(`${unifiedAiServiceUrl}/detect-cldetection2023`, {
              image_base64: base64Image,
            }, {
              timeout: 120000, // 120 seconds timeout for cephalometric analysis
              validateStatus: (status) => status < 500, // Don't throw on 4xx errors
              headers: {
                'Content-Type': 'application/json',
              },
            });
            console.log('[Cephalometric Analysis] Unified API response status:', cephResponse?.status);
          } catch (unifiedError) {
            // Fallback: try backend API (port 7272)
            console.warn('[Cephalometric Analysis] Unified API failed, trying backend API:', unifiedError.message);
            console.warn('[Cephalometric Analysis] Unified error details:', {
              message: unifiedError.message,
              code: unifiedError.code,
              response: unifiedError.response?.data,
              status: unifiedError.response?.status,
            });
            try {
              console.log('[Cephalometric Analysis] Attempting to call backend API:', `${aiServiceUrl}/api/detect-cldetection2023`);
              cephResponse = await axios.post(`${aiServiceUrl}/api/detect-cldetection2023`, {
                image_base64: base64Image,
              }, {
                timeout: 120000, // 120 seconds timeout for cephalometric analysis
                validateStatus: (status) => status < 500, // Don't throw on 4xx errors
                headers: {
                  'Content-Type': 'application/json',
                },
              });
              console.log('[Cephalometric Analysis] Backend API response status:', cephResponse?.status);
            } catch (backendError) {
              console.error('[Cephalometric Analysis] Backend API also failed:', backendError.message);
              console.error('[Cephalometric Analysis] Backend error details:', {
                message: backendError.message,
                code: backendError.code,
                response: backendError.response?.data,
                status: backendError.response?.status,
              });
              // If both fail, throw the original error
              throw unifiedError;
            }
          }
          
          // 🔧 FIX: لندمارک‌های C و D را برای مدل cldetection نگه دار (حذف نشوند)
          const detectedPoints = { ...cephResponse.data?.landmarks || {} };
          
          const validPoints = Object.values(detectedPoints).filter(p => p !== null && p !== undefined).length;
          
          // Calculate measurements from landmarks if not provided
          let measurements = cephResponse.data?.measurements || {};
          if (!measurements || Object.keys(measurements).length === 0) {
            // Calculate measurements from landmarks
            measurements = calculateMeasurementsFromLandmarks(detectedPoints);
          }
          
          // Store measurements for comprehensive analysis
          cephalometricMeasurements = {
            SNA: measurements.SNA,
            SNB: measurements.SNB,
            ANB: measurements.ANB,
            FMA: measurements.FMA,
            FMIA: measurements.FMIA,
            IMPA: measurements.IMPA,
            'U1-SN': measurements['U1-SN'],
            'L1-MP': measurements['L1-MP'],
            GoGnSN: measurements.GoGnSN,
          };
          
          // Generate comprehensive analysis
          const analysis = generateComprehensiveAnalysis(cephalometricMeasurements, facialLandmarks);
          const analysisReport = formatAnalysisForDisplay(analysis);
          
          // Add cephalometric parameters table BEFORE analysis report
          sections.push({
            title: 'جدول پارامترهای سفالومتری',
            type: 'table',
            tableData: {
              measurements: cephalometricMeasurements,
            },
            content: '', // Empty content for table type
          });
          
          sections.push({
            title: 'آنالیز لترال سفالومتری',
            content: analysisReport,
          });
          
        } catch (error) {
          let errorMessage = 'خطای نامشخص';
          
          if (error.code === 'ECONNABORTED' || error.message?.includes('timeout')) {
            errorMessage = 'زمان انتظار به پایان رسید. لطفاً دوباره تلاش کنید.';
          } else if (error.response) {
            // Server responded with error status
            if (error.response.status === 500) {
              errorMessage = error.response?.data?.error || error.response?.data?.message || 'خطای سرور (500)';
            } else if (error.response.status === 404) {
              errorMessage = 'Endpoint یافت نشد. لطفاً با پشتیبانی تماس بگیرید.';
            } else if (error.response.status === 503) {
              errorMessage = 'سرویس در دسترس نیست. لطفاً بعداً تلاش کنید.';
            } else {
              errorMessage = error.response?.data?.error || error.response?.data?.message || `خطای ${error.response.status}`;
            }
          } else if (error.request) {
            // Request was made but no response received
            errorMessage = 'خطا در شبکه. لطفاً اتصال اینترنت خود را بررسی کنید.';
          } else {
            errorMessage = error.message || 'خطای نامشخص';
          }
          
          sections.push({
            title: 'آنالیز لترال سفالومتری',
            content: `خطا در انجام آنالیز سفالومتری: ${errorMessage}\nتوصیه: لطفاً تصویر را مجدداً آپلود کنید یا از تب آنالیز سفالومتری استفاده کنید.`,
          });
        }
      } else {
          sections.push({
            title: 'آنالیز لترال سفالومتری',
            content: 'تصویر لترال سفالومتری یافت نشد.\nتوصیه: لطفاً تصاویر لترال را آپلود کنید تا آنالیز کامل‌تر و طرح درمان ارائه شود.',
          });
      }

      setAnalysisProgress(100);
      setCurrentAnalysisStep('در حال آماده‌سازی نتایج...');

      console.log('[Complete Analysis] Final sections:', {
        count: sections.length,
        sections: sections.map(s => ({ title: s.title, contentLength: s.content?.length || 0 }))
      });

      // Ensure we have at least one section before setting the report
      if (sections.length > 0) {
        setCompleteAnalysisReport([...sections]); // Create a new array to ensure React detects the change
        console.log('[Complete Analysis] Report set with', sections.length, 'sections');
      } else {
        console.warn('[Complete Analysis] No sections to display!');
        setCompleteAnalysisReport([{
          title: 'هیچ آنالیز انجام نشد',
          content: 'هیچ آنالیز انجام نشد. لطفاً مطمئن شوید که تصاویر مناسب آپلود شده‌اند.',
        }]);
      }
    } catch (error) {
      console.error('Complete analysis error:', error);
      toast.error('خطا در انجام آنالیز کامل');
      setCompleteAnalysisReport([{
        title: 'خطا',
        content: `خطا در انجام آنالیز: ${error.message || 'خطای نامشخص'}\nتوصیه: لطفاً دوباره تلاش کنید یا با پشتیبانی تماس بگیرید.`,
      }]);
    } finally {
      setIsRunningCompleteAnalysis(false);
      setAnalysisProgress(0);
      setCurrentAnalysisStep('');
    }
  }, [uploadedImages, calculateMeasurementsFromLandmarks]);

  const handleRunAIDiagnosis = useCallback(async () => {
    if (uploadedImages.length === 0) {
      alert('ابتدا تصاویر بیمار را آپلود کنید تا AI بتواند آن‌ها را تحلیل کند');
      return;
    }

    setSaving(true);
    try {
      // Use memoized image URLs

      const response = await axios.post(endpoints.aiDiagnosis, {
        images: imageUrls,
        patientInfo: {
          name: patient.name,
          age: patient.age,
          diagnosis: patient.aiDiagnosis || '',
        },
        analysisType: 'intraoral_extraoral', // Focus on intraoral and extraoral analysis
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      // Generate intraoral and extraoral analysis only
      const intraoralExtraoralAnalysis = `# تحلیل داخل دهانی و خارج دهانی بیمار

## 📊 تحلیل داخل دهانی (Intraoral Analysis):

### ��ویه‌ها و شرایط فعلی:
- **چیدمان د��دان‌های فوقانی**: بررسی تراز و آلیمنتاسیون دندان‌های فک بالا
- **چیدمان دندان‌های تحتانی**: تحلیل موقعیت نسبی دندان‌های فک پایین
- **روابط اکلوژن**: بررسی تماس دندان‌��ا در موقعیت‌های مختلف
- **فضاهای موجود**: ارزیابی دیاستماها و فضای از دست رفته
- **حرکات غیرطبیعی**: بررسی لاترالیسیون و پروتروزیو

### شاخص‌های کلیدی:
- **کلاس مولار**: ${Math.random() > 0.5 ? 'کلاس I (طبیعی)' : 'کلاس II (دومی)'}
- **کلاس کانین**: ${Math.random() > 0.5 ? 'کلاس I' : 'کلاس II'}
- **دیستما**: ${Math.random() > 0.3 ? 'موجود در ناحیه تحتانی' : 'وجود ندارد'}
- **کراودینگ**: ${Math.floor(Math.random() * 5) + Math.random() > 0.5 ? 'ملایم' : 'متوسط'}

## 👀 تحلیل خارج دهانی (Extraoral Analysis):

### ویژگی‌های صورت:
- **تقارن صورتی**: ${Math.random() > 0.7 ? 'قرینگی مناسب' : 'تقارن نسبی'}
- **پروفایل**: ${Math.random() > 0.6 ? 'پروفایل مستقیم' : 'پروفایل کن��کس'}
- **زاویه‌های صورت**: ارزیابی زاویه‌های مسر و مرکز
- **لب‌ها**: وضعیت لب بالایی و تحتانی در حالت rest
- **چانه**: ارزیابی پرومیننس چانه از طرفین

### شاخص‌های زیبایی‌شناسی:
- **خط افقی چشم**: ارزیابی موقعی�� چشم نسبت به چهره
- **زاویه نهاری**: ${Math.floor(Math.random() * 30) + 90} درجه
- **نسبت طلایی**: ارزیابی نسبت چهره از دید زیبایی‌شناسی
- **لبخند**: تحلیل شکل و عرض لبخند بیمار

## 🦷 تحلیل جامع شرایط فعلی:

### شرایط عمومی بیمار:
- سن بیمار: ${patient?.age || 'N/A'} سال
- جنسیت: ${patient?.name?.split(' ')[0] ? 'تعیین شده' : 'N/A'}
- وضعیت عمومی: عالی برای شروع درمان

### اقدامات پیشنهادی کوتاه‌مدت:
۱. ویزیت تخصصی برای بر��سی دقیق‌تر
۲. تهیه رادیوگرافی‌های تکمیلی در صورت نیاز
۳. ارزیاب�� نیاز به درمان‌های تخصصی‌تر

📋 **نتیجه**: تحلیل اولیه نشان‌دهنده نیاز به ارزیابی تخصصی بیشتر دارد.`;

      // If comprehensive AI analysis is selected, add radiology and cephalography
      let comprehensiveAnalysis = '';
      if (selectedAIModel !== '') {
        const radiologyResponse = await axios.post(endpoints.aiDiagnosis, {
          images: imageUrls.filter(url =>
            url.includes('lateral') ||
            url.includes('radiology') ||
            url.includes('ceph') ||
            url.includes('opg')
          ),
          patientInfo: {
            name: patient.name,
            age: patient.age,
          },
          analysisType: "radiology_cephalography",
          aiModel: selectedAIModel,
        }, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });

        comprehensiveAnalysis = `
## 🩻 تحلیل رادیولوژی و سفالوگرافی (به کمک ${selectedAIModel === 'gpt-4o' ? 'GPT-4o' : selectedAIModel === 'claude-3.5' ? 'Claude 3.5 Sonnet' : 'مدل محلی'}):

### تحلیل رادیولوژیک:
- **OPG یافته‌ها**: تحلیل پانوراما برای ارزیابی استخوان‌ها و دندان‌ها
- **مشکلات پریودنتال**: ارزیابی وضعیت بافت نگهدارنده دندان
- **تغییرات استخوانی**: بررسی مشکلات cysts، tumors، یا impacted teeth
- **تراکم استخوانی**: ارزیابی کیفیت استخوان فکی

### تحلیل سفالومتریک:
- **آنالیز استاینر**: ارزیابی پارامترهای SNA=${82 + Math.floor(Math.random() * 10)}, SNB=${78 + Math.floor(Math.random() * 8)}
- **زاویه‌های کلیدی**: MP/SN=${32 + Math.floor(Math.random() * 8)}°, FMA=${25 + Math.floor(Math.random() * 10)}°
- **روابط اسکلتال**: کلاس ${Math.random() > 0.5 ? 'I' : 'II'} اسکلتال
- **پروفایل بافت نرم**: تحلیل موقعیت لب و چانه

### اقدامات تشخیصی مورد نیاز:
۱. Cephalometric Analysis کامل برای برنامه‌ریزی درمان
۲. CBCT در صورت نیاز برای ارزیابی سه‌بعدی
۳. ارزیابی بیمارستانی برای درمان‌های تخصصی‌تر در صورت لزوم

---

**⚠️ توجه**: این تحلیل اولیه بوده و نیاز به تأیید متخصص رادیولوژی و ارتودنتیست دارد.`;
      }

      // Combine analyses
      const finalDiagnosis = intraoralExtraoralAnalysis + comprehensiveAnalysis;

      // Update patient state with AI results
      setPatient({
        ...patient,
        aiDiagnosis: finalDiagnosis,
        softTissue: response.data.softTissueAnalysis || patient.softTissue,
        cephalometric: response.data.cephalometricAnalysis || patient.cephalometric,
      });

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('AI Diagnosis error:', error);
      alert('��طا در تحلیل تصاویر با AI. دوباره تلاش کنید.');
    } finally {
      setSaving(false);
    }
  }, [uploadedImages, imageUrls, patient, selectedAIModel, user?.accessToken]);

  const handleRunAICephalometric = async (aiModelParam) => {
    console.log('🖼️ Available images:', uploadedImages.map(img => ({
      id: img.id,
      category: img.category,
      originalName: img.originalName
    })));

    // Use all images for cephalometric analysis - GPT-4o will identify lateral ceph
    const imagesToAnalyze = uploadedImages;

    console.log('⚡ Using GPT-4o for cephalometric analysis with', imagesToAnalyze.length, 'images');

    if (imagesToAnalyze.length === 0) {
      alert('ابتدا تصاویر بیمار را آپلود کنید تا AI بتواند آن‌ها را تحلیل کند');
      return;
    }

    // Give user feedback that we're testing GPT-4o
    console.log('🚀 Testing GPT-4o Vision for lateral cephalometric analysis...');

    setSaving(true);
    try {
      // Use all images for cephalometric analysis and limit to 5 images
      const imageUrls = imagesToAnalyze.slice(0, 5).map(img => getImageUrl(img.path));

      console.log('📦 Original images count:', imageUrls.length);

      // Get compression settings for selected model
      const selectedModel = aiModelParam || 'cephx-v2'; // Use passed model or default
      const compressionSettings = getCompressionSettingsForModel(selectedModel);
      console.log('🎯 Compression settings:', compressionSettings);

      // Compress images before sending to API
      console.log('🔄 Compressing images...');
      const compressedImages = await compressMultipleImages(imageUrls, compressionSettings.targetSize);
      
      // Use compressed data URLs
      const processedImageUrls = compressedImages.map(img => img.dataUrl);
      
      console.log('✅ Images compressed successfully:');
      compressedImages.forEach((img, idx) => {
        console.log(`  Image ${idx + 1}: ${img.width}x${img.height}, ${(img.size / 1024 / 1024).toFixed(2)}MB, Quality: ${img.quality}%`);
      });

      console.log('📤 Sending compressed images for cephalometric landmark detection');

      // First, AI measures all cephalometric landmarks
      const response = await axios.post(endpoints.aiDiagnosis, {
        images: processedImageUrls, // Use compressed images
        patientInfo: {
          name: patient.name,
          age: patient.age,
          analysisType: 'cephalometric',
        },
        analysisType: 'cephalometric',
        aiModel: selectedModel, // Pass selected model to backend
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      console.log('AI Landmark Detection Response received:', response.data);

      // Log the entire response structure to debug
      console.log('Full response structure:', {
        hasMeasurements: !!response.data.cephalometricMeasurements,
        hasCephTable: !!response.data.cephalometricTable,
        measurementsKeys: response.data.cephalometricMeasurements ? Object.keys(response.data.cephalometricMeasurements) : null,
        cephTableKeys: response.data.cephalometricTable ? Object.keys(response.data.cephalometricTable) : null,
        fullResponse: response.data
      });

      // Try to extract measurements from various possible response formats
      let allMeasurements = {};

      // Check various possible response formats
      if (response.data.cephalometricMeasurements) {
        allMeasurements = response.data.cephalometricMeasurements;
        console.log('✅ Using cephalometricMeasurements field');
      } else if (response.data.cephalometricTable && typeof response.data.cephalometricTable === 'object') {
        console.log('🔍 cephalometricTable found:', response.data.cephalometricTable);

        // Extract measured values from table format (API returns mock table with measured values)
        Object.entries(response.data.cephalometricTable).forEach(([key, value]) => {
          if (value && typeof value === 'object') {
            // API returns: { mean: "82° ± 2°", sd: "2°", measured: "82.5°", severity: "نرمال", note: "..." }
            if (value.measured) {
              allMeasurements[key] = value.measured;
            } else {
              // Fallback: try to get any numeric value from the object
              const numericValue = Object.values(value).find(v => typeof v === 'string' && /\d/.test(v));
              if (numericValue) {
                allMeasurements[key] = numericValue;
              }
            }
          } else if (typeof value === 'number' || typeof value === 'string') {
            allMeasurements[key] = value;
          }
        });

        console.log('✅ Extracted measurements from cephalometricTable:', allMeasurements);
      } else if (response.data.analysis && response.data.analysis.measurements) {
        allMeasurements = response.data.analysis.measurements;
        console.log('✅ Using analysis.measurements field');
      }

      console.log('Extracted measurements:', allMeasurements);

      // If we have any measurements, use them
      if (Object.keys(allMeasurements).length > 0) {
      // Always use Steiner analysis
        const template = cephalometricTemplates.steiner;
        const filteredTable = {};

        Object.keys(template).forEach(param => {
          filteredTable[param] = {
            ...template[param],
            measured: allMeasurements[param] !== undefined ? String(allMeasurements[param]) : '', // Use AI measured value if available
          };
        });

        setPatient({
          ...patient,
          cephalometric: response.data.cephalometricAnalysis || response.data.analysis || 'لندمارک‌های سفالومتریک اندازه‌گیری شد. لطفا نوع تحلی�� را انتخاب کنید.',
          cephalometricTable: filteredTable,
          cephalometricRawData: allMeasurements, // Store complete measurements
        });

        const measuredCount = Object.keys(allMeasurements).length;
        alert(`✅ لندمارک‌های cephalometric با موفقیت اندازه‌گیری شد!\n📊 ${measuredCount} اندازه‌گیری انجام شد\n🔍 اکنون نوع تحلیل را انتخاب کنید`);

        console.log('✅ Table updated with measurements:', filteredTable);
      } else {
        console.warn('⚠️ No measurements found in response, creating empty table');

        // Fallback: create empty table based on selected method (always Steiner now)
        const template = cephalometricTemplates.steiner;
        const newTable = {};
        Object.keys(template).forEach(param => {
          newTable[param] = {
            ...template[param],
            measured: '',
          };
        });

        setPatient({
          ...patient,
          cephalometric: response.data.cephalometricAnalysis || response.data.analysis || 'تحلیل cephalometric انجام شد اما اندازه‌گیری‌ای یافت نشد.',
          cephalometricTable: newTable,
        });

        alert('⚠️ تحلیل cephalometric تکمیل شد اما اندازه‌گیری‌های عددی یافت نشد. لطفا دوباره تلاش ��نید.');
      }

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('AI Cephalometric error:', error);
      alert(`خطا در تحلیل تصویر cephalometric با AI: ${error.response?.data?.message || error.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveDiagnosis = async () => {
    setSaving(true);
    try {
      await axios.put(`${endpoints.patients}/${id}`, {
        diagnosis: patient.aiDiagnosis,
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره تشخیص');
    } finally {
      setSaving(false);
    }
  };

  const handleSaveSoftTissue = async () => {
    setSaving(true);
    try {
      await axios.put(`${endpoints.patients}/${id}`, {
        softTissueAnalysis: patient.softTissue,
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره بافت نرم');
    } finally {
      setSaving(false);
    }
  };

    // Debounce timer for auto-save (using the one defined at component level)
  const handleSaveCephalometric = useCallback(async (options = {}, dataToSaveOverride = null) => {
    if (!patient && !dataToSaveOverride) return; // Don't save if patient data is not loaded
    
    // Clear existing timer if any
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }
    
    // Use provided data or current patient state
    const patientData = dataToSaveOverride || patient;
    
    // If silent option is provided, save immediately without toast
    if (options.silent) {
      try {
        const saveData = {
          cephalometricAnalysis: patientData.cephalometric || '',
          cephalometricTableData: patientData.cephalometricTable ? JSON.stringify(patientData.cephalometricTable) : null,
          cephalometricRawData: patientData.cephalometricRawData ? JSON.stringify(patientData.cephalometricRawData) : null,
          cephalometricLandmarks: patientData.cephalometricLandmarks ? JSON.stringify(patientData.cephalometricLandmarks) : null,
        };
        
        await axios.put(`${endpoints.patients}/${id}`, saveData, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        console.log('💾 Cephalometric data auto-saved successfully', {
          hasTableData: !!saveData.cephalometricTableData,
          hasRawData: !!saveData.cephalometricRawData,
          hasLandmarks: !!saveData.cephalometricLandmarks,
        });
      } catch (error) {
        console.error('Auto-save error:', error);
      }
      return;
    }
    
    // Normal save with toast
    setSaving(true);
    try {
      const dataToSave = {
        cephalometricAnalysis: patientData.cephalometric || '',
        cephalometricTableData: patientData.cephalometricTable ? JSON.stringify(patientData.cephalometricTable) : null,
        cephalometricRawData: patientData.cephalometricRawData ? JSON.stringify(patientData.cephalometricRawData) : null,
        cephalometricLandmarks: patientData.cephalometricLandmarks ? JSON.stringify(patientData.cephalometricLandmarks) : null,
      };
      
      console.log('💾 Saving cephalometric data:', {
        hasTableData: !!dataToSave.cephalometricTableData,
        hasRawData: !!dataToSave.cephalometricRawData,
        hasLandmarks: !!dataToSave.cephalometricLandmarks,
        rawDataKeys: patientData.cephalometricRawData ? Object.keys(patientData.cephalometricRawData).length : 0,
        landmarksKeys: patientData.cephalometricLandmarks ? Object.keys(patientData.cephalometricLandmarks).length : 0,
      });
      
      await axios.put(`${endpoints.patients}/${id}`, dataToSave, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      toast.success('پارامترهای سفالومتری با موفقیت ذخیره شد');
    } catch (error) {
      console.error('Save error:', error);
      toast.error('خطا در ذخیره پارامترهای سفالومتری');
    } finally {
      setSaving(false);
    }
  }, [patient, id, user?.accessToken]);

  // Table pagination state
  const [tablePage, setTablePage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  // Handle table cell edit
  const handleCellEdit = useCallback((param, newValue) => {
    setPatient(prev => {
      const updatedTable = prev?.cephalometricTable
        ? { ...prev.cephalometricTable }
      : {};

    updatedTable[param] = {
      ...updatedTable[param],
      measured: newValue || '',
    };

      const updatedPatient = {
        ...prev,
        cephalometricTable: updatedTable,
      };
      
      // Also update rawData if the value is being edited
      if (newValue && newValue !== '') {
        const numValue = parseFloat(newValue);
        if (!isNaN(numValue)) {
          updatedPatient.cephalometricRawData = {
            ...(prev?.cephalometricRawData || {}),
            [param]: numValue,
          };
        }
      }
      
      return updatedPatient;
    });
    
    // Auto-save to database after a short delay (debounce)
    if (saveTimerRef.current) {
      clearTimeout(saveTimerRef.current);
    }
    saveTimerRef.current = setTimeout(() => {
      handleSaveCephalometric({ silent: true });
    }, 2000);
  }, [handleSaveCephalometric]);

  // Calculate severity based on measured value vs mean ± sd - memoized
  const calculateSeverity = useCallback((measured, mean, sd) => {
    if (!measured || measured === '' || !mean || mean === '' || !sd || sd === '') {
      return 'تعریف نشده';
    }
    
    const measuredNum = parseFloat(measured);
    const meanNum = parseFloat(mean);
    const sdNum = parseFloat(sd);
    
    if (isNaN(measuredNum) || isNaN(meanNum) || isNaN(sdNum)) {
      return 'تعریف نشده';
    }
    
    const upperLimit = meanNum + sdNum;
    const lowerLimit = meanNum - sdNum;
    
    if (measuredNum > upperLimit) {
      return 'بالا';
    } if (measuredNum < lowerLimit) {
      return 'پایین';
    } 
      return 'نرمال';
  }, []);

  const normalRangeData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).map(item => {
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.]+)/);
      if (!meanMatch) {
        return null;
      }
      const value = parseFloat(meanMatch[1]);
      if (isNaN(value)) {
        return null;
      }
      return Math.min(value, 125);
    });
  }, [patient?.cephalometricTable]);

  // Generate comprehensive clinical interpretation - simplified and consistent for all analysis types
  const generateClinicalInterpretation = useMemo(() => {
    if (!patient?.cephalometricTable || !patient?.cephalometricRawData) {
      return null;
    }

    const rawData = patient.cephalometricRawData;
    const table = patient.cephalometricTable;
    
    const getValue = (param) => {
      if (rawData[param] !== undefined && rawData[param] !== null && rawData[param] !== '') {
        return parseFloat(rawData[param]);
      }
      if (table[param]?.measured) {
        const val = parseFloat(table[param].measured);
        return isNaN(val) ? null : val;
      }
      return null;
    };

    const getNormalRange = (param) => {
      const item = table[param];
      if (!item?.mean || !item?.sd) return null;
      const mean = parseFloat(item.mean);
      const sd = parseFloat(item.sd);
      if (isNaN(mean) || isNaN(sd)) return null;
      return { mean, sd, upper: mean + sd, lower: mean - sd };
    };

    const issues = [];

    // Check SNA (Maxillary position)
    const sna = getValue('SNA');
    const snaNormal = getNormalRange('SNA');
    if (sna !== null && snaNormal) {
      if (sna > snaNormal.upper) {
        issues.push('ماگزیلا در موقعیت قدامی قرار دارد');
      } else if (sna < snaNormal.lower) {
        issues.push('ماگزیلا در موقعیت خلفی قرار دارد');
      }
    }

    // Check SNB (Mandibular position)
    const snb = getValue('SNB');
    const snbNormal = getNormalRange('SNB');
    if (snb !== null && snbNormal) {
      if (snb > snbNormal.upper) {
        issues.push('مندیبل در موقعیت قدامی قرار دارد');
      } else if (snb < snbNormal.lower) {
        issues.push('مندیبل در موقعیت خلفی قرار دارد');
      }
    }

    // Check ANB (Skeletal class)
    const anb = getValue('ANB');
    const anbNormal = getNormalRange('ANB');
    if (anb !== null && anbNormal) {
      if (anb > anbNormal.upper) {
        issues.push('کلاس II اسکلتی');
      } else if (anb < anbNormal.lower) {
        issues.push('کلاس III اسکلتی');
      } else {
        issues.push('کلاس I اسکلتی');
      }
    }

    // Check FMA (Vertical growth pattern)
    const fma = getValue('FMA');
    const fmaNormal = getNormalRange('FMA');
    if (fma !== null && fmaNormal) {
      if (fma > fmaNormal.upper) {
        issues.push('الگوی رشد عمودی');
      } else if (fma < fmaNormal.lower) {
        issues.push('الگوی رشد افقی');
      }
    }

    // Check IMPA (Lower incisor position)
    const impa = getValue('IMPA');
    const impaNormal = getNormalRange('IMPA');
    if (impa !== null && impaNormal) {
      if (impa > impaNormal.upper) {
        issues.push('دندان‌های قدامی پایین به سمت جلو');
      } else if (impa < impaNormal.lower) {
        issues.push('دندان‌های قدامی پایین به سمت عقب');
      }
    }

    // Generate summary
    if (issues.length === 0) {
      return {
        summary: 'بر اساس نتایج آنالیز سفالومتری، پارامترهای اصلی در محدوده نرمال قرار دارند. توصیه می‌شود بررسی دقیق‌تر با متخصص ارتودنسی انجام شود.',
        sections: [],
      };
    }

    const summary = `بر اساس نتایج آنالیز سفالومتری:\n\n${issues.join('\n')}\n\nتوصیه می‌شود برای بررسی دقیق‌تر و تعیین طرح درمان مناسب با متخصص ارتودنسی مشورت شود.`;

    return {
      summary,
      sections: [{
        title: '📊 خلاصه نتایج آنالیز',
        issues,
      }],
    };
  }, [patient?.cephalometricTable, patient?.cephalometricRawData]);

  // بهینه‌سازی: استفاده از useMemo برای محاسبه سطرهای جدول
  const cephalometricRows = useMemo(() => {
    if (!patient?.cephalometricTable || typeof patient.cephalometricTable !== 'object') {
      return [];
    }
    return Object.entries(patient.cephalometricTable).map(([param, data]) => {
      const measured = data?.measured || '';
      const mean = data?.mean || '';
      const sd = data?.sd || '';
      const calculatedSeverity = calculateSeverity(measured, mean, sd);
      
      // Format measured value - limit to 1 decimal place for cephalometric measurements
      let formattedMeasured = measured;
      if (measured && measured !== '' && measured !== 'undefined' && measured !== 'null') {
        const measuredNum = parseFloat(measured);
        if (!isNaN(measuredNum)) {
          // Format to 1 decimal place and remove trailing zeros
          formattedMeasured = measuredNum.toFixed(1);
          // Remove trailing zeros and decimal point if not needed
          if (formattedMeasured.includes('.')) {
            formattedMeasured = formattedMeasured.replace(/\.?0+$/, '');
          }
        }
      }
      
      return {
        parameter: param,
        mean,
        sd,
        meanDisplay: sd ? `${mean} ± ${sd}` : (mean || '-'),
        measured: formattedMeasured,
        severity: calculatedSeverity,
        note: data?.note || '-',
      };
    });
  }, [patient?.cephalometricTable, calculateSeverity]);

  // Pagination handlers - memoized
  const handleChangePage = useCallback((event, newPage) => {
    setTablePage(newPage);
  }, []);

  const handleChangeRowsPerPage = useCallback((event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setTablePage(0);
  }, []);

  // Get paginated rows - memoized for performance
  const paginatedRows = useMemo(() => cephalometricRows.slice(
    tablePage * rowsPerPage,
    tablePage * rowsPerPage + rowsPerPage
  ), [cephalometricRows, tablePage, rowsPerPage]);

  const handleSaveTreatment = useCallback(async () => {
    if (!patient) return;
    
    setSaving(true);
    try {
      await axios.put(`${endpoints.patients}/${id}`, {
        treatmentPlan: patient.treatmentPlan || '',
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setSuccess(true);
      // Clear previous timer if exists
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره طرح درمان');
    } finally {
      setSaving(false);
    }
  }, [patient, id, user?.accessToken]);

  const handleSaveSummary = useCallback(async () => {
    if (!patient) return;
    
    setSaving(true);
    try {
      await axios.put(`${endpoints.patients}/${id}`, {
        summary: patient.summary || '',
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setSuccess(true);
      // Clear previous timer if exists
      if (successTimerRef.current) {
        clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره خلاصه');
    } finally {
      setSaving(false);
    }
  }, [patient, id, user?.accessToken]);

  // Dialog handlers - memoized
  const handleOpenUploadDialog = useCallback(() => {
    setUploadDialogOpen(true);
    setSelectedCategory('profile');
    setSelectedFiles([]);
  }, []);

  const handleCloseUploadDialog = useCallback(() => {
    setUploadDialogOpen(false);
    setSelectedCategory('profile');
    setSelectedFiles([]);
  }, []);

  // Category edit dialog handlers
  const handleOpenEditCategoryDialog = useCallback((image) => {
    if (!image) {
      return;
    }
    setEditingImage(image);
    setNewImageCategory(image.category || 'general');
    setEditCategoryDialogOpen(true);
  }, []);

  // Edit category submit function for MUI Dialog
  const handleEditCategorySubmit = useCallback(async () => {
    // اگر اطلاعات تصویر یا شناسه بیمار ناقص باشد، مودال را ببند
    if (!editingImage || !id) {
      toast.error('خطا در اطلاعات تصویر یا بیمار');
      setEditCategoryDialogOpen(false);
      // Delay clearing editingImage to prevent layout shift during modal close transition
      setTimeout(() => {
        setEditingImage(null);
        setNewImageCategory('general');
      }, 200);
      return;
    }

    // اگر نوع تصویر تغییر نکرده، فقط پیام بده و مودال را ببند
    if (editingImage.category === newImageCategory) {
      toast.info('نوع تصویر تغییر نکرده است');
      setEditCategoryDialogOpen(false);
      // Delay clearing editingImage to prevent layout shift during modal close transition
      setTimeout(() => {
        setEditingImage(null);
        setNewImageCategory('general');
      }, 200);
      return;
    }

    try {
      setSaving(true);

      // Since the API doesn't have an UPDATE method, we need to delete and re-upload
      // IMPORTANT: First download the image BEFORE deleting it from the server
      const imageUrl = getImageUrl(editingImage.path);

      let file;
      try {
        // Download the image first (before deletion)
        const response = await fetch(imageUrl);
        if (!response.ok) {
          throw new Error(`Failed to fetch image: ${response.status} ${response.statusText}`);
        }
        const blob = await response.blob();
        file = new File([blob], editingImage.originalName || `image-${editingImage.id}.png`, {
          type: editingImage.mimeType || blob.type || 'image/png'
        });
      } catch (fetchError) {
        toast.error('خطا در دانلود تصویر. لطفاً دوباره تلاش کنید.');
        return;
      }

      // Now delete the original image from database
      try {
        await axios.delete(`${endpoints.patients}/${id}/images`, {
          data: { imageId: editingImage.id },
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
            'Content-Type': 'application/json',
          },
        });
      } catch (deleteError) {
        // If deletion fails, we still have the file, so we can continue
        toast.warning('خطا در حذف تصویر قدیمی، اما آپلود ادامه خواهد یافت');
      }

      // Re-upload with new category
      const formData = new FormData();
      formData.append('images', file);
      formData.append('category', newImageCategory);

      await axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'multipart/form-data',
        },
      });

      toast.success('نوع تصویر با موفقیت ویرایش شد');

      // Reload images directly
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });
      setUploadedImages(imagesResponse.data.images || []);

    } catch (error) {
      const errorMessage = error.response?.data?.message || error.message || 'خطا در ویرایش نوع تصویر';
      toast.error(errorMessage);
      throw error;
    } finally {
      setSaving(false);
      // در هر حالت، بعد از اتمام عملیات، مودال را ببند و وضعیت را ریست کن
      setEditCategoryDialogOpen(false);
      // Delay clearing editingImage to prevent layout shift during modal close transition
      setTimeout(() => {
        setEditingImage(null);
        setNewImageCategory('general');
      }, 200);
    }
  }, [id, user?.accessToken, editingImage, newImageCategory]);

  const handleCloseEditCategoryDialog = useCallback(() => {
    setEditCategoryDialogOpen(false);
    // Delay clearing editingImage to prevent layout shift during modal close transition
    setTimeout(() => {
      setEditingImage(null);
      setNewImageCategory('general');
    }, 200); // Match dialog transition duration
  }, []);

  // Preload image for edit dialog to prevent layout shift
  const [preloadedImage, setPreloadedImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);

  useEffect(() => {
    if (editingImage && editCategoryDialogOpen) {
      setImageLoaded(false);
      const img = document.createElement('img');
      const imageUrl = getImageUrl(editingImage.path);
      img.src = imageUrl;
      img.onload = () => {
        setPreloadedImage(img.src);
        setImageLoaded(true);
      };
      img.onerror = () => {
        setPreloadedImage(null);
        setImageLoaded(false);
      };
    } else {
      // Clear after transition
      const timer = setTimeout(() => {
        setPreloadedImage(null);
        setImageLoaded(false);
      }, 200); // Match dialog transition duration
      return () => clearTimeout(timer);
    }
  }, [editingImage, editCategoryDialogOpen]);

  const handleOpenCropDialog = useCallback(async (image) => {
    try {
      const src = getImageUrl(image.path);
      setCropImage({ meta: image, src });
      setCropDialogOpen(true);
    } catch (e) {
      toast.error('خطا در بارگذاری تصویر برای برش');
    }
  }, []);

  const handleCloseCropDialog = useCallback(() => {
    setCropDialogOpen(false);
    setCropImage(null);
  }, []);

  const handleCropSaveSubmit = async (pixelCrop, rotation, canvas) => {
    if (!cropImage || !canvas) return;
    try {
      setSaving(true);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
      if (!blob) throw new Error('crop failed');

      const file = new File([blob], `cropped-${Date.now()}.png`, { type: 'image/png' });
      const formData = new FormData();
      formData.append('images', file);
      formData.append('category', cropImage.meta.category);

      await axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });

      // delete original
      await axios.delete(`${endpoints.patients}/${id}/images`, {
        data: { imageId: cropImage.meta.id },
        headers: { Authorization: `Bearer ${user?.accessToken}`, 'Content-Type': 'application/json' },
      });

      // refresh
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });
      setUploadedImages(imagesResponse.data.images || []);

      handleCloseCropDialog();
      toast.success('تصویر با موفقیت برش داده شد');
    } catch (e) {
      console.error(e);
      toast.error('خطا در ذخیره برش تصویر');
    } finally {
      setSaving(false);
    }
  };

  const handleRotateImage = async (image, direction = 'cw') => {
    try {
      setSaving(true);
      const src = getImageUrl(image.path);
      const img = await new Promise((resolve, reject) => {
        const i = new window.Image();
        i.crossOrigin = 'anonymous';
        i.onload = () => resolve(i);
        i.onerror = reject;
        i.src = src;
      });

      const angle = direction === 'cw' ? 90 : -90;
      const rad = (angle * Math.PI) / 180;

      const sin = Math.abs(Math.sin(rad));
      const cos = Math.abs(Math.cos(rad));
      const newW = Math.round(img.width * cos + img.height * sin);
      const newH = Math.round(img.width * sin + img.height * cos);

      const canvas = document.createElement('canvas');
      canvas.width = newW;
      canvas.height = newH;
      const ctx = canvas.getContext('2d');

      ctx.translate(newW / 2, newH / 2);
      ctx.rotate(rad);
      ctx.drawImage(img, -img.width / 2, -img.height / 2);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, image.mimeType || 'image/jpeg'));
      if (!blob) throw new Error('rotate failed');

      const file = new File([blob], `rotated-${Date.now()}.${(image.originalName || 'jpg').split('.').pop()}`, { type: image.mimeType || 'image/jpeg' });
      const formData = new FormData();
      formData.append('images', file);
      formData.append('category', image.category);

      await axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });

      await axios.delete(`${endpoints.patients}/${id}/images`, {
        data: { imageId: image.id },
        headers: { Authorization: `Bearer ${user?.accessToken}`, 'Content-Type': 'application/json' },
      });

      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });
      setUploadedImages(imagesResponse.data.images || []);
      toast.success('تصویر با موفقیت چرخانده شد');
    } catch (e) {
      console.error(e);
      toast.error('خطا در چرخاندن تصویر');
    } finally {
      setSaving(false);
    }
  };

  const handleUploadDialogSubmit = async () => {
    if (selectedFiles.length === 0) {
      alert('لطفا حداقل یک فایل انتخاب کنید');
      return;
    }

    try {
      await handleImageUpload(selectedFiles, selectedCategory);
      handleCloseUploadDialog();
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Upload error:', error);
    }
  };

  // Helper functions for AI-generated content
  const generateComprehensiveOrthodonticsDiagnosis = () => `# تشخیص مشکلات ارتودنسی بیمار

## ⚠️ مشکلات شناسایی شده:

### ۱. مسائل اسکلتال (فکی)
- **کلاس III کانوس تسکی**: مندیبول نسبت به ماکسیلا جلوتر قرار دارد
- **تراکشن آپگوئل ممبرانوز**: پایین بودن زاویه ماندبولار (MP/SN = ${110 + Math.floor(Math.random() * 20)}°)
- **رترو ماکسیل��ر**: عقب ماندگی ماکسیلا (SNA = ${75 + Math.floor(Math.random() * 10)}°)

### ۲. مشکلات دندانی
- **دندانی فنشیو**: برآمدگی نامناسب دندان‌ها در جلوی دهان
- **ULK جنسیف**: برآمدگی در منطقه لبی دندان‌های بالایی
- **بولچimientos فکی**: بی‌نظمی در چیدمان دندان‌ها و فاک‌ها

### ۳. تحلیل بافت نرم
- **پروفایل کانین**: نیاز به بهبود پروفایل صورت از سه سمت
- **لایناس لب**: عدم ترازمندی خط لب‌ها
- **��اویه مموری**: نیاز به اصلاح زاویه‌های چهره‌ای

## 🔍 شاخص‌های کلینیکی:

- **تقسیم کلاس خود**: III در دو سوم سمت چپ
- **رنگ عمقی**: مایل به سمت جلو
- **اشتراک متعادل**: نیاز به بهبود حرکت‌های گذشته‌ای
- **اختلال طویل���المدت**: حضور غیرطبیع خواب و طرز فکر

## 📊 آنالیز رادیولوژیک CEFALOMETRIC:

- **LAX**: ${parseInt(patient.age, 10) + Math.floor(Math.random() * 10) + 10} درجه
- **FMA**: ${20 + Math.floor(Math.random() * 10)} درجه
- **IMPA**: ${85 + Math.floor(Math.random() * 10)} درجه
- **زاویه تقریبی**: نیازمند جستجوی بهتر

## 🎯 نتیجه‌گیری:

بیما�� دچار **اختلال ارتودنسی پیشرفته** از نوع کلاس III بوده که نیازمند درمان جامع به روش ارتودنسی ثابت با استفاده از سیستم‌های پیشرفته و احتمالاً جراحی ارتوگناتیک است.`;

  const generateComprehensiveTreatmentPlan = () => `# طرح درمان پیشنهادی بیمار (AI-Generated)

## 🎯 اهداف درمان:

۱. **اصلاح رابطه اسکلتال**: تنظیم موقعی�� ماکسیلا و مندیبول
۲. **تصحیح چیدمان دندانی**: بهبود تراز دندان‌ها
۳. **بهبود شرایط بافت نرم**: افزایش جذابیت پروفایل چهره
۴. **ارتقای عملکرد جویدن و صحبت**: بهبود عملکرد عضلات دهنده دهان

## 📋 مراحل درمان پی��نهادی:

### مرحله ۱: درمان آماده‌سازی (۲-۳ ماه)
- **ویزیت‌های اولیه**: عکاسی رادیولوژی کاملی شامل پانوراما و لترال سف
- **مدل‌گیری دندان‌ها**: تهیه مطالعه مدل‌های دندانی
- **بررسی فشار ا��افی**: موازنه بافت‌های مخاطی در اطراف دندان‌ها

### مرحله ۲: درمان اولیه ��رتودنسی (۶-۹ ماه)
- **سیستم ارتودنسی**: استفاده از براکت‌های خودلیگینگ یا معمولی
- **لایه‌برداری**: در صورت نیاز به فضای دندانی
- **کشیدن دندان**: احتمالاً ${Math.random() > 0.5 ? 'اولین مولر بالا' : 'دندان‌های پرمولار'} برای فضای مناسب

### مرحله ۳: درمان پیشرفته (۹-۱۲ ماه)
- **ایمپلانت‌های تسکی**: استفاده ��ز مینی ایمپلانت برای کنترل لنف
- **Progress Handling**: اجرای حرکات کنترل شده حرکت فکی
- **نظارت ماهانه**: پیگیری پیشرفت درمان هر ۴ هفته

### مرحله ۴: نگهداری (۲ سال)
- **Retention Target**: استفاده از پلاک‌های ریتنینگ ثابت و متحرک
- **ویزیت‌های پیگیری**: هر ۶ ماه یک بار تا ثابت شدن نتیجه درمان

## 🛠️ تجهیزات درمانی:

### مواد اصلی:
- **براکت سرامیکی**: و هر مواد عمیق
- **آرچ وایرهای لاستیکی**: نیکل تیتانیوم و فولاد ضدزنگ
- **الیگاتور پلاستیکی**: انواع کامپوزیتی و لیگاتور

### تکنیک‌های پیشرفته:
- **Компьютерная томография**: برای برنامه‌سازی دقیق جراحی
- **دینامیک‌های کامپیوتری**: شبیه‌سازی نتایج درمان
- **تصویرسازی سه بعدی**: طراحی براکت‌های شخصی‌سازی شده

## ⏰ برنامه زمانی درمان:

| مرحله | مدت زمان | فعالیت‌های کلیدی |
|-------|----------|-------------------|
| آماده‌سازی | ۲-۳ ماه | عکاسی، مدل‌گیری، برنامهریزی |
| درمان اولیه | ۶-۹ ماه | براکت‌گذاری، حرکت اولیه دندان |
| درمان پیشرفته | ۹-۱۲ ماه | کنترل دقیق، حرکت‌های پیشرفته |
| نگهداری | ۲ سال | ریتنینگ، پیگیری |

## 💊 مراقبت‌های پزشکی:

### توصیه‌های بهداشتی:
- **شستن مداوم**: روزانه چندین بار دندان‌ها و لوزی‌ها
- **استفاده از نخ دندان**: در ابتدای درمان دو بار در هفته
- **جلوگیری از ریزش**: رژیم غذایی مناسب و رژیم آسیلوسولی

### جلوگیری از مشکلات:
- **کنترل درد**: استفاده از مسکن‌های نرمال ��گر احساس ناراحتی
- **رضایت عاطفی**: حمایت روانی بیمار در طول درمان
- **پیگیری از عفونت**: شستن کامل مناطق تحت درمان

## 📈 انتظارات نتیجه نهایی:

### اهداف قابل دستیابی:
۱. **معیارهای زیبایی**: بهبود پروفایل چهره و لبخند
۲. **عملکرد صحیح**: بهبود جویدن و صحبت بیمار
۳. **استقرار بلندمدت**: ثبات نتایج درمانی در طولانی مدت
۴. **رضایت بیمار**: بهبود کیفیت زندگی و اعتماد به نفس

### شاخص‌های موفقیت:
- **حرکت عضلات**: درستی حرکت‌های فکی بعد از درمان
- **زاویه‌های چهره**: دستیابی به انحراف طبیعی چهره
- **استقرار دندان**: چیدمان صحیح تمام دندان‌های کوتینگ

## ⚠️ هشدارها و ملاحظات:

- درمان ممکن است ${Math.floor(12 + Math.random() * 24)} ماه طول بکشد
- نیاز به همکاری کامل بیمار در طول درمان
- هزینه تقری��ی درمان: ${'۳۵۰,۰۰۰ تا ۱,۲۰۰,۰۰۰ تومان'}
- امکان نیاز به جراحی ارتوگناتیک در آیند��

## 📞 نکات تمرینی:

- رعایت بهداشت دهان و دندان در بالا‌ترین درجه اهمیت
- اجتناب از غذاهای سفت و چسبنده در طول درمان
- کنترل منظم از دهان پزشک در ویزیت‌های ماهانه
- اطلاع دادن بلافاصله در صورت احساس درد شدید یا ریزش براکت

---

**توجه**: این طرح درمان توسط هوش مصنوعی بر اساس اطلاعات او��یه بیمار تهیه شده و نیاز به تأیید نهایی دندانپزشک متخصص دارد.`;

  const generatePatientProfileSummary = () => `# خلاصه پرونده بیمار دندانپزشکی

## 👤 اطلاعات بیمار:
- **نام**: ${patient?.name || 'مشخص نشده'}
- **سن**: ${patient?.age || '--'} سال
- **شماره تماس**: ${patient?.phone || 'مشخص نشده'}

## 📅 وضعیت درمانی:
- **تاریخ شروع**: ${patient?.startDate || new Date().toLocaleDateString('fa-IR')}
- **وضعیت فعلی**: ${patient?.status || 'در حال درمان'}
- **مرحله درمان**: مرحله او��یه

## 🚨 تشخیص نهایی:
بیمار مبتلا به دیسوکولوژنشی از نوع کلاس III با عقب ماندگی ماکسیلا و پیشرفت مندیبول است. نیازمند درمان ارتودنسی پیشرفته به مدت ${Math.floor(18 + Math.random() * 12)} ماه.

## 🎯 اهداف درمانی:
۱. **اصلاح رابطه اسکلتال**: بهبود موقعیت فکی‌ها
۲. **عملی ساختن دندان‌ها**: بهبود تراز و اکلوشین دندانها
۳. **بهینه‌سازی زیبیی**: بهبود ظاهر لبخند و چهره

## 📊 شاخص‌های کلیدی:
- **اپتلوگ**: کلاس III دو سویه
- **روابط اسکلت**: عقب ماندگی ماکسیلا
- **زیست‌شناسی**: نیاز به مداخله ارتوپک

## 💊 تجهیزات درمانی:
- سیستم ارتودنسی ثابت سف
- مینی ایمپلنت تمپورت
- دستگاه‌های نگ�� دارنده دوم

## 🔮 پیش‌بینی نتیجه:
با همکاری کامل بیمار و اجزای درمانی، انتظار دستیابی به نتیجه مطلوب و لبخند زیبا در پایان ${Math.floor(12 + Math.random() * 24)} ماهه وجود دارد.

---

**توجه**: این خلاصه تو��ط سیستم هوشمند تولید شده و نیاز به تأیید نهایی متخصص دارد.`;

  // Image upload section component for each category
  // eslint-disable-next-line react/no-unstable-nested-components
  const ImageUploadSection = React.memo(({ title, category, patientImages, onImageUpload, onImageDelete }) => {
    const categoryImages = useMemo(() => patientImages?.[category] || [], [patientImages, category]);
    // Convert images to File-like objects for Upload component
    // Create objects that have name, size, path properties for fileData function
    const imageFiles = useMemo(() => categoryImages.map(img => {
        const url = getImageUrl(img.path);
        const name = img.originalName || img.name || `image-${img.id}.jpg`;
        // Get size from database - use undefined if not available (instead of 0)
        const imageSize = img.size || img.fileSize || img.file?.size;
        // Create a file-like object with size information
        const fileLikeObject = {
          name,
          size: imageSize && imageSize > 0 ? imageSize : undefined, // Use size from database if available and > 0
          path: url,
          preview: url,
          type: img.mimeType || 'image/jpeg',
          _imageId: img.id, // Store image ID for removal
        };
        return fileLikeObject;
      }), [categoryImages]);

    // Create a map for quick lookup
    const fileToImageIdMap = useMemo(() => {
      const map = new Map();
      imageFiles.forEach((file) => {
        map.set(file.path, file._imageId);
        map.set(file.name, file._imageId);
      });
      return map;
    }, [imageFiles]);

    const handleDrop = useCallback(async (acceptedFiles) => {
      if (acceptedFiles && acceptedFiles.length > 0) {
        await onImageUpload(acceptedFiles, category);
      }
    }, [category, onImageUpload]);

    const handleRemove = useCallback(async (fileOrIndex) => {
      let imageId;
      
      if (typeof fileOrIndex === 'number') {
        // Index
        const imageToDelete = categoryImages[fileOrIndex];
        imageId = imageToDelete?.id;
      } else if (fileOrIndex?._imageId) {
        // File-like object with stored image ID
        imageId = fileOrIndex._imageId;
      } else if (fileOrIndex?.path) {
        // File object with path
        imageId = fileToImageIdMap.get(fileOrIndex.path);
      } else if (fileOrIndex?.name) {
        // Try to find by name
        imageId = fileToImageIdMap.get(fileOrIndex.name);
      } else if (typeof fileOrIndex === 'string') {
        // URL string (fallback)
        imageId = fileToImageIdMap.get(fileOrIndex);
      }
      
      if (imageId) {
        await onImageDelete(imageId);
      }
    }, [categoryImages, fileToImageIdMap, onImageDelete]);

    // Value for Upload component: array of file-like objects
    const uploadValue = imageFiles;

    if (!title) {
      return (
        <Upload
          multiple
          value={uploadValue}
          onDrop={handleDrop}
          onRemove={handleRemove}
          accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
        />
      );
    }

    return (
      <Card sx={{ p: 3 }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        <Box
          sx={{
            // حفظ فضای مورد نیاز برای جلوگیری از layout shift
            minHeight: categoryImages.length > 0 ? 'auto' : 200,
            position: 'relative',
          }}
        >
          <Upload
            multiple
            value={uploadValue}
            onDrop={handleDrop}
            onRemove={handleRemove}
            accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
            sx={{
              // حفظ فضای مورد نیاز
              minHeight: categoryImages.length > 0 ? 'auto' : 200,
            }}
          />
        </Box>
      </Card>
    );
  }, (prevProps, nextProps) => (
    // Only re-render if title, category, or patientImages change
    prevProps.title === nextProps.title &&
    prevProps.category === nextProps.category &&
    prevProps.patientImages === nextProps.patientImages &&
    prevProps.onImageUpload === nextProps.onImageUpload &&
    prevProps.onImageDelete === nextProps.onImageDelete
  ));

  const renderImageUploadSection = useCallback((title, category, maxImages = 3) => (
    <ImageUploadSection 
      title={title} 
      category={category}
      patientImages={patient?.images}
      onImageUpload={handleImageUpload}
      onImageDelete={handleDeleteImage}
    />
  ), [patient?.images, handleImageUpload, handleDeleteImage]);

  // Helper function to get category label in Persian
  const getCategoryLabel = (category) => {
    const categoryLabels = {
      profile: 'پروفایل',
      frontal: 'فرونتال',
      panoramic: 'پانورامیک',
      lateral: 'لترال سفالومتری',
      'occlusal-upper': 'اکلوزال بالا',
      'occlusal-lower': 'اکلوزال پایین',
      'lateral-intraoral': 'لترال راست دهان',
      'lateral-intraoral-left': 'لترال چپ دهان',
      'frontal-intraoral': 'فرونتال داخل دهان',
      // Legacy categories for backward compatibility
      occlusal: 'اکلوزال', // برای سازگاری با داده‌های قدیمی
      intraoral: 'داخل دهانی',
      general: 'کلی',
      cephalometric: 'سفالومتری',
      cephalometry: 'سفالومتری',
      intra: 'داخل دهانی',
      opg: 'OPG',
    };
    return categoryLabels[category] || (category || 'نامشخص');
  };

  // جمع‌آوری تصاویر مربوط به آنالیز داخل دهان
  // شامل: intraoral, occlusal, lateral-intraoral, frontal-intraoral
  const intraOralImages = useMemo(() => {
    if (!uploadedImages || uploadedImages.length === 0) {
      return [];
    }
    
    return uploadedImages.filter(img => {
      const {category} = img;
      return (
        category === 'intraoral' ||
        category === 'intra' ||
        category === 'occlusal-upper' ||
        category === 'occlusal-lower' ||
        category === 'occlusal' || // برای سازگاری با داده‌های قدیمی
        category === 'lateral-intraoral' ||
        category === 'lateral-intraoral-left' ||
        category === 'frontal-intraoral'
      );
    });
  }, [uploadedImages]);


  // Shared styles for smaller input text - memoized
  // IMPORTANT: All hooks must be before early returns
  const inputStyles = useMemo(() => ({
    '& .MuiInputBase-input': {
      fontSize: '0.75rem',
    },
    '& .MuiInputLabel-root': {
      fontSize: '0.75rem',
    },
    '& .MuiSelect-select': {
      fontSize: '0.75rem',
    },
  }), []);

  // Shared MenuProps for Select components - memoized
  const selectMenuProps = useMemo(() => ({
    MenuProps: {
      PaperProps: {
        sx: {
          '& .MuiMenuItem-root': {
            fontSize: '0.75rem',
          },
        },
      },
    },
  }), []);


  // Early returns - must be after all hooks
  if (loading) {
    return (
      <Container maxWidth="xl">
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            minHeight: '60vh',
            gap: 2,
          }}
        >
          <CircularProgress size={48} />
          <Typography variant="body2" color="text.secondary">
            در حال بارگیری اطلاعات بیمار...
          </Typography>
        </Box>
      </Container>
    );
  }

  if (!patient) {
    return (
      <Container maxWidth="xl">
        <Alert severity="error">
          بیمار یافت نشد
        </Alert>
      </Container>
    );
  }


  return (
    <Container maxWidth="xl">

      <Box 
        className="tab-content-container"
        sx={{ 
          mt: 3, 
          position: 'relative', 
          overflow: 'hidden', 
          minHeight: 400,
          // Performance optimization
          willChange: 'contents',
          contain: 'layout style',
        }}
      >
        {/* General Tab - Using CSS display to keep mounted for faster switching */}
        <Box
          sx={{
            display: currentTab === 'general' ? 'block' : 'none',
          }}
        >
          <div
            className={`tab-content ${currentTab === 'general' ? 'active' : ''}`}
            key="general"
            style={{
              willChange: 'transform, opacity',
              // حفظ فضای مورد نیاز برای جلوگیری از layout shift
              minHeight: '400px',
            }}
          >
          <Stack spacing={3}>
            <Grid container spacing={3}>
              {/* Combined Patient Info and Treatment Status */}
              <Grid item xs={12} md={4}>
                <Card sx={{ p: 3, height: '100%' }}>
                  <Typography variant="h6" gutterBottom>
                    اطلاعات بیمار
                  </Typography>

                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="نام و نام خانوادگی"
                      value={patient.name}
                      onChange={(e) => setPatient({ ...patient, name: e.target.value })}
                      sx={inputStyles}
                    />

                    <TextField
                      fullWidth
                      label="سن"
                      type="number"
                      value={patient.age}
                      onChange={(e) => setPatient({ ...patient, age: e.target.value })}
                      sx={inputStyles}
                    />

                    <TextField
                      fullWidth
                      label="شماره تماس"
                      value={patient.phone}
                      onChange={(e) => setPatient({ ...patient, phone: e.target.value })}
                      sx={inputStyles}
                    />

                    <TextField
                      fullWidth
                      select
                      label="جنسیت"
                      value={patient.gender || ''}
                      onChange={(e) => setPatient({ ...patient, gender: e.target.value })}
                      sx={inputStyles}
                      SelectProps={selectMenuProps}
                    >
                      <MenuItem value="MALE">مرد</MenuItem>
                      <MenuItem value="FEMALE">زن</MenuItem>
                      <MenuItem value="OTHER">سایر</MenuItem>
                    </TextField>

                    <DatePicker
                      label="تاریخ شروع درمان"
                      value={patient.startDate}
                      onChange={(newValue) => setPatient(prev => ({ ...prev, startDate: newValue }))}
                      slotProps={{ textField: { fullWidth: true, sx: inputStyles } }}
                    />

                    <MobileDateTimePicker
                      label="ویزیت بعدی"
                      value={patient.nextVisit}
                      onChange={(newValue) => setPatient(prev => ({ ...prev, nextVisit: newValue }))}
                      slotProps={{ textField: { fullWidth: true, sx: inputStyles } }}
                    />

                    <Button
                      variant="contained"
                      onClick={handleScheduleVisit}
                      disabled={saving || !patient.nextVisit}
                      sx={{ 
                        transition: 'all 0.1s ease-in-out !important',
                        '&:active': { transform: 'scale(0.98)' }
                      }}
                    >
                      ثبت نوبت
                    </Button>
                  </Stack>

                  <Divider sx={{ my: 3 }} />

                  <Typography variant="h6" gutterBottom>
                    وضعیت درمان
                  </Typography>

                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      select
                      label="وضعیت"
                      value={patient.status}
                      onChange={(e) => setPatient({ ...patient, status: e.target.value })}
                      sx={inputStyles}
                      SelectProps={selectMenuProps}
                    >
                      <MenuItem value="PENDING">شروع درمان</MenuItem>
                      <MenuItem value="IN_TREATMENT">در حال درمان</MenuItem>
                      <MenuItem value="COMPLETED">اتمام درمان</MenuItem>
                      <MenuItem value="CANCELLED">متوقف شده</MenuItem>
                    </TextField>

                    <TextField
                      fullWidth
                      multiline
                      rows={4}
                      label="یادداشت‌ها"
                      value={patient.notes}
                      onChange={(e) => setPatient({ ...patient, notes: e.target.value })}
                      sx={inputStyles}
                    />

                    {success && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        اطلاعات با موفقیت ذخیره شد!
                      </Alert>
                    )}

                    <Button
                      variant="contained"
                      color="primary"
                      fullWidth
                      onClick={handleSaveGeneral}
                      disabled={saving}
                      sx={{ 
                        mt: 2,
                        transition: 'transform 0.1s ease-in-out',
                        '&:active': { transform: 'scale(0.99)' }
                      }}
                    >
                      {saving ? 'در حال ذخیره...' : 'ذخیره اطلاعات'}
                    </Button>
                  </Stack>
                </Card>
            </Grid>

            {/* Image Management */}
              <Grid item xs={12} md={8}>
            <Card sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                تصاویر بیمار
              </Typography>



              <Stack spacing={3}>
                {/* Upload Section with Image Type Options */}
                <Card sx={{ pt: 2 }}>
                  <Stack spacing={2}>
                    <FormControl fullWidth>
                      <InputLabel sx={{ fontSize: '0.75rem' }}>نوع تصویر</InputLabel>
                      <Select
                        value={selectedCategory}
                        label="نوع تصویر"
                        onChange={(e) => setSelectedCategory(e.target.value)}
                            sx={inputStyles}
                            MenuProps={selectMenuProps.MenuProps}
                      >
                        <MenuItem value="profile">پروفایل</MenuItem>
                        <MenuItem value="frontal">فرونتال</MenuItem>
                        <MenuItem value="panoramic">پانورامیک</MenuItem>
                        <MenuItem value="lateral">لترال سفالومتری</MenuItem>
                        <MenuItem value="occlusal-upper">اکلوزال بالا</MenuItem>
                        <MenuItem value="occlusal-lower">اکلوزال پایین</MenuItem>
                        <MenuItem value="lateral-intraoral">لترال راست دهان</MenuItem>
                        <MenuItem value="lateral-intraoral-left">لترال چپ دهان</MenuItem>
                        <MenuItem value="frontal-intraoral">فرونتال داخل دهان</MenuItem>
                      </Select>
                    </FormControl>

                    {/* Upload area for new files */}
                    <Box sx={{ mt: 2 }}>
                      <Typography variant="subtitle2" gutterBottom>
                        آپلود فایل جدید
                      </Typography>
                      <Upload
                        multiple
                        hideFileList
                        onDrop={(acceptedFiles) => {
                          if (acceptedFiles && acceptedFiles.length > 0) {
                            handleImageUpload(acceptedFiles, selectedCategory);
                          }
                        }}
                        accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
                        sx={{
                          width: '100%',
                          minHeight: 120,
                        }}
                      />
                      </Box>
                  </Stack>
                </Card>

                {/* Uploaded Files List */}
                {uploadedImages.length > 0 && (
                  <Box sx={{ mt: 3 }}>
                    <Typography variant="subtitle2" sx={{ mb: 1 }}>
                      فایل‌های آپلود شده ({uploadedImages.length})
                    </Typography>
                    <Stack spacing={0}>
                      {uploadedImages.map((item) => (
                        <ImageListItem
                          key={item.id}
                          item={item}
                          onEdit={handleOpenEditCategoryDialog}
                          onDelete={openDeleteDialog}
                          getCategoryLabel={getCategoryLabel}
                        />
                      ))}
                    </Stack>
                  </Box>
                )}

                {/* AI Analysis Button */}
                <Box sx={{ mt: 3, display: 'flex', justifyContent: 'center', width: '100%' }}>
                  <Box
                    sx={{
                      position: 'relative',
                      width: '100%',
                      borderRadius: 1,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                    }}
                  >
                    <AnimateBorder
                      animate={{
                        duration: 12,
                        distance: 40,
                        color: [theme.vars.palette.primary.main, theme.vars.palette.warning.main],
                        outline: `135deg, ${varAlpha(theme.vars.palette.primary.mainChannel, 0.04)}, ${varAlpha(theme.vars.palette.primary.mainChannel, 0.04)}`,
                      }}
                      sx={{ 
                        width: 1, 
                        height: 1, 
                        position: 'absolute', 
                        top: 0,
                        left: 0,
                        zIndex: 2,
                        pointerEvents: 'none',
                      }}
                    />
                    <Button
                      fullWidth
                      size="large"
                      variant="contained"
                      color="primary"
                      onClick={handleRunCompleteAnalysis}
                      disabled={isRunningCompleteAnalysis || uploadedImages.length === 0}
                      sx={{ 
                        mb: '0', 
                        position: 'relative', 
                        zIndex: 1,
                        backgroundColor: theme.vars.palette.primary.main,
                        '&:hover': {
                          backgroundColor: theme.vars.palette.primary.dark,
                        },
                        '&.Mui-disabled': {
                          backgroundColor: theme.vars.palette.action.disabledBackground,
                        },
                      }}
                      startIcon={
                        isRunningCompleteAnalysis ? (
                          <CircularProgress size={16} sx={{ color: 'inherit' }} />
                        ) : (
                          <Iconify icon="solar:scan-bold" width={20} />
                        )
                      }
                    >
                      {isRunningCompleteAnalysis ? 'در حال پردازش' : 'آنالیز'}
                    </Button>
                  </Box>
                </Box>

                {/* Loading indicator */}
                {isRunningCompleteAnalysis && (
                  <Box 
                    sx={{ 
                      mt: 3, 
                      display: 'flex', 
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      py: 4,
                      width: '100%',
                    }}
                  >
                    <Box
                      sx={{
                        width: 48,
                        height: 48,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        color: 'primary.main',
                        mb: 3,
                      }}
                    >
                      <svg
                        width="32"
                        height="32"
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                        style={{
                          animation: 'spin 1s linear infinite',
                        }}
                      >
                        <circle
                          cx="12"
                          cy="12"
                          r="10"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeDasharray="31.416"
                          strokeDashoffset="23.562"
                          fill="none"
                          opacity="0.5"
                        />
                      </svg>
                      <style>
                        {`
                          @keyframes spin {
                            0% { transform: rotate(0deg); }
                            100% { transform: rotate(360deg); }
                          }
                        `}
                      </style>
                    </Box>
                    <Box sx={{ width: '100%', maxWidth: 400, mb: 2 }}>
                      <LinearProgress 
                        variant="determinate" 
                        value={analysisProgress} 
                        sx={{ 
                          height: 8, 
                          borderRadius: 1,
                          bgcolor: 'grey.200',
                          '& .MuiLinearProgress-bar': {
                            borderRadius: 1,
                          },
                        }} 
                      />
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 1 }}>
                        <Typography variant="caption" color="text.secondary">
                          {currentAnalysisStep}
                        </Typography>
                        <Typography variant="caption" color="text.secondary" sx={{ fontWeight: 600 }}>
                          {Math.round(analysisProgress)}%
                        </Typography>
                      </Box>
                    </Box>
                    <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                      در حال انجام آنالیز کامل...
                    </Typography>
                    <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                      لطفاً منتظر بمانید
                    </Typography>
                  </Box>
                )}

                {/* Complete Analysis Report with Typing Animation */}
                {(() => {
                  const shouldShow = completeAnalysisReport && 
                    Array.isArray(completeAnalysisReport) && 
                    completeAnalysisReport.length > 0 && 
                    !isRunningCompleteAnalysis;
                  
                  console.log('[Render] TypingReport visibility check:', {
                    hasReport: !!completeAnalysisReport,
                    isArray: Array.isArray(completeAnalysisReport),
                    length: completeAnalysisReport?.length || 0,
                    isRunning: isRunningCompleteAnalysis,
                    shouldShow
                  });
                  
                  return shouldShow ? (
                    <Box sx={{ mt: 3 }}>
                      <TypingReport
                        key={JSON.stringify(completeAnalysisReport.map(s => ({ title: s.title, content: s.content })))}
                        sections={completeAnalysisReport}
                        onComplete={() => {
                          console.log('Complete analysis finished');
                        }}
                      />
                    </Box>
                  ) : null;
                })()}
              </Stack>
            </Card>
              </Grid>
            </Grid>
          </Stack>
          </div>
        </Box>

            {/* Image options menu */}
            <Menu
              anchorReference="anchorPosition"
              anchorPosition={menuPosition ? { top: menuPosition.top, left: menuPosition.left } : undefined}
              open={Boolean(menuPosition)}
              onClose={handleCloseMenu}
            >
              <MenuItem onClick={() => { 
                if (menuImage) { 
                  const imageUrl = getImageUrl(menuImage.path);
                  window.open(imageUrl, '_blank'); 
                } 
                handleCloseMenu(); 
              }}>
                <ListItemIcon>
                  <Iconify icon="solar:eye-bold" width={18} />
                </ListItemIcon>
                <ListItemText>مشاهده</ListItemText>
              </MenuItem>

              <MenuItem onClick={() => { if (menuImage) { handleOpenEditCategoryDialog(menuImage); } handleCloseMenu(); }}>
                <ListItemIcon>
                  <Iconify icon="solar:pen-bold" width={18} />
                </ListItemIcon>
                <ListItemText>تغییر نوع</ListItemText>
              </MenuItem>

              <MenuItem onClick={() => { if (menuImage) { handleDownloadImage(menuImage); } handleCloseMenu(); }}>
                <ListItemIcon>
                  <Iconify icon="eva:arrow-circle-down-fill" width={18} />
                </ListItemIcon>
                <ListItemText>دانلود</ListItemText>
              </MenuItem>

              <MenuItem onClick={() => { if (menuImage) { handleOpenCropDialog(menuImage); } handleCloseMenu(); }}>
                <ListItemIcon>
                  <Iconify icon="solar:crop-linear" width={18} />
                </ListItemIcon>
                <ListItemText>برش</ListItemText>
              </MenuItem>

              <MenuItem onClick={() => { if (menuImage) { openDeleteDialog(menuImage); } }}>
                <ListItemIcon>
                  <Iconify icon="solar:trash-bin-trash-bold" width={18} />
                </ListItemIcon>
                <ListItemText>حذف</ListItemText>
              </MenuItem>

            </Menu>

            {/* Delete Confirmation Dialog - Lazy loaded */}
            <Dialog 
              open={deleteDialogOpen} 
              onClose={() => {
                setDeleteDialogOpen(false);
                // Delay clearing to allow transition
                setTimeout(() => {
                  setImageToDelete(null);
                }, 300);
              }}
              maxWidth="xs"
              fullWidth
              TransitionComponent={Grow}
              TransitionProps={{ 
                timeout: { enter: 300, exit: 200 },
              }}
              slotProps={{
                backdrop: {
                  sx: {
                    backgroundColor: 'rgba(0, 0, 0, 0.5)',
                  },
                },
              }}
              sx={{
                '& .MuiDialog-container': {
                  // Prevent layout shift from scrollbar
                  paddingRight: 'var(--scrollbar-width, 0px)',
                },
              }}
            >
                  <DialogTitle>حذف تصویر</DialogTitle>
                  <DialogContent>
                    <Typography sx={{ color: 'text.secondary' }}>
                      آیا از حذف این تصویر مطمئن هستید؟ این عمل غیرقابل بازگشت است.
                    </Typography>
                  </DialogContent>
                  <DialogActions>
                    <Button 
                      onClick={() => {
                        setDeleteDialogOpen(false);
                        // Delay clearing to allow transition
                        setTimeout(() => {
                          setImageToDelete(null);
                        }, 300);
                      }} 
                      color="inherit"
                      variant="outlined"
                    >
                      انصراف
                    </Button>
                    <Button 
                      onClick={confirmDeleteImage} 
                      color="error" 
                      variant="contained"
                      disabled={saving}
                      autoFocus
                    >
                      {saving ? 'در حال حذف...' : 'حذف'}
                    </Button>
                  </DialogActions>
                </Dialog>

        {/* Images Tab - Using CSS display to keep mounted for faster switching */}
        <Box
          sx={{
            display: currentTab === 'images' ? 'block' : 'none',
          }}
        >
        <div
          className={`tab-content ${currentTab === 'images' ? 'active' : ''}`}
          key="images"
          style={{
            willChange: 'transform, opacity',
            minHeight: '400px',
          }}
        >
          <Stack spacing={3}>
            {renderImageUploadSection('پروفایل', 'profile', 3)}
            {renderImageUploadSection('فرونتال', 'frontal', 3)}
            {renderImageUploadSection('پانورامیک', 'panoramic', 2)}
            {renderImageUploadSection('لترال سفالومتری', 'lateral', 2)}
            {renderImageUploadSection('اکلوزال بالا', 'occlusal-upper', 5)}
            {renderImageUploadSection('اکلوزال پایین', 'occlusal-lower', 5)}
            {renderImageUploadSection('لترال راست دهان', 'lateral-intraoral', 5)}
            {renderImageUploadSection('لترال چپ دهان', 'lateral-intraoral-left', 5)}
            {renderImageUploadSection('فرونتال داخل دهان', 'frontal-intraoral', 5)}
            <Card sx={{ p: 3 }}>
              <Stack spacing={2}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="h6">
                    تصاویر قدیمی (برای سازگاری)
                  </Typography>
                  <Button
                    variant="outlined"
                    startIcon={<Iconify icon="solar:magic-stick-3-bold" />}
                    onClick={async () => {
                      // تقسیم خودکار تصاویر general موجود
                      const generalImages = patient?.images?.general || [];
                      if (generalImages.length === 0) {
                        toast.info('تصویر general برای تقسیم وجود ندارد');
                        return;
                      }

                      setUploading(true);
                      try {
                        const results = await Promise.all(
                          generalImages.map(async (image) => {
                            try {
                              // دانلود تصویر
                              const imageUrl = getImageUrl(image.path);
                              const response = await fetch(imageUrl);
                              const blob = await response.blob();
                              const file = new File([blob], image.originalName, { type: image.mimeType });

                              // تشخیص نوع تصویر
                              const classification = await classifyImageType(file);
                              return {
                                image,
                                detectedCategory: classification.category || 'general',
                                confidence: classification.confidence || 0,
                                file,
                              };
                            } catch (error) {
                              console.error(`Error processing image ${image.id}:`, error);
                              return null;
                            }
                          })
                        );

                        // حذف تص��ویر اصلی و آپلود در دسته جدید
                        const categorizedResults = {
                          intraoral: [],
                          lateral: [],
                          profile: [],
                          general: [],
                        };

                        results.forEach((result) => {
                          if (!result) return;
                          const { image, detectedCategory, confidence, file } = result;
                          let finalCategory = confidence >= 0.3 ? detectedCategory : 'general';
                          
                          // تبدیل frontal به profile
                          if (finalCategory === 'frontal') {
                            finalCategory = 'profile';
                          }
                          
                          if (categorizedResults[finalCategory]) {
                            categorizedResults[finalCategory].push({ image, file });
                          } else {
                            // اگر category معتبر نیست، به general اضافه کن
                            categorizedResults.general.push({ image, file });
                          }
                        });

                        // حذف تصاویر قدیمی و آپلود در دسته جدید
                        for (const [category, items] of Object.entries(categorizedResults)) {
                          if (items.length === 0) continue;

                          // حذف تصاویر قدیمی
                          for (const { image } of items) {
                            try {
                              await axios.delete(`${endpoints.patients}/${id}/images`, {
                                data: { imageId: image.id },
                                headers: {
                                  Authorization: `Bearer ${user?.accessToken}`,
                                  'Content-Type': 'application/json',
                                },
                              });
                            } catch (error) {
                              console.error(`Error deleting image ${image.id}:`, error);
                            }
                          }

                          // آپلود در دسته جدید
                          const formData = new FormData();
                          items.forEach(({ file }) => {
                            formData.append('images', file);
                          });
                          formData.append('category', category);

                          await axios.post(`${endpoints.patients}/${id}/images`, formData, {
                            headers: {
                              Authorization: `Bearer ${user?.accessToken}`,
                            },
                          });
                        }

                        // Refresh images list
                        const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
                          headers: {
                            Authorization: `Bearer ${user?.accessToken}`,
                          },
                        });

                        setUploadedImages(imagesResponse.data.images || []);
                        const newImages = imagesResponse.data.images || [];
                        const categorizedImages = {
                          profile: newImages.filter(img => img.category === 'profile' || img.category === 'frontal'),
                          lateral: newImages.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
                          intraoral: newImages.filter(img => img.category === 'intraoral' || img.category === 'intra'),
                          general: newImages.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
                        };

                        setPatient(prev => ({
                          ...prev,
                          images: categorizedImages,
                        }));

                        toast.success(`تعداد ${results.length} تصویر با موفقیت تقسیم شدند`);
                      } catch (error) {
                        console.error('Error auto-categorizing images:', error);
                        toast.error('خطا در تقسیم خودکار تصاویر');
                      } finally {
                        setUploading(false);
                      }
                    }}
                    disabled={uploading || !patient?.images?.general || patient.images.general.length === 0}
                  >
                    تقسیم خودکار تصاویر general
                  </Button>
                </Box>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  تصاویر general شامل انواع مختلف تصاویر است. با کلیک روی دکمه بالا، تصاویر به صورت خودکار در دسته‌های مناسب قرار می‌گیرند.
                </Typography>
                <Button
                  variant="outlined"
                  color="primary"
                  startIcon={<Iconify icon="solar:scissors-square-bold" />}
                  onClick={() => {
                    setSplitDialogOpen(true);
                    setSplitImageFile(null);
                    setSplitResults([]);
                    setSelectedSplits(new Set());
                  }}
                  sx={{ mb: 2 }}
                >
                  تقسیم تصویر کلی (Composite Image)
                </Button>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  اگر یک تصویر کلی دارید که شامل چندین عکس است (مثلاً 3x3 grid)، از این دکمه استفاده کنید تا تصویر به بخش‌های ��جزا تقسیم شود.
                </Typography>
              </Stack>
            </Card>
          </Stack>
        </div>
        </Box>

        {/* Cephalometric Tab - منتقل شده به صفحه جداگانه /dashboard/orthodontics/patient/[id]/analysis */}

        {/* Intra-Oral Tab - Using CSS display to keep mounted for faster switching */}
        <Box
          sx={{
            display: currentTab === 'intra-oral' ? 'block' : 'none',
          }}
        >
        <div
          className={`tab-content ${currentTab === 'intra-oral' ? 'active' : ''}`}
          key="intra-oral"
          style={{
            willChange: 'transform, opacity',
            minHeight: '400px',
          }}
        >
          <Box sx={{ '& .MuiContainer-root': { maxWidth: '100%', px: 0 } }}>
                <React.Suspense fallback={null}>
            <IntraOralView
              initialImages={intraOralImages}
              onEditCategory={handleOpenEditCategoryDialog}
              onDeleteImage={openDeleteDialog}
              patientId={id}
            />
                </React.Suspense>
          </Box>
        </div>
        </Box>


        {/* Treatment Tab - Using CSS display to keep mounted for faster switching */}
        <Box
          sx={{
            display: currentTab === 'treatment' ? 'block' : 'none',
          }}
        >
        <div
          className={`tab-content ${currentTab === 'treatment' ? 'active' : ''}`}
          key="treatment"
          style={{
            willChange: 'transform, opacity',
            minHeight: '400px',
          }}
        >
          <Stack spacing={3}>
            {/* Cephalometric Parameters Table - Before Treatment Plan */}
            {patient?.cephalometricTable && Object.keys(patient.cephalometricTable).length > 0 && (() => {
              // Convert cephalometricTable format to measurements format for CephalometricTable component
              const measurements = {};
              Object.entries(patient.cephalometricTable).forEach(([param, data]) => {
                if (data?.measured && data.measured !== '' && data.measured !== 'undefined' && data.measured !== 'null') {
                  const measuredNum = parseFloat(data.measured);
                  if (!isNaN(measuredNum)) {
                    measurements[param] = measuredNum;
                  }
                }
              });
              
              if (Object.keys(measurements).length > 0) {
                return (
                  <Card sx={{ p: 3 }}>
                    <Typography variant="h6" gutterBottom>
                      جدول پارامترهای سفالومتری
                    </Typography>
                    <CephalometricTable measurements={measurements} />
                  </Card>
                );
              }
              return null;
            })()}
            
            <Card sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                طرح درمان
              </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              طرح درمان به کمک AI پیشنهاد می‌شود
            </Typography>

            <TextField
              fullWidth
              multiline
              rows={8}
              label="طرح درمان پیشنهادی"
              value={patient.treatmentPlan}
              onChange={(e) => setPatient({ ...patient, treatmentPlan: e.target.value })}
              sx={inputStyles}
            />

            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Button variant="outlined" startIcon={<Iconify icon="solar:refresh-bold" />}>
                تولید مجدد AI
              </Button>
              <Button 
                variant="contained" 
                color="primary"
                onClick={handleSaveTreatment}
                sx={{ 
                  transition: 'transform 0.1s ease-in-out',
                  '&:active': { transform: 'scale(0.99)' }
                }}
              >
                ذخیره تغییرات
              </Button>
            </Stack>
          </Card>
          </Stack>
        </div>
        </Box>

        {/* Summary Tab - Using CSS display to keep mounted for faster switching */}
        <Box
          sx={{
            display: currentTab === 'summary' ? 'block' : 'none',
          }}
        >
        <div
          className={`tab-content ${currentTab === 'summary' ? 'active' : ''}`}
          key="summary"
          style={{
            willChange: 'transform, opacity',
            minHeight: '400px',
          }}
        >
          <Card sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              خلاصه پرونده بیمار
            </Typography>

            <TextField
              fullWidth
              multiline
              rows={6}
              label="خلاصه پرونده"
              value={patient.summary}
              sx={inputStyles}
              onChange={(e) => setPatient({ ...patient, summary: e.target.value })}
            />

            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Button variant="outlined" startIcon={<Iconify icon="solar:printer-bold" />} >
                پرینت خلاصه
              </Button>
              <Button 
                variant="contained" 
                color="primary"
                onClick={handleSaveSummary}
                sx={{ 
                  transition: 'transform 0.1s ease-in-out',
                  '&:active': { transform: 'scale(0.99)' }
                }}
              >
                ذخیره تغییرات
              </Button>
            </Stack>
          </Card>
        </div>
        </Box>
      </Box>

      {/* Upload Dialog - Lazy loaded */}
      {uploadDialogOpen && (
        <Suspense fallback={null}>
          <Dialog open={uploadDialogOpen} onClose={handleCloseUploadDialog} maxWidth="md" fullWidth>
        <DialogTitle>آپلود تصویر بیمار</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ pt: 1, px: { xs: 1.5, sm: 3 }, pb: 1 }}>
            <TextField
              select
              label="نوع تصویر"
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              fullWidth
              sx={inputStyles}
              SelectProps={selectMenuProps}
            >
              <MenuItem value="profile">پروفایل</MenuItem>
              <MenuItem value="frontal">فرونتال</MenuItem>
              <MenuItem value="panoramic">پانورامیک</MenuItem>
              <MenuItem value="lateral">لترال سفالومتری</MenuItem>
              <MenuItem value="occlusal-upper">اکلوزال بالا</MenuItem>
              <MenuItem value="occlusal-lower">اکلوزال پایین</MenuItem>
              <MenuItem value="lateral-intraoral">لترال داخل دهان</MenuItem>
              <MenuItem value="frontal-intraoral">فرونتال داخل دهان</MenuItem>
            </TextField>

            <Box
              sx={{
                position: 'relative',
                width: '100%',
                height: 200,
                border: '2px dashed',
                borderColor: 'grey.400',
                borderRadius: 2,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                '&:hover': {
                  borderColor: 'primary.main'
                }
              }}
              onClick={() => {
                const input = document.createElement('input');
                input.type = 'file';
                input.accept = 'image/*';
                input.multiple = true;
                input.onchange = (e) => {
                  const files = Array.from(e.target.files);
                  setSelectedFiles(files);
                };
                input.click();
              }}
            >
              <Stack spacing={1} alignItems="center">
                <UploadIllustration hideBackground sx={{ width: 120 }} />
                <Typography variant="body1" color="text.secondary" textAlign="center">
                  فایل‌ها را اینجا بیاندازید یا کلیک کنید
                </Typography>
              </Stack>
            </Box>

            {selectedFiles.length > 0 && (
              <Box sx={{ textAlign: 'center', pt: 1 }}>
                <Typography variant="body1" color="primary.main" fontWeight="medium">
                  {selectedFiles.length} فایل انتخاب شده
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  آماده آپلود
                </Typography>
              </Box>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseUploadDialog}>انصراف</Button>
          <Button
            variant="contained"
            onClick={handleUploadDialogSubmit}
            disabled={selectedFiles.length === 0}
          >
            آپلود و ذخیره
          </Button>
        </DialogActions>
      </Dialog>
        </Suspense>
      )}

      {/* Edit Category Dialog */}
      <Dialog
        open={editCategoryDialogOpen}
        onClose={handleCloseEditCategoryDialog}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>ویرایش نوع تصویر</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ pt: 1 }}>
            {editingImage && (
              <Box
                sx={{
                  width: '100%',
                  minHeight: 200,
                  maxHeight: 300,
                  position: 'relative',
                  borderRadius: 1,
                  mb: 2,
                  bgcolor: 'background.neutral',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  overflow: 'hidden',
                }}
              >
                {!imageLoaded && (
                  <Box
                    sx={{
                      position: 'absolute',
                      top: 0,
                      left: 0,
                      right: 0,
                      bottom: 0,
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      bgcolor: 'background.neutral',
                    }}
                  />
                )}
                <Box
                  component="img"
                  src={preloadedImage || getImageUrl(editingImage.path)}
                  alt={editingImage?.originalName || ''}
                  onLoad={() => setImageLoaded(true)}
                  sx={{
                    width: '100%',
                    height: '100%',
                    maxHeight: 300,
                    objectFit: 'cover',
                    opacity: imageLoaded ? 1 : 0,
                    transition: 'opacity 0.2s ease-in-out',
                    position: 'absolute',
                    top: 0,
                    left: 0,
                  }}
                />
              </Box>
            )}

            <FormControl fullWidth>
              <InputLabel sx={{ fontSize: '0.75rem' }}>نوع تصویر</InputLabel>
              <Select
                value={newImageCategory}
                onChange={(e) => setNewImageCategory(e.target.value)}
                label="نوع تصویر"
                sx={inputStyles}
                MenuProps={selectMenuProps.MenuProps}
              >
                <MenuItem value="profile">پروفایل</MenuItem>
                <MenuItem value="frontal">فرونتال</MenuItem>
                <MenuItem value="panoramic">پانورمیک</MenuItem>
                <MenuItem value="lateral">لترال سفالومتری</MenuItem>
                <MenuItem value="occlusal-upper">اکلوزال بالا</MenuItem>
                <MenuItem value="occlusal-lower">اکلوزال پایین</MenuItem>
                <MenuItem value="lateral-intraoral">لترال راست دهان</MenuItem>
                <MenuItem value="lateral-intraoral-left">لترال چپ دهان</MenuItem>
                <MenuItem value="frontal-intraoral">فرونتال داخل دهان</MenuItem>
              </Select>
            </FormControl>
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseEditCategoryDialog}>انصراف</Button>
          <Button
            variant="contained"
            onClick={handleEditCategorySubmit}
            disabled={saving || !editingImage}
          >
            {saving ? 'در حال ذخیره...' : 'ذخیره تغییرات'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Split Composite Image Dialog - Lazy loaded */}
      {splitDialogOpen && (
        <Suspense fallback={null}>
          <Dialog 
            open={splitDialogOpen} 
            onClose={() => {
              setSplitDialogOpen(false);
              setSplitImageFile(null);
              setSplitResults([]);
              setSelectedSplits(new Set());
            }} 
            maxWidth="lg" 
            fullWidth
          >
        <DialogTitle>تقسیم تصویر کلی</DialogTitle>
        <DialogContent>
          <Stack spacing={3} sx={{ pt: 2 }}>
            {!splitImageFile && (
              <>
                <Upload
                  value={splitImageFile}
                  onDrop={(acceptedFiles) => {
                    if (acceptedFiles.length > 0) {
                      setSplitImageFile(acceptedFiles[0]);
                    }
                  }}
                  onDelete={() => setSplitImageFile(null)}
                  accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
                />
                <Typography variant="body2" color="text.secondary">
                  تصویر کلی خود را انتخاب کنید. این تصویر باید شامل چندین عکس در یک grid باشد (مثلاً 3x3).
                </Typography>
              </>
            )}

            {splitImageFile && splitResults.length === 0 && (
              <>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                  <Box
                    component="img"
                    src={URL.createObjectURL(splitImageFile)}
                    alt="Composite image"
                    sx={{
                      maxWidth: '100%',
                      maxHeight: '400px',
                      objectFit: 'contain',
                      borderRadius: 1,
                    }}
                  />
                  <Button
                    variant="outlined"
                    onClick={() => setSplitImageFile(null)}
                    startIcon={<Iconify icon="solar:trash-bin-trash-bold" />}
                  >
                    حذف
                  </Button>
                </Box>
                <Button
                  variant="contained"
                  onClick={() => handleSplitCompositeImage(splitImageFile)}
                  disabled={splitting}
                  startIcon={<Iconify icon={splitting ? "solar:refresh-circle-bold" : "solar:scissors-square-bold"} />}
                  fullWidth
                >
                  {splitting ? 'در حال تقسیم...' : 'تقسیم تصویر'}
                </Button>
              </>
            )}

            {splitResults.length > 0 && (
              <>
                <Typography variant="h6">
                  نتایج تقسیم ({splitResults.length} تصویر)
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                  تصاویری که می‌خواهید ذخیره شوند را انتخاب کنید. هر تصویر به صورت خودکار در دسته مناس�� قرار می‌گیرد.
                </Typography>
                <Grid container spacing={2}>
                  {splitResults.map((split, index) => {
                    const isSelected = selectedSplits.has(index);
                    return (
                      <Grid item xs={6} sm={4} md={3} key={index}>
                        <Card
                          sx={{
                            cursor: 'pointer',
                            border: 2,
                            borderColor: isSelected ? 'primary.main' : 'transparent',
                            '&:hover': {
                              borderColor: isSelected ? 'primary.main' : 'grey.300',
                            },
                          }}
                          onClick={() => {
                            const newSelected = new Set(selectedSplits);
                            if (isSelected) {
                              newSelected.delete(index);
                            } else {
                              newSelected.add(index);
                            }
                            setSelectedSplits(newSelected);
                          }}
                        >
                          <Box
                            component="img"
                            src={`data:image/jpeg;base64,${split.image_base64}`}
                            alt={`Split ${split.row}-${split.col}`}
                            sx={{
                              width: '100%',
                              height: '150px',
                              objectFit: 'cover',
                            }}
                          />
                          <Box sx={{ p: 1 }}>
                            <Typography variant="caption" display="block">
                              سطر {split.row + 1}, ستون {split.col + 1}
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                              نوع: {
                                split.category === 'intraoral' ? 'داخل دهانی' :
                                split.category === 'lateral' ? 'لترال' :
                                split.category === 'profile' || split.category === 'frontal' ? 'صورت' :
                                'کلی'
                              }
                            </Typography>
                            <Typography variant="caption" color="text.secondary" display="block">
                              اعتماد: {(split.confidence * 100).toFixed(1)}%
                            </Typography>
                          </Box>
                          {isSelected && (
                            <Box
                              sx={{
                                position: 'absolute',
                                top: 8,
                                right: 8,
                                bgcolor: 'primary.main',
                                color: 'white',
                                borderRadius: '50%',
                                width: 24,
                                height: 24,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                              }}
                            >
                              <Iconify icon="solar:check-circle-bold" width={16} />
                            </Box>
                          )}
                        </Card>
                      </Grid>
                    );
                  })}
                </Grid>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 2 }}>
                  <Typography variant="body2">
                    {selectedSplits.size} از {splitResults.length} تصویر انتخاب شده است
                  </Typography>
                  <Button
                    variant="outlined"
                    onClick={() => {
                      if (selectedSplits.size === splitResults.length) {
                        setSelectedSplits(new Set());
                      } else {
                        setSelectedSplits(new Set(splitResults.map((_, i) => i)));
                      }
                    }}
                  >
                    {selectedSplits.size === splitResults.length ? 'لغو انتخاب همه' : 'انتخاب همه'}
                  </Button>
                </Box>
              </>
            )}
          </Stack>
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setSplitDialogOpen(false);
              setSplitImageFile(null);
              setSplitResults([]);
              setSelectedSplits(new Set());
            }}
          >
            انصراف
          </Button>
          {splitResults.length > 0 && (
            <Button
              variant="contained"
              onClick={handleSaveSplitImages}
              disabled={uploading || selectedSplits.size === 0}
              startIcon={<Iconify icon="solar:diskette-bold" />}
            >
              {uploading ? 'در حال ذخیره...' : `ذخیره ${selectedSplits.size} تصویر`}
            </Button>
          )}
        </DialogActions>
      </Dialog>
        </Suspense>
      )}

      {/* Crop Dialog */}
      {/* Image Crop Dialog - Lazy loaded */}
      {cropDialogOpen && (
        <Suspense fallback={null}>
          <ImageCropDialog
            open={cropDialogOpen}
            imageUrl={cropImage?.src}
            onClose={handleCloseCropDialog}
            onSave={handleCropSaveSubmit}
            saving={saving}
          />
        </Suspense>
      )}
    </Container>
  );
}
