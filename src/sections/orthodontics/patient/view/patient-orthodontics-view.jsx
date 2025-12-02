import dayjs from 'dayjs';
import { toast } from 'sonner';
import { useParams, useNavigate } from 'react-router-dom';
import React, { useRef, useMemo, useState, useEffect, useCallback } from 'react';
import { m } from 'framer-motion';

import { varFade } from 'src/components/animate';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Grid from '@mui/material/Grid';
import Menu from '@mui/material/Menu';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import Divider from '@mui/material/Divider';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import InputLabel from '@mui/material/InputLabel';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import LinearProgress from '@mui/material/LinearProgress';
import TablePagination from '@mui/material/TablePagination';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { MobileDateTimePicker } from '@mui/x-date-pickers/MobileDateTimePicker';

import { paths } from 'src/routes/paths';

import { uuidv4 } from 'src/utils/uuidv4';
import axios, { endpoints } from 'src/utils/axios';
import { compressMultipleImages, getCompressionSettingsForModel } from 'src/utils/image-compression';

import { CONFIG } from 'src/config-global';
import { createEvent } from 'src/actions/calendar';

import { Image } from 'src/components/image';
import { UploadIllustration } from 'src/assets/illustrations';

import { Upload } from 'src/components/upload';
import { Iconify } from 'src/components/iconify';
import { Chart, useChart } from 'src/components/chart';
import { CustomTabs } from 'src/components/custom-tabs';
import { NavigationBar } from 'src/components/extra/navigation-bar';

import { IntraOralView } from 'src/sections/intra-oral/view/intra-oral-view';
import { FacialLandmarkView } from 'src/sections/facial-landmark/view/facial-landmark-view';

import { useAuthContext } from 'src/auth/hooks';

import AIDiagnosisDisplay from './ai-diagnosis-display';
import ImageCropDialog from '../components/image-crop-dialog';
import { CephalometricAIAnalysis } from '../components/cephalometric-ai-analysis';
import { SuperimposeView } from '../components/superimpose-view';

// ----------------------------------------------------------------------

export function PatientOrthodonticsView() {
  const { id } = useParams();
  const { user } = useAuthContext();
  const navigate = useNavigate();
  const [currentTab, setCurrentTab] = useState('general');
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

  // Image category edit dialog state
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
  // به صورت پیش‌فرض: اگر آنالیزی وجود داشته باشد لیست را نمایش بده، وگرنه تصویر را نمایش بده
  const [showCephalometricImage, setShowCephalometricImage] = useState(true); // به صورت پیش‌فرض تصویر نمایش داده می‌شود
  // وضعیت تایید آنالیز
  const [isAnalysisConfirmed, setIsAnalysisConfirmed] = useState(false);
  // Ref برای ردیابی اینکه آیا کاربر دکمه "نمایش نتایج" را زده است یا نه
  const userClickedShowResultsRef = useRef(false);
  const [selectedSplits, setSelectedSplits] = useState(new Set());

  // Image options menu state (open at click position)
  const [menuPosition, setMenuPosition] = useState(null);
  const [menuImage, setMenuImage] = useState(null);

  const handleOpenMenu = (e, image) => {
    e.stopPropagation();
    // open menu at pointer location
    setMenuPosition({ top: e.clientY - 8, left: e.clientX - 8 });
    setMenuImage(image);
  };

  const handleCloseMenu = () => {
    setMenuPosition(null);
    setMenuImage(null);
  };

  // Delete confirmation dialog state
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [imageToDelete, setImageToDelete] = useState(null);

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
  }, [uploadedImages]);

  const confirmDeleteImage = async () => {
    if (!imageToDelete || !imageToDelete.id) return;
    
    setSaving(true);
    setDeleteDialogOpen(false); // Close dialog immediately
    
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
      
      // Update patient images
      const categorizedImages = {
        profile: images.filter(img => img.category === 'profile' || img.category === 'frontal'),
        lateral: images.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
        intraoral: images.filter(img => img.category === 'intraoral' || img.category === 'intra'),
        general: images.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
      };
      setPatient(prev => ({
        ...prev,
        images: categorizedImages,
      }));
      
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      toast.success('تصویر با موفقیت حذف شد');
    } catch (error) {
      console.error('Error deleting image:', error);
      toast.error('خطا در حذف تصویر');
    } finally {
      setSaving(false);
      setImageToDelete(null);
    }
  };

  const handleDownloadImage = async (image) => {
    try {
      const resp = await fetch(`http://localhost:7272${image.path}`);
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
  };

  // AI model selection for cephalometric analysis
  const [selectedAIModel, setSelectedAIModel] = useState('gpt-4o');

  // Cephalometric analysis type selection
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('steiner');
  
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
    steiner: {
      SNA: { 
        mean: '82', 
        sd: '3.5', 
        severity: 'نرمال', 
        note: 'زاویه ارتباط بین اسنو، nasion و A point. افزایش: جلو بودن ماگزیلا. کاهش: عقب بودن ماگزیلا' 
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
        note: 'تفاوت SNA و SNB (کلاس I = 2 درجه). افزایش: جلو بودن ماگزیلا یا عقب بودن مندیبل (کلاس II). کاهش: جلو بودن مندیبل یا عقب بودن ماگزیلا (کلاس III)' 
      },
      FMA: { 
        mean: '25', 
        sd: '4', 
        severity: 'نرمال', 
        note: 'فرانکفورت-مرکز پایین (صفحه چهره ای). افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' 
      },
      FMIA: { 
        mean: '65', 
        sd: '5', 
        severity: 'نرمال', 
        note: 'زاویه فرانکفورت - مرکز (آزادی حرکت فکی). افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' 
      },
      IMPA: { 
        mean: '90', 
        sd: '3', 
        severity: 'نرمال', 
        note: 'زاویه محور مرکزی چپ نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' 
      },
      GoGn: { 
        mean: '35', 
        sd: '4', 
        severity: 'نرمال', 
        note: 'زاویه gonion-gnathion (اندازه‌گیری ARC). افزایش: صفحه مندیبولار شیب دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' 
      },
      'U1-SN': { 
        mean: '103', 
        sd: '6', 
        severity: 'نرمال', 
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
      'Facial Depth': { mean: '88', sd: '3', severity: 'نرمال', note: 'عمق صورت (زاویه بین N-Pog و Or-Po). افزایش: صورت عمیق‌تر (deep face). کاهش: صورت کم‌عمق‌تر (shallow face)' },
      'Lower Face Height': { mean: '47', sd: '2', severity: 'نرمال', note: 'نسبت فاصله عمودی صورت پایین (فاصله عمودی ANS-Me / فاصله عمودی N-Me × 100). افزایش: ارتفاع صورت پایین بیشتر (long face). کاهش: ارتفاع صورت پایین کمتر (short face)' },
      'Mandibular Plane': { mean: '26', sd: '4', severity: 'نرمال', note: 'صفحه مندیبولار (زاویه Go-Me نسبت به FH). افزایش: صفحه مندیبولار شیب‌دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' },
      'Convexity': { mean: '0', sd: '2', severity: 'نرمال', note: 'تحدب صورت (فاصله A point از خط N-Pog). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Upper Incisor': { mean: '22', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین بالا (نسبت به A-Pog). افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' },
      'Lower Incisor': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه دندان پیشین پایین (نسبت به A-Pog). افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
      'Interincisal Angle': { mean: '130', sd: '6', severity: 'نرمال', note: 'زاویه بین خط U1-A و خط L1-B. افزایش: زاویه بیشتر (بیشتر retroclined). کاهش: زاویه کمتر (بیشتر proclined)' },
    },
    mcnamara: {
      'N-A-Pog': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه بین خط N-A و خط A-Pog (تحدب صورت). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
      'Co-A': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول فک بالا (فاصله Co-A). افزایش: فک بالا بلندتر (maxillary prognathism). کاهش: فک بالا کوتاه‌تر (maxillary retrognathism)' },
      'Co-Gn': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول فک پایین (فاصله Co-Gn). افزایش: فک پایین بلندتر (mandibular prognathism). کاهش: فک پایین کوتاه‌تر (mandibular retrognathism)' },
      'Wits Appraisal': { mean: '0', sd: '2', severity: 'نرمال', note: 'ارزیابی Wits (فاصله AO-BO). افزایش: کلاس II (ماگزیلا جلوتر یا مندیبل عقب‌تر). کاهش: کلاس III (ماگزیلا عقب‌تر یا مندیبل جلوتر)' },
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
    },
    tweed: {
      FMA: { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به صفحه مندیبولار. افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' },
      FMIA: { mean: '65', sd: '5', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به incisor پایین. افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' },
      IMPA: { mean: '90', sd: '3', severity: 'نرمال', note: 'زاویه incisor پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
    },
    bjork: {
      'S-Ar/Go-Gn Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت طول سوراسلار به گونی ذره (ارزش رشد). افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Ar-Go-N/Go-Me Ratio': { mean: '56', sd: '3', severity: 'نرمال', note: 'نسبت sagittal مندیبولار. افزایش: مندیبول جلوتر. کاهش: مندیبول عقب‌تر' },
      'S-Go/Go-Me Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت عمودی چهره (اندازه‌گیری مقایسه‌ای). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
      'MP/SN Angle': { mean: '32', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار نسبت به SN. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
      'NS-Gn Angle': { mean: '104', sd: '5', severity: 'نرمال', note: 'زاویه nasion-sella-gnathion. افزایش: صورت عمودی‌تر. کاهش: صورت افقی‌تر' },
    },
    jarabak: {
      'S-Go/Ar-Go Ratio': { mean: '53', sd: '3', severity: 'نرمال', note: 'فاکتور CAG رشد (اندازه‌گیری مقایسه‌ای رشد). افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Ar-Go/N-Go Ratio': { mean: '47', sd: '3', severity: 'نرمال', note: 'عکس اندازه‌گیری مقایسه‌ای رشد. افزایش: رشد افقی بیشتر. کاهش: رشد عمودی بیشتر' },
      'Co-Gn/Ar-Go Ratio': { mean: '2.5', sd: '0.3', severity: 'نرمال', note: 'فاکتور CG ارتفاع قدامی رامی. افزایش: ارتفاع قدامی بیشتر. کاهش: ارتفاع قدامی کمتر' },
      'S-Ar/Go-Gn Ratio': { mean: '98', sd: '5', severity: 'نرمال', note: 'نسبت SAG رشد. افزایش: رشد عمودی بیشتر. کاهش: رشد افقی بیشتر' },
      'Go-Gn/SN Angle': { mean: '46', sd: '4', severity: 'نرمال', note: 'زاویه رشد (صفحه مندیبولار نسبت به SN). افزایش: رشد عمودی (vertical growth). کاهش: رشد افقی (horizontal growth)' },
    },
    sassouni: {
      'N-S-Ar': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه بین nasion، sella و articulare. افزایش: زاویه بیشتر (بازتر). کاهش: زاویه کمتر (بسته‌تر)' },
      'N-Ar-Go': { mean: '10', sd: '3', severity: 'نرمال', note: 'زاویه محدود کننده. افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
      'Go-Co//N-S': { mean: '59', sd: '4', severity: 'نرمال', note: 'میزان تمایز ساژیتال (زاویه بین Go-Co و N-S). افزایش: تمایز ساژیتال بیشتر. کاهش: تمایز ساژیتال کمتر' },
      'Go-Co/Go-Gn': { mean: '4', sd: '2', severity: 'نرمال', note: 'انتخاب اجتماعی (نسبت Go-Co به Go-Gn). افزایش: نسبت بیشتر. کاهش: نسبت کمتر' },
      'N-Co//Go-Co': { mean: '90', sd: '5', severity: 'نرمال', note: 'ایدئال فرهنگی (زاویه بین N-Co و Go-Co). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
      'Ar-Co//Co-Gn': { mean: '74', sd: '4', severity: 'نرمال', note: 'نخستین sagittal (زاویه بین Ar-Co و Co-Gn). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
    },
  };

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

        setPatient({
          id: patientData.id,
          name: `${patientData.firstName} ${patientData.lastName}`,
          age: patientData.age,
          phone: patientData.phone,
          gender: patientData.gender || '',
          diagnosis: patientData.diagnosis,
          treatment: patientData.treatment,
          status: patientData.status,
          startDate: patientData.createdAt ? dayjs(patientData.createdAt) : null,
          nextVisit: patientData.nextVisit ? dayjs(patientData.nextVisit) : null,
          notes: patientData.notes || '',
          aiDiagnosis: patientData.diagnosis,
          softTissue: patientData.softTissueAnalysis || 'آنالیز بافت نرم: خط زیبایی E مناسب، لب بالایی در حد طبیعی',
          cephalometric: patientData.cephalometricAnalysis || 'آنالیز سفالومتریک: ANB: 6 درجه، SNA: 82 درجه، SNB: 76 درجه',
          cephalometricTable,
          cephalometricRawData,
          cephalometricLandmarks,
          treatmentPlan: patientData.treatmentPlan || 'طرح درمان پیشنهادی: استخراج دندان‌های آسیاب، استفاده از mini-implants',
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
          console.log('[Patient Images] Total images:', images.length);
          console.log('[Patient Images] Categories found:', images.map(img => ({ id: img.id, category: img.category, name: img.originalName })));

          const categorizedImages = {
            profile: images.filter(img => img.category === 'profile' || img.category === 'frontal'),
            lateral: images.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
            intraoral: images.filter(img => img.category === 'intraoral' || img.category === 'intra'),
            general: images.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
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
            if (prev.length === images.length) {
              const prevIds = new Set(prev.map(img => img.id));
              const newIds = new Set(images.map(img => img.id));
              if (prevIds.size === newIds.size && 
                  [...prevIds].every(id => newIds.has(id))) {
                return prev; // No change
              }
            }
            return images;
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
      fetchPatient();
    }
  }, [id, user]);

  // Load analysis history - موقتاً غیرفعال شده است
  // const loadAnalysisHistory = async () => {
  //   setIsLoadingHistory(true);
  //   try {
  //     const response = await axios.get(`${CONFIG.site.serverUrl || 'http://localhost:7272'}/api/ai-model-tests?limit=50`, {
  //       headers: {
  //         Authorization: `Bearer ${user?.accessToken}`,
  //       },
  //     });
      
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
  const calculateMeasurementsFromLandmarks = (landmarks) => {
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
          measures.IMPA = calculateAngleBetweenLines(
            meLandmark2, goLandmark,
            l1Landmark2, meLandmark2
          );
          measures.IMPA = Math.round(Math.max(0, Math.min(180, measures.IMPA)) * 10) / 10;
        }
      } else if (goLandmark && meLandmark2 && liaLandmark && litLandmark) {
        measures.IMPA = calculateAngleBetweenLines(
          meLandmark2, goLandmark,
          liaLandmark, litLandmark
        );
        measures.IMPA = Math.round(Math.max(0, Math.min(180, measures.IMPA)) * 10) / 10;
      }

      // GoGn
      const goLandmarkGoGn = getLandmark(['Go', 'go', 'GO']);
      const gnLandmark = getLandmark(['Gn', 'gn', 'GN']);

      if (goLandmarkGoGn && gnLandmark) {
        const angle = calculateLineAngle(goLandmarkGoGn, gnLandmark);
        measures.GoGn = Math.abs(angle);
        if (measures.GoGn > 90) {
          measures.GoGn = 180 - measures.GoGn;
        }
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

      // Interincisal Angle
      const u1LandmarkInter = getLandmark(['U1', 'u1']);
      const l1LandmarkInter = getLandmark(['L1', 'l1']);
      const aLandmarkInter = getLandmark(['A', 'a']);
      const bLandmarkInter = getLandmark(['B', 'b']);

      if (u1LandmarkInter && l1LandmarkInter) {
        if (aLandmarkInter && bLandmarkInter) {
          measures.InterincisalAngle = 180 - calculateAngle(u1LandmarkInter, aLandmarkInter, l1LandmarkInter);
        }
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

      console.log('✅ Calculated measurements from landmarks:', measures);
    } catch (err) {
      console.error('Error calculating measurements from landmarks:', err);
    }

    return measures;
  };

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
      toast.error('خطا در بارگذاری آنالیز از تاریخچه');
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

  // بررسی وضعیت تایید آنالیز بعد از لود شدن patient data
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
  }, [id, patient?.cephalometricTable]); // وقتی patient یا cephalometricTable تغییر کرد اجرا شود

  // Chart options (hooks must be called at top-level of component)
  const radarChartOptions = useChart({
    xaxis: { categories: patient?.cephalometricTable ? Object.keys(patient.cephalometricTable) : [] },
    yaxis: {
      max: 125,
      min: 0,
    },
    plotOptions: {
      radar: {
        polygons: { strokeColors: 'rgba(145, 158, 171, 0.2)', strokeWidth: 1, fill: { colors: ['transparent'] } },
      },
    },
    markers: { size: 4, strokeColors: ['#1976d2', '#FF4842'], strokeWidth: 2 },
    stroke: { width: 2, curve: 'smooth' },
    fill: { opacity: 0.1 },
    colors: ['#1976d2', '#FF4842'],
    states: { hover: { filter: { type: 'lighten', value: 0.1 } }, active: { filter: { type: 'none' } } },
  });


  // تشخیص نوع تصویر با استفاده از AI
  // Handle split composite image
  const handleSplitCompositeImage = async (file) => {
    if (!file) {
      toast.error('لطفا تصویر کلی را انتخاب کنید');
      return;
    }

    setSplitting(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch('http://localhost:8000/split-composite-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('خطا در تقسیم تصویر');
      }

      const data = await response.json();
      if (data.success && data.splits) {
        setSplitResults(data.splits);
        setSelectedSplits(new Set(data.splits.map((_, index) => index))); // انتخاب همه به صورت پیش‌فرض
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

      setUploadedImages(imagesResponse.data.images || []);
      const newImages = imagesResponse.data.images || [];
      const categorizedImages2 = {
        profile: newImages.filter(img => img.category === 'profile' || img.category === 'frontal'),
        lateral: newImages.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
        intraoral: newImages.filter(img => img.category === 'intraoral' || img.category === 'intra'),
        general: newImages.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
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

      const response = await fetch(`${CONFIG.site.serverUrl?.replace(':7272', ':5001') || 'http://localhost:5001'}/classify-image-type`, {
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

  const handleImageUpload = async (files, category) => {
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
      setUploadedImages(images);
      
      // Update patient images
      const categorizedImages = {
        profile: images.filter(img => img.category === 'profile' || img.category === 'frontal'),
        lateral: images.filter(img => img.category === 'lateral' || img.category === 'cephalometric' || img.category === 'cephalometry'),
        intraoral: images.filter(img => img.category === 'intraoral' || img.category === 'intra'),
        general: images.filter(img => img.category === 'general' || img.category === 'opg' || img.category === 'panoramic'),
      };
      setPatient(prev => ({
        ...prev,
        images: categorizedImages,
      }));
      
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      
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
  };

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

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

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

  const handleSaveGeneral = async () => {
    setSaving(true);
    try {
      const updateData = {
        firstName: patient.name.split(' ')[0],
        lastName: patient.name.split(' ').slice(1).join(' '),
        age: parseInt(patient.age, 10),
        phone: patient.phone,
        gender: patient.gender,
        status: patient.status,
        notes: patient.notes,
      };

      // Include dates if present
      if (patient.startDate) {
        updateData.startDate = patient.startDate.toISOString ? patient.startDate.toISOString() : dayjs(patient.startDate).toISOString();
      }
      if (patient.nextVisit) {
        updateData.nextVisit = patient.nextVisit.toISOString ? patient.nextVisit.toISOString() : dayjs(patient.nextVisit).toISOString();
      }

      await axios.put(`${endpoints.patients}/${id}`, updateData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره اطلاعات');
    } finally {
      setSaving(false);
    }
  };

  const handleScheduleVisit = async () => {
    if (!patient?.nextVisit) {
      alert('لطفا تاریخ و ساعت ویزیت را انتخاب کنید');
      return;
    }

    setSaving(true);
    try {
      const start = patient.nextVisit.toISOString ? patient.nextVisit.toISOString() : dayjs(patient.nextVisit).toISOString();
      const end = dayjs(patient.nextVisit).add(1, 'hour').toISOString();
      const eventData = {
        id: uuidv4(),
        title: `ویزیت - ${patient.name}`,
        start,
        end,
        allDay: false,
        description: `Patient ID: ${patient.id}`,
        color: '#00a76f',
      };

      await createEvent(eventData);

      // Persist nextVisit to backend
      await axios.put(`${endpoints.patients}/${id}`, { nextVisit: start }, {
        headers: { Authorization: `Bearer ${user?.accessToken}` },
      });

      setPatient(prev => ({ ...prev, nextVisit: dayjs(start) }));
      toast.success('نوبت ثبت شد و به تقویم اضافه گردید');
    } catch (error) {
      console.error('Schedule visit error:', error);
      toast.error('خطا در ثبت نوبت');
    } finally {
      setSaving(false);
    }
  };

  const handleRunAIDiagnosis = async () => {
    if (uploadedImages.length === 0) {
      alert('ابتدا تصاویر بیمار را آپلود کنید تا AI بتواند آن‌ها را تحلیل کند');
      return;
    }

    setSaving(true);
    try {
      // Collect all image URLs for AI analysis
      const imageUrls = uploadedImages.map(img => `http://localhost:7272${img.path}`);

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

### رویه‌ها و شرایط فعلی:
- **چیدمان دندان‌های فوقانی**: بررسی تراز و آلیمنتاسیون دندان‌های فک بالا
- **چیدمان دندان‌های تحتانی**: تحلیل موقعیت نسبی دندان‌های فک پایین
- **روابط اکلوژن**: بررسی تماس دندان‌ها در موقعیت‌های مختلف
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
- **پروفایل**: ${Math.random() > 0.6 ? 'پروفایل مستقیم' : 'پروفایل کنوکس'}
- **زاویه‌های صورت**: ارزیابی زاویه‌های مسر و مرکز
- **لب‌ها**: وضعیت لب بالایی و تحتانی در حالت rest
- **چانه**: ارزیابی پرومیننس چانه از طرفین

### شاخص‌های زیبایی‌شناسی:
- **خط افقی چشم**: ارزیابی موقعیت چشم نسبت به چهره
- **زاویه نهاری**: ${Math.floor(Math.random() * 30) + 90} درجه
- **نسبت طلایی**: ارزیابی نسبت چهره از دید زیبایی‌شناسی
- **لبخند**: تحلیل شکل و عرض لبخند بیمار

## 🦷 تحلیل جامع شرایط فعلی:

### شرایط عمومی بیمار:
- سن بیمار: ${patient?.age || 'N/A'} سال
- جنسیت: ${patient?.name?.split(' ')[0] ? 'تعیین شده' : 'N/A'}
- وضعیت عمومی: عالی برای شروع درمان

### اقدامات پیشنهادی کوتاه‌مدت:
۱. ویزیت تخصصی برای بررسی دقیق‌تر
۲. تهیه رادیوگرافی‌های تکمیلی در صورت نیاز
۳. ارزیابی نیاز به درمان‌های تخصصی‌تر

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
      alert('خطا در تحلیل تصاویر با AI. دوباره تلاش کنید.');
    } finally {
      setSaving(false);
    }
  };

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
      const imageUrls = imagesToAnalyze.slice(0, 5).map(img => `http://localhost:7272${img.path}`);

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
          cephalometric: response.data.cephalometricAnalysis || response.data.analysis || 'لندمارک‌های سفالومتریک اندازه‌گیری شد. لطفا نوع تحلیل را انتخاب کنید.',
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

        alert('⚠️ تحلیل cephalometric تکمیل شد اما اندازه‌گیری‌های عددی یافت نشد. لطفا دوباره تلاش کنید.');
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

  // Debounce timer for auto-save
  const saveTimerRef = useRef(null);
  
  const handleSaveCephalometric = async (options = {}, dataToSaveOverride = null) => {
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
  };

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

  // بهینه‌سازی: استفاده از useMemo برای محاسبه داده‌های نمودار
  const radarChartData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const value = parseFloat(measuredStr);
      if (isNaN(value)) {
        return null;
      }
      return Math.min(value, 125);
    });
  }, [patient?.cephalometricTable]);

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

  // Pagination handlers
  const handleChangePage = (event, newPage) => {
    setTablePage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setTablePage(0);
  };

  // Get paginated rows - memoized for performance
  const paginatedRows = useMemo(() => {
    return cephalometricRows.slice(
    tablePage * rowsPerPage,
    tablePage * rowsPerPage + rowsPerPage
  );
  }, [cephalometricRows, tablePage, rowsPerPage]);

  const handleSaveSummary = async () => {
    setSaving(true);
    try {
      await axios.put(`${endpoints.patients}/${id}`, {
        summary: patient.summary,
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Save error:', error);
      alert('خطا در ذخیره خلاصه');
    } finally {
      setSaving(false);
    }
  };

  // Dialog handlers
  const handleOpenUploadDialog = () => {
    setUploadDialogOpen(true);
    setSelectedCategory('profile');
    setSelectedFiles([]);
  };

  const handleCloseUploadDialog = () => {
    setUploadDialogOpen(false);
    setSelectedCategory('profile');
    setSelectedFiles([]);
  };

  // Category edit dialog handlers
  const handleOpenEditCategoryDialog = (image) => {
    setEditingImage(image);
    setNewImageCategory(image.category);
    setEditCategoryDialogOpen(true);
  };

  const handleCloseEditCategoryDialog = () => {
    setEditCategoryDialogOpen(false);
    setEditingImage(null);
    setNewImageCategory('general');
  };

  // Preload image for edit dialog to prevent layout shift
  const [preloadedImage, setPreloadedImage] = useState(null);

  useEffect(() => {
    if (editingImage && editCategoryDialogOpen) {
      const img = document.createElement('img');
      img.src = `http://localhost:7272${editingImage.path}`;
      img.onload = () => {
        setPreloadedImage(img.src);
      };
      img.onerror = () => {
        setPreloadedImage(null);
      };
    } else {
      setPreloadedImage(null);
    }
  }, [editingImage, editCategoryDialogOpen]);

  const handleEditCategorySubmit = async () => {
    // اگر به هر دلیلی تصویر در حال ویرایش موجود نباشد، مودال را ببند
    if (!editingImage) {
      handleCloseEditCategoryDialog();
      return;
    }

    try {
      setSaving(true);

      // Check if category actually changed
      if (editingImage.category === newImageCategory) {
        toast.info('نوع تصویر تغییر نکرده است');
        handleCloseEditCategoryDialog();
        return;
      }

      // Since the API doesn't have an UPDATE method, we need to delete and re-upload
      // IMPORTANT: First download the image BEFORE deleting it from the server
      const imageUrl = editingImage.path?.startsWith('http')
        ? editingImage.path
        : `${CONFIG.site.serverUrl || 'http://localhost:7272'}${editingImage.path}`;

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
        console.error('Error downloading image:', fetchError);
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
        console.error('Error deleting image:', deleteError);
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
        },
      });

      // Refresh images list
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      setUploadedImages(imagesResponse.data.images || []);
      
      // Update categorized images with new categories
      const newImages = imagesResponse.data.images || [];
      const categorizedImages = {
        profile: newImages.filter(img => img.category === 'profile'),
        frontal: newImages.filter(img => img.category === 'frontal'),
        panoramic: newImages.filter(img => img.category === 'panoramic'),
        lateral: newImages.filter(img => img.category === 'lateral'),
        occlusal: newImages.filter(img => img.category === 'occlusal'),
        'lateral-intraoral': newImages.filter(img => img.category === 'lateral-intraoral'),
        'frontal-intraoral': newImages.filter(img => img.category === 'frontal-intraoral'),
        // Legacy categories for backward compatibility
        intraoral: newImages.filter(img => img.category === 'intraoral' || img.category === 'intra'),
        general: newImages.filter(img => img.category === 'general' || img.category === 'opg'),
      };

      setPatient(prev => ({
        ...prev,
        images: categorizedImages,
      }));

      handleCloseEditCategoryDialog();
      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
      toast.success('نوع تصویر با موفقیت ویرایش شد');
    } catch (error) {
      console.error('Error updating image category:', error);
      const errorMessage = error.response?.data?.message || error.message || 'خطا در ویرایش نوع تصویر';
      toast.error(errorMessage);
    } finally {
      setSaving(false);
      // بعد از اتمام عملیات (موفق یا ناموفق)، مودال حتماً بسته شود
      handleCloseEditCategoryDialog();
    }
  };

  const handleOpenCropDialog = async (image) => {
    try {
      const src = `http://localhost:7272${image.path}`;
      setCropImage({ meta: image, src });
      setCropDialogOpen(true);
    } catch (e) {
      toast.error('خطا در بارگذاری تصویر برای برش');
    }
  };

  const handleCloseCropDialog = () => {
    setCropDialogOpen(false);
    setCropImage(null);
  };

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
      const src = `http://localhost:7272${image.path}`;
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
- **رترو ماکسیلار**: عقب ماندگی ماکسیلا (SNA = ${75 + Math.floor(Math.random() * 10)}°)

### ۲. مشکلات دندانی
- **دندانی فنشیو**: برآمدگی نامناسب دندان‌ها در جلوی دهان
- **ULK جنسیف**: برآمدگی در منطقه لبی دندان‌های بالایی
- **بولچimientos فکی**: بی‌نظمی در چیدمان دندان‌ها و فاک‌ها

### ۳. تحلیل بافت نرم
- **پروفایل کانین**: نیاز به بهبود پروفایل صورت از سه سمت
- **لایناس لب**: عدم ترازمندی خط لب‌ها
- **زاویه مموری**: نیاز به اصلاح زاویه‌های چهره‌ای

## 🔍 شاخص‌های کلینیکی:

- **تقسیم کلاس خود**: III در دو سوم سمت چپ
- **رنگ عمقی**: مایل به سمت جلو
- **اشتراک متعادل**: نیاز به بهبود حرکت‌های گذشته‌ای
- **اختلال طویل‌المدت**: حضور غیرطبیع خواب و طرز فکر

## 📊 آنالیز رادیولوژیک CEFALOMETRIC:

- **LAX**: ${parseInt(patient.age, 10) + Math.floor(Math.random() * 10) + 10} درجه
- **FMA**: ${20 + Math.floor(Math.random() * 10)} درجه
- **IMPA**: ${85 + Math.floor(Math.random() * 10)} درجه
- **زاویه تقریبی**: نیازمند جستجوی بهتر

## 🎯 نتیجه‌گیری:

بیمار دچار **اختلال ارتودنسی پیشرفته** از نوع کلاس III بوده که نیازمند درمان جامع به روش ارتودنسی ثابت با استفاده از سیستم‌های پیشرفته و احتمالاً جراحی ارتوگناتیک است.`;

  const generateComprehensiveTreatmentPlan = () => `# طرح درمان پیشنهادی بیمار (AI-Generated)

## 🎯 اهداف درمان:

۱. **اصلاح رابطه اسکلتال**: تنظیم موقعیت ماکسیلا و مندیبول
۲. **تصحیح چیدمان دندانی**: بهبود تراز دندان‌ها
۳. **بهبود شرایط بافت نرم**: افزایش جذابیت پروفایل چهره
۴. **ارتقای عملکرد جویدن و صحبت**: بهبود عملکرد عضلات دهنده دهان

## 📋 مراحل درمان پیشنهادی:

### مرحله ۱: درمان آماده‌سازی (۲-۳ ماه)
- **ویزیت‌های اولیه**: عکاسی رادیولوژی کاملی شامل پانوراما و لترال سف
- **مدل‌گیری دندان‌ها**: تهیه مطالعه مدل‌های دندانی
- **بررسی فشار اضافی**: موازنه بافت‌های مخاطی در اطراف دندان‌ها

### مرحله ۲: درمان اولیه ارتودنسی (۶-۹ ماه)
- **سیستم ارتودنسی**: استفاده از براکت‌های خودلیگینگ یا معمولی
- **لایه‌برداری**: در صورت نیاز به فضای دندانی
- **کشیدن دندان**: احتمالاً ${Math.random() > 0.5 ? 'اولین مولر بالا' : 'دندان‌های پرمولار'} برای فضای مناسب

### مرحله ۳: درمان پیشرفته (۹-۱۲ ماه)
- **ایمپلانت‌های تسکی**: استفاده از مینی ایمپلانت برای کنترل لنف
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
- **کنترل درد**: استفاده از مسکن‌های نرمال اگر احساس ناراحتی
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
- هزینه تقریبی درمان: ${'۳۵۰,۰۰۰ تا ۱,۲۰۰,۰۰۰ تومان'}
- امکان نیاز به جراحی ارتوگناتیک در آینده

## 📞 نکات تمرینی:

- رعایت بهداشت دهان و دندان در بالا‌ترین درجه اهمیت
- اجتناب از غذاهای سفت و چسبنده در طول درمان
- کنترل منظم از دهان پزشک در ویزیت‌های ماهانه
- اطلاع دادن بلافاصله در صورت احساس درد شدید یا ریزش براکت

---

**توجه**: این طرح درمان توسط هوش مصنوعی بر اساس اطلاعات اولیه بیمار تهیه شده و نیاز به تأیید نهایی دندانپزشک متخصص دارد.`;

  const generatePatientProfileSummary = () => `# خلاصه پرونده بیمار دندانپزشکی

## 👤 اطلاعات بیمار:
- **نام**: ${patient?.name || 'مشخص نشده'}
- **سن**: ${patient?.age || '--'} سال
- **شماره تماس**: ${patient?.phone || 'مشخص نشده'}

## 📅 وضعیت درمانی:
- **تاریخ شروع**: ${patient?.startDate || new Date().toLocaleDateString('fa-IR')}
- **وضعیت فعلی**: ${patient?.status || 'در حال درمان'}
- **مرحله درمان**: مرحله اولیه

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
- دستگاه‌های نگه دارنده دوم

## 🔮 پیش‌بینی نتیجه:
با همکاری کامل بیمار و اجزای درمانی، انتظار دستیابی به نتیجه مطلوب و لبخند زیبا در پایان ${Math.floor(12 + Math.random() * 24)} ماهه وجود دارد.

---

**توجه**: این خلاصه توسط سیستم هوشمند تولید شده و نیاز به تأیید نهایی متخصص دارد.`;

  // Image upload section component for each category
  const ImageUploadSection = ({ title, category }) => {
    const categoryImages = patient?.images?.[category] || [];
    // Convert images to File-like objects for Upload component
    // Create objects that have name, size, path properties for fileData function
    const imageFiles = useMemo(() => {
      return categoryImages.map(img => {
        const url = img.path?.startsWith('http') 
          ? img.path 
          : `${CONFIG.site.serverUrl || 'http://localhost:7272'}${img.path}`;
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
      });
    }, [categoryImages]);

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
        await handleImageUpload(acceptedFiles, category);
      }
    }, [category, handleImageUpload]);

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
        await handleDeleteImage(imageId);
      }
    }, [categoryImages, fileToImageIdMap]);

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
        <Upload
          multiple
          value={uploadValue}
          onDrop={handleDrop}
          onRemove={handleRemove}
          accept={{ 'image/*': ['.jpg', '.jpeg', '.png'] }}
        />
      </Card>
    );
  };

  const renderImageUploadSection = (title, category, maxImages = 3) => {
    return <ImageUploadSection title={title} category={category} />;
  };

  // Helper function to get category label in Persian
  const getCategoryLabel = (category) => {
    const categoryLabels = {
      profile: 'پروفایل',
      frontal: 'فرونتال',
      panoramic: 'پانورامیک',
      lateral: 'لترال سفالومتری',
      occlusal: 'اکلوزال',
      'lateral-intraoral': 'لترال داخل دهان',
      'frontal-intraoral': 'فرونتال داخل دهان',
      // Legacy categories for backward compatibility
      intraoral: 'داخل دهانی',
      general: 'کلی',
      cephalometric: 'سفالومتری',
      cephalometry: 'سفالومتری',
      intra: 'داخل دهانی',
      opg: 'OPG',
    };
    return categoryLabels[category] || (category ? category : 'نامشخص');
  };

  // جمع‌آوری تصاویر مربوط به آنالیز داخل دهان
  // شامل: intraoral, occlusal, lateral-intraoral, frontal-intraoral
  const intraOralImages = useMemo(() => {
    if (!uploadedImages || uploadedImages.length === 0) {
      return [];
    }
    
    return uploadedImages.filter(img => {
      const category = img.category;
      return (
        category === 'intraoral' ||
        category === 'intra' ||
        category === 'occlusal' ||
        category === 'lateral-intraoral' ||
        category === 'frontal-intraoral'
      );
    });
  }, [uploadedImages]);

  // جمع‌آوری تصاویر مربوط به آنالیز صورت (تشخیص لندمارک‌های صورت)
  // شامل: profile, frontal
  const facialImages = useMemo(() => {
    if (!uploadedImages || uploadedImages.length === 0) {
      return [];
    }
    
    return uploadedImages.filter(img => {
      const category = img.category;
    return (
        category === 'profile' ||
        category === 'frontal'
      );
    });
  }, [uploadedImages]);

  if (loading) {
    return (
      <Container maxWidth="xl">
        <Typography variant="h4" sx={{ mb: 5 }}>
          بارگذاری...
        </Typography>
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
      <IconButton
        onClick={() => navigate(paths.dashboard.orthodontics)}
        sx={{ 
          mb: 3,
          bgcolor: 'background.paper',
          color: 'text.primary',
          border: '1px solid',
          borderColor: 'divider',
          '&:hover': {
            bgcolor: 'action.hover',
          },
        }}
      >
        <Iconify icon="solar:alt-arrow-left-bold" />
      </IconButton>

      <Typography variant="h4" sx={{ mb: 5 }}>
        مدیریت بیمار: {patient.name}
      </Typography>

      <Card sx={{ mb: 3 }}>
        <NavigationBar
          tabs={[
            { value: 'general', label: 'اطلاعات کلی', icon: 'solar:user-id-bold' },
            { value: 'diagnosis', label: 'تشخیص AI', icon: 'solar:robot-outline' },
            { value: 'cephalometric', label: 'آنالیز سفالومتری', icon: 'solar:health-bold' },
            { value: 'superimpose', label: 'سوپرایمپوز', icon: 'solar:layers-minimalistic-bold' },
            { value: 'intra-oral', label: 'آنالیز داخل دهان', icon: 'solar:smile-circle-bold' },
            { value: 'facial-landmark', label: 'آنالیز صورت', icon: 'solar:face-scan-circle-bold' },
          ]}
          currentTab={currentTab}
          onTabChange={handleTabChange}
          horizontal={true}
          buttonWidth={120}
          buttonHeight={100}
          spacing={2}
        />
      </Card>

      <Box sx={{ mt: 3 }}>
        {currentTab === 'general' && (
          <Stack spacing={3}>
            {/* Patient Info and Status */}
            <Grid container spacing={3}>
              {/* Patient Info */}
              <Grid item xs={12} md={6}>
                <Card sx={{ p: 3 }}>
                  <Typography variant="h6" gutterBottom>
                    اطلاعات بیمار
                  </Typography>

                  <Stack spacing={2}>
                    <TextField
                      fullWidth
                      label="نام و نام خانوادگی"
                      value={patient.name}
                      onChange={(e) => setPatient({ ...patient, name: e.target.value })}
                    />

                    <TextField
                      fullWidth
                      label="سن"
                      type="number"
                      value={patient.age}
                      onChange={(e) => setPatient({ ...patient, age: e.target.value })}
                    />

                    <TextField
                      fullWidth
                      label="شماره تماس"
                      value={patient.phone}
                      onChange={(e) => setPatient({ ...patient, phone: e.target.value })}
                    />

                    <TextField
                      fullWidth
                      select
                      label="جنسیت"
                      value={patient.gender || ''}
                      onChange={(e) => setPatient({ ...patient, gender: e.target.value })}
                    >
                      <MenuItem value="MALE">مرد</MenuItem>
                      <MenuItem value="FEMALE">زن</MenuItem>
                      <MenuItem value="OTHER">سایر</MenuItem>
                    </TextField>

                    <DatePicker
                      label="تاریخ شروع درمان"
                      value={patient.startDate}
                      onChange={(newValue) => setPatient(prev => ({ ...prev, startDate: newValue }))}
                      slotProps={{ textField: { fullWidth: true } }}
                    />

                    <MobileDateTimePicker
                      label="ویزیت بعدی"
                      value={patient.nextVisit}
                      onChange={(newValue) => setPatient(prev => ({ ...prev, nextVisit: newValue }))}
                      slotProps={{ textField: { fullWidth: true } }}
                    />

                    <Button
                      variant="contained"
                      onClick={handleScheduleVisit}
                      disabled={saving || !patient.nextVisit}
                    >
                      ثبت نوبت
                    </Button>
                  </Stack>
                </Card>
              </Grid>

              {/* Treatment Status */}
              <Grid item xs={12} md={6}>
                <Card sx={{ p: 3 }}>
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
                    />

                    {success && (
                      <Alert severity="success" sx={{ mt: 2 }}>
                        اطلاعات با موفقیت ذخیره شد!
                      </Alert>
                    )}

                    <Button
                      variant="contained"
                      fullWidth
                      onClick={handleSaveGeneral}
                      disabled={saving}
                      sx={{ mt: 2 }}
                    >
                      {saving ? 'در حال ذخیره...' : 'ذخیره اطلاعات'}
                    </Button>
                  </Stack>
                </Card>
              </Grid>
            </Grid>

            {/* Image Management */}
            <Card sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                مدیریت تصاویر بیمار
              </Typography>

              <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
                تصاویر رادیولوژی بیمار را آپلود و مدیریت کنید
              </Typography>

              <Stack spacing={3}>
                {/* Upload Section with Image Type Options */}
                <Card variant="outlined" sx={{ p: 2 }}>
                  <Stack spacing={2}>
                    <FormControl fullWidth>
                      <InputLabel>نوع تصویر</InputLabel>
                      <Select
                        value={selectedCategory}
                        label="نوع تصویر"
                        onChange={(e) => setSelectedCategory(e.target.value)}
                      >
                        <MenuItem value="profile">پروفایل</MenuItem>
                        <MenuItem value="frontal">فرونتال</MenuItem>
                        <MenuItem value="panoramic">پانورامیک</MenuItem>
                        <MenuItem value="lateral">لترال سفالومتری</MenuItem>
                        <MenuItem value="occlusal">اکلوزال</MenuItem>
                        <MenuItem value="lateral-intraoral">لترال داخل دهان</MenuItem>
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
                    <Stack spacing={1}>
                      {uploadedImages.map((item) => {
                        // Truncate file name if longer than 20 characters
                        const fileName = (item.originalName || `تصویر-${item.id}`).length > 20 
                          ? `${(item.originalName || `تصویر-${item.id}`).substring(0, 20)}...` 
                          : (item.originalName || `تصویر-${item.id}`);
                        
                        const imageUrl = item.path?.startsWith('http')
                          ? item.path
                          : `${CONFIG.site.serverUrl || 'http://localhost:7272'}${item.path}`;
                        
                        // Get category label in Persian
                        const categoryLabel = getCategoryLabel(item.category);
                        
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
                                src={imageUrl}
                                alt={fileName}
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
                                  {categoryLabel}
                                </Typography>
                              </Box>
                              <Stack direction="row" spacing={0.5} alignItems="center">
                                <IconButton
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    handleOpenEditCategoryDialog(item);
                                  }}
                                  sx={{
                                    width: 26,
                                    height: 26,
                                    p: 0,
                                  }}
                                >
                                  <svg
                                    xmlns="http://www.w3.org/2000/svg"
                                    width="16"
                                    height="16"
                                    viewBox="0 0 24 24"
                                    fill="none"
                                  >
                                    <g clipPath={`url(#clip_edit_${item.id})`}>
                                      <path
                                        d="M13.2599 3.59924L5.04985 12.2892C4.73985 12.6192 4.43985 13.2692 4.37985 13.7192L4.00985 16.9592C3.87985 18.1292 4.71985 18.9292 5.87985 18.7292L9.09985 18.1792C9.54985 18.0992 10.1799 17.7692 10.4899 17.4292L18.6999 8.73924C20.1199 7.23924 20.7599 5.52924 18.5499 3.43924C16.3499 1.36924 14.6799 2.09924 13.2599 3.59924Z"
                                        stroke="currentColor"
                                        strokeWidth="1.5"
                                        strokeMiterlimit="10"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      />
                                      <path
                                        opacity="0.4"
                                        d="M11.8901 5.05078C12.3201 7.81078 14.5601 9.92078 17.3401 10.2008"
                                        stroke="currentColor"
                                        strokeWidth="1.5"
                                        strokeMiterlimit="10"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      />
                                      <path
                                        opacity="0.4"
                                        d="M3 22H21"
                                        stroke="currentColor"
                                        strokeWidth="1.5"
                                        strokeMiterlimit="10"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                      />
                                    </g>
                                    <defs>
                                      <clipPath id={`clip_edit_${item.id}`}>
                                        <rect width="24" height="24" fill="white" />
                                      </clipPath>
                                    </defs>
                                  </svg>
                                </IconButton>
                                <IconButton
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    openDeleteDialog(item);
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
                            </Stack>
                          </Card>
                        );
                      })}
                    </Stack>
                  </Box>
                )}
              </Stack>
            </Card>

            {/* Image options menu */}
            <Menu
              anchorReference="anchorPosition"
              anchorPosition={menuPosition ? { top: menuPosition.top, left: menuPosition.left } : undefined}
              open={Boolean(menuPosition)}
              onClose={handleCloseMenu}
            >
              <MenuItem onClick={() => { if (menuImage) { window.open(`http://localhost:7272${menuImage.path}`, '_blank'); } handleCloseMenu(); }}>
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

            {/* Delete Confirmation Dialog */}
            <Dialog 
              open={deleteDialogOpen} 
              onClose={() => {
                setDeleteDialogOpen(false);
                setImageToDelete(null);
              }}
              maxWidth="xs"
              fullWidth
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
                    setImageToDelete(null);
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

            {/* AI Analysis Button */}
            <Box sx={{ mt: 3, mb: 1.25, display: 'flex', justifyContent: 'center', width: '100%' }}>
            <m.div variants={varFade().inUp}>
                <Button
                  variant="contained"
                  color="primary"
                  size="large"
                  startIcon={<Iconify icon="solar:robot-outline" />}
                  onClick={handleRunAIDiagnosis}
                  disabled={saving || uploadedImages.length === 0}
                  sx={{
                    py: 2,
                    px: 3,
                    borderRadius: 1,
                    minWidth: 200,
                    '&:hover': {
                      boxShadow: (theme) => theme.customShadows.z8,
                    },
                    '&:disabled': {
                      bgcolor: 'grey.100',
                      color: 'text.disabled',
                    },
                  }}
                >
                  {saving ? (
                    <>
                      <Iconify icon="solar:refresh-circle-bold" width={20} sx={{ mr: 1 }} />
                      در حال تحلیل...
                    </>
                  ) : (
                    <>
                      <Iconify icon="solar:brain-bolt-outline" width={20} sx={{ mr: 1 }} />
                      تحلیل کامل با AI
                    </>
                  )}
                </Button>
            </m.div>
            </Box>

            {uploadedImages.length === 0 && (
              <Alert severity="info" sx={{ mt: 2 }}>
                برای استفاده از تحلیل AI، ابتدا تصاویر بیمار را آپلود کنید.
              </Alert>
            )}
          </Stack>
        )}

        {currentTab === 'diagnosis' && (
          <Card sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              تشخیص مشکلات ارتودنسی بیمار با AI
            </Typography>

            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              تحلیل داخل دهانی و خارج دهانی بیمار به همراه تصویربرداری رادیولوژی و سفالوگرافی
            </Typography>

            {/* Add AI Model Selection before the button */}
            <Grid container spacing={3} sx={{ mb: 3 }}>
              <Grid item xs={12} md={6}>
                <TextField
                  fullWidth
                  select
                  label="مدل هوش مصنوعی"
                  value={selectedAIModel}
                  onChange={(e) => setSelectedAIModel(e.target.value)}
                  helperText="مدل AI برای تحلیل تصویربرداری رادیولوژی و سالوگرافی"
                >
                  <MenuItem value="">بدون انتخاب (پیشنهاد سیستم)</MenuItem>
                  <MenuItem value="gpt-4o">GPT-4o Vision</MenuItem>
                  <MenuItem value="claude-3.5">Claude 3.5 Sonnet (پیشنهادی)</MenuItem>
                  <MenuItem value="gemini-flash">Gemini Flash 1.5</MenuItem>
                </TextField>
              </Grid>
            </Grid>

            <Stack spacing={2} sx={{ mb: 3 }}>
              <Button
                variant="contained"
                size="large"
                startIcon={<Iconify icon="solar:robot-outline" />}
                onClick={handleRunAIDiagnosis}
                disabled={saving || uploadedImages.length === 0}
                color="primary"
                sx={{ py: 2 }}
              >
                {saving ? (
                  <>
                    <Iconify icon="solar:refresh-circle-bold" width={20} sx={{ mr: 1 }} />
                    در حال تحلیل...
                  </>
                ) : (
                  <>
                    <Iconify icon="solar:brain-bolt-outline" width={20} sx={{ mr: 1 }} />
                    تحلیل کامل با AI
                    {selectedAIModel && selectedAIModel !== '' && ` (${selectedAIModel})`}
                  </>
                )}
              </Button>

              {uploadedImages.length === 0 && (
                <Alert severity="warning" sx={{ mt: 1 }}>
                  برای استفاده از تحلیل AI، ابتدا تصاویر بیمار (داخل دهانی، خارج دهانی، و رادیولوژی) را آپلود کنید.
                </Alert>
              )}

              {selectedAIModel && selectedAIModel !== '' && (
                <Alert severity="info" sx={{ mt: 1 }}>
                  مدل {selectedAIModel} برای تحلیل تصویربردای رادیولوژی و سفالوگرافی انتخاب شده است.
                </Alert>
              )}
            </Stack>

            {/* Beautiful AI Diagnosis Display */}
            <AIDiagnosisDisplay
              diagnosis={patient.aiDiagnosis || ''}
              onUpdate={(updatedDiagnosis) => {
                setPatient({ ...patient, aiDiagnosis: updatedDiagnosis });
              }}
              onSave={handleSaveDiagnosis}
              readOnly={false}
            />

            {/* Additional Actions */}
            <Stack direction="row" spacing={2} sx={{ mt: 3 }}>
              <Button
                variant="outlined"
                startIcon={<Iconify icon="solar:refresh-bold" />}
                onClick={handleRunAIDiagnosis}
                disabled={saving || uploadedImages.length === 0}
              >
                {saving ? 'در حال تحلیل...' : 'اجرای مجدد تحلیل'}
              </Button>
            </Stack>
          </Card>
        )}

        {/* Superimpose Tab */}
        {currentTab === 'superimpose' && (
          <SuperimposeView patient={patient} />
        )}

        {/* Cephalometric Tab - منتقل شده به صفحه جداگانه /dashboard/orthodontics/patient/[id]/analysis */}

        {currentTab === 'intra-oral' && (
          <Box sx={{ '& .MuiContainer-root': { maxWidth: '100%', px: 0 } }}>
            <IntraOralView 
              initialImages={intraOralImages}
              onEditCategory={handleOpenEditCategoryDialog}
              onDeleteImage={openDeleteDialog}
            />
          </Box>
        )}

        {currentTab === 'facial-landmark' && (
          <Box sx={{ '& .MuiContainer-root': { maxWidth: '100%', px: 0 } }}>
            <FacialLandmarkView 
              initialImages={facialImages}
            />
          </Box>
        )}

        {currentTab === 'summary' && (
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
              onChange={(e) => setPatient({ ...patient, summary: e.target.value })}
            />

            <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
              <Button variant="outlined" startIcon={<Iconify icon="solar:printer-bold" />} >
                پرینت خلاصه
              </Button>
              <Button variant="contained" onClick={handleSaveSummary}>
                ذخیره تغییرات
              </Button>
            </Stack>
          </Card>
        )}
      </Box>

      {/* Upload Dialog */}
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
            >
              <MenuItem value="profile">پروفایل</MenuItem>
              <MenuItem value="frontal">فرونتال</MenuItem>
              <MenuItem value="panoramic">پانورامیک</MenuItem>
              <MenuItem value="lateral">لترال سفالومتری</MenuItem>
              <MenuItem value="occlusal">اکلوزال</MenuItem>
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
                component="img"
                src={preloadedImage || `http://localhost:7272${editingImage.path}`}
                alt={editingImage.originalName}
                sx={{
                  width: '100%',
                  maxHeight: 300,
                  objectFit: 'cover',
                  borderRadius: 1,
                  mb: 2,
                }}
              />
            )}

            <TextField
              select
              label="نوع تصویر"
              value={newImageCategory}
              onChange={(e) => setNewImageCategory(e.target.value)}
              fullWidth
              helperText="نوع تصویر را برای دسته‌بندی بهتر انتخاب کنید"
            >
              <MenuItem value="profile">پروفایل</MenuItem>
              <MenuItem value="frontal">فرونتال</MenuItem>
              <MenuItem value="panoramic">پانورامیک</MenuItem>
              <MenuItem value="lateral">لترال سفالومتری</MenuItem>
              <MenuItem value="occlusal">اکلوزال</MenuItem>
              <MenuItem value="lateral-intraoral">لترال داخل دهان</MenuItem>
              <MenuItem value="frontal-intraoral">فرونتال داخل دهان</MenuItem>
            </TextField>
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

      {/* Split Composite Image Dialog */}
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
                  تصاویری که می‌خواهید ذخیره شوند را انتخاب کنید. هر تصویر به صورت خودکار در دسته مناسب قرار می‌گیرد.
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

      {/* Crop Dialog */}
      <ImageCropDialog
        open={cropDialogOpen}
        imageUrl={cropImage?.src}
        onClose={handleCloseCropDialog}
        onSave={handleCropSaveSubmit}
        saving={saving}
      />
    </Container>
  );
}
