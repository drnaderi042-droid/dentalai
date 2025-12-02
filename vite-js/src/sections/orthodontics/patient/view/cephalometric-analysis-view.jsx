import { useParams, useNavigate } from 'react-router-dom';
import { lazy, memo, useRef, useMemo, useState, Suspense, useEffect, useCallback, startTransition, useDeferredValue } from 'react';

import Box from '@mui/material/Box';
import Tab from '@mui/material/Tab';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Select from '@mui/material/Select';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import TableRow from '@mui/material/TableRow';
import TextField from '@mui/material/TextField';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import FormControl from '@mui/material/FormControl';
import CardContent from '@mui/material/CardContent';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';
import { useTheme, alpha as hexAlpha } from '@mui/material/styles';

import axios, { endpoints } from 'src/utils/axios';
import { getImageUrl } from 'src/utils/url-helpers';
import { formatAnalysisForDisplay, generateComprehensiveAnalysis } from 'src/utils/orthodontic-analysis.ts';

import { DashboardContent } from 'src/layouts/dashboard';
import { useHeaderContent } from 'src/contexts/header-content-context';

import { Label } from 'src/components/label';
import { toast } from 'src/components/snackbar';
import { useChart } from 'src/components/chart';
import { Iconify } from 'src/components/iconify';
import { CustomTabs } from 'src/components/custom-tabs';

import { useAuthContext } from 'src/auth/hooks';

import { ClinicalRAGAnalysis } from '../components/clinical-rag-analysis';

// Lazy load heavy components for better initial load performance
const Chart = lazy(() => import('src/components/chart').then(module => ({ default: module.Chart })));
const CephalometricAIAnalysis = lazy(() => import('../components/cephalometric-ai-analysis').then(module => ({ default: module.MemoizedCephalometricAIAnalysis })));

// Preload lazy components for smoother tab switching
const preloadComponents = () => {
  // Use requestIdleCallback for better performance
  if ('requestIdleCallback' in window) {
    requestIdleCallback(() => {
      // Preload Chart component
      import('src/components/chart').catch(() => {});
      // Preload CephalometricAIAnalysis component
      import('../components/cephalometric-ai-analysis').catch(() => {});
    }, { timeout: 2000 });
  } else {
    // Fallback for browsers without requestIdleCallback
    setTimeout(() => {
      import('src/components/chart').catch(() => {});
      import('../components/cephalometric-ai-analysis').catch(() => {});
    }, 100);
  }
};

// Loading fallback component
const LoadingFallback = memo(() => (
  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 200 }}>
    <CircularProgress size={40} />
  </Box>
));
LoadingFallback.displayName = 'LoadingFallback';

// ----------------------------------------------------------------------

// Cephalometric parameter templates for each analysis method
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
    'Overbite': { 
      mean: '3', 
      sd: '1', 
      severity: 'نرمال', 
      note: 'میزان همپوشانی دندان های اینسایزال بالا با دندان های اینسایزال پایین. مقادیر منفی: دندان ها با هم همپوشانی دارند. مقادیر مثبت: دندان ها در بعد عمودی از هم فاصله دارند و بیمار openbite است.' 
    },
    'Overjet': { 
      mean: '3', 
      sd: '1', 
      severity: 'نرمال', 
      note: 'فاصله افقی بین دندان سانترال فک بالا (U1) و دندان سانترال فک پایین (L1). افزایش: overjet زیاد (protrusion). کاهش: overjet کم. مقادیر منفی: بیمار کراس بایت قدامی دارد' 
    },
    // Ricketts Analysis Parameters
    'Facial Axis': { mean: '90', sd: '3', severity: 'نرمال', note: 'محور صورت (زاویه بین Ba-N و Pt-Gn). افزایش: رشد عمودی صورت (vertical growth pattern). کاهش: رشد افقی صورت (horizontal growth pattern)' },
    'Facial Depth': { mean: '88', sd: '3', severity: 'نرمال', note: 'عمق صورت (زاویه بین N-Pog و Or-Po). افزایش: صورت عمیق‌تر (deep face). کاهش: صورت کم‌عمق‌تر (shallow face)' },
    'Lower Face Height': { mean: '47', sd: '2', severity: 'نرمال', note: 'نسبت فاصله عمودی صورت پایین (فاصله عمودی ANS-Me / فاصله عمودی N-Me × 100). افزایش: ارتفاع صورت پاییارتفاع بخش پایین صورتن بیشتر (long face). کاهش: ارتفاع بخش پایین صورت کمتر (short face)' },
    'Mandibular Plane': { mean: '26', sd: '4', severity: 'نرمال', note: 'صفحه مندیبولار (زاویه Go-Me نسبت به FH). افزایش: صفحه مندیبولار شیب‌دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' },
    'Convexity': { mean: '0', sd: '2', severity: 'نرمال', note: 'تحدب صورت (فاصله A نقطه از خط N-Pog). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
    'Upper Incisor': { mean: '22', sd: '4', severity: 'نرمال', note: 'زاویه دندان سانترال بالا (نسبت به A-Pog). افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' },
    'Lower Incisor': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه دندان سانترال پایین (نسبت به A-Pog). افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
    'Interincisal Angle': { mean: '130', sd: '6', severity: 'نرمال', note: 'زاویه بین خط U1-U1A و خط L1-L1A. افزایش: زاویه بیشتر (بیشتر retroclined). کاهش: زاویه کمتر (بیشتر proclined)' },
    'Occlusal Plane Angle': { mean: '14', sd: '4', severity: 'نرمال', note: 'زاویه بین Occlusal Plane (L1-LMT) و خط S-N. افزایش: صفحه اکلوزال شیب‌دارتر (steep occlusal plane). کاهش: صفحه اکلوزال صاف‌تر (flat occlusal plane)' },
    'Cranial Deflection': { mean: '27', sd: '3', severity: 'نرمال', note: 'زاویه بین Ba-N-S (زاویه در نقطه N). افزایش: انحراف جمجمه بیشتر (greater cranial deflection). کاهش: انحراف جمجمه کمتر (lesser cranial deflection)' },
    'Palatal Plane Angle': { mean: '8', sd: '3', severity: 'نرمال', note: 'زاویه بین خط سقف دهان (ANS-PNS) و خط فرانکفورت (Or-Po). افزایش: سقف دهان شیب‌دارتر (steep palatal plane). کاهش: سقف دهان صاف‌تر (flat palatal plane)' },
    // McNamara Analysis Parameters
    'Skeletal Convexity': { mean: '170', sd: '5', severity: 'نرمال', note: 'زاویه بین خط N-A و خط A-Pog (تحدب صورت). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
    'Co-A': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول فک بالا (فاصله Co-A). افزایش: فک بالا بلندتر (maxillary prognathism). کاهش: فک بالا کوتاه‌تر (maxillary retrognathism)' },
    'Co-Gn': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول فک پایین (فاصله Co-Gn). افزایش: فک پایین بلندتر (mandibular prognathism). کاهش: فک پایین کوتاه‌تر (mandibular retrognathism)' },
    'Upper Face Height': { mean: '55', sd: '3', severity: 'نرمال', note: 'ارتفاع صورت بالا (N-ANS). افزایش: ارتفاع صورت بالا بیشتر. کاهش: ارتفاع صورت بالا کمتر' },
    'Facial Height Ratio': { mean: '55', sd: '2', severity: 'نرمال', note: 'نسبت ارتفاع صورت (ANS-Me/N-Me × 100). افزایش: نسبت بیشتر (long face pattern). کاهش: نسبت کمتر (short face pattern)' },
    'Mandibular Plane Angle': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
    'PFH/AFH Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع خلفی به قدامی (PFH/AFH × 100). PFH: فاصله S-Go (ارتفاع خلفی). AFH: فاصله N-Me (ارتفاع قدامی). افزایش: vertical growth pattern. کاهش: horizontal growth pattern' },
    // Tweed Analysis Parameters
    FMA: { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به صفحه مندیبولار. افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' },
    FMIA: { mean: '65', sd: '5', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به incisor پایین. افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' },
    IMPA: { mean: '90', sd: '3', severity: 'نرمال', note: 'زاویه incisor پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
    // Jarabak Analysis Parameters
    'Jarabak Ratio (Facial Height Ratio)': { mean: '63.5', sd: '1.5', severity: 'نرمال', note: 'مهم‌ترین شاخص. (S-Go / N-Me) × ۱۰۰. >۶۵٪ ⬅️ رشد افقی (Hypodivergent). <۵۹٪ ⬅️ رشد عمودی (Hyperdivergent)' },
    'Posterior Facial Height (PFH)': { mean: '75', sd: '5', severity: 'نرمال', note: 'ارتفاع خلف صورت (S-Go). برای محاسبه Jarabak Ratio استفاده می‌شود. افزایش: ارتفاع خلفی بیشتر (vertical growth pattern). کاهش: ارتفاع خلفی کمتر (horizontal growth pattern)' },
    'Anterior Facial Height (AFH)': { mean: '120', sd: '5', severity: 'نرمال', note: 'ارتفاع قدامی صورت (N-Me). برای محاسبه Jarabak Ratio استفاده می‌شود. افزایش: ارتفاع قدامی بیشتر (long face). کاهش: ارتفاع قدامی کمتر (short face)' },
    'Saddle Angle': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه سدل (∠N-S-Ar). ↑ ⬅️ عقب‌گرد، ↓ ⬅️ جلوگرد' },
    'Articular Angle': { mean: '143', sd: '6', severity: 'نرمال', note: 'زاویه آرتیکولار (∠S-Ar-Go). ↑ ⬅️ رشد افقی، ↓ ⬅️ رشد عمودی' },
    'Gonial Angle (Total)': { mean: '130', sd: '7', severity: 'نرمال', note: 'زاویه گونیال کل (∠Ar-Go-Me). ↑ ⬅️ رشد عمودی، ↓ ⬅️ رشد افقی' },
    'Upper Gonial Angle': { mean: '53.5', sd: '1.5', severity: 'نرمال', note: 'زاویه گونیال بالا (∠Ar-Go-N)' },
    'Lower Gonial Angle': { mean: '72.5', sd: '2.5', severity: 'نرمال', note: 'زاویه گونیال پایین (∠N-Go-Gn). ↑ ⬅️ رشد عمودی' },
    'Sum of Posterior Angles': { mean: '396', sd: '6', severity: 'نرمال', note: 'مجموع زوایای خلفی (Saddle + Articular + Gonial). >۳۹۶° ⬅️ عمودی، <۳۹۶° ⬅️ افقی' },
    'Y-Axis (Growth Axis)': { mean: '59.5', sd: '3.5', severity: 'نرمال', note: 'محور Y (∠SGn–FH فرانکفورت). ↑ ⬅️ رشد عمودی، ↓ ⬅️ رشد افقی' },
    'Basal Plane Angle': { mean: '26.5', sd: '4', severity: 'نرمال', note: 'زاویه صفحه بازال (∠PNS-ANS به Go-Me). مشابه MP angle. افزایش: صفحه بازال شیب‌دارتر (vertical growth pattern). کاهش: صفحه بازال صاف‌تر (horizontal growth pattern)' },
    'Ramus Height': { mean: '50', sd: '5', severity: 'نرمال', note: 'ارتفاع راموس (Ar-Go). افزایش: ارتفاع راموس بیشتر (horizontal growth pattern). کاهش: ارتفاع راموس کمتر (vertical growth pattern)' },
    'Mandibular Arc': { mean: '27', sd: '1', severity: 'نرمال', note: 'زاویه بین Corpus Axis و Condylar Axis. ↑ ⬅️ رشد عمودی' },
    'Palatal Plane to FH': { mean: '0', sd: '3', severity: 'نرمال', note: 'زاویه صفحه پالاتال به FH (∠ANS-PNS به FH). کمی ↓ به خلف' },
    'Occlusal Plane to FH': { mean: '9', sd: '1', severity: 'نرمال', note: 'زاویه صفحه اکلوژال به FH (∠Occlusal Plane به FH)' },
    // Sassouni Analysis Parameters
    'Restriction Angle': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه محدود کننده (زاویه بین N-Ar-Go). افزایش: رشد عمودی صورت. کاهش: رشد افقی صورت' },
    'Sagittal Differentiation Angle': { mean: '59', sd: '4', severity: 'نرمال', note: 'این زاویه بیانگر شیب فک پایین نسبت به قاعده جمجمه است. کاربرد در Sassouni: بررسی الگوی رشد عمودی یا افقی فک پایین. اگر زاویه زیاد باشد ⬅️ رشد عمودی. اگر زاویه کم باشد ⬅️ رشد افقی' },
    'Social Selection Angle': { mean: '4', sd: '2', severity: 'نرمال', note: 'زاویه انتخاب اجتماعی (نسبت Go-Co به Go-Gn). افزایش: نسبت بیشتر. کاهش: نسبت کمتر' },
    'Cultural Ideal Angle': { mean: '90', sd: '5', severity: 'نرمال', note: 'زاویه ایدئال فرهنگی (زاویه بین N-Co و Go-Co). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
    'First Sagittal Angle': { mean: '74', sd: '4', severity: 'نرمال', note: 'زاویه نخستین ساژیتال (زاویه بین Ar-Co و Co-Gn). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
    // Wits Analysis Parameters
    'AO-BO': { mean: '0', sd: '2', severity: 'نرمال', note: 'اخلاف فاصله نقاط A و B نسبت به اکلوزال پلن (0 = کلاس I). افزایش: کلاس II (ماگزیلا جلوتر یا مندیبل عقب‌تر). کاهش: کلاس III (ماگزیلا عقب‌تر یا مندیبل جلوتر)' },
    'PP/Go-Gn': { mean: '27', sd: '4', severity: 'نرمال', note: 'زاویه بین صفحه پالاتال و صفحه مندیبولار. افزایش: زاویه بیشتر (vertical growth pattern). کاهش: زاویه کمتر (horizontal growth pattern)' },
    'S-Go': { mean: '75', sd: '5', severity: 'نرمال', note: 'ابعاد عمودی چهره (سلا-گناتیون). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
    'Sagittal Jaw': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه ساژیتال فک (معمولاً همان ANB). افزایش: کلاس II. کاهش: کلاس III' },
  },
  steiner: {
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
    'Overbite': { 
      mean: '3', 
      sd: '1', 
      severity: 'نرمال', 
      note: 'میزان همپوشانی دندان های اینسایزال بالا با دندان های اینسایزال پایین. مقادیر منفی: دندان ها با هم همپوشانی دارند. مقادیر مثبت: دندان ها در بعد عمودی از هم فاصله دارند و بیمار openbite است.' 
    },
    'Overjet': { 
      mean: '3', 
      sd: '1', 
      severity: 'نرمال', 
      note: 'فاصله افقی بین دندان سانترال فک بالا (U1) و دندان سانترال فک پایین (L1). افزایش: overjet زیاد (protrusion). کاهش: overjet کم. مقادیر منفی: بیمار کراس بایت قدامی دارد' 
    },
  },
  ricketts: {
    'Facial Axis': { mean: '90', sd: '3', severity: 'نرمال', note: 'محور صورت (زاویه بین Ba-N و Pt-Gn). افزایش: رشد عمودی صورت (vertical growth pattern). کاهش: رشد افقی صورت (horizontal growth pattern)' },
    'Facial Depth': { mean: '88', sd: '3', severity: 'نرمال', note: 'عمق صورت (زاویه بین N-Pog و Or-Po). افزایش: صورت عمیق‌تر (deep face). کاهش: صورت کم‌عمق‌تر (shallow face)' },
    'Lower Face Height': { mean: '47', sd: '2', severity: 'نرمال', note: 'نسبت فاصله عمودی صورت پایین (فاصله عمودی ANS-Me / فاصله عمودی N-Me × 100). افزایش: ارتفاع صورت پاییارتفاع بخش پایین صورتن بیشتر (long face). کاهش: ارتفاع بخش پایین صورت کمتر (short face)' },
    'Mandibular Plane': { mean: '26', sd: '4', severity: 'نرمال', note: 'صفحه مندیبولار (زاویه Go-Me نسبت به FH). افزایش: صفحه مندیبولار شیب‌دار (steep mandibular plane). کاهش: صفحه مندیبولار صاف (flat mandibular plane)' },
    'Convexity': { mean: '0', sd: '2', severity: 'نرمال', note: 'تحدب صورت (فاصله A نقطه از خط N-Pog). افزایش: صورت محدب‌تر (convex profile - کلاس II). کاهش: صورت مقعرتر (concave profile - کلاس III)' },
    'Upper Incisor': { mean: '22', sd: '4', severity: 'نرمال', note: 'زاویه دندان سانترال بالا (نسبت به A-Pog). افزایش: incisor بالا به سمت جلو (proclined). کاهش: incisor بالا به سمت عقب (retroclined)' },
    'Lower Incisor': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه دندان سانترال پایین (نسبت به A-Pog). افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
    'Interincisal Angle': { mean: '130', sd: '6', severity: 'نرمال', note: 'زاویه بین خط U1-U1A و خط L1-L1A. افزایش: زاویه بیشتر (بیشتر retroclined). کاهش: زاویه کمتر (بیشتر proclined)' },
    'Occlusal Plane Angle': { mean: '14', sd: '4', severity: 'نرمال', note: 'زاویه بین Occlusal Plane (L1-LMT) و خط S-N. افزایش: صفحه اکلوزال شیب‌دارتر (steep occlusal plane). کاهش: صفحه اکلوزال صاف‌تر (flat occlusal plane)' },
    'Cranial Deflection': { mean: '27', sd: '3', severity: 'نرمال', note: 'زاویه بین Ba-N-S (زاویه در نقطه N). افزایش: انحراف جمجمه بیشتر (greater cranial deflection). کاهش: انحراف جمجمه کمتر (lesser cranial deflection)' },
    'Palatal Plane Angle': { mean: '8', sd: '3', severity: 'نرمال', note: 'زاویه بین خط سقف دهان (ANS-PNS) و خط فرانکفورت (Or-Po). افزایش: سقف دهان شیب‌دارتر (steep palatal plane). کاهش: سقف دهان صاف‌تر (flat palatal plane)' },
    'E-line (UL)': { mean: '-2', sd: '2', severity: 'نرمال', note: 'فاصله از Upper Lip تا خط E-line (Prn-Pog\'). مقدار منفی: لب بالا عقب‌تر از E-line. مقدار مثبت: لب بالا جلوتر از E-line' },
    'E-line (LL)': { mean: '0', sd: '2', severity: 'نرمال', note: 'فاصله از Lower Lip تا خط E-line (Prn-Pog\'). مقدار منفی: لب پایین عقب‌تر از E-line. مقدار مثبت: لب پایین جلوتر از E-line' },
  },
  mcnamara: {
    'Skeletal Convexity': { mean: '170', sd: '5', severity: 'نرمال', note: 'تفسیر: بیشتر از نرمال (زاویه بزرگ‌تر) ⬅️ نیمرخ صاف (Straight) یا حتی مقعر. کمتر از نرمال (زاویه کوچک‌تر) ⬅️ نیمرخ محدب (Convex)، معمولاً کلاس II' },
    'Co-A': { mean: '90', sd: '4', severity: 'نرمال', note: 'طول فک بالا (فاصله Co-A). . افزایش: فک بالا بلندتر (maxillary prognathism). کاهش: فک بالا کوتاه‌تر (maxillary retrognathism)' },
    'Co-Gn': { mean: '120', sd: '5', severity: 'نرمال', note: 'طول فک پایین (فاصله Co-Gn). افزایش: فک پایین بلندتر (mandibular prognathism). کاهش: فک پایین کوتاه‌تر (mandibular retrognathism)' },
    'Lower Face Height': { mean: '65', sd: '4', severity: 'نرمال', note: 'فاصله عمودی صورت پایین (فاصله عمودی ANS-Me). افزایش: ارتفاع بخش پایین صورت بیشتر (long face). کاهش: ارتفاع بخش پایین صورت کمتر (short face)' },
    'Upper Face Height': { mean: '55', sd: '3', severity: 'نرمال', note: 'ارتفاع صورت بالا (N-ANS). افزایش: ارتفاع صورت بالا بیشتر. کاهش: ارتفاع صورت بالا کمتر' },
    'Facial Height Ratio': { mean: '55', sd: '2', severity: 'نرمال', note: 'نسبت ارتفاع صورت (ANS-Me/N-Me × 100). افزایش: نسبت بیشتر (long face pattern). کاهش: نسبت کمتر (short face pattern)' },
    'Mandibular Plane Angle': { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه مندیبولار. افزایش: صفحه مندیبولار شیب‌دار (vertical growth). کاهش: صفحه مندیبولار صاف (horizontal growth)' },
    'PFH/AFH Ratio': { mean: '62', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع خلفی به قدامی (PFH/AFH × 100). PFH: فاصله S-Go (ارتفاع خلفی). AFH: فاصله N-Me (ارتفاع قدامی). افزایش: vertical growth pattern. کاهش: horizontal growth pattern' },
  },
  wits: {
    'AO-BO': { mean: '0', sd: '2', severity: 'نرمال', note: 'اخلاف فاصله نقاط A و B نسبت به اکلوزال پلن (0 = کلاس I). افزایش: کلاس II (ماگزیلا جلوتر یا مندیبل عقب‌تر). کاهش: کلاس III (ماگزیلا عقب‌تر یا مندیبل جلوتر)' },
    'PP/Go-Gn': { mean: '27', sd: '4', severity: 'نرمال', note: 'زاویه بین صفحه پالاتال و صفحه مندیبولار. افزایش: زاویه بیشتر (vertical growth pattern). کاهش: زاویه کمتر (horizontal growth pattern)' },
    'S-Go': { mean: '75', sd: '5', severity: 'نرمال', note: 'ابعاد عمودی چهره (سلا-گناتیون). افزایش: ارتفاع عمودی بیشتر (long face). کاهش: ارتفاع عمودی کمتر (short face)' },
    'Sagittal Jaw': { mean: '0', sd: '2', severity: 'نرمال', note: 'زاویه ساژیتال فک (معمولاً همان ANB). افزایش: کلاس II. کاهش: کلاس III' },
  },
  tweed: {
    FMA: { mean: '25', sd: '4', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به صفحه مندیبولار. افزایش: صورت عمودی (vertical growth pattern). کاهش: صورت افقی (horizontal growth pattern)' },
    FMIA: { mean: '65', sd: '5', severity: 'نرمال', note: 'زاویه صفحه چهره‌ای فرانکفورت نسبت به incisor پایین. افزایش: incisor پایین به سمت عقب (retroclined). کاهش: incisor پایین به سمت جلو (proclined)' },
    IMPA: { mean: '90', sd: '3', severity: 'نرمال', note: 'زاویه incisor پایین نسبت به صفحه مندیبولار. افزایش: incisor پایین به سمت جلو (proclined). کاهش: incisor پایین به سمت عقب (retroclined)' },
  },
  jarabak: {
    'Jarabak Ratio (Facial Height Ratio)': { mean: '63.5', sd: '1.5', severity: 'نرمال', note: 'مهم‌ترین شاخص. (S-Go / N-Me) × ۱۰۰. >۶۵٪ ⬅️ رشد افقی (Hypodivergent). <۵۹٪ ⬅️ رشد عمودی (Hyperdivergent)' },
    'Posterior Facial Height (PFH)': { mean: '75', sd: '5', severity: 'نرمال', note: 'ارتفاع خلف صورت (S-Go). برای محاسبه Jarabak Ratio استفاده می‌شود. افزایش: ارتفاع خلفی بیشتر (vertical growth pattern). کاهش: ارتفاع خلفی کمتر (horizontal growth pattern)' },
    'Anterior Facial Height (AFH)': { mean: '120', sd: '5', severity: 'نرمال', note: 'ارتفاع قدامی صورت (N-Me). برای محاسبه Jarabak Ratio استفاده می‌شود. افزایش: ارتفاع قدامی بیشتر (long face). کاهش: ارتفاع قدامی کمتر (short face)' },
    'Articular Angle': { mean: '143', sd: '6', severity: 'نرمال', note: 'زاویه آرتیکولار (∠S-Ar-Go). ↑ ⬅️ رشد افقی، ↓ ⬅️ رشد عمودی' },
    'Gonial Angle (Total)': { mean: '130', sd: '7', severity: 'نرمال', note: 'زاویه گونیال کل (∠Ar-Go-Me). ↑ ⬅️ رشد عمودی، ↓ ⬅️ رشد افقی' },
    'Upper Gonial Angle': { mean: '53.5', sd: '1.5', severity: 'نرمال', note: 'زاویه گونیال بالا (∠Ar-Go-N)' },
    'Lower Gonial Angle': { mean: '72.5', sd: '2.5', severity: 'نرمال', note: 'زاویه گونیال پایین (∠N-Go-Gn). ↑ ⬅️ رشد عمودی' },
    'Sum of Posterior Angles': { mean: '396', sd: '6', severity: 'نرمال', note: 'مجموع زوایای خلفی (Saddle + Articular + Gonial). >۳۹۶° ⬅️ عمودی، <۳۹۶° ⬅️ افقی' },
    'Y-Axis (Growth Axis)': { mean: '59.5', sd: '3.5', severity: 'نرمال', note: 'محور Y (∠SGn–FH فرانکفورت). ↑ ⬅️ رشد عمودی، ↓ ⬅️ رشد افقی' },
    'Mandibular Plane Angle': { mean: '26.5', sd: '1.5', severity: 'نرمال', note: 'زاویه صفحه مندیبولار (∠FH–MP، Go-Me یا Go-Gn). >۳۲° ⬅️ هایپردایورجنت، <۲۰° ⬅️ هایپودایورجنت' },
    'Basal Plane Angle': { mean: '26.5', sd: '4', severity: 'نرمال', note: 'زاویه صفحه بازال (∠PNS-ANS به Go-Me). مشابه MP angle. افزایش: صفحه بازال شیب‌دارتر (vertical growth pattern). کاهش: صفحه بازال صاف‌تر (horizontal growth pattern)' },
    'Ramus Height': { mean: '50', sd: '5', severity: 'نرمال', note: 'ارتفاع راموس (Ar-Go). افزایش: ارتفاع راموس بیشتر (horizontal growth pattern). کاهش: ارتفاع راموس کمتر (vertical growth pattern)' },
    'Mandibular Arc': { mean: '27', sd: '1', severity: 'نرمال', note: 'زاویه بین Corpus Axis و Condylar Axis. ↑ ⬅️ رشد عمودی' },
    'Palatal Plane to FH': { mean: '0', sd: '3', severity: 'نرمال', note: 'زاویه صفحه پالاتال به FH (∠ANS-PNS به FH). کمی ↓ به خلف' },
    'Occlusal Plane to FH': { mean: '9', sd: '1', severity: 'نرمال', note: 'زاویه صفحه اکلوژال به FH (∠Occlusal Plane به FH)' },
  },
  leganBurstone: {
    'Glabella-Sn-Pog\' (Facial Convexity)': { mean: '12', sd: '4', severity: 'نرمال', note: 'تحدب صورت (Glabella-Sn-Pog\'). افزایش: صورت محدب‌تر. کاهش: صورت مقعرتر' },
    'Sn-Gn\' (Lower Face Height)': { mean: '43', sd: '4', severity: 'نرمال', note: 'ارتفاع صورت پایین (Sn-Gn\')' },
    'Cm-Sn-UL (Upper Lip Protrusion)': { mean: '6', sd: '2', severity: 'نرمال', note: 'برجستگی لب بالا (Cm-Sn-UL)' },
    'Sn-Me\' (Lower Face Height)': { mean: '65', sd: '5', severity: 'نرمال', note: 'ارتفاع صورت پایین (Sn-Me\')' },
    'Glabella-Sn (Midface Length)': { mean: '55', sd: '4', severity: 'نرمال', note: 'طول صورت میانی (Glabella-Sn)' },
    'Sn-Pog\' (Lower Face Length)': { mean: '50', sd: '4', severity: 'نرمال', note: 'طول صورت پایین (Sn-Pog\')' },
    'Nasolabial Angle': { mean: '102', sd: '8', severity: 'نرمال', note: 'زاویه نازولبیال' },
    'Z-Angle': { mean: '75', sd: '5', severity: 'نرمال', note: 'زاویه Z (زاویه بین Or-Po و Pog\'-UL)' },
    'Total Soft-Tissue Convexity': { mean: '165', sd: '6', severity: 'نرمال', note: 'تحدب کل بافت نرم (Pog\'-Prn-G)' },
    'Lower Face Throat Angle': { mean: '100', sd: '7', severity: 'نرمال', note: 'زاویه گلو صورت پایین (زاویه بین Sn–Gn\' و Gn\'–C)' },
    'Facial Contour Angle': { mean: '12', sd: '4', severity: 'نرمال', note: 'زاویه کانتور صورت (N′−Prn−Pog′)' },
    'Lower Face Height / Total Face Height': { mean: '57.5', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع صورت پایین به کل (Sn-Me\' / G-Me\')' },
    'Midface Height / Total Face Height': { mean: '42.5', sd: '3', severity: 'نرمال', note: 'نسبت ارتفاع صورت میانی به کل (G-Sn / G-Me\')' },
    'Lower Face Height / Midface Height': { mean: '1.35', sd: '0.15', severity: 'نرمال', note: 'نسبت ارتفاع صورت پایین به میانی (Sn-Me\' / G-Sn)' },
    'Lip Height / Lower Face Height': { mean: '0.35', sd: '0.05', severity: 'نرمال', note: 'نسبت ارتفاع لب به صورت پایین (UL-LL / Sn-Me\')' },
    'Upper Lip Length / Lower Lip–Chin Length': { mean: '0.42', sd: '0.05', severity: 'نرمال', note: 'نسبت طول لب بالا به طول لب پایین-چانه (Sn-UL / LL-Me\')' },
  },
  arnettMcLaughlin: {
    'Upper Lip to E-line': { mean: '-2', sd: '2', severity: 'نرمال', note: 'فاصله لب بالا تا E-line (Pn-Pog\'). مقدار منفی: عقب‌تر از E-line' },
    'Lower Lip to E-line': { mean: '0', sd: '2', severity: 'نرمال', note: 'فاصله لب پایین تا E-line (Pn-Pog\')' },
    'Nasolabial Angle': { mean: '102', sd: '8', severity: 'نرمال', note: 'زاویه نازولبیال (Cm-Sn-UL)' },
    'Chin Prominence': { mean: '0', sd: '2', severity: 'نرمال', note: 'برجستگی چانه (Pog\' to N-Pog\')' },
    'Soft Facial Convexity (Glabella-Sn-Pog\')': { mean: '165', sd: '5', severity: 'نرمال', note: 'تحدب بافت نرم صورت (Glabella-Sn-Pog\')' },
    'Lower Face Height (Sn-Me\')': { mean: '65', sd: '5', severity: 'نرمال', note: 'ارتفاع صورت پایین (Sn-Me\')' },
    'Upper Lip Protrusion': { mean: '6', sd: '2', severity: 'نرمال', note: 'برجستگی لب بالا (Cm-Sn-UL)' },
  },
  holdaway: {
    'H-angle': { mean: '7.5', sd: '4.5', severity: 'نرمال', note: 'زاویه بین خط N-Pog و H-line (خط از Pog\' تا UL). افزایش: لب‌ها جلوتر. کاهش: لب‌ها عقب‌تر' },
    'Upper Lip to H-line': { mean: '0', sd: '2', severity: 'نرمال', note: 'فاصله لب بالا تا H-line. مقدار مثبت: جلوتر از H-line' },
    'Lower Lip to H-line': { mean: '0', sd: '2', severity: 'نرمال', note: 'فاصله لب پایین تا H-line. مقدار مثبت: جلوتر از H-line' },
    'Soft Tissue Facial Angle': { mean: '90', sd: '4', severity: 'نرمال', note: 'زاویه بین خط N-Pog\' و Frankfort Horizontal' },
    'Soft Tissue Chin Thickness': { mean: '10', sd: '2', severity: 'نرمال', note: 'فاصله بین Pog و Pog\'' },
  },
  softTissueAngular: {
    'Nasolabial Angle': { mean: '102', sd: '8', severity: 'نرمال', note: 'زاویه نازولبیال (Cm-Sn-UL). افزایش: لب بالا عقب‌تر. کاهش: لب بالا جلوتر' },
    'Mentolabial Angle': { mean: '130', sd: '8', severity: 'نرمال', note: 'زاویه منتولبیال (LL-Pog\'-Me\'). افزایش: چانه جلوتر. کاهش: چانه عقب‌تر' },
    'Soft Tissue Chin Angle': { mean: '12', sd: '4', severity: 'نرمال', note: 'زاویه چانه بافت نرم (Pog\'-Sn-N\')' },
    'Upper Lip Angle': { mean: '110', sd: '10', severity: 'نرمال', note: 'زاویه لب بالا (Cm-Sn-UL)' },
    'Lower Lip Angle': { mean: '120', sd: '10', severity: 'نرمال', note: 'زاویه لب پایین (Sn-LL-Pog\')' },
    'Total Facial Convexity': { mean: '165', sd: '6', severity: 'نرمال', note: 'مجموع زوایای تحدب صورت (Glabella-Sn-Pog\')' },
  },
  sassouni: {
    'Restriction Angle': { mean: '123', sd: '5', severity: 'نرمال', note: 'زاویه محدود کننده (زاویه بین N-Ar-Go). افزایش: رشد عمودی صورت. کاهش: رشد افقی صورت' },
    'Sagittal Differentiation Angle': { mean: '59', sd: '4', severity: 'نرمال', note: 'این زاویه بیانگر شیب فک پایین نسبت به قاعده جمجمه است. کاربرد در Sassouni: بررسی الگوی رشد عمودی یا افقی فک پایین. اگر زاویه زیاد باشد ⬅️ رشد عمودی. اگر زاویه کم باشد ⬅️ رشد افقی' },
    'Social Selection Angle': { mean: '4', sd: '2', severity: 'نرمال', note: 'زاویه انتخاب اجتماعی (نسبت Go-Co به Go-Gn). افزایش: نسبت بیشتر. کاهش: نسبت کمتر' },
    'Cultural Ideal Angle': { mean: '90', sd: '5', severity: 'نرمال', note: 'زاویه ایدئال فرهنگی (زاویه بین N-Co و Go-Co). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
    'First Sagittal Angle': { mean: '74', sd: '4', severity: 'نرمال', note: 'زاویه نخستین ساژیتال (زاویه بین Ar-Co و Co-Gn). افزایش: زاویه بیشتر. کاهش: زاویه کمتر' },
  },
};

// Export template for use in visualizer
export { cephalometricTemplates };

// Function to generate interpretation based on parameter and severity
const getInterpretation = (parameter, severity, measured, mean, sd) => {
  if (!severity || severity === 'تعریف نشده' || !measured || measured === '') {
    return '-';
  }

  if (severity === 'نرمال') {
    return 'در محدوده نرمال';
  }

  const measuredNum = parseFloat(measured);
  const meanNum = parseFloat(mean);
  
  // Interpretation mappings for all parameters
  const interpretations = {
    'SNA': {
      high: 'ماگزیلا جلو رفته است',
      low: 'ماگزیلا عقب رفته است',
    },
    'SNB': {
      high: 'مندیبل جلو رفته است',
      low: 'مندیبل عقب رفته است',
    },
    'ANB': {
      high: 'کلاس II اسکلتال (ماگزیلا جلوتر یا مندیبل عقب‌تر)',
      low: 'کلاس III اسکلتال (ماگزیلا عقب‌تر یا مندیبل جلوتر)',
    },
    'U1-SN': {
      high: 'دندان سانترال فک بالا به سمت جلو (proclined)',
      low: 'دندان سانترال فک بالا به سمت عقب (retroclined)',
    },
    'L1-MP': {
      high: 'دندان سانترال فک پایین به سمت جلو (proclined)',
      low: 'دندان سانترال فک پایین به سمت عقب (retroclined)',
    },
    'IMPA': {
      high: 'دندان سانترال فک پایین به سمت جلو (proclined)',
      low: 'دندان سانترال فک پایین به سمت عقب (retroclined)',
    },
    'GoGn-SN': {
      high: 'صفحه مندیبولار شیب‌دار (الگوی رشد عمودی)',
      low: 'صفحه مندیبولار صاف (الگوی رشد افقی)',
    },
    'Overbite': {
      high: 'اوربایت زیاد (deep bite)',
      low: 'اوربایت کم یا اپن بایت (open bite)',
    },
    'Overjet': {
      high: 'اورجت زیاد (protrusion)',
      low: 'اورجت کم یا کراس بایت قدامی',
    },
    'FMA': {
      high: 'الگوی رشد عمودی صورت',
      low: 'الگوی رشد افقی صورت',
    },
    'FMIA': {
      high: 'دندان سانترال فک پایین به سمت عقب (retroclined)',
      low: 'دندان سانترال فک پایین به سمت جلو (proclined)',
    },
    'Facial Axis': {
      high: 'الگوی رشد عمودی صورت',
      low: 'الگوی رشد افقی صورت',
    },
    'Facial Depth': {
      high: 'صورت عمیق‌تر (deep face)',
      low: 'صورت کم‌عمق‌تر (shallow face)',
    },
    'Lower Face Height': {
      high: 'ارتفاع صورت پایین بیشتر (long face)',
      low: 'ارتفاع صورت پایین کمتر (short face)',
    },
    'Mandibular Plane': {
      high: 'صفحه مندیبولار شیب‌دار (steep mandibular plane)',
      low: 'صفحه مندیبولار صاف (flat mandibular plane)',
    },
    'Convexity': {
      high: 'صورت محدب‌تر (convex profile - کلاس II)',
      low: 'صورت مقعرتر (concave profile - کلاس III)',
    },
    'Upper Incisor': {
      high: 'دندان سانترال بالا به سمت جلو (proclined)',
      low: 'دندان سانترال بالا به سمت عقب (retroclined)',
    },
    'Lower Incisor': {
      high: 'دندان سانترال پایین به سمت جلو (proclined)',
      low: 'دندان سانترال پایین به سمت عقب (retroclined)',
    },
    'Interincisal Angle': {
      high: 'زاویه بین اینسایزورها بیشتر است (بیشتر retroclined)',
      low: 'زاویه بین اینسایزورها کمتر است (بیشتر proclined)',
    },
    'Gonial Angle': {
      high: 'زاویه گونیال بازتر است (رشد عمودی)',
      low: 'زاویه گونیال بسته‌تر است (رشد افقی)',
    },
    'Gonial Angle (Ar-Go-Me)': {
      high: 'زاویه گونیال بازتر است (رشد عمودی شدید، اپن بایت)',
      low: 'زاویه گونیال بسته‌تر است (رشد افقی، دیپ بایت شدید)',
    },
    'Gonial Angle (Total)': {
      high: 'زاویه گونیال بازتر است (رشد عمودی)',
      low: 'زاویه گونیال بسته‌تر است (رشد افقی)',
    },
    'Lower Gonial Angle': {
      high: 'زاویه گونیال پایین بیشتر است (رشد عمودی)',
      low: 'زاویه گونیال پایین کمتر است (رشد افقی)',
    },
    'Mandibular Plane Angle': {
      high: 'صفحه مندیبولار شیب‌دار است (vertical growth pattern)',
      low: 'صفحه مندیبولار صاف است (horizontal growth pattern)',
    },
    'Mandibular Plane Angle (FH to MP)': {
      high: 'صفحه مندیبولار شیب‌دار است (هایپردایورجنت)',
      low: 'صفحه مندیبولار صاف است (هایپودایورجنت)',
    },
    'AO-BO': {
      high: 'کلاس II اسکلتال (ماگزیلا جلوتر یا مندیبل عقب‌تر)',
      low: 'کلاس III اسکلتال (ماگزیلا عقب‌تر یا مندیبل جلوتر)',
    },
    'Sagittal Jaw': {
      high: 'کلاس II اسکلتال',
      low: 'کلاس III اسکلتال',
    },
    'Facial Height Ratio': {
      high: 'الگوی صورت بلند (long face pattern)',
      low: 'الگوی صورت کوتاه (short face pattern)',
    },
    'Facial Height Ratio (Jarabak Ratio)': {
      high: 'الگوی رشد افقی (Hypodivergent)',
      low: 'الگوی رشد عمودی (Hyperdivergent)',
    },
    'Jarabak Ratio (Facial Height Ratio)': {
      high: 'الگوی رشد افقی (Hypodivergent)',
      low: 'الگوی رشد عمودی (Hyperdivergent)',
    },
    'Y-Axis (SGn to FH)': {
      high: 'رشد عمودی صورت بیشتر است',
      low: 'رشد افقی صورت بیشتر است',
    },
    'Y-Axis (Growth Axis)': {
      high: 'رشد عمودی صورت (صورت بلند)',
      low: 'رشد افقی صورت (صورت کوتاه)',
    },
    'Occlusal Plane Angle': {
      high: 'صفحه اکلوزال شیب‌دارتر است (steep occlusal plane)',
      low: 'صفحه اکلوزال صاف‌تر است (flat occlusal plane)',
    },
    'Cranial Deflection': {
      high: 'انحراف جمجمه بیشتر است (greater cranial deflection)',
      low: 'انحراف جمجمه کمتر است (lesser cranial deflection)',
    },
    'Palatal Plane Angle': {
      high: 'سقف دهان شیب‌دارتر است (steep palatal plane)',
      low: 'سقف دهان صاف‌تر است (flat palatal plane)',
    },
    'E-line (UL)': {
      high: 'لب بالا جلوتر از E-line است',
      low: 'لب بالا عقب‌تر از E-line است',
    },
    'E-line (LL)': {
      high: 'لب پایین جلوتر از E-line است',
      low: 'لب پایین عقب‌تر از E-line است',
    },
    'Skeletal Convexity': {
      high: 'نیمرخ صاف (Straight) یا حتی concave',
      low: 'نیمرخ محدب (Convex)، معمولاً کلاس II',
    },
    'Co-A': {
      high: 'فک بالا بلندتر است (maxillary prognathism)',
      low: 'فک بالا کوتاه‌تر است (maxillary retrognathism)',
    },
    'Co-Gn': {
      high: 'فک پایین بلندتر است (mandibular prognathism)',
      low: 'فک پایین کوتاه‌تر است (mandibular retrognathism)',
    },
    'Upper Face Height': {
      high: 'ارتفاع قسمت فوقانی صورت بیشتر است',
      low: 'ارتفاع قسمت فوقانی صورت کمتر است',
    },
    'PFH/AFH Ratio': {
      high: 'vertical growth pattern',
      low: 'horizontal growth pattern',
    },
    'PP/Go-Gn': {
      high: 'vertical growth pattern',
      low: 'horizontal growth pattern',
    },
    'S-Go': {
      high: 'ارتفاع عمودی صورت بیشتر است (long face)',
      low: 'ارتفاع عمودی صورت کمتر است (short face)',
    },
    'Saddle Angle (N-S-Ar)': {
      high: 'زاویه سدل بیشتر است (کلاس II اسکلتال و رشد عقب‌گرد)',
      low: 'زاویه سدل کمتر است (کلاس III و رشد جلوگرد)',
    },
    'Saddle Angle': {
      high: 'زاویه سدل بیشتر است (عقب‌گرد)',
      low: 'زاویه سدل کمتر است (جلوگرد)',
    },
    'Articular Angle (S-Ar-Go)': {
      high: 'زاویه آرتیکولار بیشتر است (رشد افقی - هایپودایورجنت)',
      low: 'زاویه آرتیکولار کمتر است (رشد عمودی - هایپردایورجنت)',
    },
    'Articular Angle': {
      high: 'زاویه آرتیکولار بیشتر است (رشد افقی)',
      low: 'زاویه آرتیکولار کمتر است (رشد عمودی)',
    },
    'Sum of Angles (Posterior Angle Sum)': {
      high: 'مجموع زوایای خلفی بیشتر است (الگوی رشد عمودی)',
      low: 'مجموع زوایای خلفی کمتر است (الگوی رشد افقی)',
    },
    'Sum of Posterior Angles': {
      high: 'مجموع زوایای خلفی بیشتر است (الگوی رشد عمودی)',
      low: 'مجموع زوایای خلفی کمتر است (الگوی رشد افقی)',
    },
    'Ramus Height (Ar-Go)': {
      high: 'ارتفاع راموس بیشتر است (رشد افقی)',
      low: 'ارتفاع راموس کمتر است (رشد عمودی)',
    },
    'Ramus Height': {
      high: 'ارتفاع راموس بیشتر است (رشد افقی)',
      low: 'ارتفاع راموس کمتر است (رشد عمودی)',
    },
    'Mandibular Arc': {
      high: 'زاویه مندیبولار آرک بیشتر است (رشد عمودی)',
      low: 'زاویه مندیبولار آرک کمتر است (رشد افقی)',
    },
    'Palatal Plane to FH': {
      high: 'سقف دهان شیب‌دارتر به سمت قدام است',
      low: 'سقف دهان شیب‌دارتر به سمت خلف است',
    },
    'Occlusal Plane to FH': {
      high: 'صفحه اکلوزال شیب‌دارتر است',
      low: 'صفحه اکلوزال صاف‌تر است',
    },
    'Glabella-Sn-Pog\' (Facial Convexity)': {
      high: 'صورت محدب‌تر است',
      low: 'صورت مقعرتر است',
    },
    'Cm-Sn-UL (Upper Lip Protrusion)': {
      high: 'لب بالا برجسته‌تر است',
      low: 'لب بالا عقب‌تر است',
    },
    'Upper Lip Protrusion': {
      high: 'لب بالا برجسته‌تر است',
      low: 'لب بالا عقب‌تر است',
    },
    'Sn-Me\' (Lower Face Height)': {
      high: 'ارتفاع صورت پایین بیشتر است',
      low: 'ارتفاع صورت پایین کمتر است',
    },
    'Lower Face Height (Sn-Me\')': {
      high: 'ارتفاع صورت پایین بیشتر است',
      low: 'ارتفاع صورت پایین کمتر است',
    },
    'Sn-Pog\' (Lower Face Length)': {
      high: 'طول صورت پایین بیشتر است',
      low: 'طول صورت پایین کمتر است',
    },
    'Nasolabial Angle': {
      high: 'زاویه نازولبیال بیشتر است (لب بالا عقب‌تر)',
      low: 'زاویه نازولبیال کمتر است (لب بالا جلوتر)',
    },
    'Z-Angle': {
      high: 'زاویه Z بیشتر است',
      low: 'زاویه Z کمتر است',
    },
    'Total Soft-Tissue Convexity': {
      high: 'تحدب کل بافت نرم بیشتر است',
      low: 'تحدب کل بافت نرم کمتر است',
    },
    'Lower Face Throat Angle': {
      high: 'زاویه گلو صورت پایین بیشتر است',
      low: 'زاویه گلو صورت پایین کمتر است',
    },
    'Facial Contour Angle': {
      high: 'زاویه کانتور صورت بیشتر است',
      low: 'زاویه کانتور صورت کمتر است',
    },
    'Upper Lip to E-line': {
      high: 'لب بالا جلوتر از E-line است',
      low: 'لب بالا عقب‌تر از E-line است',
    },
    'Lower Lip to E-line': {
      high: 'لب پایین جلوتر از E-line است',
      low: 'لب پایین عقب‌تر از E-line است',
    },
    'Chin Prominence': {
      high: 'چانه برجسته‌تر است',
      low: 'چانه عقب‌تر است',
    },
    'H-angle': {
      high: 'زاویه H بیشتر است (لب‌ها جلوتر)',
      low: 'زاویه H کمتر است (لب‌ها عقب‌تر)',
    },
    'Upper Lip to H-line': {
      high: 'لب بالا جلوتر از H-line است',
      low: 'لب بالا عقب‌تر از H-line است',
    },
    'Lower Lip to H-line': {
      high: 'لب پایین جلوتر از H-line است',
      low: 'لب پایین عقب‌تر از H-line است',
    },
    'Soft Tissue Facial Angle': {
      high: 'زاویه صورت بافت نرم بیشتر است',
      low: 'زاویه صورت بافت نرم کمتر است',
    },
    'Soft Tissue Chin Thickness': {
      high: 'ضخامت چانه بافت نرم بیشتر است',
      low: 'ضخامت چانه بافت نرم کمتر است',
    },
    'Mentolabial Angle': {
      high: 'زاویه منتولبیال بیشتر است (چانه جلوتر)',
      low: 'زاویه منتولبیال کمتر است (چانه عقب‌تر)',
    },
    'Facial Convexity Angle': {
      high: 'صورت محدب‌تر است',
      low: 'صورت مقعرتر است',
    },
    'Soft Tissue Chin Angle': {
      high: 'زاویه چانه بافت نرم بیشتر است',
      low: 'زاویه چانه بافت نرم کمتر است',
    },
    'Upper Lip Angle': {
      high: 'زاویه لب بالا بیشتر است',
      low: 'زاویه لب بالا کمتر است',
    },
    'Lower Lip Angle': {
      high: 'زاویه لب پایین بیشتر است',
      low: 'زاویه لب پایین کمتر است',
    },
    'Saddle Angle Interpretation': {
      high: 'زاویه سدل بیشتر است (کلاس II اسکلتال و رشد عقب‌گرد)',
      low: 'زاویه سدل کمتر است (کلاس III اسکلتال و رشد جلوگرد)',
    },
    'Restriction Angle': {
      high: 'زاویه محدود کننده بیشتر است (رشد عمودی صورت یا هایپردایورجنت)',
      low: 'زاویه محدود کننده کمتر است (رشد افقی صورت یا هایپودایورجنت)',
    },
    'Sagittal Differentiation Angle': {
      high: 'زاویه زیاد است ⬅️ رشد عمودی فک پایین',
      low: 'زاویه کم است ⬅️ رشد افقی فک پایین',
    },
    'Social Selection Angle': {
      high: 'زاویه انتخاب اجتماعی بیشتر است (رشد عمودی)',
      low: 'زاویه انتخاب اجتماعی کمتر است (رشد افقی)',
    },
    'Cultural Ideal Angle': {
      high: 'زاویه ایدئال فرهنگی بیشتر است (رشد عمودی)',
      low: 'زاویه ایدئال فرهنگی کمتر است (رشد افقی)',
    },
    'First Sagittal Angle': {
      high: 'زاویه نخستین ساژیتال بیشتر است (رشد عمودی)',
      low: 'زاویه نخستین ساژیتال کمتر است (رشد افقی)',
    },
    // برای سازگاری با داده‌های قدیمی
    'N-S-Ar': {
      high: 'زاویه سدل بیشتر است (کلاس II اسکلتال و رشد عقب‌گرد)',
      low: 'زاویه سدل کمتر است (کلاس III اسکلتال و رشد جلوگرد)',
    },
    'N-Ar-Go': {
      high: 'زاویه محدود کننده بیشتر است (رشد عمودی صورت یا هایپردایورجنت)',
      low: 'زاویه محدود کننده کمتر است (رشد افقی صورت یا هایپودایورجنت)',
    },
    'Go-Co-N-S': {
      high: 'تمایز ساژیتال بیشتر است (رشد عمودی صورت)',
      low: 'تمایز ساژیتال کمتر است (رشد افقی صورت)',
    },
    'Go-Co-Go-Gn': {
      high: 'نسبت Go-Co به Go-Gn بیشتر است (رشد عمودی)',
      low: 'نسبت Go-Co به Go-Gn کمتر است (رشد افقی)',
    },
    'N-Co-Go-Co': {
      high: 'ایدئال فرهنگی بیشتر است (رشد عمودی)',
      low: 'ایدئال فرهنگی کمتر است (رشد افقی)',
    },
    'Ar-Co-Co-Gn': {
      high: 'نخستین ساژیتال بیشتر است (رشد عمودی)',
      low: 'نخستین ساژیتال کمتر است (رشد افقی)',
    },
    'Sn-Gn\' (Lower Face Height)': {
      high: 'ارتفاع صورت پایین بیشتر است',
      low: 'ارتفاع صورت پایین کمتر است',
    },
    'Glabella-Sn (Midface Length)': {
      high: 'طول صورت میانی بیشتر است',
      low: 'طول صورت میانی کمتر است',
    },
    'Total Facial Convexity': {
      high: 'مجموع زوایای تحدب صورت بیشتر است',
      low: 'مجموع زوایای تحدب صورت کمتر است',
    },
    'Lower Face Height / Total Face Height': {
      high: 'نسبت ارتفاع صورت پایین به کل بیشتر است',
      low: 'نسبت ارتفاع صورت پایین به کل کمتر است',
    },
    'Midface Height / Total Face Height': {
      high: 'نسبت ارتفاع صورت میانی به کل بیشتر است',
      low: 'نسبت ارتفاع صورت میانی به کل کمتر است',
    },
    'Lower Face Height / Midface Height': {
      high: 'نسبت ارتفاع صورت پایین به میانی بیشتر است',
      low: 'نسبت ارتفاع صورت پایین به میانی کمتر است',
    },
    'Lip Height / Lower Face Height': {
      high: 'نسبت ارتفاع لب به صورت پایین بیشتر است',
      low: 'نسبت ارتفاع لب به صورت پایین کمتر است',
    },
    'Upper Lip Length / Lower Lip–Chin Length': {
      high: 'نسبت طول لب بالا به طول لب پایین-چانه بیشتر است',
      low: 'نسبت طول لب بالا به طول لب پایین-چانه کمتر است',
    },
    'Basal Plane Angle (PNS-ANS to Go-Me)': {
      high: 'زاویه صفحه بازال بیشتر است',
      low: 'زاویه صفحه بازال کمتر است',
    },
    'Basal Plane Angle': {
      high: 'زاویه صفحه بازال بیشتر است',
      low: 'زاویه صفحه بازال کمتر است',
    },
    'Anterior Facial Height (N-Me)': {
      high: 'ارتفاع قدامی صورت بیشتر است',
      low: 'ارتفاع قدامی صورت کمتر است',
    },
    'Posterior Facial Height (S-Go)': {
      high: 'ارتفاع خلف صورت بیشتر است',
      low: 'ارتفاع خلف صورت کمتر است',
    },
    'Posterior Facial Height (PFH)': {
      high: 'ارتفاع خلف صورت بیشتر است',
      low: 'ارتفاع خلف صورت کمتر است',
    },
    'Anterior Facial Height (AFH)': {
      high: 'ارتفاع قدامی صورت بیشتر است',
      low: 'ارتفاع قدامی صورت کمتر است',
    },
    'Upper Gonial Angle': {
      high: 'زاویه گونیال بالا بیشتر است',
      low: 'زاویه گونیال بالا کمتر است',
    },
  };

  // Check if we have a specific interpretation for this parameter
  const paramInterpretation = interpretations[parameter];
  if (paramInterpretation) {
    if (severity === 'بالا') {
      return paramInterpretation.high;
    }
    if (severity === 'پایین') {
      return paramInterpretation.low;
    }
  }

  // If no specific interpretation found, try to generate one from parameter name
  // This is a fallback for any parameters not explicitly defined above
  if (severity === 'بالا') {
    // Try to infer meaning from parameter name
    const paramLower = parameter.toLowerCase();
    if (paramLower.includes('angle') || paramLower.includes('زاویه')) {
      return 'زاویه بیشتر از نرمال است';
    }
    if (paramLower.includes('height') || paramLower.includes('ارتفاع')) {
      return 'ارتفاع بیشتر از نرمال است';
    }
    if (paramLower.includes('length') || paramLower.includes('طول')) {
      return 'طول بیشتر از نرمال است';
    }
    if (paramLower.includes('ratio') || paramLower.includes('نسبت')) {
      return 'نسبت بیشتر از نرمال است';
    }
    if (paramLower.includes('distance') || paramLower.includes('فاصله')) {
      return 'فاصله بیشتر از نرمال است';
    }
    return 'مقدار بیشتر از نرمال است';
  }
  if (severity === 'پایین') {
    // Try to infer meaning from parameter name
    const paramLower = parameter.toLowerCase();
    if (paramLower.includes('angle') || paramLower.includes('زاویه')) {
      return 'زاویه کمتر از نرمال است';
    }
    if (paramLower.includes('height') || paramLower.includes('ارتفاع')) {
      return 'ارتفاع کمتر از نرمال است';
    }
    if (paramLower.includes('length') || paramLower.includes('طول')) {
      return 'طول کمتر از نرمال است';
    }
    if (paramLower.includes('ratio') || paramLower.includes('نسبت')) {
      return 'نسبت کمتر از نرمال است';
    }
    if (paramLower.includes('distance') || paramLower.includes('فاصله')) {
      return 'فاصله کمتر از نرمال است';
    }
    return 'مقدار کمتر از نرمال است';
  }

  return '-';
};

export function CephalometricAnalysisView() {
  const { id } = useParams();
  const { user } = useAuthContext();
  const navigate = useNavigate();
  const { setHeaderContent, setHideRightButtons } = useHeaderContent();

  // State management
  const [patient, setPatient] = useState(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [selectedAnalysisType, setSelectedAnalysisType] = useState('general');
  const [chartType, setChartType] = useState('radar'); // 'radar', 'area', 'bar-negative' (column negative)
  const [chartNormalization, setChartNormalization] = useState('real'); // 'real' or 'normalized' (percentage)
  const [showCephalometricImage, setShowCephalometricImage] = useState(true);
  const [showCoordinateSystem, setShowCoordinateSystem] = useState(false);
  const [isAnalysisConfirmed, setIsAnalysisConfirmed] = useState(false);
  const [viewMode, setViewMode] = useState('normal'); // 'normal', 'coordinate', 'hard-tissue-only'
  const [tablePage, setTablePage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [selectedImageIndex, setSelectedImageIndex] = useState(0);
  const [activeTab, setActiveTab] = useState('image'); // 'image' or 'table'
  
  // Use deferred value for smoother tab switching
  const deferredActiveTab = useDeferredValue(activeTab);
  
  // 🔧 FIX: Update ref when activeTab changes
  useEffect(() => {
    activeTabRef.current = activeTab;
  }, [activeTab]);
  
  // Ref to track if user has interacted with buttons (to prevent useEffect from overriding user choice)
  const userInteractedRef = useRef(false);
  const saveTimerRef = useRef(null);
  
  // 🔧 FIX: State to track if we're currently uploading an image (to prevent useEffect from resetting selectedImageIndex)
  // Using state instead of ref so React re-renders when it changes
  const [isUploadingImage, setIsUploadingImage] = useState(false);
  
  // Key to force remount of CephalometricAIAnalysis when showing image
  const [analysisKey, setAnalysisKey] = useState(0);
  
  // 🔧 FIX: Use refs to prevent duplicate resets when image changes
  const lastResetImageIndexRef = useRef(selectedImageIndex);
  const lastResetImagesLengthRef = useRef(patient?.lateralImages?.length || 0);

  // Analysis history states
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [selectedAnalysisIndex, setSelectedAnalysisIndex] = useState(null);
  const [isLoadingHistory, setIsLoadingHistory] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [analysisToDelete, setAnalysisToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [showRAGAnalysis, setShowRAGAnalysis] = useState(false);
  
  // 🔧 FIX: Track if we've already initialized display state to prevent re-running
  const hasInitializedDisplayRef = useRef(false);
  
  // 🔧 FIX: Track if user is viewing image (to prevent hiding it automatically)
  const isViewingImageRef = useRef(false);
  
  // 🔧 FIX: Track active tab with ref to avoid stale closure in callbacks
  const activeTabRef = useRef('image');
  
  // 🔧 FIX: Track if we're currently auto-saving to prevent infinite loops
  const isAutoSavingRef = useRef(false);
  
  // 🔧 FIX: Track the last saved analysis timestamp to prevent duplicate saves
  const lastSavedAnalysisTimestampRef = useRef(null);
  
  // 🔧 FIX: Wrapper function for changing selected image that also clears analysis selection
  const handleSelectedImageIndexChange = useCallback((newIndex) => {
    console.log('📷 [Image Selection] Changing image index:', { old: selectedImageIndex, new: newIndex });
    
    // Only clear analysis if the image actually changed
    if (newIndex !== selectedImageIndex) {
      setSelectedAnalysisIndex(null);
      console.log('📷 [Image Selection] Cleared selectedAnalysisIndex (different image)');
    }
    
    setSelectedImageIndex(newIndex);
  }, [selectedImageIndex]);

  // Load analysis history
  const loadAnalysisHistory = useCallback(async (selectLatest = false) => {
    if (!id) return;
    
    setIsLoadingHistory(true);
    try {
      const res = await axios.get(`${endpoints.patients}/${id}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      // Parse analysis history from cephalometricAnalysis field
      const patientData = res.data?.patient || res.data;
      let analyses = [];
      
      if (patientData.cephalometricAnalysis) {
        try {
          // بررسی اینکه آیا داده JSON معتبر است
          const data = patientData.cephalometricAnalysis;
          
          // اگر داده قبلاً یک آبجکت است (نه string)
          if (typeof data === 'object') {
            if (Array.isArray(data)) {
              analyses = data;
            } else {
              const { analyses: dataAnalyses } = data;
              if (dataAnalyses && Array.isArray(dataAnalyses)) {
                analyses = dataAnalyses;
              }
            }
          } 
          // اگر داده string است، سعی کن parse کنی
          else if (typeof data === 'string') {
            // بررسی اگر با { یا [ شروع شود (JSON معتبر)
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
            } else {
              // داده قدیمی (غیر JSON) - نادیده بگیر
              console.warn('⚠️ Old format data detected in cephalometricAnalysis, ignoring:', trimmedData.substring(0, 50));
            }
          }
        } catch (parseError) {
          console.error('Failed to parse cephalometric analysis:', parseError);
          // در صورت خطا، آرایه خالی بمان
        }
      }
      
      // 🔧 DEBUG: Log what we loaded from database
      console.log('📊 [loadAnalysisHistory] Loaded from database:', {
        analysisCount: analyses.length,
        analyses: analyses.map((a, idx) => ({
          index: idx,
          hasLandmarks: !!a.landmarks,
          landmarksCount: a.landmarks ? Object.keys(a.landmarks).length : 0,
          timestamp: a.timestamp,
        })),
      });
      
      setAnalysisHistory(analyses);
      
      // 🔧 FIX: Auto-select the latest analysis (newest one is at the end of array)
      // Always select the latest analysis on page load
      // Use setTimeout to ensure state is updated before setting selectedAnalysisIndex
      if (analyses.length > 0) {
        const latestIndex = analyses.length - 1;
        console.log('🔄 [loadAnalysisHistory] Auto-selecting latest analysis at index:', latestIndex, {
          analysisCount: analyses.length,
          hasLandmarks: !!analyses[latestIndex]?.landmarks,
          hasTableData: !!analyses[latestIndex]?.tableData,
          hasAllTableData: !!analyses[latestIndex]?.allTableData,
        });
        // Use setTimeout to ensure analysisHistory state is updated first
        setTimeout(() => {
          setSelectedAnalysisIndex(latestIndex);
        }, 0);
      } else {
        setSelectedAnalysisIndex(null);
      }
    } catch (err) {
      console.error('Failed to load analysis history:', err);
    } finally {
      setIsLoadingHistory(false);
    }
  }, [id, user?.accessToken]); // 🔧 FIX: Removed selectedAnalysisIndex from deps to prevent infinite loop

  // Save new analysis to history
  // 🔧 FIX: Accept optional patientData parameter to avoid stale closure issues
  const handleSaveAnalysis = useCallback(async (patientData = null) => {
    // Use provided patientData or fall back to state
    const currentPatient = patientData || patient;
    
    if (!id || !currentPatient) return;

    try {
      // Get existing history
      const existingRes = await axios.get(`${endpoints.patients}/${id}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      const patientDataFromDb = existingRes.data?.patient || existingRes.data;
      let existingHistory = [];
      
      if (patientDataFromDb.cephalometricAnalysis) {
        try {
          const data = patientDataFromDb.cephalometricAnalysis;
          
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
            } else {
              console.warn('⚠️ Old format data in cephalometricAnalysis (handleSaveAnalysis), starting fresh');
            }
          }
        } catch (parseError) {
          console.error('Failed to parse existing history:', parseError);
        }
      }

      // Calculate all analysis types from current landmarks
      const allAnalysisTypes = ['general', 'steiner', 'mcnamara', 'ricketts', 'downs', 'tweed', 'jarabak', 'sassouni', 'wits', 'leganBurstone', 'arnettMcLaughlin', 'holdaway', 'softTissueAngular'];
      const allTableData = {};
      
      // If we have landmarks and rawData, calculate all analysis types
      if (currentPatient.cephalometricLandmarks && currentPatient.cephalometricRawData) {
        allAnalysisTypes.forEach(analysisType => {
          const template = cephalometricTemplates[analysisType];
          if (template) {
            const tableForType = {};
            Object.keys(template).forEach(param => {
              let measuredValue = currentPatient.cephalometricRawData[param] || '';
              // 🔧 FIX: برای IMPA، اگر مقدار از rawData می‌آید و کمتر از 90 است، از 180 کم می‌کنیم
              if (param === 'IMPA' && measuredValue && !isNaN(parseFloat(measuredValue))) {
                const numValue = parseFloat(measuredValue);
                if (numValue < 90) {
                  measuredValue = String(180 - numValue);
                }
              }
              tableForType[param] = {
                ...template[param],
                measured: measuredValue,
              };
            });
            allTableData[analysisType] = tableForType;
          }
        });
      }

      // 🔧 FIX: Check if analysis already exists for this image (one analysis per image)
      // Remove existing analysis for the same image
      const filteredHistory = existingHistory.filter(analysis => 
        analysis.imageIndex !== selectedImageIndex
      );
      
      // 🔧 FIX: Limit to 5 analyses (keep newest 5)
      // After filtering, we'll add the new one, so we keep 4 oldest and add the new one
      const historyToKeep = filteredHistory.slice(-4); // Keep 4 oldest (will become 5 with new one)
      
      // Add new analysis to history with all types
      const newAnalysis = {
        id: `analysis_${Date.now()}`,
        timestamp: new Date().toISOString(),
        currentAnalysisType: selectedAnalysisType, // The type user was viewing when saving
        allTableData, // All analysis types calculated from same landmarks
        rawData: currentPatient.cephalometricRawData,
        landmarks: currentPatient.cephalometricLandmarks,
        imageIndex: selectedImageIndex, // 🔧 FIX: Store which image this analysis belongs to
      };
      
      console.log('📊 [handleSaveAnalysis] Saving analysis with landmarks:', {
        landmarksCount: newAnalysis.landmarks ? Object.keys(newAnalysis.landmarks).length : 0,
        hasRawData: !!newAnalysis.rawData,
        hasAllTableData: !!newAnalysis.allTableData,
        imageIndex: selectedImageIndex,
        removedExistingForImage: existingHistory.length - filteredHistory.length,
      });
      
      // 🔧 FIX: Add new analysis and limit to 5 total
      const updatedHistory = [...historyToKeep, newAnalysis].slice(-5); // Keep only 5 newest

      // 🔧 DEBUG: Log what we're about to save
      console.log('📊 [handleSaveAnalysis] About to save to database:', {
        historyLength: updatedHistory.length,
        newAnalysisHasLandmarks: !!newAnalysis.landmarks,
        newAnalysisLandmarksCount: newAnalysis.landmarks ? Object.keys(newAnalysis.landmarks).length : 0,
        stringifiedLength: JSON.stringify(updatedHistory).length,
      });

      // Save to database using existing endpoint
      await axios.put(
        `${endpoints.patients}/${id}`,
        { cephalometricAnalysis: JSON.stringify(updatedHistory) },
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      console.log('✅ Analysis saved to history (all types)');
      toast.success('آنالیز با موفقیت ذخیره شد ');
      
      // 🔧 FIX: Reload history but DON'T auto-select to prevent infinite loops
      // Auto-selecting causes the analysis to be loaded, which triggers onLandmarksDetected again
      await loadAnalysisHistory(false); // Pass false to prevent auto-select
    } catch (err) {
      console.error('❌ Failed to save analysis:', err);
      const errorMsg = err.response?.data?.error || err.response?.data?.message || err.message || 'خطای نامشخص';
      toast.error(`خطا در ذخیره آنالیز: ${errorMsg}`);
    }
  }, [id, patient, selectedAnalysisType, selectedImageIndex, user?.accessToken, loadAnalysisHistory]);

  // Delete analysis from history
  const handleDeleteAnalysis = useCallback(async (analysisToDelete) => {
    if (!id) return;

    // Extract analysis ID - handle both object and string
    const analysisId = typeof analysisToDelete === 'string' 
      ? analysisToDelete 
      : (analysisToDelete?.id || analysisToDelete);

    if (!analysisId) {
      console.error('❌ Cannot delete analysis: no ID provided');
      toast.error('خطا: شناسه آنالیز یافت نشد');
      return;
    }

    try {
      setDeleting(true);
      
      // Get existing history
      const existingRes = await axios.get(`${endpoints.patients}/${id}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      const patientData = existingRes.data?.patient || existingRes.data;
      let existingHistory = [];
      
      if (patientData.cephalometricAnalysis) {
        try {
          const data = patientData.cephalometricAnalysis;
          
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
            } else {
              console.warn('⚠️ Old format data in cephalometricAnalysis (handleDeleteAnalysis), starting fresh');
            }
          }
        } catch (parseError) {
          console.error('Failed to parse existing history:', parseError);
        }
      }
      
      // Filter out the analysis to delete
      const updatedHistory = existingHistory.filter((a) => a.id !== analysisId);
      
      // Save updated history
      await axios.put(
        `${endpoints.patients}/${id}`,
        { cephalometricAnalysis: updatedHistory.length > 0 ? JSON.stringify(updatedHistory) : null },
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
            'Content-Type': 'application/json',
          },
        }
      );

      toast.success('آنالیز با موفقیت حذف شد');
      
      // 🔧 FIX: Find the index of the deleted analysis before reloading
      const deletedIndex = existingHistory.findIndex((a) => a.id === analysisId);
      const newHistoryLength = updatedHistory.length;
      
      // Reload history
      await loadAnalysisHistory();
      
      // 🔧 FIX: After deletion, select the appropriate analysis
      // If we deleted the last analysis, select the new last one
      // If we deleted a middle analysis, keep the same index (which will now point to the next analysis)
      // If we deleted the first analysis and it was the only one, set to null
      setSelectedAnalysisIndex(currentIndex => {
        if (newHistoryLength === 0) {
          return null;
        }
        // If we deleted the last item, select the new last item
        if (deletedIndex === existingHistory.length - 1) {
          return newHistoryLength - 1;
        }
        // If we deleted an item before the current selection, adjust the index
        if (deletedIndex < currentIndex) {
          return currentIndex - 1;
        }
        // Otherwise, keep the same index (it will now point to the next item)
        return currentIndex;
      });
    } catch (err) {
      console.error('❌ Failed to delete analysis:', err);
      toast.error('خطا در حذف آنالیز');
    } finally {
      setDeleting(false);
      setDeleteDialogOpen(false);
      setAnalysisToDelete(null);
    }
  }, [id, user?.accessToken, loadAnalysisHistory]);

  // Memoize header content to avoid unnecessary re-renders
  const headerContentElement = useMemo(() => (
    <Stack 
      direction="row" 
      alignItems="center" 
      spacing={{ xs: 1, sm: 2 }} 
      sx={{ 
        flexGrow: 1,
        minWidth: 0, // Allow flex items to shrink below their minimum content size
        overflow: 'hidden', // Prevent overflow
        width: '100%',
        maxWidth: '100%',
        position: 'relative',
      }}
    >
      <IconButton
        onClick={() => navigate(`/dashboard/orthodontics/patient/${id}`)}
        sx={{
          width: { xs: 36, sm: 40 },
          height: { xs: 36, sm: 40 },
          minWidth: { xs: 36, sm: 40 },
          flexShrink: 0, // Prevent icon from shrinking
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
          value={activeTab} 
          onChange={(e, newValue) => {
            // Use startTransition for smoother tab switching
            startTransition(() => {
              setActiveTab(newValue);
              if (newValue === 'table') {
                setShowCephalometricImage(false);
                isViewingImageRef.current = false;
                userInteractedRef.current = true;
              } else if (newValue === 'chart') {
                setShowCephalometricImage(false);
                isViewingImageRef.current = false;
                userInteractedRef.current = true;
              } else if (newValue === 'report') {
                setShowCephalometricImage(false);
                isViewingImageRef.current = false;
                userInteractedRef.current = true;
              } else if (newValue === 'image') {
                setShowCephalometricImage(true);
                isViewingImageRef.current = true; // Mark that user is viewing image
                userInteractedRef.current = true;
                // Clear localStorage flag when user explicitly clicks image tab
                if (id) {
                  localStorage.removeItem(`cephalometric_viewing_tables_${id}`);
                }
              }
            });
          }}
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
            label="تصویر" 
            value="image" 
          />
          <Tab 
            label="جدول" 
            value="table" 
          />
          <Tab 
            label="نمودار" 
            value="chart" 
          />
          <Tab 
            label="گزارش" 
            value="report" 
          />
        </CustomTabs>
      </Box>
    </Stack>
  ), [id, navigate, activeTab]);

  // Set header content when component mounts or activeTab changes
  useEffect(() => {
    setHeaderContent(headerContentElement);
    setHideRightButtons(true); // Hide wallet, notification, profile buttons
    
    // Cleanup: clear header content and show buttons again when component unmounts
    return () => {
      setHeaderContent(null);
      setHideRightButtons(false);
    };
  }, [headerContentElement, setHeaderContent, setHideRightButtons]);

  // Preload components for smoother tab switching
  useEffect(() => {
    // Preload lazy components in the background
    preloadComponents();
  }, []);

  // Load history when component mounts
  useEffect(() => {
    if (id && user?.accessToken) {
      loadAnalysisHistory();
    }
  }, [id, user?.accessToken, loadAnalysisHistory]);

  // Load selected analysis from history
  useEffect(() => {
    if (selectedAnalysisIndex !== null && analysisHistory && analysisHistory.length > 0 && selectedAnalysisIndex < analysisHistory.length) {
      const analysis = analysisHistory[selectedAnalysisIndex];
      
      // 🔧 FIX: Ensure analysis exists before accessing its properties
      if (!analysis) {
        console.warn('Analysis at index', selectedAnalysisIndex, 'is undefined');
        return;
      }
      
      // 🔧 FIX: Change selectedImageIndex to match the analysis's imageIndex
      // This ensures the correct image is displayed when selecting an analysis from history
      if (analysis.imageIndex !== undefined && analysis.imageIndex !== null) {
        const targetImageIndex = analysis.imageIndex;
        if (targetImageIndex !== selectedImageIndex && targetImageIndex >= 0 && patient?.lateralImages && targetImageIndex < patient.lateralImages.length) {
          console.log('🔄 [Load Analysis] Changing image index to match selected analysis:', {
            from: selectedImageIndex,
            to: targetImageIndex,
            analysisIndex: selectedAnalysisIndex,
          });
          setSelectedImageIndex(targetImageIndex);
        }
      }
      
      // Set the analysis type (use currentAnalysisType if available, or fallback to analysisType for old data)
      const typeToUse = analysis.currentAnalysisType || analysis.analysisType || 'steiner';
      setSelectedAnalysisType(typeToUse);
      
      // If allTableData exists (new format), use the table for current analysis type
      let { tableData } = analysis; // fallback for old format
      if (analysis.allTableData && analysis.allTableData[typeToUse]) {
        tableData = analysis.allTableData[typeToUse];
      }
      
      // 🔧 FIX: Update patient state even if tableData is empty, but ensure we have landmarks
      if (analysis.landmarks && Object.keys(analysis.landmarks).length > 0) {
        console.log('📊 [Load Analysis] Loading analysis data:', {
          selectedAnalysisIndex,
          typeToUse,
          hasTableData: !!tableData,
          tableDataKeys: tableData ? Object.keys(tableData).length : 0,
          hasAllTableData: !!analysis.allTableData,
          allTableDataKeys: analysis.allTableData ? Object.keys(analysis.allTableData) : [],
        });
        
        setPatient(prev => ({
          ...prev,
          cephalometricTable: tableData || null,
          cephalometricRawData: analysis.rawData,
          cephalometricLandmarks: analysis.landmarks,
        }));
      }
    }
  }, [selectedAnalysisIndex, analysisHistory, selectedImageIndex, patient?.lateralImages]);

  // Ensure selectedImageIndex is within bounds when lateralImages change
  useEffect(() => {
    // 🔧 FIX: Don't adjust if we're currently uploading an image
    if (isUploadingImage) {
      console.log('🔧 [useEffect] Skipping selectedImageIndex adjustment (upload in progress)');
      return;
    }
    
    if (patient?.lateralImages && patient.lateralImages.length > 0) {
      // 🔧 FIX: Only adjust if index is truly out of bounds
      // Don't reset unnecessarily
      if (selectedImageIndex < 0) {
        console.log('🔧 [useEffect] Index is negative, setting to 0 (newest image)');
        setSelectedImageIndex(0);
      } else if (selectedImageIndex >= patient.lateralImages.length) {
        // Index is beyond array length, set to 0 (newest image after sorting)
        const newIndex = 0;
        console.log('🔧 [useEffect] Index out of bounds, setting to newest image (index 0):', {
          old: selectedImageIndex,
          new: newIndex,
          totalImages: patient.lateralImages.length,
        });
        setSelectedImageIndex(newIndex);
      }
      // Otherwise, keep the current index (don't change it)
    }
  }, [patient?.lateralImages, selectedImageIndex, isUploadingImage]);

  // Reset analysis when selected image changes (but not during upload or when loading from history)
  // 🔧 FIX: Use refs to prevent duplicate resets (defined at component level)
  useEffect(() => {
    // 🔧 FIX: Don't reset during upload
    if (isUploadingImage) {
      console.log('🔧 [Reset Analysis Effect] Skipping reset (upload in progress)');
      return;
    }
    
    // 🔧 FIX: Check if image actually changed to prevent duplicate resets
    const imageIndexChanged = lastResetImageIndexRef.current !== selectedImageIndex;
    const imagesLengthChanged = lastResetImagesLengthRef.current !== (patient?.lateralImages?.length || 0);
    
    // 🔧 FIX: Don't reset if we have analysis history to load
    // If analysisHistory exists and has data, we should load from history instead of resetting
    // Also check if selectedAnalysisIndex is already set (history is being loaded)
    if (analysisHistory && analysisHistory.length > 0) {
      // If selectedAnalysisIndex is already set, we're loading from history - don't reset
      if (selectedAnalysisIndex !== null && selectedAnalysisIndex !== undefined) {
        // Only update refs to prevent duplicate checks, but don't reset
        if (imageIndexChanged) {
          lastResetImageIndexRef.current = selectedImageIndex;
        }
        if (imagesLengthChanged) {
          lastResetImagesLengthRef.current = patient?.lateralImages?.length || 0;
        }
        console.log('🔧 [Reset Analysis Effect] Skipping reset (loading from history):', {
          selectedAnalysisIndex,
          analysisHistoryLength: analysisHistory.length,
          imageIndexChanged,
          imagesLengthChanged,
        });
        return;
      }
      
      // If we have history but selectedAnalysisIndex is not set yet, it might be loading
      // Don't reset if image index hasn't actually changed (only length changed, e.g., initial load)
      if (!imageIndexChanged && imagesLengthChanged) {
        // This is likely an initial load - update refs but don't reset
        lastResetImagesLengthRef.current = patient?.lateralImages?.length || 0;
        console.log('🔧 [Reset Analysis Effect] Skipping reset (initial load with history):', {
          analysisHistoryLength: analysisHistory.length,
          imagesLengthChanged,
        });
        return;
      }
    }
    
    // Only reset if image index actually changed (not just length)
    // Length change alone (e.g., on initial load) shouldn't trigger reset if we have history to load
    if (!imageIndexChanged) {
      // Update refs even if no change to prevent future false positives
      if (imagesLengthChanged) {
        lastResetImagesLengthRef.current = patient?.lateralImages?.length || 0;
      }
      return; // No image index change, skip reset
    }
    
    // Update refs
    lastResetImageIndexRef.current = selectedImageIndex;
    lastResetImagesLengthRef.current = patient?.lateralImages?.length || 0;
    
    if (patient?.lateralImages && patient.lateralImages.length > 0) {
      console.log('🔧 [Reset Analysis Effect] Resetting analysis for image change:', {
        selectedImageIndex,
        totalImages: patient.lateralImages.length,
        imageIndexChanged,
        imagesLengthChanged,
      });
      
      // Reset analysis state when switching images
      setPatient(prev => ({
        ...prev,
        cephalometricTable: null,
        cephalometricRawData: null,
        cephalometricLandmarks: null,
      }));
      setSelectedAnalysisIndex(null);
      setIsAnalysisConfirmed(false);
      setShowCephalometricImage(true);
      setActiveTab('image');
      
      // 🔧 FIX: Only update analysisKey if not already updated by handleImageUpload
      // Check if this is a manual image change (not during upload)
      if (!isUploadingImage) {
        setAnalysisKey(prev => prev + 1);
      }
      
      // Reset localStorage for the new image
      if (id) {
        localStorage.removeItem(`cephalometric_analysis_confirmed_${id}`);
        localStorage.removeItem(`cephalometric_viewing_tables_${id}`);
      }
    }
  }, [selectedImageIndex, patient?.lateralImages, id, isUploadingImage, selectedAnalysisIndex, analysisHistory]);

  // Update table when analysis type changes (for historical data with all types)
  useEffect(() => {
    if (selectedAnalysisIndex !== null && analysisHistory && analysisHistory.length > 0 && selectedAnalysisIndex < analysisHistory.length) {
      const analysis = analysisHistory[selectedAnalysisIndex];
      
      if (!analysis) return;

      console.log('🔄 [Update Table] Analysis type changed:', {
        selectedAnalysisType,
        selectedAnalysisIndex,
        hasAllTableData: !!analysis.allTableData,
        allTableDataKeys: analysis.allTableData ? Object.keys(analysis.allTableData) : [],
        hasTableData: !!analysis.tableData,
        hasRawData: !!analysis.rawData,
      });

      // If this analysis has all table data, switch to the selected type
      if (analysis.allTableData && analysis.allTableData[selectedAnalysisType]) {
        console.log('✅ [Update Table] Using allTableData for type:', selectedAnalysisType);
        setPatient(prev => ({
          ...prev,
          cephalometricTable: analysis.allTableData[selectedAnalysisType],
        }));
      } else if (analysis.tableData && Object.keys(analysis.tableData).length > 0) {
        // Use existing tableData if available (for old format)
        console.log('✅ [Update Table] Using existing tableData');
        setPatient(prev => ({
          ...prev,
          cephalometricTable: analysis.tableData,
        }));
      } else if (analysis.rawData && Object.keys(analysis.rawData).length > 0) {
        // Fallback: calculate from rawData if allTableData not available
        console.log('✅ [Update Table] Calculating from rawData');
        const template = cephalometricTemplates[selectedAnalysisType];
        if (template) {
          const newTable = {};
          Object.keys(template).forEach(param => {
            let measuredValue = analysis.rawData[param] || '';
            // 🔧 FIX: برای IMPA، اگر مقدار از rawData می‌آید و کمتر از 90 است، از 180 کم می‌کنیم
            if (param === 'IMPA' && measuredValue && !isNaN(parseFloat(measuredValue))) {
              const numValue = parseFloat(measuredValue);
              if (numValue < 90) {
                measuredValue = String(180 - numValue);
              }
            }
            newTable[param] = {
              ...template[param],
              measured: measuredValue,
            };
          });
          setPatient(prev => ({
            ...prev,
            cephalometricTable: newTable,
          }));
        }
      } else {
        // Fallback: create empty table from template if no data available
        console.warn('⚠️ [Update Table] No table data available, creating empty table from template');
        const template = cephalometricTemplates[selectedAnalysisType];
        if (template) {
          const newTable = {};
          Object.keys(template).forEach(param => {
            newTable[param] = {
              ...template[param],
              measured: '',
            };
          });
          setPatient(prev => ({
            ...prev,
            cephalometricTable: newTable,
          }));
        } else {
          console.error('❌ [Update Table] Template not found for analysis type:', selectedAnalysisType);
        }
      }
    } else if (selectedAnalysisIndex === null) {
      // If no analysis is selected, create empty table from template
      const template = cephalometricTemplates[selectedAnalysisType];
      if (template) {
        const newTable = {};
        Object.keys(template).forEach(param => {
          newTable[param] = {
            ...template[param],
            measured: '',
          };
        });
        setPatient(prev => ({
          ...prev,
          cephalometricTable: newTable,
        }));
      }
    }
  }, [selectedAnalysisType, selectedAnalysisIndex, analysisHistory]);

  // Handle image delete
  const handleDeleteImage = useCallback(async (image) => {
    if (!image?.id || !id) return;

    try {
      await axios.delete(`${endpoints.patients}/${id}/images`, {
        data: { imageId: image.id },
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'application/json',
        },
      });

      // Refresh images after deletion
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      const images = imagesResponse.data.images || [];
      const lateralImages = images.filter(img => 
        img.category === 'lateral' || 
        img.category === 'cephalometric' || 
        img.category === 'cephalometry'
      );

      // Find deleted index before updating state
      const deletedIndex = patient?.lateralImages?.findIndex(img => img.id === image.id) ?? -1;

      // Update patient state
      setPatient(prev => ({
        ...prev,
        lateralImages,
      }));

      // Adjust selected index if needed
      if (lateralImages.length > 0) {
        if (deletedIndex >= 0 && deletedIndex === selectedImageIndex) {
          // If deleted image was selected, select the first available or adjust index
          const newIndex = Math.min(selectedImageIndex, lateralImages.length - 1);
          setSelectedImageIndex(newIndex);
        } else if (deletedIndex >= 0 && deletedIndex < selectedImageIndex) {
          // If deleted image was before selected, adjust index
          setSelectedImageIndex(selectedImageIndex - 1);
        }
      } else {
        // No images left
        setSelectedImageIndex(0);
      }

      // Always force refresh of analysis component
      setAnalysisKey(prev => prev + 1);

      toast.success('تصویر با موفقیت حذف شد');
    } catch (error) {
      console.error('Error deleting image:', error);
      toast.error('خطا در حذف تصویر');
    }
  }, [id, user?.accessToken, patient?.lateralImages, selectedImageIndex]);

  // Handle image upload
  const handleImageUpload = useCallback(async (files, category = 'lateral') => {
    if (!files || files.length === 0 || !id) return;

    // 🔧 FIX: Check limit of 5 images
    const currentImages = patient?.lateralImages || [];
    if (currentImages.length >= 5) {
      toast.error('حداکثر 5 تصویر لترال سفالومتری مجاز است. لطفاً ابتدا یک تصویر را حذف کنید.');
      return;
    }

    // 🔧 FIX: Limit number of files to upload to stay within 5 image limit
    const maxFilesToUpload = Math.min(files.length, 5 - currentImages.length);
    if (maxFilesToUpload < files.length) {
      toast.warning(`فقط ${maxFilesToUpload} تصویر از ${files.length} تصویر آپلود می‌شود (حداکثر 5 تصویر)`);
    }

    // 🔧 FIX: Set flag to prevent useEffect from interfering
    setIsUploadingImage(true);

    try {
      const formData = new FormData();
      files.slice(0, maxFilesToUpload).forEach((file) => {
        formData.append('images', file);
      });
      formData.append('category', category);

      await axios.post(`${endpoints.patients}/${id}/images`, formData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
          'Content-Type': 'multipart/form-data',
        },
      });

      // Refresh images after upload
      const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });

      const images = imagesResponse.data.images || [];
      let lateralImages = images.filter(img => 
        img.category === 'lateral' || 
        img.category === 'cephalometric' || 
        img.category === 'cephalometry'
      );

      // 🔧 FIX: Limit to 5 images (keep newest 5)
      lateralImages = lateralImages.sort((a, b) => {
        const dateA = a.createdAt ? new Date(a.createdAt).getTime() : 0;
        const dateB = b.createdAt ? new Date(b.createdAt).getTime() : 0;
        return dateB - dateA; // Newest first
      }).slice(0, 5); // Keep only 5 newest images

      // Select the newly uploaded image (index 0 = newest after sorting)
      const newImageIndex = lateralImages.length > 0 ? 0 : 0;
      
      console.log('📤 [handleImageUpload] Upload successful, updating state:', {
        newImageIndex,
        totalImages: lateralImages.length,
        newImageId: lateralImages[newImageIndex]?.id,
      });
      
      // Update patient state first
      // 🔧 FIX: Clear cephalometricLandmarks when uploading new image
      // Each analysis is specific to one image - old landmarks should not be used for new image
      setPatient(prev => {
        const updated = {
          ...prev,
          lateralImages,
          cephalometricLandmarks: null, // 🔧 FIX: Clear old landmarks for new image
        };
        console.log('📤 [handleImageUpload] Updated patient state:', {
          lateralImagesCount: updated.lateralImages.length,
          newestImageId: updated.lateralImages[0]?.id,
          clearedLandmarks: true, // 🔧 FIX: Landmarks cleared for new image
        });
        return updated;
      });
      
      // 🔧 FIX: Update selected index to 0 (newest image)
      setSelectedImageIndex(newImageIndex);
      console.log('📤 [handleImageUpload] Updated selectedImageIndex to:', newImageIndex);
      
      // 🔧 FIX: Clear selected analysis when new image is uploaded
      // This prevents loading old landmarks for the new image
      setSelectedAnalysisIndex(null);
      console.log('📤 [handleImageUpload] Cleared selectedAnalysisIndex (new image has no analysis yet)');
      
      // 🔧 FIX: Update refs to prevent duplicate reset in useEffect
      lastResetImageIndexRef.current = newImageIndex;
      lastResetImagesLengthRef.current = lateralImages.length;
      
      // Force remount of CephalometricAIAnalysis to reload with new image
      setAnalysisKey(prev => prev + 1);
      console.log('📤 [handleImageUpload] Updated analysisKey');
      
      // 🔧 FIX: Reset flag after a delay to allow state updates to complete
      setTimeout(() => {
        setIsUploadingImage(false);
        console.log('📤 [handleImageUpload] Reset isUploadingImage flag');
      }, 500);

      toast.success(`${files.length} تصویر با موفقیت آپلود شد`);
    } catch (error) {
      console.error('Error uploading images:', error);
      toast.error('خطا در آپلود تصویر');
      // Reset flag on error
      setIsUploadingImage(false);
    }
  }, [id, user?.accessToken, patient?.lateralImages]);

  // Fetch patient data
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
        
        console.log('🔍 [Fetch Patient] Raw patientData from API:', {
          hasCephalometricLandmarks: !!patientData.cephalometricLandmarks,
          cephalometricLandmarksType: typeof patientData.cephalometricLandmarks,
          cephalometricLandmarksValue: patientData.cephalometricLandmarks ? (
            typeof patientData.cephalometricLandmarks === 'string' 
              ? patientData.cephalometricLandmarks.substring(0, 200) 
              : JSON.stringify(patientData.cephalometricLandmarks).substring(0, 200)
          ) : 'null',
          allKeys: Object.keys(patientData),
        });
        
        // Parse cephalometric data from JSON strings
        let cephalometricTable = null;
        try {
          if (patientData.cephalometricTableData && typeof patientData.cephalometricTableData === 'string') {
            cephalometricTable = JSON.parse(patientData.cephalometricTableData);
          }
        } catch (parseError) {
          console.warn('Failed to parse cephalometricTableData:', parseError);
          cephalometricTable = null;
        }

        let cephalometricRawData = null;
        try {
          if (patientData.cephalometricRawData && typeof patientData.cephalometricRawData === 'string') {
            cephalometricRawData = JSON.parse(patientData.cephalometricRawData);
          }
        } catch (parseError) {
          console.warn('Failed to parse cephalometricRawData:', parseError);
          cephalometricRawData = null;
        }

        let cephalometricLandmarks = null;
        try {
          console.log('🔍 [Fetch Patient] Checking cephalometricLandmarks in patientData:', {
            hasCephalometricLandmarks: !!patientData.cephalometricLandmarks,
            type: typeof patientData.cephalometricLandmarks,
            value: patientData.cephalometricLandmarks ? (typeof patientData.cephalometricLandmarks === 'string' ? patientData.cephalometricLandmarks.substring(0, 100) : 'object') : 'null',
          });
          
          if (patientData.cephalometricLandmarks) {
            if (typeof patientData.cephalometricLandmarks === 'string') {
              console.log('📥 Parsing cephalometricLandmarks from string...');
              cephalometricLandmarks = JSON.parse(patientData.cephalometricLandmarks);
              console.log('✅ Parsed cephalometricLandmarks:', {
                landmarksCount: cephalometricLandmarks ? Object.keys(cephalometricLandmarks).length : 0,
                landmarkNames: cephalometricLandmarks ? Object.keys(cephalometricLandmarks).slice(0, 10) : [],
              });
            } else if (typeof patientData.cephalometricLandmarks === 'object') {
              // If it's already an object, use it directly
              console.log('📥 Using cephalometricLandmarks as object...');
              const { cephalometricLandmarks: patientCephalometricLandmarks } = patientData;
              cephalometricLandmarks = patientCephalometricLandmarks;
              console.log('✅ Using cephalometricLandmarks object:', {
                landmarksCount: cephalometricLandmarks ? Object.keys(cephalometricLandmarks).length : 0,
                landmarkNames: cephalometricLandmarks ? Object.keys(cephalometricLandmarks).slice(0, 10) : [],
              });
            }
          } else {
            console.warn('⚠️ patientData.cephalometricLandmarks is null or undefined');
          }
        } catch (parseError) {
          console.error('❌ Failed to parse cephalometricLandmarks:', parseError);
          console.error('❌ Parse error details:', {
            message: parseError.message,
            stack: parseError.stack,
            rawValue: patientData.cephalometricLandmarks ? (typeof patientData.cephalometricLandmarks === 'string' ? patientData.cephalometricLandmarks.substring(0, 200) : 'object') : 'null',
          });
          cephalometricLandmarks = null;
        }
        
        console.log('📊 [Fetch Patient] Final cephalometricLandmarks:', {
          isNull: cephalometricLandmarks === null,
          isUndefined: cephalometricLandmarks === undefined,
          landmarksCount: cephalometricLandmarks ? Object.keys(cephalometricLandmarks).length : 0,
        });

        const patientState = {
          id: patientData.id,
          name: `${patientData.firstName} ${patientData.lastName}`,
          age: patientData.age,
          phone: patientData.phone,
          gender: patientData.gender || '',
          cephalometricTable,
          cephalometricRawData,
          cephalometricLandmarks,
        };
        
        console.log('📊 [Fetch Patient] Setting patient state:', {
          hasCephalometricLandmarks: !!patientState.cephalometricLandmarks,
          landmarksCount: patientState.cephalometricLandmarks ? Object.keys(patientState.cephalometricLandmarks).length : 0,
        });
        
        setPatient(patientState);

        // Fetch patient images
        try {
          const imagesResponse = await axios.get(`${endpoints.patients}/${id}/images`, {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          });

          const images = imagesResponse.data.images || [];
          let lateralImages = images.filter(img => 
            img.category === 'lateral' || 
            img.category === 'cephalometric' || 
            img.category === 'cephalometry'
          );

          // 🔧 FIX: Sort images by createdAt (newest first) and limit to 5
          lateralImages = lateralImages.sort((a, b) => {
            const dateA = a.createdAt ? new Date(a.createdAt).getTime() : 0;
            const dateB = b.createdAt ? new Date(b.createdAt).getTime() : 0;
            return dateB - dateA; // Newest first
          }).slice(0, 5); // Keep only 5 newest images

          console.log('📊 [Fetch Images] Updating patient with lateralImages:', {
            lateralImagesCount: lateralImages.length,
            hasPreviousLandmarks: !!patientState.cephalometricLandmarks,
            previousLandmarksCount: patientState.cephalometricLandmarks ? Object.keys(patientState.cephalometricLandmarks).length : 0,
          });

          setPatient(prev => {
            const updated = {
              ...prev,
              lateralImages,
            };
            
            console.log('📊 [Fetch Images] Updated patient state:', {
              hasCephalometricLandmarks: !!updated.cephalometricLandmarks,
              landmarksCount: updated.cephalometricLandmarks ? Object.keys(updated.cephalometricLandmarks).length : 0,
            });
            
            return updated;
          });
        } catch (imagesError) {
          console.warn('Could not fetch patient images:', imagesError);
        }

        setLoading(false);
      } catch (error) {
        console.error('Error fetching patient:', error);
        setLoading(false);
      }
    };

    if (user && id) {
      fetchPatient();
    }
  }, [id, user]);

  // Helper function to calculate measurements from landmarks
  const calculateMeasurementsFromLandmarks = useCallback((landmarks, pixelToMmConversion = 0.11) => {
    // Helper functions are defined inside to avoid dependency issues
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
    const measures = {};

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
        measures.SNA = Math.round(calculateAngle(sLandmarkSNA, nLandmarkSNA, aLandmark) * 10) / 10;
      }

      // SNB angle
      const sLandmarkSNB = getLandmark(['S', 's']);
      const nLandmarkSNB = getLandmark(['N', 'n']);
      const bLandmark = getLandmark(['B', 'b']);

      if (sLandmarkSNB && nLandmarkSNB && bLandmark) {
        measures.SNB = Math.round(calculateAngle(sLandmarkSNB, nLandmarkSNB, bLandmark) * 10) / 10;
      }

      // ANB angle
      if (measures.SNA !== undefined && measures.SNB !== undefined) {
        measures.ANB = Math.round((measures.SNA - measures.SNB) * 10) / 10;
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

      // IMPA - 🔧 FIX: باید از L1A-L1 استفاده شود (همانند تصویر)
      const goLandmark = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']) ||
                        findLandmarkByPartial(['go', 'gonion']);
      const meLandmark2 = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) ||
                         findLandmarkByPartial(['me', 'menton']);
      const l1aLandmark = getLandmark(['L1A', 'l1a', 'LIA', 'lia', 'Lia', 'lower_incisor_apex', 'Lower_incisor_apex', 'lower incisor apex']) ||
                         findLandmarkByPartial(['l1a', 'lia', 'lower', 'incisor', 'apex']);
      const l1Landmark2 = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor', 'LIT', 'lit', 'Lit', 'lower_incisor_tip', 'Lower_incisor_tip', 'lower incisor tip']) ||
                         findLandmarkByPartial(['l1', 'lit', 'lower', 'incisor', 'li', 'tip']);

      if (l1aLandmark && l1Landmark2 && goLandmark && meLandmark2) {
        // استفاده از L1A-L1 برای محاسبه دقیق‌تر (همانند تصویر)
        // 🔧 FIX: مقدار باید از 180 کم شود
        const calculatedAngle = calculateAngleBetweenLines(
          goLandmark, meLandmark2,  // خط اول: صفحه مندیبولار (Go-Me)
          l1aLandmark, l1Landmark2   // خط دوم: incisor (L1A-L1)
        );
        measures.IMPA = Math.round(Math.max(0, Math.min(180, 180 - calculatedAngle)) * 10) / 10;
      } else if (l1Landmark2 && goLandmark && meLandmark2) {
        // Fallback: اگر L1A موجود نبود، از L1-Me استفاده می‌کنیم
        // 🔧 FIX: مقدار باید از 180 کم شود
        const calculatedAngle = calculateAngleBetweenLines(
          goLandmark, meLandmark2,  // خط اول: صفحه مندیبولار (Go-Me)
          l1Landmark2, meLandmark2  // خط دوم: incisor (L1-Me)
        );
        measures.IMPA = Math.round(Math.max(0, Math.min(180, 180 - calculatedAngle)) * 10) / 10;
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
        measures['GoGn-SN'] = Math.round(angleDiff * 10) / 10;
        measures.GoGnSN = Math.round(angleDiff * 10) / 10;
      }

      // U1-SN - زاویه بین خط U1 (UIA-UIT) و خط SN
      const sLandmark = getLandmark(['S', 's', 'sella', 'Sella']) ||
                       findLandmarkByPartial(['s', 'sella']);
      const nLandmark = getLandmark(['N', 'n', 'nasion', 'Nasion']) ||
                       findLandmarkByPartial(['n', 'nasion']);
      const uiaLandmark = getLandmark(['UIA', 'uia', 'Uia', 'upper_incisor_apex', 'Upper_incisor_apex', 'upper incisor apex']) ||
                         findLandmarkByPartial(['uia', 'upper', 'incisor', 'apex']);
      const uitLandmark = getLandmark(['UIT', 'uit', 'Uit', 'upper_incisor_tip', 'Upper_incisor_tip', 'upper incisor tip']) ||
                         findLandmarkByPartial(['uit', 'upper', 'incisor', 'tip']);

      if (uiaLandmark && uitLandmark && sLandmark && nLandmark) {
        const u1Angle = calculateLineAngle(uiaLandmark, uitLandmark);
        const snAngle = calculateLineAngle(sLandmark, nLandmark);
        const angleDiff = Math.abs(u1Angle - snAngle);
        measures['U1-SN'] = Math.round(Math.max(0, Math.min(180, angleDiff)) * 10) / 10;
      }

      // L1-MP - زاویه بین خط L1 (LIA-LIT) و صفحه مندیبولار (Go-Me)
      const goLandmark2 = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']) ||
                         findLandmarkByPartial(['go', 'gonion']);
      const meLandmark3 = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']) ||
                         findLandmarkByPartial(['me', 'menton']);
      const liaLandmark3 = getLandmark(['LIA', 'lia', 'Lia', 'lower_incisor_apex', 'Lower_incisor_apex', 'lower incisor apex']) ||
                          findLandmarkByPartial(['lia', 'lower', 'incisor', 'apex']);
      const litLandmark3 = getLandmark(['LIT', 'lit', 'Lit', 'lower_incisor_tip', 'Lower_incisor_tip', 'lower incisor tip']) ||
                          findLandmarkByPartial(['lit', 'lower', 'incisor', 'tip']);

      if (liaLandmark3 && litLandmark3 && goLandmark2 && meLandmark3) {
        measures['L1-MP'] = calculateAngleBetweenLines(meLandmark3, goLandmark2, liaLandmark3, litLandmark3);
        measures['L1-MP'] = Math.round(Math.max(0, Math.min(180, measures['L1-MP'])) * 10) / 10;
      }

      // Interincisal Angle - زاویه بین خط U1-U1A و خط L1-L1A
      const u1LandmarkInter = landmarks.U1 || landmarks.u1 || landmarks.upper_incisor || landmarks['Upper Incisor'];
      const u1aLandmarkInter = landmarks.U1A || landmarks.u1a || landmarks.U1a || landmarks.upper_incisor_apex || landmarks.Upper_incisor_apex;
      const l1LandmarkInter = landmarks.L1 || landmarks.l1 || landmarks.lower_incisor || landmarks['Lower Incisor'];
      const l1aLandmarkInter = landmarks.L1A || landmarks.l1a || landmarks.L1a || landmarks.lower_incisor_apex || landmarks.Lower_incisor_apex;

      if (u1LandmarkInter && u1aLandmarkInter && l1LandmarkInter && l1aLandmarkInter) {
        // محاسبه زاویه بین دو خط: خط U1-U1A و خط L1-L1A
        const interincisalAngle = calculateAngleBetweenLines(
          u1LandmarkInter, u1aLandmarkInter,  // خط اول: U1-U1A
          l1LandmarkInter, l1aLandmarkInter   // خط دوم: L1-L1A
        );
        measures.InterincisalAngle = Math.round(interincisalAngle * 10) / 10;
      }

      // Overbite - فاصله عمودی بین U1 و L1 (تفاوت y)
      const u1LandmarkOverbite = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper_incisor', 'upper incisor']) ||
                                  findLandmarkByPartial(['u1', 'upper', 'incisor', 'ui']);
      const l1LandmarkOverbite = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                                  findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);

      if (u1LandmarkOverbite && l1LandmarkOverbite) {
        // Overbite = فاصله عمودی (تفاوت y) - U1 بالاتر است (y کوچکتر) پس باید l1.y - u1.y باشد
        // اما در تصاویر cephalometric، y از بالا به پایین افزایش می‌یابد
        // پس اگر U1 بالاتر باشد، y آن کوچکتر است
        const verticalDistance = Math.abs(l1LandmarkOverbite.y - u1LandmarkOverbite.y);
        
        // محاسبه conversion factor از p1/p2 اگر موجود باشند، در غیر این صورت از مقدار پیش‌فرض
        let conversionFactor = pixelToMmConversion;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          // فاصله p1-p2 معمولاً 10mm است (1cm)
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        
        // تبدیل به میلی‌متر با استفاده از conversion factor
        // عدد بدست آمده باید در -1 ضرب شود
        measures.Overbite = Math.round((verticalDistance * conversionFactor) * -1 * 10) / 10;
      }

      // Overjet - فاصله افقی بین U1 و L1 (تفاوت x)
      const u1LandmarkOverjet = getLandmark(['U1', 'u1', 'upper_incisor', 'Upper_incisor', 'upper incisor']) ||
                                findLandmarkByPartial(['u1', 'upper', 'incisor', 'ui']);
      const l1LandmarkOverjet = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                                findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);

      if (u1LandmarkOverjet && l1LandmarkOverjet) {
        // Overjet = فاصله افقی (تفاوت x)
        // اگر U1.x > L1.x (U1 جلوتر است) ⬅️ مقدار مثبت
        // اگر L1.x > U1.x (L1 جلوتر است) ⬅️ مقدار منفی
        const horizontalDistance = u1LandmarkOverjet.x - l1LandmarkOverjet.x;
        
        // محاسبه conversion factor از p1/p2 اگر موجود باشند، در غیر این صورت از مقدار پیش‌فرض
        let conversionFactor = pixelToMmConversion;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          // فاصله p1-p2 معمولاً 10mm است (1cm)
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        
        // تبدیل به میلی‌متر با استفاده از conversion factor (بدون Math.abs برای حفظ علامت)
        measures.Overjet = Math.round((horizontalDistance * conversionFactor) * 10) / 10;
      }

      // SN-GoGn
      if (landmarks.S && landmarks.N && landmarks.Go && landmarks.Gn && !measures['GoGn-SN']) {
        const snAngle = calculateLineAngle(landmarks.S, landmarks.N);
        const gognAngle = calculateLineAngle(landmarks.Go, landmarks.Gn);
        measures['SN-GoGn'] = Math.round(Math.abs(snAngle - gognAngle) * 10) / 10;
      }

      // Facial Axis - زاویه بین Gn-Pt-Ba (زاویه در نقطه Pt)
      const baLandmark = landmarks.Ba || landmarks.ba || landmarks.BA;
      const ptLandmark = landmarks.Pt || landmarks.pt || landmarks.PT;
      const gnLandmark = landmarks.Gn || landmarks.gn || landmarks.GN;
      
      if (baLandmark && ptLandmark && gnLandmark) {
        // محاسبه زاویه در نقطه Pt بین خطوط Gn-Pt و Ba-Pt
        const angle1 = Math.atan2(gnLandmark.y - ptLandmark.y, gnLandmark.x - ptLandmark.x);
        const angle2 = Math.atan2(baLandmark.y - ptLandmark.y, baLandmark.x - ptLandmark.x);
        let angle = (angle2 - angle1) * (180 / Math.PI);
        if (angle < 0) angle += 360;
        const facialAxis = angle > 180 ? 360 - angle : angle;
        measures.FacialAxis = Math.round(facialAxis * 10) / 10;
      }

      // Mandibular Plane Angle
      if (landmarks.Go && landmarks.Me) {
        const mpAngle = calculateLineAngle(landmarks.Go, landmarks.Me);
        measures.MandibularPlane = Math.round(mpAngle * 10) / 10;
      }

      // Helper function to calculate distance in millimeters
      const calculateDistanceMm = (p1, p2) => {
        if (!p1 || !p2) return null;
        const pixelDistance = Math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2);
        
        // محاسبه conversion factor از p1/p2 اگر موجود باشند
        let conversionFactor = pixelToMmConversion;
        const p1Landmark = getLandmark(['p1', 'P1']);
        const p2Landmark = getLandmark(['p2', 'P2']);
        if (p1Landmark && p2Landmark) {
          const dx = Math.abs(p2Landmark.x - p1Landmark.x);
          const dy = Math.abs(p2Landmark.y - p1Landmark.y);
          const distancePixels = Math.sqrt(dx * dx + dy * dy);
          // فاصله p1-p2 معمولاً 10mm است (1cm)
          if (distancePixels > 0) {
            conversionFactor = 10.0 / distancePixels;
          }
        }
        
        return pixelDistance * conversionFactor;
      };

      // Upper Face Height - ارتفاع صورت بالا (N-ANS) در میلی‌متر
      const nLandmarkUFH = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const ansLandmarkUFH = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      if (nLandmarkUFH && ansLandmarkUFH) {
        const nAnsDist = calculateDistanceMm(nLandmarkUFH, ansLandmarkUFH);
        if (nAnsDist !== null) {
          measures['Upper Face Height'] = Math.round(nAnsDist * 10) / 10;
        }
      }

      // Lower Face Height - ارتفاع صورت پایین (ANS-Me) در میلی‌متر
      const ansLandmarkLFH = getLandmark(['ANS', 'ans', 'Anterior Nasal Spine']);
      const meLandmarkLFH = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      if (ansLandmarkLFH && meLandmarkLFH) {
        const ansMeDist = calculateDistanceMm(ansLandmarkLFH, meLandmarkLFH);
        if (ansMeDist !== null) {
          measures['Lower Face Height'] = Math.round(ansMeDist * 10) / 10;
        }
      }

      // Co-A - طول فک بالا (فاصله Co-A) در میلی‌متر - نیازمند p1/p2
      const p1LandmarkCoA = getLandmark(['p1', 'P1']);
      const p2LandmarkCoA = getLandmark(['p2', 'P2']);
      if (p1LandmarkCoA && p2LandmarkCoA) {
        const coLandmark = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                          getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
        const aLandmarkCoA = getLandmark(['A', 'a', 'Point A']);
        if (coLandmark && aLandmarkCoA) {
          const coADist = calculateDistanceMm(coLandmark, aLandmarkCoA);
          if (coADist !== null) {
            measures['Co-A'] = Math.round(coADist * 10) / 10;
          }
        }
      }

      // Co-Gn - طول فک پایین (فاصله Co-Gn) در میلی‌متر - نیازمند p1/p2
      if (p1LandmarkCoA && p2LandmarkCoA) {
        const coLandmark = getLandmark(['Co', 'co', 'CO', 'condyle', 'Condyle']) || 
                          getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
        const gnLandmarkCoGn = getLandmark(['Gn', 'gn', 'GN', 'gnathion', 'Gnathion']);
        if (coLandmark && gnLandmarkCoGn) {
          const coGnDist = calculateDistanceMm(coLandmark, gnLandmarkCoGn);
          if (coGnDist !== null) {
            measures['Co-Gn'] = Math.round(coGnDist * 10) / 10;
          }
        }
      }

      // S-Go - ابعاد عمودی چهره (فاصله S-Go) در میلی‌متر
      const sLandmarkSGo = getLandmark(['S', 's', 'sella', 'Sella']);
      const goLandmarkSGo = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      if (sLandmarkSGo && goLandmarkSGo) {
        const sGoDist = calculateDistanceMm(sLandmarkSGo, goLandmarkSGo);
        if (sGoDist !== null) {
          measures['S-Go'] = Math.round(sGoDist * 10) / 10;
        }
      }

      // PFH (Posterior Facial Height) - فاصله S-Go در میلی‌متر
      if (sLandmarkSGo && goLandmarkSGo) {
        const pfhDist = calculateDistanceMm(sLandmarkSGo, goLandmarkSGo);
        if (pfhDist !== null) {
          measures.PFH = Math.round(pfhDist * 10) / 10;
          measures['Posterior Facial Height (PFH)'] = Math.round(pfhDist * 10) / 10;
          measures['Posterior Facial Height (S-Go)'] = Math.round(pfhDist * 10) / 10;
        }
      }

      // AFH (Anterior Facial Height) - فاصله N-Me در میلی‌متر
      const nLandmarkAFH = getLandmark(['N', 'n', 'Nasion', 'nasion']);
      const meLandmarkAFH = getLandmark(['Me', 'me', 'ME', 'menton', 'Menton']);
      if (nLandmarkAFH && meLandmarkAFH) {
        const afhDist = calculateDistanceMm(nLandmarkAFH, meLandmarkAFH);
        if (afhDist !== null) {
          measures.AFH = Math.round(afhDist * 10) / 10;
          measures['Anterior Facial Height (AFH)'] = Math.round(afhDist * 10) / 10;
          measures['Anterior Facial Height (N-Me)'] = Math.round(afhDist * 10) / 10;
        }
      }

      // Ramus Height (Ar-Go) - ارتفاع راموس در میلی‌متر
      const arLandmarkRamus = getLandmark(['Ar', 'ar', 'AR', 'articulare', 'Articulare']);
      const goLandmarkRamus = getLandmark(['Go', 'go', 'GO', 'gonion', 'Gonion']);
      if (arLandmarkRamus && goLandmarkRamus) {
        const ramusHeight = calculateDistanceMm(arLandmarkRamus, goLandmarkRamus);
        if (ramusHeight !== null) {
          measures['Ramus Height (Ar-Go)'] = Math.round(ramusHeight * 10) / 10;
          measures['Ramus Height'] = Math.round(ramusHeight * 10) / 10;
        }
      }

      // PFH/AFH Ratio - نسبت ارتفاع خلفی به قدامی
      if (measures.PFH && measures.AFH && measures.AFH > 0) {
        measures['PFH/AFH Ratio'] = Math.round((measures.PFH / measures.AFH) * 100 * 10) / 10;
      }

      // Facial Height Ratio - نسبت ارتفاع صورت (ANS-Me/N-Me × 100)
      if (measures['Lower Face Height'] && measures.AFH && measures.AFH > 0) {
        measures['Facial Height Ratio'] = Math.round((measures['Lower Face Height'] / measures.AFH) * 100 * 10) / 10;
      }

      // Upper Face Height / Lower Face Height Ratio
      if (measures['Upper Face Height'] && measures['Lower Face Height'] && measures['Lower Face Height'] > 0) {
        measures.UpperLowerFaceRatio = Math.round((measures['Upper Face Height'] / measures['Lower Face Height']) * 10) / 10;
      }

      // Wits Analysis - AO-BO
      // محاسبه فاصله افقی بین AO و BO روی Occlusal Plane
      const l1LandmarkWits = getLandmark(['L1', 'l1', 'lower_incisor', 'Lower_incisor', 'lower incisor']) ||
                            findLandmarkByPartial(['l1', 'lower', 'incisor', 'li']);
      const lmtLandmarkWits = getLandmark(['LMT', 'lmt', 'lower_molar', 'Lower Molar', 'L6', 'l6']) ||
                             findLandmarkByPartial(['lmt', 'lower', 'molar', 'l6']);
      const umtLandmarkWits = getLandmark(['UMT', 'umt', 'upper_molar', 'Upper Molar', 'U6', 'u6']) ||
                             findLandmarkByPartial(['umt', 'upper', 'molar', 'u6']);
      const aLandmarkWits = getLandmark(['A', 'a', 'Point A']);
      const bLandmarkWits = getLandmark(['B', 'b', 'Point B']);

      if (l1LandmarkWits && lmtLandmarkWits && aLandmarkWits && bLandmarkWits) {
        // نقطه پایانی Occlusal Plane: نقطه میانی بین LMT و UMT (یا فقط LMT)
        const occlusalEndPoint = umtLandmarkWits ? {
          x: (lmtLandmarkWits.x + umtLandmarkWits.x) / 2,
          y: (lmtLandmarkWits.y + umtLandmarkWits.y) / 2
        } : lmtLandmarkWits;

        // بردار Occlusal Plane
        const occlusalDx = occlusalEndPoint.x - l1LandmarkWits.x;
        const occlusalDy = occlusalEndPoint.y - l1LandmarkWits.y;
        const occlusalLength = Math.sqrt(occlusalDx * occlusalDx + occlusalDy * occlusalDy);

        if (occlusalLength > 0) {
          const occlusalDirX = occlusalDx / occlusalLength;
          const occlusalDirY = occlusalDy / occlusalLength;

          // محاسبه نقطه تقاطع عمود از A با Occlusal Plane (AO)
          const getPerpendicularIntersection = (point, lineStart, lineDirX, lineDirY) => {
            const dx = point.x - lineStart.x;
            const dy = point.y - lineStart.y;
            const t = (dx * lineDirX + dy * lineDirY) / (lineDirX * lineDirX + lineDirY * lineDirY);
            return {
              x: lineStart.x + t * lineDirX,
              y: lineStart.y + t * lineDirY
            };
          };

          const aoPoint = getPerpendicularIntersection(aLandmarkWits, l1LandmarkWits, occlusalDirX, occlusalDirY);
          const boPoint = getPerpendicularIntersection(bLandmarkWits, l1LandmarkWits, occlusalDirX, occlusalDirY);

          // محاسبه فاصله افقی بین AO و BO روی Occlusal Plane
          // Projection روی بردار Occlusal Plane
          // علامت: اگر AO جلوتر از BO باشد (در راستای Occlusal Plane)، مقدار مثبت (کلاس II)
          // اگر BO جلوتر از AO باشد، مقدار منفی (کلاس III)
          const aoBoDx = boPoint.x - aoPoint.x;
          const aoBoDy = boPoint.y - aoPoint.y;
          // بدون Math.abs برای حفظ علامت
          const aoBoDistancePixels = aoBoDx * occlusalDirX + aoBoDy * occlusalDirY;

          // تبدیل به میلی‌متر
          const aoBoDistanceMmAbs = Math.abs(aoBoDistancePixels) * pixelToMmConversion;
          
          // اگر p1/p2 موجود باشند، از آن‌ها برای تبدیل استفاده می‌کنیم
          const p1Landmark = getLandmark(['p1', 'P1']);
          const p2Landmark = getLandmark(['p2', 'P2']);
          let conversionFactor = pixelToMmConversion;
          if (p1Landmark && p2Landmark) {
            const dx = Math.abs(p2Landmark.x - p1Landmark.x);
            const dy = Math.abs(p2Landmark.y - p1Landmark.y);
            const distancePixels = Math.sqrt(dx * dx + dy * dy);
            if (distancePixels > 0) {
              conversionFactor = 10.0 / distancePixels;
            }
          }
          
          // محاسبه مقدار نهایی با حفظ علامت
          const aoBoDistanceMm = Math.abs(aoBoDistancePixels) * conversionFactor;
          const signedAoBoDistanceMm = aoBoDistanceMm * (aoBoDistancePixels >= 0 ? 1 : -1);
          measures['AO-BO'] = Math.round(signedAoBoDistanceMm * 10) / 10;
        }
      }

      // Skeletal Convexity (N-A-Pog) - زاویه بین خط N-A و خط A-Pog (در نقطه A)
      // در visualizer، از calculateAngleBetweenLines استفاده می‌شود که زاویه کوچک‌تر را برمی‌گرداند
      // مقدار را مستقیماً ذخیره می‌کنیم و تبدیل (180 - angle) را در useEffect انجام می‌دهیم
      const nLandmarkNAPo = getLandmark(['N', 'n', 'Nasion', 'nasion']) || findLandmarkByPartial(['n', 'nasion']);
      const aLandmarkNAPo = getLandmark(['A', 'a', 'Point A']) || findLandmarkByPartial(['a', 'point']);
      const pogLandmarkNAPo = getLandmark(['Pog', 'pog', 'POG', 'pogonion', 'Pogonion']) || findLandmarkByPartial(['pog', 'pogonion']);
      
      if (nLandmarkNAPo && aLandmarkNAPo && pogLandmarkNAPo) {
        // زاویه بین خط N-A و خط A-Pog (در نقطه A)
        // استفاده از calculateAngleBetweenLines (مشابه visualizer)
        const calculatedAngle = calculateAngleBetweenLines(
          nLandmarkNAPo, aLandmarkNAPo,  // خط اول: N-A
          pogLandmarkNAPo, aLandmarkNAPo  // خط دوم: A-Pog
        );
        if (calculatedAngle !== null && !isNaN(calculatedAngle) && calculatedAngle > 0) {
          // مقدار را مستقیماً ذخیره می‌کنیم (بدون کم کردن از 180)
          // تبدیل (180 - angle) در useEffect انجام می‌شود تا از double conversion جلوگیری شود
          const rawAngle = Math.round(calculatedAngle * 10) / 10;
          measures['Skeletal Convexity'] = rawAngle;
          measures['N-A-Pog'] = rawAngle; // Backward compatibility
          console.log('🔍 [calculateMeasurementsFromLandmarks] Skeletal Convexity calculated:', {
            rawAngle,
            calculatedAngle,
            nLandmark: nLandmarkNAPo,
            aLandmark: aLandmarkNAPo,
            pogLandmark: pogLandmarkNAPo
          });
        }
      }

      console.log('✅ Calculated measurements from landmarks:', measures);
    } catch (err) {
      console.error('Error calculating measurements from landmarks:', err);
    }

    return measures;
  }, []); // Empty dependency array since helper functions are defined inside

  // Handle save cephalometric data (defined before onLandmarksDetected to avoid initialization error)
  const handleSaveCephalometric = useCallback(async (options = {}, dataToSaveOverride = null) => {
    if (!patient && !dataToSaveOverride) return;
    
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
          cephalometricTableData: patientData.cephalometricTable ? JSON.stringify(patientData.cephalometricTable) : null,
          cephalometricRawData: patientData.cephalometricRawData ? JSON.stringify(patientData.cephalometricRawData) : null,
          cephalometricLandmarks: patientData.cephalometricLandmarks ? JSON.stringify(patientData.cephalometricLandmarks) : null,
        };
        
        console.log('💾 Saving to database:', {
          hasTableData: !!saveData.cephalometricTableData,
          hasRawData: !!saveData.cephalometricRawData,
          hasLandmarks: !!saveData.cephalometricLandmarks,
          landmarksSize: saveData.cephalometricLandmarks ? saveData.cephalometricLandmarks.length : 0,
          landmarksSample: saveData.cephalometricLandmarks ? saveData.cephalometricLandmarks.substring(0, 100) : 'none',
        });
        
        await axios.put(`${endpoints.patients}/${id}`, saveData, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        console.log('✅ Cephalometric data auto-saved successfully to database');
      } catch (error) {
        console.error('Auto-save error:', error);
      }
      return;
    }
    
    // Normal save with toast
    setSaving(true);
    try {
      const dataToSave = {
        cephalometricTableData: patientData.cephalometricTable ? JSON.stringify(patientData.cephalometricTable) : null,
        cephalometricRawData: patientData.cephalometricRawData ? JSON.stringify(patientData.cephalometricRawData) : null,
        cephalometricLandmarks: patientData.cephalometricLandmarks ? JSON.stringify(patientData.cephalometricLandmarks) : null,
      };
      
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
  }, [id, user, patient]);

  // Handle landmarks detected callback
  const onLandmarksDetected = useCallback(async (data) => {
    const { landmarks, measurements, metadata } = data;

    console.log('📥 onLandmarksDetected called with:', {
      landmarksCount: landmarks ? Object.keys(landmarks).length : 0,
      hasMeasurements: !!measurements,
      displayedOnly: metadata?.displayedOnly,
    });

    // اگر فقط برای نمایش است، ذخیره نکن
    if (metadata?.displayedOnly) {
      console.log('📊 فقط نمایش نتایج (بدون ذخیره در دیتابیس)');

      // همچنان محاسبات را انجام بده و state را آپدیت کن، اما ذخیره نکن
      let calculatedMeasurements = measurements || calculateMeasurementsFromLandmarks(landmarks);

      // Round all measurements to 1 decimal place to ensure consistency
      const roundedMeasurements = {};
      Object.keys(calculatedMeasurements).forEach(key => {
        const value = calculatedMeasurements[key];
        if (typeof value === 'number' && !isNaN(value)) {
          roundedMeasurements[key] = Math.round(value * 10) / 10;
        } else {
          roundedMeasurements[key] = value;
        }
      });
      calculatedMeasurements = roundedMeasurements;

      // Get selected template
      const template = cephalometricTemplates[selectedAnalysisType] || cephalometricTemplates.steiner;
      const filteredTable = {};

      // First, add all parameters from the selected template
      Object.keys(template).forEach(param => {
        let measuredValue = calculatedMeasurements[param];
        
        // برای Skeletal Convexity (N-A-Pog)، اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم
        // (مقدار از calculateAngleBetweenLines می‌آید که زاویه کوچک‌تر را برمی‌گرداند)
        if ((param === 'Skeletal Convexity' || param === 'N-A-Pog') && measuredValue !== undefined && measuredValue !== null && !isNaN(parseFloat(measuredValue))) {
          const numValue = parseFloat(measuredValue);
          // اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم (مقدار نرمال 165° تا 175° است)
          if (numValue < 90) {
            const convertedValue = Math.round((180 - numValue) * 10) / 10;
            console.log('🔍 [onLandmarksDetected] Skeletal Convexity conversion:', {
              original: numValue,
              converted: convertedValue,
              fromMeasurements: measurements && measurements[param] !== undefined
            });
            measuredValue = convertedValue;
          }
        }
        
        // برای Interincisal Angle در Ricketts، مقدار باید از 180 کم شود
        // (مقدار محاسبه شده در canvas درست است، اما برای Ricketts باید از 180 کم شود)
        if (param === 'Interincisal Angle' && selectedAnalysisType === 'ricketts' && measuredValue !== undefined && measuredValue !== null && !isNaN(parseFloat(measuredValue))) {
          const numValue = parseFloat(measuredValue);
          const convertedValue = Math.round((180 - numValue) * 10) / 10;
          console.log('🔍 [onLandmarksDetected] Interincisal Angle conversion for Ricketts:', {
            original: numValue,
            converted: convertedValue,
            analysisType: selectedAnalysisType
          });
          measuredValue = convertedValue;
        }
        
        filteredTable[param] = {
          ...template[param],
          measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '')
            ? String(measuredValue)
            : '',
        };
      });

      // Then, add any additional measurements that might not be in the current template
      Object.keys(calculatedMeasurements).forEach(param => {
        if (!filteredTable[param]) {
          // Check if this parameter exists in any template
          let paramTemplate = null;
          Object.values(cephalometricTemplates).forEach(t => {
            if (t[param]) {
              paramTemplate = t[param];
            }
          });

          if (paramTemplate) {
            const measuredValue = calculatedMeasurements[param];
            filteredTable[param] = {
              ...paramTemplate,
              measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '')
                ? String(measuredValue)
                : '',
            };
          }
        }
      });

      // پردازش rawData برای تبدیل مقادیر خاص (مثل Interincisal Angle در Ricketts)
      const processedRawData = { ...calculatedMeasurements };
      
      // برای Interincisal Angle در Ricketts، مقدار را در rawData تبدیل می‌کنیم (از 180 کم می‌کنیم)
      if (selectedAnalysisType === 'ricketts') {
        const interincisalAngleValue = processedRawData['Interincisal Angle'];
        if (interincisalAngleValue !== undefined && interincisalAngleValue !== null && !isNaN(parseFloat(interincisalAngleValue))) {
          const numValue = parseFloat(interincisalAngleValue);
          const convertedValue = Math.round((180 - numValue) * 10) / 10;
          processedRawData['Interincisal Angle'] = convertedValue;
          console.log('🔍 [onLandmarksDetected] Interincisal Angle conversion for Ricketts (rawData):', {
            original: numValue,
            converted: convertedValue,
            analysisType: selectedAnalysisType
          });
        }
      }

      // 🔧 FIX: Check if this is from history - if so, don't update patient state if it's already set
      // This prevents infinite loops when loading from history
      const isFromHistory = metadata?.source === 'history';
      
      if (isFromHistory) {
        // If loading from history, only update if landmarks actually changed
        const currentLandmarks = patient?.cephalometricLandmarks;
        const landmarksChanged = !currentLandmarks || JSON.stringify(currentLandmarks) !== JSON.stringify(landmarks);
        
        if (landmarksChanged) {
          // Only update patient state if landmarks actually changed
          const updatedPatient = {
            ...patient,
            cephalometricTable: filteredTable,
            cephalometricRawData: {
              ...(patient?.cephalometricRawData || {}),
              ...processedRawData,
            },
            cephalometricLandmarks: landmarks,
          };
          setPatient(updatedPatient);
        } else {
          console.log('⏭️ [onLandmarksDetected] Skipping patient state update - landmarks unchanged (from history)');
        }
      } else {
        // Update patient state only for display (don't save to database)
        const updatedPatient = {
          ...patient,
          cephalometricTable: filteredTable,
          cephalometricRawData: {
            ...(patient?.cephalometricRawData || {}),
            ...processedRawData,
          },
          cephalometricLandmarks: landmarks,
        };
        setPatient(updatedPatient);
      }
      
      // 🔧 FIX: Only hide image if user is not actively viewing it
      // If user clicked "نمایش تصویر" or is on image tab, keep the image visible
      // But if user clicked "جدول آنالیز" button, hide the image
      if (metadata?.fromTableButton) {
        console.log('📊 [displayedOnly] User clicked "جدول آنالیز" - hiding image and showing table');
        isViewingImageRef.current = false; // Reset flag
        setShowCephalometricImage(false);
        setActiveTab('table');
        userInteractedRef.current = true;
        if (id) {
          localStorage.setItem(`cephalometric_viewing_tables_${id}`, 'true');
        }
      } else if (isViewingImageRef.current || activeTabRef.current === 'image') {
        console.log('📊 [displayedOnly] User is viewing image - keeping it visible');
        // Ensure image tab is active
        setActiveTab('image');
        setShowCephalometricImage(true);
        isViewingImageRef.current = true; // Ensure flag is set
      } else {
        console.log('📊 [displayedOnly] Hiding image and showing table');
        
        // Hide the image and show the results table
        setShowCephalometricImage(false);
        setActiveTab('table');
        
        // Mark that user has interacted
        userInteractedRef.current = true;
        
        // Save state to localStorage so it persists
        if (id) {
          localStorage.setItem(`cephalometric_viewing_tables_${id}`, 'true');
        }
      }

      console.log('✅ نتایج آنالیز برای نمایش آماده شد (بدون ذخیره)');
      return; // خروج بدون ذخیره در دیتابیس
    }

    // Calculate measurements from landmarks if not provided
    let calculatedMeasurements = measurements || calculateMeasurementsFromLandmarks(landmarks);

    // Round all measurements to 1 decimal place to ensure consistency
    const roundedMeasurements = {};
    Object.keys(calculatedMeasurements).forEach(key => {
      const value = calculatedMeasurements[key];
      if (typeof value === 'number' && !isNaN(value)) {
        roundedMeasurements[key] = Math.round(value * 10) / 10;
      } else {
        roundedMeasurements[key] = value;
      }
    });
    calculatedMeasurements = roundedMeasurements;
    
    // Get selected template
    const template = cephalometricTemplates[selectedAnalysisType] || cephalometricTemplates.steiner;
    const filteredTable = {};

    // First, add all parameters from the selected template
    Object.keys(template).forEach(param => {
      let measuredValue = calculatedMeasurements[param];
      
        // برای Skeletal Convexity (N-A-Pog)، اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم
        // (مقدار از calculateAngleBetweenLines می‌آید که زاویه کوچک‌تر را برمی‌گرداند)
        if ((param === 'Skeletal Convexity' || param === 'N-A-Pog') && measuredValue !== undefined && measuredValue !== null && !isNaN(parseFloat(measuredValue))) {
          const numValue = parseFloat(measuredValue);
          // اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم (مقدار نرمال 165° تا 175° است)
          if (numValue < 90) {
            measuredValue = Math.round((180 - numValue) * 10) / 10;
          }
        }
        
        // برای Interincisal Angle در Ricketts، مقدار باید از 180 کم شود
        // (مقدار محاسبه شده در canvas درست است، اما برای Ricketts باید از 180 کم شود)
        if (param === 'Interincisal Angle' && selectedAnalysisType === 'ricketts' && measuredValue !== undefined && measuredValue !== null && !isNaN(parseFloat(measuredValue))) {
          const numValue = parseFloat(measuredValue);
          measuredValue = Math.round((180 - numValue) * 10) / 10;
        }
      
      filteredTable[param] = {
        ...template[param],
        measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '')
          ? String(measuredValue)
          : '',
      };
    });

    // Then, add any additional measurements that might not be in the current template
    Object.keys(calculatedMeasurements).forEach(param => {
      if (!filteredTable[param]) {
        // Check if this parameter exists in any template
        let paramTemplate = null;
        Object.values(cephalometricTemplates).forEach(t => {
          if (t[param]) {
            paramTemplate = t[param];
          }
        });

        if (paramTemplate) {
          const measuredValue = calculatedMeasurements[param];
          filteredTable[param] = {
            ...paramTemplate,
            measured: (measuredValue !== undefined && measuredValue !== null && measuredValue !== '')
              ? String(measuredValue)
              : '',
          };
        }
      }
    });

    // برای Skeletal Convexity (N-A-Pog)، مقدار را در rawData تبدیل می‌کنیم (از 180 کم می‌کنیم)
    const processedRawData = { ...calculatedMeasurements };
    const skeletalConvexityValue = processedRawData['Skeletal Convexity'] || processedRawData['N-A-Pog'];
    if (skeletalConvexityValue !== undefined && skeletalConvexityValue !== null && !isNaN(parseFloat(skeletalConvexityValue))) {
      const numValue = parseFloat(skeletalConvexityValue);
      // اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم (مقدار نرمال 165° تا 175° است)
      if (numValue < 90) {
        const convertedValue = Math.round((180 - numValue) * 10) / 10;
        processedRawData['Skeletal Convexity'] = convertedValue;
        processedRawData['N-A-Pog'] = convertedValue; // Backward compatibility
        console.log('🔍 [handleSaveCephalometric] Skeletal Convexity conversion for rawData:', {
          original: numValue,
          converted: convertedValue
        });
      }
    }
    
    // برای Interincisal Angle در Ricketts، مقدار را در rawData تبدیل می‌کنیم (از 180 کم می‌کنیم)
    if (selectedAnalysisType === 'ricketts') {
      const interincisalAngleValue = processedRawData['Interincisal Angle'];
      if (interincisalAngleValue !== undefined && interincisalAngleValue !== null && !isNaN(parseFloat(interincisalAngleValue))) {
        const numValue = parseFloat(interincisalAngleValue);
        const convertedValue = Math.round((180 - numValue) * 10) / 10;
        processedRawData['Interincisal Angle'] = convertedValue;
        console.log('🔍 [handleSaveCephalometric] Interincisal Angle conversion for Ricketts (rawData):', {
          original: numValue,
          converted: convertedValue,
          analysisType: selectedAnalysisType
        });
      }
    }
    
    // Update patient state
    const updatedPatient = {
      ...patient,
      cephalometricTable: filteredTable,
      cephalometricRawData: {
        ...(patient?.cephalometricRawData || {}),
        ...processedRawData,
      },
      cephalometricLandmarks: landmarks,
    };

    setPatient(updatedPatient);
    
    // 🔧 FIX: Reset isUploadingImage flag when analysis completes
    // This ensures landmarks can be loaded properly after analysis
    setIsUploadingImage(false);

    // Save to database immediately (call handleSaveCephalometric directly)
    try {
      console.log('💾 Saving landmarks to database...', {
        landmarksCount: landmarks ? Object.keys(landmarks).length : 0,
        tableParams: Object.keys(filteredTable).length,
      });
      
      // Call handleSaveCephalometric using a ref or directly save
      const saveData = {
        cephalometricTableData: updatedPatient.cephalometricTable ? JSON.stringify(updatedPatient.cephalometricTable) : null,
        cephalometricRawData: updatedPatient.cephalometricRawData ? JSON.stringify(updatedPatient.cephalometricRawData) : null,
        cephalometricLandmarks: updatedPatient.cephalometricLandmarks ? JSON.stringify(updatedPatient.cephalometricLandmarks) : null,
      };
      
      console.log('💾 Saving to database:', {
        hasTableData: !!saveData.cephalometricTableData,
        hasRawData: !!saveData.cephalometricRawData,
        hasLandmarks: !!saveData.cephalometricLandmarks,
        landmarksSize: saveData.cephalometricLandmarks ? saveData.cephalometricLandmarks.length : 0,
      });
      
      await axios.put(`${endpoints.patients}/${id}`, saveData, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      
      console.log('✅ Landmarks saved to database successfully');
    } catch (error) {
      console.error('❌ Error saving landmarks to database:', error);
      toast.error('خطا در ذخیره لندمارک‌ها در دیتابیس');
    }

    // 🔧 FIX: Auto-save analysis to history after landmarks are saved
    // Pass updatedPatient directly to avoid stale closure issues
    // Only auto-save if we're not already saving and this is a new analysis
    const currentTimestamp = metadata?.timestamp || new Date().toISOString();
    const isDuplicate = lastSavedAnalysisTimestampRef.current === currentTimestamp;
    
    if (!isAutoSavingRef.current && !isDuplicate) {
      isAutoSavingRef.current = true;
      setTimeout(async () => {
        try {
          console.log('💾 Auto-saving analysis with landmarks:', {
            landmarksCount: updatedPatient.cephalometricLandmarks ? Object.keys(updatedPatient.cephalometricLandmarks).length : 0,
            timestamp: currentTimestamp,
          });
          await handleSaveAnalysis(updatedPatient);
          lastSavedAnalysisTimestampRef.current = currentTimestamp;
        } catch (error) {
          console.error('❌ Error auto-saving analysis:', error);
          // Don't show error toast for auto-save, as it's not critical
        } finally {
          isAutoSavingRef.current = false;
        }
      }, 1000);
    } else {
      console.log('⏭️ [Auto-save] Skipping - already saving or duplicate:', {
        isAutoSaving: isAutoSavingRef.current,
        isDuplicate,
        currentTimestamp,
        lastSaved: lastSavedAnalysisTimestampRef.current,
      });
    }

    // Show image with landmarks after analysis
    setShowCephalometricImage(true);
    setActiveTab('image');
  }, [selectedAnalysisType, calculateMeasurementsFromLandmarks, patient, id, user, handleSaveAnalysis, setIsUploadingImage]);

  // Calculate severity based on measured value vs mean ± sd
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

  // Handle view mode change - memoized for performance
  const handleViewModeChange = useCallback((newMode) => {
    setViewMode(newMode);
    // Update showCoordinateSystem based on view mode
    if (newMode === 'coordinate') {
      setShowCoordinateSystem(true);
    } else {
      setShowCoordinateSystem(false);
    }
  }, []);

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

  // Update cephalometric table when analysis type changes
  useEffect(() => {
    const template = cephalometricTemplates[selectedAnalysisType] || cephalometricTemplates.steiner;
    const rawData = patient?.cephalometricRawData;
    const currentTable = patient?.cephalometricTable;

    if (!template) return;

    // If we have raw measurement data, use it with the selected template
    if (rawData && Object.keys(rawData).length > 0) {
      // Check if table needs to be updated (different analysis type or missing data)
      const needsUpdate = !currentTable || 
        Object.keys(currentTable).length === 0 ||
        !Object.keys(template).every(param => currentTable[param]);
      
      if (needsUpdate) {
        const newTable = {};
        Object.keys(template).forEach(param => {
          let measuredValue = '';
          
          // 🔧 FIX: برای PFH و AFH، چند نام مختلف را چک می‌کنیم
          if (param === 'Posterior Facial Height (PFH)') {
            measuredValue = rawData['Posterior Facial Height (PFH)'] || 
                           rawData['Posterior Facial Height (S-Go)'] || 
                           rawData.PFH || 
                           '';
          } else if (param === 'Anterior Facial Height (AFH)') {
            measuredValue = rawData['Anterior Facial Height (AFH)'] || 
                           rawData['Anterior Facial Height (N-Me)'] || 
                           rawData.AFH || 
                           '';
          } else if (param === 'Saddle Angle') {
            // برای سازگاری با داده‌های قدیمی
            measuredValue = rawData['Saddle Angle'] || 
                           rawData['Saddle Angle (N-S-Ar)'] || 
                           rawData['N-S-Ar'] || 
                           '';
          } else if (param === 'Restriction Angle') {
            measuredValue = rawData['Restriction Angle'] || 
                           rawData['Restriction Angle (N-Ar-Go)'] || 
                           rawData['N-Ar-Go'] || 
                           '';
          } else if (param === 'Sagittal Differentiation Angle') {
            measuredValue = rawData['Sagittal Differentiation Angle'] || 
                           rawData['Sagittal Differentiation Angle (Go-Co-N-S)'] || 
                           rawData['Go-Co-N-S'] || 
                           '';
          } else if (param === 'Social Selection Angle') {
            measuredValue = rawData['Social Selection Angle'] || 
                           rawData['Social Selection Angle (Go-Co-Go-Gn)'] || 
                           rawData['Go-Co-Go-Gn'] || 
                           '';
          } else if (param === 'Cultural Ideal Angle') {
            measuredValue = rawData['Cultural Ideal Angle'] || 
                           rawData['Cultural Ideal Angle (N-Co-Go-Co)'] || 
                           rawData['N-Co-Go-Co'] || 
                           '';
          } else if (param === 'First Sagittal Angle') {
            measuredValue = rawData['First Sagittal Angle'] || 
                           rawData['First Sagittal Angle (Ar-Co-Co-Gn)'] || 
                           rawData['Ar-Co-Co-Gn'] || 
                           '';
          } else if (param === 'Skeletal Convexity') {
            // استفاده از مقدار محاسبه شده در canvas (از rawData که از visualizer می‌آید)
            // Map from old parameter names for backward compatibility
            measuredValue = rawData['Skeletal Convexity'] || 
                           rawData['N-A-Pog'] || 
                           rawData['Facial Convexity'] || 
                           '';
            
            // مقدار از visualizer می‌آید که از calculateAngleBetweenLines محاسبه شده
            // calculateAngleBetweenLines زاویه کوچک‌تر را برمی‌گرداند (مثلاً 20°)
            // برای Skeletal Convexity که مقدار نرمال 165° تا 175° است، باید از 180 کم کنیم
            // اما اگر مقدار بیشتر از 90 است (مثلاً 160°)، یعنی قبلاً از 180 کم شده است و نباید دوباره کم کنیم
            if (measuredValue && !isNaN(parseFloat(measuredValue))) {
              const numValue = parseFloat(measuredValue);
              // فقط اگر مقدار کمتر از 90 است، از 180 کم می‌کنیم
              // اگر مقدار بیشتر از 90 است (مثلاً 160°)، یعنی قبلاً تبدیل شده است
              if (numValue < 90) {
                const convertedValue = Math.round((180 - numValue) * 10) / 10;
                console.log('🔍 [useEffect] Skeletal Convexity conversion from rawData:', {
                  original: numValue,
                  converted: convertedValue,
                  rawDataValue: rawData['Skeletal Convexity'],
                  nAPogValue: rawData['N-A-Pog'],
                  facialConvexityValue: rawData['Facial Convexity']
                });
                measuredValue = String(convertedValue);
              } else {
                console.log('🔍 [useEffect] Skeletal Convexity already converted:', {
                  value: numValue,
                  rawDataValue: rawData['Skeletal Convexity'],
                  nAPogValue: rawData['N-A-Pog'],
                  facialConvexityValue: rawData['Facial Convexity']
                });
              }
            }
          } else if (param === 'E-line (UL)') {
            // Map from Arnett analysis parameter name for backward compatibility
            measuredValue = rawData['E-line (UL)'] || 
                           rawData['Upper Lip to E-line'] || 
                           '';
          } else if (param === 'E-line (LL)') {
            // Map from Arnett analysis parameter name for backward compatibility
            measuredValue = rawData['E-line (LL)'] || 
                           rawData['Lower Lip to E-line'] || 
                           '';
          } else {
            measuredValue = rawData[param] || '';
          }
          
          // 🔧 FIX: برای IMPA، اگر مقدار از rawData می‌آید و کمتر از 90 است، از 180 کم می‌کنیم
          if (param === 'IMPA' && measuredValue && !isNaN(parseFloat(measuredValue))) {
            const numValue = parseFloat(measuredValue);
            if (numValue < 90) {
              measuredValue = String(180 - numValue);
            }
          }
          newTable[param] = {
            ...template[param],
            measured: measuredValue,
          };
        });

        setPatient(prev => ({
          ...prev,
          cephalometricTable: newTable,
        }));
      }
    } else if (!currentTable || Object.keys(currentTable).length === 0) {
      // Only create empty table if no existing data and no raw data
      const newTable = {};
      Object.keys(template).forEach(param => {
        newTable[param] = {
          ...template[param],
          measured: '',
        };
      });

      setPatient(prev => ({
        ...prev,
        cephalometricTable: newTable,
      }));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedAnalysisType, patient?.cephalometricRawData]);

  // Check initial display state - only run once on mount or when patient data changes
  useEffect(() => {
    // 🔧 FIX: Skip if we've already initialized (prevents infinite loop)
    if (hasInitializedDisplayRef.current) {
      return;
    }
    
    // Skip if user has explicitly interacted with buttons
    if (userInteractedRef.current) {
      return;
    }

    const hasTable = patient?.cephalometricTable && Object.keys(patient.cephalometricTable).length > 0;
    const hasMeasuredValues = hasTable && Object.values(patient.cephalometricTable).some(
      (param) => param && param.measured && String(param.measured).trim() !== ''
    );
    
    const wasViewingTables = id ? localStorage.getItem(`cephalometric_viewing_tables_${id}`) === 'true' : false;
    
    // 🔧 FIX: Check if we have analysis history to show
    const hasAnalysisHistory = analysisHistory && analysisHistory.length > 0;
    
    if (!hasTable || !hasMeasuredValues) {
      setShowCephalometricImage(true);
      setActiveTab('image');
      setShowCoordinateSystem(false);
      // Reset analysis key to force reload when showing image for the first time
      if (analysisKey === 0) {
        setAnalysisKey(1);
      }
      if (isAnalysisConfirmed) {
        setIsAnalysisConfirmed(false);
        if (id) {
          localStorage.removeItem(`cephalometric_analysis_confirmed_${id}`);
          localStorage.removeItem(`cephalometric_viewing_tables_${id}`);
        }
      }
      // 🔧 FIX: Mark as initialized
      hasInitializedDisplayRef.current = true;
      return;
    }
    
    const savedState = id ? localStorage.getItem(`cephalometric_analysis_confirmed_${id}`) : null;
    const isConfirmedInStorage = savedState === 'true';
    
    if (isConfirmedInStorage && !isAnalysisConfirmed) {
      setIsAnalysisConfirmed(true);
    }
    
    // 🔧 FIX: Always show image with landmarks if we have analysis history
    // This ensures that when the page loads, the latest analysis is visible
    if (hasAnalysisHistory) {
      setShowCephalometricImage(true);
      setActiveTab('image');
      setShowCoordinateSystem(false);
      // Reset analysis key to force reload when showing image for the first time
      if (analysisKey === 0) {
        setAnalysisKey(1);
      }
      // 🔧 FIX: Auto-select latest image (index 0) and latest analysis
      if (patient?.lateralImages && patient.lateralImages.length > 0 && selectedImageIndex === null) {
        setSelectedImageIndex(0);
      }
      // 🔧 FIX: Mark as initialized only when we have data to show
      hasInitializedDisplayRef.current = true;
    } else {
      // Only set initial state if user hasn't interacted and no analysis history
      if (wasViewingTables) {
        setShowCephalometricImage(false);
        setActiveTab('table');
        setShowCoordinateSystem(false);
      } else {
        setShowCephalometricImage(true);
        setActiveTab('image');
        setShowCoordinateSystem(false);
        // Reset analysis key to force reload when showing image for the first time
        if (analysisKey === 0) {
          setAnalysisKey(1);
        }
      }
      // 🔧 FIX: Mark as initialized
      hasInitializedDisplayRef.current = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id, patient?.cephalometricTable, analysisHistory]); // Added analysisHistory to dependencies

  // Get lateral image URL - stable reference to prevent unnecessary re-renders
  const lateralImageUrl = useMemo(() => {
    if (!patient?.lateralImages || patient.lateralImages.length === 0) {
      console.log('🖼️ [lateralImageUrl] No lateral images available');
      return null;
    }
    // 🔧 FIX: Default to latest image (index 0 after sorting newest first)
    const imageIndex = selectedImageIndex >= 0 && selectedImageIndex < patient.lateralImages.length
      ? selectedImageIndex
      : 0; // Default to latest image (index 0)
    
    const lateralImage = patient.lateralImages[imageIndex];
    if (!lateralImage?.path) {
      console.log('🖼️ [lateralImageUrl] No path for image at index:', imageIndex);
      return null;
    }
    const imagePath = getImageUrl(lateralImage.path);
    console.log('🖼️ [lateralImageUrl] Computed URL:', {
      selectedImageIndex,
      imageIndex,
      imageId: lateralImage.id,
      path: lateralImage.path,
      imagePath,
      totalImages: patient.lateralImages.length,
      isUploading: isUploadingImage,
    });
    return imagePath;
  }, [patient?.lateralImages, selectedImageIndex, isUploadingImage]);

  // Chart data - normalized relative to mean for each parameter (max 8 parameters)
  const radarChartData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const measured = parseFloat(measuredStr);
      if (isNaN(measured)) {
        return null;
      }
      
      // Get mean value
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.-]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      
      // Get SD value for normalization when mean = 0
      const sdStr = String(item?.sd || '').trim();
      const sdMatch = sdStr.match(/^([\d.]+)/);
      const sd = sdMatch ? parseFloat(sdMatch[1]) : 2; // Default SD = 2 if not found
      
      // اگر mean = 0 باشد، از SD برای normalize کردن استفاده می‌کنیم
      if (mean === 0) {
        // برای mean = 0، مقدار نرمال (0) در مرکز (50) قرار می‌گیرد
        // هر SD معادل 10 واحد در نمودار است
        // measured = 0 → normalized = 50
        // measured = SD → normalized = 60
        // measured = -SD → normalized = 40
        if (sd > 0) {
          const normalized = 50 + (measured / sd) * 10;
          // Clamp between 25 and 65 for display
          return Math.max(25, Math.min(65, normalized));
        }
        // اگر SD هم 0 باشد، مقدار را در مرکز قرار می‌دهیم
        return 50;
      }
      
      // Normalize: center at 50 (mean), scale based on deviation from mean
      // اگر مقدار خیلی کم بود (کمتر از نصف نرمال)، به اندازه نصف مقدار نرمال روی نمودار داشته باشد (25)
      if (measured < mean / 2) {
        return 25;
      }
      
      // اگر مقدار خیلی بیشتر بود (بیشتر از 30% نرمال)، نهایتا به اندازه 30% بیشتر از مقدار نرمال روی نمودار داشته باشد (65)
      if (measured > mean * 1.3) {
        return 65;
      }
      
      // در حالت عادی: Formula: normalized = 50 + ((measured - mean) / mean) * 50
      // This makes mean = 50, and ±100% deviation = 0 or 100
      const normalized = 50 + ((measured - mean) / mean) * 50;
      
      // Clamp between 25 and 65 for display (محدود به محدوده جدید)
      return Math.max(25, Math.min(65, normalized));
    });
  }, [patient?.cephalometricTable]);

  const normalRangeData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.-]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      
      // Mean is always at center (50) in normalized scale
      // حتی اگر mean = 0 باشد، در مرکز (50) قرار می‌گیرد
      return 50;
    });
  }, [patient?.cephalometricTable]);

  // Real values data for non-radar charts (max 8 parameters)
  const realChartData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const measured = parseFloat(measuredStr);
      if (isNaN(measured)) {
        return null;
      }
      return measured;
    });
  }, [patient?.cephalometricTable]);

  const realNormalRangeData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      return mean;
    });
  }, [patient?.cephalometricTable]);

  // Difference data for column negative (measured - mean) (max 8 parameters)
  const differenceData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const measured = parseFloat(measuredStr);
      if (isNaN(measured)) {
        return null;
      }
      
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      
      // Calculate difference: measured - mean
      return measured - mean;
    });
  }, [patient?.cephalometricTable]);

  // Normalized data for area and bar-negative charts (percentage deviation from mean)
  const normalizedAreaChartData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const measured = parseFloat(measuredStr);
      if (isNaN(measured)) {
        return null;
      }
      
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.-]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      
      // Calculate percentage deviation: ((measured - mean) / mean) * 100
      if (mean === 0) {
        // If mean is 0, use SD for normalization
        const sdStr = String(item?.sd || '').trim();
        const sdMatch = sdStr.match(/^([\d.]+)/);
        const sd = sdMatch ? parseFloat(sdMatch[1]) : 2;
        if (sd > 0) {
          return (measured / sd) * 100; // Percentage of SD
        }
        return 0;
      }
      
      return ((measured - mean) / mean) * 100;
    });
  }, [patient?.cephalometricTable]);

  const normalizedAreaNormalRangeData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(() => 0);
  }, [patient?.cephalometricTable]);

  // Normalized difference data (percentage deviation) for bar-negative
  const normalizedDifferenceData = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.values(patient.cephalometricTable).slice(0, 8).map(item => {
      const measuredStr = String(item?.measured || '').trim();
      if (!measuredStr || measuredStr === '' || measuredStr === 'undefined' || measuredStr === 'null') {
        return null;
      }
      const measured = parseFloat(measuredStr);
      if (isNaN(measured)) {
        return null;
      }
      
      const meanStr = String(item?.mean || '').trim();
      const meanMatch = meanStr.match(/^([\d.-]+)/);
      if (!meanMatch) {
        return null;
      }
      const mean = parseFloat(meanMatch[1]);
      if (isNaN(mean)) {
        return null;
      }
      
      // Calculate percentage deviation: ((measured - mean) / mean) * 100
      if (mean === 0) {
        // If mean is 0, use SD for normalization
        const sdStr = String(item?.sd || '').trim();
        const sdMatch = sdStr.match(/^([\d.]+)/);
        const sd = sdMatch ? parseFloat(sdMatch[1]) : 2;
        if (sd > 0) {
          return (measured / sd) * 100; // Percentage of SD
        }
        return 0;
      }
      
      return ((measured - mean) / mean) * 100;
    });
  }, [patient?.cephalometricTable]);

  // Chart series - use normalized data for radar, real values for others (or normalized if selected)
  const chartSeries = useMemo(() => {
    const series = [];
    
    if (chartType === 'radar') {
      // Radar always uses normalized data (0-100 scale)
      // Add normal range series for radar
      if (normalRangeData.some(v => v !== null)) {
        series.push({
          name: 'محدوده نرمال',
          data: normalRangeData,
        });
      }
      
      // Add measured values series for radar
      if (radarChartData.some(v => v !== null)) {
        series.push({
          name: 'بیمار',
          data: radarChartData,
        });
      }
    } else if (chartType === 'bar-negative') {
      // For column negative, show difference (real or normalized based on selection)
      const dataToUse = chartNormalization === 'normalized' ? normalizedDifferenceData : differenceData;
      if (dataToUse.some(v => v !== null)) {
        series.push({
          name: chartNormalization === 'normalized' ? 'اختلاف از نرمال (%)' : 'اختلاف از نرمال',
          data: dataToUse,
        });
      }
    } else {
      // For area and other charts, use real or normalized values based on selection
      if (chartNormalization === 'normalized') {
        // Use normalized data (percentage deviation)
        if (normalizedAreaNormalRangeData.some(v => v !== null)) {
          series.push({
            name: 'محدوده نرمال',
            data: normalizedAreaNormalRangeData,
          });
        }
        
        if (normalizedAreaChartData.some(v => v !== null)) {
          series.push({
            name: 'بیمار',
            data: normalizedAreaChartData,
          });
        }
      } else {
        // Use real values
      if (realNormalRangeData.some(v => v !== null)) {
        series.push({
          name: 'محدوده نرمال',
          data: realNormalRangeData,
        });
      }
      
      if (realChartData.some(v => v !== null)) {
        series.push({
          name: 'بیمار',
          data: realChartData,
        });
        }
      }
    }
    
    return series;
  }, [chartType, chartNormalization, radarChartData, normalRangeData, realChartData, realNormalRangeData, differenceData, normalizedAreaChartData, normalizedAreaNormalRangeData, normalizedDifferenceData]);

  // Store original values for tooltip display (max 8 parameters)
  const chartOriginalValues = useMemo(() => {
    if (!patient?.cephalometricTable) return {};
    const values = {};
    Object.entries(patient.cephalometricTable).slice(0, 8).forEach(([param, item]) => {
      const measuredStr = String(item?.measured || '').trim();
      if (measuredStr && measuredStr !== '' && measuredStr !== 'undefined' && measuredStr !== 'null') {
        const measured = parseFloat(measuredStr);
        if (!isNaN(measured)) {
          values[param] = measured;
        }
      }
    });
    return values;
  }, [patient?.cephalometricTable]);

  // Function to format parameter names for radar chart (short and concise)
  const formatRadarChartCategory = (param) => {
    if (!param) return param;
    
    // Handle Ratio parameters - convert to fraction and remove "Ratio"
    if (param.includes('/') && param.trim().endsWith('Ratio')) {
      // Remove "Ratio" from the end
      const paramWithoutRatio = param.trim().replace(/\s*Ratio\s*$/i, '');
      // Split by "/" to get numerator and denominator
      const parts = paramWithoutRatio.split('/').map(p => p.trim());
      if (parts.length === 2) {
        // Format as fraction: numerator/denominator
        return `${parts[0]}/${parts[1]}`;
      }
      return paramWithoutRatio;
    }
    
    // Common abbreviations for cephalometric parameters
    const abbreviations = {
      // Keep short names as is, but add common long names
      'Sella-Nasion': 'S-N',
      'Nasion-Sella': 'N-S',
      'Articulare-Gonion': 'Ar-Go',
      'Gonion-Articulare': 'Go-Ar',
      'Nasion-Gonion': 'N-Go',
      'Gonion-Nasion': 'Go-N',
      'Gonion-Menton': 'Go-Me',
      'Menton-Gonion': 'Me-Go',
      'Sella-Articulare': 'S-Ar',
      'Articulare-Sella': 'Ar-S',
      'Sella-Gonion': 'S-Go',
      'Gonion-Sella': 'Go-S',
      'Nasion-Menton': 'N-Me',
      'Menton-Nasion': 'Me-N',
      'Sella-Gnathion': 'S-Gn',
      'Gnathion-Sella': 'Gn-S',
      'Nasion-Gnathion': 'N-Gn',
      'Gnathion-Nasion': 'Gn-N',
    };
    
    // Check if there's an abbreviation
    const trimmedParam = param.trim();
    if (abbreviations[trimmedParam]) {
      return abbreviations[trimmedParam];
    }
    
    // For other parameters, keep them as is (they're already short like S-N, Ar-Go, etc.)
    return param;
  };

  // Chart categories - formatted for display (max 8 parameters)
  const radarChartCategories = useMemo(() => {
    if (!patient?.cephalometricTable) return [];
    return Object.keys(patient.cephalometricTable).slice(0, 8).map(formatRadarChartCategory);
  }, [patient?.cephalometricTable]);

  // Mapping از نام‌های کوتاه به نام‌های کامل برای tooltip (max 8 parameters)
  const categoryFullNames = useMemo(() => {
    if (!patient?.cephalometricTable) return {};
    const mapping = {};
    Object.keys(patient.cephalometricTable).slice(0, 8).forEach((fullName) => {
      const shortName = formatRadarChartCategory(fullName);
      mapping[shortName] = fullName;
    });
    return mapping;
  }, [patient?.cephalometricTable]);

  // اضافه کردن tooltip به labels بعد از render شدن chart
  useEffect(() => {
    if (!radarChartCategories.length || !categoryFullNames || Object.keys(categoryFullNames).length === 0) return;

    const addTooltipsToLabels = () => {
      // پیدا کردن تمام label elements در xaxis
      const labelElements = document.querySelectorAll('#radar-chart-container .apexcharts-xaxis-label');
      
      if (labelElements.length === 0) {
        return; // Chart هنوز render نشده
      }
      
      // اضافه کردن tooltip به parent elements (g elements) که شامل text هستند
      labelElements.forEach((labelElement) => {
        const textElement = labelElement.querySelector('text');
        if (textElement) {
          const shortName = textElement.textContent?.trim();
          if (shortName) {
            // جستجو در categoryFullNames - ممکن است نام دقیقاً مطابقت نداشته باشد
            let fullName = categoryFullNames[shortName];
            
            // اگر پیدا نشد، سعی کن با حذف "..." پیدا کنی
            if (!fullName && shortName.endsWith('...')) {
              const shortNameWithoutEllipsis = shortName.replace('...', '').trim();
              fullName = categoryFullNames[shortNameWithoutEllipsis];
            }
            
            // اگر هنوز پیدا نشد، سعی کن در تمام keys جستجو کنی
            if (!fullName) {
              const matchingKey = Object.keys(categoryFullNames).find(key => {
                const formattedKey = formatRadarChartCategory(key);
                return formattedKey === shortName || formattedKey === shortName.replace('...', '').trim();
              });
              if (matchingKey) {
                fullName = categoryFullNames[matchingKey];
              }
            }
            
            if (fullName) {
              // اضافه کردن title attribute برای tooltip
              textElement.setAttribute('title', fullName);
              labelElement.setAttribute('title', fullName);
              // اضافه کردن cursor
              textElement.style.cursor = 'help';
              labelElement.style.cursor = 'help';
            }
          }
        }
      });
    };

    // اجرای تابع با تاخیر‌های مختلف برای اطمینان از render شدن chart
    const timeoutId1 = setTimeout(addTooltipsToLabels, 300);
    const timeoutId2 = setTimeout(addTooltipsToLabels, 800);
    const timeoutId3 = setTimeout(addTooltipsToLabels, 1500);
    
    // همچنین بعد از هر تغییر در DOM
    const observer = new MutationObserver(() => {
      // استفاده از debounce برای جلوگیری از اجرای زیاد
      clearTimeout(observer.timeoutId);
      observer.timeoutId = setTimeout(addTooltipsToLabels, 200);
    });

    const container = document.getElementById('radar-chart-container');
    if (container) {
      observer.observe(container, {
        childList: true,
        subtree: true,
        attributes: false,
      });
    }

    return () => {
      clearTimeout(timeoutId1);
      clearTimeout(timeoutId2);
      clearTimeout(timeoutId3);
      if (observer.timeoutId) {
        clearTimeout(observer.timeoutId);
      }
      observer.disconnect();
    };
  }, [radarChartCategories, categoryFullNames, chartSeries]);

  // Chart options - dynamically configured based on chartType
  const theme = useTheme();
  const chartOptionsConfig = useMemo(() => {
    // Use modern colors for all charts
    // Light blue for normal range, light red for patient data
    const chartColors = ['#60a5fa', '#f87171']; // Light blue and light red
    const isDarkMode = theme.palette.mode === 'dark';
    const tooltipBgColor = isDarkMode ? '#282b33' : '#ffffff';
    const tooltipTextColor = isDarkMode ? '#ffffff' : '#212b36';
    
    const baseOptions = {
      chart: {
        animations: {
          enabled: true,
          easing: 'swing', // انواع: 'linear', 'easein', 'easeout', 'easeinout', 'swing'
          speed: 800,
          animateGradually: {
            enabled: true,
            delay: 150,
          },
          dynamicAnimation: {
            enabled: true,
            speed: 350,
          },
        },
      },
      xaxis: { 
        categories: radarChartCategories,
        labels: {
          style: {
            fontSize: '12px',
          },
          maxWidth: 30,
          trim: true,
          hideOverlappingLabels: true,
          formatter: (value) => {
            // کوتاه کردن متن اگر بیشتر از 5 کاراکتر باشد
            if (value && value.length > 5) {
              return `${value.substring(0, 5)}...`;
            }
            return value;
          },
        },
      },
      yaxis: chartType === 'radar' ? {
        max: 100,
        min: 0,
        tickAmount: 5,
        labels: {
          formatter: (value) => {
            // Convert normalized value back to percentage deviation
            const deviation = ((value - 50) / 50) * 100;
            if (deviation === 0) return 'میانگین';
            return `${deviation > 0 ? '+' : ''}${deviation.toFixed(0)}%`;
          },
        },
      } : (() => {
        // For non-radar charts, calculate min/max based on normalization mode
        let allValues;
        if (chartType === 'bar-negative') {
          // For bar-negative, use difference data (real or normalized)
          allValues = chartNormalization === 'normalized' 
            ? normalizedDifferenceData.filter(v => v !== null && !isNaN(v))
            : differenceData.filter(v => v !== null && !isNaN(v));
        } else {
          // For area and other charts, use appropriate data based on normalization
          if (chartNormalization === 'normalized') {
            allValues = [...normalizedAreaChartData, ...normalizedAreaNormalRangeData].filter(v => v !== null && !isNaN(v));
          } else {
            allValues = [...realChartData, ...realNormalRangeData].filter(v => v !== null && !isNaN(v));
          }
        }
        
        if (allValues.length === 0) {
          return {
            min: chartNormalization === 'normalized' ? -50 : 0,
            max: chartNormalization === 'normalized' ? 50 : 100,
            tickAmount: 5,
            labels: {
              formatter: (value) => {
                if (chartNormalization === 'normalized') {
                  return typeof value === 'number' ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : value;
                }
                return typeof value === 'number' ? value.toFixed(1) : value;
              },
            },
          };
        }
        
        const minValue = Math.min(...allValues);
        const maxValue = Math.max(...allValues);
        const range = maxValue - minValue;
        
        // Calculate padding: use 10% of range, but ensure minimum padding for visibility
        // For very small ranges, use a percentage of the absolute values to ensure visibility
        let padding;
        if (range === 0) {
          // All values are the same - add padding based on absolute value
          padding = Math.max(Math.abs(minValue) * 0.1, Math.abs(maxValue) * 0.1, chartNormalization === 'normalized' ? 5 : 0.1);
        } else {
          // Use 10% of range, but ensure at least 5% of the larger absolute value for visibility
          const basePadding = range * 0.1;
          const minPadding = Math.max(Math.abs(minValue), Math.abs(maxValue)) * 0.05;
          padding = Math.max(basePadding, minPadding);
          // For normalized mode, ensure minimum padding of 5%
          if (chartNormalization === 'normalized' && padding < 5) {
            padding = 5;
          }
        }
        
        return {
          min: minValue - padding, // Allow negative values
          max: maxValue + padding,
          tickAmount: 5,
          labels: {
            formatter: (value) => {
              if (chartNormalization === 'normalized') {
                // Show percentage with sign
                return typeof value === 'number' ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : value;
              }
              // Show real values
              return typeof value === 'number' ? value.toFixed(1) : value;
            },
          },
        };
      })(),
      dataLabels: {
        enabled: true,
        formatter: (value, opts) => {
          const { seriesIndex, dataPointIndex } = opts || {};
          // Only show data labels for measured values series (seriesIndex === 1 for area, seriesIndex === 0 for bar-negative)
          const isMeasuredSeries = chartType === 'bar-negative' ? seriesIndex === 0 : seriesIndex === 1;
          if (isMeasuredSeries && patient?.cephalometricTable) {
            if (chartType === 'radar') {
            const paramKeys = Object.keys(patient.cephalometricTable).slice(0, 8);
            const param = paramKeys[dataPointIndex];
            const originalValue = chartOriginalValues[param];
            if (originalValue !== undefined) {
                const formatted = originalValue.toString();
                return formatted.includes('.') ? formatted.replace(/\.?0+$/, '') : formatted;
              }
            } else if (chartNormalization === 'normalized') {
              // For normalized mode, show percentage
              return typeof value === 'number' ? `${value > 0 ? '+' : ''}${value.toFixed(1)}%` : value;
              } else {
              // For real values mode, show original value
              const paramKeys = Object.keys(patient.cephalometricTable).slice(0, 8);
              const param = paramKeys[dataPointIndex];
              const originalValue = chartOriginalValues[param];
              if (originalValue !== undefined) {
                return typeof originalValue === 'number' ? originalValue.toFixed(1) : originalValue.toString();
              }
            }
          }
          // Don't show labels for normal range series
          return '';
        },
        style: {
          fontSize: '12px',
          fontWeight: 600,
          colors: ['rgba(248, 113, 113, 0.8)'], // #f87171 (light red) with 0.8 opacity
        },
        background: {
          enabled: true,
          borderRadius: 4,
          borderWidth: 1,
          borderColor: '#f87171', // Light red
          opacity: 1,
          dropShadow: {
            enabled: false,
          },
        },
        offsetY: -5,
      },
      tooltip: {
        enabled: false,
      },
      colors: chartColors,
      states: { hover: { filter: { type: 'lighten', value: 0.1 } }, active: { filter: { type: 'none' } } },
      legend: {
        show: true,
        position: 'top',
      },
    };

    // Configure options based on chart type
    switch (chartType) {
      case 'radar':
        return {
          ...baseOptions,
          plotOptions: {
            radar: {
              polygons: { strokeColors: 'rgba(145, 158, 171, 0.2)', strokeWidth: 1, fill: { colors: ['transparent'] } },
            },
          },
          markers: { size: 4, strokeColors: ['#60a5fa', '#f87171'], strokeWidth: 2 }, // Light blue and light red
          stroke: { width: 2, curve: 'smooth' },
          fill: { opacity: 0.1 },
        };
      
      case 'area': {
        // Determine strokeDashArray based on series count
        // First series (normal range) should be dashed, second (patient) should be solid
        const areaStrokeDashArray = chartSeries.length === 2 ? [5, 0] : (chartSeries.length === 1 ? [0] : []);
        
        return {
          ...baseOptions,
          colors: chartSeries.length === 2 
            ? [hexAlpha(chartColors[0], 0.6), hexAlpha(chartColors[1], 0.6)] // More transparent colors
            : chartColors.map(color => hexAlpha(color, 0.6)),
          stroke: { 
            width: 3, 
            curve: 'smooth',
            lineCap: 'round',
            strokeDashArray: areaStrokeDashArray,
          },
          fill: { 
            opacity: 0.2, // Reduced from 0.3 to 0.2 for more transparency
            type: 'solid',
          },
          markers: { 
            size: 6, 
            strokeColors: chartSeries.length === 2 
              ? [hexAlpha(chartColors[0], 0.6), hexAlpha(chartColors[1], 0.6)]
              : chartColors.map(color => hexAlpha(color, 0.6)), 
            strokeWidth: 2,
            fillColors: chartSeries.length === 2 
              ? [hexAlpha(chartColors[0], 0.6), hexAlpha(chartColors[1], 0.6)]
              : chartColors.map(color => hexAlpha(color, 0.6)),
            hover: {
              size: 8,
              sizeOffset: 2,
            },
          },
          dataLabels: { 
            enabled: false,
          },
          tooltip: { 
            enabled: true,
            shared: true,
            intersect: false,
            theme: isDarkMode ? 'dark' : 'light',
            style: {
              fontSize: '12px',
              fontFamily: theme.typography.fontFamily,
            },
            fillSeriesColor: false,
            marker: {
              show: true,
            },
          },
          grid: {
            strokeDashArray: 3,
            xaxis: {
              lines: {
                show: true,
              },
            },
            yaxis: {
              lines: {
                show: true,
              },
            },
          },

        };
      }
      
      case 'bar-negative':
        return {
          stroke: { width: 0 },
          xaxis: { 
            categories: radarChartCategories,
            labels: {
              style: {
                fontSize: '12px',
              },
              maxWidth: 30,
              trim: true,
              hideOverlappingLabels: true,
              formatter: (value) => {
                if (value && value.length > 5) {
                  return `${value.substring(0, 5)}...`;
                }
                return value;
              },
            },
          },
          tooltip: { 
            enabled: true,
            theme: isDarkMode ? 'dark' : 'light',
            style: {
              fontSize: '12px',
              fontFamily: theme.typography.fontFamily,
            },
            fillSeriesColor: false,
            y: { 
              title: { formatter: () => '' },
            },
            marker: {
              show: true,
            },
          },
          dataLabels: { enabled: false },
          plotOptions: {
            bar: {
              borderRadius: 8, // Rounded bars for column negative
              colors: {
                ranges: [
                  {
                    from: -1000000, // Use a very large negative number to cover all negative values
                    to: -0.01,
                    color: hexAlpha(chartColors[1], 0.6), // Light red with 60% opacity for negative values
                  },
                  {
                    from: 0,
                    to: 1000000, // Use a very large positive number to cover all positive values
                    color: hexAlpha(chartColors[0], 0.6), // Light blue with 60% opacity for positive values
                  },
                ],
              },
            },
          },

        };
      
      default:
        return baseOptions;
    }
  }, [chartType, chartNormalization, radarChartCategories, chartOriginalValues, patient?.cephalometricTable, realChartData, realNormalRangeData, differenceData, normalizedAreaChartData, normalizedAreaNormalRangeData, normalizedDifferenceData, theme, chartSeries]);

  const chartOptions = useChart(chartOptionsConfig);

  // Helper function to format parameter name as fraction if it contains "/"
  const formatParameterName = (param) => {
    if (param && param.includes('/')) {
      // Remove "Ratio" from the end if it exists
      const paramWithoutRatio = param.trim().replace(/\s*Ratio\s*$/i, '');
      
      const parts = paramWithoutRatio.split('/');
      if (parts.length === 2) {
        return (
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px', direction: 'ltr' }}>
            <span style={{ display: 'inline-block', textAlign: 'center' }}>
              <span style={{ display: 'block', borderBottom: '1px solid', paddingBottom: '2px' }}>
                {parts[0].trim()}
              </span>
              <span style={{ display: 'block', paddingTop: '2px' }}>
                {parts[1].trim()}
              </span>
            </span>
          </span>
        );
      }
    }
    // Remove "Ratio" from the end even if it doesn't contain "/"
    return param ? param.trim().replace(/\s*Ratio\s*$/i, '') : param;
  };

  // Memoized table rows
  const cephalometricRows = useMemo(() => {
    if (!patient?.cephalometricTable || typeof patient.cephalometricTable !== 'object') {
      console.log('⚠️ [cephalometricRows] No table data available:', {
        hasPatient: !!patient,
        hasTable: !!patient?.cephalometricTable,
        tableType: typeof patient?.cephalometricTable,
      });
      return [];
    }
    
    const entries = Object.entries(patient.cephalometricTable);
    console.log('📊 [cephalometricRows] Generating rows:', {
      tableKeys: entries.length,
      selectedAnalysisType,
      selectedAnalysisIndex,
    });
    
    return entries.map(([param, data]) => {
      const measured = data?.measured || '';
      const mean = data?.mean || '';
      const sd = data?.sd || '';
      const calculatedSeverity = calculateSeverity(measured, mean, sd);
      
      // Format measured value - show one decimal place
      let formattedMeasured = measured;
      if (measured && measured !== '' && measured !== 'undefined' && measured !== 'null') {
        const measuredNum = parseFloat(measured);
        if (!isNaN(measuredNum)) {
          formattedMeasured = measuredNum.toFixed(1);
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
        interpretation: getInterpretation(param, calculatedSeverity, formattedMeasured, mean, sd),
      };
    });
  }, [calculateSeverity, patient, selectedAnalysisIndex, selectedAnalysisType]);

  // Paginated rows
  const paginatedRows = useMemo(() => cephalometricRows.slice(
      tablePage * rowsPerPage,
      tablePage * rowsPerPage + rowsPerPage
    ), [cephalometricRows, tablePage, rowsPerPage]);

  // Pagination handlers
  const handleChangePage = (event, newPage) => {
    setTablePage(newPage);
  };

  const handleChangeRowsPerPage = (event) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setTablePage(0);
  };

  if (loading) {
    return null;
  }

  const hasTable = patient?.cephalometricTable && Object.keys(patient.cephalometricTable).length > 0;
  const hasMeasuredValues = hasTable && Object.values(patient.cephalometricTable).some(
    (param) => param && param.measured && String(param.measured).trim() !== ''
  );

  return (
    <DashboardContent>
      <Stack spacing={3}>
        {/* Analysis Type Selection and Results View - Preload for smooth tab switching */}
        {hasTable && hasMeasuredValues && (
          <Box
            sx={{
              position: 'relative',
              visibility: deferredActiveTab === 'table' ? 'visible' : 'hidden',
              opacity: deferredActiveTab === 'table' ? 1 : 0,
              transform: deferredActiveTab === 'table' ? 'translateY(0)' : 'translateY(-10px)',
              height: deferredActiveTab === 'table' ? 'auto' : 0,
              overflow: deferredActiveTab === 'table' ? 'visible' : 'hidden',
              transition: 'opacity 0.15s ease-in-out, visibility 0.15s ease-in-out, transform 0.15s ease-in-out',
              willChange: 'opacity, visibility, transform',
              contain: 'layout style paint',
              pointerEvents: deferredActiveTab === 'table' ? 'auto' : 'none',
              backfaceVisibility: 'hidden',
              WebkitBackfaceVisibility: 'hidden',
            }}
          >
            <Card>
              <CardContent>
                <Stack spacing={2}>

                  {/* Analysis Type Selection */}
                  <FormControl fullWidth>
                    <InputLabel>نوع آنالیز</InputLabel>
                    <Select
                      value={selectedAnalysisType}
                      label="نوع آنالیز"
                      onChange={(e) => setSelectedAnalysisType(e.target.value)}
                    >
                      <MenuItem value="general">عمومی</MenuItem>
                      <MenuItem value="steiner">Steiner</MenuItem>
                      <MenuItem value="ricketts">Ricketts</MenuItem>
                      <MenuItem value="mcnamara">McNamara</MenuItem>
                      <MenuItem value="wits">Wits</MenuItem>
                      <MenuItem value="tweed">Tweed</MenuItem>
                      <MenuItem value="jarabak">Jarabak</MenuItem>
                      <MenuItem value="sassouni">Sassouni</MenuItem>
                      <MenuItem value="leganBurstone">Legan & Burstone</MenuItem>
                      <MenuItem value="arnettMcLaughlin">Arnett & McLaughlin</MenuItem>
                      <MenuItem value="holdaway">Holdaway</MenuItem>
                      <MenuItem value="softTissueAngular">Soft Tissue Angular</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Analysis Table */}
                  <TableContainer sx={{ borderRadius: '10px' }}>
                    <Table>
                      <TableHead>
                        <TableRow>
                          <TableCell align="center" sx={{ minWidth: '200px', width: '15%' }}>پارامتر</TableCell>
                          <TableCell align="center">مقدار</TableCell>
                          <TableCell align="center" sx={{ minWidth: '120px' }}>میانگین</TableCell>
                          <TableCell align="center">وضعیت</TableCell>
                          <TableCell align="center" sx={{ minWidth: '250px' }}>تفسیر</TableCell>
                          <TableCell align="center" sx={{ minWidth: '400px', width: '60%' }}>یادداشت</TableCell>
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {paginatedRows.map((row) => (
                          <TableRow key={row.parameter}>
                            <TableCell align="center">
                              <Typography variant="subtitle2">
                                {formatParameterName(row.parameter)}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <TextField
                                size="small"
                                value={row.measured}
                                onChange={(e) => {
                                  const newValue = e.target.value;
                                  handleCellEdit(row.parameter, newValue);
                                }}
                                onBlur={(e) => {
                                  const {value} = e.target;
                                  if (value && value !== '') {
                                    const numValue = parseFloat(value);
                                    if (!isNaN(numValue)) {
                                      const formatted = numValue.toString();
                                      const cleaned = formatted.includes('.')
                                        ? formatted.replace(/\.?0+$/, '')
                                        : formatted;
                                      handleCellEdit(row.parameter, cleaned);
                                    }
                                  }
                                }}
                                sx={{
                                  width: 70,
                                  direction: 'ltr', // اطمینان از نمایش صحیح علامت منفی قبل از عدد
                                  '& .MuiInputBase-input': {
                                    textAlign: 'center',
                                    direction: 'ltr', // اطمینان از نمایش صحیح علامت منفی قبل از عدد
                                  },
                                }}
                              />
                            </TableCell>
                            <TableCell align="center" style={{ direction: 'ltr' }}>{row.meanDisplay}</TableCell>
                            <TableCell align="center">
                              <Label
                                variant="soft"
                                color={
                                  row.severity === 'نرمال' ? 'success' :
                                  row.severity === 'بالا' || row.severity === 'پایین' ? 'warning' :
                                  'default'
                                }
                                sx={{ fontWeight: 'normal' }}
                              >
                                {row.severity}
                              </Label>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {row.interpretation}
                              </Typography>
                            </TableCell>
                            <TableCell>
                              <Typography variant="body2" color="text.secondary">
                                {row.note}
                              </Typography>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </TableContainer>

                  {/* Table Pagination */}
                  <TablePagination
                    component="div"
                    count={cephalometricRows.length}
                    page={tablePage}
                    onPageChange={handleChangePage}
                    rowsPerPage={rowsPerPage}
                    onRowsPerPageChange={handleChangeRowsPerPage}
                    labelRowsPerPage="تعداد سطرها در هر صفحه:"
                    labelDisplayedRows={({ from, to, count }) => `${from}-${to} از ${count}`}
                  />
                </Stack>
              </CardContent>
            </Card>
          </Box>
        )}

        {/* Image and Landmark Display - Preload for smooth tab switching */}
        <Box
          sx={{
            position: 'relative',
            visibility: deferredActiveTab === 'image' ? 'visible' : 'hidden',
            opacity: deferredActiveTab === 'image' ? 1 : 0,
            transform: deferredActiveTab === 'image' ? 'translateY(0)' : 'translateY(-10px)',
            height: deferredActiveTab === 'image' ? 'auto' : 0,
            overflow: deferredActiveTab === 'image' ? 'visible' : 'hidden',
            transition: 'opacity 0.15s ease-in-out, visibility 0.15s ease-in-out, transform 0.15s ease-in-out',
            willChange: 'opacity, visibility, transform',
            contain: 'layout style paint',
            pointerEvents: deferredActiveTab === 'image' ? 'auto' : 'none',
            backfaceVisibility: 'hidden',
            WebkitBackfaceVisibility: 'hidden',
          }}
        >
          <Suspense fallback={<LoadingFallback />}>
            <CephalometricAIAnalysis
            key={`analysis-${analysisKey}-${patient?.lateralImages?.[selectedImageIndex]?.id || lateralImageUrl || 'no-image'}`}
            onLandmarksDetected={onLandmarksDetected}
            lateralImageUrl={lateralImageUrl}
            initialLandmarks={patient?.cephalometricLandmarks || null}
            showCoordinateSystem={showCoordinateSystem}
            viewMode={viewMode}
            onViewModeChange={handleViewModeChange}
            onSaveAnalysis={handleSaveAnalysis}
            saving={saving}
            hasPatient={!!patient}
            analysisHistory={analysisHistory}
            selectedAnalysisIndex={selectedAnalysisIndex}
            onSelectedAnalysisIndexChange={setSelectedAnalysisIndex}
            onDeleteAnalysis={handleDeleteAnalysis}
            deleteDialogOpen={deleteDialogOpen}
            onDeleteDialogOpenChange={setDeleteDialogOpen}
            analysisToDelete={analysisToDelete}
            onAnalysisToDeleteChange={setAnalysisToDelete}
            deleting={deleting}
            selectedImageIndex={selectedImageIndex}
            onSelectedImageIndexChange={handleSelectedImageIndexChange}
            cephalometricTable={patient?.cephalometricTable}
            selectedAnalysisType={selectedAnalysisType}
            patientInfo={patient}
            lateralImages={patient?.lateralImages || []}
            onImageUpload={handleImageUpload}
            patientId={id}
            onDeleteImage={handleDeleteImage}
            isUploadingImage={isUploadingImage}
          />
          </Suspense>
        </Box>

        {/* Chart Display - Preload for smooth tab switching */}
        {hasTable && hasMeasuredValues && chartSeries.length > 0 && (
          <Box
            sx={{
              position: 'relative',
              visibility: deferredActiveTab === 'chart' ? 'visible' : 'hidden',
              opacity: deferredActiveTab === 'chart' ? 1 : 0,
              transform: deferredActiveTab === 'chart' ? 'translateY(0)' : 'translateY(-10px)',
              height: deferredActiveTab === 'chart' ? 'auto' : 0,
              overflow: deferredActiveTab === 'chart' ? 'visible' : 'hidden',
              transition: 'opacity 0.15s ease-in-out, visibility 0.15s ease-in-out, transform 0.15s ease-in-out',
              willChange: 'opacity, visibility, transform',
              contain: 'layout style paint',
              pointerEvents: deferredActiveTab === 'chart' ? 'auto' : 'none',
              backfaceVisibility: 'hidden',
              WebkitBackfaceVisibility: 'hidden',
            }}
          >
            <Card>
              <CardContent>
                <Stack spacing={2}>
                  {/* Analysis Type Selection */}
                  <FormControl fullWidth>
                    <InputLabel>نوع آنالیز</InputLabel>
                    <Select
                      value={selectedAnalysisType}
                      label="نوع آنالیز"
                      onChange={(e) => setSelectedAnalysisType(e.target.value)}
                    >
                      <MenuItem value="general">عمومی</MenuItem>
                      <MenuItem value="steiner">Steiner</MenuItem>
                      <MenuItem value="ricketts">Ricketts</MenuItem>
                      <MenuItem value="mcnamara">McNamara</MenuItem>
                      <MenuItem value="wits">Wits</MenuItem>
                      <MenuItem value="tweed">Tweed</MenuItem>
                      <MenuItem value="jarabak">Jarabak</MenuItem>
                      <MenuItem value="sassouni">Sassouni</MenuItem>
                      <MenuItem value="leganBurstone">Legan & Burstone</MenuItem>
                      <MenuItem value="arnettMcLaughlin">Arnett & McLaughlin</MenuItem>
                      <MenuItem value="holdaway">Holdaway</MenuItem>
                      <MenuItem value="softTissueAngular">Soft Tissue Angular</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Chart Type Selection */}
                  <FormControl fullWidth>
                    <InputLabel>نوع نمودار</InputLabel>
                    <Select
                      value={chartType}
                      label="نوع نمودار"
                      onChange={(e) => setChartType(e.target.value)}
                    >
                      <MenuItem value="radar">رادار</MenuItem>
                      <MenuItem value="area">Area</MenuItem>
                      <MenuItem value="bar-negative">Column Negative</MenuItem>
                    </Select>
                  </FormControl>

                  {/* Chart Normalization Selection - Only for area and bar-negative charts */}
                  {chartType !== 'radar' && (
                    <FormControl fullWidth>
                      <InputLabel>نوع نمایش</InputLabel>
                      <Select
                        value={chartNormalization}
                        label="نوع نمایش"
                        onChange={(e) => setChartNormalization(e.target.value)}
                      >
                        <MenuItem value="real">مقادیر واقعی</MenuItem>
                        <MenuItem value="normalized">نرمالایز شده (درصد)</MenuItem>
                      </Select>
                    </FormControl>
                  )}

                  <Suspense fallback={<LoadingFallback />}>
                    <Box
                      id="radar-chart-container"
                      sx={{
                        '& .apexcharts-radar-series .apexcharts-xaxis-label text': {
                          maxWidth: '30px !important',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          cursor: 'help',
                        },
                        '& .apexcharts-xaxis-label': {
                          maxWidth: '30px !important',
                          overflow: 'hidden',
                          cursor: 'help',
                          '& text': {
                            maxWidth: '30px !important',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis',
                            cursor: 'help',
                          },
                        },
                        // برای موبایل
                        '@media (max-width: 768px)': {
                          '& .apexcharts-xaxis-label': {
                            maxWidth: '30px !important',
                            '& text': {
                              maxWidth: '30px !important',
                            },
                          },
                        },
                      }}
                    >
                      <Chart 
                        type={chartType === 'bar-negative' ? 'bar' : chartType} 
                        series={chartSeries} 
                        options={chartOptions} 
                        height={400} 
                      />
                    </Box>
                  </Suspense>
                </Stack>
              </CardContent>
            </Card>
          </Box>
        )}

        {/* Report Display - Preload for smooth tab switching */}
        {hasTable && hasMeasuredValues && (
          <Box
            sx={{
              position: 'relative',
              visibility: deferredActiveTab === 'report' ? 'visible' : 'hidden',
              opacity: deferredActiveTab === 'report' ? 1 : 0,
              transform: deferredActiveTab === 'report' ? 'translateY(0)' : 'translateY(-10px)',
              height: deferredActiveTab === 'report' ? 'auto' : 0,
              overflow: deferredActiveTab === 'report' ? 'visible' : 'hidden',
              transition: 'opacity 0.15s ease-in-out, visibility 0.15s ease-in-out, transform 0.15s ease-in-out',
              willChange: 'opacity, visibility, transform',
              contain: 'layout style paint',
              pointerEvents: deferredActiveTab === 'report' ? 'auto' : 'none',
              backfaceVisibility: 'hidden',
              WebkitBackfaceVisibility: 'hidden',
            }}
          >
            <Card>
              <CardContent>
                <Stack spacing={3}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                    <Typography variant="h5">
                      گزارش کلی
                    </Typography>
                    {!showRAGAnalysis ? (
                      <Button
                        variant="contained"
                        startIcon={<Iconify icon="solar:graph-up-bold" />}
                        onClick={() => {
                          setShowRAGAnalysis(true);
                        }}
                      >
                        آنالیز با RAG
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        startIcon={<Iconify icon="solar:close-circle-bold" />}
                        onClick={() => {
                          setShowRAGAnalysis(false);
                        }}
                      >
                        بستن آنالیز RAG
                      </Button>
                    )}
                  </Stack>

                  {showRAGAnalysis && (() => {
                    // تبدیل allTableData به فرمت مورد نیاز RAG - شامل همه آنالیزها
                    const ragMeasurements = {};
                    const ragCephalometricTable = {};
                    
                    // اولویت: استفاده از allTableData از analysisHistory (اگر موجود باشد)
                    let allTableData = null;
                    if (selectedAnalysisIndex !== null && analysisHistory && analysisHistory.length > 0 && selectedAnalysisIndex < analysisHistory.length) {
                      const analysis = analysisHistory[selectedAnalysisIndex];
                      if (analysis?.allTableData) {
                        ({ allTableData } = analysis);
                      }
                    }
                    
                    // اگر allTableData موجود نبود، از patient.cephalometricTable استفاده کن
                    if (!allTableData && patient?.cephalometricTable) {
                      // ساخت allTableData از cephalometricTable فعلی
                      allTableData = {
                        [selectedAnalysisType]: patient.cephalometricTable,
                      };
                    }
                    
                    // ترکیب فقط 3 آنالیز: Steiner, Ricketts, Tweed
                    if (allTableData) {
                      // فقط این 3 آنالیز: steiner, ricketts, tweed
                      const requiredAnalysisTypes = ['steiner', 'ricketts', 'tweed'];
                      
                      requiredAnalysisTypes.forEach(analysisType => {
                        if (allTableData[analysisType]) {
                          Object.entries(allTableData[analysisType]).forEach(([param, data]) => {
                            if (data?.measured && data.measured !== '' && data.measured !== 'undefined' && data.measured !== 'null') {
                              const value = parseFloat(data.measured);
                              if (!isNaN(value)) {
                                // اگر پارامتر قبلاً اضافه نشده، اضافه کن
                                if (!ragMeasurements[param]) {
                                  ragMeasurements[param] = value;
                                  if (!ragCephalometricTable[analysisType]) {
                                    ragCephalometricTable[analysisType] = {};
                                  }
                                  ragCephalometricTable[analysisType][param] = data;
                                }
                              }
                            }
                          });
                        }
                      });
                    }
                    
                    // اگر هنوز چیزی نداریم، از cephalometricTable استفاده کن
                    if (Object.keys(ragMeasurements).length === 0 && patient?.cephalometricTable) {
                      Object.entries(patient.cephalometricTable).forEach(([param, data]) => {
                        if (data?.measured && data.measured !== '' && data.measured !== 'undefined' && data.measured !== 'null') {
                          const value = parseFloat(data.measured);
                          if (!isNaN(value)) {
                            ragMeasurements[param] = value;
                            if (!ragCephalometricTable[selectedAnalysisType]) {
                              ragCephalometricTable[selectedAnalysisType] = {};
                            }
                            ragCephalometricTable[selectedAnalysisType][param] = data;
                          }
                        }
                      });
                    }
                    
                    // ساخت cephalometricTable ترکیبی از همه آنالیزها
                    const combinedCephalometricTable = {};
                    Object.keys(ragCephalometricTable).forEach(analysisType => {
                      Object.entries(ragCephalometricTable[analysisType]).forEach(([param, data]) => {
                        if (!combinedCephalometricTable[param]) {
                          combinedCephalometricTable[param] = data;
                        }
                      });
                    });
                    
                    // لاگ برای دیباگ
                    console.log('🔍 [RAG Analysis] Using measurements from:', {
                      analysisTypes: Object.keys(ragCephalometricTable),
                      totalParameters: Object.keys(ragMeasurements).length,
                      parametersPerType: Object.keys(ragCephalometricTable).map(type => ({
                        type,
                        count: Object.keys(ragCephalometricTable[type] || {}).length,
                      })),
                    });

                    // دریافت سن و جنسیت بیمار
                    const patientAge = patient?.age ? parseInt(String(patient.age), 10) : 0;
                    const patientGender = patient?.gender === 'male' || patient?.gender === 'female' 
                      ? patient.gender 
                      : 'male'; // default

                    return (
                      <ClinicalRAGAnalysis
                        patientData={{
                          age: patientAge || 0,
                          gender: patientGender,
                          cephalometricMeasurements: ragMeasurements,
                          medicalHistory: patient?.medicalHistory,
                          previousTreatments: patient?.previousTreatments,
                          cephalometricTable: combinedCephalometricTable, // ارسال table ترکیبی از همه آنالیزها
                        }}
                        onAnalysisComplete={(analysis) => {
                          console.log('RAG Analysis completed:', analysis);
                          console.log('Parameters used:', Object.keys(ragMeasurements).length, 'from', Object.keys(ragCephalometricTable).length, 'analysis types');
                        }}
                      />
                    );
                  })()}

                  {!showRAGAnalysis && (() => {
                    // Convert cephalometricTable to CephalometricMeasurements format
                    const measurements = {};
                    if (patient?.cephalometricTable) {
                      Object.entries(patient.cephalometricTable).forEach(([param, data]) => {
                        if (data?.measured && data.measured !== '' && data.measured !== 'undefined' && data.measured !== 'null') {
                          const value = parseFloat(data.measured);
                          if (!isNaN(value)) {
                            // Use original parameter name first
                            measurements[param] = value;
                            
                            // Also try normalized versions for common parameter formats
                            const normalized = param
                              .replace(/\s+/g, '-')
                              .replace(/\//g, '-')
                              .replace(/--+/g, '-');
                            if (normalized !== param) {
                              measurements[normalized] = value;
                            }
                            
                            // Try without spaces/dashes for some parameters
                            const noSpaces = param.replace(/\s+/g, '').replace(/-/g, '');
                            if (noSpaces !== param && noSpaces !== normalized) {
                              measurements[noSpaces] = value;
                            }
                          }
                        }
                      });
                    }

                    // Generate comprehensive analysis
                    const analysis = generateComprehensiveAnalysis(measurements);
                    const formattedReport = formatAnalysisForDisplay(analysis);

                    return (
                      <Stack spacing={3}>
                        {/* Diagnosis */}
                        <Box>
                          <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                            تشخیص
                          </Typography>
                          <Typography variant="body1" sx={{ whiteSpace: 'pre-line' }}>
                            {analysis.diagnosis}
                          </Typography>
                        </Box>

                        {/* Issues */}
                        {analysis.issues.length > 0 && (
                          <Box>
                            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                              مشکلات شناسایی شده
                            </Typography>
                            <Stack spacing={1}>
                              {analysis.issues.map((issue, index) => (
                                <Box key={index} sx={{ p: 2, bgcolor: 'background.neutral', borderRadius: 1 }}>
                                  <Stack direction="row" spacing={1} alignItems="center" sx={{ mb: 0.5 }}>
                                    <Label
                                      variant="soft"
                                      color={
                                        issue.severity === 'severe' ? 'error' :
                                        issue.severity === 'moderate' ? 'warning' :
                                        'info'
                                      }
                                      sx={{ fontWeight: 'normal' }}
                                    >
                                      {issue.severity === 'severe' ? 'شدید' :
                                       issue.severity === 'moderate' ? 'متوسط' :
                                       'خفیف'}
                                    </Label>
                                    <Label
                                      variant="soft"
                                      color={
                                        issue.type === 'skeletal' ? 'primary' :
                                        issue.type === 'dental' ? 'secondary' :
                                        issue.type === 'soft_tissue' ? 'info' :
                                        'default'
                                      }
                                      sx={{ fontWeight: 'normal' }}
                                    >
                                      {issue.type === 'skeletal' ? 'اسکلتی' :
                                       issue.type === 'dental' ? 'دندانی' :
                                       issue.type === 'soft_tissue' ? 'بافت نرم' :
                                       'عملکردی'}
                                    </Label>
                                  </Stack>
                                  <Typography variant="body2">
                                    {issue.description}
                                    {issue.parameter && issue.value !== undefined && (
                                      <span style={{ color: 'text.secondary', marginRight: '8px' }}>
                                        {' '}({issue.parameter}: {issue.value.toFixed(1)}°)
                                      </span>
                                    )}
                                  </Typography>
                                </Box>
                              ))}
                            </Stack>
                          </Box>
                        )}

                        {/* Treatment Plan */}
                        {analysis.treatmentPlan.length > 0 && (
                          <Box>
                            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                              طرح درمان
                            </Typography>
                            <Stack spacing={2}>
                              {analysis.treatmentPlan.map((phase, index) => (
                                <Box key={index} sx={{ p: 2, border: '1px solid', borderColor: 'divider', borderRadius: 1 }}>
                                  <Typography variant="subtitle1" sx={{ mb: 1, fontWeight: 'bold' }}>
                                    {phase.phase}
                                  </Typography>
                                  <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                                    مدت زمان: {phase.duration}
                                  </Typography>
                                  {phase.goals.length > 0 && (
                                    <Box sx={{ mb: 1 }}>
                                      <Typography variant="body2" sx={{ fontWeight: 'medium', mb: 0.5 }}>
                                        اهداف:
                                      </Typography>
                                      <Typography variant="body2" color="text.secondary">
                                        {phase.goals.join('، ')}
                                      </Typography>
                                    </Box>
                                  )}
                                  {phase.procedures.length > 0 && (
                                    <Box>
                                      <Typography variant="body2" sx={{ fontWeight: 'medium', mb: 0.5 }}>
                                        روش‌های درمانی:
                                      </Typography>
                                      <Stack component="ul" spacing={0.5} sx={{ pl: 2, m: 0 }}>
                                        {phase.procedures.map((procedure, procIndex) => (
                                          <Typography key={procIndex} component="li" variant="body2" color="text.secondary">
                                            {procedure}
                                          </Typography>
                                        ))}
                                      </Stack>
                                    </Box>
                                  )}
                                </Box>
                              ))}
                            </Stack>
                          </Box>
                        )}

                        {/* Prognosis */}
                        {analysis.prognosis && (
                          <Box>
                            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                              پیش‌آگهی
                            </Typography>
                            <Typography variant="body1" sx={{ whiteSpace: 'pre-line', p: 2, bgcolor: 'background.neutral', borderRadius: 1 }}>
                              {analysis.prognosis}
                            </Typography>
                          </Box>
                        )}

                        {/* Recommendations */}
                        {analysis.recommendations.length > 0 && (
                          <Box>
                            <Typography variant="h6" sx={{ mb: 1, color: 'primary.main' }}>
                              توصیه‌ها
                            </Typography>
                            <Stack component="ul" spacing={1} sx={{ pl: 2, m: 0 }}>
                              {analysis.recommendations.map((rec, index) => (
                                <Typography key={index} component="li" variant="body2" color="text.secondary">
                                  {rec}
                                </Typography>
                              ))}
                            </Stack>
                          </Box>
                        )}
                      </Stack>
                    );
                  })()}
                </Stack>
              </CardContent>
            </Card>
          </Box>
        )}
      </Stack>
    </DashboardContent>
  );
}
