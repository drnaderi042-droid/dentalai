import { memo, useMemo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import Divider from '@mui/material/Divider';
import MenuItem from '@mui/material/MenuItem';
import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import LinearProgress from '@mui/material/LinearProgress';

import axios, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';
import { Chart, useChart } from 'src/components/chart';

import { useAuthContext } from 'src/auth/hooks';

import { PeriodontalVisualization } from '../../components/periodontal-visualization';

// ----------------------------------------------------------------------

function AnalysisTabComponent({ patient, onUpdate }) {
  const theme = useTheme();
  const { user } = useAuthContext();
  const [chartData, setChartData] = useState(null);
  const [charts, setCharts] = useState([]);
  const [selectedChartId, setSelectedChartId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chartToDelete, setChartToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Helper function to load chart data
  const loadChartData = useCallback((chart) => {
    if (!chart) {
      setChartData(null);
      return;
    }

    let teethData = chart.teeth;
    
    // Parse teeth if it's a string
    if (typeof teethData === 'string') {
      try {
        teethData = JSON.parse(teethData);
      } catch (parseError) {
        console.error('Error parsing teeth data:', parseError);
        teethData = {};
      }
    }
    
    // Ensure teeth is an object
    if (!teethData || typeof teethData !== 'object') {
      teethData = {};
    }
    
    setChartData({
      ...chart,
      teeth: teethData,
    });
  }, []);

  // Fetch all charts
  useEffect(() => {
    const fetchCharts = async () => {
      try {
        setLoading(true);
        const response = await axios.get(`${endpoints.patients}/${patient.id}/periodontal-charts`, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        const fetchedCharts = response.data.charts || [];
        setCharts(fetchedCharts);

        // Load the latest chart by default
        if (fetchedCharts.length > 0) {
          const latestChart = fetchedCharts[0];
          setSelectedChartId(latestChart.id);
          loadChartData(latestChart);
        } else {
          setChartData(null);
          setSelectedChartId(null);
        }
      } catch (error) {
        console.error('Error fetching charts:', error);
        setChartData(null);
        setCharts([]);
        setSelectedChartId(null);
      } finally {
        setLoading(false);
      }
    };

    if (user && patient) {
      fetchCharts();
    } else {
      setChartData(null);
      setCharts([]);
      setSelectedChartId(null);
      setLoading(false);
    }
  }, [user, patient, loadChartData]);

  // Handle chart selection change
  const handleChartChange = (event) => {
    const chartId = event.target.value;
    setSelectedChartId(chartId);

    if (chartId && chartId !== '') {
      const selectedChart = charts.find((c) => c.id === chartId);
      if (selectedChart) {
        loadChartData(selectedChart);
      }
    } else {
      setChartData(null);
    }
  };

  // Handle delete chart
  const handleDeleteChart = (chart) => {
    setChartToDelete(chart);
    setDeleteDialogOpen(true);
  };

  const confirmDeleteChart = async () => {
    if (!chartToDelete) return;

    try {
      setDeleting(true);
      await axios.delete(
        `${endpoints.patients}/${patient.id}/periodontal-charts/${chartToDelete.id}`,
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        }
      );

      toast.success('چارت با موفقیت حذف شد');

      // Refresh charts list
      const response = await axios.get(`${endpoints.patients}/${patient.id}/periodontal-charts`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      const fetchedCharts = response.data.charts || [];
      setCharts(fetchedCharts);

      // If deleted chart was selected, select the latest one or clear selection
      if (selectedChartId === chartToDelete.id) {
        if (fetchedCharts.length > 0) {
          const latestChart = fetchedCharts[0];
          setSelectedChartId(latestChart.id);
          loadChartData(latestChart);
        } else {
          setSelectedChartId(null);
          setChartData(null);
        }
      }

      setDeleteDialogOpen(false);
      setChartToDelete(null);
    } catch (error) {
      console.error('Error deleting chart:', error);
      toast.error('خطا در حذف چارت');
    } finally {
      setDeleting(false);
    }
  };

  // Calculate analysis metrics
  const analysis = useMemo(() => {
    if (!chartData || !chartData.teeth) {
      console.log('Analysis: No chart data or teeth found', { chartData });
      return null;
    }

    const { teeth } = chartData;
    
    // Debug: Log teeth data structure
    if (Object.keys(teeth).length === 0) {
      console.log('Analysis: No teeth found in chart data');
      return null;
    }
    const results = {
      bopPercentage: 0,
      avgPocketDepth: 0,
      avgCAL: 0,
      pocketDepthDistribution: { healthy: 0, mild: 0, moderate: 0, severe: 0 },
      diseaseExtent: 'Localized',
      diseaseSeverity: 'Stage I',
      totalSites: 0, // Total measurement sites (for PD and CAL)
      bopSites: 0, // Number of surfaces with BOP
      totalBopSurfaces: 0, // Total number of surfaces (2 per active tooth)
      affectedTeeth: 0,
      totalTeeth: 0,
    };

    let totalPD = 0;
    let totalCAL = 0;
    let totalSites = 0; // Total measurement sites (for PD and CAL)
    let bopSurfacesCount = 0; // Count of surfaces with BOP (each tooth has 2 surfaces: facial and lingual)
    let totalBopSurfaces = 0; // Total number of surfaces (2 per tooth, excluding missing teeth)
    let teethWithCAL = 0;
    let activeTeeth = 0;

    Object.keys(teeth).forEach((toothNum) => {
      const tooth = teeth[toothNum];
      
      // Skip missing teeth
      if (!tooth || tooth.missing) return;
      
      // Ensure facial and lingual surfaces exist
      const facial = tooth.facial || {};
      const lingual = tooth.lingual || {};
      const facialPD = facial.pocketDepth || [];
      const facialGM = facial.gingivalMargin || [];
      const lingualPD = lingual.pocketDepth || [];
      const lingualGM = lingual.gingivalMargin || [];
      const facialBleeding = facial.bleeding || [];
      const lingualBleeding = lingual.bleeding || [];
      
      activeTeeth += 1;
      // Each tooth has 2 surfaces (facial and lingual) for BOP calculation
      totalBopSurfaces += 2;

      let maxCAL = 0;

      // Check if facial surface has BOP (if any bleeding point is true)
      const facialHasBOP = facialBleeding.some((b) => b === true);
      if (facialHasBOP) {
        bopSurfacesCount += 1;
      }

      // Check if lingual surface has BOP (if any bleeding point is true)
      const lingualHasBOP = lingualBleeding.some((b) => b === true);
      if (lingualHasBOP) {
        bopSurfacesCount += 1;
      }

      // Process facial surface for PD and CAL
      facialPD.forEach((pd, i) => {
        // Skip if pocket depth is null or undefined
        if (pd === null || pd === undefined) return;
        
        const gm = facialGM[i] !== null && facialGM[i] !== undefined ? facialGM[i] : 0;
        const cal = pd + gm;
        
        // Only count valid measurements
        totalPD += pd;
        totalCAL += cal;
        totalSites += 1;
        
        maxCAL = Math.max(maxCAL, cal);

        // Pocket depth distribution
        if (pd <= 3) results.pocketDepthDistribution.healthy += 1;
        else if (pd <= 5) results.pocketDepthDistribution.mild += 1;
        else if (pd === 6) results.pocketDepthDistribution.moderate += 1;
        else if (pd >= 7) results.pocketDepthDistribution.severe += 1;
      });

      // Process lingual surface for PD and CAL
      lingualPD.forEach((pd, i) => {
        // Skip if pocket depth is null or undefined
        if (pd === null || pd === undefined) return;
        
        const gm = lingualGM[i] !== null && lingualGM[i] !== undefined ? lingualGM[i] : 0;
        const cal = pd + gm;
        
        // Only count valid measurements
        totalPD += pd;
        totalCAL += cal;
        totalSites += 1;
        
        maxCAL = Math.max(maxCAL, cal);

        // Pocket depth distribution
        if (pd <= 3) results.pocketDepthDistribution.healthy += 1;
        else if (pd <= 5) results.pocketDepthDistribution.mild += 1;
        else if (pd === 6) results.pocketDepthDistribution.moderate += 1;
        else if (pd >= 7) results.pocketDepthDistribution.severe += 1;
      });

      // Count affected teeth (CAL >= 3mm)
      if (maxCAL >= 3) {
        teethWithCAL += 1;
      }
    });

    results.totalSites = totalSites;
    results.bopSites = bopSurfacesCount; // Number of surfaces with BOP
    results.totalBopSurfaces = totalBopSurfaces; // Total number of surfaces (2 per active tooth)
    results.totalTeeth = activeTeeth;
    results.affectedTeeth = teethWithCAL;

    // Calculate percentages and averages
    if (totalSites > 0) {
      results.avgPocketDepth = totalPD / totalSites;
      results.avgCAL = totalCAL / totalSites;
    }

    // Calculate BOP percentage based on surfaces (not sites)
    if (totalBopSurfaces > 0) {
      results.bopPercentage = (bopSurfacesCount / totalBopSurfaces) * 100;
    }

    // Disease extent
    if (activeTeeth > 0) {
      const affectedPercentage = (teethWithCAL / activeTeeth) * 100;
      results.diseaseExtent = affectedPercentage < 30 ? 'Localized' : 'Generalized';
    }

    // Disease severity (based on max CAL)
    const allCALValues = Object.values(teeth)
      .filter((t) => t && !t.missing)
      .flatMap((t) => {
        const facial = t.facial || {};
        const lingual = t.lingual || {};
        const facialPD = facial.pocketDepth || [];
        const facialGM = facial.gingivalMargin || [];
        const lingualPD = lingual.pocketDepth || [];
        const lingualGM = lingual.gingivalMargin || [];
        
        const calValues = [];
        
        // Process facial CAL values
        facialPD.forEach((pd, i) => {
          if (pd !== null && pd !== undefined) {
            const gm = facialGM[i] !== null && facialGM[i] !== undefined ? facialGM[i] : 0;
            calValues.push(pd + gm);
          }
        });
        
        // Process lingual CAL values
        lingualPD.forEach((pd, i) => {
          if (pd !== null && pd !== undefined) {
            const gm = lingualGM[i] !== null && lingualGM[i] !== undefined ? lingualGM[i] : 0;
            calValues.push(pd + gm);
          }
        });
        
        return calValues;
      });

    const maxCAL = allCALValues.length > 0 ? Math.max(...allCALValues) : 0;

    if (maxCAL <= 2) results.diseaseSeverity = 'Stage I - Mild';
    else if (maxCAL <= 4) results.diseaseSeverity = 'Stage II - Moderate';
    else if (maxCAL >= 5 && activeTeeth === 32) results.diseaseSeverity = 'Stage III - Severe';
    else if (maxCAL >= 5) results.diseaseSeverity = 'Stage IV - Very Severe';
    else results.diseaseSeverity = 'Stage I - Mild';

    return results;
  }, [chartData]);

  // Generate treatment plan
  const treatmentPlan = useMemo(() => {
    if (!analysis || !chartData) return null;

    const plan = {
      phase1: { items: [], description: '', duration: '', priority: 'high' },
      phase2: { items: [], description: '', duration: '', priority: 'medium' },
      phase3: { items: [], description: '', duration: '', priority: 'low' },
      phase4: { items: [], description: '', duration: '', priority: 'high' },
    };

    // Phase 1: Initial Therapy (Always)
    plan.phase1.description = 'درمان اولیه شامل کنترل پلاک، اسکلینگ و روت پلنینگ برای حذف عوامل بیماری‌زا';
    plan.phase1.duration = '4-6 هفته';
    plan.phase1.items.push({
      title: 'آموزش بهداشت دهان و دندان',
      description: 'آموزش تکنیک‌های صحیح مسواک زدن، استفاده از نخ دندان و دهانشویه',
      icon: 'solar:book-bookmark-bold',
    });
    plan.phase1.items.push({
      title: 'اسکلینگ و روت پلنینگ (SRP)',
      description: 'حذف پلاک و جرم از سطح دندان و ریشه در تمام دهان',
      icon: 'solar:scissors-bold',
    });

    if (analysis.bopPercentage > 30) {
      plan.phase1.items.push({
        title: 'کنترل شدید پلاک',
        description: 'برنامه کنترل پلاک فشرده با ویزیت‌های مکرر',
        icon: 'solar:shield-check-bold',
      });
      plan.phase1.items.push({
        title: 'دهانشویه ضدمیکروبی',
        description: 'کلرهگزیدین 0.12% دو بار در روز به مدت 2-4 هفته',
        icon: 'solar:droplet-bold',
      });
    }

    if (analysis.avgPocketDepth > 5) {
      plan.phase1.items.push({
        title: 'آنتی‌بیوتیک موضعی',
        description: 'تجویز آنتی‌بیوتیک موضعی در پاکت‌های عمیق (در صورت نیاز)',
        icon: 'solar:pill-bold',
      });
    }

    // Phase 2: Re-evaluation & Surgery
    plan.phase2.description = 'ارزیابی پاسخ به درمان اولیه و در صورت نیاز، انجام درمان‌های جراحی';
    plan.phase2.duration = '4-8 هفته پس از فاز 1';
    
    if (analysis.avgPocketDepth > 5 || analysis.avgCAL > 4) {
      plan.phase2.items.push({
        title: 'ارزیابی مجدد',
        description: 'بررسی بهبودی BOP، CAL و عمق پاکت 4-6 هفته پس از SRP',
        icon: 'solar:clipboard-check-bold',
      });
      
      if (analysis.avgPocketDepth > 6) {
        plan.phase2.items.push({
          title: 'جراحی فلپ',
          description: 'جراحی فلپ برای دسترسی بهتر به ریشه و حذف پاکت‌های عمیق',
          icon: 'solar:scalpel-bold',
        });
        plan.phase2.items.push({
          title: 'جراحی استخوان',
          description: 'جراحی استخوان برای اصلاح نقایص استخوانی',
          icon: 'solar:bone-bold',
        });
      }

      plan.phase2.items.push({
        title: 'پیوند استخوان',
        description: 'پیوند استخوان و بازسازی بافت هدایت شده (GBR) در صورت نیاز',
        icon: 'solar:heart-pulse-bold',
      });
    }

    // Phase 3: Restorative
    plan.phase3.description = 'درمان‌های بازسازی‌ای و پروتزی برای بهبود عملکرد و زیبایی';
    plan.phase3.duration = 'پس از بهبودی کامل فاز 2';
    
    const { teeth } = chartData;
    const teethWithMobility = Object.values(teeth).filter(
      (t) => !t.missing && (t.facial?.mobility > 1 || t.lingual?.mobility > 1)
    ).length;

    if (teethWithMobility > 0) {
      plan.phase3.items.push({
        title: `اسپلینت کردن ${teethWithMobility} دندان`,
        description: `اسپلینت کردن دندان‌های با تحرک بالا برای کاهش تحرک و بهبود ثبات`,
        icon: 'solar:link-bold',
      });
    }

    const missingTeeth = Object.values(teeth).filter((t) => t.missing).length;
    if (missingTeeth > 0) {
      plan.phase3.items.push({
        title: `بازسازی پروتزی ${missingTeeth} دندان`,
        description: `برنامه‌ریزی برای جایگزینی ${missingTeeth} دندان از دست رفته با ایمپلنت یا پروتز`,
        icon: 'solar:tooth-bold',
      });
    }

    if (plan.phase3.items.length === 0) {
      plan.phase3.items.push({
        title: 'نیازی به درمان بازسازی‌ای نیست',
        description: 'وضعیت فعلی نیاز به درمان بازسازی‌ای ندارد',
        icon: 'solar:check-circle-bold',
      });
    }

    // Phase 4: Maintenance
    plan.phase4.description = 'برنامه نگهداری طولانی‌مدت برای حفظ سلامت پریودونتال';
    const maintenanceInterval = analysis.bopPercentage < 10 ? '6 ماه' : '3 ماه';
    plan.phase4.duration = `هر ${maintenanceInterval}`;
    
    plan.phase4.items.push({
      title: `بازدید دوره‌ای هر ${maintenanceInterval}`,
      description: `ویزیت منظم هر ${maintenanceInterval} برای مانیتورینگ وضعیت پریودونتال`,
      icon: 'solar:calendar-mark-bold',
    });
    plan.phase4.items.push({
      title: 'مانیتورینگ شاخص‌ها',
      description: 'بررسی BOP، CAL، عمق پاکت و وضعیت کلی لثه',
      icon: 'solar:graph-up-bold',
    });
    plan.phase4.items.push({
      title: 'اسکلینگ و پولیش',
      description: 'اسکلینگ و پولیش حرفه‌ای در صورت نیاز',
      icon: 'solar:brush-bold',
    });
    plan.phase4.items.push({
      title: 'بررسی و اصلاح تکنیک‌های بهداشتی',
      description: 'بررسی تکنیک‌های بهداشت دهان بیمار و ارائه راهنمایی‌های لازم',
      icon: 'solar:book-bookmark-bold',
    });

    // Medical history considerations
    const medicalHistory =
      typeof patient.medicalHistory === 'string'
        ? JSON.parse(patient.medicalHistory)
        : patient.medicalHistory || {};

    if (medicalHistory.diabetes) {
      plan.phase1.items.unshift({
        title: 'هماهنگی با پزشک برای کنترل قند خون',
        description: 'هماهنگی با پزشک معالج برای کنترل بهینه قند خون (HbA1c < 7%)',
        icon: 'solar:heart-pulse-bold',
        warning: true,
      });
    }

    if (medicalHistory.smoking) {
      plan.phase1.items.unshift({
        title: 'مشاوره ترک سیگار',
        description: 'مشاوره و راهنمایی برای ترک سیگار (بسیار مهم برای موفقیت درمان)',
        icon: 'solar:smoking-bold',
        warning: true,
      });
    }

    if (medicalHistory.pregnancy) {
      plan.phase1.items.push({
        title: 'توجه ویژه به ژنژیویت بارداری',
        description: 'درمان محافظه‌کارانه با توجه به وضعیت بارداری',
        icon: 'solar:heart-bold',
        warning: true,
      });
    }

    return plan;
  }, [analysis, chartData, patient]);

  // Chart options for BOP distribution
  const bopChartOptions = useChart({
    chart: { type: 'bar' },
    xaxis: {
      categories: ['فک بالا راست', 'فک بالا چپ', 'فک پایین چپ', 'فک پایین راست'],
    },
    colors: [theme.palette.error.main],
    plotOptions: {
      bar: {
        borderRadius: 8,
        distributed: false,
      },
    },
  });

  const bopChartSeries = useMemo(() => {
    if (!chartData) return [];

    const quadrants = {
      q1: { bop: 0, total: 0 }, // Upper right (1-8)
      q2: { bop: 0, total: 0 }, // Upper left (9-16)
      q3: { bop: 0, total: 0 }, // Lower left (17-24)
      q4: { bop: 0, total: 0 }, // Lower right (25-32)
    };

    Object.keys(chartData.teeth).forEach((toothNum) => {
      const tooth = chartData.teeth[toothNum];
      const num = parseInt(toothNum, 10);

      if (!tooth || tooth.missing) return;

      let quadrant;
      if (num >= 1 && num <= 8) quadrant = 'q1';
      else if (num >= 9 && num <= 16) quadrant = 'q2';
      else if (num >= 17 && num <= 24) quadrant = 'q3';
      else quadrant = 'q4';

      // Count BOP surfaces - each tooth has 2 surfaces (facial and lingual)
      const facialBleeding = (tooth.facial && tooth.facial.bleeding) || [];
      const lingualBleeding = (tooth.lingual && tooth.lingual.bleeding) || [];
      
      // Check if surface has BOP (if any bleeding point is true)
      const facialHasBOP = facialBleeding.some((b) => b === true);
      const lingualHasBOP = lingualBleeding.some((b) => b === true);

      if (facialHasBOP) quadrants[quadrant].bop += 1;
      if (lingualHasBOP) quadrants[quadrant].bop += 1;
      quadrants[quadrant].total += 2; // 2 surfaces per tooth (facial + lingual)
    });

    return [
      {
        name: 'BOP %',
        data: [
          quadrants.q1.total > 0 ? (quadrants.q1.bop / quadrants.q1.total) * 100 : 0,
          quadrants.q2.total > 0 ? (quadrants.q2.bop / quadrants.q2.total) * 100 : 0,
          quadrants.q3.total > 0 ? (quadrants.q3.bop / quadrants.q3.total) * 100 : 0,
          quadrants.q4.total > 0 ? (quadrants.q4.bop / quadrants.q4.total) * 100 : 0,
        ],
      },
    ];
  }, [chartData]);

  // Calculate actual values for tooltip (before chart options)
  const polygonChartActualValues = useMemo(() => {
    if (!analysis || !chartData || !patient) return null;

    const missingTeeth = Object.values(chartData.teeth).filter((t) => t.missing).length;
    const medicalHistory =
      typeof patient.medicalHistory === 'string'
        ? JSON.parse(patient.medicalHistory)
        : patient.medicalHistory || {};

    let systemicDiseasesCount = 0;
    if (medicalHistory.diabetes) systemicDiseasesCount += 1;
    if (medicalHistory.hypertension) systemicDiseasesCount += 1;
    if (medicalHistory.heartDisease) systemicDiseasesCount += 1;
    if (medicalHistory.kidneyDisease) systemicDiseasesCount += 1;
    if (medicalHistory.liverDisease) systemicDiseasesCount += 1;
    if (medicalHistory.autoimmune) systemicDiseasesCount += 1;
    if (medicalHistory.cancer) systemicDiseasesCount += 1;
    if (medicalHistory.otherDiseases) systemicDiseasesCount += 1;

    const smokingPerDay = medicalHistory.smoking ? (medicalHistory.smokingCigarettesPerDay || 0) : 0;

    return {
      missingTeeth,
      bop: analysis.bopPercentage,
      pd: analysis.avgPocketDepth,
      cal: analysis.avgCAL,
      systemicDiseases: systemicDiseasesCount,
      smoking: smokingPerDay,
    };
  }, [analysis, chartData, patient]);

  // Polygon (Radar) chart for comprehensive analysis
  const polygonChartOptionsConfig = useMemo(
    () => ({
      xaxis: {
        categories: [
          'دندان از دست رفته',
          'میانگین BOP (%)',
          'میانگین PD (mm)',
          'میانگین CAL (mm)',
          'بیماری‌های سیستمیک',
          'مصرف سیگار (نخ/روز)',
        ],
      },
      yaxis: {
        show: false,
        max: 100,
        min: 0,
      },
      plotOptions: {
        radar: {
          polygons: {
            strokeColors: theme.palette.divider,
            strokeWidth: 1,
            fill: { colors: ['transparent'] },
          },
        },
      },
      markers: {
        size: 4,
        strokeColors: [theme.palette.primary.main],
        strokeWidth: 2,
      },
      stroke: {
        width: 2,
        curve: 'smooth',
      },
      fill: {
        opacity: 0.2,
      },
      colors: [theme.palette.primary.main],
      tooltip: {
        y: {
          formatter: (value, opts) => {
            const {dataPointIndex} = opts;
            const actualValues = polygonChartActualValues;

            if (!actualValues) {
              // Fallback if actual values not available
              if (dataPointIndex === 0) {
                return `${Math.round((value / 100) * 32)} دندان`;
              }
              return `${value.toFixed(1)}`;
            }

            // For missing teeth, show actual number
            if (dataPointIndex === 0) {
              return `${actualValues.missingTeeth} دندان`;
            }

            // For BOP, show percentage
            if (dataPointIndex === 1) {
              return `${actualValues.bop.toFixed(1)}%`;
            }

            // For PD and CAL, show in mm
            if (dataPointIndex === 2) {
              return `${actualValues.pd.toFixed(1)} mm`;
            }
            if (dataPointIndex === 3) {
              return `${actualValues.cal.toFixed(1)} mm`;
            }

            // For systemic diseases, show count
            if (dataPointIndex === 4) {
              return `${actualValues.systemicDiseases} بیماری`;
            }

            // For smoking, show cigarettes per day
            if (dataPointIndex === 5) {
              return `${actualValues.smoking} نخ/روز`;
            }

            return `${value.toFixed(1)}`;
          },
        },
      },
      states: {
        hover: {
          filter: { type: 'lighten', value: 0.1 },
        },
        active: {
          filter: { type: 'none' },
        },
      },
    }),
    [theme.palette.divider, theme.palette.primary.main, polygonChartActualValues]
  );

  const polygonChartOptions = useChart(polygonChartOptionsConfig);

  const polygonChartSeries = useMemo(() => {
    if (!analysis || !chartData || !patient) return [];

    // Calculate missing teeth count
    const missingTeeth = Object.values(chartData.teeth).filter((t) => t.missing).length;

    // Get medical history
    const medicalHistory =
      typeof patient.medicalHistory === 'string'
        ? JSON.parse(patient.medicalHistory)
        : patient.medicalHistory || {};

    // Count systemic diseases
    let systemicDiseasesCount = 0;
    if (medicalHistory.diabetes) systemicDiseasesCount += 1;
    if (medicalHistory.hypertension) systemicDiseasesCount += 1;
    if (medicalHistory.heartDisease) systemicDiseasesCount += 1;
    if (medicalHistory.kidneyDisease) systemicDiseasesCount += 1;
    if (medicalHistory.liverDisease) systemicDiseasesCount += 1;
    if (medicalHistory.autoimmune) systemicDiseasesCount += 1;
    if (medicalHistory.cancer) systemicDiseasesCount += 1;
    if (medicalHistory.otherDiseases) systemicDiseasesCount += 1;

    // Get smoking (cigarettes per day)
    const smokingPerDay = medicalHistory.smoking ? (medicalHistory.smokingCigarettesPerDay || 0) : 0;

    // Normalize values for radar chart - each axis normalized independently
    // Each axis uses a reasonable maximum value that makes small values visible
    // The max values are set lower than theoretical maximums to ensure visibility
    
    // Define reasonable normalization max values for each category
    // These are lower than theoretical maximums to ensure small values are visible
    const NORMALIZATION_MAX = {
      missingTeeth: 10,      // Normalize against 10 teeth (instead of 32) - makes 5 teeth show as 50%
      bop: 100,              // 100% BOP (already a percentage)
      pd: 6,                 // Normalize against 6mm (instead of 10) - makes smaller values more visible
      cal: 8,                // Normalize against 8mm (instead of 12) - makes smaller values more visible
      systemicDiseases: 5,   // Normalize against 5 diseases (instead of 8) - makes smaller counts visible
      smoking: 30,           // Normalize against 30 cigarettes/day (instead of 40) - makes smaller values visible
    };

    // Normalize each value independently (0-100 scale)
    // Each axis uses its own normalization max to ensure proportional display
    // If actual value exceeds the normalization max, it will be capped at 100%
    const normalizedMissingTeeth = missingTeeth > 0 
      ? Math.min((missingTeeth / NORMALIZATION_MAX.missingTeeth) * 100, 100)
      : 0;

    const normalizedBOP = Math.min((analysis.bopPercentage / NORMALIZATION_MAX.bop) * 100, 100);

    const normalizedPD = analysis.avgPocketDepth > 0
      ? Math.min((analysis.avgPocketDepth / NORMALIZATION_MAX.pd) * 100, 100)
      : 0;

    const normalizedCAL = analysis.avgCAL > 0
      ? Math.min((analysis.avgCAL / NORMALIZATION_MAX.cal) * 100, 100)
      : 0;

    const normalizedSystemicDiseases = systemicDiseasesCount > 0
      ? Math.min((systemicDiseasesCount / NORMALIZATION_MAX.systemicDiseases) * 100, 100)
      : 0;

    const normalizedSmoking = smokingPerDay > 0
      ? Math.min((smokingPerDay / NORMALIZATION_MAX.smoking) * 100, 100)
      : 0;

    return [
      {
        name: 'وضعیت پریودونتال',
        data: [
          normalizedMissingTeeth, // Normalized for chart display (0-100), actual number shown in tooltip
          normalizedBOP,
          normalizedPD,
          normalizedCAL,
          normalizedSystemicDiseases,
          normalizedSmoking,
        ],
      },
    ];
  }, [analysis, chartData, patient]);

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography>در حال بارگیری آنالیز...</Typography>
      </Box>
    );
  }

  if (!chartData || !analysis) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography variant="h6" color="text.secondary">
          چارت پریودونتال ثبت نشده است
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          لطفاً ابتدا به تب &quot;چارت پریودونتال&quot; بروید و اطلاعات را ثبت کنید
        </Typography>
      </Box>
    );
  }

  return (
    <Stack spacing={3}>
      {/* Chart Selection */}
      {charts.length > 0 && (
        <Box
          sx={{
            p: 2,
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 1,
            bgcolor: 'background.paper',
          }}
        >
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControl fullWidth size="small">
              <InputLabel>انتخاب چارت برای آنالیز</InputLabel>
              <Select
                value={selectedChartId || ''}
                onChange={handleChartChange}
                label="انتخاب چارت برای آنالیز"
              >
                {charts.map((chart) => (
                  <MenuItem key={chart.id} value={chart.id}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, width: '100%' }}>
                      <Iconify icon="solar:calendar-bold" width={18} />
                      <Box>
                        <Typography variant="body2">
                          {new Date(chart.date).toLocaleDateString('fa-IR', {
                            year: 'numeric',
                            month: 'long',
                            day: 'numeric',
                          })}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(chart.date).toLocaleTimeString('fa-IR', {
                            hour: '2-digit',
                            minute: '2-digit',
                          })}
                        </Typography>
                      </Box>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {selectedChartId && (
              <IconButton
                color="error"
                size="small"
                onClick={() => {
                  const chart = charts.find((c) => c.id === selectedChartId);
                  if (chart) {
                    handleDeleteChart(chart);
                  }
                }}
              >
                <Iconify icon="solar:trash-bin-trash-bold" width={20} />
              </IconButton>
            )}
          </Stack>
        </Box>
      )}

      {/* Delete Confirmation Dialog */}
      <Dialog open={deleteDialogOpen} onClose={() => !deleting && setDeleteDialogOpen(false)}>
        <DialogTitle>حذف چارت پریودونتال</DialogTitle>
        <DialogContent>
          <Typography>
            آیا از حذف چارت تاریخ{' '}
            {chartToDelete &&
              new Date(chartToDelete.date).toLocaleDateString('fa-IR', {
                year: 'numeric',
                month: 'long',
                day: 'numeric',
              })}{' '}
            مطمئن هستید؟ این عمل غیرقابل بازگشت است.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)} color="inherit" disabled={deleting}>
            انصراف
          </Button>
          <Button
            onClick={confirmDeleteChart}
            color="error"
            variant="contained"
            disabled={deleting}
            startIcon={deleting ? <Iconify icon="eva:loader-fill" /> : <Iconify icon="solar:trash-bin-trash-bold" />}
          >
            {deleting ? 'در حال حذف...' : 'حذف'}
          </Button>
        </DialogActions>
      </Dialog>

      {/* Key Metrics */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Stack spacing={1.5}>
              <Typography variant="body2" color="text.secondary" fontWeight={500}>
                BOP درصد
              </Typography>
              <Typography variant="h4" fontWeight={600}>
                {analysis.bopPercentage.toFixed(1)}%
              </Typography>
              <LinearProgress
                variant="determinate"
                value={Math.min(analysis.bopPercentage, 100)}
                color={analysis.bopPercentage > 30 ? 'error' : 'success'}
                sx={{ height: 6, borderRadius: 1 }}
              />
              <Typography variant="caption" color="text.secondary">
                {analysis.bopSites} از {analysis.totalBopSurfaces} سطح
              </Typography>
            </Stack>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Stack spacing={1.5}>
              <Typography variant="body2" color="text.secondary" fontWeight={500}>
                میانگین عمق پاکت
              </Typography>
              <Typography variant="h4" fontWeight={600}>
                {analysis.avgPocketDepth.toFixed(1)} mm
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Pocket Depth
              </Typography>
            </Stack>
          </Box>
        </Grid>

        <Grid item xs={12} md={4}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Stack spacing={1.5}>
              <Typography variant="body2" color="text.secondary" fontWeight={500}>
                میانگین CAL
              </Typography>
              <Typography variant="h4" fontWeight={600}>
                {analysis.avgCAL.toFixed(1)} mm
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Clinical Attachment Level
              </Typography>
            </Stack>
          </Box>
        </Grid>
      </Grid>

      {/* Disease Classification */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              گستردگی بیماری
            </Typography>
            <Box
              sx={{
                mt: 2,
                p: 2,
                textAlign: 'center',
                borderRadius: 1,
                bgcolor:
                  analysis.diseaseExtent === 'Generalized'
                    ? theme.palette.error.lighter
                    : theme.palette.success.lighter,
              }}
            >
              <Typography variant="h6" fontWeight={600}>
                {analysis.diseaseExtent}
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                {analysis.affectedTeeth} از {analysis.totalTeeth} دندان درگیر (
                {((analysis.affectedTeeth / analysis.totalTeeth) * 100).toFixed(0)}%)
              </Typography>
            </Box>
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              شدت بیماری
            </Typography>
            <Box
              sx={{
                mt: 2,
                p: 2,
                textAlign: 'center',
                borderRadius: 1,
                bgcolor: theme.palette.warning.lighter,
              }}
            >
              <Typography variant="h6" fontWeight={600}>
                {analysis.diseaseSeverity}
              </Typography>
              <Stack spacing={0.5} sx={{ mt: 1.5 }}>
                <Typography variant="caption" color="text.secondary">
                  Stage I: CAL 1-2mm
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Stage II: CAL 3-4mm
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Stage III: CAL ≥5mm
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Stage IV: CAL ≥5mm + tooth loss
                </Typography>
              </Stack>
            </Box>
          </Box>
        </Grid>
      </Grid>

      {/* Charts */}
      <Grid container spacing={2}>
        <Grid item xs={12} md={6}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              توزیع BOP در کادران‌ها
            </Typography>
            <Chart type="bar" series={bopChartSeries} options={bopChartOptions} height={300} />
          </Box>
        </Grid>

        <Grid item xs={12} md={6}>
          <Box
            sx={{
              p: 2.5,
              border: `1px solid ${theme.palette.divider}`,
              borderRadius: 1,
              bgcolor: 'background.paper',
            }}
          >
            <Typography variant="subtitle1" fontWeight={600} gutterBottom>
              تحلیل جامع پریودونتال
            </Typography>
            <Typography variant="caption" color="text.secondary" sx={{ mb: 2, display: 'block' }}>
              نمایش چندبعدی عوامل مؤثر بر سلامت پریودونتال
            </Typography>
            <Chart type="radar" series={polygonChartSeries} options={polygonChartOptions} height={300} />
          </Box>
        </Grid>
      </Grid>

      {/* Periodontal Visualization */}
      <PeriodontalVisualization chartData={chartData} />

      {/* Treatment Plan */}
      {treatmentPlan && (
        <Box
          sx={{
            border: `1px solid ${theme.palette.divider}`,
            borderRadius: 2,
            bgcolor: 'background.paper',
            overflow: 'hidden',
          }}
        >
          {/* Header */}
          <Box
            sx={{
              p: 3,
              borderBottom: `1px solid ${theme.palette.divider}`,
              bgcolor: 'background.paper',
            }}
          >
            <Stack direction="row" spacing={2} alignItems="center">
              <Iconify 
                icon="medical-icon:i-あprescription" 
                width={32} 
                sx={{ color: theme.palette.text.primary }}
              />
              <Box>
                <Typography variant="h6" fontWeight={600}>
                  طرح درمان پیشنهادی
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>
                  بر اساس ارزیابی جامع وضعیت پریودونتال بیمار
                </Typography>
              </Box>
            </Stack>
          </Box>

          {/* Content */}
          <Box sx={{ p: 3 }}>
            <Grid container spacing={3}>
              {/* Phase 1 */}
              <Grid item xs={12} md={6}>
                <Box
                  sx={{
                    p: 2.5,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 2,
                    bgcolor: 'background.paper',
                    height: '100%',
                  }}
                >
                  <Stack spacing={2.5}>
                    <Box>
                      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                        <Box
                          sx={{
                            width: 28,
                            height: 28,
                            borderRadius: 1,
                            bgcolor: 'transparent',
                            border: `2px solid ${theme.palette.primary.main}`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: theme.palette.primary.main,
                            fontWeight: 600,
                            fontSize: '0.875rem',
                          }}
                        >
                          1
                        </Box>
                        <Box>
                          <Typography variant="subtitle1" fontWeight={600}>
                            فاز 1: درمان اولیه
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            مدت زمان: {treatmentPlan.phase1.duration}
                          </Typography>
                        </Box>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        {treatmentPlan.phase1.description}
                      </Typography>
                    </Box>
                    <Divider />
                    <Stack spacing={1.5}>
                      {treatmentPlan.phase1.items.map((item, index) => (
                        <Box
                          key={index}
                          sx={{
                            p: 1.5,
                            borderRadius: 1,
                            bgcolor: 'background.paper',
                            border: item.warning ? `1px solid ${theme.palette.warning.main}` : `1px solid ${theme.palette.divider}`,
                          }}
                        >
                          <Stack direction="row" spacing={1.5} alignItems="flex-start">
                            <Iconify
                              icon={item.icon}
                              width={18}
                              sx={{
                                color: item.warning ? theme.palette.warning.main : theme.palette.text.secondary,
                                mt: 0.25,
                              }}
                            />
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="body2" fontWeight={500} sx={{ mb: 0.5 }}>
                                {item.title}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                {item.description}
                              </Typography>
                            </Box>
                          </Stack>
                        </Box>
                      ))}
                    </Stack>
                  </Stack>
                </Box>
              </Grid>

              {/* Phase 2 */}
              <Grid item xs={12} md={6}>
                <Box
                  sx={{
                    p: 2.5,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 2,
                    bgcolor: 'background.paper',
                    height: '100%',
                  }}
                >
                  <Stack spacing={2.5}>
                    <Box>
                      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                        <Box
                          sx={{
                            width: 28,
                            height: 28,
                            borderRadius: 1,
                            bgcolor: 'transparent',
                            border: `2px solid ${treatmentPlan.phase2.items.length > 0 ? theme.palette.warning.main : theme.palette.text.disabled}`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: treatmentPlan.phase2.items.length > 0 ? theme.palette.warning.main : theme.palette.text.disabled,
                            fontWeight: 600,
                            fontSize: '0.875rem',
                          }}
                        >
                          2
                        </Box>
                        <Box>
                          <Typography 
                            variant="subtitle1" 
                            fontWeight={600}
                            color={treatmentPlan.phase2.items.length > 0 ? 'text.primary' : 'text.secondary'}
                          >
                            فاز 2: ارزیابی مجدد و جراحی
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {treatmentPlan.phase2.duration}
                          </Typography>
                        </Box>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        {treatmentPlan.phase2.description}
                      </Typography>
                    </Box>
                    <Divider />
                    <Stack spacing={1.5}>
                      {treatmentPlan.phase2.items.length > 0 ? (
                        treatmentPlan.phase2.items.map((item, index) => (
                          <Box
                            key={index}
                            sx={{
                              p: 1.5,
                              borderRadius: 1,
                              bgcolor: 'background.paper',
                              border: `1px solid ${theme.palette.divider}`,
                            }}
                          >
                            <Stack direction="row" spacing={1.5} alignItems="flex-start">
                              <Iconify
                                icon={item.icon}
                                width={18}
                                sx={{ color: theme.palette.text.secondary, mt: 0.25 }}
                              />
                              <Box sx={{ flex: 1 }}>
                                <Typography variant="body2" fontWeight={500} sx={{ mb: 0.5 }}>
                                  {item.title}
                                </Typography>
                                <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                  {item.description}
                                </Typography>
                              </Box>
                            </Stack>
                          </Box>
                        ))
                      ) : (
                        <Box
                          sx={{
                            p: 2,
                            borderRadius: 1,
                            bgcolor: 'background.paper',
                            border: `1px dashed ${theme.palette.divider}`,
                            textAlign: 'center',
                          }}
                        >
                          <Typography variant="body2" color="text.secondary">
                            بر اساس نتایج فاز 1 تعیین می‌شود
                          </Typography>
                        </Box>
                      )}
                    </Stack>
                  </Stack>
                </Box>
              </Grid>

              {/* Phase 3 */}
              <Grid item xs={12} md={6}>
                <Box
                  sx={{
                    p: 2.5,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 2,
                    bgcolor: 'background.paper',
                    height: '100%',
                  }}
                >
                  <Stack spacing={2.5}>
                    <Box>
                      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                        <Box
                          sx={{
                            width: 28,
                            height: 28,
                            borderRadius: 1,
                            bgcolor: 'transparent',
                            border: `2px solid ${treatmentPlan.phase3.items.length > 0 && treatmentPlan.phase3.items[0].title !== 'نیازی به درمان بازسازی‌ای نیست' ? theme.palette.info.main : theme.palette.text.disabled}`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: treatmentPlan.phase3.items.length > 0 && treatmentPlan.phase3.items[0].title !== 'نیازی به درمان بازسازی‌ای نیست' ? theme.palette.info.main : theme.palette.text.disabled,
                            fontWeight: 600,
                            fontSize: '0.875rem',
                          }}
                        >
                          3
                        </Box>
                        <Box>
                          <Typography 
                            variant="subtitle1" 
                            fontWeight={600}
                            color={treatmentPlan.phase3.items.length > 0 && treatmentPlan.phase3.items[0].title !== 'نیازی به درمان بازسازی‌ای نیست' ? 'text.primary' : 'text.secondary'}
                          >
                            فاز 3: بازسازی
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {treatmentPlan.phase3.duration}
                          </Typography>
                        </Box>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        {treatmentPlan.phase3.description}
                      </Typography>
                    </Box>
                    <Divider />
                    <Stack spacing={1.5}>
                      {treatmentPlan.phase3.items.map((item, index) => (
                        <Box
                          key={index}
                          sx={{
                            p: 1.5,
                            borderRadius: 1,
                            bgcolor: 'background.paper',
                            border: `1px solid ${theme.palette.divider}`,
                          }}
                        >
                          <Stack direction="row" spacing={1.5} alignItems="flex-start">
                            <Iconify
                              icon={item.icon}
                              width={18}
                              sx={{ color: theme.palette.text.secondary, mt: 0.25 }}
                            />
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="body2" fontWeight={500} sx={{ mb: 0.5 }}>
                                {item.title}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                {item.description}
                              </Typography>
                            </Box>
                          </Stack>
                        </Box>
                      ))}
                    </Stack>
                  </Stack>
                </Box>
              </Grid>

              {/* Phase 4 */}
              <Grid item xs={12} md={6}>
                <Box
                  sx={{
                    p: 2.5,
                    border: `1px solid ${theme.palette.divider}`,
                    borderRadius: 2,
                    bgcolor: 'background.paper',
                    height: '100%',
                  }}
                >
                  <Stack spacing={2.5}>
                    <Box>
                      <Stack direction="row" spacing={1.5} alignItems="center" sx={{ mb: 1.5 }}>
                        <Box
                          sx={{
                            width: 28,
                            height: 28,
                            borderRadius: 1,
                            bgcolor: 'transparent',
                            border: `2px solid ${theme.palette.success.main}`,
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: theme.palette.success.main,
                            fontWeight: 600,
                            fontSize: '0.875rem',
                          }}
                        >
                          4
                        </Box>
                        <Box>
                          <Typography variant="subtitle1" fontWeight={600}>
                            فاز 4: نگهداری
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {treatmentPlan.phase4.duration}
                          </Typography>
                        </Box>
                      </Stack>
                      <Typography variant="body2" color="text.secondary">
                        {treatmentPlan.phase4.description}
                      </Typography>
                    </Box>
                    <Divider />
                    <Stack spacing={1.5}>
                      {treatmentPlan.phase4.items.map((item, index) => (
                        <Box
                          key={index}
                          sx={{
                            p: 1.5,
                            borderRadius: 1,
                            bgcolor: 'background.paper',
                            border: `1px solid ${theme.palette.divider}`,
                          }}
                        >
                          <Stack direction="row" spacing={1.5} alignItems="flex-start">
                            <Iconify
                              icon={item.icon}
                              width={18}
                              sx={{ color: theme.palette.text.secondary, mt: 0.25 }}
                            />
                            <Box sx={{ flex: 1 }}>
                              <Typography variant="body2" fontWeight={500} sx={{ mb: 0.5 }}>
                                {item.title}
                              </Typography>
                              <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                                {item.description}
                              </Typography>
                            </Box>
                          </Stack>
                        </Box>
                      ))}
                    </Stack>
                  </Stack>
                </Box>
              </Grid>
            </Grid>
          </Box>
        </Box>
      )}
    </Stack>
  );
}

export const AnalysisTab = memo(AnalysisTabComponent, (prevProps, nextProps) => (
  // Only re-render if patient ID or onUpdate function changes
  prevProps.patient?.id === nextProps.patient?.id &&
  prevProps.onUpdate === nextProps.onUpdate
));



