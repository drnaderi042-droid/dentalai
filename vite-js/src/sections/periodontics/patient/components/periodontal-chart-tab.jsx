import { memo, useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';

import axios, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';

import { useAuthContext } from 'src/auth/hooks';

import { PeriodontalChart } from '../../components/periodontal-chart';

// ----------------------------------------------------------------------

// Initialize empty chart data structure
const createEmptyChart = () => {
  const teeth = {};
  // Upper teeth: 1-16 (right to left)
  // Lower teeth: 17-32 (left to right)
  for (let i = 1; i <= 32; i += 1) {
          teeth[i] = {
        facial: {
          pocketDepth: [null, null, null],
          gingivalMargin: [null, null, null],
          bleeding: [false, false, false],
          suppuration: [false, false, false],
          furcation: null,
          mobility: 0,
          plaque: false,
        },
        lingual: {
          pocketDepth: [null, null, null],
          gingivalMargin: [null, null, null],
          bleeding: [false, false, false],
          suppuration: [false, false, false],
          furcation: null,
          mobility: 0,
          plaque: false,
        },
        missing: false,
        implant: false,
      };
  }
  return teeth;
};

// ----------------------------------------------------------------------

function PeriodontalChartTabComponent({ patient, onUpdate, onNavigateToAnalysis }) {
  const { user } = useAuthContext();
  const [chartData, setChartData] = useState({
    teeth: createEmptyChart(),
    date: new Date(),
  });
  const [charts, setCharts] = useState([]);
  const [selectedChartId, setSelectedChartId] = useState(null);
  const [saving, setSaving] = useState(false);
  const [loading, setLoading] = useState(true);
  const [selectedQuadrant, setSelectedQuadrant] = useState('Upper Right');
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [chartToDelete, setChartToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch all charts for this patient
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

        // Load the latest chart if available
        if (fetchedCharts.length > 0) {
          const latestChart = fetchedCharts[0];
          setSelectedChartId(latestChart.id);
          setChartData({
            teeth:
              typeof latestChart.teeth === 'string'
                ? JSON.parse(latestChart.teeth)
                : latestChart.teeth,
            date: new Date(latestChart.date),
          });
        }
      } catch (error) {
        console.error('Error fetching charts:', error);
        toast.error('خطا در بارگیری چارت‌ها');
      } finally {
        setLoading(false);
      }
    };

    if (user && patient) {
      fetchCharts();
    }
  }, [user, patient]);

  const handleChartChange = (event) => {
    const chartId = event.target.value;
    setSelectedChartId(chartId);

    if (chartId === 'new') {
      setChartData({
        teeth: createEmptyChart(),
        date: new Date(),
      });
    } else {
      const selected = charts.find((c) => c.id === chartId);
      if (selected) {
        setChartData({
          teeth:
            typeof selected.teeth === 'string' ? JSON.parse(selected.teeth) : selected.teeth,
          date: new Date(selected.date),
        });
      }
    }
  };

  const handleSaveChart = async () => {
    try {
      setSaving(true);

      const payload = {
        patientId: patient.id,
        teeth: chartData.teeth,
        date: chartData.date,
      };

      if (selectedChartId && selectedChartId !== 'new') {
        // Update existing chart
        await axios.put(
          `${endpoints.patients}/${patient.id}/periodontal-charts/${selectedChartId}`,
          payload,
          {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          }
        );
        toast.success('چارت با موفقیت بروزرسانی شد');
      } else {
        // Create new chart
        const response = await axios.post(
          `${endpoints.patients}/${patient.id}/periodontal-charts`,
          payload,
          {
            headers: {
              Authorization: `Bearer ${user?.accessToken}`,
            },
          }
        );
        toast.success('چارت جدید با موفقیت ذخیره شد');
        setSelectedChartId(response.data.chart.id);
      }

      onUpdate?.();

      // Refresh charts list
      const response = await axios.get(`${endpoints.patients}/${patient.id}/periodontal-charts`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setCharts(response.data.charts || []);
    } catch (error) {
      console.error('Error saving chart:', error);
      toast.error('خطا در ذخیره چارت');
    } finally {
      setSaving(false);
    }
  };

  const handleUpdateTooth = useCallback((toothNumber, field, value) => {
    setChartData((prev) => ({
      ...prev,
      teeth: {
        ...prev.teeth,
        [toothNumber]: {
          ...prev.teeth[toothNumber],
          [field]: value,
        },
      },
    }));
  }, []);

  const handleUpdateToothFields = useCallback((toothNumber, fields) => {
    setChartData((prev) => ({
      ...prev,
      teeth: {
        ...prev.teeth,
        [toothNumber]: {
          ...prev.teeth[toothNumber],
          ...fields,
        },
      },
    }));
  }, []);

  if (loading) {
    return (
      <Box sx={{ textAlign: 'center', py: 4 }}>
        <Typography>در حال بارگیری چارت...</Typography>
      </Box>
    );
  }

  // Quadrant options
  const QUADRANTS = [
    { value: 'Upper Right', label: 'بالا راست' },
    { value: 'Upper Left', label: 'بالا چپ' },
    { value: 'Lower Left', label: 'پایین چپ' },
    { value: 'Lower Right', label: 'پایین راست' },
  ];

  const handleQuadrantChange = (event) => {
    setSelectedQuadrant(event.target.value);
  };

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

      // If deleted chart was selected, select the latest one or create new
      if (selectedChartId === chartToDelete.id) {
        if (fetchedCharts.length > 0) {
          const latestChart = fetchedCharts[0];
          setSelectedChartId(latestChart.id);
          setChartData({
            teeth:
              typeof latestChart.teeth === 'string'
                ? JSON.parse(latestChart.teeth)
                : latestChart.teeth,
            date: new Date(latestChart.date),
          });
        } else {
          setSelectedChartId('new');
          setChartData({
            teeth: createEmptyChart(),
            date: new Date(),
          });
        }
      }

      setDeleteDialogOpen(false);
      setChartToDelete(null);
      onUpdate?.();
    } catch (error) {
      console.error('Error deleting chart:', error);
      toast.error('خطا در حذف چارت');
    } finally {
      setDeleting(false);
    }
  };

  return (
    <Stack spacing={3}>
      {/* Chart and Quadrant Selector */}
      <Card>
        <CardContent>
          <Stack direction={{ xs: 'column', md: 'row' }} spacing={2} alignItems="center">
            <FormControl fullWidth>
              <InputLabel>انتخاب چارت</InputLabel>
              <Select value={selectedChartId || 'new'} onChange={handleChartChange} label="انتخاب چارت">
                <MenuItem value="new">
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Iconify icon="solar:add-circle-bold" />
                    چارت جدید
                  </Box>
                </MenuItem>
                {charts.map((chart) => (
                  <MenuItem key={chart.id} value={chart.id}>
                    {new Date(chart.date).toLocaleDateString('fa-IR')} -{' '}
                    {new Date(chart.date).toLocaleTimeString('fa-IR', {
                      hour: '2-digit',
                      minute: '2-digit',
                    })}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControl fullWidth>
              <InputLabel>انتخاب کوادرانت</InputLabel>
              <Select
                value={selectedQuadrant}
                onChange={handleQuadrantChange}
                label="انتخاب کوادرانت"
              >
                {QUADRANTS.map((quad) => (
                  <MenuItem key={quad.value} value={quad.value}>
                    {quad.label}
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            {selectedChartId && selectedChartId !== 'new' && (
              <IconButton
                color="error"
                onClick={() => {
                  const chart = charts.find((c) => c.id === selectedChartId);
                  if (chart) {
                    handleDeleteChart(chart);
                  }
                }}
                sx={{ minWidth: 48, height: 48 }}
              >
                <Iconify icon="solar:trash-bin-trash-bold" width={24} />
              </IconButton>
            )}
          </Stack>
        </CardContent>
      </Card>

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

      {/* Periodontal Chart Component */}
      <PeriodontalChart
        chartData={chartData}
        onUpdateTooth={handleUpdateTooth}
        onUpdateToothFields={handleUpdateToothFields}
        selectedQuadrant={selectedQuadrant}
        onQuadrantChange={handleQuadrantChange}
      />

      {/* Save and Analysis Buttons */}
      <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
        <Button
          variant="soft"
          color="primary"
          onClick={handleSaveChart}
          disabled={saving}
          startIcon={<Iconify icon="solar:diskette-bold" />}
          sx={{ minWidth: 150 }}
        >
          {saving ? 'در حال ذخیره...' : 'ذخیره چارت'}
        </Button>
        <Button
          variant="contained"
          color="primary"
          onClick={() => {
            if (onNavigateToAnalysis) {
              onNavigateToAnalysis();
            }
          }}
          startIcon={<Iconify icon="solar:chart-2-bold" />}
          sx={{ minWidth: 150 }}
        >
          آنالیز
        </Button>
      </Box>
    </Stack>
  );
}

export const PeriodontalChartTab = memo(PeriodontalChartTabComponent, (prevProps, nextProps) => (
  // Only re-render if patient ID or callbacks change
  prevProps.patient?.id === nextProps.patient?.id &&
  prevProps.onUpdate === nextProps.onUpdate &&
  prevProps.onNavigateToAnalysis === nextProps.onNavigateToAnalysis
));



