import { useForm } from 'react-hook-form';
import { memo, useMemo, useState, useEffect } from 'react';

import Box from '@mui/material/Box';
import Grid from '@mui/material/Grid';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import { Checkbox } from '@mui/material';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import CardHeader from '@mui/material/CardHeader';
import Typography from '@mui/material/Typography';
import CardContent from '@mui/material/CardContent';
import FormControlLabel from '@mui/material/FormControlLabel';

import axios, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';
import { Form, Field } from 'src/components/hook-form';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const SYSTEMIC_DISEASES = [
  { id: 'diabetes', label: 'دیابت', icon: 'solar:medicine-bold-duotone' },
  { id: 'hypertension', label: 'فشار خون بالا', icon: 'solar:heart-pulse-bold-duotone' },
  { id: 'cardiovascularDisease', label: 'بیماری‌های قلبی', icon: 'solar:heart-pulse-2-bold-duotone' },
  { id: 'rheumatoidArthritis', label: 'آرتریت روماتوئید', icon: 'solar:bone-bold-duotone' },
  { id: 'osteoporosis', label: 'استئوپروز', icon: 'solar:bone-crack-bold-duotone' },
  { id: 'hiv', label: 'HIV/AIDS', icon: 'solar:virus-bold-duotone' },
  { id: 'hepatitis', label: 'هپاتیت', icon: 'solar:medical-kit-bold-duotone' },
  { id: 'smoking', label: 'مصرف سیگار', icon: 'solar:cigarette-bold-duotone' },
  { id: 'alcohol', label: 'مصرف الکل', icon: 'solar:wineglass-triangle-bold-duotone' },
  { id: 'stress', label: 'استرس', icon: 'solar:emoji-funny-bold-duotone' },
  { id: 'pregnancy', label: 'بارداری', icon: 'solar:heart-pulse-rounded-bold-duotone' },
];

const STATUS_OPTIONS = [
  { value: 'PENDING', label: 'شروع درمان' },
  { value: 'IN_TREATMENT', label: 'در حال درمان' },
  { value: 'COMPLETED', label: 'اتمام درمان' },
  { value: 'CANCELLED', label: 'متوقف شده' },
];

// ----------------------------------------------------------------------

function PatientInfoTabComponent({ patient, onUpdate }) {
  const { user } = useAuthContext();
  const [saving, setSaving] = useState(false);
  const [medicalHistory, setMedicalHistory] = useState({});

  // Shared styles for smaller input text - memoized
  const inputStyles = useMemo(() => ({
    '& .MuiInputBase-input': {
      fontSize: '0.8125rem', // Smaller than orthodontics (0.875rem) - one size smaller
    },
    '& .MuiInputLabel-root': {
      fontSize: '0.8125rem',
    },
    '& .MuiSelect-select': {
      fontSize: '0.8125rem',
    },
    '& .MuiFormHelperText-root': {
      fontSize: '0.75rem',
    },
  }), []);

  // Shared slotProps for Select components - memoized
  const selectSlotProps = useMemo(() => ({
    paper: {
      '& .MuiMenuItem-root': {
        fontSize: '0.8125rem',
      },
    },
  }), []);

  const methods = useForm({
    defaultValues: {
      firstName: patient?.firstName || '',
      lastName: patient?.lastName || '',
      phone: patient?.phone || '',
      age: patient?.age || '',
      gender: patient?.gender || '',
      diagnosis: patient?.diagnosis || '',
      treatment: patient?.treatment || '',
      status: patient?.status || 'PENDING',
      notes: patient?.notes || '',
      nextVisitTime: patient?.nextVisitTime ? new Date(patient.nextVisitTime) : null,
      treatmentStartDate: patient?.treatmentStartDate ? new Date(patient.treatmentStartDate) : null,
    },
  });

  useEffect(() => {
    if (patient?.medicalHistory) {
      setMedicalHistory(
        typeof patient.medicalHistory === 'string'
          ? JSON.parse(patient.medicalHistory)
          : patient.medicalHistory
      );
    }
  }, [patient]);

  const handleMedicalHistoryChange = (diseaseId, checked) => {
    setMedicalHistory((prev) => ({
      ...prev,
      [diseaseId]: checked,
    }));
  };

  const onSubmit = async (data) => {
    try {
      setSaving(true);
      await axios.put(
        `${endpoints.patients}/${patient.id}`,
        {
          ...data,
          medicalHistory,
        },
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        }
      );
      toast.success('اطلاعات بیمار با موفقیت ذخیره شد');
      onUpdate?.();
    } catch (error) {
      console.error('Error updating patient:', error);
      toast.error('خطا در ذخیره اطلاعات');
    } finally {
      setSaving(false);
    }
  };

  return (
    <Form methods={methods} onSubmit={methods.handleSubmit(onSubmit)}>
      <Stack spacing={3}>
        {/* Basic Information */}
        <Card>
          <CardHeader
            title="اطلاعات پایه"
            avatar={<Iconify icon="solar:user-id-bold-duotone" width={24} />}
          />
          <CardContent>
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Field.Text name="firstName" label="نام" sx={inputStyles} />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.Text name="lastName" label="نام خانوادگی" sx={inputStyles} />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.Text name="phone" label="شماره تماس" sx={inputStyles} />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.Text name="age" label="سن" type="number" sx={inputStyles} />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.Select 
                  name="gender" 
                  label="جنسیت"
                  sx={inputStyles}
                  slotProps={selectSlotProps}
                >
                  <MenuItem value="MALE">مرد</MenuItem>
                  <MenuItem value="FEMALE">زن</MenuItem>
                  <MenuItem value="OTHER">سایر</MenuItem>
                </Field.Select>
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.DatePicker 
                  name="treatmentStartDate" 
                  label="تاریخ شروع درمان"
                  slotProps={{ textField: { sx: inputStyles } }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.DatePicker 
                  name="nextVisitTime" 
                  label="ویزیت بعدی"
                  slotProps={{ textField: { sx: inputStyles } }}
                />
              </Grid>
              <Grid item xs={12} md={6}>
                <Field.Select 
                  name="status" 
                  label="وضعیت"
                  sx={inputStyles}
                  slotProps={selectSlotProps}
                >
                  {STATUS_OPTIONS.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </Field.Select>
              </Grid>
            </Grid>
          </CardContent>
        </Card>

        {/* Medical History */}
        <Card>
          <CardHeader
            title="بیماری‌های زمینه‌ای و سابقه پزشکی"
            subheader="بیماری‌های سیستمیک که می‌توانند بر وضعیت پریودونشیال تأثیر بگذارند"
            avatar={<Iconify icon="solar:health-bold-duotone" width={24} />}
          />
          <CardContent>
            <Grid container spacing={1}>
              {SYSTEMIC_DISEASES.map((disease) => (
                <Grid item xs={12} sm={6} md={4} key={disease.id}>
                  <FormControlLabel
                    control={
                      <Checkbox
                        checked={medicalHistory[disease.id] || false}
                        onChange={(e) =>
                          handleMedicalHistoryChange(disease.id, e.target.checked)
                        }
                        size="small"
                      />
                    }
                    label={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Iconify icon={disease.icon} width={18} />
                        <Typography variant="body2" sx={{ fontSize: '0.875rem' }}>
                          {disease.label}
                        </Typography>
                      </Box>
                    }
                    sx={{
                      margin: 0,
                      padding: '4px 8px',
                      '& .MuiFormControlLabel-label': {
                        marginLeft: '4px',
                      },
                    }}
                  />
                </Grid>
              ))}
            </Grid>

            {/* Additional fields for smoking */}
            {medicalHistory.smoking && (
              <Box sx={{ mt: 3 }}>
                <Typography variant="subtitle2" gutterBottom>
                  اطلاعات تکمیلی سیگار:
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={12} md={6}>
                    <Field.Text
                      name="smokingPackYears"
                      label="تعداد پاکت-سال (Pack-Years)"
                      type="number"
                      helperText="تعداد پاکت در روز × تعداد سال"
                      sx={inputStyles}
                    />
                  </Grid>
                </Grid>
              </Box>
            )}

            {/* Other diseases text field */}
            <Box sx={{ mt: 3 }}>
              <Field.Text
                name="otherDiseases"
                label="سایر بیماری‌ها"
                multiline
                rows={2}
                placeholder="سایر بیماری‌های زمینه‌ای را وارد کنید..."
                sx={inputStyles}
              />
            </Box>
          </CardContent>
        </Card>

        {/* Clinical Information */}
        <Card>
          <CardHeader
            title="اطلاعات بالینی"
            avatar={<Iconify icon="solar:clipboard-text-bold-duotone" width={24} />}
          />
          <CardContent>
            <Stack spacing={3}>
              <Field.Text 
                name="diagnosis" 
                label="تشخیص" 
                multiline 
                rows={3}
                sx={inputStyles}
              />
              <Field.Text 
                name="treatment" 
                label="طرح درمان" 
                multiline 
                rows={3}
                sx={inputStyles}
              />
              <Field.Text 
                name="notes" 
                label="یادداشت‌ها" 
                multiline 
                rows={4}
                sx={inputStyles}
              />
            </Stack>
          </CardContent>
        </Card>

        {/* Submit Button */}
        <Box sx={{ display: 'flex', justifyContent: 'flex-end', gap: 2 }}>
          <Button
            type="submit"
            variant="contained"
            size="large"
            disabled={saving}
            startIcon={<Iconify icon="solar:diskette-bold-duotone" />}
          >
            {saving ? 'در حال ذخیره...' : 'ذخیره اطلاعات'}
          </Button>
        </Box>
      </Stack>
    </Form>
  );
}

export const PatientInfoTab = memo(PatientInfoTabComponent, (prevProps, nextProps) => (
  // Only re-render if patient ID changes or onUpdate function changes
  // We check patient ID to ensure we re-render when switching between patients
  prevProps.patient?.id === nextProps.patient?.id &&
  prevProps.onUpdate === nextProps.onUpdate
));



