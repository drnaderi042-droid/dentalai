import { useForm } from 'react-hook-form';
import { useState, useEffect } from 'react';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Grid from '@mui/material/Grid';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import { alpha } from '@mui/material/styles';
import MenuItem from '@mui/material/MenuItem';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';

import axios, { endpoints } from 'src/utils/axios';

import { CONFIG } from 'src/config-global';
import { DashboardContent } from 'src/layouts/dashboard';

import { Iconify } from 'src/components/iconify';
import { Form, Field } from 'src/components/hook-form';
import { ConfirmDialog } from 'src/components/custom-dialog';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const statusMap = {
  'PENDING': 'شروع درمان',
  'IN_TREATMENT': 'در حال درمان',
  'COMPLETED': 'اتمام درمان',
  'CANCELLED': 'متوقف شده',
};

export function ProsthodonticsView() {
  const { user } = useAuthContext();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientToDelete, setPatientToDelete] = useState(null);

  const addForm = useForm({
    defaultValues: {
      firstName: '',
      lastName: '',
      phone: '',
      age: '',
      diagnosis: '',
      treatment: '',
      status: 'PENDING',
      notes: '',
      nextVisitTime: null,
      treatmentStartDate: null,
    },
  });

  const editForm = useForm({
    defaultValues: {
      firstName: '',
      lastName: '',
      phone: '',
      age: '',
      diagnosis: '',
      treatment: '',
      status: 'PENDING',
      notes: '',
      nextVisitTime: null,
      treatmentStartDate: null,
    },
  });

  // Fetch patients on mount
  useEffect(() => {
    const fetchPatients = async () => {
      try {
        setLoading(true);
        const response = await axios.get(endpoints.patients, {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        });
        setPatients(response.data.patients || []);
        setError(null);
      } catch (err) {
        console.error('Error fetching patients:', err);
        setError('خطا در بارگیری بیماران');
        setPatients([]);
      } finally {
        setLoading(false);
      }
    };

    if (user) {
      fetchPatients();
    }
  }, [user]);

  const handleOpenAddDialog = () => {
    addForm.reset({
      firstName: '',
      lastName: '',
      phone: '',
      age: '',
      diagnosis: '',
      treatment: '',
      status: 'PENDING',
      notes: '',
      nextVisitTime: null,
      treatmentStartDate: null,
    });
    setOpenAddDialog(true);
  };

  const handleCloseAddDialog = () => {
    setOpenAddDialog(false);
    addForm.reset();
  };

  const handleOpenEditDialog = (patient) => {
    setSelectedPatient(patient);
    editForm.reset({
      firstName: patient.firstName,
      lastName: patient.lastName,
      phone: patient.phone,
      age: patient.age,
      diagnosis: patient.diagnosis,
      treatment: patient.treatment,
      status: patient.status,
      notes: patient.notes || '',
      nextVisitTime: patient.nextVisitTime ? new Date(patient.nextVisitTime) : null,
      treatmentStartDate: patient.treatmentStartDate ? new Date(patient.treatmentStartDate) : null,
    });
    setOpenEditDialog(true);
  };

  const handleCloseEditDialog = () => {
    setOpenEditDialog(false);
    setSelectedPatient(null);
    editForm.reset();
  };

  const handleOpenDeleteDialog = (patient) => {
    setPatientToDelete(patient);
    setOpenDeleteDialog(true);
  };

  const handleCloseDeleteDialog = () => {
    setOpenDeleteDialog(false);
    setPatientToDelete(null);
  };

  const handleAddPatient = async (data) => {
    try {
      const response = await axios.post(endpoints.patients, {
        ...data,
        specialty: 'PROSTHODONTICS',
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setPatients(prev => [...prev, response.data.patient]);
      handleCloseAddDialog();
    } catch (err) {
      console.error('Error adding patient:', err);
      alert('خطا در افزودن بیمار');
    }
  };

  const handleUpdatePatient = async (data) => {
    try {
      const response = await axios.put(`${endpoints.patients}/${selectedPatient.id}`, data, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setPatients(prev =>
        prev.map(patient =>
          patient.id === selectedPatient.id
            ? response.data.patient
            : patient
        )
      );
      handleCloseEditDialog();
    } catch (err) {
      console.error('Error updating patient:', err);
      alert('خطا در بروزرسانی بیمار');
    }
  };

  const handleDeletePatient = async (patientId) => {
    try {
      await axios.delete(`${endpoints.patients}/${patientId}`, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setPatients(prev => prev.filter(patient => patient.id !== patientId));
    } catch (err) {
      console.error('Error deleting patient:', err);
      alert('خطا در حذف بیمار');
    }
  };

  if (loading) {
    return (
      <DashboardContent>
        <Container>
          <Typography variant="h4" sx={{ textAlign: 'center', mt: 4 }}>
            بارگیری بیماران...
          </Typography>
        </Container>
      </DashboardContent>
    );
  }

  return (
    <DashboardContent>
      <Container>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4">
            بخش پروتزهای دندانی
          </Typography>
          <Button
            variant="contained"
            startIcon={<Iconify icon="mingcute:add-line" />}
            onClick={handleOpenAddDialog}
          >
            افزودن بیمار جدید
          </Button>
        </Box>

        {error && (
          <Typography variant="body1" color="error" sx={{ mb: 4 }}>
            {error}
          </Typography>
        )}

        <Typography variant="body1" color="text.secondary" mb={4}>
          مدیریت بیماران پروتزهای دندانی و ثبت اطلاعات تشخیصی
        </Typography>

        <Grid container spacing={3}>
          {patients.map((patient) => {
            // Get first available profile image
            const profileImage = patient.radiologyImages?.find(img => img.category === 'profile');

            return (
              <Grid item xs={12} sm={6} md={4} key={patient.id}>
                <Card>
                  <CardContent>
                    <Box display="flex" alignItems="center" mb={2}>
                      <Avatar
                        sx={{ mr: 2, width: 56, height: 56 }}
                        src={profileImage ? `${CONFIG.site.serverUrl}${profileImage.path}` : undefined}
                      >
                        {profileImage ? null : `${patient.firstName.charAt(0)}${patient.lastName.charAt(0)}`}
                      </Avatar>
                      <Box>
                        <Typography variant="h6">{patient.firstName} {patient.lastName}</Typography>
                        <Typography variant="body2" color="text.secondary">
                          {patient.age} سال
                        </Typography>
                      </Box>
                    </Box>

                    {patient.nextVisitTime && (
                      <Typography variant="body2" color="text.secondary" mb={1}>
                        <strong>ویزیت بعدی:</strong> {new Date(patient.nextVisitTime).toLocaleDateString('fa-IR')}
                      </Typography>
                    )}

                    {patient.treatmentStartDate && (
                      <Typography variant="body2" color="text.secondary" mb={1}>
                        <strong>شروع درمان:</strong> {new Date(patient.treatmentStartDate).toLocaleDateString('fa-IR')}
                      </Typography>
                    )}

                    <Typography variant="body2" color="text.secondary" mb={1}>
                      <strong>تشخیص:</strong> {patient.diagnosis}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" mb={1}>
                      <strong>درمان:</strong> {patient.treatment}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" mb={1}>
                      <strong>وضعیت:</strong> {statusMap[patient.status] || patient.status}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      <strong>آخرین ویزیت:</strong> {new Date(patient.updatedAt).toLocaleDateString('fa-IR')}
                    </Typography>
                  </CardContent>

                  <CardActions>
                    <Button
                      size="small"
                      onClick={() => handleOpenEditDialog(patient)}
                      startIcon={<Iconify icon="solar:pen-bold" />}
                    >
                      اصلاح
                    </Button>
                    <Button
                      size="small"
                      onClick={() => {
                        window.location.href = `/dashboard/prosthodontics/patient/${patient.id}`;
                      }}
                      startIcon={<Iconify icon="solar:user-bold" />}
                    >
                      مدیریت بیمار
                    </Button>
                    <Button
                      size="small"
                      color="error"
                      onClick={() => handleOpenDeleteDialog(patient)}
                      startIcon={<Iconify icon="solar:trash-bin-minimalistic-bold" />}
                    >
                      حذف
                    </Button>
                  </CardActions>
                </Card>
              </Grid>
            );
          })}
        </Grid>

        {/* Delete Dialog */}
        <ConfirmDialog
          open={openDeleteDialog}
          onClose={handleCloseDeleteDialog}
          title="حذف بیمار"
          content={`آیا مطمئن هستید که می‌خواهید بیمار ${patientToDelete?.firstName} ${patientToDelete?.lastName} را حذف کنید؟`}
          action={
            <Button
              variant="contained"
              color="error"
              onClick={() => {
                if (patientToDelete) {
                  handleDeletePatient(patientToDelete.id);
                }
                handleCloseDeleteDialog();
              }}
            >
              حذف
            </Button>
          }
        />

        {/* Add Patient Dialog */}
        <Dialog open={openAddDialog} onClose={handleCloseAddDialog} maxWidth="md" fullWidth>
          <DialogTitle>افزودن بیمار جدید</DialogTitle>
          <Form methods={addForm} onSubmit={addForm.handleSubmit(handleAddPatient)}>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="firstName" label="نام" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="lastName" label="نام خانوادگی" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="phone" label="شماره تلفن" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="age" label="سن" type="number" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.DatePicker name="treatmentStartDate" label="تاریخ شروع درمان" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.DatePicker name="nextVisitTime" label="تایم ویزیت بعدی" />
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="diagnosis"
                    label="تشخیص"
                    multiline
                    rows={2}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="treatment"
                    label="طرح درمان"
                    multiline
                    rows={2}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Field.Select name="status" label="وضعیت">
                    <MenuItem value="PENDING">شروع درمان</MenuItem>
                    <MenuItem value="IN_TREATMENT">در حال درمان</MenuItem>
                    <MenuItem value="COMPLETED">اتمام درمان</MenuItem>
                    <MenuItem value="CANCELLED">متوقف شده</MenuItem>
                  </Field.Select>
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="notes"
                    label="یادداشت ها"
                    multiline
                    rows={3}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    آپلود تصاویر رادیولوژی
                  </Typography>
                  <Paper
                    variant="outlined"
                    sx={{
                      p: 3,
                      textAlign: 'center',
                      borderStyle: 'dashed',
                      bgcolor: (theme) => alpha(theme.palette.grey[500], 0.08),
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: (theme) => alpha(theme.palette.grey[500], 0.16),
                      },
                    }}
                  >
                    <Stack spacing={1} alignItems="center">
                      <Iconify icon="solar:cloud-upload-bold" width={32} />
                      <Typography variant="body2">
                        تصاویر رادیولوژی را اینجا بکشید یا کلیک کنید
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        PNG, JPG, JPEG (حداکثر ۱۰ مگابایت)
                      </Typography>
                    </Stack>
                    <label htmlFor="radiology-upload-add">
                      <input
                        type="file"
                        multiple
                        accept="image/*"
                        style={{ display: 'none' }}
                        id="radiology-upload-add"
                      />
                      <Button variant="outlined" component="span" sx={{ mt: 2 }}>
                        انتخاب فایل
                      </Button>
                    </label>
                  </Paper>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseAddDialog}>انصراف</Button>
              <Button type="submit" variant="contained">
                افزودن بیمار
              </Button>
            </DialogActions>
          </Form>
        </Dialog>

        {/* Edit Patient Dialog */}
        <Dialog open={openEditDialog} onClose={handleCloseEditDialog} maxWidth="md" fullWidth>
          <DialogTitle>ویرایش بیمار</DialogTitle>
          <Form methods={editForm} onSubmit={editForm.handleSubmit(handleUpdatePatient)}>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="firstName" label="نام" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="lastName" label="نام خانوادگی" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="phone" label="شماره تلفن" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.Text name="age" label="سن" type="number" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.DatePicker name="treatmentStartDate" label="تاریخ شروع درمان" />
                </Grid>
                <Grid item xs={12} sm={6}>
                  <Field.DatePicker name="nextVisitTime" label="تایم ویزیت بعدی" />
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="diagnosis"
                    label="تشخیص"
                    multiline
                    rows={2}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="treatment"
                    label="طرح درمان"
                    multiline
                    rows={2}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Field.Select name="status" label="وضعیت">
                    <MenuItem value="PENDING">شروع درمان</MenuItem>
                    <MenuItem value="IN_TREATMENT">در حال درمان</MenuItem>
                    <MenuItem value="COMPLETED">اتمام درمان</MenuItem>
                    <MenuItem value="CANCELLED">متوقف شده</MenuItem>
                  </Field.Select>
                </Grid>
                <Grid item xs={12}>
                  <Field.Text
                    name="notes"
                    label="یادداشت ها"
                    multiline
                    rows={3}
                  />
                </Grid>
                <Grid item xs={12}>
                  <Typography variant="h6" gutterBottom>
                    آپلود تصاویر رادیولوژی
                  </Typography>
                  <Paper
                    variant="outlined"
                    sx={{
                      p: 3,
                      textAlign: 'center',
                      borderStyle: 'dashed',
                      bgcolor: (theme) => alpha(theme.palette.grey[500], 0.08),
                      cursor: 'pointer',
                      '&:hover': {
                        bgcolor: (theme) => alpha(theme.palette.grey[500], 0.16),
                      },
                    }}
                  >
                    <Stack spacing={1} alignItems="center">
                      <Iconify icon="solar:cloud-upload-bold" width={32} />
                      <Typography variant="body2">
                        تصاویر رادیولوژی را اینجا بکشید یا کلیک کنید
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        PNG, JPG, JPEG (حداکثر ۱۰ مگابایت)
                      </Typography>
                    </Stack>
                    <label htmlFor="radiology-upload-edit">
                      <input
                        type="file"
                        multiple
                        accept="image/*"
                        style={{ display: 'none' }}
                        id="radiology-upload-edit"
                      />
                      <Button variant="outlined" component="span" sx={{ mt: 2 }}>
                        انتخاب فایل
                      </Button>
                    </label>
                  </Paper>
                </Grid>
              </Grid>
            </DialogContent>
            <DialogActions>
              <Button onClick={handleCloseEditDialog}>انصراف</Button>
              <Button type="submit" variant="contained">
                بروزرسانی
              </Button>
            </DialogActions>
          </Form>
        </Dialog>
      </Container>
    </DashboardContent>
  );
}
