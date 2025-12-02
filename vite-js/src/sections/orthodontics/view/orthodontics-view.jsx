import { useForm } from 'react-hook-form';
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Grid from '@mui/material/Grid';
import Chip from '@mui/material/Chip';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Table from '@mui/material/Table';
import Avatar from '@mui/material/Avatar';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Tooltip from '@mui/material/Tooltip';
import Divider from '@mui/material/Divider';
import TableRow from '@mui/material/TableRow';
import MenuItem from '@mui/material/MenuItem';
import TableBody from '@mui/material/TableBody';
import TableCell from '@mui/material/TableCell';
import TableHead from '@mui/material/TableHead';
import Container from '@mui/material/Container';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import ListItemText from '@mui/material/ListItemText';
import ToggleButton from '@mui/material/ToggleButton';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';
import TableContainer from '@mui/material/TableContainer';
import TablePagination from '@mui/material/TablePagination';
import CircularProgress from '@mui/material/CircularProgress';
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup';

import axios, { endpoints } from 'src/utils/axios';
import { getImageUrl } from 'src/utils/url-helpers';

import { CONFIG } from 'src/config-global';
import { varAlpha } from 'src/theme/styles';
import { AvatarShape } from 'src/assets/illustrations';
import { DashboardContent } from 'src/layouts/dashboard';

import { Image } from 'src/components/image';
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

const statusMapReverse = {
  'شروع درمان': 'PENDING',
  'در حال درمان': 'IN_TREATMENT',
  'اتمام درمان': 'COMPLETED',
  'متوقف شده': 'CANCELLED',
};

export function OrthodonticsView() {
  const { user } = useAuthContext();
  const navigate = useNavigate();
  const [patients, setPatients] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [openAddDialog, setOpenAddDialog] = useState(false);
  const [openEditDialog, setOpenEditDialog] = useState(false);
  const [openDeleteDialog, setOpenDeleteDialog] = useState(false);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [patientToDelete, setPatientToDelete] = useState(null);
  const [viewMode, setViewMode] = useState('card'); // 'card' or 'table'
  const [tablePage, setTablePage] = useState(0);
  const [tableRowsPerPage, setTableRowsPerPage] = useState(10);
  // Form state used for add/edit operations
  const [formData, setFormData] = useState({});
  const addForm = useForm({
    defaultValues: {
      firstName: '',
      age: '',
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
        // Filter only Orthodontics patients
        const orthoPatients = (response.data.patients || []).filter(
          (p) => p.specialty === 'ORTHODONTICS'
        );
        
        // Fetch images for each patient in parallel
        const patientsWithImages = await Promise.all(
          orthoPatients.map(async (patient) => {
            try {
              const imagesResponse = await axios.get(`${endpoints.patients}/${patient.id}/images`, {
                headers: {
                  Authorization: `Bearer ${user?.accessToken}`,
                },
              });
              const images = imagesResponse.data.images || [];
              // Find profile or frontal image
              const profileImage = images.find(img => img.category === 'profile') ||
                                   images.find(img => img.category === 'frontal');
              
              return {
                ...patient,
                images: images,
                profileImage: profileImage || null,
              };
            } catch (imageError) {
              // If image fetch fails, just return patient without images
              return {
                ...patient,
                images: [],
                profileImage: null,
              };
            }
          })
        );
        
        setPatients(patientsWithImages);
        setError(null);
      } catch (err) {
        console.error('Error fetching patients:', err);
        // Only set error if it's an actual error, not an empty list
        if (err.response?.status !== 200 && err.response?.status !== 404) {
          setError('خطا در بارگیری بیماران');
        } else {
          setError(null);
        }
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
      age: '',
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

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleAddPatient = async (data) => {
    try {
      const response = await axios.post(endpoints.patients, {
        ...data,
        lastName: '-',
        phone: '-',
        diagnosis: '-',
        treatment: '-',
        specialty: 'ORTHODONTICS',
      }, {
        headers: {
          Authorization: `Bearer ${user?.accessToken}`,
        },
      });
      setPatients(prev => [...prev, response.data.patient]);
      handleCloseAddDialog();
    } catch (err) {
      console.error('Error adding patient:', err);
      const errorMessage = err?.response?.data?.message || err?.message || 'خطا در افزودن بیمار';
      alert(errorMessage);
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
          <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '400px' }}>
            <CircularProgress />
          </Box>
        </Container>
      </DashboardContent>
    );
  }

  return (
    <DashboardContent>
      <Container>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
          <Typography variant="h4">
            بخش ارتودنسی
          </Typography>
          <Stack direction="row" spacing={1} alignItems="center">
            {/* View Mode Toggle */}
            <ToggleButtonGroup
              value={viewMode}
              exclusive
              onChange={(event, newViewMode) => {
                if (newViewMode !== null) {
                  setViewMode(newViewMode);
                }
              }}
              size="small"
              color="primary"
            >
              <ToggleButton value="card">
                <Iconify icon="solar:widget-2-bold" width={20} />
              </ToggleButton>
              <ToggleButton value="table">
                <Iconify icon="solar:list-bold" width={20} />
              </ToggleButton>
            </ToggleButtonGroup>
            
          <IconButton
            onClick={handleOpenAddDialog}
            sx={{
              borderRadius: '50%',
              width: 40,
              height: 40,
              padding: 0,
              display: 'inline-flex',
              alignItems: 'center',
              justifyContent: 'center'
            }}
            aria-label="add-patient"
          >
            <Iconify icon="mingcute:add-line" />
          </IconButton>
          </Stack>
        </Box>

        {error && (
          <Typography variant="body1" color="error" sx={{ mb: 4 }}>
            {error}
          </Typography>
        )}

        {!error && !loading && patients.length === 0 && (
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Typography variant="h6" color="text.secondary" sx={{ mb: 2 }}>
              هیچ بیماری یافت نشد
            </Typography>
            <Typography variant="body2" color="text.secondary">
              برای افزودن بیمار جدید، روی دکمه + کلیک کنید
            </Typography>
          </Box>
        )}

        {/* Card View */}
        {viewMode === 'card' && (
        <Grid container spacing={3}>
          {patients.map((patient) => {
            // Use profileImage from patient data (already fetched)
            const { profileImage } = patient;

            return (
              <Grid item xs={12} sm={6} md={4} key={patient.id}>
                <Card
                  sx={{ textAlign: 'center', cursor: 'pointer' }}
                  role="button"
                  tabIndex={0}
                  onClick={() => { navigate(`/dashboard/orthodontics/patient/${patient.id}`); }}
                  onKeyDown={(e) => { if (e.key === 'Enter') { navigate(`/dashboard/orthodontics/patient/${patient.id}`); } }}
                >
                  <Box sx={{ position: 'relative' }}>
                    <AvatarShape
                      sx={{
                        left: 0,
                        right: 0,
                        zIndex: 10,
                        mx: 'auto',
                        bottom: -26,
                        position: 'absolute',
                      }}
                    />

                    <Avatar
                      alt={patient.firstName}
                      src={profileImage ? getImageUrl(profileImage.path) : undefined}
                      sx={{
                        width: 64,
                        height: 64,
                        zIndex: 11,
                        left: 0,
                        right: 0,
                        bottom: -32,
                        mx: 'auto',
                        position: 'absolute',
                        cursor: 'pointer'
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        navigate(`/dashboard/orthodontics/patient/${patient.id}`);
                      }}
                    />

                    <Image
                      src={`${CONFIG.site.serverUrl}/assets/images/cover/cover-2.webp`}
                      alt="Card cover"
                      ratio="16/9"
                      slotProps={{
                        overlay: {
                          background: (theme) => varAlpha(theme.vars.palette.grey['900Channel'], 0.48),
                        },
                      }}
                      sx={{
                        height: 180,
                        borderRadius: 0,
                      }}
                    />

                  </Box>

                  <ListItemText
                    sx={{ mt: 7, mb: 1 }}
                    primary={`${patient.firstName} ${patient.lastName || ''}`}
                    primaryTypographyProps={{ typography: 'subtitle1' }}
                    secondaryTypographyProps={{ component: 'span', mt: 0.5 }}
                  />


                  <Divider sx={{ borderStyle: 'dashed' }} />

                  <Box
                    display="grid"
                    gridTemplateColumns="repeat(3, 1fr)"
                    sx={{ py: 3, typography: 'subtitle1' }}
                  >
                    <div>
                      <Typography variant="caption" component="div" sx={{ mb: 0.5, color: 'text.secondary' }}>
                        سن
                      </Typography>
                      {patient.age || '-'}
                    </div>

                    <div>
                      <Typography variant="caption" component="div" sx={{ mb: 0.5, color: 'text.secondary' }}>
                        وضعیت
                      </Typography>
                      <Typography variant="caption" sx={{ fontSize: '0.7rem !important' }}>
                        {statusMap[patient.status] || 'مشخص نشده'}
                      </Typography>
                    </div>

                    <div>
                      <Typography variant="caption" component="div" sx={{ mb: 0.5, color: 'text.secondary' }}>
                        تاریخ افزودن
                      </Typography>
                      <Typography variant="caption" sx={{ fontSize: '0.7rem !important' }}>
                        {new Date(patient.createdAt).toLocaleDateString('fa-IR')}
                      </Typography>
                    </div>
                  </Box>
                </Card>
              </Grid>
            );
          })}
        </Grid>
        )}

        {/* Table View */}
        {viewMode === 'table' && (
          <Paper sx={{ width: '100%', borderRadius: '16px', overflow: 'hidden' }}>
            <TableContainer>
              <Table>
                <TableHead>
                  <TableRow sx={{ bgcolor: 'background.neutral' }}>
                    <TableCell>نام بیمار</TableCell>
                    <TableCell>وضعیت</TableCell>
                    <TableCell>تاریخ ثبت</TableCell>
                    <TableCell align="center">عملیات</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {loading ? (
                    <TableRow>
                      <TableCell colSpan={4} align="center" sx={{ py: 3 }}>
                        <Typography variant="body2">در حال بارگیری...</Typography>
                      </TableCell>
                    </TableRow>
                  ) : patients.length === 0 ? (
                    <TableRow>
                      <TableCell colSpan={4} align="center" sx={{ py: 3 }}>
                        <Typography variant="body2">بیماری یافت نشد</Typography>
                      </TableCell>
                    </TableRow>
                  ) : (
                    patients
                      .slice(tablePage * tableRowsPerPage, tablePage * tableRowsPerPage + tableRowsPerPage)
                      .map((patient) => {
                        // Use profileImage from patient data (already fetched)
                        const { profileImage } = patient;
                        
                        return (
                          <TableRow
                            key={patient.id}
                            hover
                            sx={{ cursor: 'pointer' }}
                            onClick={() => {
                              navigate(`/dashboard/orthodontics/patient/${patient.id}`);
                            }}
                          >
                            <TableCell>
                              <Stack direction="row" spacing={2} alignItems="center">
                                <Avatar
                                  src={profileImage 
                                    ? getImageUrl(profileImage.path)
                                    : undefined
                                  }
                                  sx={{ width: 32, height: 32 }}
                                >
                                  {patient.firstName[0]}
                                </Avatar>
                                <Typography variant="body2">
                                  {`${patient.firstName} ${patient.lastName || ''}`}
                                </Typography>
                              </Stack>
                            </TableCell>
                            <TableCell>
                              <Chip 
                                label={statusMap[patient.status] || 'مشخص نشده'}
                                size="small"
                                color={
                                  patient.status === 'IN_TREATMENT' ? 'primary' :
                                  patient.status === 'COMPLETED' ? 'success' :
                                  patient.status === 'CANCELLED' ? 'error' : 'default'
                                }
                                sx={{ fontSize: '0.75rem' }}
                              />
                            </TableCell>
                            <TableCell>
                              <Typography variant="caption">
                                {new Date(patient.createdAt).toLocaleDateString('fa-IR')}
                              </Typography>
                            </TableCell>
                            <TableCell align="center">
                              <Stack direction="row" spacing={0.5} justifyContent="center">
                                <Tooltip title="ویرایش">
                                  <IconButton
                                    size="small"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleOpenEditDialog(patient);
                                    }}
                                  >
                                    <Iconify icon="solar:pen-bold" width={18} />
                                  </IconButton>
                                </Tooltip>
                                <Tooltip title="حذف">
                                  <IconButton
                                    size="small"
                                    color="error"
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      handleOpenDeleteDialog(patient);
                                    }}
                                  >
                                    <Iconify icon="solar:trash-bin-trash-bold" width={18} />
                                  </IconButton>
                                </Tooltip>
                              </Stack>
                            </TableCell>
                          </TableRow>
                        );
                      })
                  )}
                </TableBody>
              </Table>
            </TableContainer>
            <TablePagination
              component="div"
              count={patients.length}
              page={tablePage}
              onPageChange={(event, newPage) => setTablePage(newPage)}
              rowsPerPage={tableRowsPerPage}
              onRowsPerPageChange={(event) => {
                setTableRowsPerPage(parseInt(event.target.value, 10));
                setTablePage(0);
              }}
              rowsPerPageOptions={[5, 10, 25, 50]}
              labelRowsPerPage="تعداد ردیف در صفحه:"
              labelDisplayedRows={({ from, to, count }) => `${from}-${to} از ${count}`}
            />
          </Paper>
        )}

        {/* Add Patient Dialog */}
        <Dialog open={openAddDialog} onClose={handleCloseAddDialog} maxWidth="sm" fullWidth>
          <DialogTitle>افزودن بیمار جدید</DialogTitle>
          <Form methods={addForm} onSubmit={addForm.handleSubmit(handleAddPatient)}>
            <DialogContent>
              <Grid container spacing={2} sx={{ mt: 1 }}>
                <Grid item xs={12}>
                  <Field.Text name="firstName" label="نام" />
                </Grid>
                <Grid item xs={12}>
                  <Field.Text name="age" label="سن" type="number" />
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

        {/* Delete Confirm Dialog */}
        <ConfirmDialog
          open={openDeleteDialog}
          onClose={handleCloseDeleteDialog}
          title="حذف بیمار"
          content={`آیا از حذف بیمار "${patientToDelete?.firstName} ${patientToDelete?.lastName || ''}" مطمئن هستید؟`}
          action={
            <Button
              variant="contained"
              color="error"
              onClick={async () => {
                if (patientToDelete) {
                  await handleDeletePatient(patientToDelete.id);
                }
                handleCloseDeleteDialog();
              }}
            >
              حذف
            </Button>
          }
        />
      </Container>
    </DashboardContent>
  );
}
