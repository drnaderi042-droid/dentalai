import { z as zod } from 'zod';
import { useForm } from 'react-hook-form';
import { useState, useEffect } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import MenuItem from '@mui/material/MenuItem';
import Grid from '@mui/material/Unstable_Grid2';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';

import axios from 'src/utils/axios';
import { fData } from 'src/utils/format-number';
import { fDateJalali } from 'src/utils/format-time';

import { toast } from 'src/components/snackbar';
import { Form, Field } from 'src/components/hook-form';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

export const UpdateUserSchema = zod.object({
  firstName: zod.string().min(1, { message: 'نام الزامی است!' }),
  lastName: zod.string().min(1, { message: 'نام خانوادگی الزامی است!' }),
  email: zod
    .string()
    .min(1, { message: 'ایمیل الزامی است!' })
    .email({ message: 'ایمیل معتبر نیست!' }),
  phone: zod.string().optional(),
  licenseNumber: zod.string().optional(),
  specialty: zod.string().optional(),
  city: zod.string().optional(),
  province: zod.string().optional(),
  avatarUrl: zod.any().optional(),
});

// Specialty options for doctors
const SPECIALTY_OPTIONS = [
  'ارتودنسی',
  'پروستودونتیکس',
  'پریودونتیکس',
  'دندانپزشکی عمومی',
  'دندانپزشکی کودکان',
  'جراحی دهان و فک',
];

// Province options for Iran
const PROVINCE_OPTIONS = [
  'آذربایجان شرقی',
  'آذربایجان غربی',
  'اردبیل',
  'اصفهان',
  'البرز',
  'ایلام',
  'بوشهر',
  'تهران',
  'چهارمحال و بختیاری',
  'خراسان جنوبی',
  'خراسان رضوی',
  'خراسان شمالی',
  'خوزستان',
  'زنجان',
  'سمنان',
  'سیستان و بلوچستان',
  'فارس',
  'قزوین',
  'قم',
  'کردستان',
  'کرمان',
  'کرمانشاه',
  'کهگیلویه و بویراحمد',
  'گلستان',
  'گیلان',
  'لرستان',
  'مازندران',
  'مرکزی',
  'هرمزگان',
  'همدان',
  'یزد',
];

export function AccountGeneral() {
  const { user, checkUserSession } = useAuthContext();
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchUserProfile = async () => {
      if (user?.accessToken) {
        try {
          const response = await axios.get('/api/auth/me', {
            headers: {
              Authorization: `Bearer ${user.accessToken}`,
            },
          });
          setUserProfile(response.data.user);
        } catch (error) {
          console.error('Error fetching user profile:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchUserProfile();
  }, [user]);

  const defaultValues = {
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    licenseNumber: '',
    specialty: '',
    city: '',
    province: '',
    avatarUrl: null,
  };

  const methods = useForm({
    mode: 'all',
    resolver: zodResolver(UpdateUserSchema),
    defaultValues,
  });

  const {
    handleSubmit,
    formState: { isSubmitting },
    reset,
  } = methods;

  // Update form values when userProfile is loaded
  useEffect(() => {
    if (userProfile && !loading) {
      const formValues = {
        firstName: userProfile.firstName || '',
        lastName: userProfile.lastName || '',
        email: userProfile.email || '',
        phone: userProfile.phone || '',
        licenseNumber: userProfile.licenseNumber || '',
        specialty: userProfile.specialty || '',
        city: userProfile.city || '',
        province: userProfile.province || '',
        avatarUrl: userProfile.avatarUrl || userProfile.avatar || null,
      };
      
      reset(formValues, {
        keepDefaultValues: false,
        keepErrors: false,
      });
    }
  }, [userProfile, loading, reset]);

  const onSubmit = handleSubmit(async (data) => {
    try {
      // If avatar is a File, use FormData, otherwise use JSON
      const hasAvatarFile = data.avatarUrl instanceof File;
      
      let requestData;
      const headers = {
        Authorization: `Bearer ${user?.accessToken}`,
      };

      if (hasAvatarFile) {
        // Use FormData for file upload
        const formData = new FormData();
        formData.append('firstName', data.firstName);
        formData.append('lastName', data.lastName);
        formData.append('email', data.email);
        if (data.phone) formData.append('phone', data.phone);
        if (data.licenseNumber) formData.append('licenseNumber', data.licenseNumber);
        if (data.specialty) formData.append('specialty', data.specialty);
        if (data.city) formData.append('city', data.city);
        if (data.province) formData.append('province', data.province);
        formData.append('avatar', data.avatarUrl);
        
        requestData = formData;
        // Don't set Content-Type header for FormData - browser will set it automatically with boundary
      } else {
        // Use JSON for regular updates
        requestData = {
          firstName: data.firstName,
          lastName: data.lastName,
          email: data.email,
          phone: data.phone || null,
          licenseNumber: data.licenseNumber || null,
          specialty: data.specialty || null,
          city: data.city || null,
          province: data.province || null,
        };
        headers['Content-Type'] = 'application/json';
      }

      console.log('Updating profile:', { hasAvatarFile, data: requestData });

      const response = await axios.put('/api/auth/profile', requestData, {
        headers,
      });

      console.log('Profile update response:', response.data);

      const updatedUser = response.data.user;
      setUserProfile(updatedUser);
      
      // Update auth context with new user data
      if (checkUserSession) {
        await checkUserSession();
      }
      
      toast.success('پروفایل با موفقیت بروزرسانی شد!');
    } catch (error) {
      console.error('Error updating profile:', error);
      const errorMessage = error.response?.data?.message || error.message || 'خطا در بروزرسانی پروفایل';
      toast.error(errorMessage);
    }
  });

  if (loading) {
    return (
      <Card sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6">بارگیری اطلاعات پروفایل...</Typography>
      </Card>
    );
  }

  if (!userProfile) {
    return (
      <Card sx={{ p: 3, textAlign: 'center' }}>
        <Typography variant="h6" color="error">
          خطا در بارگیری اطلاعات پروفایل
        </Typography>
      </Card>
    );
  }

  return (
    <Form methods={methods} onSubmit={onSubmit}>
      <Grid container spacing={3}>
        <Grid xs={12} md={4}>
          <Card
            sx={{
              pt: 10,
              pb: 5,
              px: 3,
              textAlign: 'center',
            }}
          >
            <Box sx={{ mb: 5 }}>
              <Field.UploadAvatar
                name="avatarUrl"
                maxSize={3145728}
                helperText={
                  <Typography
                    variant="caption"
                    sx={{
                      mt: 3,
                      mx: 'auto',
                      display: 'block',
                      textAlign: 'center',
                      color: 'text.disabled',
                    }}
                  >
                    فرمت‌های مجاز: *.jpeg, *.jpg, *.png, *.gif
                    <br /> حداکثر حجم: {fData(3145728)}
                  </Typography>
                }
              />
            </Box>



            <Stack spacing={2} sx={{ mb: 3 }}>
              <Typography variant="body1">
                <strong>نقش:</strong> {
                  userProfile.role === 'DOCTOR' || userProfile.role === 'doctor' ? 'دکتر' : 
                  userProfile.role === 'ADMIN' || userProfile.role === 'admin' ? 'ادمین' : 
                  'بیمار'
                }
              </Typography>

              <Typography variant="body2" color="text.secondary">
                تاریخ عضویت: {userProfile.createdAt ? fDateJalali(userProfile.createdAt, 'jYYYY/jMM/jDD') : 'نامشخص'}
              </Typography>

              {userProfile.isVerified && (
                <Typography variant="body2" color="success.main">
                  ✓ حساب تایید شده
                </Typography>
              )}
            </Stack>
          </Card>
        </Grid>

        <Grid xs={12} md={8}>
          <Card sx={{ p: 3 }}>
            <Typography variant="h6" sx={{ mb: 3 }}>
              اطلاعات شخصی
            </Typography>

            <Box
              rowGap={3}
              columnGap={2}
              display="grid"
              gridTemplateColumns={{
                xs: 'repeat(1, 1fr)',
                sm: 'repeat(2, 1fr)',
              }}
            >
              <Field.Text name="firstName" label="نام" />
              <Field.Text name="lastName" label="نام خانوادگی" />
              <Field.Text name="email" label="آدرس ایمیل" />

              <Field.Phone name="phone" label="شماره تلفن" />

              <Field.Select name="province" label="استان">
                <MenuItem value="">
                  <em>انتخاب کنید</em>
                </MenuItem>
                {PROVINCE_OPTIONS.map((option) => (
                  <MenuItem key={option} value={option}>
                    {option}
                  </MenuItem>
                ))}
              </Field.Select>

              <Field.Text name="city" label="شهر" />

              {/* Doctor-specific fields */}
              {(userProfile.role === 'DOCTOR' || userProfile.role === 'doctor') && (
                <>
                  <Field.Text name="licenseNumber" label="شماره نظام پزشکی" />
                  <Field.Select name="specialty" label="تخصص">
                    <MenuItem value="">
                      <em>انتخاب کنید</em>
                    </MenuItem>
                    {SPECIALTY_OPTIONS.map((option) => (
                      <MenuItem key={option} value={option}>
                        {option}
                      </MenuItem>
                    ))}
                  </Field.Select>
                </>
              )}
            </Box>

            <Stack spacing={3} alignItems="flex-end" sx={{ mt: 3 }}>
              <LoadingButton
                type="submit"
                variant="contained"
                loading={isSubmitting}
                sx={{ minWidth: 120 }}
              >
                ذخیره تغییرات
              </LoadingButton>
            </Stack>
          </Card>
        </Grid>
      </Grid>
    </Form>
  );
}
