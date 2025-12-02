import { z as zod } from 'zod';
import { useMemo } from 'react';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm, Controller } from 'react-hook-form';
import { isValidPhoneNumber } from 'react-phone-number-input/input';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Switch from '@mui/material/Switch';
import Grid from '@mui/material/Unstable_Grid2';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';
import FormControlLabel from '@mui/material/FormControlLabel';

import { paths } from 'src/routes/paths';
import { useRouter } from 'src/routes/hooks';

import { fData } from 'src/utils/format-number';
import axiosInstance, { endpoints } from 'src/utils/axios';

import { Label } from 'src/components/label';
import { toast } from 'src/components/snackbar';
import { Form, Field, schemaHelper } from 'src/components/hook-form';

// ----------------------------------------------------------------------

export const NewUserSchema = zod.object({
  avatarUrl: schemaHelper.file({
    message: { required_error: 'تصویر پروفایل الزامی است!' },
  }).optional(),
  name: zod.string().min(1, { message: 'نام الزامی است!' }),
  email: zod
    .string()
    .min(1, { message: 'ایمیل الزامی است!' })
    .email({ message: 'ایمیل باید معتبر باشد!' }),
  phoneNumber: schemaHelper.phoneNumber({ isValidPhoneNumber }).optional(),
  country: schemaHelper.objectOrNull({
    message: { required_error: 'کشور الزامی است!' },
  }).optional(),
  address: zod.string().optional(),
  company: zod.string().optional(),
  state: zod.string().optional(),
  city: zod.string().optional(),
  role: zod.string().min(1, { message: 'نقش کاربری الزامی است!' }),
  zipCode: zod.string().optional(),
  // Not required
  status: zod.string().optional(),
  isVerified: zod.boolean().optional(),
});

// ----------------------------------------------------------------------

export function UserNewEditForm({ currentUser }) {
  const router = useRouter();

  const defaultValues = useMemo(
    () => ({
      status: currentUser?.status || 'active',
      avatarUrl: currentUser?.avatarUrl || currentUser?.avatar || null,
      isVerified: currentUser?.isVerified !== undefined ? currentUser?.isVerified : true,
      name: currentUser?.name || currentUser?.firstName || '',
      email: currentUser?.email || '',
      phoneNumber: currentUser?.phoneNumber || currentUser?.phone || '',
      country: currentUser?.country || null,
      state: currentUser?.state || '',
      city: currentUser?.city || '',
      address: currentUser?.address || '',
      zipCode: currentUser?.zipCode || '',
      company: currentUser?.company || '',
      role: currentUser?.role || '',
    }),
    [currentUser]
  );

  const methods = useForm({
    mode: 'onSubmit',
    resolver: zodResolver(NewUserSchema),
    defaultValues,
  });

  const {
    reset,
    watch,
    control,
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const values = watch();

  const onSubmit = handleSubmit(async (data) => {
    try {
      if (currentUser) {
        // Update existing user
        await axiosInstance.put(endpoints.user.update(currentUser.id), data);
        toast.success('اطلاعات کاربر با موفقیت به‌روزرسانی شد!');
      } else {
        // Create new user
        await axiosInstance.post(endpoints.user.list, data);
        toast.success('کاربر جدید با موفقیت ایجاد شد!');
      }
      router.push(paths.dashboard.user.list);
    } catch (error) {
      console.error('Error saving user:', error);
      toast.error(error.message || 'خطا در ذخیره اطلاعات کاربر');
    }
  });

  const handleDelete = async () => {
    if (!currentUser?.id) return;
    
    if (window.confirm('آیا از حذف این کاربر مطمئن هستید؟ این عمل غیرقابل بازگشت است.')) {
      try {
        await axiosInstance.delete(endpoints.user.delete(currentUser.id));
        toast.success('کاربر با موفقیت حذف شد!');
        router.push(paths.dashboard.user.list);
      } catch (error) {
        console.error('Error deleting user:', error);
        toast.error(error.message || 'خطا در حذف کاربر');
      }
    }
  };

  return (
    <Form methods={methods} onSubmit={onSubmit}>
      <Grid container spacing={3}>
        <Grid xs={12} md={4}>
          <Card sx={{ pt: 10, pb: 5, px: 3 }}>
            {currentUser && (
              <Label
                color={
                  (values.status === 'active' && 'success') ||
                  (values.status === 'banned' && 'error') ||
                  'warning'
                }
                sx={{ position: 'absolute', top: 24, right: 24 }}
              >
                {values.status === 'active' ? 'فعال' : values.status === 'banned' ? 'مسدود' : values.status}
              </Label>
            )}

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

            {currentUser && (
              <FormControlLabel
                labelPlacement="start"
                control={
                  <Controller
                    name="status"
                    control={control}
                    render={({ field }) => (
                      <Switch
                        {...field}
                        checked={field.value !== 'active'}
                        onChange={(event) =>
                          field.onChange(event.target.checked ? 'banned' : 'active')
                        }
                      />
                    )}
                  />
                }
                label={
                  <>
                    <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                      مسدود کردن
                    </Typography>
                    <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                      غیرفعال کردن حساب کاربری
                    </Typography>
                  </>
                }
                sx={{
                  mx: 0,
                  mb: 3,
                  width: 1,
                  justifyContent: 'space-between',
                }}
              />
            )}

            <Field.Switch
              name="isVerified"
              labelPlacement="start"
              label={
                <>
                  <Typography variant="subtitle2" sx={{ mb: 0.5 }}>
                    ایمیل تأیید شده
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    غیرفعال کردن این گزینه به طور خودکار ایمیل تأیید برای کاربر ارسال می‌کند
                  </Typography>
                </>
              }
              sx={{ mx: 0, width: 1, justifyContent: 'space-between' }}
            />

            {currentUser && (
              <Stack justifyContent="center" alignItems="center" sx={{ mt: 3 }}>
                <Button variant="soft" color="error" onClick={handleDelete}>
                  حذف کاربر
                </Button>
              </Stack>
            )}
          </Card>
        </Grid>

        <Grid xs={12} md={8}>
          <Card sx={{ p: 3 }}>
            <Box
              rowGap={3}
              columnGap={2}
              display="grid"
              gridTemplateColumns={{
                xs: 'repeat(1, 1fr)',
                sm: 'repeat(2, 1fr)',
              }}
            >
              <Field.Text name="name" label="نام کامل" />
              <Field.Text name="email" label="آدرس ایمیل" />
              <Field.Phone name="phoneNumber" label="شماره تلفن" />

              <Field.CountrySelect
                fullWidth
                name="country"
                label="کشور"
                placeholder="انتخاب کشور"
              />

              <Field.Text name="state" label="استان/منطقه" />
              <Field.Text name="city" label="شهر" />
              <Field.Text name="address" label="آدرس" />
              <Field.Text name="zipCode" label="کد پستی" />
              <Field.Text name="company" label="شرکت" />
            </Box>

            <Stack alignItems="flex-end" sx={{ mt: 3 }}>
              <LoadingButton type="submit" variant="contained" loading={isSubmitting}>
                {!currentUser ? 'ایجاد کاربر' : 'ذخیره تغییرات'}
              </LoadingButton>
            </Stack>
          </Card>
        </Grid>
      </Grid>
    </Form>
  );
}
