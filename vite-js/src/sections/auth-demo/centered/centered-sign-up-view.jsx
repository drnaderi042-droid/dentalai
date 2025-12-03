import { z as zod } from 'zod';
import { useForm } from 'react-hook-form';
import { useNavigate } from 'react-router-dom';
import { zodResolver } from '@hookform/resolvers/zod';

import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';
import InputAdornment from '@mui/material/InputAdornment';

import { paths } from 'src/routes/paths';
import { RouterLink } from 'src/routes/components';

import { useBoolean } from 'src/hooks/use-boolean';

import { Iconify } from 'src/components/iconify';
import { AnimateLogo2 } from 'src/components/animate';
import { Form, Field } from 'src/components/hook-form';

import axios, { endpoints } from 'src/utils/axios';
import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

export const SignUpSchema = zod.object({
  firstName: zod.string().min(1, { message: 'نام الزامی است!' }),
  lastName: zod.string().min(1, { message: 'نام خانوادگی الزامی است!' }),
  email: zod
    .string()
    .min(1, { message: 'ایمیل الزامی است!' })
    .email({ message: 'ایمیل باید معتبر باشد!' }),
  phone: zod.string().optional(),
  password: zod
    .string()
    .min(1, { message: 'رمز عبور الزامی است!' })
    .min(6, { message: 'رمز عبور باید حداقل ۶ کاراکتر باشد!' }),

  specialty: zod.string().optional(),
  licenseNumber: zod.string().optional(),
});

// ----------------------------------------------------------------------

export function CenteredSignUpView() {
  const password = useBoolean();
  const navigate = useNavigate();
  const { login } = useAuthContext();

  const defaultValues = {
    firstName: '',
    lastName: '',
    email: '',
    phone: '',
    password: '',
    specialty: '',
    licenseNumber: '',
  };

  const methods = useForm({
    resolver: zodResolver(SignUpSchema),
    defaultValues,
  });

  const {
    handleSubmit,
    formState: { isSubmitting },
    watch,
  } = methods;

  const onSubmit = handleSubmit(async (data) => {
    // Set default role as DOCTOR for medical professionals
    const registrationData = {
      email: data.email,
      password: data.password,
      firstName: data.firstName,
      lastName: data.lastName,
      role: 'DOCTOR' // Changed to DOCTOR for medical professionals
    };

    try {
      await axios.post(endpoints.auth.signUp, registrationData);

      // Auto-login after successful registration
      await login?.(data.email, data.password);

      // Redirect to dashboard
      navigate(paths.dashboard.root);
    } catch (error) {
      console.error('Registration error:', error);
      // Show user-friendly error message
      const errorMessage = error.message || 'خطا در ثبت نام. لطفاً دوباره تلاش کنید.';
      alert(`خطا در ثبت نام: ${errorMessage}`);
    }
  });

  const renderLogo = <AnimateLogo2 sx={{ mb: 3, mx: 'auto' }} />;

  const renderHead = (
    <Stack alignItems="center" spacing={1.5} sx={{ mb: 5 }}>
      <Typography variant="h5">ثبت نام پزشکان</Typography>

      <Stack direction="row" spacing={0.5}>
        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          حساب کاربری دارید؟
        </Typography>

        <Link component={RouterLink} href={paths.authDemo.centered.signIn} variant="subtitle2">
          وارد شوید
        </Link>
      </Stack>
    </Stack>
  );

  const renderForm = (
    <Stack spacing={3}>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
        <Field.Text name="firstName" label="نام" InputLabelProps={{ shrink: true }} />
        <Field.Text name="lastName" label="نام خانوادگی" InputLabelProps={{ shrink: true }} />
      </Stack>

      <Field.Text name="email" label="ایمیل" InputLabelProps={{ shrink: true }} />

      <Field.Text name="phone" label="شماره تلفن" InputLabelProps={{ shrink: true }} />

      <Field.Text
        name="password"
        label="رمز عبور"
        placeholder="۶ کاراکتر یا بیشتر"
        type={password.value ? 'text' : 'password'}
        InputLabelProps={{ shrink: true }}
        InputProps={{
          endAdornment: (
            <InputAdornment position="end">
              <IconButton onClick={password.onToggle} edge="end">
                <Iconify icon={password.value ? 'solar:eye-bold' : 'solar:eye-closed-bold'} />
              </IconButton>
            </InputAdornment>
          ),
        }}
      />

      <LoadingButton
        fullWidth
        color="inherit"
        size="large"
        type="submit"
        variant="contained"
        loading={isSubmitting}
        loadingIndicator="ایجاد حساب..."
      >
        ایجاد حساب
      </LoadingButton>
    </Stack>
  );

  return (
    <>
      {renderLogo}

      {renderHead}

      <Form methods={methods} onSubmit={onSubmit}>
        {renderForm}
      </Form>
    </>
  );
}
