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
    // Set default role as PATIENT since role field is removed
    const registrationData = { ...data, role: 'PATIENT' };

    try {
      const response = await fetch('http://localhost:7272/api/auth/sign-up', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(registrationData),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.message || 'Registration failed');
      }

      // Auto-login after successful registration
      await login?.(data.email, data.password);

      // Redirect to dashboard
      navigate(paths.dashboard.root);
    } catch (error) {
      console.error('Registration error:', error);
      // You might want to show an error message to the user here
      alert(`خطا در ثبت نام: ${error.message}`);
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
