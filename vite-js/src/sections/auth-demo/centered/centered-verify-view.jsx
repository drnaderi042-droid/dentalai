import { z as zod } from 'zod';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import LoadingButton from '@mui/lab/LoadingButton';

import { paths } from 'src/routes/paths';
import { RouterLink } from 'src/routes/components';

import { EmailInboxIcon } from 'src/assets/icons';

import { Iconify } from 'src/components/iconify';
import { Form, Field } from 'src/components/hook-form';

// ----------------------------------------------------------------------

export const VerifySchema = zod.object({
  code: zod
    .string()
    .min(1, { message: 'کد الزامی است!' })
    .min(6, { message: 'کد باید حداقل ۶ کاراکتر باشد!' }),
  email: zod
    .string()
    .min(1, { message: 'ایمیل الزامی است!' })
    .email({ message: 'ایمیل باید معتبر باشد!' }),
});

// ----------------------------------------------------------------------

export function CenteredVerifyView() {
  const defaultValues = { code: '', email: '' };

  const methods = useForm({
    resolver: zodResolver(VerifySchema),
    defaultValues,
  });

  const {
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const onSubmit = handleSubmit(async (data) => {
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      console.info('DATA', data);
    } catch (error) {
      console.error(error);
    }
  });

  const renderHead = (
    <>
      <EmailInboxIcon sx={{ mx: 'auto' }} />

      <Stack spacing={1} sx={{ mt: 3, mb: 5, textAlign: 'center', whiteSpace: 'pre-line' }}>
        <Typography variant="h5">لطفاً ایمیل خود را بررسی کنید!</Typography>

        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
          {`ما یک کد تأیید ۶ رقمی به ایمیل شما ارسال کرده‌ایم. \nلطفاً کد را در کادر زیر وارد کنید تا ایمیل خود را تأیید کنید.`}
        </Typography>
      </Stack>
    </>
  );

  const renderForm = (
    <Stack spacing={3}>
      <Field.Text
        name="email"
        label="آدرس ایمیل"
        placeholder="example@gmail.com"
        InputLabelProps={{ shrink: true }}
      />

      <Field.Code name="code" />

      <LoadingButton
        fullWidth
        size="large"
        type="submit"
        variant="contained"
        loading={isSubmitting}
        loadingIndicator="در حال تأیید..."
      >
        تأیید
      </LoadingButton>

      <Typography variant="body2" sx={{ mx: 'auto' }}>
        {`کد ندارید؟ `}
        <Link variant="subtitle2" sx={{ cursor: 'pointer' }}>
          ارسال مجدد کد
        </Link>
      </Typography>

      <Link
        component={RouterLink}
        href={paths.authDemo.centered.signIn}
        color="inherit"
        variant="subtitle2"
        sx={{ mx: 'auto', alignItems: 'center', display: 'inline-flex' }}
      >
        <Iconify icon="eva:arrow-ios-back-fill" width={16} sx={{ mr: 0.5 }} />
        بازگشت به ورود
      </Link>
    </Stack>
  );

  return (
    <>
      {renderHead}

      <Form methods={methods} onSubmit={onSubmit}>
        {renderForm}
      </Form>
    </>
  );
}
