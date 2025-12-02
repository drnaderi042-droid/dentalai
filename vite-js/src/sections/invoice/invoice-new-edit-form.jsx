import { z as zod } from 'zod';
import { useMemo } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';

import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import LoadingButton from '@mui/lab/LoadingButton';

import { paths } from 'src/routes/paths';
import { useRouter } from 'src/routes/hooks';

import { useBoolean } from 'src/hooks/use-boolean';

import { today, fIsAfter } from 'src/utils/format-time';
import axiosInstance, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Form, schemaHelper } from 'src/components/hook-form';

import { useAuthContext } from 'src/auth/hooks';

import { InvoiceNewEditDetails } from './invoice-new-edit-details';
import { InvoiceNewEditAddress } from './invoice-new-edit-address';
import { InvoiceNewEditStatusDate } from './invoice-new-edit-status-date';

export const NewInvoiceSchema = zod
  .object({
    invoiceTo: zod.custom().refine((data) => data !== null, { message: 'مشتری الزامی است!' }),
    createDate: schemaHelper.date({ message: { required_error: 'تاریخ ایجاد الزامی است!' } }),
    dueDate: schemaHelper.date({ message: { required_error: 'تاریخ سررسید الزامی است!' } }),
    items: zod.array(
      zod.object({
        title: zod.string().min(1, { message: 'عنوان الزامی است!' }),
        service: zod.string().min(1, { message: 'خدمات الزامی است!' }),
        quantity: zod.number().min(1, { message: 'تعداد باید بیشتر از 0 باشد' }),
        // Not required
        price: zod.number(),
        total: zod.number(),
        description: zod.string(),
      })
    ),
    // Not required
    taxes: zod.number(),
    status: zod.string(),
    discount: zod.number(),
    shipping: zod.number(),
    totalAmount: zod.number(),
    invoiceNumber: zod.string(),
    invoiceFrom: zod.custom().nullable(),
  })
  .refine((data) => !fIsAfter(data.createDate, data.dueDate), {
    message: 'تاریخ سررسید نمی‌تواند زودتر از تاریخ ایجاد باشد!',
    path: ['dueDate'],
  });

// ----------------------------------------------------------------------

export function InvoiceNewEditForm({ currentInvoice }) {
  const router = useRouter();
  const { user } = useAuthContext();

  const loadingSave = useBoolean();

  const loadingSend = useBoolean();

  const isAdmin = user?.role?.toUpperCase() === 'ADMIN';

  const defaultValues = useMemo(
    () => ({
      invoiceNumber: currentInvoice?.invoiceNumber || 'INV-1990',
      createDate: currentInvoice?.createDate || today(),
      dueDate: currentInvoice?.dueDate || null,
      taxes: currentInvoice?.taxes || 0,
      shipping: currentInvoice?.shipping || 0,
      status: currentInvoice?.status || 'draft',
      discount: currentInvoice?.discount || 0,
      invoiceFrom: currentInvoice?.invoiceFrom || null,
      invoiceTo: currentInvoice?.invoiceTo || null,
      totalAmount: currentInvoice?.totalAmount || 0,
      items: currentInvoice?.items || [
        {
          title: '',
          description: '',
          service: '',
          quantity: 1,
          price: 0,
          total: 0,
        },
      ],
    }),
    [currentInvoice]
  );

  const methods = useForm({
    mode: 'all',
    resolver: zodResolver(NewInvoiceSchema),
    defaultValues,
  });

  const {
    reset,
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const handleSaveAsDraft = handleSubmit(async (data) => {
    loadingSave.onTrue();

    try {
      const payload = {
        ...data,
        status: 'draft',
        createDate: data.createDate?.toISOString(),
        dueDate: data.dueDate?.toISOString(),
      };

      if (currentInvoice) {
        await axiosInstance.put(endpoints.invoice.update(currentInvoice.id), payload);
        toast.success('فاکتور با موفقیت به‌روزرسانی شد!');
      } else {
        await axiosInstance.post(endpoints.invoice.create, payload);
        toast.success('فاکتور با موفقیت ایجاد شد!');
      }
      
      loadingSave.onFalse();
      router.push(paths.dashboard.invoice.root);
    } catch (error) {
      console.error(error);
      toast.error(error.message || 'خطا در ذخیره فاکتور');
      loadingSave.onFalse();
    }
  });

  const handleCreateAndSend = handleSubmit(async (data) => {
    loadingSend.onTrue();

    try {
      const payload = {
        ...data,
        status: data.status || 'pending',
        sentDate: new Date().toISOString(),
        createDate: data.createDate?.toISOString(),
        dueDate: data.dueDate?.toISOString(),
      };

      if (currentInvoice) {
        await axiosInstance.put(endpoints.invoice.update(currentInvoice.id), payload);
        toast.success('فاکتور با موفقیت به‌روزرسانی و ارسال شد!');
      } else {
        await axiosInstance.post(endpoints.invoice.create, payload);
        toast.success('فاکتور با موفقیت ایجاد و ارسال شد!');
      }
      
      loadingSend.onFalse();
      router.push(paths.dashboard.invoice.root);
    } catch (error) {
      console.error(error);
      toast.error(error.message || 'خطا در ایجاد و ارسال فاکتور');
      loadingSend.onFalse();
    }
  });

  return (
    <Form methods={methods}>
      <Card>
        <InvoiceNewEditAddress />

        <InvoiceNewEditStatusDate />

        <InvoiceNewEditDetails />
      </Card>

      <Stack justifyContent="flex-end" direction="row" spacing={2} sx={{ mt: 3 }}>
        <LoadingButton
          color="inherit"
          size="large"
          variant="outlined"
          loading={loadingSave.value && isSubmitting}
          onClick={handleSaveAsDraft}
        >
          ذخیره به عنوان پیش‌نویس
        </LoadingButton>

        {isAdmin && (
          <LoadingButton
            size="large"
            variant="contained"
            loading={loadingSend.value && isSubmitting}
            onClick={handleCreateAndSend}
          >
            {currentInvoice ? 'به‌روزرسانی' : 'ایجاد'} و ارسال
          </LoadingButton>
        )}
      </Stack>
    </Form>
  );
}
