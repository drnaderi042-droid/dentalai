import { useFormContext } from 'react-hook-form';

import Stack from '@mui/material/Stack';
import MenuItem from '@mui/material/MenuItem';

import { Field } from 'src/components/hook-form';

// ----------------------------------------------------------------------

export function InvoiceNewEditStatusDate() {
  const { watch } = useFormContext();

  const values = watch();

  return (
    <Stack
      spacing={2}
      direction={{ xs: 'column', sm: 'row' }}
      sx={{ p: 3, bgcolor: 'background.neutral' }}
    >
      <Field.Text
        disabled
        name="invoiceNumber"
        label="شماره فاکتور"
        value={values.invoiceNumber}
      />

      <Field.Select fullWidth name="status" label="وضعیت" InputLabelProps={{ shrink: true }}>
        {[
          { value: 'paid', label: 'پرداخت شده' },
          { value: 'pending', label: 'در انتظار' },
          { value: 'overdue', label: 'سررسید گذشته' },
          { value: 'draft', label: 'پیش‌نویس' },
        ].map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </Field.Select>

      <Field.DatePicker name="createDate" label="تاریخ ایجاد" />
      <Field.DatePicker name="dueDate" label="تاریخ سررسید" />
    </Stack>
  );
}
