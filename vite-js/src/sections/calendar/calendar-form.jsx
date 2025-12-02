import { z as zod } from 'zod';
import { useCallback } from 'react';
import moment from 'moment-jalaali';
import { zodResolver } from '@hookform/resolvers/zod';
import { useForm, Controller } from 'react-hook-form';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Tooltip from '@mui/material/Tooltip';
import IconButton from '@mui/material/IconButton';
import LoadingButton from '@mui/lab/LoadingButton';
import DialogActions from '@mui/material/DialogActions';

import { uuidv4 } from 'src/utils/uuidv4';
import { fIsAfter } from 'src/utils/format-time';

import { createEvent, updateEvent, deleteEvent } from 'src/actions/calendar';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';
import { Scrollbar } from 'src/components/scrollbar';
import { Form, Field } from 'src/components/hook-form';
import { ColorPicker } from 'src/components/color-utils';

// ----------------------------------------------------------------------

export const EventSchema = zod.object({
  title: zod
    .string()
    .min(1, { message: 'عنوان الزامی است!' })
    .max(100, { message: 'عنوان باید کمتر از 100 کاراکتر باشد' }),
  description: zod
    .string()
    .optional()
    .default(''),
  // Not required
  color: zod.string(),
  allDay: zod.boolean(),
  start: zod.union([zod.string(), zod.number()]),
  end: zod.union([zod.string(), zod.number()]),
});

// ----------------------------------------------------------------------

export function CalendarForm({ currentEvent, colorOptions, onClose }) {
  const methods = useForm({
    mode: 'all',
    resolver: zodResolver(EventSchema),
    defaultValues: currentEvent,
  });

  const {
    reset,
    watch,
    control,
    handleSubmit,
    formState: { isSubmitting },
  } = methods;

  const values = watch();

  const dateError = fIsAfter(values.start, values.end);

  const onSubmit = handleSubmit(async (data) => {
    // The RHFMobileDateTimePicker already converts dates to ISO strings
    // So data.start and data.end should already be ISO strings
    let start = data?.start;
    let end = data?.end;
    
    // Ensure they are valid ISO strings
    if (start) {
      // If it's already a string, validate it
      if (typeof start === 'string') {
        const testMoment = moment(start);
        if (!testMoment.isValid()) {
          toast.error('تاریخ شروع نامعتبر است');
          return;
        }
        start = testMoment.toISOString();
      } else if (moment.isMoment(start)) {
        start = start.toISOString();
      } else {
        const testMoment = moment(start);
        if (testMoment.isValid()) {
          start = testMoment.toISOString();
        } else {
          toast.error('تاریخ شروع نامعتبر است');
          return;
        }
      }
    }
    
    if (end) {
      // If it's already a string, validate it
      if (typeof end === 'string') {
        const testMoment = moment(end);
        if (!testMoment.isValid()) {
          toast.error('تاریخ پایان نامعتبر است');
          return;
        }
        end = testMoment.toISOString();
      } else if (moment.isMoment(end)) {
        end = end.toISOString();
      } else {
        const testMoment = moment(end);
        if (testMoment.isValid()) {
          end = testMoment.toISOString();
        } else {
          toast.error('تاریخ پایان نامعتبر است');
          return;
        }
      }
    }
    
    // Debug: Log the values to ensure they are ISO strings
    console.log('Event data before save:', { start, end, startType: typeof start, endType: typeof end });
    
    // Ensure start and end are valid ISO strings (not format strings)
    if (typeof start === 'string' && (start.includes('DD') || start.includes('MM') || start.includes('YYYY'))) {
      console.error('Invalid start date format detected:', start);
      toast.error('فرمت تاریخ شروع نامعتبر است. لطفاً دوباره انتخاب کنید.');
      return;
    }
    
    if (typeof end === 'string' && (end.includes('DD') || end.includes('MM') || end.includes('YYYY'))) {
      console.error('Invalid end date format detected:', end);
      toast.error('فرمت تاریخ پایان نامعتبر است. لطفاً دوباره انتخاب کنید.');
      return;
    }
    
    const eventData = {
      id: currentEvent?.id ? currentEvent?.id : uuidv4(),
      color: data?.color,
      title: data?.title,
      allDay: data?.allDay,
      description: data?.description || '',
      end,
      start,
    };

    // Validate required fields
    if (!start || !end) {
      toast.error('لطفا تاریخ شروع و پایان را انتخاب کنید');
      return;
    }
    
    // Final validation: ensure they are valid ISO strings
    if (!moment(start).isValid() || !moment(end).isValid()) {
      console.error('Invalid ISO strings:', { start, end });
      toast.error('تاریخ‌های انتخاب شده نامعتبر هستند');
      return;
    }

    try {
      if (!dateError) {
        if (currentEvent?.id) {
          await updateEvent(eventData);
          toast.success('رویداد با موفقیت به‌روزرسانی شد!');
        } else {
          await createEvent(eventData);
          toast.success('رویداد با موفقیت ایجاد شد!');
        }
        onClose();
        reset();
      }
    } catch (error) {
      console.error('Error saving event:', error);
      toast.error(error?.response?.data?.message || 'خطا در ذخیره رویداد');
    }
  });

  const onDelete = useCallback(async () => {
    try {
      await deleteEvent(`${currentEvent?.id}`);
      toast.success('رویداد با موفقیت حذف شد!');
      onClose();
    } catch (error) {
      console.error(error);
    }
  }, [currentEvent?.id, onClose]);

  return (
    <Form methods={methods} onSubmit={onSubmit}>
      <Scrollbar sx={{ p: 3, bgcolor: 'background.neutral' }}>
        <Stack spacing={3}>
          <Field.Text name="title" label="عنوان" />

          <Field.Text name="description" label="توضیحات" multiline rows={3} />

          <Field.Switch name="allDay" label="تمام روز" />

          <Field.MobileDateTimePicker name="start" label="تاریخ شروع" />

          <Field.MobileDateTimePicker
            name="end"
            label="تاریخ پایان"
            slotProps={{
              textField: {
                error: dateError,
                helperText: dateError ? 'تاریخ پایان باید بعد از تاریخ شروع باشد' : null,
              },
            }}
          />

          <Controller
            name="color"
            control={control}
            render={({ field }) => (
              <ColorPicker
                selected={field.value}
                onSelectColor={(color) => field.onChange(color)}
                colors={colorOptions}
              />
            )}
          />
        </Stack>
      </Scrollbar>

      <DialogActions sx={{ flexShrink: 0 }}>
        {!!currentEvent?.id && (
          <Tooltip title="حذف رویداد">
            <IconButton onClick={onDelete}>
              <Iconify icon="solar:trash-bin-trash-bold" />
            </IconButton>
          </Tooltip>
        )}

        <Box sx={{ flexGrow: 1 }} />

        <Button variant="outlined" color="inherit" onClick={onClose}>
          لغو
        </Button>

        <LoadingButton
          type="submit"
          variant="contained"
          loading={isSubmitting}
          disabled={dateError}
        >
          ذخیره تغییرات
        </LoadingButton>
      </DialogActions>
    </Form>
  );
}
