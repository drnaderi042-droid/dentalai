import dayjs from 'dayjs';
import moment from 'moment-jalaali';
import { Controller, useFormContext } from 'react-hook-form';

import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import { MobileDateTimePicker } from '@mui/x-date-pickers/MobileDateTimePicker';

import { formatStr } from 'src/utils/format-time';

// ----------------------------------------------------------------------

// Helper function to convert value to appropriate date object
const getDateValue = (value) => {
  if (!value) return null;
  
  // If using moment-jalaali adapter, use moment, otherwise use dayjs
  // Check if the value is already a moment object
  if (moment.isMoment(value)) {
    return value;
  }
  
  // If it's already a dayjs object, convert to moment
  if (dayjs.isDayjs(value)) {
    return moment(value.toDate());
  }
  
  // Try to parse as ISO string or various formats
  // First try moment (for jalali support)
  const momentValue = moment(value);
  if (momentValue.isValid()) {
    return momentValue;
  }
  
  // If moment fails, try dayjs
  const dayjsValue = dayjs(value);
  if (dayjsValue.isValid()) {
    return moment(dayjsValue.toDate());
  }
  
  // If both fail, return null
  console.warn('Invalid date value:', value);
  return null;
};

// Helper function to format date value to ISO string
const formatDateValue = (value) => {
  if (!value) return null;
  
  // Reject format strings (like "DD/MM/YYYY" or "jYYYY/jMM/jDD")
  if (typeof value === 'string') {
    // Check if it's a format string pattern
    const formatPatterns = [
      /DD\/MM\/YYYY/i,
      /YYYY\/MM\/DD/i,
      /jYYYY\/jMM\/jDD/i,
      /DD-MM-YYYY/i,
      /YYYY-MM-DD/i,
      /jYYYY-jMM-jDD/i,
      /DD MMM YYYY/i,
      /YYYY MMM DD/i,
    ];
    
    for (const pattern of formatPatterns) {
      if (pattern.test(value) && !/^\d/.test(value)) {
        // It's a format string, not a date value
        console.warn('Format string detected, rejecting:', value);
        return null;
      }
    }
  }
  
  // If it's a moment object, convert to ISO string
  if (moment.isMoment(value)) {
    if (!value.isValid()) {
      console.warn('Invalid moment object:', value);
      return null;
    }
    return value.toISOString();
  }
  
  // If it's a dayjs object, convert to ISO string
  if (dayjs.isDayjs(value)) {
    if (!value.isValid()) {
      console.warn('Invalid dayjs object:', value);
      return null;
    }
    return value.toISOString();
  }
  
  // If it's already a string, check if it's a valid ISO string
  if (typeof value === 'string') {
    // Check if it's already a valid ISO string format
    const isoRegex = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{3})?Z?$/;
    if (isoRegex.test(value)) {
      // Validate it's a valid date
      const testMoment = moment(value);
      if (testMoment.isValid()) {
        return value; // Return as is if it's already a valid ISO string
      }
    }
    // Try to parse and convert to ISO
    const testMoment = moment(value);
    if (testMoment.isValid()) {
      return testMoment.toISOString();
    }
  }
  
  // If it's a number (timestamp), convert to ISO string
  if (typeof value === 'number') {
    const testMoment = moment(value);
    if (testMoment.isValid()) {
      return testMoment.toISOString();
    }
  }
  
  // Try to parse with dayjs as last resort
  const dayjsValue = dayjs(value);
  if (dayjsValue.isValid()) {
    return dayjsValue.toISOString();
  }
  
  console.warn('Could not format date value:', value);
  return null;
};

export function RHFDatePicker({ name, slotProps, ...other }) {
  const { control } = useFormContext();

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState: { error } }) => (
        <DatePicker
          {...field}
          value={getDateValue(field.value)}
          onChange={(newValue) => field.onChange(formatDateValue(newValue))}
          format={formatStr.split.date}
          slotProps={{
            textField: {
              fullWidth: true,
              error: !!error,
              helperText: error?.message ?? slotProps?.textField?.helperText,
              ...slotProps?.textField,
            },
            ...slotProps,
          }}
          {...other}
        />
      )}
    />
  );
}

// ----------------------------------------------------------------------

export function RHFMobileDateTimePicker({ name, slotProps, ...other }) {
  const { control } = useFormContext();

  return (
    <Controller
      name={name}
      control={control}
      render={({ field, fieldState: { error } }) => (
        <MobileDateTimePicker
          {...field}
          value={getDateValue(field.value)}
          onChange={(newValue) => field.onChange(formatDateValue(newValue))}
          format={formatStr.split.dateTime}
          slotProps={{
            textField: {
              fullWidth: true,
              error: !!error,
              helperText: error?.message ?? slotProps?.textField?.helperText,
              ...slotProps?.textField,
            },
            ...slotProps,
          }}
          {...other}
        />
      )}
    />
  );
}
