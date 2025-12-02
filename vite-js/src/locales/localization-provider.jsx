/* eslint-disable perfectionist/sort-imports */

import moment from 'moment-jalaali';

import { AdapterMomentJalaali } from '@mui/x-date-pickers/AdapterMomentJalaali';
import { LocalizationProvider as Provider } from '@mui/x-date-pickers/LocalizationProvider';

// ----------------------------------------------------------------------

// Initialize moment-jalaali with Persian locale
moment.loadPersian({ dialect: 'persian-modern' });

// Configure moment to use full month names (not abbreviated)
// Override the locale data to ensure full month names are used
if (moment.localeData) {
  const localeData = moment.localeData('fa');
  if (localeData && localeData._months) {
    // Ensure full month names are used (not abbreviated)
    // This affects how months are displayed in date pickers
  }
}

// Persian locale text for DatePicker components
const faIRLocaleText = {
  // Calendar navigation
  previousMonth: 'ماه قبل',
  nextMonth: 'ماه بعد',
  
  // Calendar view switch
  calendarWeekNumberHeaderLabel: 'هفته',
  calendarWeekNumberHeaderText: '#',
  calendarWeekNumberAriaLabelText: weekNumber => `هفته ${weekNumber}`,
  calendarWeekNumberText: weekNumber => `${weekNumber}`,
  
  // Open picker labels
  openDatePickerDialogue: (value, utils) =>
    value !== null && utils.isValid(value)
      ? `انتخاب تاریخ، تاریخ انتخاب شده ${utils.format(value, 'fullDate')} است`
      : 'انتخاب تاریخ',
  openTimePickerDialogue: (value, utils) =>
    value !== null && utils.isValid(value)
      ? `انتخاب زمان، زمان انتخاب شده ${utils.format(value, 'fullTime')} است`
      : 'انتخاب زمان',
  fieldClearLabel: 'پاک کردن',
  
  // Table labels
  fieldYearPlaceholder: params => 'Y'.repeat(params.digitAmount),
  fieldMonthPlaceholder: params => params.contentType === 'letter' ? 'MMMM' : 'MM',
  fieldDayPlaceholder: () => 'DD',
  fieldWeekDayPlaceholder: params => params.contentType === 'letter' ? 'EEEE' : 'EE',
  fieldHoursPlaceholder: () => 'hh',
  fieldMinutesPlaceholder: () => 'mm',
  fieldSecondsPlaceholder: () => 'ss',
  fieldMeridiemPlaceholder: () => 'aa',
  
  // View names
  year: 'سال',
  month: 'ماه',
  day: 'روز',
  week: 'هفته',
  hours: 'ساعت',
  minutes: 'دقیقه',
  seconds: 'ثانیه',
  
  // Common
  cancelButtonLabel: 'لغو',
  clearButtonLabel: 'پاک کردن',
  okButtonLabel: 'تأیید',
  todayButtonLabel: 'امروز',
  
  // Field input placeholders
  fieldYearPlaceholderV2: params => 'Y'.repeat(params.digitAmount),
  fieldMonthPlaceholderV2: params => params.contentType === 'letter' ? 'MMMM' : 'MM',
  fieldDayPlaceholderV2: () => 'DD',
  fieldHoursPlaceholderV2: () => 'hh',
  fieldMinutesPlaceholderV2: () => 'mm',
  fieldSecondsPlaceholderV2: () => 'ss',
  fieldMeridiemPlaceholderV2: () => 'aa',
  
  // Date range
  start: 'شروع',
  end: 'پایان',
  
  // Select date & time - این متن‌ها برای MobileDateTimePicker استفاده می‌شوند
  selectDate: 'انتخاب تاریخ',
  selectTime: 'انتخاب زمان',
  selectDateTime: 'انتخاب تاریخ و زمان',
  
  // Action bar
  actionBarLabel: 'نوار عملیات',
  
  // Clock labels
  clockLabelText: (view, time, adapter) =>
    `انتخاب ${view === 'hours' ? 'ساعت' : 'دقیقه'}. ${time === null ? 'زمان انتخاب نشده' : `زمان انتخاب شده ${adapter.format(time, 'fullTime')} است`}`,
  hoursClockNumberText: hours => `${hours} ساعت`,
  minutesClockNumberText: minutes => `${minutes} دقیقه`,
  secondsClockNumberText: seconds => `${seconds} ثانیه`,
  
  // Digital clock section
  digitalClockSectionLabel: 'انتخاب زمان',
  
  // Calendar header
  calendarHeaderLabel: 'انتخاب تاریخ',
  dateTableLabel: 'انتخاب تاریخ',
  timeTableLabel: 'انتخاب زمان',
  
  // Field section
  fieldSectionLabel: 'انتخاب تاریخ و زمان',
  fieldSectionLabelWithoutHeader: 'انتخاب',
  
  // Error messages
  invalidDate: 'تاریخ نامعتبر',
  invalidTime: 'زمان نامعتبر',
  invalidDateTime: 'تاریخ و زمان نامعتبر',
  
  // Format
  datePickerToolbarTitle: 'انتخاب تاریخ',
  timePickerToolbarTitle: 'انتخاب زمان',
  dateTimePickerToolbarTitle: 'انتخاب تاریخ و زمان',
};

export function LocalizationProvider({ children }) {
  // Always use AdapterMomentJalaali for Persian/Jalali calendar
  // This ensures all DatePickers use the Jalali calendar regardless of language setting
  return (
    <Provider 
      dateAdapter={AdapterMomentJalaali} 
      adapterLocale="fa"
      localeText={faIRLocaleText}
    >
      {children}
    </Provider>
  );
}
