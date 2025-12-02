import SvgIcon from '@mui/material/SvgIcon';
import { buttonClasses } from '@mui/material/Button';
import { dialogActionsClasses } from '@mui/material/DialogActions';

import { stylesMode } from '../../styles';

// ----------------------------------------------------------------------

/**
 * Icons
 */
/* https://icon-sets.iconify.design/eva/chevron-down-fill */
export const PickerSwitchIcon = (props) => (
  <SvgIcon {...props}>
    <path
      fill="currentColor"
      d="M12 15.5a1 1 0 0 1-.71-.29l-4-4a1 1 0 1 1 1.42-1.42L12 13.1l3.3-3.18a1 1 0 1 1 1.38 1.44l-4 3.86a1 1 0 0 1-.68.28"
    />
  </SvgIcon>
);

/* Left arrow icon - points to the left */
export const PickerLeftIcon = (props) => (
  <SvgIcon {...props} sx={{ transform: 'none' }}>
    <path
      fill="currentColor"
      d="M15.41 7.41L14 6l-6 6 6 6 1.41-1.41L10.83 12z"
    />
  </SvgIcon>
);

/* Right arrow icon - points to the right */
export const PickerRightIcon = (props) => (
  <SvgIcon {...props} sx={{ transform: 'none' }}>
    <path
      fill="currentColor"
      d="M10 6L8.59 7.41 13.17 12l-4.58 4.59L10 18l6-6z"
    />
  </SvgIcon>
);

/* https://icon-sets.iconify.design/solar/calendar-mark-bold-duotone */
export const PickerCalendarIcon = (props) => (
  <SvgIcon {...props}>
    <path
      fill="currentColor"
      d="M6.96 2c.418 0 .756.31.756.692V4.09c.67-.012 1.422-.012 2.268-.012h4.032c.846 0 1.597 0 2.268.012V2.692c0-.382.338-.692.756-.692s.756.31.756.692V4.15c1.45.106 2.403.368 3.103 1.008c.7.641.985 1.513 1.101 2.842v1H2V8c.116-1.329.401-2.2 1.101-2.842c.7-.64 1.652-.902 3.103-1.008V2.692c0-.382.339-.692.756-.692"
    />
    <path
      fill="currentColor"
      d="M22 14v-2c0-.839-.013-2.335-.026-3H2.006c-.013.665 0 2.161 0 3v2c0 3.771 0 5.657 1.17 6.828C4.349 22 6.234 22 10.004 22h4c3.77 0 5.654 0 6.826-1.172C22 19.657 22 17.771 22 14"
      opacity="0.5"
    />
    <path fill="currentColor" d="M18 16.5a1.5 1.5 0 1 1-3 0a1.5 1.5 0 0 1 3 0" />
  </SvgIcon>
);

/* https://icon-sets.iconify.design/solar/clock-circle-outline */
export const PickerClockIcon = (props) => (
  <SvgIcon {...props}>
    <path
      fill="currentColor"
      fillRule="evenodd"
      d="M12 2.75a9.25 9.25 0 1 0 0 18.5a9.25 9.25 0 0 0 0-18.5M1.25 12C1.25 6.063 6.063 1.25 12 1.25S22.75 6.063 22.75 12S17.937 22.75 12 22.75S1.25 17.937 1.25 12M12 7.25a.75.75 0 0 1 .75.75v3.69l2.28 2.28a.75.75 0 1 1-1.06 1.06l-2.5-2.5a.75.75 0 0 1-.22-.53V8a.75.75 0 0 1 .75-.75"
      clipRule="evenodd"
    />
  </SvgIcon>
);

const defaultProps = {
  date: {
    openPickerIcon: PickerCalendarIcon,
    leftArrowIcon: PickerLeftIcon,
    rightArrowIcon: PickerRightIcon,
    switchViewIcon: PickerSwitchIcon,
  },
  time: {
    openPickerIcon: PickerClockIcon,
    rightArrowIcon: PickerRightIcon,
    switchViewIcon: PickerSwitchIcon,
  },
};

const MuiDatePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان" (day name, day number, full month name)
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiDateTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiStaticDatePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiDesktopDatePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiDesktopDateTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiMobileDatePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiMobileDateTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { 
    slots: defaultProps.date,
    slotProps: {
      toolbar: {
        toolbarFormat: 'dddd jD jMMMM', // Full format: "دوشنبه 26 آبان"
        format: 'dddd jD jMMMM', // Also set format as fallback
      },
    },
  },
};

const MuiTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { slots: defaultProps.time },
};

const MuiMobileTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { slots: defaultProps.time },
};

const MuiStaticTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { slots: defaultProps.time },
};

const MuiDesktopTimePicker = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  defaultProps: { slots: defaultProps.time },
};

const MuiPickersLayout = {
  /** **************************************
   * STYLE
   *************************************** */
  styleOverrides: {
    root: ({ theme }) => ({
      [`& .${dialogActionsClasses.root}`]: {
        [`& .${buttonClasses.root}`]: {
          [`&:last-of-type`]: {
            color: theme.vars.palette.common.white,
            backgroundColor: theme.vars.palette.text.primary,
            [stylesMode.dark]: { color: theme.vars.palette.grey[800] },
          },
        },
      },
      // Fix arrow icons in RTL mode
      '& [data-testid="ArrowLeftIcon"]': {
        transform: 'none !important',
      },
      '& [data-testid="ArrowRightIcon"]': {
        transform: 'none !important',
      },
    }),
  },
};

const MuiPickersArrowSwitcher = {
  /** **************************************
   * STYLE
   *************************************** */
  styleOverrides: {
    root: {
      // Ensure arrows are not flipped in RTL
      '& button:first-of-type svg': {
        transform: 'none !important',
      },
      '& button:last-of-type svg': {
        transform: 'none !important',
      },
    },
  },
};

const MuiPickersPopper = {
  /** **************************************
   * DEFAULT PROPS
   *************************************** */
  styleOverrides: {
    paper: ({ theme }) => ({
      boxShadow: theme.customShadows.dropdown,
      borderRadius: theme.shape.borderRadius * 1.5,
    }),
  },
};

const MuiPickersToolbar = {
  /** **************************************
   * STYLE
   *************************************** */
  styleOverrides: {
    root: {
      // Fix layout shift by setting a fixed width for the month/year display
      '& .MuiPickersToolbar-title': {
        minWidth: '220px', // Fixed width to prevent layout shift (enough for "دوشنبه 26 آبان")
        width: '220px', // Fixed width to prevent layout shift
        textAlign: 'center',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        flexShrink: 0, // Prevent shrinking
      },
      // Also fix the container to prevent layout shift
      '& .MuiPickersToolbar-content': {
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        minHeight: '56px', // Fixed height to prevent vertical shift
        height: '56px', // Fixed height to prevent vertical shift
      },
      // Fix layout shift for toolbar buttons (month and year)
      '& button': {
        minHeight: '48px', // Fixed height for buttons
        height: '48px', // Fixed height for buttons
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        '& .MuiPickersToolbarText-root': {
          textOverflow: 'unset !important',
          overflow: 'visible !important',
          whiteSpace: 'nowrap !important',
          lineHeight: '1.5 !important',
          minHeight: '24px', // Fixed height for text
          display: 'flex',
          alignItems: 'center',
        },
      },
      // Specifically target year button to prevent layout shift
      '& button[data-testid="pickers-toolbar-year"]': {
        minHeight: '48px !important',
        height: '48px !important',
        display: 'flex !important',
        alignItems: 'center !important',
        justifyContent: 'center !important',
      },
      // Specifically target month button to prevent layout shift
      '& button[data-testid="pickers-toolbar-month"]': {
        minHeight: '48px !important',
        height: '48px !important',
        display: 'flex !important',
        alignItems: 'center !important',
        justifyContent: 'center !important',
      },
    },
  },
};

const MuiPickersToolbarText = {
  /** **************************************
   * STYLE
   *************************************** */
  styleOverrides: {
    root: {
      // Ensure full month name is displayed (not abbreviated)
      '&.MuiPickersToolbarText-monthButton': {
        // Force full month name display - prevent text truncation
        textOverflow: 'unset !important',
        overflow: 'visible !important',
        whiteSpace: 'nowrap !important',
        maxWidth: 'none !important',
        width: 'auto !important',
        minHeight: '24px !important',
        height: 'auto !important',
        lineHeight: '1.5 !important',
        display: 'flex !important',
        alignItems: 'center !important',
      },
      // Target the month text specifically
      '&[data-testid="pickers-toolbar-month"]': {
        textOverflow: 'unset !important',
        overflow: 'visible !important',
        whiteSpace: 'nowrap !important',
        maxWidth: 'none !important',
        minHeight: '24px !important',
        lineHeight: '1.5 !important',
        display: 'flex !important',
        alignItems: 'center !important',
      },
      // Target year text to prevent layout shift
      '&[data-testid="pickers-toolbar-year"]': {
        textOverflow: 'unset !important',
        overflow: 'visible !important',
        whiteSpace: 'nowrap !important',
        maxWidth: 'none !important',
        minHeight: '24px !important',
        height: 'auto !important',
        lineHeight: '1.5 !important',
        display: 'flex !important',
        alignItems: 'center !important',
      },
      // Target all toolbar text elements that might contain month names
      '&[role="button"]': {
        textOverflow: 'unset !important',
        overflow: 'visible !important',
        whiteSpace: 'nowrap !important',
        minHeight: '24px !important',
        lineHeight: '1.5 !important',
        display: 'flex !important',
        alignItems: 'center !important',
      },
    },
  },
};

// ----------------------------------------------------------------------

export const datePicker = {
  MuiPickersPopper,
  MuiPickersLayout,
  MuiPickersArrowSwitcher,
  MuiPickersToolbar,
  MuiPickersToolbarText,
  // Date
  MuiDatePicker,
  MuiDateTimePicker,
  MuiStaticDatePicker,
  MuiDesktopDatePicker,
  MuiDesktopDateTimePicker,
  MuiMobileDatePicker,
  MuiMobileDateTimePicker,
  // Time
  MuiTimePicker,
  MuiMobileTimePicker,
  MuiStaticTimePicker,
  MuiDesktopTimePicker,
};
