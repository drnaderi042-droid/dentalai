import { varAlpha } from '../../styles';

// ----------------------------------------------------------------------

const MuiBackdrop = {
  /** **************************************
   * STYLE
   *************************************** */
  styleOverrides: {
    root: ({ theme }) => ({
      backgroundColor: theme.palette.mode === 'dark' 
        ? varAlpha(theme.vars.palette.background.defaultChannel, 0.8)
        : varAlpha(theme.vars.palette.grey['800Channel'], 0.48),

    }),
    invisible: { background: 'transparent' },
  },
};

// ----------------------------------------------------------------------

export const backdrop = { MuiBackdrop };
