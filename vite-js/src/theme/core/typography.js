import { setFont, pxToRem, responsiveFontSizes } from '../styles/utils';

// ----------------------------------------------------------------------

export const defaultFont = 'Yekan Bakh';

export const primaryFont = setFont(defaultFont);

export const secondaryFont = setFont('Barlow');

// ----------------------------------------------------------------------

export const typography = {
  fontFamily: primaryFont,
  fontSecondaryFamily: secondaryFont,
  fontWeightLight: '300',
  fontWeightRegular: '400',
  fontWeightMedium: '500',
  fontWeightSemiBold: '600',
  fontWeightBold: '700',
  h1: {
    fontWeight: 800,
    lineHeight: 80 / 64,
    fontSize: pxToRem(36),
    fontFamily: primaryFont, // Changed from secondaryFont to primaryFont
    ...responsiveFontSizes({ sm: 48, md: 54, lg: 60 }),
  },
  h2: {
    fontWeight: 800,
    lineHeight: 64 / 48,
    fontSize: pxToRem(30),
    fontFamily: primaryFont, // Changed from secondaryFont to primaryFont
    ...responsiveFontSizes({ sm: 38, md: 42, lg: 46 }),
  },
  h3: {
    fontWeight: 700,
    lineHeight: 1.5,
    fontSize: pxToRem(22),
    fontFamily: primaryFont, // Changed from secondaryFont to primaryFont
    ...responsiveFontSizes({ sm: 24, md: 28, lg: 30 }),
  },
  h4: {
    fontWeight: 700,
    lineHeight: 1.5,
    fontSize: pxToRem(18),
    ...responsiveFontSizes({ sm: 18, md: 22, lg: 22 }),
  },
  h5: {
    fontWeight: 700,
    lineHeight: 1.5,
    fontSize: pxToRem(16),
    ...responsiveFontSizes({ sm: 17, md: 18, lg: 18 }),
  },
  h6: {
    fontWeight: 600,
    lineHeight: 28 / 18,
    fontSize: pxToRem(15),
    ...responsiveFontSizes({ sm: 16, md: 16, lg: 16 }),
  },
  subtitle1: {
    fontWeight: 600,
    lineHeight: 1.5,
    fontSize: pxToRem(15),
  },
  subtitle2: {
    fontWeight: 600,
    lineHeight: 22 / 14,
    fontSize: pxToRem(13),
  },
  body1: {
    lineHeight: 1.5,
    fontSize: pxToRem(15),
  },
  body2: {
    lineHeight: 22 / 14,
    fontSize: pxToRem(13),
  },
  caption: {
    lineHeight: 1.5,
    fontSize: pxToRem(11),
  },
  overline: {
    fontWeight: 700,
    lineHeight: 1.5,
    fontSize: pxToRem(11),
    textTransform: 'uppercase',
  },
  button: {
    fontWeight: 400,
    lineHeight: 24 / 14,
    fontSize: pxToRem(13),
    textTransform: 'unset',
  },
};
