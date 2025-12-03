import Stack from '@mui/material/Stack';
import { useTheme } from '@mui/material/styles';

import { CONFIG } from 'src/config-global';
import { varAlpha, textGradient } from 'src/theme/styles';

import { MotionViewport } from 'src/components/animate';

import { SectionTitle } from './components/section-title';

// ----------------------------------------------------------------------

export function HomeForDesigner({ sx, ...other }) {
  const theme = useTheme();


  return (
    <Stack
      component="section"
      sx={{
        position: 'relative',
        backgroundImage: `linear-gradient(135deg, ${varAlpha(theme.vars.palette.grey['900Channel'], 0.8)} 0%, ${theme.vars.palette.grey[900]} 75%), url(${CONFIG.site.basePath}/assets/images/home/for-designer.webp)`,
        backgroundRepeat: 'no-repeat',
        backgroundPosition: 'center center',
        backgroundSize: 'cover',
        backgroundColor: 'grey.700',
        ...sx,
      }}
      {...other}
    >
      <MotionViewport>
        <Stack
          spacing={5}
          sx={{
            px: 2,
            py: 15,
            alignItems: 'center',
          }}
        >

          <SectionTitle
            title="برای حرفه ای ها"
            description="بسپرش به AI"
            sx={{
              zIndex: 1,
              textAlign: 'center',
              alignItems: 'center',
            }}
            slotProps={{
              caption: {
                sx: {
                  ...textGradient(
                    `to right, ${theme.vars.palette.common.white}, ${varAlpha(theme.vars.palette.common.whiteChannel, 0.2)}`
                  ),
                },
              },
              title: {
                sx: {
                  ...textGradient(
                    `135deg, ${theme.vars.palette.warning.main}, ${theme.vars.palette.primary.main}`
                  ),
                },
              },
              description: { sx: { maxWidth: 320, color: 'common.white', textAlign: 'center' } },
            }}
          />

        </Stack>
      </MotionViewport>
    </Stack>
  );
}
