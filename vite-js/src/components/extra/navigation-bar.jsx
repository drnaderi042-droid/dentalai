import React from 'react';
import { m } from 'framer-motion';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import { useTheme } from '@mui/material/styles';
import Typography from '@mui/material/Typography';

import { varFade } from 'src/components/animate';
import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export function NavigationBar({ 
  tabs = [], 
  currentTab, 
  onTabChange, 
  horizontal = true,
  spacing = 2,
  buttonWidth = 120,
  buttonHeight = 100,
  ...other 
}) {
  const theme = useTheme();
  const isHorizontal = horizontal;
  const isDarkMode = theme.palette.mode === 'dark';

  const renderHorizontalNav = () => (
    <Stack
      direction="row"
      spacing={spacing}
      sx={{
        overflowX: 'auto',
        pb: 1,
        px: 2,
        bgcolor: isDarkMode ? '#282b33' : '#f2f6ff',
        borderRadius: 1,
      }}
      {...other}
    >
      {tabs.map((tab, index) => (
        <m.div
          key={tab.value || tab.id}
          variants={varFade().inUp}
          custom={index}
        >
          <Button
            onClick={() => onTabChange?.({}, tab.value || tab.id)}
            sx={{
              minWidth: buttonWidth,
              height: buttonHeight,
              flexDirection: 'column',
              gap: 1,
              p: 2,
              borderRadius: 1.5,
              bgcolor: currentTab === (tab.value || tab.id) 
                ? 'primary.main' 
                : 'transparent',
              color: currentTab === (tab.value || tab.id) 
                ? 'primary.contrastText' 
                : 'text.secondary',
              '&:hover': {
                bgcolor: currentTab === (tab.value || tab.id) 
                  ? 'primary.dark' 
                  : 'action.hover',
                color: currentTab === (tab.value || tab.id) 
                  ? 'primary.contrastText' 
                  : 'text.primary',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            {tab.icon && (
              <Iconify 
                icon={tab.icon} 
                width={24} 
                height={24} 
              />
            )}
            <Typography 
              variant="caption" 
              sx={{ 
                textAlign: 'center',
                fontSize: '0.75rem',
                lineHeight: 1.2,
                maxWidth: buttonWidth - 16,
                wordBreak: 'break-word',
              }}
            >
              {tab.label}
            </Typography>
          </Button>
        </m.div>
      ))}
    </Stack>
  );

  const renderVerticalNav = () => (
    <Stack
      direction="column"
      spacing={spacing}
      {...other}
    >
      {tabs.map((tab, index) => (
        <m.div
          key={tab.value || tab.id}
          variants={varFade().inLeft}
          custom={index}
        >
          <Button
            onClick={() => onTabChange?.({}, tab.value || tab.id)}
            sx={{
              minWidth: buttonWidth,
              height: buttonHeight,
              flexDirection: 'column',
              gap: 1,
              p: 2,
              justifyContent: 'flex-start',
              textAlign: 'center',
              borderRadius: 1.5,
              bgcolor: currentTab === (tab.value || tab.id) 
                ? 'primary.main' 
                : 'transparent',
              color: currentTab === (tab.value || tab.id) 
                ? 'primary.contrastText' 
                : 'text.secondary',
              '&:hover': {
                bgcolor: currentTab === (tab.value || tab.id) 
                  ? 'primary.dark' 
                  : 'action.hover',
                color: currentTab === (tab.value || tab.id) 
                  ? 'primary.contrastText' 
                  : 'text.primary',
              },
              transition: 'all 0.2s ease-in-out',
            }}
          >
            {tab.icon && (
              <Iconify 
                icon={tab.icon} 
                width={24} 
                height={24} 
              />
            )}
            <Typography 
              variant="caption" 
              sx={{ 
                textAlign: 'center',
                fontSize: '0.75rem',
                lineHeight: 1.2,
                maxWidth: buttonWidth - 16,
                wordBreak: 'break-word',
              }}
            >
              {tab.label}
            </Typography>
          </Button>
        </m.div>
      ))}
    </Stack>
  );

  return (
    <Box>
      {isHorizontal ? renderHorizontalNav() : renderVerticalNav()}
    </Box>
  );
}
