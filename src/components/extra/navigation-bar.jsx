import React from 'react';
import { m } from 'framer-motion';

import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Stack from '@mui/material/Stack';
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
  const isHorizontal = horizontal;

  const renderHorizontalNav = () => (
    <Stack
      direction="row"
      spacing={spacing}
      sx={{
        overflowX: 'auto',
        bgcolor: '#f2f6ff',
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
                ? 'grey.100'
                : 'transparent',
              color: currentTab === (tab.value || tab.id)
                ? 'text.primary'
                : 'text.secondary',
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
                ? 'grey.100'
                : 'transparent',
              color: currentTab === (tab.value || tab.id)
                ? 'text.primary'
                : 'text.secondary',
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
