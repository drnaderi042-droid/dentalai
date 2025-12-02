import { m } from 'framer-motion';

import Box from '@mui/material/Box';
import Divider from '@mui/material/Divider';
import Popover from '@mui/material/Popover';
import SvgIcon from '@mui/material/SvgIcon';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';

import { useRouter, usePathname } from 'src/routes/hooks';

import { varHover } from 'src/components/animate';
import { usePopover } from 'src/components/custom-popover';

import { navData } from '../config-nav-dashboard';

// ----------------------------------------------------------------------

export function MenuPopover({ sx, ...other }) {
  const popover = usePopover();
  const router = useRouter();
  const pathname = usePathname();

  const handleClickItem = (path) => {
    popover.onClose();
    router.push(path);
  };

  // Flatten nav data to get all items
  const allNavItems = navData.flatMap((section) => 
    section.items?.map((item) => ({
      ...item,
      subheader: section.subheader,
    })) || []
  );

  return (
    <>
      <IconButton
        component={m.button}
        whileTap="tap"
        whileHover="hover"
        variants={varHover(1.05)}
        onClick={popover.onOpen}
        sx={{
          ...(popover.open && { bgcolor: (theme) => theme.vars.palette.action.selected }),
          ...sx,
        }}
        {...other}
      >
        <SvgIcon>
          <path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z" />
        </SvgIcon>
      </IconButton>

      <Popover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        transformOrigin={{ vertical: 'top', horizontal: 'left' }}
        slotProps={{
          paper: { sx: { width: 240, p: 0, mt: 1 } },
        }}
      >
        {navData.map((section, sectionIndex) => (
          <Box key={section.subheader || sectionIndex}>
            {section.subheader && (
              <Typography
                variant="caption"
                sx={{
                  px: 2,
                  py: 1,
                  display: 'block',
                  color: 'text.secondary',
                  fontWeight: 600,
                  textTransform: 'uppercase',
                  fontSize: '0.65rem',
                }}
              >
                {section.subheader}
              </Typography>
            )}

            {section.items?.map((item) => {
              const isActive = pathname === item.path || pathname.startsWith(`${item.path}/`);
              
              return (
                <MenuItem
                  key={item.title}
                  onClick={() => handleClickItem(item.path)}
                  selected={isActive}
                  sx={{
                    py: 1,
                    px: 2,
                    color: isActive ? 'primary.main' : 'text.secondary',
                    '& svg': { width: 20, height: 20 },
                    '&:hover': { color: 'text.primary' },
                    '&.Mui-selected': {
                      bgcolor: 'action.selected',
                      '&:hover': {
                        bgcolor: 'action.hover',
                      },
                    },
                  }}
                >
                  {item.icon}
                  <Box component="span" sx={{ ml: 2 }}>
                    {item.title}
                  </Box>
                </MenuItem>
              );
            })}

            {sectionIndex < navData.length - 1 && <Divider sx={{ my: 0.5 }} />}
          </Box>
        ))}
      </Popover>
    </>
  );
}

