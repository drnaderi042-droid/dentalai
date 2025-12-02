import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Drawer from '@mui/material/Drawer';
import MenuItem from '@mui/material/MenuItem';
import { useTheme } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import { paths } from 'src/routes/paths';
import { useRouter, usePathname } from 'src/routes/hooks';

import axios from 'src/utils/axios';

import { CONFIG } from 'src/config-global';
import { varAlpha } from 'src/theme/styles';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';
import { Scrollbar } from 'src/components/scrollbar';
import { AnimateAvatar } from 'src/components/animate';

import { useAuthContext } from 'src/auth/hooks';

import { AccountButton } from './account-button';
import { SignOutButton } from './sign-out-button';

// ----------------------------------------------------------------------

export function AccountDrawer({ data = [], sx, ...other }) {
  const theme = useTheme();

  const router = useRouter();

  const pathname = usePathname();

  const { user } = useAuthContext();

  const [open, setOpen] = useState(false);
  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch user profile when drawer opens
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (open && user?.accessToken) {
        try {
          setLoading(true);
          const response = await axios.get('/api/auth/me', {
            headers: {
              Authorization: `Bearer ${user.accessToken}`,
            },
          });

          setUserProfile(response.data.user);
        } catch (error) {
          console.error('Error fetching user profile:', error);
        } finally {
          setLoading(false);
        }
      }
    };

    fetchUserProfile();
  }, [open, user]);

  const handleOpenDrawer = useCallback(() => {
    setOpen(true);
  }, []);

  const handleCloseDrawer = useCallback(() => {
    setOpen(false);
  }, []);

  const handleClickItem = useCallback(
    (path) => {
      handleCloseDrawer();
      router.push(path);
    },
    [handleCloseDrawer, router]
  );

  // Build full URL if avatar URL is a relative path
  const getAvatarUrl = useCallback((url) => {
    if (!url) return null;
    
    // If it's already a full URL, return as is
    if (url.startsWith('http://') || url.startsWith('https://')) {
      return url;
    }
    
    // If it's a relative path, prepend server URL
    if (url.startsWith('/')) {
      const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
      return `${serverUrl}${url}`;
    }
    
    // If it's a path without leading slash
    const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
    return `${serverUrl}/${url}`;
  }, []);

  // Get avatar URL from userProfile or user
  const avatarUrl = userProfile?.photoURL || userProfile?.avatarUrl || userProfile?.avatar || user?.photoURL || user?.avatarUrl || user?.avatar;
  const fullAvatarUrl = getAvatarUrl(avatarUrl);
  const displayName = userProfile?.displayName || (userProfile?.firstName && userProfile?.lastName ? `${userProfile.firstName} ${userProfile.lastName}` : user?.displayName || 'کاربر');
  const firstName = userProfile?.firstName || user?.firstName || '';
  const lastName = userProfile?.lastName || user?.lastName || '';

  const renderAvatar = (
    <AnimateAvatar
      width={96}
      slotProps={{
        avatar: { 
          src: fullAvatarUrl, 
          alt: `${firstName} ${lastName}`.trim() || 'کاربر'
        },
        overlay: {
          border: 2,
          spacing: 3,
          color: `linear-gradient(135deg, ${varAlpha(theme.vars.palette.primary.mainChannel, 0)} 25%, ${theme.vars.palette.primary.main} 100%)`,
        },
      }}
    >
      {displayName?.charAt(0).toUpperCase() || 'ک'}
    </AnimateAvatar>
  );

  return (
    <>
      <AccountButton
        open={open}
        onClick={handleOpenDrawer}
        photoURL={avatarUrl}
        displayName={displayName}
        sx={sx}
        {...other}
      />

      <Drawer
        open={open}
        onClose={handleCloseDrawer}
        anchor="right"
        slotProps={{ backdrop: { invisible: true } }}
        PaperProps={{ sx: { width: 320 } }}
      >
        <IconButton
          onClick={handleCloseDrawer}
          sx={{ top: 12, left: 12, zIndex: 9, position: 'absolute' }}
        >
          <Iconify icon="mingcute:close-line" />
        </IconButton>

        <Scrollbar>
          <Stack alignItems="center" sx={{ pt: 8 }}>
            {renderAvatar}

            <Typography variant="subtitle1" noWrap sx={{ mt: 2 }}>
              {displayName}
            </Typography>

            <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }} noWrap>
              {userProfile?.email || user?.email || ''}
            </Typography>

            {(userProfile?.role || user?.role) && (
              <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }} noWrap>
                {(() => {
                  const role = (userProfile?.role || user?.role || '').toUpperCase();
                  return role === 'DOCTOR' ? 'دکتر' : role === 'ADMIN' ? 'ادمین' : role === 'PATIENT' ? 'بیمار' : userProfile?.role || user?.role;
                })()}
                {userProfile?.specialty && ` - ${userProfile.specialty}`}
              </Typography>
            )}

            {userProfile?.licenseNumber && (
              <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }} noWrap>
                شماره نظام: {userProfile.licenseNumber}
              </Typography>
            )}
          </Stack>

          {/* Account switching removed - single account system for dental app */}

          <Stack
            sx={{
              py: 3,
              px: 2.5,
              borderTop: `dashed 1px ${theme.vars.palette.divider}`,
              borderBottom: `dashed 1px ${theme.vars.palette.divider}`,
            }}
          >
            {(userProfile?.role?.toUpperCase() === 'DOCTOR' || user?.role?.toUpperCase() === 'DOCTOR') && (
              <>
                <MenuItem
                  onClick={() => handleClickItem(paths.dashboard.orthodontics)}
                  sx={{
                    py: 1,
                    color: 'text.secondary',
                    '& svg': { width: 24, height: 24 },
                    '&:hover': { color: 'text.primary' },
                  }}
                >
                  <Iconify icon="solar:heart-bold" />
                  <Box component="span" sx={{ ml: 2 }}>
                    بخش ارتودنسی
                  </Box>
                </MenuItem>

                <MenuItem
                  onClick={() => handleClickItem(`${paths.dashboard.root  }/wallet`)}
                  sx={{
                    py: 1,
                    color: 'text.secondary',
                    '& svg': { width: 24, height: 24 },
                    '&:hover': { color: 'text.primary' },
                  }}
                >
                  <Iconify icon="solar:wallet-money-bold" />
                  <Box component="span" sx={{ ml: 2 }}>
                    کیف پول
                  </Box>
                </MenuItem>

                <MenuItem
                  onClick={() => handleClickItem(paths.dashboard.invoice.list)}
                  sx={{
                    py: 1,
                    color: 'text.secondary',
                    '& svg': { width: 24, height: 24 },
                    '&:hover': { color: 'text.primary' },
                  }}
                >
                  <Iconify icon="solar:bill-list-bold" />
                  <Box component="span" sx={{ ml: 2 }}>
                    فاکتورها
                  </Box>
                </MenuItem>
              </>
            )}


            {data.map((option) => {
              const rootLabel = pathname.includes('/dashboard') ? 'خانه' : 'Dashboard';

              const rootHref = pathname.includes('/dashboard') ? '/' : paths.dashboard.root;

              return (
                <MenuItem
                  key={option.label}
                  onClick={() => handleClickItem(option.label === 'Home' ? rootHref : option.href)}
                  sx={{
                    py: 1,
                    color: 'text.secondary',
                    '& svg': { width: 24, height: 24 },
                    '&:hover': { color: 'text.primary' },
                  }}
                >
                  {option.icon}

                  <Box component="span" sx={{ ml: 2 }}>
                    {option.label === 'Home' ? rootLabel : option.label}
                  </Box>

                  {option.info && (
                    <Label color="error" sx={{ ml: 1 }}>
                      {option.info}
                    </Label>
                  )}
                </MenuItem>
              );
            })}
          </Stack>


        </Scrollbar>

        <Box sx={{ p: 2.5 }}>
          <SignOutButton onClose={handleCloseDrawer} />
        </Box>
      </Drawer>
    </>
  );
}
