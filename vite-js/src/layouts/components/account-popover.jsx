import { useState, useEffect, useCallback } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Popover from '@mui/material/Popover';
import MenuItem from '@mui/material/MenuItem';
import { useTheme } from '@mui/material/styles';
import Typography from '@mui/material/Typography';

import { paths } from 'src/routes/paths';
import { useRouter, usePathname } from 'src/routes/hooks';

import axios from 'src/utils/axios';

import { CONFIG } from 'src/config-global';
import { varAlpha } from 'src/theme/styles';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';
import { AnimateAvatar } from 'src/components/animate';
import { usePopover } from 'src/components/custom-popover';

import { useAuthContext } from 'src/auth/hooks';

import { AccountButton } from './account-button';
import { SignOutButton } from './sign-out-button';

// ----------------------------------------------------------------------

export function AccountPopover({ data = [], sx, ...other }) {
  const theme = useTheme();
  const popover = usePopover();
  const router = useRouter();
  const pathname = usePathname();
  const { user } = useAuthContext();

  const [userProfile, setUserProfile] = useState(null);
  const [loading, setLoading] = useState(false);

  // Fetch user profile when popover opens
  useEffect(() => {
    const fetchUserProfile = async () => {
      if (popover.open && user?.accessToken) {
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
  }, [popover.open, user]);

  const handleClickItem = useCallback(
    (path) => {
      popover.onClose();
      router.push(path);
    },
    [popover, router]
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

  return (
    <>
      <AccountButton
        open={popover.open}
        onClick={popover.onOpen}
        photoURL={avatarUrl}
        displayName={displayName}
        sx={sx}
        {...other}
      />

      <Popover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: { sx: { width: 200, p: 0 } },
        }}
      >
        <Stack alignItems="center" sx={{ p: 2, pt: 3 }}>
          <AnimateAvatar
            width={64}
            slotProps={{
              avatar: { 
                src: fullAvatarUrl, 
                alt: `${firstName} ${lastName}`.trim() || 'کاربر'
              },
              overlay: {
                border: 2,
                spacing: 2,
                color: `linear-gradient(135deg, ${varAlpha(theme.vars.palette.primary.mainChannel, 0)} 25%, ${theme.vars.palette.primary.main} 100%)`,
              },
            }}
          >
            {displayName?.charAt(0).toUpperCase() || 'ک'}
          </AnimateAvatar>

          <Typography variant="subtitle2" noWrap sx={{ mt: 1.5 }}>
            {displayName}
          </Typography>





        </Stack>

        <Divider />

        <Stack sx={{ py: 1 }}>
          {(userProfile?.role?.toUpperCase() === 'DOCTOR' || user?.role?.toUpperCase() === 'DOCTOR') && (
            <>
              <MenuItem
                onClick={() => handleClickItem(paths.dashboard.orthodontics)}
                sx={{
                  py: 0.5,
                  px: 2,
                  minHeight: 36,
                  '& svg': { width: 20, height: 20, color: 'var(--palette-text-secondary)' },
                }}
              >
                <Iconify icon="solar:heart-bold" />
                <Box component="span" sx={{ ml: 2 }}>
                  بخش ارتودنسی
                </Box>
              </MenuItem>

              <MenuItem
                onClick={() => handleClickItem(`${paths.dashboard.root}/wallet`)}
                sx={{
                  py: 0.5,
                  px: 2,
                  minHeight: 36,
                  '& svg': { width: 20, height: 20, color: 'var(--palette-text-secondary)' },
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
                  py: 0.5,
                  px: 2,
                  minHeight: 36,
                  '& svg': { width: 20, height: 20, color: 'var(--palette-text-secondary)' },
                }}
              >
                <Iconify icon="solar:bill-list-bold" />
                <Box component="span" sx={{ ml: 2 }}>
                  فاکتورها
                </Box>
              </MenuItem>
            </>
          )}


          {data
            .filter((option) => {
              // نمایش دکمه "فاکتور جدید" فقط برای ادمین
              if (option.label === 'فاکتور جدید') {
                return userProfile?.role?.toUpperCase() === 'ADMIN' || user?.role?.toUpperCase() === 'ADMIN';
              }
              return true;
            })
            .map((option) => {
              const rootLabel = pathname.includes('/dashboard') ? 'خانه' : 'Dashboard';
              const rootHref = pathname.includes('/dashboard') ? '/' : paths.dashboard.root;

              return (
                <MenuItem
                  key={option.label}
                  onClick={() => handleClickItem(option.label === 'Home' ? rootHref : option.href)}
                  sx={{
                    py: 0.5,
                    px: 2,
                    minHeight: 36,
                    '& svg': { width: 20, height: 20, color: 'var(--palette-text-secondary)' },
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

        <Divider />

        <Box sx={{ p: 1.5 }}>
          <SignOutButton onClose={popover.onClose} />
        </Box>
      </Popover>
    </>
  );
}
