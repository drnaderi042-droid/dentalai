import { useCallback } from 'react';

import Stack from '@mui/material/Stack';
import Avatar from '@mui/material/Avatar';
import Divider from '@mui/material/Divider';
import Tooltip from '@mui/material/Tooltip';
import MenuList from '@mui/material/MenuList';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import ListItemText from '@mui/material/ListItemText';

import { useRouter } from 'src/routes/hooks';

import { getAvatarUrl } from 'src/utils/avatar-url';

import { Iconify } from 'src/components/iconify';
import { usePopover, CustomPopover } from 'src/components/custom-popover';

import { useMockedUser, useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

export function ChatNavAccount() {
  const { user: mockedUser } = useMockedUser();
  const { user } = useAuthContext();
  const router = useRouter();

  const popover = usePopover();

  const currentUser = user || mockedUser;

  const handleProfileClick = useCallback(() => {
    popover.onClose();
    // Profile page removed
  }, [popover]);

  return (
    <>
      <Avatar
        src={getAvatarUrl(currentUser?.photoURL || currentUser?.avatarUrl)}
        alt={currentUser?.displayName}
        onClick={popover.onOpen}
        sx={{ cursor: 'pointer', width: 48, height: 48 }}
      >
        {currentUser?.displayName?.charAt(0).toUpperCase()}
      </Avatar>

      <CustomPopover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        slotProps={{
          paper: { sx: { p: 0 } },
          arrow: { placement: 'top-left' },
        }}
      >
        <Stack direction="row" alignItems="center" spacing={2} sx={{ py: 2, pr: 1, pl: 2 }}>
          <ListItemText
            primary={currentUser?.displayName || `${currentUser?.firstName || ''} ${currentUser?.lastName || ''}`.trim()}
            secondary={currentUser?.email}
            secondaryTypographyProps={{ component: 'span' }}
          />

          <Tooltip title="خروج">
            <IconButton color="error">
              <Iconify icon="ic:round-power-settings-new" />
            </IconButton>
          </Tooltip>
        </Stack>

        <Divider sx={{ borderStyle: 'dashed' }} />

        <MenuList sx={{ my: 0.5, px: 0.5 }}>
          <MenuItem onClick={handleProfileClick}>
            <Iconify icon="solar:user-id-bold" width={24} />
            پروفایل
          </MenuItem>

          <MenuItem>
            <Iconify icon="eva:settings-2-fill" width={24} />
            تنظیمات
          </MenuItem>
        </MenuList>
      </CustomPopover>
    </>
  );
}
