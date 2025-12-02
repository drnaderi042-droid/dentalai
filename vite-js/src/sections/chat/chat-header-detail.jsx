import { useState, useCallback } from 'react';

import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Avatar from '@mui/material/Avatar';
import Divider from '@mui/material/Divider';
import MenuList from '@mui/material/MenuList';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import ListItemText from '@mui/material/ListItemText';
import AvatarGroup, { avatarGroupClasses } from '@mui/material/AvatarGroup';

import { useResponsive } from 'src/hooks/use-responsive';

import { fToNow } from 'src/utils/format-time';
import { getAvatarUrl } from 'src/utils/avatar-url';
import axiosInstance, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';
import { ConfirmDialog } from 'src/components/custom-dialog';
import { usePopover, CustomPopover } from 'src/components/custom-popover';

import { useAuthContext } from 'src/auth/hooks';

import { ChatHeaderSkeleton } from './chat-skeleton';
import { ChatReportDialog } from './chat-report-dialog';

// ----------------------------------------------------------------------

export function ChatHeaderDetail({ collapseNav, participants, loading }) {
  const popover = usePopover();
  const { user } = useAuthContext();
  const [blockConfirmOpen, setBlockConfirmOpen] = useState(false);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);
  const [blocking, setBlocking] = useState(false);

  const lgUp = useResponsive('up', 'lg');

  const group = participants.length > 1;

  const singleParticipant = participants[0];

  const { collapseDesktop, onCollapseDesktop, onOpenMobile } = collapseNav;

  const handleToggleNav = useCallback(() => {
    if (lgUp) {
      onCollapseDesktop();
    } else {
      onOpenMobile();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lgUp]);

  const handleBlockClick = useCallback(() => {
    popover.onClose();
    setBlockConfirmOpen(true);
  }, [popover]);

  const handleBlockConfirm = useCallback(async () => {
    if (!singleParticipant?.id || !user?.accessToken) {
      return;
    }

    try {
      setBlocking(true);
      await axiosInstance.post(
        endpoints.chat.block,
        { blockedId: singleParticipant.id },
        {
          headers: {
            Authorization: `Bearer ${user.accessToken}`,
          },
        }
      );

      toast.success('کاربر با موفقیت مسدود شد');
      setBlockConfirmOpen(false);
      // Optionally redirect or refresh chat list
      window.location.href = '/dashboard/chat';
    } catch (error) {
      console.error('Error blocking user:', error);
      const errorMessage = error.response?.data?.message || error.message || 'خطا در مسدود کردن کاربر';
      toast.error(errorMessage);
    } finally {
      setBlocking(false);
    }
  }, [singleParticipant, user]);

  const handleReportClick = useCallback(() => {
    popover.onClose();
    setReportDialogOpen(true);
  }, [popover]);

  const renderGroup = (
    <AvatarGroup max={3} sx={{ [`& .${avatarGroupClasses.avatar}`]: { width: 32, height: 32 } }}>
      {participants.map((participant) => (
        <Avatar key={participant.id} alt={participant.name} src={getAvatarUrl(participant.avatarUrl)} />
      ))}
    </AvatarGroup>
  );

  const renderSingle = (
    <Stack direction="row" alignItems="center" spacing={2}>
      <Avatar src={getAvatarUrl(singleParticipant?.avatarUrl)} alt={singleParticipant?.name} />

      <ListItemText
        primary={singleParticipant?.name}
        secondary={
          singleParticipant?.lastActivity
            ? `آخرین بازدید: ${fToNow(singleParticipant.lastActivity)}`
            : 'آخرین بازدید: نامشخص'
        }
        secondaryTypographyProps={{
          component: 'span',
        }}
      />
    </Stack>
  );

  if (loading) {
    return <ChatHeaderSkeleton />;
  }

  return (
    <>
      {group ? renderGroup : renderSingle}

      <Stack direction="row" flexGrow={1} justifyContent="flex-end">


        <IconButton onClick={handleToggleNav}>
          <Iconify icon={!collapseDesktop ? 'ri:sidebar-unfold-fill' : 'ri:sidebar-fold-fill'} />
        </IconButton>

        <IconButton onClick={popover.onOpen}>
          <Iconify icon="eva:more-vertical-fill" />
        </IconButton>
      </Stack>

      <CustomPopover open={popover.open} anchorEl={popover.anchorEl} onClose={popover.onClose}>
        <MenuList>


          <MenuItem onClick={handleBlockClick}>
            <Iconify icon="solar:forbidden-circle-bold" />
            مسدود کردن
          </MenuItem>

          <MenuItem onClick={handleReportClick}>
            <Iconify icon="solar:danger-triangle-bold" />
            گزارش
          </MenuItem>

          <Divider sx={{ borderStyle: 'dashed' }} />

          <MenuItem
            onClick={() => {
              popover.onClose();
            }}
            sx={{ color: 'error.main' }}
          >
            <Iconify icon="solar:trash-bin-trash-bold" />
            حذف
          </MenuItem>
        </MenuList>
      </CustomPopover>

      <ConfirmDialog
        open={blockConfirmOpen}
        onClose={() => setBlockConfirmOpen(false)}
        title="مسدود کردن کاربر"
        content={
          <>
            آیا از مسدود کردن <strong>{singleParticipant?.name}</strong> مطمئن هستید؟
            <br />
            پس از مسدود کردن، امکان ارسال یا دریافت پیام از این کاربر را نخواهید داشت.
          </>
        }
        action={
          <Button
            variant="contained"
            color="error"
            onClick={handleBlockConfirm}
            disabled={blocking}
          >
            {blocking ? 'در حال مسدود کردن...' : 'مسدود کردن'}
          </Button>
        }
      />

      <ChatReportDialog
        open={reportDialogOpen}
        onClose={() => setReportDialogOpen(false)}
        reportedUserId={singleParticipant?.id}
        reportedUserName={singleParticipant?.name}
      />
    </>
  );
}
