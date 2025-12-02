import { toast } from 'sonner';
import { m } from 'framer-motion';
import { useState, useEffect, useCallback } from 'react';

import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Badge from '@mui/material/Badge';
import Popover from '@mui/material/Popover';
import SvgIcon from '@mui/material/SvgIcon';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import CircularProgress from '@mui/material/CircularProgress';

import axios, { endpoints } from 'src/utils/axios';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';
import { varHover } from 'src/components/animate';
import { Scrollbar } from 'src/components/scrollbar';
import { CustomTabs } from 'src/components/custom-tabs';
import { usePopover } from 'src/components/custom-popover';

import { useAuthContext } from 'src/auth/hooks';

import { NotificationItem } from './notifications-drawer/notification-item';

// ----------------------------------------------------------------------

const TABS = [
  { value: 'all', label: 'همه', count: 0 },
  { value: 'unread', label: 'خوانده نشده', count: 0 },
  { value: 'archived', label: 'بایگانی شده', count: 0 },
];

// ----------------------------------------------------------------------

export function NotificationsPopover({ sx, ...other }) {
  const popover = usePopover();
  const { user } = useAuthContext();
  const [currentTab, setCurrentTab] = useState('all');
  const [notifications, setNotifications] = useState([]);
  const [loading, setLoading] = useState(false);
  const [markingAsRead, setMarkingAsRead] = useState(false);

  const handleChangeTab = useCallback((event, newValue) => {
    setCurrentTab(newValue);
  }, []);

  // Fetch notifications from API
  const fetchNotifications = useCallback(async () => {
    if (!user?.accessToken) return;

    try {
      setLoading(true);
      const response = await axios.get(endpoints.notifications.list, {
        headers: {
          Authorization: `Bearer ${user.accessToken}`,
        },
      });
      setNotifications(response.data.notifications || response.data || []);
    } catch (error) {
      console.error('Error fetching notifications:', error);
      setNotifications([]);
    } finally {
      setLoading(false);
    }
  }, [user?.accessToken]);

  // Fetch notifications when popover opens
  useEffect(() => {
    if (popover.open && user?.accessToken) {
      fetchNotifications();
    }
  }, [popover.open, user?.accessToken, fetchNotifications]);

  const totalUnRead = notifications.filter((item) => item.isUnRead === true).length;

  // Update tab counts dynamically inside the component
  const updatedTabs = [
    { value: 'all', label: 'همه', count: notifications.length },
    { value: 'unread', label: 'خوانده نشده', count: totalUnRead },
  ];

  const handleMarkAllAsRead = useCallback(async () => {
    if (!user?.accessToken || markingAsRead) return;

    try {
      setMarkingAsRead(true);
      await axios.post(
        endpoints.notifications.markAllAsRead,
        {},
        {
          headers: {
            Authorization: `Bearer ${user.accessToken}`,
          },
        }
      );
      
      // Refresh notifications after marking as read
      await fetchNotifications();
      toast.success('همه اعلان‌ها به عنوان خوانده شده علامت زده شدند');
    } catch (error) {
      console.error('Error marking all notifications as read:', error);
      toast.error('خطا در علامت‌گذاری اعلان‌ها');
    } finally {
      setMarkingAsRead(false);
    }
  }, [user?.accessToken, markingAsRead, fetchNotifications]);

  const filteredNotifications = notifications?.filter((notification) => {
    if (currentTab === 'all') return true;
    if (currentTab === 'unread') return notification.isUnRead === true;
    if (currentTab === 'archived') return notification.isUnRead === false;
    return true;
  });

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
        <Badge badgeContent={totalUnRead} color="error">
          <SvgIcon>
            {/* https://icon-sets.iconify.design/solar/bell-bing-bold-duotone/ */}
            <path
              fill="currentColor"
              d="M18.75 9v.704c0 .845.24 1.671.692 2.374l1.108 1.723c1.011 1.574.239 3.713-1.52 4.21a25.794 25.794 0 0 1-14.06 0c-1.759-.497-2.531-2.636-1.52-4.21l1.108-1.723a4.393 4.393 0 0 0 .693-2.374V9c0-3.866 3.022-7 6.749-7s6.75 3.134 6.75 7"
              opacity="0.5"
            />
            <path
              fill="currentColor"
              d="M12.75 6a.75.75 0 0 0-1.5 0v4a.75.75 0 0 0 1.5 0zM7.243 18.545a5.002 5.002 0 0 0 9.513 0c-3.145.59-6.367.59-9.513 0"
            />
          </SvgIcon>
        </Badge>
      </IconButton>

      <Popover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
        slotProps={{
          paper: { sx: { width: 360, maxWidth: '90vw', p: 0 } },
        }}
      >
        {/* Header */}
        <Stack direction="row" alignItems="center" sx={{ py: 2, pl: 2.5, pr: 1, minHeight: 68 }}>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            اعلان‌ها
          </Typography>

          {!!totalUnRead && (
            <Tooltip title="همه را خوانده شده علامت بزن">
              <IconButton 
                color="primary" 
                onClick={handleMarkAllAsRead} 
                size="small"
                disabled={markingAsRead}
              >
                {markingAsRead ? (
                  <CircularProgress size={20} />
                ) : (
                  <Iconify icon="eva:done-all-fill" />
                )}
              </IconButton>
            </Tooltip>
          )}

        </Stack>

        {/* Tabs */}
        <CustomTabs variant="fullWidth" value={currentTab} onChange={handleChangeTab}>
          {updatedTabs.map((tab) => (
            <Tab
              key={tab.value}
              iconPosition="end"
              value={tab.value}
              label={tab.label}
              icon={
                <Label
                  variant={((tab.value === 'all' || tab.value === currentTab) && 'filled') || 'soft'}
                  color={
                    (tab.value === 'unread' && 'info') ||
                    (tab.value === 'archived' && 'success') ||
                    'default'
                  }
                >
                  {tab.count}
                </Label>
              }
            />
          ))}
        </CustomTabs>

        {/* Notifications List */}
        <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
          {loading ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
              <CircularProgress size={32} />
            </Box>
          ) : filteredNotifications.length === 0 ? (
            <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary">
                اعلانی وجود ندارد
              </Typography>
            </Box>
          ) : (
            <Scrollbar>
              <Box component="ul" sx={{ p: 0, m: 0, listStyle: 'none' }}>
                {filteredNotifications.map((notification) => (
                  <Box component="li" key={notification.id} sx={{ display: 'flex' }}>
                    <NotificationItem notification={notification} />
                  </Box>
                ))}
              </Box>
            </Scrollbar>
          )}
        </Box>

        {/* Footer */}

      </Popover>
    </>
  );
}

