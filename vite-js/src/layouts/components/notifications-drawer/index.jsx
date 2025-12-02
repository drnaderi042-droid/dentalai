import { m } from 'framer-motion';
import { useState, useEffect, useCallback } from 'react';

import Tab from '@mui/material/Tab';
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Badge from '@mui/material/Badge';
import Drawer from '@mui/material/Drawer';
import Button from '@mui/material/Button';
import SvgIcon from '@mui/material/SvgIcon';
import Tooltip from '@mui/material/Tooltip';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';

import { useBoolean } from 'src/hooks/use-boolean';

import { Label } from 'src/components/label';
import { Iconify } from 'src/components/iconify';
import { varHover } from 'src/components/animate';
import { Scrollbar } from 'src/components/scrollbar';
import { CustomTabs } from 'src/components/custom-tabs';

import { useAuthContext } from 'src/auth/hooks';

import { NotificationItem } from './notification-item';

// ----------------------------------------------------------------------

const TABS = [
  { value: 'all', label: 'همه', count: 0 },
  { value: 'unread', label: 'خوانده نشده', count: 0 },
  { value: 'archived', label: 'بایگانی شده', count: 0 },
];

// ----------------------------------------------------------------------

export function NotificationsDrawer({ data = [], sx, ...other }) {
  const drawer = useBoolean();
  const { user } = useAuthContext();

  const [currentTab, setCurrentTab] = useState('all');
  const [notifications, setNotifications] = useState([]);

  const handleChangeTab = useCallback((event, newValue) => {
    setCurrentTab(newValue);
  }, []);

  // Fetch notifications
  useEffect(() => {
    const fetchNotifications = async () => {
      try {
        // Mock notifications data for dental app
        setNotifications([
          {
            id: '1',
            title: 'صورتحساب جدید صادر شده',
            type: 'invoice',
            category: 'صورتحساب',
            avatarUrl: '/assets/images/avatar/avatar_1.jpg',
            isUnRead: true,
            createdAt: new Date(),
          },
          {
            id: '2',
            title: 'بیمار احمد رضایی ارجاع داده شد',
            type: 'patient',
            category: 'ارجاع بیمار',
            avatarUrl: '/assets/images/avatar/avatar_2.jpg',
            isUnRead: true,
            createdAt: new Date(Date.now() - 1000 * 60 * 30), // 30 minutes ago
          },
          {
            id: '3',
            title: 'پیام جدید از دکتر محمدی',
            type: 'message',
            category: 'پیام',
            avatarUrl: '/assets/images/avatar/avatar_3.jpg',
            isUnRead: false,
            createdAt: new Date(Date.now() - 1000 * 60 * 60 * 2), // 2 hours ago
          },
          {
            id: '4',
            title: 'صورتحساب پرداخت شد',
            type: 'payment',
            category: 'پرداخت',
            avatarUrl: '/assets/images/avatar/avatar_4.jpg',
            isUnRead: false,
            createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24), // 1 day ago
          },
          {
            id: '5',
            title: 'بیمار نازنین احمدی ارجاع داده شد',
            type: 'patient',
            category: 'ارجاع بیمار',
            avatarUrl: '/assets/images/avatar/avatar_3.jpg',
            isUnRead: true,
            createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24 * 2), // 2 days ago
          },
        ]);
      } catch (error) {
        console.error('Error fetching notifications:', error);
      }
    };

    if (user) {
      fetchNotifications();
    }
  }, [user]);

  const totalUnRead = notifications.filter((item) => item.isUnRead === true).length;

  // Update tab counts dynamically inside the component
  const updatedTabs = [
    { value: 'all', label: 'همه', count: notifications.length },
    { value: 'unread', label: 'خوانده نشده', count: totalUnRead },
    { value: 'archived', label: 'بایگانی شده', count: Math.max(0, notifications.length - totalUnRead) },
  ];

  const handleMarkAllAsRead = () => {
    setNotifications(notifications.map(notification => ({ ...notification, isUnRead: false })));
  };

  const renderHead = (
    <Stack direction="row" alignItems="center" sx={{ py: 2, pl: 2.5, pr: 1, minHeight: 68 }}>
      <Typography variant="h6" sx={{ flexGrow: 1 }}>
        اعلان‌ها
      </Typography>

      {!!totalUnRead && (
        <Tooltip title="همه را خوانده شده علامت بزن">
          <IconButton color="primary" onClick={handleMarkAllAsRead}>
            <Iconify icon="eva:done-all-fill" />
          </IconButton>
        </Tooltip>
      )}

      <IconButton onClick={drawer.onFalse} sx={{ display: { xs: 'inline-flex', sm: 'none' } }}>
        <Iconify icon="mingcute:close-line" />
      </IconButton>

      <IconButton>
        <Iconify icon="solar:settings-bold-duotone" />
      </IconButton>
    </Stack>
  );

  const renderTabs = (
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
  );

  const renderList = (
    <Scrollbar>
      <Box component="ul">
        {notifications?.filter((notification) => {
          if (currentTab === 'all') return true;
          if (currentTab === 'unread') return notification.isUnRead;
          if (currentTab === 'archived') return !notification.isUnRead;
          return true;
        }).map((notification) => (
          <Box component="li" key={notification.id} sx={{ display: 'flex' }}>
            <NotificationItem notification={notification} />
          </Box>
        ))}
      </Box>
    </Scrollbar>
  );

  return (
    <>
      <IconButton
        component={m.button}
        whileTap="tap"
        whileHover="hover"
        variants={varHover(1.05)}
        onClick={drawer.onTrue}
        sx={sx}
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

      <Drawer
        open={drawer.value}
        onClose={drawer.onFalse}
        anchor="right"
        slotProps={{ backdrop: { invisible: true } }}
        PaperProps={{ sx: { width: 1, maxWidth: 420 } }}
      >
        {renderHead}

        {renderTabs}

        {renderList}

        <Box sx={{ p: 1 }}>
          <Button fullWidth size="large">
            View all
          </Button>
        </Box>
      </Drawer>
    </>
  );
}
