import Tab from '@mui/material/Tab';
import Tabs from '@mui/material/Tabs';

import { paths } from 'src/routes/paths';

import { useTabs } from 'src/hooks/use-tabs';

import { DashboardContent } from 'src/layouts/dashboard';

import { Iconify } from 'src/components/iconify';
import { CustomBreadcrumbs } from 'src/components/custom-breadcrumbs';

import { AccountGeneral } from '../account-general';
import { AccountChangePassword } from '../account-change-password';

// ----------------------------------------------------------------------

const TABS = [
  { value: 'general', label: 'اطلاعات حساب', icon: <Iconify icon="solar:user-id-bold" width={24} /> },
  { value: 'security', label: 'امنیت', icon: <Iconify icon="ic:round-vpn-key" width={24} /> },
];

// ----------------------------------------------------------------------

export function AccountView() {
  const tabs = useTabs('general');

  return (
    <DashboardContent>
      <CustomBreadcrumbs
        heading="ویرایش حساب کاربری"
        links={[
          { name: 'داشبورد', href: paths.dashboard.root },
          { name: 'حساب کاربری' },
        ]}
        sx={{ mb: { xs: 3, md: 5 } }}
      />

      <Tabs 
        value={tabs.value} 
        onChange={tabs.onChange} 
        sx={{ 
          mb: { xs: 3, md: 5 },
          '& .MuiTab-root': {
            transition: 'none !important',
            transform: 'none !important',
            '&:active': {
              transform: 'none !important',
            },
            '&:hover': {
              transform: 'none !important',
            },
          },
        }}
      >
        {TABS.map((tab) => (
          <Tab 
            key={tab.value} 
            label={tab.label} 
            icon={tab.icon} 
            value={tab.value}
            sx={{
              transition: 'none !important',
              transform: 'none !important',
              '&:active': {
                transform: 'none !important',
              },
              '&:hover': {
                transform: 'none !important',
              },
            }}
          />
        ))}
      </Tabs>

      {tabs.value === 'general' && <AccountGeneral />}

      {tabs.value === 'security' && <AccountChangePassword />}
    </DashboardContent>
  );
}
