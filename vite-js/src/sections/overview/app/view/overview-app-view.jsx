import { useState, useEffect } from 'react';

import { useTheme } from '@mui/material/styles';
import Grid from '@mui/material/Unstable_Grid2';

import axios from 'src/utils/axios';

import { DashboardContent } from 'src/layouts/dashboard';
import { SeoIllustration } from 'src/assets/illustrations';

import { useAuthContext } from 'src/auth/hooks';

import { AppWelcome } from '../app-welcome';

// ----------------------------------------------------------------------

export function OverviewAppView() {
  const { user } = useAuthContext();
  const [userProfile, setUserProfile] = useState(null);
  const theme = useTheme();

  useEffect(() => {
    const fetchUserProfile = async () => {
      if (user?.accessToken) {
        try {
          const response = await axios.get('/api/auth/me', {
            headers: {
              Authorization: `Bearer ${user.accessToken}`,
            },
          });
          setUserProfile(response.data.user);
        } catch (error) {
          console.error('Error fetching user profile:', error);
        }
      }
    };

    fetchUserProfile();
  }, [user]);

  // Get user's full name
  const getUserName = () => {
    if (userProfile) {
      const firstName = userProfile.firstName || '';
      const lastName = userProfile.lastName || '';
      if (firstName || lastName) {
        return `${firstName} ${lastName}`.trim();
      }
    }
    return user?.displayName || 'کاربر';
  };

  return (
    <DashboardContent maxWidth="xl">
      <Grid container spacing={3}>
        <Grid xs={12} md={12}>
          <AppWelcome
            title={`خوش آمدید 👋 \n ${getUserName()}`}
            description="همه چیز از یک ذهن خلاق شروع می شود. به آن بیاندیشید. شاید در آینده نصیب آن شدید"
            img={<SeoIllustration hideBackground />}

          />
        </Grid>










      </Grid>
    </DashboardContent>
  );
}
