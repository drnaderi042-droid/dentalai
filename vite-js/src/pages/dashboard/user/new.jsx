import { Helmet } from 'react-helmet-async';
import { Navigate } from 'react-router-dom';

import { CONFIG } from 'src/config-global';

import { UserCreateView } from 'src/sections/user/view';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const metadata = { title: `کاربر جدید | داشبورد - ${CONFIG.site.name}` };

export default function Page() {
  const { user } = useAuthContext();

  // Check if user is admin
  const userRole = user?.role?.toUpperCase();
  if (!user || userRole !== 'ADMIN') {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <>
      <Helmet>
        <title> {metadata.title}</title>
      </Helmet>

      <UserCreateView />
    </>
  );
}

