import { Helmet } from 'react-helmet-async';
import { Navigate } from 'react-router-dom';

import { CONFIG } from 'src/config-global';

import { ChatView } from 'src/sections/chat/view';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const metadata = { title: `چت | داشبورد - ${CONFIG.site.name}` };

export default function Page() {
  const { user } = useAuthContext();

  // Allow access for DOCTOR and ADMIN roles
  const userRole = user?.role?.toUpperCase();
  if (!user || (userRole !== 'DOCTOR' && userRole !== 'ADMIN')) {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <>
      <Helmet>
        <title> {metadata.title}</title>
      </Helmet>

      <ChatView />
    </>
  );
}
