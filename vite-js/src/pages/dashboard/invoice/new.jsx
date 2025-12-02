import { Helmet } from 'react-helmet-async';
import { Navigate } from 'react-router-dom';

import { CONFIG } from 'src/config-global';

import { InvoiceCreateView } from 'src/sections/invoice/view';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const metadata = { title: `ایجاد فاکتور جدید | داشبورد - ${CONFIG.site.name}` };

export default function Page() {
  const { user } = useAuthContext();

  // Only admins can create invoices
  const userRole = user?.role?.toUpperCase();
  if (!user || userRole !== 'ADMIN') {
    return <Navigate to="/dashboard" replace />;
  }

  return (
    <>
      <Helmet>
        <title> {metadata.title}</title>
      </Helmet>

      <InvoiceCreateView />
    </>
  );
}
