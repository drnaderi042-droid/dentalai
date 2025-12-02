import { useState, useEffect } from 'react';
import { Helmet } from 'react-helmet-async';

import { useParams } from 'src/routes/hooks';

import axiosInstance, { endpoints } from 'src/utils/axios';

import { CONFIG } from 'src/config-global';

import { InvoiceEditView } from 'src/sections/invoice/view';

// ----------------------------------------------------------------------

const metadata = { title: `ویرایش فاکتور | داشبورد - ${CONFIG.site.name}` };

export default function Page() {
  const { id = '' } = useParams();
  const [currentInvoice, setCurrentInvoice] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchInvoice = async () => {
      if (!id) {
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const response = await axiosInstance.get(endpoints.invoice.details(id));
        setCurrentInvoice(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching invoice:', err);
        setError(err.message || 'خطا در دریافت اطلاعات فاکتور');
        setCurrentInvoice(null);
      } finally {
        setLoading(false);
      }
    };

    fetchInvoice();
  }, [id]);

  return (
    <>
      <Helmet>
        <title> {metadata.title}</title>
      </Helmet>

      <InvoiceEditView invoice={currentInvoice} loading={loading} error={error} />
    </>
  );
}
