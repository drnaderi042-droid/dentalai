import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

import Card from '@mui/material/Card';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Unstable_Grid2';
import Typography from '@mui/material/Typography';

import { paths } from 'src/routes/paths';

import axios, { endpoints } from 'src/utils/axios';

import { PaymentMethods } from '../payment-methods';
import { PaymentSummary } from '../payment-summary';

// ----------------------------------------------------------------------

export function PaymentView() {
  const location = useLocation();
  const navigate = useNavigate();
  const [selectedGateway, setSelectedGateway] = useState('');
  const [exchangeRate, setExchangeRate] = useState(null);
  const [loading, setLoading] = useState(false);

  // Get payment data from location state
  const paymentData = location.state || {
    amount: 0,
    type: 'wallet_charge',
    currency: 'تومان',
  };

  useEffect(() => {
    // Fetch exchange rate from API
    const fetchExchangeRate = async () => {
      try {
        const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:7272'}/api/exchange-rate`);
        const data = await response.json();
        
        if (data.success) {
          setExchangeRate({
            usd_to_irr: data.data.usd_to_irr,
            eur_to_irr: data.data.eur_to_irr,
            last_updated: data.data.fetched_at,
          });
        } else {
          // Fallback rate
          setExchangeRate({
            usd_to_irr: 70000,
            eur_to_irr: 75000,
            last_updated: new Date().toISOString(),
          });
        }
      } catch (error) {
        console.error('Error fetching exchange rate:', error);
        // Fallback rate
        setExchangeRate({
          usd_to_irr: 70000,
          eur_to_irr: 75000,
          last_updated: new Date().toISOString(),
        });
      }
    };

    fetchExchangeRate();
  }, []);

  // Redirect if no amount
  useEffect(() => {
    if (!paymentData.amount) {
      navigate(paths.dashboard.wallet);
    }
  }, [paymentData.amount, navigate]);

  const handlePayment = async () => {
    if (!selectedGateway) {
      alert('لطفاً یک درگاه پرداخت انتخاب کنید');
      return;
    }

    setLoading(true);

    try {
      // Create invoice
      const response = await axios.post(endpoints.invoice.create, {
        amount: paymentData.amount,
        type: paymentData.type,
        paymentGateway: selectedGateway,
        description: `شارژ کیف پول به مبلغ ${paymentData.amount.toLocaleString('fa-IR')} تومان`,
        items: [
          {
            description: 'شارژ کیف پول',
            quantity: 1,
            unitPrice: paymentData.amount,
            totalPrice: paymentData.amount,
          },
        ],
      });

      if (response.data.success) {
        const invoice = response.data.data;
        console.log('Invoice created:', invoice);

        // Redirect to invoice page or payment gateway
        // TODO: Implement actual payment gateway redirect
        alert(`صورت‌حساب ${invoice.invoiceNumber} با موفقیت ایجاد شد!\n\nدر حال انتقال به درگاه ${selectedGateway === 'zarinpal' ? 'زرین‌پال' : 'NowPayments'}...`);
        
        // Navigate to invoice page
        navigate(`${paths.dashboard.invoice}/${invoice.id}`, {
          state: { invoice },
        });
      } else {
        throw new Error(response.data.error || 'خطا در ایجاد صورت‌حساب');
      }
    } catch (error) {
      console.error('Payment error:', error);
      alert(error.message || 'خطا در پردازش تراکنش. لطفاً دوباره تلاش کنید.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container sx={{ pt: 5, pb: 10 }}>
      {/* Header */}
      <Typography variant="h3" align="center" sx={{ mb: 2 }}>
        پرداخت
      </Typography>



      <Grid container spacing={3}>
        {/* Payment Methods */}
        <Grid xs={12} md={8}>
          <Card sx={{ p: 4 }}>
            <PaymentMethods
              onSelectGateway={setSelectedGateway}
              selectedGateway={selectedGateway}
              exchangeRate={exchangeRate}
              amount={paymentData.amount}
            />

            <Button
              variant="contained"
              size="large"
              fullWidth
              onClick={handlePayment}
              disabled={!selectedGateway || loading}
              sx={{ mt: 4 }}
            >
              {loading ? 'در حال پردازش...' : 'پرداخت'}
            </Button>
          </Card>
        </Grid>

        {/* Summary */}
        <Grid xs={12} md={4}>
          <PaymentSummary
            amount={paymentData.amount}
            currency={paymentData.currency}
            type={paymentData.type}
            selectedGateway={selectedGateway}
            exchangeRate={exchangeRate}
          />
        </Grid>
      </Grid>
    </Container>
  );
}
