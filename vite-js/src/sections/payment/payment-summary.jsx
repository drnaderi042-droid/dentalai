import PropTypes from 'prop-types';

import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Typography from '@mui/material/Typography';

// ----------------------------------------------------------------------

export function PaymentSummary({ amount, currency, type, selectedGateway, exchangeRate }) {
  const formatCurrency = (value) => new Intl.NumberFormat('fa-IR').format(value);

  const getGatewayName = () => {
    if (selectedGateway === 'zarinpal') return 'زرین‌پال';
    if (selectedGateway === 'nowpayments') return 'NowPayments';
    return 'انتخاب نشده';
  };

  const getTransactionType = () => {
    if (type === 'wallet_charge') return 'شارژ کیف پول';
    return 'پرداخت';
  };

  const getAmountInUSD = () => {
    if (!exchangeRate || !amount) return null;
    const usdAmount = amount / exchangeRate.usd_to_irr;
    return usdAmount.toFixed(2);
  };

  // Calculate fee (example: 1% for zarinpal, 0.5% for nowpayments)
  const getFee = () => {
    if (!selectedGateway || !amount) return 0;
    const feePercent = selectedGateway === 'zarinpal' ? 0.01 : 0.005;
    return Math.round(amount * feePercent);
  };

  const getTotalAmount = () => amount + getFee();
  
  const fee = getFee();

  return (
    <Card sx={{ p: 3, position: 'sticky', top: 100 }}>
      <Stack spacing={3}>
        {/* Header */}
        <Typography variant="h6">خلاصه سفارش</Typography>

        <Divider sx={{ borderStyle: 'dashed' }} />

        {/* Transaction Details */}
        <Stack spacing={2}>
          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">
              نوع تراکنش
            </Typography>
            <Typography variant="subtitle2">{getTransactionType()}</Typography>
          </Stack>

          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">
              درگاه پرداخت
            </Typography>
            <Typography variant="subtitle2" color={selectedGateway ? 'text.primary' : 'text.disabled'}>
              {getGatewayName()}
            </Typography>
          </Stack>

          <Divider sx={{ borderStyle: 'dashed' }} />

          <Stack direction="row" justifyContent="space-between">
            <Typography variant="body2" color="text.secondary">
              مبلغ
            </Typography>
            <Typography variant="subtitle2">
              {formatCurrency(amount)} {currency}
            </Typography>
          </Stack>

          {selectedGateway && fee > 0 && (
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                کارمزد درگاه
              </Typography>
              <Typography variant="subtitle2" color="text.secondary">
                {formatCurrency(fee)} {currency}
              </Typography>
            </Stack>
          )}

          {selectedGateway === 'nowpayments' && exchangeRate && (
            <Stack direction="row" justifyContent="space-between">
              <Typography variant="body2" color="text.secondary">
                معادل USD
              </Typography>
              <Typography variant="subtitle2" color="info.main">
                ${getAmountInUSD()}
              </Typography>
            </Stack>
          )}

          <Divider sx={{ borderStyle: 'dashed' }} />

          <Stack direction="row" justifyContent="space-between">
            <Typography variant="subtitle1">مبلغ کل</Typography>
            <Typography variant="h5" color="error.main">
              {formatCurrency(getTotalAmount())} {currency}
            </Typography>
          </Stack>
        </Stack>
      </Stack>
    </Card>
  );
}

PaymentSummary.propTypes = {
  amount: PropTypes.number.isRequired,
  currency: PropTypes.string.isRequired,
  type: PropTypes.string,
  selectedGateway: PropTypes.string,
  exchangeRate: PropTypes.object,
};
