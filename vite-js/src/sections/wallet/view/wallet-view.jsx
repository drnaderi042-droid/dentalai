import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Grid from '@mui/material/Grid';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import Container from '@mui/material/Container';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';

import { paths } from 'src/routes/paths';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

const QUICK_AMOUNTS = [
  { value: 50000, label: '۵۰ هزار تومان' },
  { value: 100000, label: '۱۰۰ هزار تومان' },
  { value: 200000, label: '۲۰۰ هزار تومان' },
  { value: 500000, label: '۵۰۰ هزار تومان' },
  { value: 1000000, label: '۱ میلیون تومان' },
  { value: 2000000, label: '۲ میلیون تومان' },
];

export function WalletView() {
  const navigate = useNavigate();
  const [depositAmount, setDepositAmount] = useState('');
  const [error, setError] = useState('');
  const [walletData, setWalletData] = useState({
    balance: 0,
    currency: 'تومان',
    recentTransactions: [],
  });

  const formatCurrency = (amount) => new Intl.NumberFormat('fa-IR').format(Math.abs(amount));

  const handleDeposit = () => {
    const amount = parseInt(depositAmount, 10);
    
    console.log('[Wallet] Deposit clicked:', { depositAmount, amount });
    
    if (!depositAmount || Number.isNaN(amount)) {
      setError('لطفا مبلغ را وارد کنید');
      return;
    }

    if (amount < 10000) {
      setError('حداقل مبلغ شارژ ۱۰,۰۰۰ تومان است');
      return;
    }

    console.log('[Wallet] Navigating to payment with:', {
      path: paths.dashboard.payment,
      amount,
      type: 'wallet_charge',
    });

    // Navigate to payment page with amount
    navigate(paths.dashboard.payment, {
      state: {
        amount,
        type: 'wallet_charge',
        currency: 'تومان',
      },
    });
  };

  return (
    <Container maxWidth="lg" sx={{ py: 5 }}>
      {/* Header */}
      <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 5 }}>
        <Box
          sx={{
            width: 56,
            height: 56,
            borderRadius: 2,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
        >
          <Iconify icon="solar:wallet-bold-duotone" width={32} sx={{ color: 'var(--palette-text-secondary)' }} />
        </Box>
        <Box>
          <Typography variant="h4">کیف پول</Typography>
          <Typography variant="body2" color="text.secondary">
            مدیریت موجودی و تراکنش‌های مالی
          </Typography>
        </Box>
      </Stack>

      <Grid container spacing={3}>
        {/* Balance Card */}
        <Grid item xs={12} md={4}>
          <Card
            sx={{
              p: 3,
              height: '100%',
            }}
          >
            <Stack direction="row" alignItems="center" spacing={1} sx={{ mb: 3 }}>
              <Iconify icon="solar:wallet-2-bold" width={24} sx={{ color: 'var(--palette-text-secondary)' }} />
              <Typography variant="subtitle2" color="text.secondary">
                موجودی فعلی
              </Typography>
            </Stack>

            <Stack direction="row" alignItems="baseline" spacing={1}>
              <Typography variant="h3" sx={{ fontWeight: 700 }}>
                {formatCurrency(walletData.balance)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {walletData.currency}
              </Typography>
            </Stack>



          </Card>
        </Grid>

        {/* Charge Card */}
        <Grid item xs={12} md={8}>
          <Card sx={{ p: 4, height: '100%' }}>
            <Stack direction="row" alignItems="center" spacing={2} sx={{ mb: 3 }}>
              <Iconify icon="solar:card-bold-duotone" width={28} sx={{ color: 'var(--palette-text-secondary)' }} />
              <Typography variant="h5">شارژ کیف پول</Typography>
            </Stack>

            {error && (
              <Alert severity="error" sx={{ mb: 3 }} onClose={() => setError('')}>
                {error}
              </Alert>
            )}

            <Stack spacing={3}>
              <TextField
                fullWidth
                label="مبلغ شارژ (تومان)"
                type="number"
                value={depositAmount}
                onChange={(e) => {
                  setDepositAmount(e.target.value);
                  setError('');
                }}
                placeholder="مثلاً: 100000"
                helperText="حداقل مبلغ شارژ: ۱۰,۰۰۰ تومان"
                InputProps={{
                  startAdornment: (
                    <Iconify icon="solar:money-bag-bold" width={24} sx={{ mr: 1, color: 'var(--palette-text-secondary)' }} />
                  ),
                }}
              />

              {/* Quick Amount Buttons */}
              <Box>
                <Typography variant="subtitle2" sx={{ mb: 2 }}>
                  انتخاب سریع مبلغ:
                </Typography>
                <Grid container spacing={1.5}>
                  {QUICK_AMOUNTS.map((item) => (
                    <Grid item xs={6} sm={4} key={item.value}>
                      <Chip
                        label={item.label}
                        variant={depositAmount === String(item.value) ? 'filled' : 'outlined'}
                        color={depositAmount === String(item.value) ? 'primary' : 'default'}
                        onClick={() => {
                          setDepositAmount(String(item.value));
                          setError('');
                        }}
                        sx={{
                          width: '100%',
                          height: 40,
                          fontSize: '0.875rem',
                          cursor: 'pointer',
                        }}
                      />
                    </Grid>
                  ))}
                </Grid>
              </Box>

              <Button
                variant="contained"
                size="large"
                fullWidth
                onClick={handleDeposit}
                disabled={!depositAmount}
                
                sx={{
                  py: 1.5,
                  fontSize: '0.9rem',
                  fontWeight: 400,
                }}
              >
                ادامه و انتخاب درگاه پرداخت
              </Button>
            </Stack>
          </Card>
        </Grid>
      </Grid>

      {/* Recent Transactions */}
      <Card sx={{ p: 4, mt: 3 }}>
        <Stack direction="row" alignItems="center" justifyContent="space-between" sx={{ mb: 3 }}>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Iconify icon="solar:history-bold-duotone" width={28} sx={{ color: 'var(--palette-text-secondary)' }} />
            <Typography variant="h5">تراکنش‌های اخیر</Typography>
          </Stack>
          <Chip
            label={`${walletData.recentTransactions.length} تراکنش`}
            size="small"
            color="primary"
            variant="outlined"
          />
        </Stack>

        <Stack spacing={2}>
          {walletData.recentTransactions.map((transaction, index) => (
            <Box key={transaction.id}>
              <Stack direction="row" alignItems="center" spacing={2}>
                <Box
                  sx={{
                    width: 48,
                    height: 48,
                    borderRadius: 2,

                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}
                >
                  <Iconify
                    icon={
                      transaction.type === 'deposit'
                        ? 'solar:arrow-down-bold'
                        : transaction.type === 'payment'
                          ? 'solar:arrow-up-bold'
                          : 'solar:refresh-bold'
                    }
                    width={24}
                    sx={{ color: 'var(--palette-text-secondary)' }}
                  />
                </Box>

                <Box sx={{ flex: 1 }}>
                  <Typography variant="body1" fontWeight={600} sx={{ mb: 0.5 }}>
                    {transaction.description}
                  </Typography>
                  <Stack direction="row" alignItems="center" spacing={1}>
                    <Iconify icon="solar:calendar-bold" width={14} sx={{ color: 'var(--palette-text-secondary)' }} />
                    <Typography variant="caption" color="text.secondary">
                      {transaction.date}
                    </Typography>
                    <Chip
                      label={transaction.status === 'completed' ? 'تکمیل شده' : 'در انتظار'}
                      size="small"
                      color={transaction.status === 'completed' ? 'success' : 'warning'}
                      sx={{ height: 20, fontSize: '0.65rem' }}
                    />
                  </Stack>
                </Box>

                <Box sx={{ textAlign: 'left' }}>
                  <Typography
                    variant="h6"
                    color={transaction.amount > 0 ? 'success.main' : 'error.main'}
                    fontWeight="bold"
                  >
                    {transaction.amount > 0 ? '+' : ''}{formatCurrency(transaction.amount)}
                  </Typography>
                  <Typography variant="caption" color="text.secondary">
                    {walletData.currency}
                  </Typography>
                </Box>
              </Stack>

              {index < walletData.recentTransactions.length - 1 && <Divider sx={{ my: 2 }} />}
            </Box>
          ))}
        </Stack>


      </Card>
    </Container>
  );
}
