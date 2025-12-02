import { useState } from 'react';
import PropTypes from 'prop-types';

import Box from '@mui/material/Box';
import Paper from '@mui/material/Paper';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import ListItemText from '@mui/material/ListItemText';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

const PAYMENT_GATEWAYS = [
  {
    value: 'zarinpal',
    label: 'زرین‌پال',
    description: 'پرداخت با کارت‌های بانکی ایرانی',
    logo: '/payment-icons/zarinpal.avif',
    icon: 'mdi:credit-card-outline',
  },
  {
    value: 'zibal',
    label: 'زیبال',
    description: 'پرداخت با کارت‌های بانکی ایرانی',
    logo: '/payment-icons/zibal.svg',
    icon: 'mdi:credit-card-outline',
  },
  {
    value: 'nowpayments',
    label: 'NowPayments',
    description: 'پرداخت با ارزهای دیجیتال',
    logo: '/payment-icons/40f189daa5c0279718484ca5f5569f78-crypto-icon.webp',
    icon: 'cryptocurrency:btc',
  },
];

export function PaymentMethods({ onSelectGateway, selectedGateway, exchangeRate, amount }) {
  const formatCurrency = (value) => new Intl.NumberFormat('fa-IR').format(value);

  const getAmountInUSD = () => {
    if (!exchangeRate || !amount) return null;
    const usdAmount = amount / exchangeRate.usd_to_irr;
    return usdAmount.toFixed(2);
  };

  return (
    <Stack spacing={3}>
      <Typography variant="h6">روش پرداخت</Typography>

      <Stack spacing={2}>
        {PAYMENT_GATEWAYS.map((gateway) => (
          <GatewayOption
            key={gateway.value}
            gateway={gateway}
            selected={selectedGateway === gateway.value}
            onClick={() => onSelectGateway(gateway.value)}
            amountInUSD={gateway.value === 'nowpayments' ? getAmountInUSD() : null}
          />
        ))}
      </Stack>
    </Stack>
  );
}

PaymentMethods.propTypes = {
  onSelectGateway: PropTypes.func.isRequired,
  selectedGateway: PropTypes.string,
  exchangeRate: PropTypes.object,
  amount: PropTypes.number,
};

// ----------------------------------------------------------------------

function GatewayOption({ gateway, selected, onClick, amountInUSD }) {
  const { value, label, description, logo, icon } = gateway;
  const [logoError, setLogoError] = useState(false);

  return (
    <Paper
      variant="outlined"
      onClick={onClick}
      sx={{
        p: 2.5,
        cursor: 'pointer',
        ...(selected && {
          boxShadow: (theme) => `0 0 0 2px ${theme.vars.palette.text.primary}`,
        }),
      }}
    >
      <ListItemText
        primary={
          <Stack direction="row" alignItems="center">
            <Iconify
              icon={selected ? 'eva:checkmark-circle-2-fill' : 'eva:radio-button-off-fill'}
              width={24}
              sx={{
                mr: 2,
                color: selected ? 'primary.main' : 'text.secondary',
              }}
            />

            <Box component="span" sx={{ flexGrow: 1 }}>
              {label}
            </Box>

            {/* Logo or Icon */}
            <Box
              sx={{
                width: 80,
                height: 32,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              {!logoError ? (
                <Box
                  component="img"
                  src={logo}
                  alt={label}
                  onError={() => setLogoError(true)}
                  sx={{
                    maxWidth: '100%',
                    maxHeight: '100%',
                    objectFit: 'contain',
                  }}
                />
              ) : (
                <Iconify
                  icon={icon}
                  width={32}
                  sx={{ color: 'text.secondary' }}
                />
              )}
            </Box>
          </Stack>
        }
        secondary={
          <Stack spacing={0.5} sx={{ mt: 1 }}>
            <Typography variant="body2" color="text.secondary" component="div">
              {description}
            </Typography>
            {value === 'nowpayments' && amountInUSD && (
              <Typography variant="caption" color="text.secondary" component="div">
                معادل: ${amountInUSD} USD
              </Typography>
            )}
          </Stack>
        }
        primaryTypographyProps={{ typography: 'subtitle2', component: 'div' }}
        secondaryTypographyProps={{ component: 'div' }}
      />
    </Paper>
  );
}

GatewayOption.propTypes = {
  gateway: PropTypes.object.isRequired,
  selected: PropTypes.bool,
  onClick: PropTypes.func,
  amountInUSD: PropTypes.string,
};
