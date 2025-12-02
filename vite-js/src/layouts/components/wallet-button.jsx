import { useState } from 'react';
import { m } from 'framer-motion';

import Divider from '@mui/material/Divider';
import Popover from '@mui/material/Popover';
import SvgIcon from '@mui/material/SvgIcon';
import MenuItem from '@mui/material/MenuItem';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';
import ListItemText from '@mui/material/ListItemText';

import { varHover } from 'src/components/animate';
import { usePopover } from 'src/components/custom-popover';

// ----------------------------------------------------------------------

export function WalletButton({ sx, ...other }) {
  const popover = usePopover();
  const [walletData, setWalletData] = useState({
    balance: 0,
    currency: 'تومان',
  });

  const formatCurrency = (amount) => new Intl.NumberFormat('fa-IR').format(Math.abs(amount));

  return (
    <>
      <IconButton
        component={m.button}
        whileTap="tap"
        whileHover="hover"
        variants={varHover(1.05)}
        onClick={popover.onOpen}
        sx={{
          ...(popover.open && { bgcolor: (theme) => theme.vars.palette.action.selected }),
          ...sx,
        }}
        {...other}
      >
        <SvgIcon viewBox="0 0 24 24">
          <g clipPath="url(#clip0_4418_169476)">
            <path
              opacity="0.4"
              d="M18.04 13.55C17.62 13.96 17.38 14.55 17.44 15.18C17.53 16.26 18.52 17.05 19.6 17.05H21.5V18.24C21.5 20.31 19.81 22 17.74 22H6.26C4.19 22 2.5 20.31 2.5 18.24V11.51C2.5 9.44001 4.19 7.75 6.26 7.75H17.74C19.81 7.75 21.5 9.44001 21.5 11.51V12.95H19.48C18.92 12.95 18.41 13.17 18.04 13.55Z"
              style={{ fill: 'var(--fillg)' }}
            />
            <path
              d="M14.85 3.94964V7.74962H6.26C4.19 7.74962 2.5 9.43963 2.5 11.5096V7.83965C2.5 6.64965 3.23 5.58961 4.34 5.16961L12.28 2.16961C13.52 1.70961 14.85 2.61963 14.85 3.94964Z"
              style={{ fill: 'var(--fillg)' }}
            />
            <path
              d="M22.5598 13.9702V16.0302C22.5598 16.5802 22.1198 17.0302 21.5598 17.0502H19.5998C18.5198 17.0502 17.5298 16.2602 17.4398 15.1802C17.3798 14.5502 17.6198 13.9602 18.0398 13.5502C18.4098 13.1702 18.9198 12.9502 19.4798 12.9502H21.5598C22.1198 12.9702 22.5598 13.4202 22.5598 13.9702Z"
              style={{ fill: 'var(--fillg)' }}
            />
            <path
              d="M14 12.75H7C6.59 12.75 6.25 12.41 6.25 12C6.25 11.59 6.59 11.25 7 11.25H14C14.41 11.25 14.75 11.59 14.75 12C14.75 12.41 14.41 12.75 14 12.75Z"
              style={{ fill: 'var(--fillg)' }}
            />
          </g>
          <defs>
            <clipPath id="clip0_4418_169476">
              <rect width="24" height="24" fill="white" />
            </clipPath>
          </defs>
        </SvgIcon>
      </IconButton>

      <Popover
        open={popover.open}
        anchorEl={popover.anchorEl}
        onClose={popover.onClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        transformOrigin={{ vertical: 'top', horizontal: 'right' }}
      >
        {/* Balance Display with Charge Button as li */}
        <MenuItem
          component="li"
          sx={{
            p: 2,
            flexDirection: 'row',
            alignItems: 'center',
            justifyContent: 'space-between',
            gap: 2,
            cursor: 'pointer',
            '&:hover': {
              bgcolor: 'action.hover',
            },
          }}
          onClick={() => {
            window.location.href = '/dashboard/wallet';
            popover.onClose();
          }}
        >
          <ListItemText
            primary="شارژ کیف پول"
            secondary={`موجودی: ${formatCurrency(walletData.balance)} ${walletData.currency}`}
            primaryTypographyProps={{
              typography: 'body2',
              fontWeight: 400,
              color: 'text.primary',
            }}
            secondaryTypographyProps={{
              typography: 'body2',
              color: walletData.balance > 0 ? 'success.main' : 'error.main',
              fontWeight: 400,
              sx: { mt: 0.5 },
            }}
          />
          <SvgIcon sx={{ color: 'primary.main' }}>
            <path d="M9 5v2h6.59L4 18.59 5.41 20 17 8.41V15h2V5z" />
          </SvgIcon>
        </MenuItem>

        <Divider />

        {/* View All Transactions Link */}
        <MenuItem
          sx={{ p: 1.5, justifyContent: 'center' }}
          onClick={() => {
            window.location.href = '/dashboard/wallet';
            popover.onClose();
          }}
        >
          <Typography variant="body2">
            مشاهده جزئیات و تراکنش‌ها
          </Typography>
        </MenuItem>
      </Popover>
    </>
  );
}
