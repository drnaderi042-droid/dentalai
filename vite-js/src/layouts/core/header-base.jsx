import Box from '@mui/material/Box';
import NoSsr from '@mui/material/NoSsr';
import Button from '@mui/material/Button';
import { styled, useTheme } from '@mui/material/styles';

import { paths } from 'src/routes/paths';
import { usePathname } from 'src/routes/hooks';

import { useHeaderContent } from 'src/contexts/header-content-context';

import { Logo } from 'src/components/logo';

import { HeaderSection } from './header-section';
import { Searchbar } from '../components/searchbar';
import { MenuPopover } from '../components/menu-popover';
import { WalletButton } from '../components/wallet-button';
import { SignInButton } from '../components/sign-in-button';
import { AccountPopover } from '../components/account-popover';
import { LanguagePopover } from '../components/language-popover';
import { ContactsPopover } from '../components/contacts-popover';
import { WorkspacesPopover } from '../components/workspaces-popover';
import { ThemeToggleButton } from '../components/theme-toggle-button';
import { NotificationsPopover } from '../components/notifications-popover';

// ----------------------------------------------------------------------

const StyledDivider = styled('span')(({ theme }) => ({
  width: 1,
  height: 10,
  flexShrink: 0,
  display: 'none',
  position: 'relative',
  alignItems: 'center',
  flexDirection: 'column',
  marginLeft: theme.spacing(2.5),
  marginRight: theme.spacing(2.5),
  backgroundColor: 'currentColor',
  color: theme.vars.palette.divider,
  '&::before, &::after': {
    top: -5,
    width: 3,
    height: 3,
    content: '""',
    flexShrink: 0,
    borderRadius: '50%',
    position: 'absolute',
    backgroundColor: 'currentColor',
  },
  '&::after': { bottom: -5, top: 'auto' },
}));

// ----------------------------------------------------------------------

export function HeaderBase({
  sx,
  data,
  slots,
  slotProps,
  onOpenNav,
  layoutQuery,

  slotsDisplay: {
    signIn = true,
    account = true,
    helpLink = true,
    settings = false,
    purchase = true,
    contacts = true,
    searchbar = true,
    workspaces = true,
    menuButton = true,
    localization = true,
    notifications = true,
    wallet = true,
  } = {},

  ...other
}) {
  const theme = useTheme();
  const pathname = usePathname();
  const { headerContent, hideRightButtons } = useHeaderContent();
  
  const isDashboard = pathname?.includes('/dashboard') || pathname === paths.dashboard.root;

  return (
    <HeaderSection
      sx={sx}
      layoutQuery={layoutQuery}
      slots={{
        ...slots,
        leftAreaStart: slots?.leftAreaStart,
        leftArea: (
          <>
            {slots?.leftAreaStart}

            {/* -- Custom Header Content -- */}
            {headerContent}

            {/* -- Menu button -- */}
            {menuButton && !headerContent && isDashboard && (
              <MenuPopover
                data-slot="menu-button"
                sx={{
                  mr: 1,
                  ml: -1,
                  [theme.breakpoints.up(layoutQuery)]: { display: 'none' },
                }}
              />
            )}

            {/* -- Logo -- */}
            {!headerContent && <Logo data-slot="logo" />}

            {/* -- Divider -- */}
            {!headerContent && <StyledDivider data-slot="divider" />}

            {/* -- Workspace popover -- */}
            {workspaces && <WorkspacesPopover data-slot="workspaces" data={data?.workspaces} />}

            {slots?.leftAreaEnd}
          </>
        ),
        rightArea: (
          <>
            {slots?.rightAreaStart}

            {!hideRightButtons && (
              <NoSsr>
                <Box
                  data-area="right"
                  sx={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: { xs: 1, sm: 1.5 },
                  }}
                >


                  {/* -- Searchbar -- */}
                  {searchbar && <Searchbar data-slot="searchbar" data={data?.nav} />}

                  {/* -- Wallet button -- */}
                  {wallet && <WalletButton data-slot="wallet" />}

                  {/* -- Language popover -- */}
                  {localization && <LanguagePopover data-slot="localization" data={data?.langs} />}

                  {/* -- Theme toggle button -- */}
                  <ThemeToggleButton data-slot="theme-toggle" />

                  {/* -- Notifications popover -- */}
                  {notifications && (
                    <NotificationsPopover data-slot="notifications" />
                  )}

                  {/* -- Contacts popover -- */}
                  {contacts && <ContactsPopover data-slot="contacts" data={data?.contacts} />}


                  {/* -- Account popover -- */}
                  {account && <AccountPopover data-slot="account" data={data?.account} />}

                  {/* -- Sign in button -- */}
                  {signIn && <SignInButton />}

                  {/* -- Purchase button -- */}
                  {purchase && (
                  <Button
                    data-slot="purchase"
                    variant="contained"
                    rel="noopener"
                    target="_blank"
                    href={paths.minimalStore}
                    sx={{
                      display: 'none',
                      [theme.breakpoints.up(layoutQuery)]: {
                        display: 'inline-flex',
                      },
                    }}
                  >
                    خرید
                  </Button>
                  )}
                </Box>
              </NoSsr>
            )}

            {slots?.rightAreaEnd}
          </>
        ),
      }}
      slotProps={slotProps}
      {...other}
    />
  );
}
