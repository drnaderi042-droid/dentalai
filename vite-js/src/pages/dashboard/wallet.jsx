import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { WalletView } from 'src/sections/wallet/view';

// ----------------------------------------------------------------------

export default function Page() {
  return (
    <>
      <Helmet>
        <title> {`کیف پول - ${CONFIG.appName}`}</title>
      </Helmet>

      <WalletView />
    </>
  );
}
