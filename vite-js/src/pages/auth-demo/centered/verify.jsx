import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { CenteredVerifyView } from 'src/sections/auth-demo/centered';

// ----------------------------------------------------------------------

const metadata = { title: `تأیید | طرح مرکزی - ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <CenteredVerifyView />
    </>
  );
}
