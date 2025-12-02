import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { IntraOralView } from 'src/sections/intra-oral/view/intra-oral-view';

// ----------------------------------------------------------------------

const metadata = { title: `آنالیز عکس‌های داخل دهانی | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <IntraOralView />
    </>
  );
}

















