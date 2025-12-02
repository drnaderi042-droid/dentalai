import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { CephalometricAnalysisView } from 'src/sections/orthodontics/cephalometric-analysis';

// ----------------------------------------------------------------------

const metadata = { title: `تحلیل سفالومتری | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <CephalometricAnalysisView />
    </>
  );
}