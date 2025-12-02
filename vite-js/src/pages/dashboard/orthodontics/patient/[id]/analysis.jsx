import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { CephalometricAnalysisView } from 'src/sections/orthodontics/patient/view/cephalometric-analysis-view';

// ----------------------------------------------------------------------

export default function Page() {
  return (
    <>
      <Helmet>
        <title> {`آنالیز سفالومتری - ${CONFIG.appName}`}</title>
      </Helmet>

      <CephalometricAnalysisView />
    </>
  );
}














