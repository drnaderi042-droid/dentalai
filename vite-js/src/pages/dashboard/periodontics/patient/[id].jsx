import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { PatientPeriodonticsView } from 'src/sections/periodontics/patient/view';

// ----------------------------------------------------------------------

const metadata = { title: `جزئیات بیمار پریودونتیکس | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <PatientPeriodonticsView />
    </>
  );
}



