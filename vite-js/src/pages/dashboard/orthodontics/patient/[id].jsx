import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { PatientOrthodonticsView } from 'src/sections/orthodontics/patient/view/patient-orthodontics-view.jsx';

// ----------------------------------------------------------------------

export default function Page() {
  return (
    <>
      <Helmet>
        <title> {`مدیریت بیمار - ${CONFIG.appName}`}</title>
      </Helmet>

      <PatientOrthodonticsView />
    </>
  );
}
