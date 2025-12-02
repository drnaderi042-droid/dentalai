import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { PeriodonticsView } from 'src/sections/periodontics/view/periodontics-view';

// ----------------------------------------------------------------------

const metadata = { title: `پریودونتیکس | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <PeriodonticsView />
    </>
  );
}
