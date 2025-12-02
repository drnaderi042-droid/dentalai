import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { ProsthodonticsView } from 'src/sections/prosthodontics/view/prosthodontics-view';

// ----------------------------------------------------------------------

const metadata = { title: `پروستودونتیکس | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <ProsthodonticsView />
    </>
  );
}
