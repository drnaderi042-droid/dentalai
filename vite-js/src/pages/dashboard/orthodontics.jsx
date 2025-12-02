import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { OrthodonticsView } from 'src/sections/orthodontics/view';

// ----------------------------------------------------------------------

const metadata = { title: `ارتودنسی | ${CONFIG.site.name}` };

export default function Page() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <OrthodonticsView />
    </>
  );
}
