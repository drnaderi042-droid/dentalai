import { Helmet } from 'react-helmet-async';

import { CONFIG } from 'src/config-global';

import { FacialLandmarkView } from 'src/sections/facial-landmark/view/facial-landmark-view';

// ----------------------------------------------------------------------

const metadata = { title: `تشخیص لندمارک‌های صورت | ${CONFIG.site.name}` };

export default function FacialLandmarkPage() {
  return (
    <>
      <Helmet>
        <title>{metadata.title}</title>
      </Helmet>

      <FacialLandmarkView />
    </>
  );
}

















