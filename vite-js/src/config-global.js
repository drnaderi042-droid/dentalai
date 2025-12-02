import { paths } from 'src/routes/paths';

import packageJson from '../package.json';

// ----------------------------------------------------------------------

// Get server URL dynamically based on current hostname
const getServerUrl = () => {
  // If VITE_SERVER_URL is explicitly set and doesn't contain localhost, use it
  if (import.meta.env.VITE_SERVER_URL && !import.meta.env.VITE_SERVER_URL.includes('localhost') && !import.meta.env.VITE_SERVER_URL.includes('127.0.0.1')) {
    return import.meta.env.VITE_SERVER_URL;
  }

  // In browser, use the same hostname and port as the frontend
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    return `${protocol}//${hostname}:7272`;
  }

  // Fallback for server-side rendering
  return 'http://localhost:7272';
};

// Get AI server URL dynamically based on current hostname (for mobile access)
const getAiServerUrl = (defaultPort = 5001) => {
  // If VITE_AI_SERVER_URL is explicitly set, use it
  if (import.meta.env.VITE_AI_SERVER_URL) {
    const url = import.meta.env.VITE_AI_SERVER_URL;
    // If it contains localhost, replace with current hostname for mobile access
    if (typeof window !== 'undefined' && (url.includes('localhost') || url.includes('127.0.0.1'))) {
      const {protocol} = window.location;
      const {hostname} = window.location;
      // Extract port from URL or use default
      try {
        const urlObj = new URL(url);
        const port = urlObj.port || defaultPort;
        return `${protocol}//${hostname}:${port}`;
      } catch (e) {
        // If URL parsing fails, just replace localhost/127.0.0.1 with current hostname
        const port = url.match(/:(\d+)/)?.[1] || defaultPort;
        return `${protocol}//${hostname}:${port}`;
      }
    }
    return url;
  }
  
  // In browser, use the same hostname as the frontend with the specified port
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    return `${protocol}//${hostname}:${defaultPort}`;
  }
  
  // Fallback for server-side rendering
  return `http://localhost:${defaultPort}`;
};

// Get HRNet server URL dynamically (port 5000)
const getHrnetServerUrl = () => getAiServerUrl(5000);

export const CONFIG = {
  site: {
    name: 'Minimals',
    serverUrl: getServerUrl(),
    assetURL: import.meta.env.VITE_ASSET_URL ?? '',
    basePath: import.meta.env.VITE_BASE_PATH ?? '',
    version: packageJson.version,
  },
  /**
   * AI Server URL for facial landmark, intra-oral, and cephalometric analysis
   * Dynamically resolved based on current hostname (works for mobile access)
   */
  aiServerUrl: getAiServerUrl(5001),
  /**
   * HRNet Server URL for cephalometric analysis
   * Dynamically resolved based on current hostname (works for mobile access)
   */
  hrnetServerUrl: getHrnetServerUrl(),
  /**
   * Auth
   * @method jwt | amplify | firebase | supabase | auth0
   */
  auth: {
    method: 'jwt',
    skip: false,
    redirectPath: paths.dashboard.root,
  },
  /**
   * Mapbox
   */
  mapbox: {
    apiKey: import.meta.env.VITE_MAPBOX_API_KEY ?? '',
  },
  /**
   * Firebase
   */
  firebase: {
    apiKey: import.meta.env.VITE_FIREBASE_API_KEY ?? '',
    authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN ?? '',
    projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID ?? '',
    storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET ?? '',
    messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID ?? '',
    appId: import.meta.env.VITE_FIREBASE_APPID ?? '',
    measurementId: import.meta.env.VITE_FIREBASE_MEASUREMENT_ID ?? '',
  },
  /**
   * Amplify
   */
  amplify: {
    userPoolId: import.meta.env.VITE_AWS_AMPLIFY_USER_POOL_ID ?? '',
    userPoolWebClientId: import.meta.env.VITE_AWS_AMPLIFY_USER_POOL_WEB_CLIENT_ID ?? '',
    region: import.meta.env.VITE_AWS_AMPLIFY_REGION ?? '',
  },
  /**
   * Auth0
   */
  auth0: {
    clientId: import.meta.env.VITE_AUTH0_CLIENT_ID ?? '',
    domain: import.meta.env.VITE_AUTH0_DOMAIN ?? '',
    callbackUrl: import.meta.env.VITE_AUTH0_CALLBACK_URL ?? '',
  },
  /**
   * Supabase
   */
  supabase: {
    url: import.meta.env.VITE_SUPABASE_URL ?? '',
    key: import.meta.env.VITE_SUPABASE_ANON_KEY ?? '',
  },
};
