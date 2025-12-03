/**
 * @type {import('next').NextConfig}
 */

const nextConfig = {
  env: {
    DEV_API: `http://localhost:7272`,
    PRODUCTION_API: 'https://api-dev-minimal-v6.vercel.app',
  },

  eslint: {
    // Warning: This allows production builds to successfully complete even if
    // your project has ESLint errors.
    ignoreDuringBuilds: true,
  },

  typescript: {
    // !! WARN !!
    // Dangerously allow production builds to successfully complete even if
    // your project has type errors.
    // !! WARN !!
    ignoreBuildErrors: true,
  },

  // MEMORY OPTIMIZATION: Configure Node.js memory settings
  // These settings help reduce memory usage
  serverRuntimeConfig: {
    // Limit request body size
    maxRequestBodySize: 10 * 1024 * 1024, // 10MB
  },

  // Optimize webpack for memory
  webpack: (config, { isServer }) => {
    if (isServer) {
      // Reduce memory usage during builds
      config.optimization = {
        ...config.optimization,
        minimize: false, // Don't minify in development to save memory
      };
    }
    return config;
  },

  // Disable default Next.js pages - serve static index.html for all non-API routes
  async rewrites() {
    return [
      {
        source: '/((?!api/).*)',
        destination: '/index.html',
      },
    ];
  },

  // Disable Next.js pages and use only static files
  trailingSlash: true,
  exportPathMap: async function() {
    return {
      '/': { page: '/' }
    };
  },
};

export default nextConfig;
