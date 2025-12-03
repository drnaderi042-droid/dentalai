import path from 'path';
import checker from 'vite-plugin-checker';
import { loadEnv, defineConfig } from 'vite';
import react from '@vitejs/plugin-react-swc';

// ----------------------------------------------------------------------

const PORT = 3030;

const env = loadEnv('all', process.cwd());

export default defineConfig({
  // base: env.VITE_BASE_PATH,
  plugins: [
    react(),
    checker({
      eslint: {
        lintCommand: 'eslint "./src/**/*.{js,jsx,ts,tsx}"',
      },
      overlay: {
        position: 'tl',
        initialIsOpen: false,
      },
    }),
  ],
  resolve: {
    alias: [
      {
        find: /^~(.+)/,
        replacement: path.join(process.cwd(), 'node_modules/$1'),
      },
      {
        find: /^src(.+)/,
        replacement: path.join(process.cwd(), 'src/$1'),
      },
    ],
  },
  server: {
    port: PORT,
    host: true, // '0.0.0.0' equivalent - allows access from network
    hmr: {
      // تنظیمات HMR برای جلوگیری از خطاهای اتصال
      // برای دسترسی از شبکه خارجی، از IP سرور استفاده کنیم
      host: '31.56.233.34',
      port: PORT,
    },
    watch: {
      // جلوگیری از خطاهای زیاد در watch mode
      usePolling: false,
      interval: 100,
    },
    proxy: {
      // Forward API calls to the backend server (mock server)
      '/api': {
        target: env.VITE_SERVER_URL || 'http://localhost:7272',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  preview: {
    port: PORT,
    host: true,
    proxy: {
      '/api': {
        target: env.VITE_SERVER_URL || 'http://localhost:7272',
        changeOrigin: true,
        secure: false,
      },
    },
  },
  optimizeDeps: {
    // Exclude optional LangChain packages (only needed if useEmbeddings=true)
    exclude: [
      '@langchain/openai',
      '@langchain/community',
      '@langchain/core',
    ],
  },
});
