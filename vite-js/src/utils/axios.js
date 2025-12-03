import axios from 'axios';

import { CONFIG } from 'src/config-global';

// ----------------------------------------------------------------------

// Get server URL dynamically at runtime (for mobile/network access)
const getServerUrl = () => {
  // If VITE_SERVER_URL is explicitly set, use it
  if (CONFIG.site.serverUrl && CONFIG.site.serverUrl !== '') {
    // Check if it's a relative URL or contains localhost
    if (CONFIG.site.serverUrl.includes('localhost') || CONFIG.site.serverUrl.includes('127.0.0.1')) {
      // In browser, use the current domain (nginx handles proxying)
      if (typeof window !== 'undefined') {
        const {protocol} = window.location;
        const {hostname} = window.location;
        return `${protocol}//${hostname}`;
      }
    }
    return CONFIG.site.serverUrl;
  }

  // In browser, use the same hostname as the frontend (nginx handles proxying)
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    // For HTTPS, don't specify port (nginx handles SSL termination)
    if (protocol === 'https:') {
      return `${protocol}//${hostname}`;
    }
    // For HTTP (development), use port 7272
    return `${protocol}//${hostname}:7272`;
  }

  // Fallback
  return 'http://localhost:7272';
};

const axiosInstance = axios.create({ baseURL: getServerUrl() });

axiosInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    // Handle network errors (CORS, connection refused, etc.)
    if (!error.response) {
      let errorMessage = 'خطا در اتصال به سرور';
      
      if (error.code === 'ECONNABORTED') {
        errorMessage = 'درخواست به موقعیت پاسخ نداد';
      } else if (error.message === 'Network Error') {
        errorMessage = 'خطا در شبکه. لطفاً اتصال اینترنت خود را بررسی کنید';
      } else if (error.message && error.message.includes('CORS')) {
        errorMessage = 'خطا در تنظیمات CORS سرور';
      }
      
      const networkError = new Error(errorMessage);
      networkError.isNetworkError = true;
      networkError.originalError = error;
      return Promise.reject(networkError);
    }
    
    // Handle HTTP errors with response
    const errorData = error.response.data;
    const errorMessage = errorData?.message || errorData?.error || error.message || 'خطایی رخ داد';
    
    const httpError = new Error(errorMessage);
    httpError.status = error.response.status;
    httpError.data = errorData;
    httpError.originalError = error;
    
    return Promise.reject(httpError);
  }
);

export default axiosInstance;

// ----------------------------------------------------------------------

export const fetcher = async (args) => {
  try {
    const [url, config] = Array.isArray(args) ? args : [args];

    const res = await axiosInstance.get(url, { ...config });

    return res.data;
  } catch (error) {
    console.error('Failed to fetch:', error);
    throw error;
  }
};

// ----------------------------------------------------------------------

export const endpoints = {
  kanban: '/api/kanban',
  calendar: '/api/calendar',
  patients: '/api/patients',
  aiDiagnosis: '/api/ai/dental-diagnosis',
  auth: {
    me: '/api/auth/me',
    signIn: '/api/auth/sign-in',
    signUp: '/api/auth/sign-up',
  },
  user: {
    list: '/api/users',
    details: (id) => `/api/users/${id}`,
    update: (id) => `/api/users/${id}`,
    delete: (id) => `/api/users/${id}`,
  },
  chat: {
    base: '/api/chat',
    block: '/api/chat/block',
    unblock: '/api/chat/unblock',
    report: '/api/chat/report',
    blocked: '/api/chat/blocked',
  },
  mail: {
    list: '/api/mail/list',
    details: '/api/mail/details',
    labels: '/api/mail/labels',
  },
  post: {
    list: '/api/post/list',
    details: '/api/post/details',
    latest: '/api/post/latest',
    search: '/api/post/search',
  },
  product: {
    list: '/api/product/list',
    details: '/api/product/details',
    search: '/api/product/search',
  },
  invoice: {
    create: '/api/invoice/create',
    list: '/api/invoice/list',
    details: (id) => `/api/invoice/${id}`,
  },
  exchangeRate: '/api/exchange-rate',
  notifications: {
    list: '/api/notifications',
    markAllAsRead: '/api/notifications/mark-all-read',
    markAsRead: (id) => `/api/notifications/${id}/mark-read`,
  },
};
