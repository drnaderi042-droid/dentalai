import { CONFIG } from 'src/config-global';

/**
 * Get the full URL for an image or file path
 * If path already starts with http/https, replaces localhost with current hostname
 * Otherwise, prepends the server URL
 * 
 * @param {string} path - The image/file path (can be relative or absolute URL)
 * @returns {string} - Full URL to the resource
 */
export const getImageUrl = (path) => {
  if (!path) return '';

  // If path is already a full URL
  if (path.startsWith('http://') || path.startsWith('https://')) {
    // Replace localhost/127.0.0.1 with current hostname for network access
    if (typeof window !== 'undefined') {
      const url = new URL(path);
      if (url.hostname === 'localhost' || url.hostname === '127.0.0.1') {
        const {protocol} = window.location;
        const {hostname} = window.location;
        return `${protocol}//${hostname}:${url.port || '7272'}${url.pathname}${url.search}${url.hash}`;
      }
    }
    return path;
  }

  // Otherwise, prepend server URL
  return `${CONFIG.site.serverUrl || 'http://localhost:7272'}${path}`;
};

/**
 * Get the full URL for an API endpoint
 * 
 * @param {string} endpoint - The API endpoint path (e.g., '/api/patients')
 * @returns {string} - Full URL to the API endpoint
 */
export const getApiUrl = (endpoint) => {
  if (!endpoint) return '';
  
  // Remove leading slash if present
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  
  return `${CONFIG.site.serverUrl || 'http://localhost:7272'}${cleanEndpoint}`;
};

/**
 * Get the full URL for an AI service endpoint
 * 
 * @param {string} endpoint - The AI service endpoint path
 * @param {number} defaultPort - Default port for AI service (default: 5001)
 * @returns {string} - Full URL to the AI service endpoint
 */
export const getAiServiceUrl = (endpoint, defaultPort = 5001) => {
  if (!endpoint) return '';
  
  // Remove leading slash if present
  const cleanEndpoint = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  
  // Use CONFIG.aiServerUrl if available (this is set to port 5001 in config-global.js)
  if (CONFIG.aiServerUrl) {
    return `${CONFIG.aiServerUrl}${cleanEndpoint}`;
  }
  
  // Fallback: construct from current hostname with defaultPort
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    return `${protocol}//${hostname}:${defaultPort}${cleanEndpoint}`;
  }
  
  // Fallback for server-side rendering
  return `http://localhost:${defaultPort}${cleanEndpoint}`;
};

/**
 * Get the current frontend URL dynamically
 * Useful for navigation links
 * 
 * @returns {string} - Current frontend URL
 */
export const getFrontendUrl = () => {
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    const port = window.location.port || '3030';
    return `${protocol}//${hostname}:${port}`;
  }
  return 'http://localhost:3030';
};

/**
 * Get service URL with custom port
 * 
 * @param {number} port - The port number
 * @returns {string} - Service URL with specified port
 */
export const getServiceUrl = (port) => {
  if (typeof window !== 'undefined') {
    const {protocol} = window.location;
    const {hostname} = window.location;
    return `${protocol}//${hostname}:${port}`;
  }
  
  // Fallback based on port
  const defaultHost = port === 7272 ? 'http://localhost:7272' : `http://localhost:${port}`;
  return defaultHost;
};

