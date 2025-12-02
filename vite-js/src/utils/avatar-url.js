import { CONFIG } from 'src/config-global';

/**
 * Build full URL for avatar image
 * @param {string|null|undefined} avatarUrl - Avatar URL (can be relative or absolute)
 * @returns {string|null} Full URL or null if avatarUrl is empty
 */
export function getAvatarUrl(avatarUrl) {
  if (!avatarUrl) return null;
  
  // If it's already a full URL, return as is
  if (avatarUrl.startsWith('http://') || avatarUrl.startsWith('https://')) {
    return avatarUrl;
  }
  
  // If it's a relative path, prepend server URL
  if (avatarUrl.startsWith('/')) {
    const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
    return `${serverUrl}${avatarUrl}`;
  }
  
  // If it's a path without leading slash
  const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
  return `${serverUrl}/${avatarUrl}`;
}


