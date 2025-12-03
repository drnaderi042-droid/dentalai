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

  // For uploads, use HTTPS domain - proxied through nginx
  if (avatarUrl.startsWith('/uploads/') || avatarUrl.includes('/uploads/')) {
    return `https://ceph2.bioritalin.ir${avatarUrl.startsWith('/') ? avatarUrl : `/${avatarUrl}`}`;
  }

  // For other relative paths, use backend server URL
  if (avatarUrl.startsWith('/')) {
    const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
    return `${serverUrl}${avatarUrl}`;
  }

  // If it's a path without leading slash
  const serverUrl = CONFIG.site.serverUrl || 'http://localhost:7272';
  return `${serverUrl}/${avatarUrl}`;
}
