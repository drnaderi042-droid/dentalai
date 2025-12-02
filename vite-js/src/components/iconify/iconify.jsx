import { useEffect, forwardRef } from 'react';
import { Icon, disableCache } from '@iconify/react';

import Box from '@mui/material/Box';
import NoSsr from '@mui/material/NoSsr';

import { iconifyClasses } from './classes';

// ----------------------------------------------------------------------

// Utility function to preload icons
export const preloadIcons = (iconNames) => {
  if (typeof window === 'undefined' || !iconNames || iconNames.length === 0) return;
  
  // Use Set to avoid duplicate preloads
  const uniqueIcons = [...new Set(iconNames.filter(Boolean))];
  
  uniqueIcons.forEach((iconName) => {
    try {
      // Create a temporary icon element to trigger Iconify's loading mechanism
      // This will cache the icon data in Iconify's internal cache
      const tempIcon = document.createElement('iconify-icon');
      tempIcon.setAttribute('icon', iconName);
      tempIcon.style.cssText = 'position: absolute; visibility: hidden; width: 0; height: 0;';
      document.body.appendChild(tempIcon);
      
      // Remove after icon is loaded (Iconify loads asynchronously)
      // Use a longer timeout to ensure icon is cached
      setTimeout(() => {
        if (tempIcon.parentNode) {
          tempIcon.parentNode.removeChild(tempIcon);
        }
      }, 500);
    } catch (e) {
      // Silently fail if icon can't be preloaded
      console.debug('Failed to preload icon:', iconName, e);
    }
  });
};

// Hook to preload icons on mount
export const usePreloadIcons = (iconNames) => {
  useEffect(() => {
    if (iconNames && iconNames.length > 0) {
      preloadIcons(iconNames);
    }
  }, [iconNames]);
};

// ----------------------------------------------------------------------

export const Iconify = forwardRef(({ className, width = 20, sx, ...other }, ref) => {
  const baseStyles = {
    width,
    height: width,
    minWidth: width, // Prevent layout shift
    minHeight: width, // Prevent layout shift
    flexShrink: 0,
    display: 'inline-flex',
    color: 'inherit', // Use default color instead of black
    // Ensure icon container maintains size during loading
    position: 'relative',
    '& > *': {
      width: '100%',
      height: '100%',
    },
  };

  const renderFallback = (
    <Box
      component="span"
      className={iconifyClasses.root.concat(className ? ` ${className}` : '')}
      sx={{ ...baseStyles, ...sx }}
      aria-hidden="true"
    />
  );

  return (
    <NoSsr fallback={renderFallback}>
      <Box
        ref={ref}
        component={Icon}
        className={iconifyClasses.root.concat(className ? ` ${className}` : '')}
        sx={{ ...baseStyles, ...sx }}
        {...other}
      />
    </NoSsr>
  );
});

// https://iconify.design/docs/iconify-icon/disable-cache.html
disableCache('local');
