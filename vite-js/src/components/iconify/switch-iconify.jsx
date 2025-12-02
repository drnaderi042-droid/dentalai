import { useMemo } from 'react';

import { Iconify, usePreloadIcons } from './iconify';

// ----------------------------------------------------------------------

/**
 * SwitchIconify - A wrapper component for icons that switch between states
 * Preloads all possible icons to prevent layout shift
 * 
 * @param {string} icon - Current icon name
 * @param {string[]} possibleIcons - Array of all possible icon names that might be used
 * @param {number} width - Icon width (default: 20)
 * @param {object} sx - Additional styles
 * @param {object} other - Other props to pass to Iconify
 */
export function SwitchIconify({ icon, possibleIcons, width = 20, sx, ...other }) {
  // Preload all possible icons on mount
  usePreloadIcons(possibleIcons || [icon]);

  // Memoize to prevent unnecessary re-renders
  const iconToRender = useMemo(() => icon, [icon]);

  return (
    <Iconify 
      icon={iconToRender} 
      width={width} 
      sx={sx} 
      {...other} 
    />
  );
}





