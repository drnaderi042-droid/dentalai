import React from 'react';

import { Box, Card, Stack, Typography, IconButton } from '@mui/material';

import { getImageUrl } from 'src/utils/url-helpers';

import { Iconify } from 'src/components/iconify';

/**
 * Optimized Image List Item Component
 * Memoized to prevent unnecessary re-renders
 */
const ImageListItem = React.memo(({ item, onEdit, onDelete, getCategoryLabel }) => {
  // Safety check: ensure item exists
  if (!item) {
    return null;
  }
  
  // Truncate file name if longer than 20 characters
  const fileName = (item.originalName || item.name || `تصویر-${item.id}`).length > 20 
    ? `${(item.originalName || item.name || `تصویر-${item.id}`).substring(0, 20)}...` 
    : (item.originalName || item.name || `تصویر-${item.id}`);
  
  const imageUrl = getImageUrl(item.path || item.url || '');
  
  // Get category label in Persian
  const categoryLabel = getCategoryLabel ? getCategoryLabel(item.category) : (item.category || 'نامشخص');

  return (
    <Card
      sx={{
        p: 1.5,
        border: 1,
        borderColor: 'divider',
        bgcolor: 'background.paper',
        marginTop: '0 !important',
      }}
    >
      <Stack direction="row" spacing={1} alignItems="center">
        <Box
          component="img"
          src={imageUrl}
          alt={fileName}
          sx={{
            width: 36,
            height: 36,
            objectFit: 'cover',
            borderRadius: 1,
          }}
        />
        <Box sx={{ flex: 1, minWidth: 0 }}>
          <Typography variant="body2" noWrap>
            {fileName}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {categoryLabel}
          </Typography>
        </Box>
        <Stack direction="row" spacing={0.5} alignItems="center">
          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              e.preventDefault();
              if (onEdit && typeof onEdit === 'function') {
                onEdit(item);
              }
            }}
            sx={{
              width: 26,
              height: 26,
              p: 0,
            }}
          >
            <Iconify icon="solar:pen-bold" width={18} />
          </IconButton>
          <IconButton
            onClick={(e) => {
              e.stopPropagation();
              onDelete(item);
            }}
            sx={{
              width: 26,
              height: 26,
              p: 0,
            }}
          >
            <Iconify icon="mingcute:close-line" width={16} />
          </IconButton>
        </Stack>
      </Stack>
    </Card>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function for React.memo
  // Safety check: if item is undefined, always re-render
  if (!prevProps.item || !nextProps.item) {
    return false;
  }
  // Check if handlers changed - if so, always re-render
  if (prevProps.onEdit !== nextProps.onEdit || prevProps.onDelete !== nextProps.onDelete) {
    return false; // Re-render if handlers changed
  }
  // Otherwise, only re-render if item data changed
  return (
    prevProps.item.id === nextProps.item.id &&
    (prevProps.item.originalName || prevProps.item.name) === (nextProps.item.originalName || nextProps.item.name) &&
    prevProps.item.category === nextProps.item.category &&
    prevProps.item.path === nextProps.item.path
  );
});

ImageListItem.displayName = 'ImageListItem';

export default ImageListItem;

