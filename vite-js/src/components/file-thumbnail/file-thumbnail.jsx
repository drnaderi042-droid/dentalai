import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';

import { fileThumbnailClasses } from './classes';
import { fileData, fileThumb, fileFormat } from './utils';
import { RemoveButton, DownloadButton } from './action-buttons';

// ----------------------------------------------------------------------

export function FileThumbnail({
  sx,
  file,
  tooltip,
  onRemove,
  imageView,
  slotProps,
  onDownload,
  ...other
}) {
  // Get preview URL - handle both string URLs and File objects
  let previewUrl;
  if (typeof file === 'string') {
    previewUrl = file;
  } else if (file instanceof File || file instanceof Blob) {
    // Real File or Blob object - can use createObjectURL
    previewUrl = URL.createObjectURL(file);
  } else if (file?.preview) {
    // File-like object with preview property (URL string)
    previewUrl = file.preview;
  } else if (file?.path) {
    // File-like object with path property (URL string)
    previewUrl = file.path;
  } else {
    // Fallback: try to use file as URL string
    previewUrl = file || '';
  }

  const { name, path } = fileData(file);

  const format = fileFormat(path || previewUrl);

  const renderImg = (
    <Box
      component="img"
      src={previewUrl}
      className={fileThumbnailClasses.img}
      sx={{
        width: 1,
        height: 1,
        objectFit: 'cover',
        borderRadius: 'inherit',
        // جلوگیری از layout shift با حفظ aspect ratio
        aspectRatio: '1 / 1',
        backgroundColor: 'grey.100',
        ...slotProps?.img,
      }}
      onLoad={(e) => {
        // اطمینان از اینکه تصویر لود شده است
        e.currentTarget.style.backgroundColor = 'transparent';
      }}
      onError={(e) => {
        // در صورت خطا، placeholder نمایش داده شود
        e.currentTarget.style.display = 'none';
      }}
    />
  );

  const renderIcon = (
    <Box
      component="img"
      src={fileThumb(format)}
      className={fileThumbnailClasses.icon}
      sx={{ width: 1, height: 1, ...slotProps?.icon }}
    />
  );

  const renderContent = (
    <Stack
      component="span"
      className={fileThumbnailClasses.root}
      sx={{
        width: 36,
        height: 36,
        flexShrink: 0,
        borderRadius: 1.25,
        alignItems: 'center',
        position: 'relative',
        display: 'inline-flex',
        justifyContent: 'center',
        ...sx,
      }}
      {...other}
    >
      {format === 'image' && imageView ? renderImg : renderIcon}

      {onRemove && (
        <RemoveButton
          onClick={onRemove}
          className={`file-thumbnail-remove-btn ${fileThumbnailClasses.removeBtn}`}
          sx={{
            opacity: 1,
            zIndex: 10,
            ...slotProps?.removeBtn,
          }}
        />
      )}

      {onDownload && (
        <DownloadButton
          onClick={onDownload}
          className={fileThumbnailClasses.downloadBtn}
          sx={slotProps?.downloadBtn}
        />
      )}
    </Stack>
  );

  if (tooltip) {
    return (
      <Tooltip
        arrow
        title={name}
        slotProps={{ popper: { modifiers: [{ name: 'offset', options: { offset: [0, -12] } }] } }}
      >
        {renderContent}
      </Tooltip>
    );
  }

  return renderContent;
}
