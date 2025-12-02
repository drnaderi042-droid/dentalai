import { useDropzone } from 'react-dropzone';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import FormHelperText from '@mui/material/FormHelperText';

import { varAlpha } from 'src/theme/styles';

import { Iconify } from '../iconify';
import { UploadPlaceholder } from './components/placeholder';
import { RejectionFiles } from './components/rejection-files';
import { MultiFilePreview } from './components/preview-multi-file';
import { DeleteButton, SingleFilePreview } from './components/preview-single-file';

// ----------------------------------------------------------------------

export function Upload({
  sx,
  value,
  error,
  disabled,
  onDelete,
  onUpload,
  onRemove,
  thumbnail,
  helperText,
  onRemoveAll,
  multiple = false,
  hideFileList = false,
  hideUploadButton = false, // مخفی کردن دکمه آپلود و نمایش فقط thumbnail ها
  ...other
}) {
  const { getRootProps, getInputProps, isDragActive, isDragReject, fileRejections } = useDropzone({
    multiple,
    disabled,
    ...other,
  });

  const isArray = Array.isArray(value) && multiple;

  const hasFile = !isArray && !!value;

  const hasFiles = isArray && !!value.length;

  const hasError = isDragReject || !!error;

  const renderMultiPreview = hasFiles && !hideFileList && (
    <>
      <Box
        sx={{
          // حفظ فضای مورد نیاز برای جلوگیری از layout shift
          minHeight: thumbnail ? 80 : 'auto',
          position: 'relative',
        }}
      >
        <MultiFilePreview files={value} thumbnail={thumbnail} onRemove={onRemove} sx={{ my: 3 }} />
      </Box>

      {onUpload && (
        <Stack direction="row" justifyContent="flex-end" spacing={1.5} sx={{ mt: 3 }}>
          <Button
            size="small"
            variant="contained"
            onClick={onUpload}
            startIcon={<Iconify icon="eva:cloud-upload-fill" />}
          >
            آپلود
          </Button>
        </Stack>
      )}
    </>
  );

  return (
    <Box sx={{ width: 1, position: 'relative', ...sx }}>
      {/* دکمه آپلود - فقط اگر hideUploadButton false باشد نمایش داده می‌شود */}
      {!hideUploadButton ? (
        <Box
          {...getRootProps()}
          sx={{
            p: 5,
            outline: 'none',
            borderRadius: '16px',
            cursor: 'pointer',
            overflow: 'hidden',
            position: 'relative',
            bgcolor: (theme) => varAlpha(theme.vars.palette.grey['500Channel'], 0.08),
            border: (theme) => `1px dashed ${varAlpha(theme.vars.palette.grey['500Channel'], 0.2)}`,
            transition: (theme) => theme.transitions.create(['opacity', 'padding']),
            '&:hover': { opacity: 0.72 },
            ...(isDragActive && { opacity: 0.72 }),
            ...(disabled && { opacity: 0.48, pointerEvents: 'none' }),
            ...(hasError && {
              color: 'error.main',
              borderColor: 'error.main',
              bgcolor: (theme) => varAlpha(theme.vars.palette.error.mainChannel, 0.08),
            }),
            ...(hasFile && { padding: '28% 0' }),
          }}
        >
          <input {...getInputProps()} />

          {/* Single file */}
          {hasFile ? <SingleFilePreview file={value} /> : <UploadPlaceholder />}
        </Box>
      ) : (
        // اگر hideUploadButton true باشد، input را در یک Box نامرئی قرار می‌دهیم تا drag and drop کار کند
        <Box {...getRootProps()} sx={{ display: 'none' }}>
          <input {...getInputProps()} />
        </Box>
      )}

      {/* Single file */}
      {hasFile && !hideUploadButton && <DeleteButton onClick={onDelete} />}

      {helperText && (
        <FormHelperText error={!!error} sx={{ px: 2 }}>
          {helperText}
        </FormHelperText>
      )}

      <RejectionFiles files={fileRejections} />

      {/* Multi files */}
      {renderMultiPreview}
    </Box>
  );
}
