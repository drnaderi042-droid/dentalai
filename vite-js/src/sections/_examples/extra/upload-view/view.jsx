import { useState, useCallback } from 'react';

import Stack from '@mui/material/Stack';
import Switch from '@mui/material/Switch';
import Typography from '@mui/material/Typography';
import FormControlLabel from '@mui/material/FormControlLabel';

import { paths } from 'src/routes/paths';

import { useBoolean } from 'src/hooks/use-boolean';

import { fData } from 'src/utils/format-number';

import { Iconify } from 'src/components/iconify';
import { CustomBreadcrumbs } from 'src/components/custom-breadcrumbs';
import { Upload, UploadBox, UploadAvatar } from 'src/components/upload';

import { ComponentHero } from '../../component-hero';
import { ScrollToViewTemplate } from '../../component-template';

// ----------------------------------------------------------------------

export function UploadView() {
  const preview = useBoolean();

  const [files, setFiles] = useState([]);

  const [file, setFile] = useState(null);

  const [avatarUrl, setAvatarUrl] = useState(null);

  const handleDropSingleFile = useCallback((acceptedFiles) => {
    const newFile = acceptedFiles[0];
    setFile(newFile);
  }, []);

  const handleDropAvatar = useCallback((acceptedFiles) => {
    const newFile = acceptedFiles[0];
    setAvatarUrl(newFile);
  }, []);

  const handleDropMultiFile = useCallback(
    (acceptedFiles) => {
      setFiles([...files, ...acceptedFiles]);
    },
    [files]
  );

  const handleRemoveFile = (inputFile) => {
    const filesFiltered = files.filter((fileFiltered) => fileFiltered !== inputFile);
    setFiles(filesFiltered);
  };

  const handleRemoveAllFiles = () => {
    setFiles([]);
  };

  const DEMO = [
    {
      name: 'آپلود چند فایل',
      component: (
        <>
          <FormControlLabel
            control={<Switch checked={preview.value} onClick={preview.onToggle} />}
            label="نمایش تصویر بندانگشتی"
            sx={{ mb: 3, width: 1, justifyContent: 'flex-end' }}
          />
          <Upload
            multiple
            thumbnail={preview.value}
            value={files}
            onDrop={handleDropMultiFile}
            onRemove={handleRemoveFile}
            onRemoveAll={handleRemoveAllFiles}
            onUpload={() => console.info('ON UPLOAD')}
          />
        </>
      ),
    },
    {
      name: 'آپلود تک فایل',
      component: (
        <Upload value={file} onDrop={handleDropSingleFile} onDelete={() => setFile(null)} />
      ),
    },
    {
      name: 'آپلود آواتار',
      component: (
        <UploadAvatar
          value={avatarUrl}
          onDrop={handleDropAvatar}
          validator={(fileData) => {
            if (fileData.size > 1000000) {
              return { code: 'file-too-large', message: `فایل بزرگ‌تر از ${fData(1000000)} است` };
            }
            return null;
          }}
          helperText={
            <Typography
              variant="caption"
              sx={{
                mt: 3,
                mx: 'auto',
                display: 'block',
                textAlign: 'center',
                color: 'text.disabled',
              }}
            >
              فرمت‌های مجاز: *.jpeg, *.jpg, *.png, *.gif
              <br /> حداکثر اندازه: {fData(3145728)}
            </Typography>
          }
        />
      ),
    },
    {
      name: 'جعبه آپلود',
      component: (
        <Stack direction="row" spacing={2}>
          <UploadBox />
          <UploadBox
            placeholder={
              <Stack spacing={0.5} alignItems="center">
                <Iconify icon="eva:cloud-upload-fill" width={40} />
                <Typography variant="body2">آپلود فایل</Typography>
              </Stack>
            }
            sx={{ mb: 3, py: 2.5, flexGrow: 1, height: 'auto' }}
          />
        </Stack>
      ),
    },
  ];

  return (
    <>
      <ComponentHero>
        <CustomBreadcrumbs
          heading="آپلود"
          links={[{ name: 'کامپوننت‌ها', href: paths.components }, { name: 'آپلود' }]}
          moreLink={['https://react-dropzone.js.org/#section-basic-example']}
        />
      </ComponentHero>

      <ScrollToViewTemplate data={DEMO} />
    </>
  );
}
