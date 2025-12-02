import 'react-image-crop/dist/ReactCrop.css';

import PropTypes from 'prop-types';
import { useRef, useState } from 'react';
import ReactCrop from 'react-image-crop';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Slider from '@mui/material/Slider';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import DialogActions from '@mui/material/DialogActions';
import DialogContent from '@mui/material/DialogContent';

import { Iconify } from 'src/components/iconify';

// ----------------------------------------------------------------------

export default function ImageCropDialog({
  open,
  imageUrl,
  onClose,
  onSave,
  saving = false,
}) {
  const imgRef = useRef(null);
  const [crop, setCrop] = useState({
    unit: '%',
    width: 80,
    height: 60,
    x: 10,
    y: 20,
  });
  const [completedCrop, setCompletedCrop] = useState(null);
  const [rotation, setRotation] = useState(0);
  const [scale, setScale] = useState(1);

  const handleSave = async () => {
    if (!completedCrop || !imgRef.current) return;

    const image = imgRef.current;
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const scaleX = image.naturalWidth / image.width;
    const scaleY = image.naturalHeight / image.height;

    // Calculate crop area in pixels
    const pixelCrop = {
      x: completedCrop.x * scaleX,
      y: completedCrop.y * scaleY,
      width: completedCrop.width * scaleX,
      height: completedCrop.height * scaleY,
    };

    // Apply rotation
    if (rotation !== 0) {
      const rotRad = (rotation * Math.PI) / 180;
      const { width: origWidth, height: origHeight } = pixelCrop;
      
      // Calculate rotated dimensions
      const cos = Math.abs(Math.cos(rotRad));
      const sin = Math.abs(Math.sin(rotRad));
      const newWidth = origWidth * cos + origHeight * sin;
      const newHeight = origWidth * sin + origHeight * cos;

      canvas.width = newWidth;
      canvas.height = newHeight;

      ctx.translate(newWidth / 2, newHeight / 2);
      ctx.rotate(rotRad);
      ctx.scale(scale, scale);
      ctx.drawImage(
        image,
        pixelCrop.x,
        pixelCrop.y,
        pixelCrop.width,
        pixelCrop.height,
        -origWidth / 2,
        -origHeight / 2,
        origWidth,
        origHeight
      );
    } else {
      canvas.width = pixelCrop.width * scale;
      canvas.height = pixelCrop.height * scale;
      ctx.scale(scale, scale);
      ctx.drawImage(
        image,
        pixelCrop.x,
        pixelCrop.y,
        pixelCrop.width,
        pixelCrop.height,
        0,
        0,
        pixelCrop.width,
        pixelCrop.height
      );
    }

    onSave(pixelCrop, rotation, canvas);
  };

  const handleClose = () => {
    setCrop({
      unit: '%',
      width: 80,
      height: 60,
      x: 10,
      y: 20,
    });
    setCompletedCrop(null);
    setRotation(0);
    setScale(1);
    onClose();
  };

  const rotateLeft = () => setRotation((prev) => (prev - 90 + 360) % 360);
  const rotateRight = () => setRotation((prev) => (prev + 90) % 360);
  const resetRotation = () => setRotation(0);

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="lg" fullWidth>
      <DialogTitle sx={{ direction: 'rtl' }}>برش و ویرایش تصویر</DialogTitle>

      <DialogContent sx={{ minHeight: 500 }}>
        {/* Crop Container */}
        <Box
          sx={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            minHeight: 400,
            backgroundColor: '#f5f5f5',
            borderRadius: 1,
            overflow: 'hidden',
          }}
        >
          {imageUrl && (
            <ReactCrop
              crop={crop}
              onChange={(c) => setCrop(c)}
              onComplete={(c) => setCompletedCrop(c)}
              aspect={undefined}
              style={{
                maxWidth: '100%',
                maxHeight: 400,
              }}
            >
              <img
                ref={imgRef}
                src={imageUrl}
                alt="Crop"
                style={{
                  transform: `rotate(${rotation}deg) scale(${scale})`,
                  maxWidth: '100%',
                  maxHeight: 400,
                  display: 'block',
                }}
                crossOrigin="anonymous"
              />
            </ReactCrop>
          )}
        </Box>

        {/* Controls */}
        <Stack spacing={3} sx={{ mt: 3, direction: 'rtl' }}>
          {/* Scale Control */}
          <Box>
            <Stack direction="row" spacing={2} alignItems="center">
              <Iconify icon="solar:magnifer-zoom-out-linear" width={24} />
              <Typography variant="body2" sx={{ minWidth: 60 }}>
                مقیاس
              </Typography>
              <Slider
                value={scale}
                min={0.5}
                max={2}
                step={0.1}
                onChange={(e, value) => setScale(value)}
                sx={{ flexGrow: 1 }}
              />
              <Typography variant="body2" sx={{ minWidth: 50, textAlign: 'right' }}>
                {Math.round(scale * 100)}%
              </Typography>
            </Stack>
          </Box>

          {/* Rotation Control */}
          <Box>
            <Stack direction="row" spacing={2} alignItems="center">
              <Iconify icon="solar:refresh-linear" width={24} />
              <Typography variant="body2" sx={{ minWidth: 60 }}>
                چرخش
              </Typography>
              <Slider
                value={rotation}
                min={0}
                max={360}
                step={1}
                onChange={(e, value) => setRotation(value)}
                sx={{ flexGrow: 1 }}
              />
              <Typography variant="body2" sx={{ minWidth: 50, textAlign: 'right' }}>
                {rotation}°
              </Typography>
            </Stack>
          </Box>

          {/* Quick Rotation Buttons */}
          <Stack direction="row" spacing={1} justifyContent="center">
            <Button
              variant="outlined"
              size="small"
              startIcon={<Iconify icon="solar:restart-linear" />}
              onClick={resetRotation}
            >
              بازنشانی
            </Button>
            <Button
              variant="outlined"
              size="small"
              startIcon={<Iconify icon="solar:arrow-left-linear" />}
              onClick={rotateLeft}
            >
              90° چپ
            </Button>
            <Button
              variant="outlined"
              size="small"
              endIcon={<Iconify icon="solar:arrow-right-linear" />}
              onClick={rotateRight}
            >
              90° راست
            </Button>
          </Stack>
        </Stack>
      </DialogContent>

      <DialogActions sx={{ direction: 'rtl', px: 3, pb: 3 }}>
        <Button onClick={handleClose} disabled={saving}>
          انصراف
        </Button>
        <Button
          variant="contained"
          onClick={handleSave}
          disabled={saving || !completedCrop}
          startIcon={
            saving ? (
              <Iconify icon="svg-spinners:ring-resize" />
            ) : (
              <Iconify icon="solar:check-circle-bold" />
            )
          }
        >
          {saving ? 'در حال ذخیره...' : 'ذخیره برش'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

ImageCropDialog.propTypes = {
  open: PropTypes.bool.isRequired,
  imageUrl: PropTypes.string,
  onClose: PropTypes.func.isRequired,
  onSave: PropTypes.func.isRequired,
  saving: PropTypes.bool,
};
