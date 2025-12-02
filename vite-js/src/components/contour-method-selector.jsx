import { useState } from 'react';
import PropTypes from 'prop-types';
import { Icon } from '@iconify/react';

import {
  Box,
  Card,
  Chip,
  Alert,
  Stack,
  Select,
  MenuItem,
  InputLabel,
  Typography,
  FormControl
} from '@mui/material';

// روش‌های تشخیص کانتور
const CONTOUR_METHODS = [
  {
    id: 'auto',
    name: 'خودکار (Auto)',
    nameEn: 'Auto',
    description: 'انتخاب خودکار بهترین روش',
    icon: 'solar:magic-stick-3-bold',
    color: 'primary',
    speed: 3,
    accuracy: 4,
    recommendedFor: ['همه موارد']
  },
  {
    id: 'ryanchoi',
    name: 'RyanChoi پیشرفته',
    nameEn: 'RyanChoi Advanced',
    description: 'روش پیشرفته با Active Contour و Level Set',
    icon: 'solar:atom-bold',
    color: 'success',
    speed: 2,
    accuracy: 5,
    recommendedFor: ['همه موارد', 'دقت بالا']
  },
  {
    id: 'enhanced_landmark',
    name: 'هدایت با لندمارک',
    nameEn: 'Landmark-Guided',
    description: 'استفاده از دانش آناتومیک',
    icon: 'solar:target-bold',
    color: 'info',
    speed: 4,
    accuracy: 4,
    recommendedFor: ['Sella Turcica', 'Orbital Rim', 'ساختارهای دایره‌ای']
  },
  {
    id: 'enhanced_watershed',
    name: 'Watershed',
    nameEn: 'Watershed',
    description: 'جداسازی Foreground/Background',
    icon: 'solar:layer-bold',
    color: 'warning',
    speed: 4,
    accuracy: 3,
    recommendedFor: ['Mandible', 'Maxilla', 'استخوان‌ها']
  },
  {
    id: 'enhanced_grabcut',
    name: 'GrabCut',
    nameEn: 'GrabCut',
    description: 'بهینه برای بافت نرم',
    icon: 'solar:scissors-square-bold',
    color: 'secondary',
    speed: 3,
    accuracy: 4,
    recommendedFor: ['Soft Tissue', 'Lips', 'بافت نرم']
  }
];

export default function ContourMethodSelector({ 
  value = 'auto', 
  onChange, 
  disabled = false,
  showDetails = true,
  variant = 'outlined',
  size = 'medium'
}) {
  const [selectedMethod, setSelectedMethod] = useState(value);

  const handleChange = (event) => {
    const newValue = event.target.value;
    setSelectedMethod(newValue);
    if (onChange) {
      onChange(newValue);
    }
  };

  const currentMethod = CONTOUR_METHODS.find(m => m.id === selectedMethod) || CONTOUR_METHODS[0];
  
  // ستاره‌های امتیاز
  const renderStars = (count, max = 5) => Array.from({ length: max }, (_, i) => (
      <Icon
        key={i}
        icon={i < count ? 'solar:star-bold' : 'solar:star-outline'}
        width={14}
        style={{ color: i < count ? '#FFD700' : '#DDD' }}
      />
    ));

  return (
    <Box>
      <FormControl fullWidth variant={variant} size={size} disabled={disabled}>
        <InputLabel>روش تشخیص کانتور</InputLabel>
        <Select
          value={selectedMethod}
          onChange={handleChange}
          label="روش تشخیص کانتور"
        >
          {CONTOUR_METHODS.map((method) => (
            <MenuItem key={method.id} value={method.id}>
              <Stack direction="row" spacing={1} alignItems="center">
                <Icon icon={method.icon} width={20} />
                <Typography variant="body2">{method.name}</Typography>
                <Chip 
                  label={method.nameEn} 
                  size="small" 
                  color={method.color}
                  variant="outlined"
                />
              </Stack>
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      {showDetails && currentMethod && (
        <Card sx={{ mt: 2, p: 2, bgcolor: 'background.neutral' }}>
          <Stack spacing={2}>
            <Stack direction="row" spacing={1} alignItems="center">
              <Icon icon={currentMethod.icon} width={24} />
              <Typography variant="subtitle1" fontWeight="bold">
                {currentMethod.name}
              </Typography>
              <Chip 
                label={currentMethod.nameEn} 
                size="small" 
                color={currentMethod.color}
              />
            </Stack>

            <Typography variant="body2" color="text.secondary">
              {currentMethod.description}
            </Typography>

            <Stack spacing={1}>
              <Stack direction="row" spacing={2} alignItems="center">
                <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
                  سرعت
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  {renderStars(currentMethod.speed)}
                </Box>
              </Stack>

              <Stack direction="row" spacing={2} alignItems="center">
                <Typography variant="caption" color="text.secondary" sx={{ minWidth: 60 }}>
                  دقت
                </Typography>
                <Box sx={{ display: 'flex', gap: 0.5 }}>
                  {renderStars(currentMethod.accuracy)}
                </Box>
              </Stack>
            </Stack>

            {currentMethod.recommendedFor && currentMethod.recommendedFor.length > 0 && (
              <Box>
                <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
                  مناسب برای:
                </Typography>
                <Stack direction="row" spacing={0.5} flexWrap="wrap" useFlexGap>
                  {currentMethod.recommendedFor.map((item, index) => {
                    const chipColor = currentMethod.color;
                    return (
                      <Chip
                        key={index}
                        label={item}
                        size="small"
                        variant="filled"
                        color={chipColor}
                        sx={{
                          fontSize: '0.7rem',
                          height: 24,
                          bgcolor: (theme) => theme.palette[chipColor].lighter,
                          color: (theme) => theme.palette[chipColor].darker,
                        }}
                      />
                    );
                  })}
                </Stack>
              </Box>
            )}
          </Stack>
        </Card>
      )}

      {currentMethod.id === 'ryanchoi' && (
        <Alert severity="info" sx={{ mt: 1 }} icon={<Icon icon="solar:info-circle-bold" />}>
          این روش دقت‌ترین است اما کمی کندتر. برای نتایج بهتر توصیه می‌شود.
        </Alert>
      )}
    </Box>
  );
}

ContourMethodSelector.propTypes = {
  value: PropTypes.string,
  onChange: PropTypes.func,
  disabled: PropTypes.bool,
  showDetails: PropTypes.bool,
  variant: PropTypes.oneOf(['outlined', 'filled', 'standard']),
  size: PropTypes.oneOf(['small', 'medium'])
};

// Export methods for use in other components
export { CONTOUR_METHODS };
