import { useState } from 'react';

import Box from '@mui/material/Box';
import Alert from '@mui/material/Alert';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Typography from '@mui/material/Typography';
import DialogTitle from '@mui/material/DialogTitle';
import FormControl from '@mui/material/FormControl';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import CircularProgress from '@mui/material/CircularProgress';

import { Iconify } from 'src/components/iconify';

// AI Models for Cephalometric Analysis
const AI_MODELS = [
  {
    id: 'cephx-v1',
    name: 'CephX v1.0',
    description: 'مدل پایه - سریع و کارآمد برای تحلیل‌های استاندارد',
    accuracy: 92,
    speed: 'سریع',
    icon: 'solar:cpu-bold',
  },
  {
    id: 'cephx-v2',
    name: 'CephX v2.0',
    description: 'مدل پیشرفته - دقت بالاتر برای موارد پیچیده',
    accuracy: 95,
    speed: 'متوسط',
    icon: 'solar:cpu-bolt-bold',
  },
  {
    id: 'deepceph',
    name: 'DeepCeph',
    description: 'مدل عمیق - بهترین دقت برای تحلیل‌های تخصصی',
    accuracy: 97,
    speed: 'کند',
    icon: 'solar:atom-bold',
  },
  {
    id: 'gpt-4o-vision',
    name: 'GPT-4o Vision',
    description: 'مدل چندمنظوره OpenAI - تحلیل هوشمند با توضیحات',
    accuracy: 94,
    speed: 'متوسط',
    icon: 'solar:mind-bold',
  },
  {
    id: 'claude-vision',
    name: 'Claude 3.5 Vision',
    description: 'مدل Anthropic - تحلیل دقیق با رعایت اصول پزشکی',
    accuracy: 95,
    speed: 'متوسط',
    icon: 'solar:brain-bold',
  },
  {
    id: 'train_p1_p2_heatmap',
    name: 'P1/P2 Heatmap Detector',
    description: 'مدل تشخیص نقاط کالیبراسیون P1 و P2 با استفاده از HRNet Heatmap - خطای حدود 2px',
    accuracy: 98,
    speed: 'سریع',
    icon: 'solar:target-bold',
  },
];

export function AIModelSelector({ open, onClose, onDetect, currentModel, isDetecting }) {
  const [selectedModel, setSelectedModel] = useState(currentModel || 'cephx-v2');
  const [showAdvanced, setShowAdvanced] = useState(false);

  const handleDetect = () => {
    onDetect(selectedModel);
  };

  const currentModelInfo = AI_MODELS.find((m) => m.id === selectedModel) || AI_MODELS[1];

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Iconify icon="solar:cpu-bolt-bold-duotone" width={28} />
          <Typography variant="h6">تشخیص خودکار لندمارک‌ها با هوش مصنوعی</Typography>
        </Box>
      </DialogTitle>

      <DialogContent>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
          یک مدل هوش مصنوعی برای تشخیص خودکار نقاط آناتومیک انتخاب کنید
        </Typography>

        <FormControl fullWidth sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            مدل هوش مصنوعی
          </Typography>
          <Select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
            disabled={isDetecting}
          >
            {AI_MODELS.map((model) => (
              <MenuItem key={model.id} value={model.id}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, py: 0.5 }}>
                  <Iconify icon={model.icon} width={20} />
                  <Box>
                    <Typography variant="body2" fontWeight="medium">
                      {model.name}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      {model.description}
                    </Typography>
                  </Box>
                </Box>
              </MenuItem>
            ))}
          </Select>
        </FormControl>

        {/* Model Info Card */}
        <Box
          sx={{
            p: 2,
            borderRadius: 2,
            bgcolor: 'background.neutral',
            border: '1px solid',
            borderColor: 'divider',
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
            <Iconify icon={currentModelInfo.icon} width={24} color="primary.main" />
            <Typography variant="subtitle1" fontWeight="bold">
              {currentModelInfo.name}
            </Typography>
          </Box>

          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            {currentModelInfo.description}
          </Typography>

          <Box sx={{ display: 'flex', gap: 3 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">
                دقت
              </Typography>
              <Typography variant="h6" color="success.main">
                {currentModelInfo.accuracy}%
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">
                سرعت
              </Typography>
              <Typography variant="body2" fontWeight="medium">
                {currentModelInfo.speed}
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* Advanced Options */}
        <Box sx={{ mt: 2 }}>
          <Button
            size="small"
            onClick={() => setShowAdvanced(!showAdvanced)}
            endIcon={
              <Iconify icon={showAdvanced ? 'eva:arrow-up-fill' : 'eva:arrow-down-fill'} />
            }
          >
            تنظیمات پیشرفته
          </Button>

          {showAdvanced && (
            <Alert severity="info" sx={{ mt: 2 }}>
              <Typography variant="body2">
                • مدل‌های CephX بر روی داده‌های سفالومتریک فارسی‌زبان آموزش دیده‌اند
                <br />
                • مدل‌های GPT-4o و Claude برای تحلیل‌های پیچیده‌تر مناسب هستند
                <br />
                • همه نتایج قابل ویرایش دستی هستند
                <br />• زمان پردازش: 5-15 ثانیه بسته به مدل انتخابی
              </Typography>
            </Alert>
          )}
        </Box>

        {isDetecting && (
          <Box
            sx={{
              mt: 3,
              p: 2,
              borderRadius: 2,
              bgcolor: 'info.lighter',
              display: 'flex',
              alignItems: 'center',
              gap: 2,
            }}
          >
            <CircularProgress size={24} />
            <Box>
              <Typography variant="body2" fontWeight="medium">
                در حال پردازش تصویر...
              </Typography>
              <Typography variant="caption" color="text.secondary">
                لطفاً صبر کنید، مدل در حال تحلیل تصویر است
              </Typography>
            </Box>
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={onClose} disabled={isDetecting}>
          انصراف
        </Button>
        <Button
          variant="contained"
          onClick={handleDetect}
          disabled={isDetecting}
          startIcon={
            isDetecting ? <CircularProgress size={16} /> : <Iconify icon="solar:cpu-bolt-bold" />
          }
        >
          {isDetecting ? 'در حال تشخیص...' : 'شروع تشخیص'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default AIModelSelector;

