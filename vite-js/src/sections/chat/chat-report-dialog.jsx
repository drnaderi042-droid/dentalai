import { useState } from 'react';

import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Radio from '@mui/material/Radio';
import Button from '@mui/material/Button';
import Dialog from '@mui/material/Dialog';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import RadioGroup from '@mui/material/RadioGroup';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import FormControlLabel from '@mui/material/FormControlLabel';

import axiosInstance, { endpoints } from 'src/utils/axios';

import { toast } from 'src/components/snackbar';
import { Iconify } from 'src/components/iconify';

import { useAuthContext } from 'src/auth/hooks';

// ----------------------------------------------------------------------

const REPORT_REASONS = [
  { value: 'spam', label: 'اسپم' },
  { value: 'harassment', label: 'آزار و اذیت' },
  { value: 'inappropriate_content', label: 'محتوای نامناسب' },
  { value: 'fake_account', label: 'حساب جعلی' },
  { value: 'other', label: 'سایر' },
];

// ----------------------------------------------------------------------

export function ChatReportDialog({ open, onClose, reportedUserId, reportedUserName }) {
  const { user } = useAuthContext();
  const [reason, setReason] = useState('');
  const [description, setDescription] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const handleReasonChange = (event) => {
    setReason(event.target.value);
    if (event.target.value !== 'other') {
      setDescription('');
    }
  };

  const handleSubmit = async () => {
    if (!reason) {
      toast.error('لطفاً دلیل گزارش را انتخاب کنید');
      return;
    }

    if (reason === 'other' && !description.trim()) {
      toast.error('لطفاً توضیحات را وارد کنید');
      return;
    }

    try {
      setSubmitting(true);
      await axiosInstance.post(
        endpoints.chat.report,
        {
          reportedId: reportedUserId,
          reason,
          description: reason === 'other' ? description : null,
        },
        {
          headers: {
            Authorization: `Bearer ${user?.accessToken}`,
          },
        }
      );

      toast.success('گزارش با موفقیت ارسال شد. ادمین آن را بررسی خواهد کرد.');
      setReason('');
      setDescription('');
      onClose();
    } catch (error) {
      console.error('Error submitting report:', error);
      const errorMessage = error.response?.data?.message || error.message || 'خطا در ارسال گزارش';
      toast.error(errorMessage);
    } finally {
      setSubmitting(false);
    }
  };

  const handleClose = () => {
    if (!submitting) {
      setReason('');
      setDescription('');
      onClose();
    }
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="sm" fullWidth>
      <DialogTitle>
        <Stack direction="row" alignItems="center" spacing={2}>
          <Iconify icon="solar:danger-triangle-bold" width={24} />
          <Typography variant="h6">گزارش کاربر</Typography>
        </Stack>
      </DialogTitle>

      <DialogContent>
        <Typography variant="body2" sx={{ mb: 3, color: 'text.secondary' }}>
          گزارش کاربر: <strong>{reportedUserName}</strong>
        </Typography>

        <Typography variant="subtitle2" sx={{ mb: 2 }}>
          دلیل گزارش را انتخاب کنید:
        </Typography>

        <RadioGroup value={reason} onChange={handleReasonChange}>
          {REPORT_REASONS.map((option) => (
            <FormControlLabel
              key={option.value}
              value={option.value}
              control={<Radio />}
              label={option.label}
            />
          ))}
        </RadioGroup>

        {reason === 'other' && (
          <Box sx={{ mt: 3 }}>
            <TextField
              fullWidth
              multiline
              rows={4}
              label="توضیحات"
              placeholder="لطفاً دلیل گزارش را به صورت کامل توضیح دهید..."
              value={description}
              onChange={(e) => setDescription(e.target.value)}
            />
          </Box>
        )}
      </DialogContent>

      <DialogActions>
        <Button onClick={handleClose} disabled={submitting}>
          انصراف
        </Button>
        <Button
          variant="contained"
          color="error"
          onClick={handleSubmit}
          disabled={submitting || !reason || (reason === 'other' && !description.trim())}
        >
          {submitting ? 'در حال ارسال...' : 'ارسال گزارش'}
        </Button>
      </DialogActions>
    </Dialog>
  );
}


