import Stack from '@mui/material/Stack';
import Avatar from '@mui/material/Avatar';
import Collapse from '@mui/material/Collapse';
import Typography from '@mui/material/Typography';

import { useBoolean } from 'src/hooks/use-boolean';

import { getAvatarUrl } from 'src/utils/avatar-url';

import { Iconify } from 'src/components/iconify';

import { CollapseButton } from './styles';

// ----------------------------------------------------------------------

export function ChatRoomSingle({ participant }) {
  const collapse = useBoolean(true);

  const renderInfo = (
    <Stack alignItems="center" sx={{ py: 5 }}>
      <Avatar
        alt={participant?.name}
        src={getAvatarUrl(participant?.avatarUrl)}
        sx={{ width: 96, height: 96, mb: 2 }}
      />
      <Typography variant="subtitle1">{participant?.name}</Typography>
      <Typography variant="body2" sx={{ color: 'text.secondary', mt: 0.5 }}>
        {(() => {
          const role = (participant?.role || '').toUpperCase();
          return role === 'DOCTOR' ? 'دکتر' : role === 'ADMIN' ? 'ادمین' : role === 'PATIENT' ? 'بیمار' : participant?.role || '';
        })()}
      </Typography>
    </Stack>
  );

  const renderContact = (
    <Stack spacing={2} sx={{ px: 2, py: 2.5 }}>
      {[
        { icon: 'solar:phone-bold', label: 'شماره تلفن', value: participant?.phone || participant?.phoneNumber },
        { icon: 'fluent:mail-24-filled', label: 'ایمیل', value: participant?.email },
        { icon: 'solar:user-id-bold', label: 'تخصص', value: participant?.specialty },
      ]
        .filter((item) => item.value) // Only show items with values
        .map((item) => (
          <Stack
            key={item.icon}
            spacing={1}
            direction="row"
            sx={{ typography: 'body2', wordBreak: 'break-all' }}
          >
            <Iconify icon={item.icon} sx={{ flexShrink: 0, color: 'text.disabled' }} />
            <Typography variant="body2" sx={{ color: 'text.secondary' }}>
              {item.label}: {item.value}
            </Typography>
          </Stack>
        ))}
    </Stack>
  );

  return (
    <>
      {renderInfo}

      <CollapseButton selected={collapse.value} onClick={collapse.onToggle}>
        اطلاعات
      </CollapseButton>

      <Collapse in={collapse.value}>{renderContact}</Collapse>
    </>
  );
}
