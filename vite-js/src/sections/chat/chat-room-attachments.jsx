import Stack from '@mui/material/Stack';
import Collapse from '@mui/material/Collapse';
import ListItemText from '@mui/material/ListItemText';

import { useBoolean } from 'src/hooks/use-boolean';

import { fDateTime } from 'src/utils/format-time';

import { FileThumbnail } from 'src/components/file-thumbnail';

import { CollapseButton } from './styles';

// ----------------------------------------------------------------------

export function ChatRoomAttachments({ attachments }) {
  const collapse = useBoolean(true);

  const totalAttachments = attachments.length;

  const renderList = attachments
    .filter((attachment) => attachment && attachment.name) // Filter out invalid attachments
    .map((attachment, index) => (
      <Stack key={attachment.name + index} spacing={1.5} direction="row" alignItems="center">
        <FileThumbnail
          imageView
          file={attachment.preview || attachment.url || attachment.path || attachment}
          onDownload={() => console.info('DOWNLOAD')}
          slotProps={{ icon: { width: 24, height: 24 } }}
          sx={{ width: 40, height: 40, bgcolor: 'background.neutral' }}
        />

        <ListItemText
          primary={attachment.name || 'پیوست بدون نام'}
          secondary={attachment.createdAt ? fDateTime(attachment.createdAt) : ''}
          primaryTypographyProps={{ noWrap: true, typography: 'body2' }}
          secondaryTypographyProps={{
            mt: 0.25,
            noWrap: true,
            component: 'span',
            typography: 'caption',
            color: 'text.disabled',
          }}
        />
      </Stack>
    ));

  return (
    <>
      <CollapseButton
        selected={collapse.value}
        disabled={!totalAttachments}
        onClick={collapse.onToggle}
      >
        {`پیوست‌ها (${totalAttachments})`}
      </CollapseButton>

      {!!totalAttachments && (
        <Collapse in={collapse.value}>
          <Stack spacing={2} sx={{ p: 2 }}>
            {renderList}
          </Stack>
        </Collapse>
      )}
    </>
  );
}
