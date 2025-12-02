import axios from 'axios';
import { useRef, useMemo, useState, useCallback } from 'react';

import Stack from '@mui/material/Stack';
import InputBase from '@mui/material/InputBase';
import IconButton from '@mui/material/IconButton';
import Typography from '@mui/material/Typography';

import { paths } from 'src/routes/paths';
import { useRouter } from 'src/routes/hooks';

import { uuidv4 } from 'src/utils/uuidv4';
import { fSub, today } from 'src/utils/format-time';

import { CONFIG } from 'src/config-global';
import { sendMessage } from 'src/actions/chat';

import { Iconify } from 'src/components/iconify';

import { useMockedUser , useAuthContext } from 'src/auth/hooks';


// ----------------------------------------------------------------------

export function ChatMessageInput({
  disabled,
  recipients,
  onAddRecipients,
  selectedConversationId,
  participants = [],
}) {
  const router = useRouter();

  const { user } = useMockedUser();
  const { user: authUser } = useAuthContext(); // Use proper auth user

  const fileRef = useRef(null);

  const [message, setMessage] = useState('');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);

  const myContact = useMemo(
    () => ({
      id: `${authUser?.id || user?.id}`,
      role: `${authUser?.role || user?.role}`,
      email: `${authUser?.email || user?.email}`,
      address: `${authUser?.address || user?.address}`,
      name: `${authUser?.displayName || user?.displayName}`,
      lastActivity: today(),
      avatarUrl: `${authUser?.photoURL || user?.photoURL}`,
      phoneNumber: `${authUser?.phoneNumber || user?.phoneNumber}`,
      status: 'online',
    }),
    [authUser, user]
  );

  const messageData = useMemo(
    () => {
      // Determine recipient ID
      let recipientId = null;

      // If we have an existing conversation, get recipient from participants
      if (selectedConversationId && participants && participants.length > 0) {
        const firstParticipant = participants[0];
        if (firstParticipant && firstParticipant.id) {
          recipientId = firstParticipant.id;
        }
      }
      // If we're starting a new conversation, get recipient from recipients array
      else if (recipients && recipients.length > 0) {
        const firstRecipient = recipients[0];
        if (firstRecipient && firstRecipient.id) {
          recipientId = String(firstRecipient.id);
        }
      }

      return {
        id: uuidv4(),
        attachments: selectedFiles,
        content: message.trim(),
        contentType: selectedFiles.length > 0 ? 'image' : 'text',
        createdAt: fSub({ minutes: 1 }),
        senderId: myContact.id,
        recipientId,
      };
    },
    [message, selectedFiles, myContact.id, selectedConversationId, participants, recipients]
  );

  const conversationData = useMemo(
    () => ({
      id: uuidv4(),
      messages: [messageData],
      participants: [...recipients, myContact],
      type: recipients.length > 1 ? 'GROUP' : 'ONE_TO_ONE',
      unreadCount: 0,
    }),
    [messageData, myContact, recipients]
  );

  const handleAttach = useCallback(() => {
    if (fileRef.current) {
      fileRef.current.click();
    }
  }, []);

  const handleFileUpload = useCallback(async (files, type) => {
    setUploading(true);
    try {
      const uploadPromises = files.map(async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type === 'image' ? 'image' : 'document');

        const target = `${(CONFIG.site.serverUrl || 'http://localhost:7272').replace(/\/$/, '')}/api/upload/chat`;

        const response = await axios.post(target, formData, {
          headers: {
            Authorization: `Bearer ${authUser?.accessToken || user?.accessToken}`,
          },
        });

        return {
          name: file.name,
          size: file.size,
          type: file.type,
          url: response.data.url,
          preview: type === 'image' ? response.data.url : null,
        };
      });

      const uploadedFiles = await Promise.all(uploadPromises);

      setSelectedFiles(uploadedFiles);
    } catch (error) {
      console.error('File upload error:', error);
      alert('خطا در آپلود فایل');
    } finally {
      setUploading(false);
    }
  }, [user, authUser]);

  const handleAttachImage = useCallback(() => {
    const imageInput = document.createElement('input');
    imageInput.type = 'file';
    imageInput.accept = 'image/*';
    imageInput.onchange = async (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        await handleFileUpload(files, 'image');
      }
    };
    imageInput.click();
  }, [handleFileUpload]);

  const handleAttachFile = useCallback(() => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.onchange = async (e) => {
      const files = Array.from(e.target.files || []);
      if (files.length > 0) {
        await handleFileUpload(files, 'file');
      }
    };
    fileInput.click();
  }, [handleFileUpload]);

  const handleRemoveAttachment = useCallback((index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleChangeMessage = useCallback((event) => {
    setMessage(event.target.value);
  }, []);

  const handleSendMessage = useCallback(
    async () => {
      try {
        const hasContent = message.trim() || selectedFiles.length > 0;

        if (hasContent) {
          // For both existing and new conversations, just send the message
          // The send API will handle creating the conversation if it doesn't exist
          const res = await sendMessage(selectedConversationId, messageData);

          // If this was a new conversation (no selectedConversationId), navigate to the created chat
          if (!selectedConversationId && res.chatId) {
            router.push(`${paths.dashboard.chat}?id=${res.chatId}`);
            onAddRecipients([]);
          }

          // Reset state
          setMessage('');
          setSelectedFiles([]);
        }
      } catch (error) {
        console.error('Send message error:', error);
        alert('خطا در ارسال پیام');
      }
    },
    [message, messageData, onAddRecipients, router, selectedConversationId, selectedFiles]
  );

  return (
    <>
      {/* Show selected files if any */}
      {selectedFiles.length > 0 && (
        <Stack direction="row" spacing={1} sx={{ p: 1, borderTop: (theme) => `solid 1px ${theme.vars.palette.divider}` }}>
          {selectedFiles.map((file, index) => (
            <Stack key={index} direction="row" spacing={1} alignItems="center" sx={{ bgcolor: 'grey.100', p: 1, borderRadius: 1 }}>
              <Typography variant="body2" sx={{ maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                {file.name}
              </Typography>
              <IconButton size="small" onClick={() => handleRemoveAttachment(index)}>
                <Iconify icon="eva:close-fill" width={16} />
              </IconButton>
            </Stack>
          ))}
        </Stack>
      )}

      <InputBase
        name="chat-message"
        id="chat-message-input"
        value={message}
        onChange={handleChangeMessage}
        placeholder="یک پیام بفرستید"
        disabled={disabled || uploading}
        onKeyDown={(event) => {
          if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            handleSendMessage();
          }
        }}
        startAdornment={
			<Stack direction="row" sx={{ flexShrink: 0 }}>
          <IconButton onClick={handleSendMessage} disabled={disabled || uploading || (!message.trim() && selectedFiles.length === 0)}>
            <Iconify icon="carbon:send-filled" />
          </IconButton>

		  </Stack>
        }
        endAdornment={
          <Stack direction="row" sx={{ flexShrink: 0 }}>
		              <IconButton onClick={handleAttachFile} disabled={uploading}>
              <Iconify icon="eva:attach-2-fill" />
            </IconButton>

          </Stack>
        }
        sx={{
          px: 1,
          height: 56,
          flexShrink: 0,
          borderTop: (theme) => `solid 1px ${theme.vars.palette.divider}`,
          '& input::placeholder': {
            fontSize: '0.875rem', // Smaller font size for placeholder
          },
        }}
      />

      <input type="file" ref={fileRef} style={{ display: 'none' }} />
    </>
  );
}
