import { useState, useEffect, useCallback } from 'react';

import Typography from '@mui/material/Typography';

import { CONFIG } from 'src/config-global';
import { DashboardContent } from 'src/layouts/dashboard';
import { useGetContacts, useGetConversation, useGetConversations } from 'src/actions/chat';

import { EmptyContent } from 'src/components/empty-content';

import { useMockedUser } from 'src/auth/hooks';

import { Layout } from '../layout';
import { ChatNav } from '../chat-nav';
import { ChatRoom } from '../chat-room';
import { ChatMessageList } from '../chat-message-list';
import { ChatMessageInput } from '../chat-message-input';
import { ChatHeaderDetail } from '../chat-header-detail';
import { ChatHeaderCompose } from '../chat-header-compose';
import { useCollapseNav } from '../hooks/use-collapse-nav';

// ----------------------------------------------------------------------

export default function ChatView() {
  const { contacts } = useGetContacts();

  const searchParams = new URLSearchParams(window.location.search);
  const selectedConversationId = searchParams.get('id') || '';

  const [recipients, setRecipients] = useState([]);

  const { conversations, conversationsLoading } = useGetConversations();

  const { conversation, conversationError, conversationLoading } = useGetConversation(
    `${selectedConversationId}`
  );

  const roomNav = useCollapseNav();
  const conversationsNav = useCollapseNav();

  const { user } = useMockedUser();

  let participants = [];
  if (conversation) {
    const raw = conversation.participants;
    if (Array.isArray(raw) && raw.length && typeof raw[0] === 'object') {
      participants = raw.filter((p) => String(p.id) !== String(user?.id));
    } else if (conversation.otherParticipant) {
      participants = [{
        id: conversation.otherParticipant.id,
        name: `${conversation.otherParticipant.firstName} ${conversation.otherParticipant.lastName}`,
        firstName: conversation.otherParticipant.firstName,
        lastName: conversation.otherParticipant.lastName,
        email: conversation.otherParticipant.email,
        phone: conversation.otherParticipant.phone,
        phoneNumber: conversation.otherParticipant.phone,
        specialty: conversation.otherParticipant.specialty,
        role: conversation.otherParticipant.role || conversation.otherParticipant.specialty,
        avatarUrl: conversation.otherParticipant.avatarUrl || null,
        lastActivity: conversation.otherParticipant.lastActivity || conversation.otherParticipant.updatedAt,
        status: 'offline',
      }];
    } else {
      participants = [];
    }
  }

  useEffect(() => {
    if (conversationError || !selectedConversationId) {
      if (window?.history?.pushState) window.history.pushState({}, '', '/dashboard/chat');
    }
  }, [conversationError, selectedConversationId]);

  const handleAddRecipients = useCallback((selected) => {
    setRecipients(selected);
  }, []);

  return (
    <DashboardContent
      maxWidth={false}
      sx={{ display: 'flex', flex: '1 1 auto', flexDirection: 'column' }}
    >
      <Typography variant="h4" sx={{ mb: { xs: 3, md: 5 } }}>
        چت
      </Typography>

      <Layout
        sx={{
          minHeight: 0,
          flex: '1 1 0',
          borderRadius: 2,
          position: 'relative',
          bgcolor: 'background.paper',
          boxShadow: (theme) => theme.customShadows.card,
        }}
        slots={{
          header: selectedConversationId ? (
            <ChatHeaderDetail collapseNav={roomNav} participants={participants} loading={conversationLoading} />
          ) : (
            <ChatHeaderCompose contacts={contacts} onAddRecipients={handleAddRecipients} />
          ),
          nav: (
            <ChatNav
              contacts={contacts}
              conversations={conversations}
              loading={conversationsLoading}
              selectedConversationId={selectedConversationId}
              collapseNav={conversationsNav}
            />
          ),
          main: (
            <>
              {selectedConversationId ? (
                <ChatMessageList messages={conversation?.messages ?? []} participants={participants} loading={conversationLoading} />
              ) : (
                <EmptyContent
                  imgUrl={`${CONFIG.site.basePath}/assets/icons/empty/ic-chat-active.svg`}
                  title="خوش آمدید!"
                  description="یک مکالمه را انتخاب کنید یا مکالمه جدیدی شروع کنید..."
                />
              )}

              <ChatMessageInput
                recipients={recipients}
                onAddRecipients={handleAddRecipients}
                selectedConversationId={selectedConversationId}
                participants={participants}
                disabled={!recipients.length && !selectedConversationId}
              />
            </>
          ),
          details: selectedConversationId && (
            <ChatRoom collapseNav={roomNav} participants={participants} loading={conversationLoading} messages={conversation?.messages ?? []} />
          ),
        }}
      />
    </DashboardContent>
  );
}
