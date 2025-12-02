import { useMemo } from 'react';
import useSWR, { mutate } from 'swr';

import { keyBy } from 'src/utils/helper';
import axios, { fetcher } from 'src/utils/axios';

// Endpoints (DB-backed API)
const ENDPOINTS = {
  doctors: '/api/chat/doctors',
  list: '/api/chat/list',
  conversation: (id) => `/api/chat/${id}`,
  send: '/api/chat/send',
};

const swrOptions = {
  revalidateIfStale: true,
  revalidateOnFocus: true,
  revalidateOnReconnect: true,
};

export function useGetContacts() {
  // Instead of using doctors endpoint, we want to show actual contacts from chat
  // First, we need to get the conversations to extract unique contacts
  const { conversations, conversationsLoading, conversationsError, conversationsValidating } = useGetConversations();

  const memoizedValue = useMemo(() => {
    // Extract unique contacts from conversations
    const contactsSet = new Set();
    const contacts = [];

    if (conversations?.byId) {
      // Loop through all conversations to get unique other participants
      Object.values(conversations.byId).forEach(chat => {
        if (chat.otherParticipant && !contactsSet.has(chat.otherParticipant.id)) {
          contactsSet.add(chat.otherParticipant.id);
          contacts.push({
            id: chat.otherParticipant.id,
            name: chat.otherParticipant.name,
            firstName: chat.otherParticipant.firstName,
            lastName: chat.otherParticipant.lastName,
            email: chat.otherParticipant.email,
            specialty: chat.otherParticipant.specialty,
            phone: chat.otherParticipant.phone,
            isVerified: chat.otherParticipant.isVerified,
            lastActivity: new Date(chat.updatedAt).toISOString().split('T')[0], // Use last chat date
            avatarUrl: chat.otherParticipant.avatarUrl || null,
            status: 'offline', // We'll determine online status later if needed
          });
        }
      });
    }

    return {
      contacts,
      contactsLoading: conversationsLoading,
      contactsError: conversationsError,
      contactsValidating: conversationsValidating,
      contactsEmpty: !conversationsLoading && !contacts.length,
    };
  }, [conversations, conversationsLoading, conversationsError, conversationsValidating]);

  return memoizedValue;
}

export function useGetConversations() {
  const url = ENDPOINTS.list;

  const { data, isLoading, error, isValidating } = useSWR(url, fetcher, swrOptions);

  const memoizedValue = useMemo(() => {
    const conv = data?.chats || [];
    const byId = conv.length ? keyBy(conv, 'id') : {};
    const allIds = Object.keys(byId);

    return {
      conversations: { byId, allIds },
      conversationsLoading: isLoading,
      conversationsError: error,
      conversationsValidating: isValidating,
      conversationsEmpty: !isLoading && !allIds.length,
    };
  }, [data?.chats, error, isLoading, isValidating]);

  return memoizedValue;
}

export function useGetConversation(conversationId) {
  const url = conversationId ? ENDPOINTS.conversation(conversationId) : null;

  const { data, isLoading, error, isValidating } = useSWR(url, fetcher, swrOptions);

  const memoizedValue = useMemo(
    () => ({
      conversation: data?.chat,
      conversationLoading: isLoading,
      conversationError: error,
      conversationValidating: isValidating,
    }),
    [data?.chat, error, isLoading, isValidating]
  );

  return memoizedValue;
}

export async function sendMessage(conversationId, messageData) {
  // If conversationId is undefined, send will create the chat on server (server handles finding/creating chat)
  const payload = {
    recipientId: messageData.recipientId || null,
    content: messageData.content,
  };

  const res = await axios.post(ENDPOINTS.send, payload);

  // Update SWR caches
  const sentMessage = res.data?.message;
  const chatId = res.data?.chatId;

  if (chatId) {
    mutate(ENDPOINTS.conversation(chatId));
    mutate(ENDPOINTS.list);
  }

  // Return the chatId so frontend can navigate to the new conversation
  return {
    message: sentMessage,
    chatId,
  };
}

export async function createConversation(conversationData) {
  const res = await axios.post(ENDPOINTS.list, { conversationData });
  mutate(ENDPOINTS.list);
  return res.data;
}

export async function clickConversation(conversationId) {
  // mark-as-seen endpoint not implemented in DB API yet; fallback: revalidate list
  mutate(ENDPOINTS.list);
}
