// ----------------------------------------------------------------------

export function useNavItem({ currentUserId, conversation }) {
  // Defensive defaults
  const rawMessages = (conversation && conversation.messages) || (conversation && conversation.lastMessage ? [conversation.lastMessage] : []) || [];

  // Participants can be an array of user objects or array of userIds, and some APIs provide otherParticipant
  const participantsRaw = conversation && conversation.participants;
  let participantsInConversation = [];

  if (Array.isArray(participantsRaw) && participantsRaw.length) {
    // If participants are objects
    if (typeof participantsRaw[0] === 'object') {
      participantsInConversation = participantsRaw.filter((p) => String(p.id) !== String(currentUserId));
    } else if (conversation.otherParticipant) {
      // participants are ids; try to use otherParticipant
      participantsInConversation = [conversation.otherParticipant];
    } else {
      participantsInConversation = [];
    }
  } else if (conversation && conversation.otherParticipant) {
    participantsInConversation = [conversation.otherParticipant];
  } else {
    participantsInConversation = [];
  }

  const lastMessage = rawMessages.length ? rawMessages[rawMessages.length - 1] : null;

  const group = participantsInConversation.length > 1;

  const displayName = participantsInConversation
    .map((participant) => participant.name || `${participant.firstName || ''} ${participant.lastName || ''}`.trim())
    .filter(Boolean)
    .join(', ');

  const hasOnlineInGroup = group
    ? participantsInConversation.some((item) => (item.status || '').toLowerCase() === 'online')
    : false;

  let displayText = '';

  if (lastMessage) {
    const sender = String(lastMessage.senderId) === String(currentUserId) ? 'You: ' : '';

    const messageContent = lastMessage.content ?? lastMessage.body ?? '';

    const isImage = (lastMessage.contentType && lastMessage.contentType === 'image') ||
      (!!lastMessage.attachments && lastMessage.attachments.length > 0 && !messageContent);

    const message = isImage ? 'Sent a photo' : messageContent;

    displayText = `${sender}${message}`;
  }

  return {
    group,
    displayName,
    displayText,
    participants: participantsInConversation,
    lastActivity: lastMessage ? lastMessage.createdAt : null,
    hasOnlineInGroup,
  };
}
