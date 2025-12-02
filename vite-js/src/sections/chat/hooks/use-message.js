// ----------------------------------------------------------------------

export function useMessage({ message, participants, currentUserId }) {
  // Use the isOwn field from message to determine if it's from current user
  const me = message.isOwn;

  // Get sender details from message.sender object
  const senderInfo = message.sender;

  // Get participant info for avatar
  const participant = participants?.find(p => p.id === message.senderId || String(p.id) === String(message.senderId));
  
  const senderDetails =
    me
      ? { type: 'me' }
      : { 
          avatarUrl: participant?.avatarUrl || null, 
          firstName: senderInfo?.name || participant?.firstName || 'Unknown' 
        };

  const hasImage = message.contentType === 'image';

  return { hasImage, me, senderDetails };
}
