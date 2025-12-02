import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

import cors from 'src/utils/cors';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// ----------------------------------------------------------------------

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

    if (req.method !== 'GET') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    // Verify authentication
    const { authorization } = req.headers;
    if (!authorization) {
      return res.status(401).json({ message: 'Authorization token missing' });
    }

    const accessToken = `${authorization}`.split(' ')[1];
    const data = verify(accessToken, JWT_SECRET) as { userId: string };

    if (!data.userId) {
      return res.status(401).json({ message: 'Invalid authorization token' });
    }

    // Get user info
    const user = await prisma.user.findUnique({
      where: { id: data.userId },
    });

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    // Only doctors and admins can access this endpoint (case-insensitive check)
    const userRole = user.role?.toUpperCase();
    if (userRole !== 'DOCTOR' && userRole !== 'ADMIN') {
      return res.status(403).json({ message: 'Access denied. Only doctors and admins can access the chat feature.' });
    }

    const { chatId } = req.query;

    if (!chatId || typeof chatId !== 'string') {
      return res.status(400).json({ message: 'Valid chat ID is required' });
    }

    // Find the chat and verify the user is a participant
    const participantCheck = await prisma.chatParticipant.findFirst({
      where: {
        chatId,
        userId: data.userId,
      },
      include: {
        chat: {
          include: {
            messages: {
              include: {
                sender: {
                  select: {
                    id: true,
                    firstName: true,
                    lastName: true,
                    avatarUrl: true,
                  },
                },
              },
              orderBy: {
                createdAt: 'asc',
              },
            },
            participants: {
              include: {
                user: {
                  select: {
                    id: true,
                    firstName: true,
                    lastName: true,
                    email: true,
                    specialty: true,
                    phone: true,
                    isVerified: true,
                    avatarUrl: true,
                    updatedAt: true,
                  },
                },
              },
              where: {
                userId: {
                  not: data.userId,
                },
              },
            },
          },
        },
      },
    });

    if (!participantCheck || !participantCheck.chat) {
      return res.status(404).json({ message: 'Chat not found or access denied' });
    }

    const {chat} = participantCheck;
    const otherParticipant = chat.participants[0]; // Should be the other participant

    // Format messages
    const formattedMessages = chat.messages.map(message => ({
      id: message.id,
      chatId: message.chatId,
      content: message.content,
      contentType: message.contentType || 'text',
      attachments: message.attachments ? JSON.parse(message.attachments) : null,
      sender: {
        id: message.sender.id,
        name: `${message.sender.firstName} ${message.sender.lastName}`,
        avatarUrl: message.sender.avatarUrl,
      },
      isOwn: message.senderId === data.userId,
      read: message.read,
      readAt: message.readAt,
      createdAt: message.createdAt,
    }));

    // Mark messages from other participant as read
    if (otherParticipant) {
      await prisma.message.updateMany({
        where: {
          chatId,
          senderId: otherParticipant.userId,
          read: false,
        },
        data: {
          read: true,
          readAt: new Date(),
        },
      });
    }

    const chatData = {
      id: chat.id,
      participants: chat.participants.map(p => p.userId),
      otherParticipant: otherParticipant ? {
        id: otherParticipant.user.id,
        name: `${otherParticipant.user.firstName} ${otherParticipant.user.lastName}`,
        firstName: otherParticipant.user.firstName,
        lastName: otherParticipant.user.lastName,
        email: otherParticipant.user.email,
        specialty: otherParticipant.user.specialty,
        phone: otherParticipant.user.phone,
        isVerified: otherParticipant.user.isVerified,
        avatarUrl: otherParticipant.user.avatarUrl,
        lastActivity: otherParticipant.user.updatedAt,
      } : null,
      messages: formattedMessages,
      createdAt: chat.createdAt,
      updatedAt: chat.updatedAt,
    };

    return res.status(200).json({
      chat: chatData,
    });
  } catch (error) {
    console.error('[Chat Messages API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
