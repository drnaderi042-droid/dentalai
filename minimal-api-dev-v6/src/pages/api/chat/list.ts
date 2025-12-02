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

    // Get blocked users
    const blockedUsers = await prisma.blockedUser.findMany({
      where: {
        OR: [
          { blockerId: data.userId },
          { blockedId: data.userId },
        ],
      },
    });

    const blockedIds = new Set([
      ...blockedUsers.map(b => b.blockerId),
      ...blockedUsers.map(b => b.blockedId),
    ]);
    blockedIds.delete(data.userId); // Remove self from blocked list

    // Get all chats where the current user is a participant
    const chatParticipations = await prisma.chatParticipant.findMany({
      where: {
        userId: data.userId,
      },
      include: {
        chat: {
          include: {
            messages: {
              orderBy: {
                createdAt: 'desc',
              },
              take: 1, // Just the latest message
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
                  not: data.userId, // Exclude current user
                },
              },
            },
          },
        },
      },
      orderBy: {
        chat: {
          updatedAt: 'desc',
        },
      },
    });

    // Format the response
    const formattedChats = chatParticipations
      .map((participation) => {
        const { chat } = participation;
        const otherParticipant = chat.participants[0]; // Should be only one other participant

        if (!otherParticipant) {
          return null;
        }

        // Filter out blocked users
        if (blockedIds.has(otherParticipant.userId)) {
          return null;
        }

        const lastMessage = chat.messages[0]; // We requested desc order, so first one is latest

      // For now, we'll skip unread count calculation due to complexity
      // In a production app, you might want to optimize this
      const unreadCount = 0; // Could be calculated separately

      return {
        id: chat.id,
        participants: chat.participants.map(p => p.userId),
        otherParticipant: {
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
        },
        lastMessage: lastMessage ? {
          id: lastMessage.id,
          content: lastMessage.content,
          contentType: lastMessage.contentType || 'text',
          attachments: lastMessage.attachments ? JSON.parse(lastMessage.attachments) : null,
          createdAt: lastMessage.createdAt,
          isOwn: lastMessage.senderId === data.userId,
        } : null,
        unreadCount,
        updatedAt: chat.updatedAt,
      };
    }).filter(chat => chat !== null);

    return res.status(200).json({
      chats: formattedChats,
      total: formattedChats.length,
    });
  } catch (error) {
    console.error('[Chat List API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
