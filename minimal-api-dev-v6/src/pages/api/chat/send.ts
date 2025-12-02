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

    if (req.method !== 'POST') {
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
    const sender = await prisma.user.findUnique({
      where: { id: data.userId },
    });

    if (!sender) {
      return res.status(401).json({ message: 'User not found' });
    }

    // Only doctors and admins can send messages (case-insensitive check)
    const senderRole = sender.role?.toUpperCase();
    if (senderRole !== 'DOCTOR' && senderRole !== 'ADMIN') {
      return res.status(403).json({ message: 'Access denied. Only doctors and admins can send messages.' });
    }

    const { recipientId, content, contentType, attachments } = req.body;

    // Validate request body
    if (!recipientId) {
      return res.status(400).json({ message: 'Recipient ID is required' });
    }

    // Content must be provided unless there are attachments
    if ((!content || content.trim().length === 0) && (!attachments || attachments.length === 0)) {
      return res.status(400).json({ message: 'Message content or attachments are required' });
    }

    // Verify recipient exists and is a doctor or admin (case-insensitive check)
    const recipient = await prisma.user.findUnique({
      where: { id: recipientId },
    });

    const recipientRole = recipient?.role?.toUpperCase();
    if (!recipient || (recipientRole !== 'DOCTOR' && recipientRole !== 'ADMIN')) {
      return res.status(404).json({ message: 'Recipient not found or is not a doctor or admin' });
    }

    // Check if recipient is not the sender
    if (recipientId === data.userId) {
      return res.status(400).json({ message: 'You cannot send messages to yourself' });
    }

    // Check if user is blocked
    const isBlocked = await prisma.blockedUser.findFirst({
      where: {
        OR: [
          { blockerId: data.userId, blockedId: recipientId },
          { blockerId: recipientId, blockedId: data.userId },
        ],
      },
    });

    if (isBlocked) {
      return res.status(403).json({ message: 'You cannot send messages to this user. One of you has blocked the other.' });
    }

    // Find existing chat between these two users (doctors/admins)
    let chat = await prisma.chat.findFirst({
      where: {
        AND: [
          {
            participants: {
              some: { userId: data.userId }
            }
          },
          {
            participants: {
              some: { userId: recipientId }
            }
          }
        ]
      }
    });

    // If chat doesn't exist, create it with participants
    if (!chat) {
      chat = await prisma.chat.create({
        data: {
          participants: {
            create: [
              { userId: data.userId },
              { userId: recipientId }
            ]
          },
        },
      });
    }

    // Create the message
    const message = await prisma.message.create({
      data: {
        chatId: chat.id,
        senderId: data.userId,
        content: content?.trim() || '',
        contentType: contentType || 'text',
        attachments: attachments ? JSON.stringify(attachments) : null,
      },
      include: {
        sender: {
          select: {
            id: true,
            firstName: true,
            lastName: true,
          },
        },
      },
    });

    // Update chat updatedAt timestamp
    await prisma.chat.update({
      where: { id: chat.id },
      data: { updatedAt: new Date() },
    });

    // Return the message with sender info
    const messageResponse = {
      id: message.id,
      chatId: message.chatId,
      content: message.content,
      sender: {
        id: message.sender.id,
        name: `${message.sender.firstName} ${message.sender.lastName}`,
      },
      isOwn: true,
      read: message.read,
      createdAt: message.createdAt,
    };

    return res.status(201).json({
      message: messageResponse,
      chatId: chat.id,
    });
  } catch (error) {
    console.error('[Chat Send API] Detailed Error:', error);
    console.error('[Chat Send API] Error name:', error?.name);
    console.error('[Chat Send API] Error message:', error?.message);
    console.error('[Chat Send API] Error stack:', error?.stack);
    if (error instanceof Error) {
      console.error('[Chat Send API] Full error object:', error);
    }
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
