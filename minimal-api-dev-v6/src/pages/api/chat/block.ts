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
    const user = await prisma.user.findUnique({
      where: { id: data.userId },
    });

    if (!user) {
      return res.status(401).json({ message: 'User not found' });
    }

    // Only doctors and admins can block users
    const userRole = user.role?.toUpperCase();
    if (userRole !== 'DOCTOR' && userRole !== 'ADMIN') {
      return res.status(403).json({ message: 'Access denied. Only doctors and admins can block users.' });
    }

    const { blockedId } = req.body;

    if (!blockedId) {
      return res.status(400).json({ message: 'Blocked user ID is required' });
    }

    if (blockedId === data.userId) {
      return res.status(400).json({ message: 'You cannot block yourself' });
    }

    // Check if user exists
    const blockedUser = await prisma.user.findUnique({
      where: { id: blockedId },
    });

    if (!blockedUser) {
      return res.status(404).json({ message: 'User to block not found' });
    }

    // Check if already blocked
    const existingBlock = await prisma.blockedUser.findUnique({
      where: {
        blockerId_blockedId: {
          blockerId: data.userId,
          blockedId,
        },
      },
    });

    if (existingBlock) {
      return res.status(400).json({ message: 'User is already blocked' });
    }

    // Create block
    await prisma.blockedUser.create({
      data: {
        blockerId: data.userId,
        blockedId,
      },
    });

    res.status(200).json({ message: 'User blocked successfully' });
  } catch (error) {
    console.error('[Chat Block API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}


