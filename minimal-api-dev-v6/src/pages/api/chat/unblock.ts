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

    const { blockedId } = req.body;

    if (!blockedId) {
      return res.status(400).json({ message: 'Blocked user ID is required' });
    }

    // Remove block
    await prisma.blockedUser.deleteMany({
      where: {
        blockerId: data.userId,
        blockedId,
      },
    });

    res.status(200).json({ message: 'User unblocked successfully' });
  } catch (error) {
    console.error('[Chat Unblock API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}


