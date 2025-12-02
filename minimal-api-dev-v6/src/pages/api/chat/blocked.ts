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

    // Get all blocked users
    const blockedUsers = await prisma.blockedUser.findMany({
      where: {
        blockerId: data.userId,
      },
      include: {
        blocked: {
          select: {
            id: true,
            firstName: true,
            lastName: true,
            email: true,
          },
        },
      },
    });

    const blockedIds = blockedUsers.map(b => b.blockedId);

    res.status(200).json({ blockedIds });
  } catch (error) {
    console.error('[Chat Blocked API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}


