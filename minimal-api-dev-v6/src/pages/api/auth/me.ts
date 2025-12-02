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

    const { authorization } = req.headers;

    if (!authorization) {
      return res.status(401).json({
        message: 'Authorization token missing',
      });
    }

    const accessToken = `${authorization}`.split(' ')[1];

    const data = verify(accessToken, JWT_SECRET) as { userId: string };

    if (!data.userId) {
      return res.status(401).json({
        message: 'Invalid authorization token',
      });
    }

    // Get user from database
    const user = await prisma.user.findUnique({
      where: { id: data.userId },
          select: {
            id: true,
            email: true,
            firstName: true,
            lastName: true,
            phone: true,
            role: true,
            specialty: true,
            licenseNumber: true,
            city: true,
            province: true,
            avatarUrl: true,
            isVerified: true,
            createdAt: true,
          },
    });

    if (!user) {
      return res.status(401).json({
        message: 'User not found',
      });
    }

    // Format user data for response
    const userResponse = {
      id: user.id,
      displayName: `${user.firstName} ${user.lastName}`,
      email: user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      phone: user.phone,
      phoneNumber: user.phone, // Keep for backward compatibility
      role: user.role.toLowerCase(),
      specialty: user.specialty,
      licenseNumber: user.licenseNumber,
      city: user.city,
      province: user.province,
      avatarUrl: user.avatarUrl,
      avatar: user.avatarUrl, // Keep for backward compatibility
      photoURL: user.avatarUrl, // Keep for backward compatibility
      isVerified: user.isVerified,
      createdAt: user.createdAt,
    };

    res.status(200).json({ user: userResponse });
  } catch (error) {
    console.error('[Auth API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
