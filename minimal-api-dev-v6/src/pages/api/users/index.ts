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
      return res.status(401).json({ message: 'Authorization token missing' });
    }

    const accessToken = `${authorization}`.split(' ')[1];
    const decodedToken = verify(accessToken, JWT_SECRET) as { userId: string };

    if (!decodedToken.userId) {
      return res.status(401).json({ message: 'Invalid authorization token' });
    }

    // Check if the requesting user is an admin
    const requestingUser = await prisma.user.findUnique({
      where: { id: decodedToken.userId },
      select: { role: true },
    });

    if (!requestingUser) {
      return res.status(401).json({ message: 'User not found' });
    }

    // Only ADMIN can access user list
    if (requestingUser.role.toUpperCase() !== 'ADMIN') {
      return res.status(403).json({ message: 'Forbidden: Admin access required' });
    }

    // Get all users from database
    const users = await prisma.user.findMany({
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        phone: true,
        role: true,
        specialty: true,
        licenseNumber: true,
        avatarUrl: true,
        isVerified: true,
        createdAt: true,
        updatedAt: true,
      },
      orderBy: {
        createdAt: 'desc',
      },
    });

    // Format users for response
    const formattedUsers = users.map((user) => ({
      id: user.id,
      email: user.email,
      firstName: user.firstName,
      lastName: user.lastName,
      phone: user.phone,
      phoneNumber: user.phone, // For backward compatibility
      role: user.role.toLowerCase(),
      specialty: user.specialty,
      licenseNumber: user.licenseNumber,
      avatarUrl: user.avatarUrl,
      avatar: user.avatarUrl, // For backward compatibility
      photoURL: user.avatarUrl, // For backward compatibility
      isVerified: user.isVerified,
      status: user.isVerified ? 'active' : 'pending', // Map isVerified to status
      createdAt: user.createdAt,
      updatedAt: user.updatedAt,
    }));

    res.status(200).json({ users: formattedUsers });
  } catch (error) {
    console.error('[Users API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}


