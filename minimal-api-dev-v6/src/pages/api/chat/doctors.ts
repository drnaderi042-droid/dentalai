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

    // Get all doctors and admins except the current user (case-insensitive filtering in app)
    const allUsers = await prisma.user.findMany({
      where: {
        // Get all users except current user
        id: { not: data.userId },
        // Filter will be applied client-side to handle case-insensitive role check
      },
      select: {
        id: true,
        firstName: true,
        lastName: true,
        email: true,
        specialty: true,
        phone: true,
        isVerified: true,
        role: true,
        avatarUrl: true,
      },
      orderBy: { createdAt: 'desc' },
    });

    // Filter doctors and admins client-side to handle case-insensitive role checking
    const doctors = allUsers.filter(user => {
      const role = user.role?.toUpperCase();
      return role === 'DOCTOR' || role === 'ADMIN';
    });

    // Format the response
    const formattedDoctors = doctors.map(doctor => ({
      id: doctor.id,
      name: `${doctor.firstName} ${doctor.lastName}`,
      firstName: doctor.firstName,
      lastName: doctor.lastName,
      email: doctor.email,
      specialty: doctor.specialty,
      phone: doctor.phone,
      isVerified: doctor.isVerified,
      avatarUrl: doctor.avatarUrl,
    }));

    return res.status(200).json({
      doctors: formattedDoctors,
      total: formattedDoctors.length,
    });
  } catch (error) {
    console.error('[Chat Doctors API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
