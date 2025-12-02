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

    // Only doctors and admins can report users
    const userRole = user.role?.toUpperCase();
    if (userRole !== 'DOCTOR' && userRole !== 'ADMIN') {
      return res.status(403).json({ message: 'Access denied. Only doctors and admins can report users.' });
    }

    const { reportedId, reason, description } = req.body;

    if (!reportedId) {
      return res.status(400).json({ message: 'Reported user ID is required' });
    }

    if (!reason) {
      return res.status(400).json({ message: 'Report reason is required' });
    }

    if (reportedId === data.userId) {
      return res.status(400).json({ message: 'You cannot report yourself' });
    }

    // Check if user exists
    const reportedUser = await prisma.user.findUnique({
      where: { id: reportedId },
    });

    if (!reportedUser) {
      return res.status(404).json({ message: 'User to report not found' });
    }

    // Validate reason
    const validReasons = ['spam', 'harassment', 'inappropriate_content', 'fake_account', 'other'];
    if (!validReasons.includes(reason)) {
      return res.status(400).json({ message: 'Invalid report reason' });
    }

    // If reason is "other", description is required
    if (reason === 'other' && !description) {
      return res.status(400).json({ message: 'Description is required when reason is "other"' });
    }

    // Create report
    await prisma.report.create({
      data: {
        reporterId: data.userId,
        reportedId,
        reason,
        description: reason === 'other' ? description : null,
        status: 'pending',
      },
    });

    res.status(201).json({ message: 'Report submitted successfully. Admin will review it.' });
  } catch (error) {
    console.error('[Chat Report API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}


