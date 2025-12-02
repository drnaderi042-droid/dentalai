import type { NextApiRequest, NextApiResponse } from 'next';

import bcrypt from 'bcryptjs';
import { sign } from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

import cors from 'src/utils/cors';

const prisma = new PrismaClient();

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';
const JWT_EXPIRES_IN = '7d';

// ----------------------------------------------------------------------

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
    await cors(req, res);

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  try {
    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        message: 'Email and password are required',
      });
    }

    // Find user by email
    const user = await prisma.user.findUnique({
      where: { email },
    });

    if (!user) {
      return res.status(400).json({
        message: 'There is no user corresponding to the email address.',
      });
    }

    // Check password
    const isPasswordValid = await bcrypt.compare(password, user.password);
    if (!isPasswordValid) {
      return res.status(400).json({
        message: 'Wrong password',
      });
    }

    // Check if user is verified (for doctors)
    if (user.role === 'DOCTOR' && !user.isVerified) {
      return res.status(403).json({
        message: 'Your account is pending verification by an administrator.',
      });
    }

    // Generate JWT token
    const accessToken = sign({ userId: user.id }, JWT_SECRET, {
      expiresIn: JWT_EXPIRES_IN,
    });

    // Format user data for response
    const userResponse = {
      id: user.id,
      displayName: `${user.firstName} ${user.lastName}`,
      email: user.email,
      phoneNumber: user.phone,
      role: user.role.toLowerCase(),
      specialty: user.specialty,
      licenseNumber: user.licenseNumber,
      isVerified: user.isVerified,
    };

    res.status(200).json({
      accessToken,
      user: userResponse,
    });
  } catch (error) {
    console.error('[Auth API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
