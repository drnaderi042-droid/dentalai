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
  try {
    await cors(req, res);

    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    const { email, password, firstName, lastName, phone, role, specialty, licenseNumber } = req.body;

    // Validate required fields
    if (!email || !password || !firstName || !lastName) {
      return res.status(400).json({
        message: 'Missing required fields',
      });
    }

    // Check if user already exists
    const existingUser = await prisma.user.findUnique({
      where: { email },
    });

    if (existingUser) {
      return res.status(400).json({
        message: 'There already exists an account with the given email address.',
      });
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(password, 12);

    // Create user
    const user = await prisma.user.create({
      data: {
        email,
        password: hashedPassword,
        firstName,
        lastName,
        phone,
        role: role || 'PATIENT',
        specialty: specialty || null,
        licenseNumber: role === 'DOCTOR' ? licenseNumber : null,
        isVerified: true, // Auto-verify all users for demo
      },
      select: {
        id: true,
        email: true,
        firstName: true,
        lastName: true,
        phone: true,
        role: true,
        specialty: true,
        licenseNumber: true,
        isVerified: true,
        createdAt: true,
      },
    });

    // Create wallet for the user
    await prisma.wallet.create({
      data: {
        userId: user.id,
      },
    });

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

    res.status(201).json({
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
