import type { NextApiRequest, NextApiResponse } from 'next';

import jwt from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Verify JWT token
function verifyToken(token: string): any {
  try {
    return jwt.verify(token, process.env.JWT_SECRET || 'your-secret-key');
  } catch (error) {
    return null;
  }
}

export default async function handler(
  req: NextApiRequest,
  res: NextApiResponse
) {
  if (req.method !== 'GET') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  // Verify authentication
  const token = req.headers.authorization?.replace('Bearer ', '');
  if (!token) {
    return res.status(401).json({ error: 'Unauthorized' });
  }

  const decoded = verifyToken(token);
  if (!decoded) {
    return res.status(401).json({ error: 'Invalid token' });
  }

  const userId = decoded.userId || decoded.id;

  try {
    // Get query parameters
    const { status, type, limit = '50', offset = '0' } = req.query;

    const where: any = {
      userId,
    };

    if (status && typeof status === 'string') {
      where.status = status;
    }

    if (type && typeof type === 'string') {
      where.type = type;
    }

    // Get invoices
    const invoices = await prisma.invoice.findMany({
      where,
      orderBy: {
        createdAt: 'desc',
      },
      take: parseInt(limit as string, 10),
      skip: parseInt(offset as string, 10),
    });

    // Get total count
    const total = await prisma.invoice.count({ where });

    return res.status(200).json({
      success: true,
      data: {
        invoices: invoices.map((invoice) => ({
          ...invoice,
          items: JSON.parse(invoice.items),
        })),
        pagination: {
          total,
          limit: parseInt(limit as string, 10),
          offset: parseInt(offset as string, 10),
        },
      },
    });
  } catch (error) {
    console.error('[Invoice List API] Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to fetch invoices',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}




















