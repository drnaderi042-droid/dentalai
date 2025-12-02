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
  const { id } = req.query;

  if (typeof id !== 'string') {
    return res.status(400).json({ error: 'Invalid invoice ID' });
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
    if (req.method === 'GET') {
      // Get invoice by ID
      const invoice = await prisma.invoice.findUnique({
        where: { id },
      });

      if (!invoice) {
        return res.status(404).json({ error: 'Invoice not found' });
      }

      // Check if user owns this invoice
      if (invoice.userId !== userId) {
        return res.status(403).json({ error: 'Forbidden' });
      }

      return res.status(200).json({
        success: true,
        data: {
          ...invoice,
          items: JSON.parse(invoice.items),
        },
      });
    }

    if (req.method === 'PATCH') {
      // Update invoice
      const { status, paymentStatus, transactionId, paidAt } = req.body;

      const invoice = await prisma.invoice.findUnique({
        where: { id },
      });

      if (!invoice) {
        return res.status(404).json({ error: 'Invoice not found' });
      }

      if (invoice.userId !== userId) {
        return res.status(403).json({ error: 'Forbidden' });
      }

      const updatedInvoice = await prisma.invoice.update({
        where: { id },
        data: {
          ...(status && { status }),
          ...(paymentStatus && { paymentStatus }),
          ...(transactionId && { transactionId }),
          ...(paidAt && { paidAt: new Date(paidAt) }),
        },
      });

      return res.status(200).json({
        success: true,
        data: {
          ...updatedInvoice,
          items: JSON.parse(updatedInvoice.items),
        },
      });
    }

    return res.status(405).json({ error: 'Method not allowed' });
  } catch (error) {
    console.error('[Invoice API] Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to process invoice',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}




















