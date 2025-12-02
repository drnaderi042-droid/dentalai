import type { NextApiRequest, NextApiResponse } from 'next';

import jwt from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

// Generate unique invoice number
function generateInvoiceNumber(): string {
  const timestamp = Date.now().toString(36).toUpperCase();
  const random = Math.random().toString(36).substring(2, 7).toUpperCase();
  return `INV-${timestamp}-${random}`;
}

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
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
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

    // Check if user is admin
    const user = await prisma.user.findUnique({
      where: { id: userId },
      select: { role: true },
    });

    if (!user || user.role?.toUpperCase() !== 'ADMIN') {
      return res.status(403).json({ error: 'Only admins can create invoices' });
    }

    // Extract invoice data from request
    const {
      amount,
      type = 'wallet_charge',
      paymentGateway,
      description,
      items = [],
      patientId,
    } = req.body;

    if (!amount || amount <= 0) {
      return res.status(400).json({ error: 'Invalid amount' });
    }

    if (!paymentGateway) {
      return res.status(400).json({ error: 'Payment gateway is required' });
    }

    // Calculate fees (example: 1% for zarinpal, 0.5% for nowpayments)
    const feePercent = paymentGateway === 'zarinpal' ? 0.01 : 0.005;
    const feeAmount = Math.round(amount * feePercent);
    const totalAmount = amount + feeAmount;

    // Generate invoice number
    const invoiceNumber = generateInvoiceNumber();

    // Prepare items
    const invoiceItems = items.length > 0 ? items : [
      {
        description: type === 'wallet_charge' ? 'شارژ کیف پول' : 'پرداخت',
        quantity: 1,
        unitPrice: amount,
        totalPrice: amount,
      },
    ];

    // Add fee as an item
    if (feeAmount > 0) {
      invoiceItems.push({
        description: `کارمزد ${paymentGateway === 'zarinpal' ? 'زرین‌پال' : 'NowPayments'}`,
        quantity: 1,
        unitPrice: feeAmount,
        totalPrice: feeAmount,
      });
    }

    // Create invoice
    const invoice = await prisma.invoice.create({
      data: {
        invoiceNumber,
        status: 'pending',
        totalAmount,
        taxAmount: 0,
        discountAmount: 0,
        paymentGateway,
        paymentStatus: 'pending',
        userId,
        patientId: patientId || null,
        title: type === 'wallet_charge' ? 'شارژ کیف پول' : 'صورت‌حساب پرداخت',
        description: description || null,
        type,
        currency: 'IRR',
        items: JSON.stringify(invoiceItems),
        dueDate: new Date(Date.now() + 24 * 60 * 60 * 1000), // 24 hours
      },
    });

    console.log('[Invoice] Created:', invoice.invoiceNumber);

    return res.status(201).json({
      success: true,
      data: {
        id: invoice.id,
        invoiceNumber: invoice.invoiceNumber,
        totalAmount: invoice.totalAmount,
        status: invoice.status,
        paymentGateway: invoice.paymentGateway,
        items: JSON.parse(invoice.items),
      },
    });
  } catch (error) {
    console.error('[Invoice API] Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Failed to create invoice',
      message: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}



















