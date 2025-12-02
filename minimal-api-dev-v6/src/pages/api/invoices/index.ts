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

    if (req.method === 'GET') {
      // Get invoices based on user role
      let invoices;

      const userRole = user.role?.toUpperCase();

      if (userRole === 'DOCTOR') {
        // Doctors see invoices they created
        invoices = await prisma.invoice.findMany({
          where: { userId: user.id },
          orderBy: { createdAt: 'desc' },
        });
      } else if (userRole === 'ADMIN') {
        // Admins see all invoices
        invoices = await prisma.invoice.findMany({
          orderBy: { createdAt: 'desc' },
        });
      } else {
        // Patients see their own invoices (if patientId matches)
        invoices = await prisma.invoice.findMany({
          where: { patientId: user.id },
          orderBy: { createdAt: 'desc' },
        });
      }

      // Format invoices for response
      const formattedInvoices = invoices.map((invoice) => ({
        id: invoice.id,
        invoiceNumber: invoice.invoiceNumber,
        status: invoice.status,
        totalAmount: invoice.totalAmount,
        taxAmount: invoice.taxAmount,
        discountAmount: invoice.discountAmount,
        createDate: invoice.createDate,
        dueDate: invoice.dueDate,
        sentDate: invoice.sentDate,
        items: typeof invoice.items === 'string' ? JSON.parse(invoice.items) : invoice.items,
        title: invoice.title,
        description: invoice.description,
        type: invoice.type,
        currency: invoice.currency,
        paymentGateway: invoice.paymentGateway,
        paymentStatus: invoice.paymentStatus,
        transactionId: invoice.transactionId,
        paidAt: invoice.paidAt,
        createdAt: invoice.createdAt,
        updatedAt: invoice.updatedAt,
      }));

      return res.status(200).json({ invoices: formattedInvoices });
    }

    if (req.method === 'POST') {
      // Only admins can create invoices
      if (user.role?.toUpperCase() !== 'ADMIN') {
        return res.status(403).json({ message: 'Only admins can create invoices' });
      }

      const {
        patientId,
        title,
        description,
        items,
        taxAmount,
        discountAmount,
        dueDate,
      } = req.body;

      // Calculate total amount
      let totalAmount = 0;
      if (items && Array.isArray(items)) {
        totalAmount = items.reduce((sum, item) => sum + (item.quantity * (item.price || item.unitPrice || 0)), 0);
      }

      totalAmount = totalAmount + (taxAmount || 0) - (discountAmount || 0);

      // Generate invoice number
      const invoiceCount = await prisma.invoice.count({
        where: { userId: user.id },
      });
      const invoiceNumber = `INV-${user.id.slice(-4).toUpperCase()}-${String(invoiceCount + 1).padStart(4, '0')}`;

      const invoice = await prisma.invoice.create({
        data: {
          invoiceNumber,
          userId: user.id,
          patientId: patientId || null,
          title: title || null,
          description: description || null,
          items: JSON.stringify(items || []),
          taxAmount: taxAmount || 0,
          discountAmount: discountAmount || 0,
          totalAmount,
          dueDate: dueDate ? new Date(dueDate) : null,
          status: 'draft',
        },
      });

      return res.status(201).json({ 
        invoice: {
          ...invoice,
          items: typeof invoice.items === 'string' ? JSON.parse(invoice.items) : invoice.items,
        }
      });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Invoices API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
