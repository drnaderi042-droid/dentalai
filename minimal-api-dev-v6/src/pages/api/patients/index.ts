import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

import { prisma } from 'src/lib/prisma';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// ----------------------------------------------------------------------

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

    // Handle preflight request
    if (req.method === 'OPTIONS') {
      return res.status(200).end();
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

    if (req.method === 'GET') {
      // Get patients based on user role

      // Get pagination parameters
      const page = parseInt(req.query.page as string, 10) || 1;
      const limit = parseInt(req.query.limit as string, 10) || 20;
      const skip = (page - 1) * limit;

      if (user.role === 'DOCTOR') {
        // Doctors see only their patients
        // MEMORY OPTIMIZATION: Don't load images by default, only count
        const [patientList, total] = await Promise.all([
          prisma.patient.findMany({
            where: { doctorId: user.id },
            select: {
              id: true,
              firstName: true,
              lastName: true,
              phone: true,
              age: true,
              gender: true,
              diagnosis: true,
              treatment: true,
              status: true,
              notes: true,
              specialty: true,
              nextVisitTime: true,
              treatmentStartDate: true,
              createdAt: true,
              updatedAt: true,
              doctorId: true,
              // Only include image count, not full images
              _count: {
                select: {
                  radiologyImages: true,
                },
              },
            },
            orderBy: { createdAt: 'desc' },
            skip,
            take: limit,
          }),
          prisma.patient.count({
            where: { doctorId: user.id },
          }),
        ]);

        return res.status(200).json({
          patients: patientList,
          pagination: {
            page,
            limit,
            total,
            totalPages: Math.ceil(total / limit),
          },
        });
      }

      if (user.role === 'ADMIN') {
        // Admins see all patients
        // MEMORY OPTIMIZATION: Don't load images by default, only count
        const [patientList, total] = await Promise.all([
          prisma.patient.findMany({
            select: {
              id: true,
              firstName: true,
              lastName: true,
              phone: true,
              age: true,
              gender: true,
              diagnosis: true,
              treatment: true,
              status: true,
              notes: true,
              specialty: true,
              nextVisitTime: true,
              treatmentStartDate: true,
              createdAt: true,
              updatedAt: true,
              doctorId: true,
              // Only include image count, not full images
              _count: {
                select: {
                  radiologyImages: true,
                },
              },
              doctor: {
                select: {
                  firstName: true,
                  lastName: true,
                  specialty: true,
                },
              },
            },
            orderBy: { createdAt: 'desc' },
            skip,
            take: limit,
          }),
          prisma.patient.count(),
        ]);

        return res.status(200).json({
          patients: patientList,
          pagination: {
            page,
            limit,
            total,
            totalPages: Math.ceil(total / limit),
          },
        });
      }

      return res.status(403).json({ message: 'Access denied' });
    }

    if (req.method === 'POST') {
      // Only doctors can create patients
      if (user.role !== 'DOCTOR') {
        return res.status(403).json({ message: 'Only doctors can create patients' });
      }

      const { firstName, lastName, phone, age, diagnosis, treatment, status, notes, specialty, nextVisitTime, treatmentStartDate } = req.body;

      if (!firstName || !lastName || !phone || !age || !diagnosis || !treatment || !specialty) {
        return res.status(400).json({ message: 'Missing required fields' });
      }

      const patient = await prisma.patient.create({
        data: {
          firstName,
          lastName,
          phone,
          age: parseInt(age, 10),
          diagnosis,
          treatment,
          status: status || 'PENDING',
          notes,
          specialty,
          nextVisitTime: nextVisitTime ? new Date(nextVisitTime) : null,
          treatmentStartDate: treatmentStartDate ? new Date(treatmentStartDate) : null,
          doctorId: user.id,
        },
        // MEMORY OPTIMIZATION: Don't load images on create
        select: {
          id: true,
          firstName: true,
          lastName: true,
          phone: true,
          age: true,
          gender: true,
          diagnosis: true,
          treatment: true,
          status: true,
          notes: true,
          specialty: true,
          nextVisitTime: true,
          treatmentStartDate: true,
          createdAt: true,
          updatedAt: true,
          doctorId: true,
          _count: {
            select: {
              radiologyImages: true,
            },
          },
        },
      });

      return res.status(201).json({ patient });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Patients API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  }
  // MEMORY OPTIMIZATION: Don't disconnect prisma - use singleton connection pool
}
