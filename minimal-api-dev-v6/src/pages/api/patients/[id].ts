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

    const { id } = req.query;

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    // Check if patient exists and user has access
    // MEMORY OPTIMIZATION: Don't load images by default, only count
    const patient = await prisma.patient.findUnique({
      where: { id },
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
        softTissueAnalysis: true,
        cephalometricAnalysis: true,
        cephalometricTableData: true,
        cephalometricRawData: true,
        cephalometricLandmarks: true,
        treatmentPlan: true,
        summary: true,
        medicalHistory: true,
        intraOralAnalysis: true,
        facialLandmarkAnalysis: true,
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
    });

    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Check permissions
    if (user.role === 'DOCTOR' && patient.doctorId !== user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }

    if (req.method === 'GET') {
      return res.status(200).json({ patient });
    }

    if (req.method === 'PUT') {
      // Only doctors can update their own patients, admins can update any
      if (user.role === 'DOCTOR' && patient.doctorId !== user.id) {
        return res.status(403).json({ message: 'Access denied' });
      }

      const { firstName, lastName, phone, age, gender, diagnosis, treatment, status, notes, softTissueAnalysis, cephalometricAnalysis, cephalometricTableData, cephalometricRawData, cephalometricLandmarks, treatmentPlan, summary, nextVisitTime, treatmentStartDate, medicalHistory, intraOralAnalysis, facialLandmarkAnalysis } = req.body;

      // Debug logging for analysis data
      if (intraOralAnalysis) {
        console.log('🔍 [Backend] Received intraOralAnalysis:', {
          type: typeof intraOralAnalysis,
          length: intraOralAnalysis.length,
          isString: typeof intraOralAnalysis === 'string'
        });
      }

      try {
        const updatedPatient = await prisma.patient.update({
          where: { id },
          data: {
          ...(firstName && { firstName }),
          ...(lastName && { lastName }),
          ...(phone && { phone }),
          ...(age && { age: parseInt(age, 10) }),
          ...(gender !== undefined && { gender }),
          ...(diagnosis && { diagnosis }),
          ...(treatment && { treatment }),
          ...(status && { status }),
          ...(notes !== undefined && { notes }),
          ...(nextVisitTime !== undefined && { nextVisitTime: nextVisitTime ? new Date(nextVisitTime) : null }),
          ...(treatmentStartDate !== undefined && { treatmentStartDate: treatmentStartDate ? new Date(treatmentStartDate) : null }),
          ...(softTissueAnalysis && { softTissueAnalysis }),
          ...(cephalometricAnalysis && { cephalometricAnalysis }),
          ...(cephalometricTableData !== undefined && { cephalometricTableData }),
          ...(cephalometricRawData !== undefined && { cephalometricRawData }),
          ...(cephalometricLandmarks !== undefined && { cephalometricLandmarks }),
          ...(treatmentPlan && { treatmentPlan }),
          ...(summary && { summary }),
          ...(medicalHistory !== undefined && { medicalHistory: typeof medicalHistory === 'string' ? medicalHistory : JSON.stringify(medicalHistory) }),
          ...(intraOralAnalysis !== undefined && { intraOralAnalysis }),
          ...(facialLandmarkAnalysis !== undefined && { facialLandmarkAnalysis }),
        },
        // MEMORY OPTIMIZATION: Don't load images on update
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
          softTissueAnalysis: true,
          cephalometricAnalysis: true,
          cephalometricTableData: true,
          cephalometricRawData: true,
          cephalometricLandmarks: true,
          treatmentPlan: true,
          summary: true,
          medicalHistory: true,
          intraOralAnalysis: true,
          facialLandmarkAnalysis: true,
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

      return res.status(200).json({ patient: updatedPatient });
      } catch (updateError) {
        console.error('❌ [Backend] Failed to update patient:', updateError);
        console.error('❌ [Backend] Error details:', {
          message: updateError.message,
          code: updateError.code,
          meta: updateError.meta
        });
        return res.status(500).json({
          error: 'Failed to update patient',
          details: updateError.message
        });
      }
    }

    if (req.method === 'DELETE') {
      // Only doctors can delete their own patients, admins can delete any
      if (user.role === 'DOCTOR' && patient.doctorId !== user.id) {
        return res.status(403).json({ message: 'Access denied' });
      }

      await prisma.patient.delete({
        where: { id },
      });

      return res.status(200).json({ message: 'Patient deleted successfully' });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Patient API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  }
  // MEMORY OPTIMIZATION: Don't disconnect prisma - use singleton connection pool
}
