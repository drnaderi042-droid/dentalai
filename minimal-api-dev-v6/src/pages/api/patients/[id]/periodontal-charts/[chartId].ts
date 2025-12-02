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

    const { id: patientId, chartId } = req.query;

    if (!patientId || typeof patientId !== 'string') {
      return res.status(400).json({ message: 'Patient ID required' });
    }

    if (!chartId || typeof chartId !== 'string') {
      return res.status(400).json({ message: 'Chart ID required' });
    }

    // Verify patient access
    const patient = await prisma.patient.findUnique({
      where: { id: patientId },
    });

    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Check access rights
    if (user.role === 'DOCTOR' && patient.doctorId !== user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }

    // Verify chart belongs to patient
    const existingChart = await prisma.periodontalChart.findUnique({
      where: { id: chartId },
    });

    if (!existingChart || existingChart.patientId !== patientId) {
      return res.status(404).json({ message: 'Chart not found' });
    }

    if (req.method === 'GET') {
      return res.status(200).json({ chart: existingChart });
    }

    if (req.method === 'PUT') {
      // Update periodontal chart
      const { teeth, date, notes } = req.body;

      // Calculate analysis metrics if teeth data is provided
      let updateData: any = {
        notes: notes !== undefined ? notes : existingChart.notes,
        date: date ? new Date(date) : existingChart.date,
      };

      if (teeth) {
        const teethData = typeof teeth === 'string' ? JSON.parse(teeth) : teeth;
        const analysis = calculateAnalysis(teethData);

        updateData = {
          ...updateData,
          teeth: typeof teeth === 'string' ? teeth : JSON.stringify(teeth),
          bopPercentage: analysis.bopPercentage,
          avgPocketDepth: analysis.avgPocketDepth,
          avgCAL: analysis.avgCAL,
          diseaseExtent: analysis.diseaseExtent,
          diseaseSeverity: analysis.diseaseSeverity,
        };
      }

      const chart = await prisma.periodontalChart.update({
        where: { id: chartId },
        data: updateData,
      });

      return res.status(200).json({ chart });
    }

    if (req.method === 'DELETE') {
      // Delete periodontal chart
      await prisma.periodontalChart.delete({
        where: { id: chartId },
      });

      return res.status(200).json({ message: 'Chart deleted successfully' });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Periodontal Chart API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  } finally {
    await prisma.$disconnect();
  }
}

// ----------------------------------------------------------------------

function calculateAnalysis(teeth: any) {
  const results = {
    bopPercentage: 0,
    avgPocketDepth: 0,
    avgCAL: 0,
    diseaseExtent: 'Localized',
    diseaseSeverity: 'Stage I',
  };

  let totalPD = 0;
  let totalCAL = 0;
  let totalSites = 0;
  let bopCount = 0;
  let teethWithCAL = 0;
  let activeTeeth = 0;

  Object.keys(teeth).forEach((toothNum) => {
    const tooth = teeth[toothNum];

    // Skip missing teeth
    if (tooth.missing) return;

    activeTeeth += 1;
    let maxCAL = 0;

    // Process facial surface
    tooth.facial.pocketDepth.forEach((pd: number, i: number) => {
      const gm = tooth.facial.gingivalMargin[i];
      const cal = pd + gm;

      totalPD += pd;
      totalCAL += cal;
      totalSites += 1;

      maxCAL = Math.max(maxCAL, cal);

      // Count BOP
      if (tooth.facial.bleeding[i]) {
        bopCount += 1;
      }
    });

    // Process lingual surface
    tooth.lingual.pocketDepth.forEach((pd: number, i: number) => {
      const gm = tooth.lingual.gingivalMargin[i];
      const cal = pd + gm;

      totalPD += pd;
      totalCAL += cal;
      totalSites += 1;

      maxCAL = Math.max(maxCAL, cal);

      // Count BOP
      if (tooth.lingual.bleeding[i]) {
        bopCount += 1;
      }
    });

    // Count affected teeth (CAL >= 3mm)
    if (maxCAL >= 3) {
      teethWithCAL += 1;
    }
  });

  // Calculate percentages and averages
  if (totalSites > 0) {
    results.bopPercentage = (bopCount / totalSites) * 100;
    results.avgPocketDepth = totalPD / totalSites;
    results.avgCAL = totalCAL / totalSites;
  }

  // Disease extent
  if (activeTeeth > 0) {
    const affectedPercentage = (teethWithCAL / activeTeeth) * 100;
    results.diseaseExtent = affectedPercentage < 30 ? 'Localized' : 'Generalized';
  }

  // Disease severity (based on avg CAL)
  if (results.avgCAL <= 2) results.diseaseSeverity = 'Stage I - Mild';
  else if (results.avgCAL <= 4) results.diseaseSeverity = 'Stage II - Moderate';
  else if (results.avgCAL >= 5 && activeTeeth === 32)
    results.diseaseSeverity = 'Stage III - Severe';
  else results.diseaseSeverity = 'Stage IV - Very Severe';

  return results;
}



