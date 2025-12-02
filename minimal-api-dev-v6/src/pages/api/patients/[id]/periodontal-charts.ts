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

    const { id: patientId } = req.query;

    if (!patientId || typeof patientId !== 'string') {
      return res.status(400).json({ message: 'Patient ID required' });
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

    if (req.method === 'GET') {
      // Get all periodontal charts for this patient
      const charts = await prisma.periodontalChart.findMany({
        where: { patientId },
        orderBy: { date: 'desc' },
      });

      return res.status(200).json({ charts });
    }

    if (req.method === 'POST') {
      // Create new periodontal chart
      const { teeth, date, notes } = req.body;

      if (!teeth) {
        return res.status(400).json({ message: 'Teeth data required' });
      }

      // Calculate analysis metrics
      const teethData = typeof teeth === 'string' ? JSON.parse(teeth) : teeth;
      const analysis = calculateAnalysis(teethData);

      const chart = await prisma.periodontalChart.create({
        data: {
          patientId,
          teeth: typeof teeth === 'string' ? teeth : JSON.stringify(teeth),
          date: date ? new Date(date) : new Date(),
          notes: notes || null,
          bopPercentage: analysis.bopPercentage,
          avgPocketDepth: analysis.avgPocketDepth,
          avgCAL: analysis.avgCAL,
          diseaseExtent: analysis.diseaseExtent,
          diseaseSeverity: analysis.diseaseSeverity,
        },
      });

      return res.status(201).json({ chart });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Periodontal Charts API]: ', error);
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



