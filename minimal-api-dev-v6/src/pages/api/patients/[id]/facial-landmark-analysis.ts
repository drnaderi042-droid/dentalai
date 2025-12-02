import type { NextApiRequest, NextApiResponse } from 'next';

import { PrismaClient } from '@prisma/client';

import cors from 'src/utils/cors';

const prisma = new PrismaClient();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Apply CORS middleware
  await cors(req, res);

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  const { id } = req.query;

  if (!id || typeof id !== 'string') {
    return res.status(400).json({ error: 'Patient ID is required' });
  }

  try {
    switch (req.method) {
      case 'GET': {
        // Get the facial landmark analysis for the patient
        const patient = await prisma.patient.findUnique({
          where: { id },
          select: {
            id: true,
            facialLandmarkAnalysis: true,
            updatedAt: true,
          },
        });

        if (!patient) {
          return res.status(404).json({ error: 'Patient not found' });
        }

        // Parse the facial landmark analysis data
        let analysis = null;
        if (patient.facialLandmarkAnalysis) {
          try {
            analysis = JSON.parse(patient.facialLandmarkAnalysis);
          } catch (parseError) {
            console.error('Error parsing facial landmark analysis data:', parseError);
            analysis = null;
          }
        }

        return res.status(200).json({
          analysis,
          lastUpdated: patient.updatedAt,
        });
      }

      case 'POST': {
        // Save facial landmark analysis data
        const { analyses } = req.body;

        if (!analyses || !Array.isArray(analyses)) {
          return res.status(400).json({ error: 'Analysis data must be an array' });
        }

        // Validate the analysis data structure
        for (const analysis of analyses) {
          if (!analysis.result) {
            return res.status(400).json({
              error: 'Each analysis must have result data'
            });
          }
        }

        // Structure the data for storage
        const analysisData = {
          analyses,
          totalAnalyses: analyses.length,
          lastUpdated: new Date().toISOString(),
        };

        // Update the patient record
        const updatedPatient = await prisma.patient.update({
          where: { id },
          data: {
            facialLandmarkAnalysis: JSON.stringify(analysisData),
            updatedAt: new Date(),
          },
          select: {
            id: true,
            facialLandmarkAnalysis: true,
            updatedAt: true,
          },
        });

        return res.status(200).json({
          success: true,
          message: 'Facial landmark analysis saved successfully',
          analysis: analysisData,
          lastUpdated: updatedPatient.updatedAt,
        });
      }

      default:
        res.setHeader('Allow', ['GET', 'POST']);
        return res.status(405).json({ error: `Method ${req.method} not allowed` });
    }
  } catch (error) {
    console.error('Facial landmark analysis API error:', error);
    return res.status(500).json({
      error: 'Internal server error',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  } finally {
    await prisma.$disconnect();
  }
}
