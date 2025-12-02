import type { NextApiRequest, NextApiResponse } from 'next';

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Add CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  const { id } = req.query;

  if (typeof id !== 'string') {
    return res.status(400).json({ success: false, error: 'Invalid ID' });
  }

  if (req.method === 'GET') {
    // دریافت جزئیات یک تست
    try {
      const test = await prisma.aIModelTest.findUnique({
        where: { id },
      });

      if (!test) {
        return res.status(404).json({
          success: false,
          error: 'Test not found',
        });
      }

      // Parse JSON fields
      const parsedTest = {
        ...test,
        imageSize: test.imageSize ? JSON.parse(test.imageSize) : null,
        landmarks: test.landmarks ? JSON.parse(test.landmarks) : null,
        tokensUsed: test.tokensUsed ? JSON.parse(test.tokensUsed) : null,
        scalingInfo: test.scalingInfo ? JSON.parse(test.scalingInfo) : null,
      };

      res.status(200).json({
        success: true,
        data: parsedTest,
      });
    } catch (error: any) {
      console.error('Error fetching AI model test:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch test',
        message: error.message,
      });
    }
  } else if (req.method === 'DELETE') {
    // حذف یک تست
    try {
      await prisma.aIModelTest.delete({
        where: { id },
      });

      res.status(200).json({
        success: true,
        message: 'Test deleted successfully',
      });
    } catch (error: any) {
      console.error('Error deleting AI model test:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to delete test',
        message: error.message,
      });
    }
  } else {
    res.status(405).json({ success: false, error: 'Method not allowed' });
  }
}

