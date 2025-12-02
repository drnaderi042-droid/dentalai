import type { NextApiRequest, NextApiResponse } from 'next';

import { PrismaClient } from '@prisma/client';

const prisma = new PrismaClient();

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Add CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, DELETE, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method === 'GET') {
    // دریافت تاریخچه تست‌ها
    try {
      const { userId, limit = '50', offset = '0' } = req.query;

      const where = userId && userId !== 'undefined' ? { userId: String(userId) } : {};

      const tests = await prisma.aIModelTest.findMany({
        where,
        orderBy: {
          createdAt: 'desc',
        },
        take: parseInt(String(limit), 10),
        skip: parseInt(String(offset), 10),
        select: {
          id: true,
          modelId: true,
          modelName: true,
          modelProvider: true,
          success: true,
          confidence: true,
          processingTime: true,
          createdAt: true,
        },
      });

      const total = await prisma.aIModelTest.count({ where });

      res.status(200).json({
        success: true,
        data: tests,
        pagination: {
          total,
          limit: parseInt(String(limit), 10),
          offset: parseInt(String(offset), 10),
        },
      });
    } catch (error: any) {
      console.error('Error fetching AI model tests:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch tests',
        message: error.message,
      });
    }
  } else if (req.method === 'POST') {
    // ذخیره تست جدید
    try {
      const {
        modelId,
        modelName,
        modelProvider,
        imageUrl,
        imageSize,
        prompt,
        success,
        landmarks,
        rawResponse,
        error,
        processingTime,
        tokensUsed,
        scalingInfo,
        confidence,
        userId,
      } = req.body;

      // Validation
      if (!modelId || !modelName || !modelProvider || typeof success !== 'boolean') {
        return res.status(400).json({
          success: false,
          error: 'Missing required fields: modelId, modelName, modelProvider, success',
        });
      }

      const test = await prisma.aIModelTest.create({
        data: {
          modelId,
          modelName,
          modelProvider,
          imageUrl: imageUrl || null,
          imageSize: imageSize ? JSON.stringify(imageSize) : null,
          prompt: prompt || '',
          success,
          landmarks: landmarks ? JSON.stringify(landmarks) : null,
          rawResponse: rawResponse || null,
          error: error || null,
          processingTime: processingTime || null,
          tokensUsed: tokensUsed ? JSON.stringify(tokensUsed) : null,
          scalingInfo: scalingInfo ? JSON.stringify(scalingInfo) : null,
          confidence: confidence || null,
          userId: userId || null,
        },
      });

      res.status(201).json({
        success: true,
        data: test,
      });
    } catch (error: any) {
      console.error('Error creating AI model test:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to create test',
        message: error.message,
      });
    }
  } else if (req.method === 'DELETE') {
    // حذف تمام تست‌ها (برای ذخیره فقط آخرین آنالیز)
    try {
      const deleted = await prisma.aIModelTest.deleteMany({});
      
      res.status(200).json({
        success: true,
        message: 'All tests deleted successfully',
        count: deleted.count,
      });
    } catch (error: any) {
      console.error('Error deleting all AI model tests:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to delete tests',
        message: error.message,
      });
    }
  } else {
    res.status(405).json({ success: false, error: 'Method not allowed' });
  }
}

