import type { NextApiRequest, NextApiResponse } from 'next';

import path from 'path';
import multer from 'multer';
import { promises as fs } from 'fs';
import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

import { prisma } from 'src/lib/prisma';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Configure multer for file upload
// MEMORY OPTIMIZATION: Use disk storage instead of memory storage
// This prevents large files from consuming RAM
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const uploadsDir = path.join(process.cwd(), 'uploads', 'radiology');
    await fs.mkdir(uploadsDir, { recursive: true });
    cb(null, uploadsDir);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2);
    const extension = path.extname(file.originalname);
    cb(null, `${timestamp}-${random}${extension}`);
  },
});

const upload = multer({
  storage,
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    // Check if file is an image
    if (file.mimetype.startsWith('image/')) {
      cb(null, true);
    } else {
      cb(new Error('Only image files are allowed'));
    }
  },
});

// Helper function to run multer middleware
function runMiddleware(req: NextApiRequest, res: NextApiResponse, fn: Function) {
  return new Promise((resolve, reject) => {
    fn(req, res, (result: any) => {
      if (result instanceof Error) {
        return reject(result);
      }
      return resolve(result);
    });
  });
}

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

    const { id } = req.query;

    if (!id || typeof id !== 'string') {
      return res.status(400).json({ message: 'Invalid patient ID' });
    }

    // Check if patient exists and user has access
    const patient = await prisma.patient.findUnique({
      where: { id },
    });

    if (!patient) {
      return res.status(404).json({ message: 'Patient not found' });
    }

    // Check permissions
    if (user.role === 'DOCTOR' && patient.doctorId !== user.id) {
      return res.status(403).json({ message: 'Access denied' });
    }

    if (req.method === 'POST') {
      // Handle file upload
      try {
        await runMiddleware(req, res, upload.array('images', 10));

        const {files} = (req as any);
        console.log('[Patient Images API] Received files:', files ? files.length : 0);
        if (!files || files.length === 0) {
          console.warn('[Patient Images API] No files found in request. req.body keys:', Object.keys(req.body || {}));
          return res.status(400).json({ message: 'No files uploaded' });
        }

        // Get category from form data
        const category = req.body.category || 'general';
        const type = req.body.type || null;

        const uploadedImages = [];

        // MEMORY OPTIMIZATION: Files are already saved to disk by multer
        // Just read the filename and save metadata to database
        for (const file of files) {
          const {filename} = file;
          const filePath = file.path;

          // Verify file exists and get stats
          let fileSize = file.size;
          try {
            const stats = await fs.stat(filePath);
            fileSize = stats.size;
          } catch (statError) {
            console.warn('[Patient Images API] Could not get file stats:', statError);
          }

          // Save to database
          const image = await prisma.radiologyImage.create({
            data: {
              filename,
              originalName: file.originalname,
              mimeType: file.mimetype,
              size: fileSize,
              path: `/uploads/radiology/${filename}`,
              category: category || 'general',
              type: type || null,
              patientId: id,
            },
          });

          uploadedImages.push(image);
        }

        return res.status(201).json({
          message: 'Images uploaded successfully',
          images: uploadedImages,
        });
      } catch (uploadError) {
        // Enhanced debug logging for multer/FS errors
        try {
          const {files} = (req as any);
          console.error('[Patient Images API] Upload error:', uploadError && uploadError.message ? uploadError.message : uploadError);
          console.error('[Patient Images API] Upload error code:', uploadError && uploadError.code ? uploadError.code : 'N/A');
          console.error('[Patient Images API] Request headers:', req.headers);
          console.error('[Patient Images API] Request body keys:', Object.keys(req.body || {}));
          console.error('[Patient Images API] Files present:', files ? files.map((f: any) => ({ originalname: f.originalname, mimetype: f.mimetype, size: f.size })) : 'No files');
        } catch (logErr) {
          console.error('[Patient Images API] Error while logging upload error details:', logErr);
        }
        return res.status(400).json({ message: 'File upload failed' });
      }
    }

    if (req.method === 'GET') {
      // Get patient's images with pagination
      // MEMORY OPTIMIZATION: Add pagination to prevent loading all images at once
      const page = parseInt(req.query.page as string) || 1;
      const limit = parseInt(req.query.limit as string) || 20;
      const skip = (page - 1) * limit;

      const [images, total] = await Promise.all([
        prisma.radiologyImage.findMany({
          where: { patientId: id },
          select: {
            id: true,
            filename: true,
            originalName: true,
            mimeType: true,
            size: true,
            path: true,
            category: true,
            type: true,
            uploadedAt: true,
            patientId: true,
            // Don't load file content, only metadata
          },
          orderBy: { uploadedAt: 'desc' },
          skip,
          take: limit,
        }),
        prisma.radiologyImage.count({
          where: { patientId: id },
        }),
      ]);

      return res.status(200).json({
        images,
        pagination: {
          page,
          limit,
          total,
          totalPages: Math.ceil(total / limit),
        },
      });
    }

    if (req.method === 'DELETE') {
      // Parse JSON body for DELETE request
      let body;
      try {
        const chunks = [];
        for await (const chunk of req) {
          chunks.push(chunk);
        }
        const buffer = Buffer.concat(chunks);
        const bodyString = buffer.toString();
        body = bodyString ? JSON.parse(bodyString) : {};
      } catch (parseError) {
        console.error('Error parsing DELETE request body:', parseError);
        return res.status(400).json({ message: 'Invalid request body' });
      }

      const { imageId } = body;

      if (!imageId) {
        return res.status(400).json({ message: 'Image ID is required' });
      }

      // Find and delete the image
      const image = await prisma.radiologyImage.findUnique({
        where: { id: imageId },
      });

      if (!image) {
        return res.status(404).json({ message: 'Image not found' });
      }

      if (image.patientId !== id) {
        return res.status(403).json({ message: 'Access denied' });
      }

      // Delete file from disk
      try {
        const filePath = path.join(process.cwd(), 'uploads', 'radiology', image.filename);
        await fs.unlink(filePath);
      } catch (fileError) {
        console.warn('Could not delete file from disk:', fileError);
      }

      // Delete from database
      await prisma.radiologyImage.delete({
        where: { id: imageId },
      });

      return res.status(200).json({ message: 'Image deleted successfully' });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Patient Images API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  }
  // MEMORY OPTIMIZATION: Don't disconnect prisma - use singleton connection pool
}

// Disable Next.js body parsing for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};
