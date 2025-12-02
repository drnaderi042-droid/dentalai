import type { NextApiRequest, NextApiResponse } from 'next';

import fs from 'fs';
import path from 'path';
import multer from 'multer';
import { verify } from 'jsonwebtoken';
import { PrismaClient } from '@prisma/client';

import cors from 'src/utils/cors';

const prisma = new PrismaClient();
const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Configure multer for file upload
const upload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const uploadDir = path.join(process.cwd(), 'public', 'uploads', 'avatars');
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      cb(null, uploadDir);
    },
    filename: (req, file, cb) => {
      const uniqueSuffix = `${Date.now()}-${Math.round(Math.random() * 1e9)}`;
      cb(null, `avatar-${uniqueSuffix}${path.extname(file.originalname)}`);
    },
  }),
  limits: {
    fileSize: 3 * 1024 * 1024, // 3MB
  },
  fileFilter: (req, file, cb) => {
    const allowedTypes = /jpeg|jpg|png|gif/;
    const extname = allowedTypes.test(path.extname(file.originalname).toLowerCase());
    const mimetype = allowedTypes.test(file.mimetype);
    
    if (mimetype && extname) {
      return cb(null, true);
    }
    cb(new Error('Only image files are allowed!'));
  },
});

// Helper function to run multer middleware
const runMiddleware = (req: any, res: any, fn: any) => new Promise((resolve, reject) => {
    fn(req, res, (result: any) => {
      if (result instanceof Error) {
        return reject(result);
      }
      return resolve(result);
    });
  });

// ----------------------------------------------------------------------

// Disable body parsing for multipart/form-data (we'll handle it with multer)
export const config = {
  api: {
    bodyParser: false,
  },
};

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

    if (req.method === 'PUT') {
      let body: any = {};
      let avatarPath = null;

      // Check if request is multipart/form-data
      const contentType = req.headers['content-type'] || '';
      if (contentType.includes('multipart/form-data')) {
        // Run multer middleware to parse multipart/form-data
        try {
          await runMiddleware(req, res, upload.single('avatar'));
          
          // Get form fields from req.body (multer populates this)
          body = req.body || {};
          
          // Get file path if avatar was uploaded
          // Next.js serves files from /public directory, so path should be relative to public
          if ((req as any).file) {
            avatarPath = `/uploads/avatars/${(req as any).file.filename}`;
          }
        } catch (multerError: any) {
          console.error('[Profile API] Multer error:', multerError);
          return res.status(400).json({
            message: `File upload error: ${multerError.message || 'Unknown error'}`,
          });
        }
      } else {
        // Parse JSON body manually
        let rawBody = '';
        req.on('data', (chunk) => {
          rawBody += chunk.toString();
        });
        await new Promise((resolve) => {
          req.on('end', resolve);
        });
        try {
          body = JSON.parse(rawBody);
        } catch (parseError) {
          return res.status(400).json({ message: 'Invalid JSON body' });
        }
      }

      const {
        firstName,
        lastName,
        phone,
        licenseNumber,
        specialty,
        email,
        city,
        province,
      } = body;

      // Validate email uniqueness if email is being changed
      if (email && email !== user.email) {
        const existingUser = await prisma.user.findUnique({
          where: { email },
        });

        if (existingUser) {
          return res.status(400).json({ message: 'Email already exists' });
        }
      }

      // Delete old avatar if new one is uploaded
      if (avatarPath && user.avatarUrl) {
        // Remove leading slash if present
        const oldAvatarPath = user.avatarUrl.startsWith('/')
          ? path.join(process.cwd(), 'public', user.avatarUrl)
          : path.join(process.cwd(), 'public', user.avatarUrl);
        if (fs.existsSync(oldAvatarPath)) {
          try {
            fs.unlinkSync(oldAvatarPath);
          } catch (error) {
            console.error('[Profile API] Error deleting old avatar:', error);
          }
        }
      }

      const updatedUser = await prisma.user.update({
        where: { id: data.userId },
        data: {
          ...(firstName && { firstName }),
          ...(lastName && { lastName }),
          ...(phone !== undefined && phone !== null && { phone }),
          ...(email && { email }),
          ...(licenseNumber !== undefined && licenseNumber !== null && { licenseNumber }),
          ...(specialty !== undefined && specialty !== null && { specialty }),
          ...(city !== undefined && city !== null && city !== '' && { city }),
          ...(province !== undefined && province !== null && province !== '' && { province }),
          ...(avatarPath && { avatarUrl: avatarPath }),
        },
        select: {
          id: true,
          email: true,
          firstName: true,
          lastName: true,
          phone: true,
          role: true,
          specialty: true,
          licenseNumber: true,
          city: true,
          province: true,
          avatarUrl: true,
          isVerified: true,
          createdAt: true,
        },
      });

      return res.status(200).json({ user: updatedUser });
    }

    if (req.method === 'GET') {
      // Return user profile info (already handled by /auth/me)
      return res.status(200).json({ user });
    }

    return res.status(405).json({ message: 'Method not allowed' });
  } catch (error) {
    console.error('[Profile API]: ', error);
    res.status(500).json({
      message: 'Internal server error',
    });
  } finally {
    await prisma.$disconnect();
  }
}
