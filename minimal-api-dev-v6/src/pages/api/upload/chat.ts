import type { NextApiRequest, NextApiResponse } from 'next';

import fs from 'fs';
import path from 'path';
import multer from 'multer';
import { v4 as uuidv4 } from 'uuid';

import cors from 'src/utils/cors';

// Set up multer for file uploads
const upload = multer({
  storage: multer.diskStorage({
    destination (req, file, cb) {
      // Create uploads/chat directory if it doesn't exist
      const uploadDir = path.join(process.cwd(), 'uploads', 'chat');
      if (!fs.existsSync(uploadDir)) {
        fs.mkdirSync(uploadDir, { recursive: true });
      }
      cb(null, uploadDir);
    },
    filename (req, file, cb) {
      // Generate unique filename
      const uniqueId = uuidv4();
      const extension = path.extname(file.originalname);
      const filename = `${uniqueId}${extension}`;
      cb(null, filename);
    },
  }),
  fileFilter: (req, file, cb) => {
    // Check file type
    const allowedTypes = [
      'image/jpeg',
      'image/png',
      'image/gif',
      'image/webp',
      'application/pdf',
      'application/msword',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'text/plain',
      'application/zip',
      'application/x-zip-compressed',
    ];

    if (allowedTypes.includes(file.mimetype)) {
      cb(null, true);
    } else {
      cb(null, false);
    }
  },
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
});

// Middleware to handle multipart/form-data
const runMiddleware = (req: NextApiRequest, res: NextApiResponse, fn: Function) => new Promise((resolve, reject) => {
    fn(req, res, (result: any) => {
      if (result instanceof Error) {
        return reject(result);
      }
      return resolve(result);
    });
  });

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  // Apply CORS middleware
  await cors(req as any, res as any);

  // Allow preflight
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  try {
    // Run multer middleware
    await runMiddleware(req, res, upload.single('file'));

    // Access uploaded file
    const {file} = (req as any);
    if (!file) {
      return res.status(400).json({ message: 'No file uploaded' });
    }

    // Create file URL for frontend
    const fileUrl = `/serve-upload?path=chat/${file.filename}`;

    // Return success response
    res.status(200).json({
      message: 'File uploaded successfully',
      url: fileUrl,
      filename: file.filename,
      originalName: file.originalname,
      size: file.size,
      type: file.mimetype,
    });
  } catch (error) {
    console.error('File upload error:', error);
    res.status(500).json({
      message: 'File upload failed',
      error: error instanceof Error ? error.message : 'Unknown error'
    });
  }
}

// Disable body parsing for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};
