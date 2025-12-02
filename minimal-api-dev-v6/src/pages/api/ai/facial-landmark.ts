import type { NextApiRequest, NextApiResponse } from 'next';

import axios from 'axios';
import multer from 'multer';
import FormData from 'form-data';
import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

// Set up multer for file uploads (memory storage)
const upload = multer({
  storage: multer.memoryStorage(),
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

/**
 * Proxy endpoint for Facial Landmark Detection
 * This endpoint forwards requests to the Python AI server
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    // 🔧 FIX: CORS must be called first, before any other operations
    await cors(req, res);

    // Handle preflight request
    if (req.method === 'OPTIONS') {
      return res.status(200).end();
    }

    if (req.method !== 'POST') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    // Verify authentication (skip in development mode)
    const isDevelopment = process.env.NODE_ENV === 'development';
    
    if (!isDevelopment) {
      const { authorization } = req.headers;
      if (!authorization) {
        return res.status(401).json({ message: 'Authorization token missing' });
      }

      const accessToken = `${authorization}`.split(' ')[1];
      const data = verify(accessToken, JWT_SECRET) as { userId: string };

      if (!data.userId) {
        return res.status(401).json({ message: 'Invalid authorization token' });
      }
    }

    // Debug: Log request details before multer
    console.log('[Facial Landmark] Request received:', {
      method: req.method,
      contentType: req.headers['content-type'],
      contentLength: req.headers['content-length'],
      hasBody: !!req.body,
      bodyType: typeof req.body,
      query: req.query,
    });

    // Run multer middleware to parse multipart/form-data
    try {
      await runMiddleware(req, res, upload.single('file'));
    } catch (multerError: any) {
      console.error('[Facial Landmark] Multer error:', multerError);
      return res.status(400).json({
        success: false,
        error: `File upload error: ${multerError.message || 'Unknown error'}`,
      });
    }

    // Get uploaded file
    const { file } = (req as any);
    
    console.log('[Facial Landmark] After multer:', {
      hasFile: !!file,
      file: file ? {
        fieldname: file.fieldname,
        originalname: file.originalname,
        encoding: file.encoding,
        mimetype: file.mimetype,
        size: file.size,
        bufferLength: file.buffer?.length || 0,
      } : null,
      files: (req as any).files ? 'Files object exists' : 'No files object',
    });
    
    if (!file) {
      console.error('[Facial Landmark] No file found after multer processing');
      // Log request headers for debugging
      console.error('[Facial Landmark] Request headers:', {
        'content-type': req.headers['content-type'],
        'content-length': req.headers['content-length'],
        'content-encoding': req.headers['content-encoding'],
      });
      return res.status(400).json({ 
        success: false,
        error: 'No file uploaded. Please ensure the file is sent as multipart/form-data with field name "file".' 
      });
    }

    // Get AI server URL from environment or use default
    const aiServerUrl = process.env.AI_SERVER_URL || 'http://localhost:5001';
    const model = (req.query.model as string) || 'mediapipe';

    console.log('[Facial Landmark] Proxying request to:', `${aiServerUrl}/facial-landmark?model=${model}`);
    console.log('[Facial Landmark] File:', {
      originalName: file.originalname,
      mimetype: file.mimetype,
      size: file.size,
    });

    try {
      // Create FormData for Python server
      const formData = new FormData();
      
      // بررسی اینکه file.buffer وجود دارد
      if (!file.buffer || file.buffer.length === 0) {
        console.error('[Facial Landmark] File buffer is empty or missing');
        return res.status(400).json({
          success: false,
          error: 'File buffer is empty or missing',
        });
      }
      
      // اطمینان از اینکه filename معتبر است و خالی نیست
      let filename = file.originalname;
      if (!filename || filename.trim() === '') {
        // اگر filename خالی است، از mimetype برای تعیین extension استفاده کن
        const extension = file.mimetype?.includes('jpeg') || file.mimetype?.includes('jpg') 
          ? 'jpg' 
          : file.mimetype?.includes('png') 
            ? 'png' 
            : 'jpg';
        filename = `image.${extension}`;
      }
      const contentType = file.mimetype || 'image/jpeg';
      
      console.log('[Facial Landmark] File details before sending:', {
        originalname: file.originalname,
        filename,
        contentType,
        bufferSize: file.buffer.length,
        mimetype: file.mimetype,
      });
      
      // استفاده از Stream به جای buffer برای ارسال فایل
      formData.append('file', file.buffer, {
        filename,
        contentType,
        knownLength: file.buffer.length,
      });
      
      console.log('[Facial Landmark] Forwarding to Python server:', {
        filename,
        contentType,
        bufferSize: file.buffer.length,
        hasBuffer: !!file.buffer && file.buffer.length > 0,
      });

      // Forward the request to Python AI server using axios (better FormData support in Node.js)
      // Get headers from form-data (this includes Content-Type with boundary)
      const formDataHeaders = formData.getHeaders();
      
      // Use axios for better FormData support in Node.js
      const pythonResponse = await axios.post(
        `${aiServerUrl}/facial-landmark?model=${model}`,
        formData,
        {
          headers: formDataHeaders,
          timeout: 60000, // 60 seconds
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        }
      );

      console.log('[Facial Landmark] Success:', {
        success: pythonResponse.data?.success,
        total_landmarks: pythonResponse.data?.total_landmarks || 0,
        model: pythonResponse.data?.model,
      });

      return res.status(200).json(pythonResponse.data);

    } catch (axiosError: any) {
      console.error('[Facial Landmark] Axios error:', axiosError);
      
      // Handle axios errors
      if (axiosError.response) {
        // The request was made and the server responded with a status code
        // that falls out of the range of 2xx
        const {status} = axiosError.response;
        const errorData = axiosError.response.data;
        
        console.error('[Facial Landmark] Python server error:', status, errorData);
        
        return res.status(status).json({
          success: false,
          error: errorData?.error || errorData?.message || errorData?.detail || `Python server error: ${status}`,
        });
      } if (axiosError.request) {
        // The request was made but no response was received
        console.error('[Facial Landmark] No response from Python server:', axiosError.message);
        
        return res.status(503).json({
          success: false,
          error: 'AI server is not available. Please ensure the Python AI server is running on port 5001.',
          message: `Failed to connect to ${aiServerUrl}. Make sure the unified_ai_api_server.py is running.`,
        });
      } 
        // Something happened in setting up the request that triggered an Error
        console.error('[Facial Landmark] Request setup error:', axiosError.message);
        throw axiosError;
      
    }

  } catch (error: any) {
    console.error('[Facial Landmark] Error:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Internal server error',
    });
  }
}

// Disable body parsing for file uploads
export const config = {
  api: {
    bodyParser: false,
  },
};


