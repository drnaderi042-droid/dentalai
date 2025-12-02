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
 * Proxy endpoint for Intraoral Detection (Predict)
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
    console.log('[Predict] Request received:', {
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
      console.error('[Predict] Multer error:', multerError);
      return res.status(400).json({
        success: false,
        error: `File upload error: ${multerError.message || 'Unknown error'}`,
      });
    }

    // Get uploaded file
    const { file } = (req as any);
    
    console.log('[Predict] After multer:', {
      hasFile: !!file,
      file: file ? {
        fieldname: file.fieldname,
        originalname: file.originalname,
        encoding: file.encoding,
        mimetype: file.mimetype,
        size: file.size,
        bufferLength: file.buffer?.length || 0,
      } : null,
    });
    
    if (!file) {
      console.error('[Predict] No file found after multer processing');
      return res.status(400).json({ 
        success: false,
        error: 'No file uploaded. Please ensure the file is sent as multipart/form-data with field name "file".' 
      });
    }

    // Get AI server URL from environment or use default
    const aiServerUrl = process.env.AI_SERVER_URL || 'http://localhost:5001';
    const model = (req.query.model as string) || (req.body?.model as string) || 'fyp2';

    console.log('[Predict] Proxying request to:', `${aiServerUrl}/predict?model=${model}`);

    try {
      // Create FormData for Python server
      const formData = new FormData();
      
      if (!file.buffer || file.buffer.length === 0) {
        console.error('[Predict] File buffer is empty or missing');
        return res.status(400).json({
          success: false,
          error: 'File buffer is empty or missing',
        });
      }
      
      let filename = file.originalname;
      if (!filename || filename.trim() === '') {
        const extension = file.mimetype?.includes('jpeg') || file.mimetype?.includes('jpg') 
          ? 'jpg' 
          : file.mimetype?.includes('png') 
            ? 'png' 
            : 'jpg';
        filename = `image.${extension}`;
      }
      const contentType = file.mimetype || 'image/jpeg';
      
      formData.append('file', file.buffer, {
        filename,
        contentType,
        knownLength: file.buffer.length,
      });
      
      // Add model parameter if provided
      if (model) {
        formData.append('model', model);
      }

      // Forward the request to Python AI server
      const formDataHeaders = formData.getHeaders();
      
      const pythonResponse = await axios.post(
        `${aiServerUrl}/predict${model ? `?model=${model}` : ''}`,
        formData,
        {
          headers: formDataHeaders,
          timeout: 60000, // 60 seconds
          maxContentLength: Infinity,
          maxBodyLength: Infinity,
        }
      );

      console.log('[Predict] Success:', {
        success: pythonResponse.data?.success,
        detections: pythonResponse.data?.detections?.length || 0,
        model: pythonResponse.data?.model,
      });

      return res.status(200).json(pythonResponse.data);

    } catch (axiosError: any) {
      console.error('[Predict] Axios error:', axiosError);
      
      if (axiosError.response) {
        const {status} = axiosError.response;
        const errorData = axiosError.response.data;
        
        console.error('[Predict] Python server error:', status, errorData);
        
        return res.status(status).json({
          success: false,
          error: errorData?.error || errorData?.message || errorData?.detail || `Python server error: ${status}`,
        });
      } if (axiosError.request) {
        console.error('[Predict] No response from Python server:', axiosError.message);
        
        return res.status(503).json({
          success: false,
          error: 'AI server is not available. Please ensure the Python AI server is running on port 5001.',
          message: `Failed to connect to ${aiServerUrl}. Make sure the unified_ai_api_server.py is running.`,
        });
      } 
        console.error('[Predict] Request setup error:', axiosError.message);
        throw axiosError;
      
    }

  } catch (error: any) {
    console.error('[Predict] Error:', error);
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








