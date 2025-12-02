import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

/**
 * Proxy endpoint for Cephalometric Detection
 * This endpoint forwards requests to the Cephalometric Detection API
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

    const { image_url, image_base64, options } = req.body;

    if (!image_url && !image_base64) {
      return res.status(400).json({ 
        success: false,
        message: 'Image URL or base64 image is required' 
      });
    }

    // Get service URL from environment or use default
    const serviceUrl = process.env.CEPHX_SERVICE_URL || 'http://localhost:5000';

    console.log('[Detect] Proxying request to:', `${serviceUrl}/detect`);
    console.log('[Detect] Has image_url:', !!image_url);
    console.log('[Detect] Has image_base64:', !!image_base64);

    try {
      const requestBody: any = {
        options: options || {
          enhance: true,
          use_subpixel: true,
          threshold: 0.1
        }
      };

      // Use image_url if provided, otherwise use image_base64
      if (image_url) {
        requestBody.image_url = image_url;
      } else if (image_base64) {
        requestBody.image_base64 = image_base64;
      }

      const response = await fetch(`${serviceUrl}/detect`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody),
        signal: AbortSignal.timeout(60000), // 60 seconds
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('[Detect] Error response:', errorText);
        return res.status(response.status).json({
          success: false,
          error: `Python service error: ${response.status} - ${errorText}`,
        });
      }

      const result = await response.json();
      
      console.log('[Detect] Success! Response:', {
        success: result.success,
        num_landmarks: Object.keys(result.landmarks || {}).length,
        num_measurements: Object.keys(result.measurements || {}).length,
        method: result.metadata?.model
      });

      return res.status(200).json(result);
      
    } catch (error: any) {
      console.error('[Detect] Detection failed:', error.message);
      
      // Check if it's a connection error
      if (error.message.includes('fetch failed') || error.message.includes('ECONNREFUSED')) {
        return res.status(503).json({
          success: false,
          error: 'Python service is not running. Please start it with: python unified_ai_api_server.py',
          message: `Failed to connect to ${serviceUrl}. Make sure the unified_ai_api_server.py is running.`,
        });
      }
      
      return res.status(500).json({
        success: false,
        error: error.message || 'Internal server error',
      });
    }

  } catch (error: any) {
    console.error('[Detect] Error:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Internal server error',
    });
  }
}








