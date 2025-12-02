import type { NextApiRequest, NextApiResponse } from 'next';

import cors from 'src/utils/cors';

/**
 * Proxy endpoint for Contour Detection
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

    // Get AI server URL from environment or use default
    const aiServerUrl = process.env.AI_SERVER_URL || 'http://localhost:5001';

    console.log('[Detect Contours] Proxying request to:', `${aiServerUrl}/detect-contours`);

    try {
      const pythonResponse = await fetch(`${aiServerUrl}/detect-contours`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(req.body),
        signal: AbortSignal.timeout(60000), // 60 seconds
      });

      if (!pythonResponse.ok) {
        let errorMessage = `Python server error: ${pythonResponse.status}`;
        try {
          const errorData = await pythonResponse.json();
          errorMessage = errorData.detail || errorData.error || errorData.message || errorMessage;
        } catch (e) {
          try {
            const errorText = await pythonResponse.text();
            if (errorText) errorMessage = errorText;
          } catch (e2) {
            // Ignore
          }
        }
        console.error('[Detect Contours] Python server error:', pythonResponse.status, errorMessage);
        return res.status(pythonResponse.status).json({
          success: false,
          error: errorMessage,
        });
      }

      const result = await pythonResponse.json();
      
      console.log('[Detect Contours] Success:', {
        success: result.success,
        contours_count: result.contours?.length || 0,
      });

      return res.status(200).json(result);

    } catch (fetchError: any) {
      console.error('[Detect Contours] Fetch error:', fetchError);
      
      // Check if it's a connection error
      if (fetchError.message.includes('fetch failed') || 
          fetchError.message.includes('ECONNREFUSED') ||
          fetchError.message.includes('timeout') ||
          fetchError.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'AI server is not available. Please ensure the Python AI server is running on port 5001.',
          message: `Failed to connect to ${aiServerUrl}. Make sure the unified_ai_api_server.py is running.`,
        });
      }
      
      throw fetchError;
    }

  } catch (error: any) {
    console.error('[Detect Contours] Error:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Internal server error',
    });
  }
}







