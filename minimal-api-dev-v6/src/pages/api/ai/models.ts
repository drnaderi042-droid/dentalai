import type { NextApiRequest, NextApiResponse } from 'next';

import cors from 'src/utils/cors';

/**
 * Proxy endpoint for getting available AI models
 * This endpoint forwards requests to the Python AI server
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

    if (req.method !== 'GET') {
      return res.status(405).json({ message: 'Method not allowed' });
    }

    // Get AI server URL from environment or use default
    const aiServerUrl = process.env.AI_SERVER_URL || 'http://localhost:5001';

    console.log('[Models] Fetching models from:', `${aiServerUrl}/models`);

    try {
      const pythonResponse = await fetch(`${aiServerUrl}/models`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: AbortSignal.timeout(10000), // 10 seconds
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
        console.error('[Models] Python server error:', pythonResponse.status, errorMessage);
        return res.status(pythonResponse.status).json({
          success: false,
          error: errorMessage,
        });
      }

      const result = await pythonResponse.json();
      
      console.log('[Models] Success:', {
        models_count: result.models?.length || 0,
        available_models: result.facial_landmark?.available_models || [],
      });

      return res.status(200).json(result);

    } catch (fetchError: any) {
      console.error('[Models] Fetch error:', fetchError);
      
      // Check if it's a connection error
      if (fetchError.message.includes('fetch failed') || 
          fetchError.message.includes('ECONNREFUSED') ||
          fetchError.message.includes('timeout') ||
          fetchError.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'AI server is not available. Please ensure the Python AI server is running on port 5001.',
          message: `Failed to connect to ${aiServerUrl}. Make sure the unified_ai_api_server.py is running.`,
          // Return default models as fallback
          models: [
            {
              name: 'mediapipe',
              available: false,
              description: 'MediaPipe Face Mesh (468 points) - Server not available'
            }
          ],
          facial_landmark: {
            available_models: ['mediapipe'] // fallback
          }
        });
      }
      
      throw fetchError;
    }

  } catch (error: any) {
    console.error('[Models] Error:', error);
    return res.status(500).json({
      success: false,
      error: error.message || 'Internal server error',
      // Return default models as fallback
      models: [
        {
          name: 'mediapipe',
          available: false,
          description: 'MediaPipe Face Mesh (468 points) - Error occurred'
        }
      ],
      facial_landmark: {
        available_models: ['mediapipe'] // fallback
      }
    });
  }
}














