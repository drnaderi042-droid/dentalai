import type { NextApiRequest, NextApiResponse } from 'next';

import * as fs from 'fs';
import * as path from 'path';
import { verify } from 'jsonwebtoken';
import { spawn } from 'child_process';

import cors from 'src/utils/cors';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

/**
 * API endpoint for P1/P2 calibration point detection using trained heatmap model
 * 
 * This endpoint uses the trained HRNet heatmap model to detect p1 and p2
 * calibration points (1cm apart on ruler) with high accuracy (< 10px error)
 */
export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

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

    const { imageUrl, imageBase64 } = req.body;

    if (!imageUrl && !imageBase64) {
      return res.status(400).json({ message: 'Image URL or base64 is required' });
    }

    console.log('[P1P2 Detect] Processing calibration point detection request');

    // Path to Python inference script
    const projectRoot = process.cwd();
    const scriptPath = path.join(projectRoot, 'aariz', 'infer_p1p2_heatmap.py');
    const modelPath = path.join(projectRoot, 'aariz', 'models', 'hrnet_p1p2_heatmap_best.pth');

    // Check if model exists
    if (!fs.existsSync(modelPath)) {
      console.error('[P1P2 Detect] Model not found:', modelPath);
      return res.status(500).json({ 
        message: 'P1/P2 detection model not found. Please train the model first.',
        error: 'MODEL_NOT_FOUND'
      });
    }

    // Prepare image data
    let imageData: string;
    if (imageBase64) {
      // Remove data URL prefix if present
      imageData = imageBase64.replace(/^data:image\/[a-z]+;base64,/, '');
    } else if (imageUrl) {
      // For URL, we'll need to download it in Python script
      imageData = imageUrl;
    } else {
      return res.status(400).json({ message: 'Invalid image data' });
    }

    // Call Python inference script
    const startTime = Date.now();
    
    try {
      const result = await runPythonInference(scriptPath, imageData, modelPath);
      
      const processingTime = Date.now() - startTime;
      
      console.log('[P1P2 Detect] ✅ Detection successful');
      console.log('[P1P2 Detect] Processing time:', processingTime, 'ms');
      console.log('[P1P2 Detect] Detected p1:', result.p1);
      console.log('[P1P2 Detect] Detected p2:', result.p2);

      return res.status(200).json({
        success: true,
        p1: result.p1,
        p2: result.p2,
        confidence: result.confidence || 0.95,
        processingTime,
        method: 'heatmap-hrnet',
      });
    } catch (error: any) {
      console.error('[P1P2 Detect] ❌ Detection failed:', error.message);
      return res.status(500).json({
        message: 'P1/P2 detection failed',
        error: error.message,
      });
    }
  } catch (error: any) {
    console.error('[P1P2 Detect] Error:', error);
    return res.status(500).json({
      message: 'Internal server error',
      error: error.message,
    });
  }
}

/**
 * Run Python inference script
 */
function runPythonInference(scriptPath: string, imageData: string, modelPath: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const python = spawn('python', [scriptPath, '--image', imageData, '--model', modelPath]);
    
    let stdout = '';
    let stderr = '';

    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    python.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(`Python script failed: ${stderr || 'Unknown error'}`));
        return;
      }

      try {
        // Parse JSON output
        const lines = stdout.trim().split('\n');
        const jsonLine = lines.find(line => line.startsWith('{'));
        
        if (!jsonLine) {
          reject(new Error('No JSON output from Python script'));
          return;
        }

        const result = JSON.parse(jsonLine);
        resolve(result);
      } catch (error: any) {
        reject(new Error(`Failed to parse Python output: ${error.message}`));
      }
    });

    python.on('error', (error) => {
      reject(new Error(`Failed to spawn Python process: ${error.message}`));
    });
  });
}













