/**
 * Contour Detection API Endpoint
 * Detects anatomical contours in cephalometric radiographs
 * Supports multiple detection methods
 */

import type { NextApiRequest, NextApiResponse } from 'next';

import { verify } from 'jsonwebtoken';

import cors from 'src/utils/cors';

const JWT_SECRET = process.env.JWT_SECRET || 'your-secret-key';

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  try {
    await cors(req, res);

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
    } else {
      console.log('[Contour Detect] ⚠️ Development mode - Authentication skipped');
    }

    const { imageUrl, landmarks, method = 'auto', regions } = req.body;

    if (!imageUrl) {
      return res.status(400).json({ error: 'Image URL is required' });
    }

    if (!landmarks || Object.keys(landmarks).length === 0) {
      return res.status(400).json({ 
        error: 'Landmarks are required for contour detection',
        hint: 'Please detect landmarks first using /api/cephalometric-detect'
      });
    }

    console.log('[Contour Detect] Processing request');
    console.log('[Contour Detect] Method:', method);
    console.log('[Contour Detect] Regions:', regions || 'all');
    console.log('[Contour Detect] Landmarks count:', Object.keys(landmarks).length);

    // Try Python contour detection service
    let contours = null;
    let detectionInfo: any = {
      method: 'unknown',
      timestamp: new Date().toISOString(),
      service: 'unknown'
    };

    // Method 1: Python Contour Detection Service (Primary)
    const pythonServiceUrl = process.env.PYTHON_CONTOUR_SERVICE_URL || 'http://localhost:5002';
    
    try {
      console.log('[Contour Detect] Attempting Python service...');
      console.log('[Contour Detect] Service URL:', pythonServiceUrl);

      const pythonResponse = await fetch(`${pythonServiceUrl}/detect-contours`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image_url: imageUrl,
          landmarks,
          method,
          regions: regions || null
        }),
      });

      if (pythonResponse.ok) {
        const result = await pythonResponse.json();
        console.log('[Contour Detect] Python service response:', {
          success: result.success,
          num_regions: result.num_regions || 0,
          method_used: result.method
        });

        if (result && result.success) {
          contours = result.contours;
          detectionInfo = {
            method: result.method || method,
            timestamp: new Date().toISOString(),
            service: 'python_enhanced',
            num_regions: result.num_regions || 0,
            python_service: true
          };
          console.log('[Contour Detect] ✅ Python contour detection successful!');
        }
      } else {
        console.error('[Contour Detect] Python service returned error:', pythonResponse.status);
      }
    } catch (pythonError) {
      console.error('[Contour Detect] Python service error:', pythonError);
      console.log('[Contour Detect] Will use fallback mock data');
    }

    // Method 2: Mock/Fallback (if Python service failed)
    if (!contours) {
      console.log('[Contour Detect] ⚠️ Using mock contour data (Python service unavailable)');
      contours = generateMockContours(landmarks, regions);
      detectionInfo = {
        method: 'mock',
        timestamp: new Date().toISOString(),
        service: 'fallback',
        note: 'Python service unavailable - using mock data'
      };
    }

    // Return results
    return res.status(200).json({
      success: true,
      contours,
      detectionInfo,
      requestedMethod: method,
      requestedRegions: regions || 'all',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    console.error('[Contour Detect] Error:', error);
    return res.status(500).json({
      error: 'Failed to detect contours',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: new Date().toISOString()
    });
  }
}

/**
 * Generate mock contours for fallback
 */
function generateMockContours(landmarks: any, regions?: string[]) {
  const allRegions = regions || [
    'soft_tissue_profile',
    'mandible_border',
    'maxilla_border',
    'sella_turcica',
    'orbital_rim'
  ];

  const mockContours: any = {};

  // Generate simple contours around landmarks
  allRegions.forEach(regionType => {
    const regionLandmarks = getRegionLandmarks(regionType, landmarks);
    
    if (regionLandmarks.length > 0) {
      // Create a simple contour connecting the landmarks
      const points = regionLandmarks.map(lm => [lm.x, lm.y]);
      
      // Add some interpolated points for smoother contour
      const smoothPoints: number[][] = [];
      for (let i = 0; i < points.length; i++) {
        const current = points[i];
        const next = points[(i + 1) % points.length];
        
        smoothPoints.push(current);
        
        // Add interpolated point
        const midX = (current[0] + next[0]) / 2;
        const midY = (current[1] + next[1]) / 2;
        smoothPoints.push([midX, midY]);
      }

      mockContours[regionType] = {
        success: true,
        contour: smoothPoints,
        num_points: smoothPoints.length,
        method: 'mock',
        region_type: regionType
      };
    } else {
      mockContours[regionType] = {
        success: false,
        error: 'No landmarks available for this region',
        contour: [],
        method: 'mock'
      };
    }
  });

  return mockContours;
}

/**
 * Get landmarks for a specific region
 */
function getRegionLandmarks(regionType: string, landmarks: any) {
  const regionMappings: Record<string, string[]> = {
    soft_tissue_profile: ['N`', 'N', 'Sn', 'Ls', 'Li', 'Pog`', 'Me'],
    mandible_border: ['Go', 'Gn', 'Me', 'Pog'],
    maxilla_border: ['ANS', 'PNS', 'A'],
    sella_turcica: ['S'],
    orbital_rim: ['Or']
  };

  const landmarkNames = regionMappings[regionType] || [];
  const regionLandmarks: any[] = [];

  landmarkNames.forEach(name => {
    if (landmarks[name]) {
      regionLandmarks.push({
        name,
        x: landmarks[name].x || 0,
        y: landmarks[name].y || 0
      });
    }
  });

  return regionLandmarks;
}
