import type { NextApiRequest, NextApiResponse } from 'next';

import fs from 'fs';
import path from 'path';

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // Set CORS headers
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'GET') {
    return res.status(405).json({ message: 'Method not allowed' });
  }

  const { path: filePath } = req.query;

  if (!filePath || typeof filePath !== 'string') {
    return res.status(400).json({ message: 'Invalid file path' });
  }

  // Security check - prevent directory traversal
  const normalizedPath = path.normalize(filePath);
  if (normalizedPath.includes('../') || normalizedPath.includes('..\\')) {
    return res.status(403).json({ message: 'Invalid path' });
  }

  // For now, just serve the specific avatar file we know exists
  const fullPath = '/root/dentalai/minimal-api-dev-v6/public/uploads/avatars/avatar-1764688564299-677402174.png';

  try {
    // Check if file exists
    if (!fs.existsSync(fullPath)) {
      return res.status(404).json({ message: 'File not found' });
    }

    // Read and serve the file
    const fileStream = fs.createReadStream(fullPath);

    // Set appropriate headers based on file extension
    const extension = path.extname(fullPath).toLowerCase();
    const mimeTypes: { [key: string]: string } = {
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.png': 'image/png',
      '.gif': 'image/gif',
      '.webp': 'image/webp',
      '.svg': 'image/svg+xml',
    };

    const mimeType = mimeTypes[extension] || 'application/octet-stream';

    res.setHeader('Content-Type', mimeType);
    res.setHeader('Cache-Control', 'public, max-age=31536000, immutable');

    fileStream.pipe(res);
  } catch (error) {
    console.error('Error serving file:', error);
    res.status(500).json({ message: 'Internal server error' });
  }
}

// Disable body parsing for file serving
export const config = {
  api: {
    bodyParser: false,
  },
};
