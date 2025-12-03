import type { NextApiRequest, NextApiResponse } from 'next';

import Cors from 'cors';

// ----------------------------------------------------------------------

type Middleware = (req: NextApiRequest, res: NextApiResponse, next: (result: any) => void) => void;

const initMiddleware = (middleware: Middleware) => (req: NextApiRequest, res: NextApiResponse) =>
  new Promise<void>((resolve, reject) => {
    middleware(req, res, (result: any) => {
      if (result instanceof Error) {
        return reject(result);
      }

      return resolve();
    });
  });

// ----------------------------------------------------------------------

// You can read more about the available options here: https://github.com/expressjs/cors#configuration-options
const cors = initMiddleware(
  Cors({
    origin: (origin, callback) => {
      // Allow requests with no origin (like mobile apps or curl requests)
      if (!origin) return callback(null, true);
      
      // Allow localhost and 127.0.0.1
      if (origin.includes('localhost') || origin.includes('127.0.0.1')) {
        return callback(null, true);
      }
      
      // Allow any IP address in development (192.168.x.x, 10.x.x.x, etc.)
      if (process.env.NODE_ENV !== 'production') {
        // Check if it's an IP address pattern
        const ipPattern = /^https?:\/\/(\d{1,3}\.){3}\d{1,3}(:\d+)?$/;
        if (ipPattern.test(origin)) {
          return callback(null, true);
        }
      }
      
      // Allow the server IP
      if (origin.startsWith('http://31.56.233.34:')) {
        return callback(null, true);
      }

      // In production, you might want to restrict to specific domains
      // For now, allow all in development
      if (process.env.NODE_ENV !== 'production') {
        return callback(null, true);
      }
      
      // Default: reject
      callback(new Error('Not allowed by CORS'));
    },
    methods: ['GET', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization', 'Content-Length', 'X-Requested-With', 'Accept'],
    credentials: true,
    optionsSuccessStatus: 200,
  })
);

export default cors;
