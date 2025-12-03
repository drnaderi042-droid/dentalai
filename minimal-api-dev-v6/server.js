const { createServer } = require('https');
const { parse } = require('url');
const next = require('next');
const fs = require('fs');
const path = require('path');

const dev = true; // Force development mode for now
const app = next({ dev });
const handle = app.getRequestHandler();

const httpsOptions = {
  key: fs.readFileSync('/etc/letsencrypt/live/ceph2.bioritalin.ir/privkey.pem'),
  cert: fs.readFileSync('/etc/letsencrypt/live/ceph2.bioritalin.ir/fullchain.pem'),
};

// Static file server for Vite.js frontend and uploads
const serveStaticFile = (req, res, filePath) => {
  const fullPath = path.join(__dirname, 'public', filePath);

  if (fs.existsSync(fullPath)) {
    const ext = path.extname(fullPath).toLowerCase();
    const contentType = {
      '.html': 'text/html',
      '.js': 'application/javascript',
      '.css': 'text/css',
      '.json': 'application/json',
      '.png': 'image/png',
      '.jpg': 'image/jpeg',
      '.jpeg': 'image/jpeg',
      '.gif': 'image/gif',
      '.svg': 'image/svg+xml',
      '.ico': 'image/x-icon',
      '.woff': 'font/woff',
      '.woff2': 'font/woff2'
    }[ext] || 'text/plain';

    res.writeHead(200, { 'Content-Type': contentType });
    fs.createReadStream(fullPath).pipe(res);
  } else {
    res.writeHead(404);
    res.end('File not found');
  }
};

app.prepare().then(() => {
  createServer(httpsOptions, (req, res) => {
    const parsedUrl = parse(req.url, true);
    const { pathname } = parsedUrl;

    // Handle uploads - serve static files directly
    if (pathname.startsWith('/uploads/')) {
      serveStaticFile(req, res, pathname);
      return;
    }

    // Handle assets - serve static files directly
    if (pathname.startsWith('/assets/') || pathname.startsWith('/fonts/') || pathname.startsWith('/favicon.ico')) {
      serveStaticFile(req, res, pathname);
      return;
    }

    // Handle API routes with Next.js
    if (pathname.startsWith('/api/')) {
      handle(req, res, parsedUrl);
      return;
    }

    // For all other routes (SPA routes), serve the Vite.js index.html
    serveStaticFile(req, res, 'index.html');
  }).listen(443, '0.0.0.0', (err) => {
    if (err) throw err;
    console.log('> Ready on https://ceph2.bioritalin.ir');
    console.log('> Serving Vite.js frontend with Next.js API');
    console.log('> SSL certificates loaded from Let\'s Encrypt');
  });
});
