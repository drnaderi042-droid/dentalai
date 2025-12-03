#!/usr/bin/env python3
import http.server
import socketserver
import urllib.request
import urllib.error
import os

class SPAServer(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Proxy /api/* and /uploads/* requests to backend server
        # /assets/* should be served by the frontend server directly
        if self.path.startswith('/api/') or self.path.startswith('/uploads/'):
            self.proxy_to_backend()
        else:
            # If the path doesn't exist, serve index.html
            if not os.path.exists(self.translate_path(self.path)):
                self.path = '/index.html'
            return super().do_GET()

    def proxy_to_backend(self):
        """Proxy requests to backend server"""
        backend_url = f"http://localhost:7272{self.path}"

        try:
            # Make request to backend
            with urllib.request.urlopen(backend_url) as response:
                # Copy headers
                self.send_response(response.status)
                for header, value in response.headers.items():
                    if header.lower() not in ['transfer-encoding', 'connection']:
                        self.send_header(header, value)
                self.end_headers()

                # Copy response body
                self.wfile.write(response.read())

        except urllib.error.HTTPError as e:
            self.send_error(e.code, e.reason)
        except Exception as e:
            self.send_error(500, f"Proxy error: {str(e)}")

# Change to dist directory
os.chdir('dist')

with socketserver.TCPServer(("", 3030), SPAServer) as httpd:
    print("Serving SPA on port 3030 (with backend proxy for /api/* and /uploads/*)")
    httpd.serve_forever()
