#!/bin/bash

# Production deployment script for Ubuntu server
# Make this file executable with: chmod +x deploy-ubuntu.sh

set -e  # Exit on any error

echo "🚀 Starting production deployment for DentalAI project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "vite-js/package.json" ]; then
    print_error "vite-js/package.json not found. Please run this script from the project root directory."
    exit 1
fi

# Navigate to vite-js directory
cd vite-js

print_status "Installing dependencies..."
npm ci --production=false

print_status "Running lint check..."
npm run lint

print_status "Building project for production..."
npm run build

if [ ! -d "dist" ]; then
    print_error "Build failed - dist directory not found"
    exit 1
fi

print_status "Build completed successfully!"
print_status "Build size: $(du -sh dist/ | cut -f1)"

# Optional: Install and configure nginx
read -p "Do you want to install/configure nginx? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing nginx..."
    sudo apt update
    sudo apt install -y nginx

    print_status "Configuring nginx..."
    sudo tee /etc/nginx/sites-available/dentalai > /dev/null <<EOF
server {
    listen 80;
    server_name _;
    root /var/www/dentalai;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Gzip compression
    gzip on;
    gzip_types text/css application/javascript text/javascript application/json;
}
EOF

    sudo ln -sf /etc/nginx/sites-available/dentalai /etc/nginx/sites-enabled/
    sudo rm -f /etc/nginx/sites-enabled/default

    print_status "Copying built files to nginx..."
    sudo mkdir -p /var/www/dentalai
    sudo cp -r dist/* /var/www/dentalai/
    sudo chown -R www-data:www-data /var/www/dentalai

    print_status "Testing nginx configuration..."
    sudo nginx -t

    print_status "Restarting nginx..."
    sudo systemctl restart nginx
    sudo systemctl enable nginx

    print_status "Deployment completed!"
    print_status "Your app is now running at: http://your-server-ip"
else
    print_warning "Skipping nginx installation."
    print_status "You can manually serve the built files from: $(pwd)/dist/"
fi

print_status "Deployment script completed successfully! 🎉"



