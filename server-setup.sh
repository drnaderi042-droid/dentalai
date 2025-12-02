#!/bin/bash

# DentalAI - Ubuntu Server Setup and Deployment Script
# Run this script on your Ubuntu server after transferring files

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check if we're in the vite-js directory
if [ ! -f "package.json" ]; then
    print_error "package.json not found. Please run this script from the vite-js directory."
    exit 1
fi

print_header "DentalAI - Ubuntu Server Deployment"

print_info "Updating system packages..."
sudo apt update && sudo apt upgrade -y

print_info "Installing Node.js and npm..."
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
else
    print_warning "Node.js is already installed"
fi

print_info "Node.js version: $(node --version)"
print_info "npm version: $(npm --version)"

print_info "Installing project dependencies..."
npm install

if [ $? -ne 0 ]; then
    print_error "Failed to install dependencies"
    exit 1
fi

print_info "Building project..."
npm run build

if [ $? -ne 0 ]; then
    print_error "Build failed"
    exit 1
fi

if [ ! -d "dist" ]; then
    print_error "Build failed - dist directory not found"
    exit 1
fi

print_success "Build completed successfully!"
print_info "Build size: $(du -sh dist/ | cut -f1)"

# Ask user about web server setup
echo
read -p "Do you want to install and configure Nginx? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Installing Nginx..."
    sudo apt install -y nginx

    print_info "Configuring Nginx..."
    sudo tee /etc/nginx/sites-available/dentalai > /dev/null <<EOF
server {
    listen 80;
    server_name _;
    root /home/$(whoami)/vite-js/dist;
    index index.html;

    location / {
        try_files \$uri \$uri/ /index.html;
    }

    # Cache static assets
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }

    # Enable gzip
    gzip on;
    gzip_types text/css application/javascript text/javascript application/json;
}
EOF

    # Enable site
    sudo ln -sf /etc/nginx/sites-available/dentalai /etc/nginx/sites-enabled/

    # Remove default site if it exists
    if [ -L "/etc/nginx/sites-enabled/default" ]; then
        sudo rm /etc/nginx/sites-enabled/default
    fi

    print_info "Testing Nginx configuration..."
    sudo nginx -t

    if [ $? -ne 0 ]; then
        print_error "Nginx configuration test failed"
        exit 1
    fi

    print_info "Starting Nginx..."
    sudo systemctl restart nginx
    sudo systemctl enable nginx

    print_success "Nginx configured successfully!"

    # Configure firewall
    print_info "Configuring firewall..."
    sudo ufw allow 80
    sudo ufw allow 22
    echo "y" | sudo ufw enable

    print_success "Firewall configured!"
fi

# Final status
print_header "Deployment Summary"
print_success "Project deployed successfully!"
echo
print_info "Project location: $(pwd)"
print_info "Build files: $(pwd)/dist/"

if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
    print_info "Your app is now running at: http://$SERVER_IP"
    print_info "Nginx status: $(sudo systemctl is-active nginx)"
else
    print_warning "To serve the app manually, run:"
    print_info "  npx serve -s dist/"
    print_info "  # or"
    print_info "  npm install -g serve && serve -s dist/"
fi

echo
print_success "🎉 Deployment completed successfully!"
echo
print_info "Useful commands:"
echo "  - Check app: curl http://localhost"
echo "  - View logs: sudo tail -f /var/log/nginx/access.log"
echo "  - Restart nginx: sudo systemctl restart nginx"
echo "  - Rebuild app: npm run build"
echo



