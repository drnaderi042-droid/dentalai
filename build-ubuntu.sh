#!/bin/bash

# Build script for Ubuntu server
# Make this file executable with: chmod +x build-ubuntu.sh

echo "🚀 Starting build process for DentalAI project..."

# Check if we're in the right directory
if [ ! -f "vite-js/package.json" ]; then
    echo "❌ Error: vite-js/package.json not found. Please run this script from the project root directory."
    exit 1
fi

# Navigate to vite-js directory
cd vite-js

echo "📦 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "🔨 Building project..."
npm run build

if [ $? -ne 0 ]; then
    echo "❌ Build failed"
    exit 1
fi

echo "✅ Build completed successfully!"
echo "📁 Built files are in: $(pwd)/dist/"

# Optional: Create a simple deployment structure
echo "📋 Build summary:"
echo "   - Build directory: $(pwd)/dist/"
echo "   - Main files: index.html, assets/, etc."
echo ""
echo "To serve the built files, you can:"
echo "1. Use nginx/apache to serve the dist/ folder"
echo "2. Use a simple HTTP server: npx serve dist/"
echo "3. Copy files to web server directory"



