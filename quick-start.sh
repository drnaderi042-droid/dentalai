#!/bin/bash

# DentalAI Quick Start Script
# Run this script to deploy the entire DentalAI system with Docker

set -e

echo "🐼 DentalAI - Full System Deployment"
echo "===================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "📋 Creating .env file from template..."
    if [ -f "env.example" ]; then
        cp env.example .env
        echo "✅ .env file created. Please edit it with your settings:"
        echo "   nano .env"
        echo ""
        read -p "Press Enter after editing .env file..."
    else
        echo "⚠️  env.example not found. Creating basic .env file..."
        cat > .env << EOF
DB_PASSWORD=dentalai2024
DATABASE_URL="postgresql://postgres:dentalai2024@postgres:5432/dentalai"
NEXTAUTH_SECRET=dentalai-secret-key-change-this
NEXTAUTH_URL=http://localhost:3001
VITE_API_URL=http://localhost:3001
VITE_AI_API_URL=http://localhost:8000
NEXT_PUBLIC_API_URL=http://localhost:3001
EOF
    fi
fi

echo "🔨 Building and starting all services..."
echo "This may take several minutes on first run..."

# Build and start all services
if command -v docker-compose &> /dev/null; then
    docker-compose up -d --build
else
    docker compose up -d --build
fi

echo "⏳ Waiting for services to start..."
sleep 30

echo "📊 Checking service status..."
if command -v docker-compose &> /dev/null; then
    docker-compose ps
else
    docker compose ps
fi

echo ""
echo "🎉 DentalAI deployment completed!"
echo "=================================="
echo ""
echo "🌐 Access your application:"
echo "   Frontend:     http://localhost:3030"
echo "   Dashboard:    http://localhost:3000"
echo "   API:          http://localhost:7272"
echo "   AI Server:    http://localhost:5001"
echo "   Database:     localhost:5432"
echo ""
echo "📋 Useful commands:"
echo "   View logs:    docker-compose logs -f [service-name]"
echo "   Stop all:     docker-compose down"
echo "   Restart:      docker-compose restart"
echo "   Update:       docker-compose pull && docker-compose up -d"
echo ""
echo "🔍 Check service health:"
echo "   curl http://localhost:3001/api/health"
echo "   curl http://localhost:8000/health"
echo ""
echo "📚 For troubleshooting, see DEPLOYMENT-README.md"
echo ""
echo "Happy DentalAI! 🦷🤖"
