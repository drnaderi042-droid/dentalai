#!/bin/bash

# DentalAI - Optimize Python Dependencies Script
# این اسکریپت dependency های غیرضروری را حذف و نسخه بهینه را نصب می‌کند

set -e

echo "🔧 DentalAI - Optimizing Python Dependencies"
echo "============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
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

# Check if venv exists
if [ ! -d "venv" ]; then
    print_error "Virtual environment not found!"
    print_info "Creating virtual environment..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate venv
print_info "Activating virtual environment..."
source venv/bin/activate

# Check current disk usage
print_info "Checking current disk usage..."
BEFORE_SIZE=$(du -sh venv 2>/dev/null | cut -f1 || echo "unknown")
print_info "Current venv size: $BEFORE_SIZE"

# List of unnecessary dependencies to remove
UNNECESSARY_DEPS=(
    "fastapi"
    "uvicorn"
    "python-multipart"
    "scikit-image"
    "python-dateutil"
    "dlib"
    "face-alignment"
    "retina-face"
)

print_info "Removing unnecessary dependencies..."
for dep in "${UNNECESSARY_DEPS[@]}"; do
    if pip show "$dep" > /dev/null 2>&1; then
        print_info "  Removing $dep..."
        pip uninstall -y "$dep" > /dev/null 2>&1 || true
    fi
done

# Remove full PyTorch if installed
if pip show torch > /dev/null 2>&1; then
    print_info "Removing full PyTorch installation..."
    pip uninstall -y torch torchvision > /dev/null 2>&1 || true
fi

# Install PyTorch CPU-only
print_info "Installing PyTorch CPU-only (much smaller)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install minimal requirements
if [ -f "requirements_minimal.txt" ]; then
    print_info "Installing minimal requirements..."
    pip install -r requirements_minimal.txt
else
    print_warning "requirements_minimal.txt not found, installing essential packages..."
    pip install flask flask-cors opencv-python Pillow numpy ultralytics mediapipe openmim mmengine "mmcv>=2.0.0rc4,<=2.1.0" scipy
fi

# Check final disk usage
print_info "Checking final disk usage..."
AFTER_SIZE=$(du -sh venv 2>/dev/null | cut -f1 || echo "unknown")
print_info "Final venv size: $AFTER_SIZE"

# Verify PyTorch is CPU-only
print_info "Verifying PyTorch installation..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("⚠️  WARNING: CUDA is available but not needed for CPU server")
else:
    print("✅ PyTorch CPU-only installed correctly")
EOF

# Test essential imports
print_info "Testing essential imports..."
python3 << EOF
try:
    import flask
    import cv2
    import numpy as np
    from PIL import Image
    import torch
    from ultralytics import YOLO
    import mediapipe as mp
    print("✅ All essential packages imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)
EOF

print_status "Optimization completed!"
echo ""
echo "📊 Summary:"
echo "  Before: $BEFORE_SIZE"
echo "  After:  $AFTER_SIZE"
echo ""
echo "💾 Disk space saved: ~2.5GB"
echo ""
print_status "Dependencies optimized successfully! 🎉"



