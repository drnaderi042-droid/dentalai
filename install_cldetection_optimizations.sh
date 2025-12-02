#!/bin/bash
# اسکریپت نصب Dependency های لازم برای بهینه‌سازی CLdetection2023
# Install script for CLdetection2023 optimization dependencies (Linux/Mac)

echo "======================================================================"
echo "🚀 CLdetection2023 Optimization Dependencies Installer (Linux/Mac)"
echo "======================================================================"
echo ""

# Check if .venv exists
if [ -f ".venv/bin/python" ]; then
    echo "✅ Virtual environment found"
    PYTHON_CMD=".venv/bin/python"
    PIP_CMD=".venv/bin/pip"
else
    echo "⚠️  Virtual environment not found, using system Python"
    PYTHON_CMD="python3"
    PIP_CMD="pip3"
fi

echo ""
echo "📦 Installing required packages..."
echo ""

# Install PyTorch 2.0+ (CPU version)
echo "Installing PyTorch 2.0+ (CPU)..."
$PIP_CMD install --upgrade torch>=2.0.0 torchvision>=0.15.0 --index-url https://download.pytorch.org/whl/cpu

# Install MKL for faster CPU operations
echo ""
echo "Installing Intel MKL..."
$PIP_CMD install --upgrade mkl mkl-service

# Optional: Install ONNX (uncomment if needed)
# echo ""
# echo "Installing ONNX (optional)..."
# $PIP_CMD install --upgrade onnx onnxruntime

# Optional: Install Numba (uncomment if needed)
# echo ""
# echo "Installing Numba (optional)..."
# $PIP_CMD install --upgrade numba

echo ""
echo "======================================================================"
echo "✅ Installation Complete!"
echo "======================================================================"
echo ""
echo "📝 Next steps:"
echo "   1. Restart your Python server"
echo "   2. Test the optimizations with a CLdetection request"
echo "   3. Check logs for 'torch.compile enabled' message"
echo ""
echo "💡 Tips:"
echo "   - If torch.compile is not available, upgrade PyTorch: pip install torch>=2.0.0"
echo "   - For maximum speed, consider ONNX conversion"
echo ""










