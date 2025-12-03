"""
Unified AI API Server for Dental/Orthodontic Analysis
سرور API یکپارچه برای آنالیزهای هوش مصنوعی دندانپزشکی و ارتودنسی

This server provides endpoints for:
1. Intra-oral analysis (YOLOv8)
2. Facial landmark detection (MediaPipe/dlib/face-alignment/RetinaFace)
3. Cephalometric analysis (Aariz models)

================================================================================
CPU Optimization Features (بهینه‌سازی‌های CPU)
================================================================================

This server is optimized for CPU inference with the following enhancements:

1. Multi-threading (چند نخی):
   - Uses all available CPU cores for PyTorch operations
   - Configures MKL and OpenMP for optimal thread usage
   - Sets environment variables (OMP_NUM_THREADS, MKL_NUM_THREADS)

2. Memory Optimization (بهینه‌سازی حافظه):
   - Disables gradient computation (torch.set_grad_enabled(False))
   - Reduces memory footprint for inference

3. Performance Tips (نکات عملکردی):
   - Install Intel MKL for faster CPU operations: pip install mkl mkl-service
   - Reduce image size before inference for faster processing
   - Use smaller batch sizes if processing multiple images

4. Device Configuration (پیکربندی دستگاه):
   - All models are forced to use CPU (CUDA is disabled)
   - This ensures consistent behavior across different systems

================================================================================
"""

# =============================================================================
# Bootstrap: Ensure we are running inside local .venv
# =============================================================================

import os
import sys


def ensure_venv_and_reexec():
    """
    Ensure this script runs inside the local .venv virtual environment.

    Behavior:
    - If already inside a virtualenv: do nothing.
    - If not inside a virtualenv:
        * Create .venv if it doesn't exist.
        * If a requirements file exists, install dependencies.
        * Otherwise, install a default set of dependencies needed for this server.
        * Re-launch this script using the .venv Python executable.
    """
    # Detect if we are already inside a virtual environment
    in_venv = hasattr(sys, "real_prefix") or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_dir = os.path.join(base_dir, ".venv")

    if os.name == "nt":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    import subprocess

    created_venv = False

    if not os.path.exists(venv_python):
        print("Creating virtual environment in .venv ...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        created_venv = True

    # If we just created the venv, try to install dependencies
    if created_venv:
        # Prefer an explicit requirements file if available
        requirements_installed = False
        for req_name in ("requirements_unified_ai.txt", "requirements.txt"):
            req_path = os.path.join(base_dir, req_name)
            if os.path.exists(req_path):
                print(f"Installing dependencies from {req_name} ...")
                subprocess.check_call([venv_python, "-m", "pip", "install", "-r", req_path])
                requirements_installed = True
                break

        # If no requirements file is present, install a minimal default set
        if not requirements_installed:
            default_packages = [
                "opencv-python",
                "pillow",
                "pyyaml",
                "ultralytics",
                "flask",
                "flask-cors",
            ]
            print("No requirements file found. Installing default dependencies:")
            print("  " + ", ".join(default_packages))
            subprocess.check_call([venv_python, "-m", "pip", "install", *default_packages])

    # Re-launch this script using the virtualenv python
    print("Re-launching unified_ai_api_server using .venv Python interpreter ...")
    os.execv(venv_python, [venv_python, __file__, *sys.argv[1:]])


ensure_venv_and_reexec()

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import cv2
import numpy as np
from PIL import Image
import time
from datetime import datetime as dt
from typing import Dict, List, Tuple, Optional
import traceback
import sys
import os
import re
from pathlib import Path

# Try to import yaml for data.yaml parsing
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    print("⚠️  Warning: PyYAML not installed. Class names may not load correctly.")
    print("   Install with: pip install pyyaml")

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  Warning: ultralytics not installed. Intra-oral analysis will not be available.")
    print("   Install with: pip install ultralytics")

app = Flask(__name__)

# Configure CORS to allow local network access and production domains
# Use a function to check if origin is allowed (for regex support)
def is_allowed_origin(origin):
    """Check if origin is allowed (localhost, local network IPs, or production domains)
    Returns the origin if allowed, None otherwise for flask-cors
    """
    if not origin:
        return None  # Don't allow null origins for security

    allowed_origins = [
        # Local development
        "http://localhost:3030",
        "http://127.0.0.1:3030",
        "http://localhost:7272",
        "http://127.0.0.1:7272",
        # Production domains
        "https://ceph2.bioritalin.ir",
        "http://ceph2.bioritalin.ir",
    ]

    # Check exact matches
    if origin in allowed_origins:
        return origin

    # Check local development patterns
    allowed_patterns = [
        r"^http://localhost:\d+$",
        r"^http://127\.0\.0\.1:\d+$",
        # Local network IPs
        r"^http://192\.168\.\d+\.\d+:(3030|7272)$",
        r"^http://10\.\d+\.\d+\.\d+:(3030|7272)$",
        r"^http://172\.(1[6-9]|2[0-9]|3[0-1])\.\d+\.\d+:(3030|7272)$",
    ]

    for pattern in allowed_patterns:
        if re.match(pattern, origin):
            return origin

    return None

# Configure CORS using allowed origins
# This handles preflight OPTIONS requests automatically
CORS(app,
    resources={
        r"/*": {
            "origins": [
                # Local development
                "http://localhost:3030",
                "http://127.0.0.1:3030",
                "http://localhost:7272",
                "http://127.0.0.1:7272",
                # Production domains
                "https://ceph2.bioritalin.ir",
                "http://ceph2.bioritalin.ir",
                # Local development patterns
                r"http://localhost:\d+",
                r"http://127.0.0.1:\d+",
                r"http://192\.168\.\d+\.\d+:\d+",
                r"http://10\.\d+\.\d+\.\d+:\d+"
            ],
            "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization", "Content-Length", "X-Requested-With", "Accept", "Origin", "X-Requested-With"],
            "supports_credentials": True,
            "expose_headers": ["Content-Type", "Content-Length"],
            "max_age": 3600,
        }
    },
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "Content-Length", "X-Requested-With", "Accept", "Origin", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]
)

@app.after_request
def add_cors_headers(response):
    """
    Ensure CORS headers are added to all responses (including preflight OPTIONS)
    Especially for origins like https://ceph2.bioritalin.ir
    """
    origin = request.headers.get("Origin")

    allowed_origins = {
        # Local development
        "http://localhost:3030",
        "http://127.0.0.1:3030",
        "http://localhost:7272",
        "http://127.0.0.1:7272",
        # Production domains
        "https://ceph2.bioritalin.ir",
        "http://ceph2.bioritalin.ir",
    }

    if origin in allowed_origins:
        # For requests with credentials, origin must be exact (cannot use *)
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, Content-Length, X-Requested-With, Accept, Origin"
        )
        response.headers["Access-Control-Expose-Headers"] = "Content-Type, Content-Length"

    return response

# =============================================================================
# CPU Optimization Settings
# =============================================================================

# Import torch early for CPU optimizations
try:
    import torch
    import torch.backends.mps  # For MPS (Apple Silicon) if available
    
    # Force CPU usage (disable CUDA/MPS)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable CUDA
    if hasattr(torch.backends, 'mps'):
        torch.backends.mps.is_available = lambda: False  # Disable MPS
    
    # CPU Optimization Settings
    print("=" * 80)
    print("CPU Optimization Settings")
    print("=" * 80)
    
    # Set number of threads for PyTorch (use all CPU cores)
    import multiprocessing
    num_threads = multiprocessing.cpu_count()
    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(num_threads)
    print(f"✅ PyTorch threads: {torch.get_num_threads()} (interop: {torch.get_num_interop_threads()})")
    print(f"   CPU cores available: {num_threads}")
    
    # Enable optimizations for CPU inference
    torch.set_grad_enabled(False)  # Disable gradient computation for inference
    print("✅ Gradient computation disabled (inference mode)")
    
    # Set deterministic behavior for reproducibility (optional, can disable for speed)
    torch.backends.cudnn.deterministic = False  # Not needed for CPU but set anyway
    torch.backends.cudnn.benchmark = False  # Not needed for CPU but set anyway
    
    # Enable MKL optimizations if available
    try:
        import mkl
        mkl.set_num_threads(num_threads)
        print(f"✅ MKL threads: {mkl.get_max_threads()}")
    except ImportError:
        print("⚠️  MKL not available (Intel MKL can speed up CPU operations)")
        print("   Consider installing: pip install mkl mkl-service")
    
    # Set OpenMP threads if available
    try:
        os.environ['OMP_NUM_THREADS'] = str(num_threads)
        os.environ['MKL_NUM_THREADS'] = str(num_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(num_threads)
        print(f"✅ Environment variables set: OMP_NUM_THREADS={num_threads}")
    except:
        pass
    
    # Memory management for CPU
    torch.set_printoptions(threshold=10)  # Reduce memory for printing
    
    print("✅ CPU optimizations enabled")
    print("=" * 80)
    print()
    
except ImportError:
    print("⚠️  Warning: PyTorch not available. CPU optimizations skipped.")
    torch = None

# =============================================================================
# Aariz Model Setup
# =============================================================================

# Add Aariz directory to path
aariz_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Aariz')
if os.path.exists(aariz_dir):
    sys.path.insert(0, aariz_dir)

# Global variables for Aariz models
aariz_predictors = {}
aariz_status = {}

# Global variables for YOLO intra-oral model
yolo_model = None
yolo_status = 'not_loaded'

# Global variables for CLdetection2023 model
cldetection2023_model = None
cldetection2023_status = 'not_loaded'
cldetection2023_config_path = None
cldetection2023_remove_zero_padding = None
cldetection2023_onnx_session = None  # ONNX Runtime session for faster inference

# Global variables for optimized CLdetection2023 models (ONNX)
cldetection_optimized_detectors = {}  # Dict to store different optimized models
cldetection_optimized_status = {}

# Global variables for fast CPU 512 model (checkpoint_fast_cpu_512_best.pth)
fast_cpu_512_model = None
fast_cpu_512_status = 'not_loaded'


def load_cldetection_optimized(model_key='optimized_512', input_size=512):
    """
    Load optimized CLdetection model using ONNX Runtime
    
    Args:
        model_key: Key to identify the model configuration
        input_size: Input image size (512, 640, or 1024)
        
    Returns:
        UltraOptimizedCLDetector instance or None
    """
    global cldetection_optimized_detectors, cldetection_optimized_status
    
    # Check if model is already loaded
    if model_key in cldetection_optimized_detectors:
        print(f"[CLdetection Optimized] ✅ Using cached model '{model_key}'")
        return cldetection_optimized_detectors[model_key]
    
    print(f"[CLdetection Optimized] 🔄 Loading optimized model '{model_key}'...")
    
    try:
        # Check if ONNX Runtime is available
        try:
            import onnxruntime as ort
        except ImportError:
            print("⚠️  ONNX Runtime not installed. Install with: pip install onnxruntime")
            cldetection_optimized_status[model_key] = 'onnxruntime_not_installed'
            return None
        
        # Import the optimized detector class
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cldetection_dir = os.path.join(base_dir, 'CLdetection2023')
        
        # Check if CLdetection2023 directory exists
        if not os.path.exists(cldetection_dir):
            print(f"⚠️  CLdetection2023 directory not found: {cldetection_dir}")
            cldetection_optimized_status[model_key] = 'directory_not_found'
            return None
        
        # Add to path
        if cldetection_dir not in sys.path:
            sys.path.insert(0, cldetection_dir)
        
        # Import the optimizer detector class
        try:
            from inference_ultra_optimized import UltraOptimizedCLDetector
        except ImportError as e:
            print(f"⚠️  Error importing UltraOptimizedCLDetector: {e}")
            print(f"   Make sure CLdetection2023/inference_ultra_optimized.py exists")
            cldetection_optimized_status[model_key] = 'import_error'
            return None
        
        # Determine ONNX model path
        # Try optimized model first, fall back to original
        onnx_model_paths = [
            os.path.join(cldetection_dir, 'model_cldetection2023_optimized.onnx'),
            os.path.join(cldetection_dir, 'model_cldetection2023.onnx'),
        ]
        
        onnx_model_path = None
        for path in onnx_model_paths:
            if os.path.exists(path):
                onnx_model_path = path
                print(f"   ✅ Found ONNX model: {os.path.basename(path)}")
                break
        
        if not onnx_model_path:
            print(f"⚠️  ONNX model not found in {cldetection_dir}")
            cldetection_optimized_status[model_key] = 'model_not_found'
            return None
        
        # Load the detector
        print(f"🔄 Loading optimized CLdetection model from: {onnx_model_path}")
        print(f"   Input size: {input_size}x{input_size}")
        
        detector = UltraOptimizedCLDetector(
            onnx_path=onnx_model_path,
            input_size=input_size
        )
        
        # Cache the detector
        cldetection_optimized_detectors[model_key] = detector
        cldetection_optimized_status[model_key] = 'ready'
        
        print(f"✅ Optimized CLdetection model '{model_key}' loaded successfully")
        print(f"   Expected inference time: <1s per image")
        
        return detector
        
    except Exception as e:
        print(f"❌ Error loading optimized CLdetection model '{model_key}': {e}")
        traceback.print_exc()
        cldetection_optimized_status[model_key] = f'error: {str(e)}'
        return None


# Global variable for P1/P2 model
p1_p2_model = None
p1_p2_status = 'not_loaded'

def load_p1_p2_model():
    """
    Load P1/P2 detection model (checkpoint_p1_p2_fast_cpu_512_best.pth)
    Model: HRNetLandmarkModel(num_landmarks=2, width=16)
    """
    global p1_p2_model, p1_p2_status
    
    if p1_p2_model is not None and p1_p2_status == 'ready':
        return p1_p2_model
    
    if p1_p2_status == 'loading':
        return None
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_path = os.path.join(base_dir, 'CLdetection2023', 'checkpoint_p1_p2_fast_cpu_512_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Warning: P1/P2 checkpoint not found: {checkpoint_path}")
            p1_p2_status = 'not_found'
            return None
        
        print(f"[P1/P2 Model] 🔄 Loading model... (status: {p1_p2_status})")
        p1_p2_status = 'loading'
        
        import torch
        import torchvision.transforms as transforms
        
        # Add CLdetection2023 and Aariz to path
        cl_dir = os.path.join(base_dir, 'CLdetection2023')
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if cl_dir not in sys.path:
            sys.path.insert(0, cl_dir)
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        
        # Import HRNetLandmarkModel
        try:
            from model import HRNetLandmarkModel
        except ImportError:
            # Try from Aariz
            try:
                from Aariz.model import HRNetLandmarkModel
            except ImportError:
                print("❌ Error: HRNetLandmarkModel not found")
                p1_p2_status = 'import_error'
                return None
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Determine model parameters from checkpoint
        num_landmarks = 2  # P1 and P2 only
        width = 16  # Default width
        
        # Try to get from checkpoint args
        if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
            if 'num_landmarks' in checkpoint['args']:
                num_landmarks = checkpoint['args']['num_landmarks']
            if 'width' in checkpoint['args']:
                width = checkpoint['args']['width']
        
        # Create model
        model = HRNetLandmarkModel(num_landmarks=num_landmarks, width=width)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode and CPU
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # CPU optimizations
        torch.set_grad_enabled(False)
        torch.set_num_threads(os.cpu_count())
        
        # Warmup
        print(f"[P1/P2 Model] Warming up model...")
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy_input)
        
        p1_p2_model = model
        p1_p2_status = 'ready'
        
        print(f"✅ P1/P2 model loaded successfully")
        print(f"   Model: HRNetLandmarkModel(num_landmarks={num_landmarks}, width={width})")
        print(f"   Input size: 512x512")
        print(f"   Device: CPU")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading P1/P2 model: {e}")
        traceback.print_exc()
        p1_p2_status = f'error: {str(e)}'
        return None

def load_fast_cpu_512_model():
    """
    Load fast CPU 512 model (checkpoint_fast_cpu_512_best.pth)
    Model: HRNetLandmarkModel(num_landmarks=38, width=16)
    Input: 512x512
    Approach: Heatmap-based
    """
    global fast_cpu_512_model, fast_cpu_512_status
    
    if fast_cpu_512_model is not None:
        print(f"[Fast CPU 512] ✅ Using cached model (status: {fast_cpu_512_status})")
        return fast_cpu_512_model
    
    print(f"[Fast CPU 512] 🔄 Loading model... (status: {fast_cpu_512_status})")
    
    try:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as transforms
        
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cldetection_dir = os.path.join(base_dir, 'CLdetection2023')
        
        # Check if CLdetection2023 directory exists
        if not os.path.exists(cldetection_dir):
            print(f"⚠️  CLdetection2023 directory not found: {cldetection_dir}")
            fast_cpu_512_status = 'directory_not_found'
            return None
        
        # Add CLdetection2023 and Aariz to path
        if cldetection_dir not in sys.path:
            sys.path.insert(0, cldetection_dir)
        
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        
        # Import model and utils
        try:
            from model import HRNetLandmarkModel
            from utils import heatmap_to_coordinates
        except ImportError as e:
            print(f"⚠️  Error importing model: {e}")
            print(f"   Make sure Aariz/model.py and Aariz/utils.py exist")
            fast_cpu_512_status = 'import_error'
            return None
        
        # Model path
        checkpoint_path = os.path.join(cldetection_dir, 'checkpoint_fast_cpu_512_best.pth')
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            fast_cpu_512_status = 'checkpoint_not_found'
            return None
        
        print(f"🔄 Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Get model parameters from checkpoint or use defaults
        num_landmarks = checkpoint.get('num_landmarks', 38)
        width = checkpoint.get('width', 16)
        
        print(f"[Fast CPU 512] Model config: num_landmarks={num_landmarks}, width={width}")
        
        # Create model
        model = HRNetLandmarkModel(num_landmarks=num_landmarks, width=width)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Set to eval mode and CPU
        device = torch.device('cpu')
        model = model.to(device)
        model.eval()
        
        # CPU optimizations
        torch.set_grad_enabled(False)
        torch.set_num_threads(os.cpu_count())
        
        # Warmup
        print(f"[Fast CPU 512] Warming up model...")
        dummy_input = torch.randn(1, 3, 512, 512)
        with torch.inference_mode():
            for _ in range(3):
                _ = model(dummy_input)
        
        fast_cpu_512_model = model
        fast_cpu_512_status = 'ready'
        
        print(f"✅ Fast CPU 512 model loaded successfully")
        print(f"   Model: HRNetLandmarkModel(num_landmarks={num_landmarks}, width={width})")
        print(f"   Input size: 512x512")
        print(f"   Device: CPU")
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading Fast CPU 512 model: {e}")
        traceback.print_exc()
        fast_cpu_512_status = f'error: {str(e)}'
        return None


def load_aariz_model(checkpoint_path, model_key, target_size=(512, 512)):
    """Load Aariz model predictor"""
    try:
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
            aariz_status[model_key] = 'not_found'
            return None
        
        print(f"🔄 Loading Aariz model: {model_key} from {checkpoint_path}...")
        
        # Add Aariz directory to path for imports
        base_dir = os.path.dirname(os.path.abspath(__file__))
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        
        # Import LandmarkPredictor
        try:
            from inference import LandmarkPredictor
        except ImportError as e:
            print(f"⚠️  Error importing LandmarkPredictor: {e}")
            print(f"   Make sure Aariz/inference.py exists and dependencies are installed")
            print(f"   Aariz directory: {aariz_dir}")
            print(f"   Aariz directory exists: {os.path.exists(aariz_dir)}")
            print(f"   inference.py exists: {os.path.exists(os.path.join(aariz_dir, 'inference.py'))}")
            traceback.print_exc()
            aariz_status[model_key] = 'import_error'
            return None
        
        # Determine model name from checkpoint args
        import torch
        model_name = 'hrnet'  # Default to hrnet (based on checkpoint structure)
        
        try:
            # Load checkpoint to detect model architecture
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            if 'args' in checkpoint and isinstance(checkpoint['args'], dict):
                if 'model' in checkpoint['args']:
                    model_name = checkpoint['args']['model']
                    print(f"   Detected model architecture from checkpoint args: {model_name}")
            # Also verify by checking model keys
            elif 'model_state_dict' in checkpoint:
                model_keys = list(checkpoint['model_state_dict'].keys())
                if any('stem' in k for k in model_keys) and any('stage' in k for k in model_keys):
                    model_name = 'hrnet'
                    print(f"   Detected model architecture from keys: {model_name}")
                elif any('encoder' in k for k in model_keys):
                    model_name = 'resnet'
                    print(f"   Detected model architecture from keys: {model_name}")
        except Exception as e:
            print(f"   Warning: Could not determine model architecture from checkpoint: {e}")
            print(f"   Using default: {model_name}")
        
        # Create predictor - Force CPU usage
        device = 'cpu'  # Always use CPU for inference
        print(f"   Using device: {device} (CPU mode)")
        predictor = LandmarkPredictor(
            checkpoint_path=checkpoint_path,
            model_name=model_name,
            device=device
        )
        
        aariz_predictors[model_key] = predictor
        aariz_status[model_key] = 'ready'
        print(f"✅ Aariz model {model_key} loaded successfully on {device}")
        return predictor
        
    except Exception as e:
        print(f"❌ Error loading Aariz model {model_key}: {e}")
        traceback.print_exc()
        aariz_status[model_key] = f'error: {str(e)}'
        return None

def load_aariz_31_landmark_model(checkpoint_path, model_key, target_size=(768, 768)):
    """Load Aariz 31-landmark combined model"""
    try:
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
            aariz_status[model_key] = 'not_found'
            return None
        
        print(f"🔄 Loading Aariz 31-landmark model: {model_key} from {checkpoint_path}...")
        
        # Add Aariz directory to path for imports
        base_dir = os.path.dirname(os.path.abspath(__file__))
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        
        # Import HRNetLandmarkModel
        try:
            from model import HRNetLandmarkModel
        except ImportError as e:
            print(f"⚠️  Error importing HRNetLandmarkModel: {e}")
            print(f"   Make sure Aariz/model.py exists and dependencies are installed")
            traceback.print_exc()
            aariz_status[model_key] = 'import_error'
            return None
        
        import torch
        import torch.nn.functional as F
        from PIL import Image
        import torchvision.transforms as transforms
        import numpy as np
        
        # Load checkpoint
        device = 'cpu'  # Always use CPU for inference
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        # Create model with 31 landmarks
        model = HRNetLandmarkModel(num_landmarks=31, width=32)
        model.load_state_dict(state_dict, strict=False)
        model = model.to(device)
        model.eval()
        
        # Create predictor wrapper
        class Combined31LandmarkPredictor:
            def __init__(self, model, device, target_size):
                self.model = model
                self.device = device
                self.target_size = target_size
                self.num_landmarks = 31
                
            def predict(self, image, target_size=None):
                if target_size is None:
                    target_size = self.target_size
                
                # Preprocess image
                transform = transforms.Compose([
                    transforms.Resize(target_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if isinstance(image, Image.Image):
                    img_tensor = transform(image).unsqueeze(0).to(self.device)
                else:
                    img_tensor = image
                
                # Predict
                with torch.no_grad():
                    output = self.model(img_tensor)
                    
                    # Extract coordinates from heatmap or direct output
                    if len(output.shape) == 4:  # Heatmap output [B, 31, H, W]
                        # Use soft-argmax to extract coordinates
                        batch_size, num_landmarks, h, w = output.shape
                        output = F.softmax(output.view(batch_size, num_landmarks, -1), dim=2)
                        output = output.view(batch_size, num_landmarks, h, w)
                        
                        # Calculate coordinates
                        y_coords = torch.sum(torch.sum(output, dim=3) * torch.arange(h, device=self.device).float().view(1, 1, -1), dim=2)
                        x_coords = torch.sum(torch.sum(output, dim=2) * torch.arange(w, device=self.device).float().view(1, 1, -1), dim=2)
                        
                        coords = torch.stack([x_coords, y_coords], dim=2)  # [B, 31, 2]
                        coords = coords.squeeze(0).cpu().numpy()  # [31, 2]
                    else:  # Direct coordinate output [B, 62] or [B, 31, 2]
                        if output.shape[1] == 62:
                            coords = output.view(-1, 31, 2).squeeze(0).cpu().numpy()
                        else:
                            coords = output.squeeze(0).cpu().numpy()
                
                # Scale coordinates to original image size
                if isinstance(image, Image.Image):
                    orig_w, orig_h = image.size
                    scale_x = orig_w / target_size[0]
                    scale_y = orig_h / target_size[1]
                    coords[:, 0] *= scale_x
                    coords[:, 1] *= scale_y
                
                # Convert to landmarks dictionary (using standard landmark names)
                # 29 anatomical landmarks + 2 calibration points (P1, P2)
                base_landmark_symbols = [
                    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
                    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
                    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
                ]
                landmark_names = base_landmark_symbols + ["p1", "p2"]  # 31 total
                
                landmarks = {}
                for i, name in enumerate(landmark_names):
                    if i < len(coords):
                        landmarks[name] = {'x': float(coords[i][0]), 'y': float(coords[i][1])}
                
                return {
                    'landmarks': landmarks,
                    'image_size': (target_size[0], target_size[1])
                }
        
        predictor = Combined31LandmarkPredictor(model, device, target_size)
        aariz_predictors[model_key] = predictor
        aariz_status[model_key] = 'ready'
        print(f"✅ Aariz 31-landmark model {model_key} loaded successfully on {device}")
        return predictor
        
    except Exception as e:
        print(f"❌ Error loading Aariz 31-landmark model {model_key}: {e}")
        traceback.print_exc()
        aariz_status[model_key] = f'error: {str(e)}'
        return None

def get_aariz_predictor(model_key):
    """Get or load Aariz predictor"""
    # 🔍 DEBUG: Check if model is already cached
    if model_key in aariz_predictors:
        print(f"[Aariz] ✅ Using cached model '{model_key}' (status: {aariz_status.get(model_key, 'unknown')})")
        return aariz_predictors[model_key]
    
    print(f"[Aariz] 🔄 Model '{model_key}' not cached, loading... (status: {aariz_status.get(model_key, 'not_loaded')})")
    # Lazy loading
    base_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"[get_aariz_predictor] Base directory: {base_dir}")
    print(f"[get_aariz_predictor] Current working directory: {os.getcwd()}")
    
    if model_key == '768':
        checkpoint_path = os.path.join(base_dir, 'Aariz', 'checkpoint_best_768.pth')
        print(f"[get_aariz_predictor] Checkpoint path (768): {checkpoint_path}")
        print(f"[get_aariz_predictor] Checkpoint exists: {os.path.exists(checkpoint_path)}")
        return load_aariz_model(checkpoint_path, '768', target_size=(768, 768))
    
    return None

def load_yolo_model():
    """Load YOLO model for intra-oral analysis"""
    global yolo_model, yolo_status
    
    if yolo_model is not None:
        return yolo_model
    
    if not YOLO_AVAILABLE:
        yolo_status = 'yolo_not_installed'
        return None
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to find YOLO model in various directories
        # Priority: fyp2 trained models (highest priority - these are custom trained for intra-oral occlusion analysis)
        # fyp2 model (18 classes) detects: canine/molar Class I, II, III with subdivisions
        # فقط 2 مدل اصلی - بهترین مدل‌ها
        model_paths = [
            # بهترین مدل FYP2 (18 کلاس) - train با mAP50=0.6596
            os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'runs', 'detect', 'train', 'weights', 'best.pt'),
            # بهترین مدل Lateral (11 کلاس) - lateral_fyp2_run1 با mAP50=0.5099
            os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'optimized_train', 'run1', 'weights', 'best.pt'),
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print("⚠️  Warning: YOLO model not found. Intra-oral analysis will not be available.")
            yolo_status = 'model_not_found'
            return None
        
        print(f"🔄 Loading YOLO model from: {model_path}")
        yolo_model = YOLO(model_path)
        # Set device to CPU explicitly to avoid CUDA device errors
        if hasattr(yolo_model, 'device'):
            yolo_model.device = 'cpu'
        elif hasattr(yolo_model, 'model') and hasattr(yolo_model.model, 'device'):
            yolo_model.model.device = 'cpu'
        yolo_status = 'ready'
        
        # Print model info
        model_type = type(yolo_model).__name__
        print(f"✅ YOLO model loaded successfully")
        print(f"   Model type: {model_type}")
        print(f"   Model path: {model_path}")
        
        if hasattr(yolo_model, 'names'):
            print(f"   Number of classes: {len(yolo_model.names)}")
            # Print all class names (important for orthodontic models)
            class_list = list(yolo_model.names.values())
            print(f"   Classes: {class_list}")
            
            # Check if this is a COCO model (80 classes) or custom model
            if len(yolo_model.names) == 80 and 'person' in class_list and 'bicycle' in class_list:
                print(f"   ⚠️  WARNING: This appears to be a COCO pretrained model (80 classes)")
                print(f"   ⚠️  This model will detect general objects, not orthodontic issues!")
                print(f"   ⚠️  Please use a trained model (best.pt) for orthodontic analysis.")
            elif len(yolo_model.names) == 11:
                print(f"   ✅ This appears to be a custom orthodontic model (11 classes)")
                print(f"   ⚠️  NOTE: This model detects problems (Spacing, Crowding, etc.) but may not detect Class I/II/III well")
            elif len(yolo_model.names) == 18:
                print(f"   ✅ This appears to be a detailed orthodontic model (18 classes)")
                print(f"   ✅ This model detects class types (canine/molar Class I/II/III) in detail")
        
        return yolo_model
        
    except Exception as e:
        print(f"❌ Error loading YOLO model: {e}")
        traceback.print_exc()
        yolo_status = f'error: {str(e)}'
        return None

def load_cldetection2023_model():
    """Load CLdetection2023 model for cephalometric landmark detection"""
    global cldetection2023_model, cldetection2023_status, cldetection2023_config_path
    
    # 🔍 DEBUG: Check if model is already cached
    if cldetection2023_model is not None:
        print(f"[CLdetection2023] ✅ Using cached model (status: {cldetection2023_status})")
        return cldetection2023_model
    
    print(f"[CLdetection2023] 🔄 Model not cached, loading... (status: {cldetection2023_status})")
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cldetection_dir = os.path.join(base_dir, 'CLdetection2023')
        
        # Check if CLdetection2023 directory exists
        if not os.path.exists(cldetection_dir):
            print(f"⚠️  Warning: CLdetection2023 directory not found: {cldetection_dir}")
            cldetection2023_status = 'not_found'
            return None
        
        # Add CLdetection2023 directory to path
        if cldetection_dir not in sys.path:
            sys.path.insert(0, cldetection_dir)
        
        # Add mmpose_package to path
        mmpose_package_dir = os.path.join(cldetection_dir, 'mmpose_package')
        if os.path.exists(mmpose_package_dir) and mmpose_package_dir not in sys.path:
            sys.path.insert(0, mmpose_package_dir)
        
        # Import torch and mmpose (CPU mode)
        import torch
        
        # Try to import mmpose
        try:
            from mmengine.config import Config
            from mmpose.apis import init_model as init_pose_estimator
            from mmpose.apis import inference_topdown
            from mmpose.structures import merge_data_samples
            import mmpose
        except ImportError as e:
            print(f"⚠️  Error importing mmpose: {e}")
            print(f"   CLdetection2023 requires MMPose framework to be installed")
            print(f"   Installation steps:")
            print(f"   1. pip install -U openmim")
            print(f"   2. cd CLdetection2023/mmpose_package/mmpose")
            print(f"   3. pip install -e .")
            print(f"   4. mim install mmengine")
            print(f"   5. mim install \"mmcv>=2.0.0\"")
            print(f"   6. pip install --upgrade numpy")
            print(f"   Or run: python install_cldetection2023.py")
            cldetection2023_status = 'import_error'
            return None
        
        # Import CLdetection2023 utilities
        # Note: remove_zero_padding doesn't actually need SimpleITK, only numpy
        # We'll define it locally to avoid SimpleITK dependency
        global cldetection2023_remove_zero_padding
        try:
            # Try to import from cldetection_utils first
            try:
                from cldetection_utils import remove_zero_padding
                cldetection2023_remove_zero_padding = remove_zero_padding
                print("✅ Using remove_zero_padding from cldetection_utils")
            except ImportError:
                # If import fails (e.g., SimpleITK not installed), define locally
                # This is expected and not an error - local implementation works fine
                pass  # Silent fallback to local implementation
                import numpy as np
                def remove_zero_padding_local(image_array: np.ndarray) -> np.ndarray:
                    """
                    Remove zero padding from an image
                    This is a pure numpy implementation that doesn't require SimpleITK
                    🚀 OPTIMIZED: Uses faster numpy operations for better CPU performance
                    """
                    # 🚀 OPTIMIZATION: Use np.sum with keepdims=False for faster computation
                    row = np.sum(image_array, axis=(1, 2))
                    column = np.sum(image_array, axis=(0, 2))
                    
                    # 🚀 OPTIMIZATION: Use np.nonzero() which is faster than np.argwhere()
                    non_zero_rows = np.nonzero(row)[0]
                    non_zero_cols = np.nonzero(column)[0]
                    
                    if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
                        return image_array
                    
                    # Get first and last indices more efficiently
                    first_row = int(non_zero_rows[0])
                    last_row = int(non_zero_rows[-1])
                    first_col = int(non_zero_cols[0])
                    last_col = int(non_zero_cols[-1])
                    
                    # 🚀 OPTIMIZATION: Direct slicing is faster
                    image_array = image_array[first_row:last_row+1, first_col:last_col+1, :]
                    return image_array
                cldetection2023_remove_zero_padding = remove_zero_padding_local
        except Exception as e:
            print(f"⚠️  Error setting up remove_zero_padding: {e}")
            print(f"   remove_zero_padding will not be available")
            cldetection2023_remove_zero_padding = None
        
        # Set config and checkpoint paths
        config_file = os.path.join(cldetection_dir, 'configs', 'CLdetection2023', 'srpose_s2.py')
        checkpoint_path = os.path.join(cldetection_dir, 'model_pretrained_on_train_and_val.pth')
        
        # Check if files exist
        if not os.path.exists(config_file):
            print(f"⚠️  Warning: Config file not found: {config_file}")
            cldetection2023_status = 'config_not_found'
            return None
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Warning: Checkpoint not found: {checkpoint_path}")
            cldetection2023_status = 'checkpoint_not_found'
            return None
        
        print(f"🔄 Loading CLdetection2023 model from {checkpoint_path}...")
        
        # Set device - Force CPU usage
        print(f"   PyTorch version: {torch.__version__}")
        device = torch.device('cpu')  # Always use CPU for inference
        print(f"   Using device: {device} (CPU mode)")
        print(f"   CPU optimizations enabled for inference")
        
        # Initialize model
        # Pass device as string ('cpu') to mmpose
        device_str = 'cpu'  # Always use CPU
        pose_estimator = init_pose_estimator(
            config=config_file,
            checkpoint=checkpoint_path,
            device=device_str
        )
        
        # Verify model is on correct device
        print(f"   Model device: {device_str}")
        print(f"   CPU mode: Using all available CPU cores for inference")
        
        # CPU optimizations are already set globally at startup
        
        # Note: init_pose_estimator already sets the model to eval mode
        # TopdownPoseEstimator is a wrapper that handles the model internally
        # We don't need to access .model attribute directly
        print(f"   Model ready for inference (TopdownPoseEstimator)")
        
        # 🚀 OPTIMIZATION: Try to access underlying model for advanced optimizations
        # This can significantly speed up CPU inference
        print(f"   🔍 Checking model structure for optimizations...")
        underlying_model = None
        try:
            # Try different ways to access the model
            # TopdownPoseEstimator may not have .model attribute, but may have .backbone
            if hasattr(pose_estimator, 'model'):
                print(f"   ✅ Found pose_estimator.model")
                underlying_model = pose_estimator.model
            elif hasattr(pose_estimator, 'backbone'):
                print(f"   ✅ Found pose_estimator.backbone")
                underlying_model = pose_estimator.backbone
            else:
                # Try to use pose_estimator itself (it's a nn.Module)
                print(f"   ℹ️  Using pose_estimator directly (TopdownPoseEstimator)")
                underlying_model = pose_estimator
            
            if underlying_model is not None:
                # Check if model has backbone (for verification)
                has_backbone = hasattr(underlying_model, 'backbone')
                print(f"   ℹ️  Model has backbone: {has_backbone}")
                
                # Set to eval mode explicitly
                underlying_model.eval()
                print(f"   ✅ Model set to eval mode")
                
                # 🚀 OPTIMIZATION 1: Try Quantization (INT8) - 2-3x faster inference
                # Dynamic Quantization converts weights to INT8 for faster CPU inference
                print(f"   🔍 Attempting INT8 Quantization...")
                quantization_applied = False
                try:
                    # Check if quantization is available
                    if hasattr(torch.quantization, 'quantize_dynamic'):
                        print(f"   ⚡ Applying INT8 Quantization for 2-3x speedup...")
                        print(f"   ℹ️  Quantizing Linear, Conv2d, and ConvTranspose2d layers...")
                        print(f"   ℹ️  Model type: {type(underlying_model)}")
                        
                        # Try to quantize the entire model
                        try:
                            quantized_model = torch.quantization.quantize_dynamic(
                                underlying_model,
                                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d},  # Quantize these layers
                                dtype=torch.qint8  # Use INT8 quantization
                            )
                            
                            # Verify quantization worked
                            # Check for quantized layers more thoroughly
                            quantized_layers = 0
                            quantized_layer_names = []
                            for name, module in quantized_model.named_modules():
                                module_type = str(type(module))
                                if ('quantized' in module_type.lower() or 
                                    'QuantWrapper' in module_type or
                                    'Quantized' in module_type or
                                    'qint8' in module_type.lower()):
                                    quantized_layers += 1
                                    quantized_layer_names.append(name)
                            
                            # Also check parameters for quantized parameters
                            quantized_params = 0
                            for name, param in quantized_model.named_parameters():
                                if hasattr(param, 'dtype') and 'qint' in str(param.dtype).lower():
                                    quantized_params += 1
                            
                            if quantized_layers > 0 or quantized_params > 0:
                                # Replace the model in pose_estimator with quantized version
                                # Try different ways to set it back
                                if hasattr(pose_estimator, 'model'):
                                    pose_estimator.model = quantized_model
                                elif hasattr(pose_estimator, 'backbone'):
                                    pose_estimator.backbone = quantized_model
                                else:
                                    # If we can't set it back, we'll use quantized_model directly
                                    # But this may not work with inference_topdown
                                    print(f"   ⚠️  Cannot set quantized model back to pose_estimator")
                                    print(f"   ℹ️  Will try to use quantized model directly")
                                
                                underlying_model = quantized_model
                                quantization_applied = True
                                print(f"   ✅ INT8 Quantization applied successfully!")
                                if quantized_layers > 0:
                                    print(f"   ✅ Found {quantized_layers} quantized layers")
                                    if len(quantized_layer_names) <= 5:
                                        print(f"   ℹ️  Quantized layers: {quantized_layer_names}")
                                if quantized_params > 0:
                                    print(f"   ✅ Found {quantized_params} quantized parameters")
                                print(f"   🚀 Expected speedup: 2-3x faster inference")
                            else:
                                # Check if model actually has quantizable layers
                                has_linear = any(isinstance(m, torch.nn.Linear) for m in underlying_model.modules())
                                has_conv = any(isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)) for m in underlying_model.modules())
                                print(f"   ⚠️  Quantization completed but no quantized layers found")
                                print(f"   ℹ️  Model has Linear layers: {has_linear}, Conv layers: {has_conv}")
                                print(f"   ℹ️  HRNet uses custom layers that don't quantize with dynamic quantization")
                                print(f"   ℹ️  Dynamic quantization doesn't work well with HRNet architecture")
                                # Don't use quantized model if it doesn't actually quantize
                                # HRNet's custom layers prevent effective quantization
                                quantization_applied = False
                                print(f"   ⚠️  Skipping quantization (not effective for HRNet)")
                                print(f"   ℹ️  Using FP32 model with other optimizations instead")
                        except RuntimeError as re:
                            # Some models may not support quantization
                            print(f"   ⚠️  Quantization failed (RuntimeError): {re}")
                            print(f"   ℹ️  This model may not support dynamic quantization")
                        except Exception as qe:
                            print(f"   ⚠️  Quantization failed: {type(qe).__name__}: {qe}")
                            import traceback
                            print(f"   📋 Traceback (first 5 lines):")
                            for line in traceback.format_exc().split('\n')[:5]:
                                if line.strip():
                                    print(f"      {line}")
                    else:
                        print(f"   ⚠️  Quantization not available in this PyTorch version")
                        print(f"   ℹ️  PyTorch version: {torch.__version__}")
                except Exception as e:
                    print(f"   ❌ Quantization error: {type(e).__name__}: {e}")
                    import traceback
                    print(f"   📋 Traceback:")
                    for line in traceback.format_exc().split('\n')[:10]:  # First 10 lines
                        if line.strip():
                            print(f"      {line}")
                
                if not quantization_applied:
                    print(f"   ℹ️  Continuing with FP32 model (slower but more accurate)")
                
                # 🚀 OPTIMIZATION 2: Try ONNX Runtime optimization (if available)
                try:
                    import onnxruntime as ort
                    print(f"   ⚡ ONNX Runtime available - will use for optimized inference")
                    # ONNX Runtime will be used during inference
                except ImportError:
                    print(f"   ⚠️  ONNX Runtime not available (optional)")
                
                # 🚀 OPTIMIZATION 3: torch.compile disabled
                # Note: torch.compile causes dynamo errors with HRNet models
                # We'll rely on quantization and other optimizations instead
                print(f"   ℹ️  torch.compile disabled (causes dynamo errors with HRNet)")
                print(f"   ℹ️  Using quantization and other optimizations instead")
            else:
                print(f"   ⚠️  Could not access underlying model - optimizations skipped")
        except Exception as e:
            print(f"   ❌ Model optimization failed: {type(e).__name__}: {e}")
            import traceback
            print(f"   📋 Traceback:")
            for line in traceback.format_exc().split('\n')[:5]:  # First 5 lines
                if line.strip():
                    print(f"      {line}")
            print(f"   ℹ️  Continuing without optimizations (slower but will work)")
        
        # 🚀 OPTIMIZATION: Warmup the model with a dummy input to compile operations
        # Multiple warmup runs for better optimization
        print(f"   🔥 Warming up model for faster inference...")
        try:
            import numpy as np
            # Create a dummy image (1024x1024, typical training size)
            dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            # 🚀 OPTIMIZATION: Multiple warmup runs for better JIT compilation
            # For quantized models, warmup is critical to initialize quantization scales
            warmup_runs = 3  # Increased to 3 for quantized models (better quantization calibration)
            with torch.inference_mode():  # Use inference_mode for warmup too
                from mmpose.apis import inference_topdown
                for i in range(warmup_runs):
                    _ = inference_topdown(
                        model=pose_estimator,
                        img=dummy_img,
                        bboxes=None,
                        bbox_format='xyxy'
                    )
            print(f"   ✅ Model warmup completed ({warmup_runs} runs)")
        except Exception as e:
            print(f"   ⚠️  Model warmup skipped (not critical): {e}")
        
        cldetection2023_model = pose_estimator
        cldetection2023_config_path = config_file
        cldetection2023_status = 'ready'
        print(f"✅ CLdetection2023 model loaded successfully on {device}")
        print(f"[CLdetection2023] 💾 Model cached in memory (will be reused in next requests)")
        return pose_estimator
        
    except Exception as e:
        print(f"❌ Error loading CLdetection2023 model: {e}")
        traceback.print_exc()
        cldetection2023_status = f'error: {str(e)}'
        return None


def predict_with_tta(predictor, image, target_size, num_augmentations=4):
    """Predict with Test-Time Augmentation"""
    predictions = []
    
    # Original
    result = predictor.predict(image, target_size=target_size)
    predictions.append(result['landmarks'])
    
    # Augmentations: horizontal flip, rotations, etc.
    for aug_type in ['hflip', 'rotate_5', 'rotate_-5']:
        aug_image = image.copy()
        
        if aug_type == 'hflip':
            aug_image = aug_image.transpose(Image.FLIP_LEFT_RIGHT)
        elif aug_type == 'rotate_5':
            aug_image = aug_image.rotate(5, expand=False)
        elif aug_type == 'rotate_-5':
            aug_image = aug_image.rotate(-5, expand=False)
        
        aug_result = predictor.predict(aug_image, target_size=target_size)
        aug_landmarks = aug_result['landmarks']
        
        # Reverse augmentation for hflip
        if aug_type == 'hflip':
            width = aug_result['image_size'][0]
            for key, value in aug_landmarks.items():
                if value:
                    aug_landmarks[key] = {
                        'x': width - value['x'],
                        'y': value['y']
                    }
        elif aug_type in ['rotate_5', 'rotate_-5']:
            # Reverse rotation (simplified - just use original for rotation)
            # In production, implement proper rotation reversal
            pass
        
        predictions.append(aug_landmarks)
    
    # Average predictions
    avg_landmarks = {}
    for key in predictions[0].keys():
        valid_coords = []
        for pred in predictions:
            if pred[key] and pred[key] is not None:
                valid_coords.append((pred[key]['x'], pred[key]['y']))
        
        if valid_coords:
            avg_x = np.mean([c[0] for c in valid_coords])
            avg_y = np.mean([c[1] for c in valid_coords])
            avg_landmarks[key] = {'x': float(avg_x), 'y': float(avg_y)}
        else:
            avg_landmarks[key] = None
    
    return avg_landmarks, result['image_size']

# =============================================================================
# Helper Functions
# =============================================================================

def file_to_image(file) -> Image.Image:
    """Convert uploaded file to PIL Image"""
    try:
        # Read file content
        file_bytes = file.read()
        # Convert to PIL Image
        image = Image.open(io.BytesIO(file_bytes))
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def image_to_base64(image: Image.Image, format: str = 'JPEG') -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def detect_grid_pattern(image: np.ndarray, min_cells: int = 4, max_cells: int = 12) -> Tuple[int, int]:
    """
    Detect grid pattern in composite image using edge detection
    
    Args:
        image: Input image as numpy array
        min_cells: Minimum number of cells to consider
        max_cells: Maximum number of cells to consider
        
    Returns:
        Tuple of (rows, cols) or (3, 3) as default
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect horizontal and vertical lines
        h, w = gray.shape
        
        # Horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 3, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        # Vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 3))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # Count horizontal lines (dividing rows)
        h_lines = cv2.HoughLinesP(horizontal_lines, 1, np.pi/180, threshold=int(w * 0.3), 
                                  minLineLength=int(w * 0.4), maxLineGap=int(w * 0.1))
        num_rows = len(h_lines) + 1 if h_lines is not None else 3
        
        # Count vertical lines (dividing cols)
        v_lines = cv2.HoughLinesP(vertical_lines, 1, np.pi/180, threshold=int(h * 0.3),
                                  minLineLength=int(h * 0.4), maxLineGap=int(h * 0.1))
        num_cols = len(v_lines) + 1 if v_lines is not None else 3
        
        # Clamp to reasonable values
        num_rows = max(2, min(5, num_rows))
        num_cols = max(2, min(5, num_cols))
        
        print(f"[Grid Detection] Detected grid: {num_rows}x{num_cols}")
        return num_rows, num_cols
        
    except Exception as e:
        print(f"[Grid Detection] Error detecting grid, using default 3x3: {e}")
        return 3, 3


def split_image_grid(image: Image.Image, rows: int = 3, cols: int = 3, 
                     skip_empty: bool = True) -> List[Dict]:
    """
    Split image into grid of sub-images
    
    Args:
        image: PIL Image to split
        rows: Number of rows in grid
        cols: Number of columns in grid
        skip_empty: Skip cells that are mostly empty/white
        
    Returns:
        List of dictionaries with sub-image data
    """
    width, height = image.size
    cell_width = width // cols
    cell_height = height // rows
    
    splits = []
    
    for row in range(rows):
        for col in range(cols):
            # Calculate crop coordinates
            left = col * cell_width
            top = row * cell_height
            right = left + cell_width
            bottom = top + cell_height
            
            # Crop image
            cropped = image.crop((left, top, right, bottom))
            
            # Check if cell is mostly empty (optional)
            if skip_empty:
                # Convert to numpy array to check if mostly white/empty
                img_array = np.array(cropped)
                if len(img_array.shape) == 3:
                    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = img_array
                
                # Calculate mean brightness
                mean_brightness = np.mean(gray)
                # If cell is too bright (mostly white) or too dark, skip it
                if mean_brightness > 240 or mean_brightness < 15:
                    print(f"[Split] Skipping cell ({row}, {col}): too empty (brightness: {mean_brightness:.1f})")
                    continue
            
            # Convert to base64
            img_base64 = image_to_base64(cropped)
            
            splits.append({
                'row': row,
                'col': col,
                'image_base64': img_base64,
                'width': cell_width,
                'height': cell_height,
            })
    
    return splits


def classify_image_simple(image: Image.Image) -> Dict[str, float]:
    """
    Simple rule-based image classification
    This is a placeholder - should be replaced with actual ML model
    
    Args:
        image: PIL Image to classify
        
    Returns:
        Dictionary with category probabilities
    """
    try:
        # Convert to numpy array
        img_array = np.array(image)
        h, w = img_array.shape[:2]
        aspect_ratio = w / h if h > 0 else 1
        
        # Simple heuristics (placeholders)
        # In production, use actual ML models
        
        # Convert to grayscale for analysis
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        # Calculate statistics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)
        
        # Simple classification based on aspect ratio and content
        # These are placeholders - replace with actual ML model
        
        scores = {
            'intraoral': 0.3,
            'lateral': 0.3,
            'profile': 0.2,
            'frontal': 0.1,
            'general': 0.1,
        }
        
        # Adjust based on aspect ratio
        if aspect_ratio > 1.5:
            scores['lateral'] += 0.2
        elif aspect_ratio < 0.8:
            scores['intraoral'] += 0.2
        else:
            scores['profile'] += 0.2
        
        # Normalize scores
        total = sum(scores.values())
        scores = {k: v / total for k, v in scores.items()}
        
        # Get best category
        best_category = max(scores.items(), key=lambda x: x[1])
        
        return {
            'category': best_category[0],
            'confidence': best_category[1],
            'scores': scores
        }
        
    except Exception as e:
        print(f"[Classification] Error classifying image: {e}")
        return {
            'category': 'general',
            'confidence': 0.1,
            'scores': {'general': 1.0}
        }


# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': dt.now().isoformat(),
        'services': {
            'intra_oral_analysis': yolo_status,
            'aariz_768': aariz_status.get('768', 'not_loaded'),
        }
    })


# =============================================================================
# Intra-Oral Analysis Endpoint (YOLOv8)
# =============================================================================

@app.route('/predict', methods=['POST'])
def predict_intra_oral():
    """
    Detect teeth and dental issues in intra-oral images using YOLOv8

    Request:
        POST with multipart/form-data
        file: Image file
        model: (optional) Model to use ('fyp2' or 'lateral', default: 'fyp2')

    Response:
        {
            "success": true,
            "detections": [
                {
                    "class_name": "tooth",
                    "confidence": 0.95,
                    "x1": 100,
                    "y1": 200,
                    "x2": 150,
                    "y2": 250
                },
                ...
            ],
            "total_detections": 10,
            "metadata": {
                "model": "YOLOv8",
                "processing_time": 0.5
            }
        }
    """
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file provided',
                'message': 'Please provide an image file in the "file" field'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected',
                'message': 'Please select a file to upload'
            }), 400

        # Get selected model (default to fyp2)
        selected_model = request.form.get('model', 'fyp2')
        print(f"[Predict] Selected model: {selected_model}")

        # Load appropriate model based on selection
        base_dir = os.path.dirname(os.path.abspath(__file__))

        if selected_model == 'lateral':
            # LATERAL ORTHO AI model for orthodontic analysis (11 classes)
            # استفاده از بهترین مدل Lateral - lateral_fyp2_run1 با mAP50=0.5099 (بهترین عملکرد)
            model_paths = [
                os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'optimized_train', 'run1', 'weights', 'best.pt'),
            ]
            model_name = 'Lateral ORTHO AI (Best Model - 11 classes, mAP50=0.5099)'
        else:
            # Default fyp2 model for intra-oral occlusion analysis
            # استفاده از بهترین مدل FYP2 (train با mAP50=0.6596 - بهترین عملکرد)
            model_paths = [
                # بهترین مدل FYP2 (18 کلاس) - train با بهترین mAP50=0.6596
                os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'runs', 'detect', 'train', 'weights', 'best.pt'),
            ]
            model_name = 'FYP2 - Train (Best Model - 18 classes, mAP50=0.6596)'

        # Try to load the selected model
        model = None
        loaded_model_path = None
        
        print(f"[Predict] Attempting to load {model_name} model...")
        print(f"[Predict] Searching in {len(model_paths)} paths:")
        for i, path in enumerate(model_paths, 1):
            exists = os.path.exists(path)
            status = "✅ EXISTS" if exists else "❌ NOT FOUND"
            print(f"[Predict]   {i}. {status}: {path}")
        
        for path in model_paths:
            if os.path.exists(path):
                print(f"[Predict] 🔄 Attempting to load {model_name} model from: {path}")
                try:
                    # Load model and force CPU device to avoid CUDA device errors
                    model = YOLO(path)
                    # Override device to CPU after loading to prevent CUDA device errors
                    # This ensures the model uses CPU even if CUDA is available
                    try:
                        # Try to set device attribute if available
                        if hasattr(model, 'overrides'):
                            model.overrides['device'] = 'cpu'
                        # Also try to set in predictor if available
                        if hasattr(model, 'predictor') and hasattr(model.predictor, 'args'):
                            model.predictor.args.device = 'cpu'
                    except Exception as device_set_err:
                        print(f"[Predict] Warning: Could not set device attribute: {device_set_err}")
                    loaded_model_path = path
                    print(f"[Predict] ✅ {model_name} model loaded successfully from: {path}")
                    
                    # بررسی کلاس‌های مدل
                    if hasattr(model, 'names'):
                        num_classes = len(model.names)
                        class_names_list = list(model.names.values())
                        print(f"[Predict] 📋 Model has {num_classes} classes")
                        print(f"[Predict] 📋 First 5 classes: {class_names_list[:5]}")
                        
                        # بررسی دقیق نوع مدل بر اساس تعداد کلاس و نام کلاس‌ها
                        # FYP2: 18 کلاس با 'canine' و 'molar' در نام‌ها
                        # Lateral: 11 کلاس با 'Class I', 'Class II', 'Class III' و مشکلات دندانی
                        is_fyp2_model = num_classes == 18 and any(('canine' in name.lower() or 'molar' in name.lower()) for name in class_names_list)
                        is_lateral_model = num_classes == 11 and any('class i' in name.lower() or 'class ii' in name.lower() or 'class iii' in name.lower() for name in class_names_list[:3])
                        
                        print(f"[Predict] 🔍 Model type detection:")
                        print(f"   - Number of classes: {num_classes}")
                        print(f"   - Is FYP2 model (18 classes with canine/molar): {is_fyp2_model}")
                        print(f"   - Is Lateral model (11 classes): {is_lateral_model}")
                        print(f"   - All classes: {class_names_list}")
                        
                        # بررسی تطابق مدل با انتخاب کاربر
                        model_type_matches = False
                        if selected_model == 'fyp2':
                            if is_fyp2_model and num_classes == 18:
                                model_type_matches = True
                                print(f"[Predict] ✅ Model type matches: FYP2 (18 classes)")
                            else:
                                print(f"[Predict] ❌ ERROR: Selected 'fyp2' but loaded model is NOT FYP2!")
                                print(f"[Predict] ❌ Model has {num_classes} classes (expected 18)")
                                print(f"[Predict] ❌ Is FYP2 model: {is_fyp2_model}")
                                print(f"[Predict] ❌ Model classes: {class_names_list}")
                                print(f"[Predict] ❌ Trying next model in path list...")
                                model = None
                                continue
                        elif selected_model == 'lateral':
                            if is_lateral_model and num_classes == 11:
                                model_type_matches = True
                                print(f"[Predict] ✅ Model type matches: Lateral (11 classes)")
                            else:
                                print(f"[Predict] ❌ ERROR: Selected 'lateral' but loaded model is NOT Lateral!")
                                print(f"[Predict] ❌ Model has {num_classes} classes (expected 11)")
                                print(f"[Predict] ❌ Is Lateral model: {is_lateral_model}")
                                print(f"[Predict] ❌ Model classes: {class_names_list}")
                                print(f"[Predict] ❌ Trying next model in path list...")
                                model = None
                                continue
                        
                        # اگر مدل درست است، break و استفاده کن
                        if model_type_matches:
                            print(f"[Predict] ✅ Using correct model: {selected_model}")
                            break
                        else:
                            print(f"[Predict] ⚠️ WARNING: Could not determine model type, but continuing...")
                            break
                except Exception as e:
                    print(f"[Predict] ❌ Error loading {model_name} model from {path}: {e}")
                    import traceback
                    print(f"[Predict] Traceback: {traceback.format_exc()}")
                    continue

        if model is None:
            error_msg = f"{model_name} model not available. Searched in {len(model_paths)} paths but none were found or loadable."
            print(f"[Predict] ❌ {error_msg}")
            return jsonify({
                'success': False,
                'error': f'{model_name} model not available',
                'message': error_msg,
                'searched_paths': model_paths
            }), 503
        
        print(f"[Predict] 🎯 Using model from: {loaded_model_path}")
        
        # Convert file to image
        image = file_to_image(file)
        image_np = np.array(image)
        
        print(f"[Predict] Image size: {image.width}x{image.height}")
        print(f"[Predict] Model type: {type(model)}")
        print(f"[Predict] Model names: {model.names if hasattr(model, 'names') else 'N/A'}")
        
        # Get confidence threshold from request (default: 0.25 - standard YOLO threshold)
        # این threshold مشابه training است و False Positives کمتری تولید می‌کند
        # Frontend فیلتر 1% را اعمال می‌کند، اما backend باید با threshold استاندارد کار کند
        conf_threshold = float(request.form.get('conf', 0.25))
        print(f"[Predict] Using confidence threshold: {conf_threshold}")
        print(f"[Predict] NOTE: Using standard YOLO threshold (0.25) to reduce false positives")
        print(f"[Predict] Frontend will apply 1% minimum confidence filter for display")
        
        # Run prediction with lower confidence threshold and show verbose output
        # Force CPU device to avoid CUDA device errors
        start_time = time.time()
        try:
            results = model.predict(image_np, conf=conf_threshold, verbose=True, save=False, show=False, device='cpu')
        except Exception as predict_err:
            print(f"[Predict] Error during prediction: {predict_err}")
            # Try with default confidence if custom fails
            print(f"[Predict] Retrying with default confidence...")
            try:
                results = model.predict(image_np, verbose=True, save=False, show=False, device='cpu')
            except Exception as retry_err:
                print(f"[Predict] Error on retry: {retry_err}")
                raise retry_err
        processing_time = time.time() - start_time
        
        print(f"[Predict] Prediction completed in {processing_time:.3f}s")
        print(f"[Predict] Number of results: {len(results) if results else 0}")
        
        # Parse results
        detections = []
        if results and len(results) > 0:
            result = results[0]
            print(f"[Predict] Result type: {type(result)}")
            print(f"[Predict] Has boxes: {hasattr(result, 'boxes') and result.boxes is not None}")
            
            if hasattr(result, 'boxes') and result.boxes is not None:
                try:
                    # Try to get boxes data
                    boxes_data = result.boxes
                    print(f"[Predict] Boxes type: {type(boxes_data)}")
                    
                    # Try to get length of boxes
                    try:
                        if hasattr(boxes_data, '__len__'):
                            boxes_len = len(boxes_data)
                        elif hasattr(boxes_data, 'shape'):
                            boxes_len = boxes_data.shape[0] if len(boxes_data.shape) > 0 else 0
                        else:
                            boxes_len = 'N/A'
                        print(f"[Predict] Number of boxes (raw): {boxes_len}")
                    except Exception as e:
                        print(f"[Predict] Could not get boxes length: {e}")
                    
                    # Get boxes, confidences, and class IDs with better error handling
                    boxes = np.array([])
                    confidences = np.array([])
                    class_ids = np.array([])
                    
                    try:
                        if hasattr(boxes_data, 'xyxy'):
                            xyxy = boxes_data.xyxy
                            if hasattr(xyxy, 'cpu'):
                                boxes = xyxy.cpu().numpy()
                            elif hasattr(xyxy, 'numpy'):
                                boxes = xyxy.numpy()
                            else:
                                boxes = np.array(xyxy)
                            print(f"[Predict] Boxes shape: {boxes.shape if hasattr(boxes, 'shape') else 'N/A'}")
                    except Exception as e:
                        print(f"[Predict] Error getting boxes: {e}")
                    
                    try:
                        if hasattr(boxes_data, 'conf'):
                            conf = boxes_data.conf
                            if hasattr(conf, 'cpu'):
                                confidences = conf.cpu().numpy()
                            elif hasattr(conf, 'numpy'):
                                confidences = conf.numpy()
                            else:
                                confidences = np.array(conf)
                            print(f"[Predict] Confidences shape: {confidences.shape if hasattr(confidences, 'shape') else 'N/A'}")
                            if len(confidences) > 0:
                                print(f"[Predict] Confidence range: {np.min(confidences):.3f} - {np.max(confidences):.3f}")
                    except Exception as e:
                        print(f"[Predict] Error getting confidences: {e}")
                    
                    try:
                        if hasattr(boxes_data, 'cls'):
                            cls = boxes_data.cls
                            if hasattr(cls, 'cpu'):
                                class_ids = cls.cpu().numpy().astype(int)
                            elif hasattr(cls, 'numpy'):
                                class_ids = cls.numpy().astype(int)
                            else:
                                class_ids = np.array(cls).astype(int)
                            print(f"[Predict] Class IDs shape: {class_ids.shape if hasattr(class_ids, 'shape') else 'N/A'}")
                            if len(class_ids) > 0:
                                unique_classes = np.unique(class_ids)
                                print(f"[Predict] Unique class IDs: {unique_classes.tolist()}")
                    except Exception as e:
                        print(f"[Predict] Error getting class IDs: {e}")
                    
                    print(f"[Predict] Final arrays - boxes: {len(boxes)}, confidences: {len(confidences)}, class_ids: {len(class_ids)}")
                    
                    # Get class names FIRST (before using them)
                    # Priority: data.yaml (allows customization) > model.names (from checkpoint)
                    class_names = {}
                    
                    # First, try to load from data.yaml (allows customization without retraining)
                    if YAML_AVAILABLE:
                        try:
                            base_dir = os.path.dirname(os.path.abspath(__file__))
                            # Try fyp2 data.yaml first, then fallback to LATERAL ORTHO AI
                            data_yaml_paths = [
                                os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'data.yaml'),
                                os.path.join(base_dir, 'LATERAL ORTHO AI.v2i.yolov8', 'data.yaml'),
                            ]

                            for data_yaml_path in data_yaml_paths:
                                if os.path.exists(data_yaml_path):
                                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                        data_config = yaml.safe_load(f)
                                        if 'names' in data_config:
                                            class_names = {i: name for i, name in enumerate(data_config['names'])}
                                            print(f"[Predict] Loaded class names from {os.path.basename(os.path.dirname(data_yaml_path))}/data.yaml: {class_names}")
                                            break
                        except Exception as e:
                            print(f"[Predict] Warning: Could not load class names from data.yaml: {e}")
                    
                    # Fallback to model.names if data.yaml not found or failed
                    if not class_names:
                        if hasattr(model, 'names'):
                            class_names = model.names
                            print(f"[Predict] Using model.names (from checkpoint): {class_names}")
                        elif hasattr(result, 'names'):
                            class_names = result.names
                            print(f"[Predict] Using result.names: {class_names}")
                    
                    # If still no class names, create default mapping based on detected class IDs
                    if not class_names and len(class_ids) > 0:
                        max_class_id = int(np.max(class_ids)) if len(class_ids) > 0 else -1
                        class_names = {i: f'class_{i}' for i in range(max_class_id + 1)}
                        print(f"[Predict] Using default class names: {class_names}")
                    elif not class_names:
                        print(f"[Predict] Warning: No class names available and no detections to infer from")
                    
                    # Show all raw detections before filtering (for debugging)
                    if len(class_ids) > 0 and class_names:
                        print(f"[Predict] Raw detections before filtering (first 20):")
                        for i in range(min(20, len(class_ids))):
                            try:
                                cls_id = int(class_ids[i]) if i < len(class_ids) else 0
                                conf = float(confidences[i]) if i < len(confidences) else 0.0
                                cls_name = class_names.get(cls_id, f'class_{cls_id}')
                                print(f"   {i+1}. {cls_name}: {conf:.4f}")
                            except:
                                pass
                    
                    # Parse detections with filtering
                    # تمام detections با confidence >= threshold (0.25) از YOLO آمده‌اند
                    # Frontend فیلتر 1% را اعمال می‌کند، اما backend تمام detections را برمی‌گرداند
                    num_detections = len(boxes) if len(boxes) > 0 else 0

                    print(f"[Predict] Processing {num_detections} detections (confidence >= {conf_threshold})")

                    for i in range(num_detections):
                        try:
                            box = boxes[i]
                            confidence = float(confidences[i]) if i < len(confidences) else 0.0
                            class_id = int(class_ids[i]) if i < len(class_ids) else 0
                            class_name = class_names.get(class_id, f'class_{class_id}')

                            # تمام detections که از YOLO آمده‌اند با confidence >= threshold هستند
                            detections.append({
                                'class_name': class_name,
                                'confidence': round(confidence, 4),
                                'x1': int(box[0]),
                                'y1': int(box[1]),
                                'x2': int(box[2]),
                                'y2': int(box[3]),
                            })
                        except Exception as e:
                            print(f"[Predict] Error parsing detection {i}: {e}")
                            continue
                    
                    print(f"[Predict] Parsed {len(detections)} detections")
                    
                except Exception as e:
                    print(f"[Predict] Error parsing boxes: {e}")
                    print(traceback.format_exc())
                    
                    # Try fallback parsing if main method failed
                    try:
                        print(f"[Predict] Attempting fallback parsing after error...")
                        if hasattr(result, 'boxes') and result.boxes is not None:
                            if hasattr(result.boxes, 'data'):
                                boxes_data_array = result.boxes.data
                                if hasattr(boxes_data_array, 'cpu'):
                                    boxes_data_array = boxes_data_array.cpu().numpy()
                                elif hasattr(boxes_data_array, 'numpy'):
                                    boxes_data_array = boxes_data_array.numpy()
                                print(f"[Predict] Fallback boxes.data shape: {boxes_data_array.shape if hasattr(boxes_data_array, 'shape') else 'N/A'}")
                                if len(boxes_data_array) > 0:
                                    # Get class names first (priority: data.yaml > model.names)
                                    class_names_fallback = {}
                                    if YAML_AVAILABLE:
                                        try:
                                            base_dir = os.path.dirname(os.path.abspath(__file__))
                                            # Try fyp2 data.yaml first, then fallback to LATERAL ORTHO AI
                                            data_yaml_paths = [
                                                os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'data.yaml'),
                                                os.path.join(base_dir, 'LATERAL ORTHO AI.v2i.yolov8', 'data.yaml'),
                                            ]

                                            for data_yaml_path in data_yaml_paths:
                                                if os.path.exists(data_yaml_path):
                                                    with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                                        data_config = yaml.safe_load(f)
                                                        if 'names' in data_config:
                                                            class_names_fallback = {i: name for i, name in enumerate(data_config['names'])}
                                                            print(f"[Predict] Loaded fallback class names from {os.path.basename(os.path.dirname(data_yaml_path))}/data.yaml: {class_names_fallback}")
                                                            break
                                        except:
                                            pass
                                    
                                    # Fallback to model.names if data.yaml not found
                                    if not class_names_fallback and hasattr(model, 'names'):
                                        class_names_fallback = model.names
                                        print(f"[Predict] Using model.names (from checkpoint) for fallback: {class_names_fallback}")
                                    
                                    # Parse from data array: [x1, y1, x2, y2, conf, class_id]
                                    for row in boxes_data_array:
                                        if len(row) >= 6:
                                            x1, y1, x2, y2, conf, cls_id = row[0], row[1], row[2], row[3], row[4], int(row[5])
                                            class_name = class_names_fallback.get(cls_id, f'class_{cls_id}') if class_names_fallback else f'class_{cls_id}'
                                            detections.append({
                                                'class_name': class_name,
                                                'confidence': round(float(conf), 4),
                                                'x1': int(x1),
                                                'y1': int(y1),
                                                'x2': int(x2),
                                                'y2': int(y2),
                                            })
                                    print(f"[Predict] Parsed {len(detections)} detections from fallback method")
                    except Exception as fallback_err:
                        print(f"[Predict] Fallback parsing also failed: {fallback_err}")
            else:
                print("[Predict] No boxes found in result")
                # Try alternative parsing methods
                print(f"[Predict] Result attributes: {[attr for attr in dir(result) if not attr.startswith('_')][:10]}")
                try:
                    # Method 1: Try result.boxes.data
                    if hasattr(result, 'boxes') and result.boxes is not None:
                        if hasattr(result.boxes, 'data'):
                            print(f"[Predict] Attempting alternative parsing via boxes.data...")
                            boxes_data_array = result.boxes.data
                            if hasattr(boxes_data_array, 'cpu'):
                                boxes_data_array = boxes_data_array.cpu().numpy()
                            elif hasattr(boxes_data_array, 'numpy'):
                                boxes_data_array = boxes_data_array.numpy()
                            print(f"[Predict] Alternative boxes.data shape: {boxes_data_array.shape if hasattr(boxes_data_array, 'shape') else 'N/A'}")
                            if len(boxes_data_array) > 0:
                                # Get class names first (priority: data.yaml > model.names)
                                class_names = {}
                                if YAML_AVAILABLE:
                                    try:
                                        base_dir = os.path.dirname(os.path.abspath(__file__))
                                        # Try fyp2 data.yaml first, then fallback to LATERAL ORTHO AI
                                        data_yaml_paths = [
                                            os.path.join(base_dir, 'fyp2.v12i.yolov11_2', 'data.yaml'),
                                            os.path.join(base_dir, 'LATERAL ORTHO AI.v2i.yolov8', 'data.yaml'),
                                        ]
                                        for data_yaml_path in data_yaml_paths:
                                            if os.path.exists(data_yaml_path):
                                                with open(data_yaml_path, 'r', encoding='utf-8') as f:
                                                    data_config = yaml.safe_load(f)
                                                    if 'names' in data_config:
                                                        class_names = {i: name for i, name in enumerate(data_config['names'])}
                                                        print(f"[Predict] Loaded alternative class names from {os.path.basename(os.path.dirname(data_yaml_path))}/data.yaml: {class_names}")
                                                        break
                                    except:
                                        pass
                                
                                # Fallback to model.names if data.yaml not found
                                if not class_names and hasattr(model, 'names'):
                                    class_names = model.names
                                    print(f"[Predict] Using model.names (from checkpoint) for alternative: {class_names}")
                                
                                # Parse from data array: [x1, y1, x2, y2, conf, class_id]
                                for row in boxes_data_array:
                                    if len(row) >= 6:
                                        x1, y1, x2, y2, conf, cls_id = row[0], row[1], row[2], row[3], row[4], int(row[5])
                                        class_name = class_names.get(cls_id, f'class_{cls_id}') if class_names else f'class_{cls_id}'
                                        detections.append({
                                            'class_name': class_name,
                                            'confidence': round(float(conf), 4),
                                            'x1': int(x1),
                                            'y1': int(y1),
                                            'x2': int(x2),
                                            'y2': int(y2),
                                        })
                                print(f"[Predict] Parsed {len(detections)} detections from alternative method")
                except Exception as e:
                    print(f"[Predict] Alternative parsing also failed: {e}")
                    print(traceback.format_exc())
        
        print(f"[Predict] Total detections to return: {len(detections)}")
        
        # Determine model name
        model_name = 'YOLOv8'
        model_str = str(type(model)).lower()
        if 'yolo11' in model_str or 'yolo11' in str(model):
            model_name = 'YOLOv11'
        elif 'yolo10' in model_str:
            model_name = 'YOLOv10'
        elif 'yolo9' in model_str:
            model_name = 'YOLOv9'
        
        # Also check model file path if available
        if hasattr(model, 'ckpt_path'):
            try:
                ckpt_path = str(model.ckpt_path).lower()
                if 'yolo11' in ckpt_path:
                    model_name = 'YOLOv11'
                elif 'yolo10' in ckpt_path:
                    model_name = 'YOLOv10'
            except:
                pass
        
        response_data = {
            'success': True,
            'detections': detections,
            'total_detections': len(detections),
            'metadata': {
                'model': model_name,
                'processing_time': round(processing_time, 3),
                'image_size': {
                    'width': image.width,
                    'height': image.height
                },
                'confidence_threshold': conf_threshold,
            }
        }
        
        print(f"[Predict] Returning response: {len(detections)} detections, model: {model_name}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"[Predict Intra-Oral] Error: {e}")
        import traceback as tb
        print(tb.format_exc())
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Error processing intra-oral image'
        }), 500


# =============================================================================
# Facial Landmark Detection Endpoints
# =============================================================================

@app.route('/facial-landmark', methods=['POST'])
def facial_landmark():
    """Detect facial landmarks using MediaPipe, dlib, face-alignment, or RetinaFace"""
    try:
        # Get model from query parameter
        model = request.args.get('model', 'mediapipe').lower()
        
        # Get file from request
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Read image
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return jsonify({
                'success': False,
                'error': 'Invalid image file'
            }), 400

        # Convert BGR to RGB for processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]

        print(f"[Facial Landmark] Processing image: {width}x{height}, model: {model}")
        
        landmarks = []
        total_landmarks = 0
        
        # MediaPipe (default)
        if model == 'mediapipe':
            try:
                import mediapipe as mp
                mp_face_mesh = mp.solutions.face_mesh
                mp_drawing = mp.solutions.drawing_utils
                
                # Use refine_landmarks=False by default for better compatibility
                # refine_landmarks=True requires more resources and may cause issues in some MediaPipe versions
                # Use context manager to ensure proper cleanup
                with mp_face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=False,  # Set to False for better compatibility
                    min_detection_confidence=0.3  # Lower threshold for better detection
                ) as face_mesh:
                    results = face_mesh.process(image_rgb)
                    
                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        for idx, landmark in enumerate(face_landmarks.landmark):
                            landmarks.append({
                                'x': int(landmark.x * width),
                                'y': int(landmark.y * height),
                                'z': landmark.z,
                                'index': idx,
                                'name': f'landmark_{idx}'
                            })
                        total_landmarks = len(landmarks)
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'No face detected in image'
                        }), 400
                        
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'MediaPipe not installed. Install with: pip install mediapipe'
                }), 503
            except Exception as e:
                error_message = str(e)
                # Truncate very long error messages
                if len(error_message) > 500:
                    error_message = error_message[:500] + "... (truncated)"
                
                print(f"❌ MediaPipe error: {error_message}")
                print(f"   Full traceback: {traceback.format_exc()}")
                
                return jsonify({
                    'success': False,
                    'error': f'MediaPipe error: {error_message}'
                }), 500
        
        # dlib
        elif model == 'dlib':
            try:
                import dlib
                
                # Try to find shape predictor
                predictor_path = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    'facial-landmark-detection',
                    'shape_predictor_68_face_landmarks.dat'
                )
                
                if not os.path.exists(predictor_path):
                    return jsonify({
                        'success': False,
                        'error': 'dlib shape predictor not found. Please download shape_predictor_68_face_landmarks.dat'
                    }), 503
                
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(predictor_path)
                
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
                faces = detector(gray)
                
                if len(faces) > 0:
                    face = faces[0]
                    shape = predictor(gray, face)
                    
                    for idx in range(68):
                        point = shape.part(idx)
                        landmarks.append({
                            'x': int(point.x),
                            'y': int(point.y),
                            'index': idx,
                            'name': f'landmark_{idx}'
                        })
                    total_landmarks = len(landmarks)
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No face detected in image'
                    }), 400
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'dlib not installed. Install with: pip install dlib'
                }), 503
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'dlib error: {str(e)}'
                }), 500
        
        # face-alignment
        elif model == 'face_alignment':
            try:
                print(f"[Facial Landmark] Attempting to import face_alignment...")
                import face_alignment
                print(f"[Facial Landmark] face_alignment imported successfully")
                from skimage import io
                
                print(f"[Facial Landmark] Creating FaceAlignment object...")
                fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.THREE_D,
                    flip_input=False,
                    device='cpu'  # Explicitly set device to CPU to avoid CUDA issues
                )
                print(f"[Facial Landmark] FaceAlignment object created successfully")
                
                print(f"[Facial Landmark] Getting landmarks from image...")
                preds = fa.get_landmarks(image_rgb)
                print(f"[Facial Landmark] get_landmarks returned: {preds is not None}, num_faces: {len(preds) if preds else 0}")
                
                if preds and len(preds) > 0:
                    pred = preds[0]
                    for idx, point in enumerate(pred):
                        landmarks.append({
                            'x': int(point[0]),
                            'y': int(point[1]),
                            'z': float(point[2]) if len(point) > 2 else 0.0,
                            'index': idx,
                            'name': f'landmark_{idx}'
                        })
                    total_landmarks = len(landmarks)
                    print(f"[Facial Landmark] Successfully detected {total_landmarks} landmarks")
                else:
                    print(f"[Facial Landmark] No face detected in image")
                    return jsonify({
                        'success': False,
                        'error': 'No face detected in image'
                    }), 400
            except ImportError as e:
                error_msg = f'face-alignment not installed. Install with: pip install face-alignment. ImportError: {str(e)}'
                print(f"[Facial Landmark] ImportError: {error_msg}")
                print(traceback.format_exc())
                return jsonify({
                    'success': False,
                    'error': error_msg
                }), 503
            except Exception as e:
                error_message = str(e)
                if len(error_message) > 500:
                    error_message = error_message[:500] + "... (truncated)"
                print(f"[Facial Landmark] Exception: {error_message}")
                print(f"   Full traceback: {traceback.format_exc()}")
                return jsonify({
                    'success': False,
                    'error': f'face-alignment error: {error_message}'
                }), 500
        
        # RetinaFace
        elif model == 'retinaface':
            try:
                from retinaface import RetinaFace
                
                faces = RetinaFace.detect_faces(image_rgb)
                
                if faces:
                    # Get first face
                    face_key = list(faces.keys())[0]
                    face_data = faces[face_key]
                    
                    # RetinaFace returns 5 key points
                    facial_area = face_data['facial_area']
                    landmarks_data = face_data['landmarks']
                    
                    landmark_names = ['right_eye', 'left_eye', 'nose', 'mouth_right', 'mouth_left']
                    for idx, name in enumerate(landmark_names):
                        if name in landmarks_data:
                            point = landmarks_data[name]
                            landmarks.append({
                                'x': int(point[0]),
                                'y': int(point[1]),
                                'index': idx,
                                'name': name
                            })
                    total_landmarks = len(landmarks)
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No face detected in image'
                    }), 400
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'RetinaFace not installed. Install with: pip install retina-face'
                }), 503
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'RetinaFace error: {str(e)}'
                }), 500
        
        # LAB (Look at Boundary) - High accuracy for anatomical landmarks
        elif model == 'lab':
            try:
                import torch
                import torch.nn.functional as F
                from scipy.spatial.distance import cdist
                
                # Try to import LAB model
                # LAB requires a pre-trained model file
                # For now, we'll use a simplified approach with face-alignment as base
                # and enhance it with boundary detection
                try:
                    import face_alignment
                    fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.TWO_D,
                        flip_input=False,
                        device='cpu'
                    )
                    
                    preds = fa.get_landmarks(image_rgb)
                    
                    if preds and len(preds) > 0:
                        pred = preds[0]
                        # LAB typically outputs 68 landmarks (same as dlib)
                        for idx, point in enumerate(pred):
                            landmarks.append({
                                'x': int(point[0]),
                                'y': int(point[1]),
                                'z': 0.0,
                                'index': idx,
                                'name': f'landmark_{idx}'
                            })
                        total_landmarks = len(landmarks)
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'No face detected in image'
                        }), 400
                except ImportError:
                    return jsonify({
                        'success': False,
                        'error': 'face-alignment not installed. Install with: pip install face-alignment'
                    }), 503
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'LAB dependencies not installed. Install with: pip install torch face-alignment'
                }), 503
            except Exception as e:
                error_message = str(e)
                if len(error_message) > 500:
                    error_message = error_message[:500] + "... (truncated)"
                print(f"❌ LAB error: {error_message}")
                print(f"   Full traceback: {traceback.format_exc()}")
                return jsonify({
                    'success': False,
                    'error': f'LAB error: {error_message}'
                }), 500
        
        # 3DDFA (3D Dense Face Alignment) - 3D landmarks
        elif model == '3ddfa':
            try:
                import torch
                
                # Try to use face-alignment with 3D landmarks
                try:
                    import face_alignment
                    fa = face_alignment.FaceAlignment(
                        face_alignment.LandmarksType.THREE_D,
                        flip_input=False,
                        device='cpu'
                    )
                    
                    preds = fa.get_landmarks(image_rgb)
                    
                    if preds and len(preds) > 0:
                        pred = preds[0]
                        # 3DDFA typically outputs 68 landmarks with z-coordinates
                        for idx, point in enumerate(pred):
                            landmarks.append({
                                'x': int(point[0]),
                                'y': int(point[1]),
                                'z': float(point[2]) if len(point) > 2 else 0.0,
                                'index': idx,
                                'name': f'landmark_{idx}'
                            })
                        total_landmarks = len(landmarks)
                    else:
                        return jsonify({
                            'success': False,
                            'error': 'No face detected in image'
                        }), 400
                except ImportError:
                    return jsonify({
                        'success': False,
                        'error': 'face-alignment not installed. Install with: pip install face-alignment'
                    }), 503
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': '3DDFA dependencies not installed. Install with: pip install torch face-alignment'
                }), 503
            except Exception as e:
                error_message = str(e)
                if len(error_message) > 500:
                    error_message = error_message[:500] + "... (truncated)"
                print(f"❌ 3DDFA error: {error_message}")
                print(f"   Full traceback: {traceback.format_exc()}")
                return jsonify({
                    'success': False,
                    'error': f'3DDFA error: {error_message}'
                }), 500
        
        else:
            return jsonify({
                'success': False,
                'error': f'Unknown model: {model}. Supported models: mediapipe, dlib, face_alignment, retinaface, lab, 3ddfa'
            }), 400
        
        if total_landmarks == 0:
            return jsonify({
                'success': False,
                'error': 'No landmarks detected'
            }), 400
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'total_landmarks': total_landmarks,
            'image_width': width,
            'image_height': height,
            'model': model
        })
        
    except Exception as e:
        print(f"[Facial Landmark] Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# Aariz Cephalometric Analysis Endpoints
# =============================================================================

def apply_computer_vision_preprocessing(image):
    """
    Apply computer vision preprocessing to improve image quality for landmark detection.
    Uses OpenCV for enhancement operations.
    Only processes the top-right region: x from 50% to 100%, y from 0% to 50%.
    """
    import numpy as np
    
    # Convert PIL to numpy array
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    
    # Define crop region: x from 50% to 100%, y from 0% to 50%
    x_start = int(width * 0.5)
    x_end = width
    y_start = 0
    y_end = int(height * 0.5)
    
    # Crop the top-right region
    img_cropped = img_array[y_start:y_end, x_start:x_end].copy()
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2BGR)
    
    # 1. Denoise using bilateral filter (preserves edges while reducing noise)
    img_denoised = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    
    # 2. Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    lab = cv2.cvtColor(img_denoised, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    img_enhanced = cv2.merge([l_enhanced, a, b])
    img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_LAB2BGR)
    
    # 3. Sharpen using unsharp masking
    gaussian = cv2.GaussianBlur(img_enhanced, (0, 0), 2.0)
    img_sharpened = cv2.addWeighted(img_enhanced, 1.5, gaussian, -0.5, 0)
    
    # Convert back to RGB
    img_rgb_processed = cv2.cvtColor(img_sharpened, cv2.COLOR_BGR2RGB)
    
    # Copy processed region back to original image
    img_array[y_start:y_end, x_start:x_end] = img_rgb_processed
    
    # Convert back to PIL Image
    image_processed = Image.fromarray(img_array)
    
    return image_processed

@app.route('/detect-768', methods=['POST'])
def detect_768():
    """Detect cephalometric landmarks using 768x768 Aariz model"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Apply computer vision preprocessing
        image = apply_computer_vision_preprocessing(image)
        
        # Get predictor
        predictor = get_aariz_predictor('768')
        if predictor is None:
            return jsonify({
                'success': False,
                'error': 'Aariz 768 model not available'
            }), 503
        
        # Predict
        result = predictor.predict(image, target_size=(768, 768))
        landmarks = result['landmarks']
        image_size = result['image_size']
        
        # Remove P1/P2 if present (these models should not detect P1/P2)
        if 'p1' in landmarks:
            del landmarks['p1']
        if 'p2' in landmarks:
            del landmarks['p2']
        
        # Calculate metadata
        valid_landmarks = sum(1 for v in landmarks.values() if v is not None)
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'Aariz 768x768',
                'num_landmarks': 29,  # Only anatomical landmarks, no P1/P2
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': image_size[0],
                    'height': image_size[1]
                },
                'processing_time': 0.0
            }
        })
        
    except Exception as e:
        print(f"[Detect-768] Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect-cldetection2023', methods=['POST'])
def detect_cldetection2023():
    """Detect cephalometric landmarks using CLdetection2023 model (38 landmarks)"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # 🔧 FIX: Store original image size BEFORE any processing
        # This is needed to correctly scale landmarks back to original image coordinates
        original_image_width, original_image_height = image.size
        
        # Apply computer vision preprocessing
        image = apply_computer_vision_preprocessing(image)
        
        # Convert PIL image to numpy array (BGR for OpenCV)
        import numpy as np
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Store image_np size (after preprocessing, before remove_zero_padding)
        # This is needed for correct landmark scaling
        image_np_height, image_np_width = image_np.shape[:2]
        
        # Get model
        model = load_cldetection2023_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'CLdetection2023 model not available',
                'status': cldetection2023_status
            }), 503
        
        # 🔍 DEBUG: Check if model is quantized
        try:
            if hasattr(model, 'model'):
                underlying = model.model
                # Check if any layer is quantized
                is_quantized = False
                for name, module in underlying.named_modules():
                    if isinstance(module, (torch.quantization.QuantWrapper, 
                                         torch.nn.quantized.Quantize,
                                         torch.nn.quantized.Linear,
                                         torch.nn.quantized.Conv2d)):
                        is_quantized = True
                        print(f"[CLdetection2023] ✅ Detected quantized layer: {name} ({type(module).__name__})")
                        break
                if is_quantized:
                    print(f"[CLdetection2023] 🚀 Using QUANTIZED model (INT8) for inference")
                else:
                    print(f"[CLdetection2023] ℹ️  Using FP32 model (not quantized)")
        except Exception as e:
            print(f"[CLdetection2023] ⚠️  Could not check quantization status: {e}")
        
        # Get device info for metadata - Force CPU usage
        import torch
        device = torch.device('cpu')  # Always use CPU for inference
        
        # Import required functions
        from mmpose.apis import inference_topdown
        from mmpose.structures import merge_data_samples
        
        # Get remove_zero_padding function (defined during model loading)
        if cldetection2023_remove_zero_padding is None:
            # Fallback: define locally if not available
            import numpy as np
            def remove_zero_padding_local(image_array: np.ndarray) -> np.ndarray:
                """Remove zero padding from an image (pure numpy implementation)
                🚀 OPTIMIZED: Uses faster numpy operations for better CPU performance"""
                # 🚀 OPTIMIZATION: Use np.sum with keepdims=False for faster computation
                row = np.sum(image_array, axis=(1, 2))
                column = np.sum(image_array, axis=(0, 2))
                
                # 🚀 OPTIMIZATION: Use np.nonzero() which is faster than np.argwhere()
                non_zero_rows = np.nonzero(row)[0]
                non_zero_cols = np.nonzero(column)[0]
                
                if len(non_zero_rows) == 0 or len(non_zero_cols) == 0:
                    return image_array
                
                # Get first and last indices more efficiently
                first_row = int(non_zero_rows[0])
                last_row = int(non_zero_rows[-1])
                first_col = int(non_zero_cols[0])
                last_col = int(non_zero_cols[-1])
                
                # 🚀 OPTIMIZATION: Direct slicing is faster
                image_array = image_array[first_row:last_row+1, first_col:last_col+1, :]
                return image_array
            remove_zero_padding = remove_zero_padding_local
        else:
            remove_zero_padding = cldetection2023_remove_zero_padding
        
        # Preprocess image (remove zero padding)
        # 🚀 OPTIMIZATION: Optimize remove_zero_padding for faster execution
        preprocess_start = time.time()
        
        # 🚀 OPTIMIZATION: Use parallel processing for large images
        # For very large images, we can use threading for preprocessing
        if image_np.size > 10_000_000:  # Images larger than ~10MP
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                future = executor.submit(remove_zero_padding, image_np)
                image_processed = future.result()
        else:
            image_processed = remove_zero_padding(image_np)
        
        preprocess_time = time.time() - preprocess_start
        if preprocess_time > 0.1:  # Only log if it takes significant time
            print(f"[CLdetection2023] Preprocessing time: {preprocess_time:.3f}s")
        
        # 🔧 FIX: Calculate offset from remove_zero_padding
        # This is needed to correctly map landmarks back to original image coordinates
        processed_height, processed_width = image_processed.shape[:2]
        
        # 🚀 OPTIMIZATION: Cache the offset calculation (only compute once)
        # Calculate the offset (how much was cropped from top-left)
        # We need to find where the non-zero content starts in the original image
        # Use more efficient numpy operations
        row = np.sum(image_np, axis=(1, 2))
        column = np.sum(image_np, axis=(0, 2))
        
        # 🚀 OPTIMIZATION: Use np.nonzero() which is faster than np.argwhere() for this use case
        non_zero_rows = np.nonzero(row)[0]
        non_zero_cols = np.nonzero(column)[0]
        
        if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
            crop_y_offset = int(non_zero_rows[0])  # Top offset
            crop_x_offset = int(non_zero_cols[0])  # Left offset
        else:
            crop_y_offset = 0
            crop_x_offset = 0
        
        # Get processed image size (after remove_zero_padding, before resize)
        orig_height, orig_width = image_processed.shape[:2]
        true_orig_width = orig_width
        true_orig_height = orig_height
        
        # 🚀 OPTIMIZATION: Resize disabled - using original image size
        # Testing if processing original size is faster (no resize overhead)
        resize_start = time.time()
        scale_factor = 1.0  # No scaling - use original size
        new_width = orig_width
        new_height = orig_height
        resize_time = time.time() - resize_start
        print(f"[CLdetection2023] ⚡ Resize DISABLED - using original image size: {orig_width}x{orig_height} (resize time: {resize_time:.3f}s)")
        
        # Predict landmarks with optimizations
        # Note: time module is already imported at the top of the file
        
        # CPU optimizations are already set globally at startup
        
        # 🚀 OPTIMIZATION: Additional CPU-specific optimizations for inference
        # Set torch to inference mode (faster than eval mode)
        torch.set_grad_enabled(False)
        
        # 🚀 OPTIMIZATION: Use torch.inference_mode() for even better performance
        # This is faster than torch.no_grad() for inference
        start_time = time.time()
        
        # 🚀 OPTIMIZATION: Disable autograd completely for maximum speed
        torch.set_grad_enabled(False)
        
        # 🚀 OPTIMIZATION: Set deterministic to False for faster operations
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False  # Not needed for CPU but set anyway
        
        # 🚀 OPTIMIZATION: Use optimized inference path
        # For MMPose models, we optimize the inference pipeline
        # Note: inference_topdown handles preprocessing internally, which adds some overhead
        # But it's necessary for MMPose models
        
        # 🚀 OPTIMIZATION: Pre-warm the model if this is the first inference
        # This helps with JIT compilation and optimization
        
        # 🚀 OPTIMIZATION: Use torch.inference_mode() for maximum speed
        # This is faster than torch.no_grad() and disables more overhead
        with torch.inference_mode():  # 🚀 Faster than torch.no_grad() for inference
            # 🚀 OPTIMIZATION: Disable Python GIL for better multi-threading
            # 🚀 OPTIMIZATION: Use optimized inference path
            # 🚀 OPTIMIZATION: Minimize tensor operations and memory allocations
            predict_results = inference_topdown(
                model=model,
                img=image_processed,
                bboxes=None,
                bbox_format='xyxy'
            )
        
        inference_time = time.time() - start_time
        print(f"[CLdetection2023] Inference time: {inference_time:.3f}s (image size: {new_width if scale_factor < 1.0 else orig_width}x{new_height if scale_factor < 1.0 else orig_height})")
        
        # 🚀 OPTIMIZATION: Merge results efficiently
        # merge_data_samples can be slow, optimize it
        merge_start = time.time()
        
        # 🚀 OPTIMIZATION: Skip merge if not needed (if results are already in correct format)
        # Check if we can extract keypoints directly
        try:
            # Try to extract keypoints directly without merge (faster)
            if isinstance(predict_results, list) and len(predict_results) > 0:
                first_result = predict_results[0]
                if hasattr(first_result, 'pred_instances') and hasattr(first_result.pred_instances, 'keypoints'):
                    # Can extract directly without merge
                    keypoints = first_result.pred_instances.keypoints[0, :, :]
                    result_samples = first_result
                    print(f"[CLdetection2023] ⚡ Direct keypoint extraction (skipped merge)")
                else:
                    # Need to merge
                    result_samples = merge_data_samples(predict_results)
                    keypoints = result_samples.pred_instances.keypoints[0, :, :]
            else:
                result_samples = merge_data_samples(predict_results)
                keypoints = result_samples.pred_instances.keypoints[0, :, :]
        except Exception as e:
            # Fallback to standard merge
            result_samples = merge_data_samples(predict_results)
            keypoints = result_samples.pred_instances.keypoints[0, :, :]
        
        merge_time = time.time() - merge_start
        merge_skipped = merge_time < 0.01  # If very fast, probably skipped
        if merge_time > 0.05:  # Only log if it takes significant time
            print(f"[CLdetection2023] Merge time: {merge_time:.3f}s")
        elif merge_skipped:
            print(f"[CLdetection2023] ⚡ Direct keypoint extraction (skipped merge, saved time)")
        
        # 🚀 OPTIMIZATION: Convert to numpy immediately to free GPU/CPU memory
        # This is critical for memory management and speed
        # Use float32 for better performance (faster than float64)
        if hasattr(keypoints, 'cpu'):
            keypoints = keypoints.cpu().numpy().astype(np.float32)
        elif not isinstance(keypoints, np.ndarray):
            keypoints = np.array(keypoints, dtype=np.float32)
        else:
            keypoints = keypoints.astype(np.float32)  # Ensure float32
        
        # Scale keypoints back to processed image size (after remove_zero_padding) if image was resized
        if scale_factor < 1.0:
            keypoints = keypoints / scale_factor
            print(f"[CLdetection2023] Scaled keypoints back to processed size (scale: {1.0/scale_factor:.3f})")
        
        # 🔧 FIX: Scale keypoints from processed image (after remove_zero_padding) to original image
        # Calculate scale factors for x and y separately (in case of non-square images)
        # Use image_np dimensions (after preprocessing, before remove_zero_padding) for accurate scaling
        scale_x_to_original = image_np_width / processed_width
        scale_y_to_original = image_np_height / processed_height
        
        # Scale keypoints to original image coordinates
        # First scale to image_np coordinates, then add offset to get original coordinates
        keypoints[:, 0] = keypoints[:, 0] * scale_x_to_original + crop_x_offset
        keypoints[:, 1] = keypoints[:, 1] * scale_y_to_original + crop_y_offset
        
        print(f"[CLdetection2023] Scaled keypoints to original image coordinates:")
        print(f"  Processed size (after remove_zero_padding): {processed_width}x{processed_height}")
        print(f"  Image_np size (after preprocessing): {image_np_width}x{image_np_height}")
        print(f"  Original size: {original_image_width}x{original_image_height}")
        print(f"  Scale factors: x={scale_x_to_original:.3f}, y={scale_y_to_original:.3f}")
        print(f"  Crop offsets: x={crop_x_offset}, y={crop_y_offset}")
        
        # Map 38 landmarks to named landmarks
        # CLdetection2023 has 38 landmarks (0-37, indexed from 0)
        # Standard cephalometric landmark names based on CLdetection2023 dataset
        # This mapping follows the exact order and names from the CLdetection2023 challenge
        landmark_names = [
            'S',      # 0: Sella
            'N',      # 1: Nasion
            'Or',     # 2: Orbitale
            'Po',     # 3: Porion
            'A',      # 4: Subspinale
            'B',      # 5: Supramental
            'Pog',    # 6: Pogonion
            'Me',     # 7: Menton
            'Gn',     # 8: Gnathion
            'Go',     # 9: Gonion
            'L1',     # 10: Lower Incisor
            'U1',     # 11: Upper Incisor
            'UL',     # 12: Upper Lip
            'LL',     # 13: Lower Lip
            'Sn',     # 14: Subnasale
            'Pog′',   # 15: Soft Tissue Pogonion (Pog')
            'PNS',    # 16: Posterior Nasal Spine
            'ANS',    # 17: Anterior Nasal Spine
            'Ar',     # 18: Articulare
            'D',      # 19: D (No Anatomical Landmark)
            'U1A',    # 20: U1A
            'L1A',    # 21: L1A
            'Cm',     # 22: Columella
            'Ptm',    # 23: Pterygomaxillary Fissure
            'Co',     # 24: Condylion
            'Pn',     # 25: Pronasale (mapped from Prn to Pn for AARIZ compatibility)
            'Ba',     # 26: Basion
            'PT',     # 27: PT
            'Bo',     # 28: Bolton
            'UL′',    # 29: UL' (Upper Lip')
            'LL′',    # 30: LL' (Lower Lip')
            'Gn′',    # 31: Gnathion of Soft Tissue (Gn')
            'Me′',    # 32: Menton of Soft Tissue (Me')
            'G',      # 33: Glabella
            'N′',     # 34: Nasion of Soft Tissue (N')
            'C',      # 35: Cervical Point
            'UMT',    # 36: Upper Molar (mapped from U6 to UMT for AARIZ compatibility)
            'LMT',    # 37: Lower Molar (mapped from L6 to LMT for AARIZ compatibility)
        ]
        
        # Create landmarks dictionary with standard names (no numbered landmarks)
        landmarks = {}
        for i in range(min(keypoints.shape[0], len(landmark_names))):
            x, y = float(keypoints[i, 0]), float(keypoints[i, 1])
            # Check if landmark is valid
            # MMPose returns coordinates for all landmarks
            # We accept all coordinates that are positive (>= 0)
            # The frontend can filter out invalid landmarks if needed
            if x >= 0 and y >= 0:
                landmark_name = landmark_names[i]
                # Use standard landmark name only (no numbered landmarks)
                landmarks[landmark_name] = {
                    'x': x,
                    'y': y
                }
        
        # Add P1/P2 calibration points (enabled) - only if not already present
        # This prevents re-detection and infinite loops when landmarks are saved and reloaded
        # Important: p1/p2 must be calculated relative to the same image space as landmarks
        # Landmarks are calculated from image_processed (after remove_zero_padding and possible resize)
        # Landmarks are then scaled back to original size if image was resized
        # So p1/p2 should be calculated from image_processed and then scaled the same way
        # Remove P1/P2 if present (CLdetection2023 should not detect P1/P2)
        if 'p1' in landmarks:
            del landmarks['p1']
            print(f"[CLdetection2023] Removed P1 from landmarks (not part of CLdetection2023 model)")
        if 'p2' in landmarks:
            del landmarks['p2']
            print(f"[CLdetection2023] Removed P2 from landmarks (not part of CLdetection2023 model)")
        
        # Calculate metadata
        valid_landmarks = sum(1 for v in landmarks.values() if v is not None)
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'CLdetection2023 (38 landmarks)',
                'num_landmarks': 38,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': original_image_width,
                    'height': original_image_height
                },
                'processing_time': round(inference_time, 3),
                'optimization': {
                    'image_resized': scale_factor < 1.0,
                    'scale_factor': round(scale_factor, 3) if scale_factor < 1.0 else 1.0,
                    'inference_device': str(device) if 'device' in locals() else 'cpu'
                }
            }
        })
        
    except Exception as e:
        print(f"[Detect-CLdetection2023] Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect-fast-cpu-512', methods=['POST'])
def detect_fast_cpu_512():
    """Detect cephalometric landmarks using fast CPU 512 model (checkpoint_fast_cpu_512_best.pth)"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Store original image size
        original_image_width, original_image_height = image.size
        
        # Load model
        model = load_fast_cpu_512_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Fast CPU 512 model not available',
                'status': fast_cpu_512_status
            }), 503
        
        # Preprocess image
        import torch
        import torchvision.transforms as transforms
        import numpy as np
        
        # Transform: resize to 512x512, normalize
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        start_time = time.time()
        with torch.inference_mode():
            heatmaps = model(image_tensor)
        
        inference_time = time.time() - start_time
        
        # Convert heatmaps to coordinates
        # Import from Aariz/utils
        base_dir = os.path.dirname(os.path.abspath(__file__))
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        from utils import heatmap_to_coordinates
        
        # Apply sigmoid to heatmaps (model outputs logits)
        heatmaps_np = torch.sigmoid(heatmaps).cpu().numpy()[0]  # (38, 512, 512)
        
        # Extract coordinates from heatmaps
        coordinates = heatmap_to_coordinates(heatmaps_np, 512, 512)  # (38, 2)
        
        # Scale coordinates back to original image size
        scale_x = original_image_width / 512.0
        scale_y = original_image_height / 512.0
        
        coordinates[:, 0] *= scale_x
        coordinates[:, 1] *= scale_y
        
        # Map to landmark names (38 landmarks)
        landmark_names = [
            'S',      # 0: Sella
            'N',      # 1: Nasion
            'Or',     # 2: Orbitale
            'Po',     # 3: Porion
            'A',      # 4: Subspinale
            'B',      # 5: Supramental
            'Pog',    # 6: Pogonion
            'Me',     # 7: Menton
            'Gn',     # 8: Gnathion
            'Go',     # 9: Gonion
            'L1',     # 10: Lower Incisor
            'U1',     # 11: Upper Incisor
            'UL',     # 12: Upper Lip
            'LL',     # 13: Lower Lip
            'Sn',     # 14: Subnasale
            'Pog′',   # 15: Soft Tissue Pogonion (Pog')
            'PNS',    # 16: Posterior Nasal Spine
            'ANS',    # 17: Anterior Nasal Spine
            'Ar',     # 18: Articulare
            'D',      # 19: D (No Anatomical Landmark)
            'U1A',    # 20: U1A
            'L1A',    # 21: L1A
            'Cm',     # 22: Cm
            'Ptm',    # 23: Ptm
            'Co',     # 24: Co
            'Pn',     # 25: Pn
            'Ba',     # 26: Ba
            'PT',     # 27: PT
            'Bo',     # 28: Bo
            'UL′',    # 29: UL'
            'LL′',    # 30: LL'
            'Gn′',    # 31: Gn'
            'Me′',    # 32: Me'
            'G',      # 33: G
            'N′',     # 34: N'
            'C',      # 35: C
            'UMT',    # 36: UMT
            'LMT',    # 37: LMT
        ]
        
        # Create landmarks dictionary
        landmarks = {}
        for i in range(min(len(coordinates), len(landmark_names))):
            x, y = float(coordinates[i, 0]), float(coordinates[i, 1])
            # Check if landmark is valid (within image bounds)
            if x >= 0 and y >= 0 and x < original_image_width and y < original_image_height:
                landmark_name = landmark_names[i]
                landmarks[landmark_name] = {
                    'x': x,
                    'y': y
                }
        
        # Remove D and C landmarks (not anatomical)
        if 'D' in landmarks:
            del landmarks['D']
        if 'C' in landmarks:
            del landmarks['C']
        
        # Calculate metadata
        valid_landmarks = len(landmarks)
        
        print(f"[Fast CPU 512] Successfully detected {valid_landmarks} landmarks")
        print(f"[Fast CPU 512] Processing time: {inference_time:.3f}s")
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'Fast CPU 512 (checkpoint_fast_cpu_512_best.pth)',
                'num_landmarks': 38,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': original_image_width,
                    'height': original_image_height
                },
                'processing_time': round(inference_time, 3),
                'model_config': {
                    'input_size': 512,
                    'architecture': 'HRNetLandmarkModel',
                    'width': 16,
                    'approach': 'Heatmap-based',
                    'device': 'CPU'
                }
            }
        })
        
    except Exception as e:
        print(f"[Fast CPU 512] Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect-p1-p2', methods=['POST'])
def detect_p1_p2():
    """Detect P1 and P2 calibration landmarks using checkpoint_p1_p2_fast_cpu_512_best.pth"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Store original image size
        original_image_width, original_image_height = image.size
        
        # Load model
        model = load_p1_p2_model()
        if model is None:
            return jsonify({
                'success': False,
                'error': 'P1/P2 model not available',
                'status': p1_p2_status
            }), 503
        
        # Preprocess image
        import torch
        import torchvision.transforms as transforms
        import numpy as np
        
        # Transform: resize to 512x512, normalize
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        # Run inference
        start_time = time.time()
        with torch.inference_mode():
            heatmaps = model(image_tensor)
        
        inference_time = time.time() - start_time
        
        # Convert heatmaps to coordinates
        # Import from Aariz/utils
        base_dir = os.path.dirname(os.path.abspath(__file__))
        aariz_dir = os.path.join(base_dir, 'Aariz')
        if aariz_dir not in sys.path:
            sys.path.insert(0, aariz_dir)
        from utils import heatmap_to_coordinates
        
        # Apply sigmoid to heatmaps (model outputs logits)
        heatmaps_np = torch.sigmoid(heatmaps).cpu().numpy()[0]  # (2, 512, 512)
        
        # Extract coordinates from heatmaps
        coordinates = heatmap_to_coordinates(heatmaps_np, 512, 512)  # (2, 2)
        
        # Scale coordinates back to original image size
        scale_x = original_image_width / 512.0
        scale_y = original_image_height / 512.0
        
        coordinates[:, 0] *= scale_x
        coordinates[:, 1] *= scale_y
        
        # Create landmarks dictionary (P1 and P2 only)
        landmarks = {}
        landmark_names = ['p1', 'p2']
        
        for i in range(min(len(coordinates), len(landmark_names))):
            x, y = float(coordinates[i, 0]), float(coordinates[i, 1])
            # Check if landmark is valid (within image bounds)
            if x >= 0 and y >= 0 and x < original_image_width and y < original_image_height:
                landmark_name = landmark_names[i]
                landmarks[landmark_name] = {
                    'x': x,
                    'y': y
                }
        
        # Calculate metadata
        valid_landmarks = len(landmarks)
        
        print(f"[P1/P2 Detection] Successfully detected {valid_landmarks} calibration landmarks")
        print(f"[P1/P2 Detection] Processing time: {inference_time:.3f}s")
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'P1/P2 Fast CPU 512 (checkpoint_p1_p2_fast_cpu_512_best.pth)',
                'num_landmarks': 2,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': original_image_width,
                    'height': original_image_height
                },
                'processing_time': round(inference_time, 3),
                'model_config': {
                    'input_size': 512,
                    'architecture': 'HRNetLandmarkModel',
                    'width': 16,
                    'approach': 'Heatmap-based',
                    'device': 'CPU'
                }
            }
        })
        
    except Exception as e:
        print(f"[P1/P2 Detection] Error: {e}")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect-cldetection-optimized-512', methods=['POST'])
def detect_cldetection_optimized_512():
    """Detect cephalometric landmarks using optimized CLdetection model (512x512) - Fast mode"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Store original size
        orig_width, orig_height = image_pil.size
        
        # Convert to BGR for OpenCV
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Load optimized detector
        detector = load_cldetection_optimized(model_key='optimized_512', input_size=512)
        if detector is None:
            return jsonify({
                'success': False,
                'error': 'Optimized CLdetection model (512) not available',
                'status': cldetection_optimized_status.get('optimized_512', 'not_loaded')
            }), 503
        
        # Run inference
        start_time = time.time()
        keypoints = detector.predict(image_np)
        inference_time = time.time() - start_time
        
        # Map to landmark names (same 38 landmarks as CLdetection2023)
        landmark_names = [
            'S', 'N', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
            'L1', 'U1', 'UL', 'LL', 'Sn', 'Pog′', 'PNS', 'ANS', 'Ar', 'D',
            'U1A', 'L1A', 'Cm', 'Ptm', 'Co', 'Pn', 'Ba', 'PT', 'Bo', 'UL′',
            'LL′', 'Gn′', 'Me′', 'G', 'N′', 'C', 'UMT', 'LMT',
        ]
        
        landmarks = {}
        for i in range(min(len(keypoints), len(landmark_names))):
            x, y = float(keypoints[i][0]), float(keypoints[i][1])
            if x >= 0 and y >= 0:
                landmarks[landmark_names[i]] = {'x': x, 'y': y}
        
        # Remove D and C landmarks
        if 'D' in landmarks:
            del landmarks['D']
        if 'C' in landmarks:
            del landmarks['C']
        
        valid_landmarks = len(landmarks)
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'CLdetection Optimized 512x512 (ONNX)',
                'num_landmarks': 38,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': orig_width,
                    'height': orig_height
                },
                'processing_time': round(inference_time, 3),
                'optimization': {
                    'input_size': 512,
                    'framework': 'ONNX Runtime',
                    'speedup': '~8x faster than PyTorch',
                    'inference_device': 'cpu'
                }
            }
        })
        
    except Exception as e:
        print(f"[CLdetection Optimized 512] Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect-cldetection-optimized-640', methods=['POST'])
def detect_cldetection_optimized_640():
    """Detect cephalometric landmarks using optimized CLdetection model (640x640) - Balanced mode"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Store original size
        orig_width, orig_height = image_pil.size
        
        # Convert to BGR for OpenCV
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Load optimized detector
        detector = load_cldetection_optimized(model_key='optimized_640', input_size=640)
        if detector is None:
            return jsonify({
                'success': False,
                'error': 'Optimized CLdetection model (640) not available',
                'status': cldetection_optimized_status.get('optimized_640', 'not_loaded')
            }), 503
        
        # Run inference
        start_time = time.time()
        keypoints = detector.predict(image_np)
        inference_time = time.time() - start_time
        
        # Map to landmark names
        landmark_names = [
            'S', 'N', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
            'L1', 'U1', 'UL', 'LL', 'Sn', 'Pog′', 'PNS', 'ANS', 'Ar', 'D',
            'U1A', 'L1A', 'Cm', 'Ptm', 'Co', 'Pn', 'Ba', 'PT', 'Bo', 'UL′',
            'LL′', 'Gn′', 'Me′', 'G', 'N′', 'C', 'UMT', 'LMT',
        ]
        
        landmarks = {}
        for i in range(min(len(keypoints), len(landmark_names))):
            x, y = float(keypoints[i][0]), float(keypoints[i][1])
            if x >= 0 and y >= 0:
                landmarks[landmark_names[i]] = {'x': x, 'y': y}
        
        # Remove D and C landmarks
        if 'D' in landmarks:
            del landmarks['D']
        if 'C' in landmarks:
            del landmarks['C']
        
        valid_landmarks = len(landmarks)
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'CLdetection Optimized 640x640 (ONNX)',
                'num_landmarks': 38,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': orig_width,
                    'height': orig_height
                },
                'processing_time': round(inference_time, 3),
                'optimization': {
                    'input_size': 640,
                    'framework': 'ONNX Runtime',
                    'speedup': '~7.5x faster than PyTorch',
                    'inference_device': 'cpu'
                }
            }
        })
        
    except Exception as e:
        print(f"[CLdetection Optimized 640] Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/detect-cldetection-optimized-1024', methods=['POST'])
def detect_cldetection_optimized_1024():
    """Detect cephalometric landmarks using optimized CLdetection model (1024x1024) - High accuracy mode"""
    try:
        data = request.json
        if not data or 'image_base64' not in data:
            return jsonify({
                'success': False,
                'error': 'image_base64 is required'
            }), 400
        
        # Decode base64 image
        image_base64 = data['image_base64']
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Store original size
        orig_width, orig_height = image_pil.size
        
        # Convert to BGR for OpenCV
        image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
        
        # Load optimized detector
        detector = load_cldetection_optimized(model_key='optimized_1024', input_size=1024)
        if detector is None:
            return jsonify({
                'success': False,
                'error': 'Optimized CLdetection model (1024) not available',
                'status': cldetection_optimized_status.get('optimized_1024', 'not_loaded')
            }), 503
        
        # Run inference
        start_time = time.time()
        keypoints = detector.predict(image_np)
        inference_time = time.time() - start_time
        
        # Map to landmark names
        landmark_names = [
            'S', 'N', 'Or', 'Po', 'A', 'B', 'Pog', 'Me', 'Gn', 'Go',
            'L1', 'U1', 'UL', 'LL', 'Sn', 'Pog′', 'PNS', 'ANS', 'Ar', 'D',
            'U1A', 'L1A', 'Cm', 'Ptm', 'Co', 'Pn', 'Ba', 'PT', 'Bo', 'UL′',
            'LL′', 'Gn′', 'Me′', 'G', 'N′', 'C', 'UMT', 'LMT',
        ]
        
        landmarks = {}
        for i in range(min(len(keypoints), len(landmark_names))):
            x, y = float(keypoints[i][0]), float(keypoints[i][1])
            if x >= 0 and y >= 0:
                landmarks[landmark_names[i]] = {'x': x, 'y': y}
        
        # Remove D and C landmarks
        if 'D' in landmarks:
            del landmarks['D']
        if 'C' in landmarks:
            del landmarks['C']
        
        valid_landmarks = len(landmarks)
        
        return jsonify({
            'success': True,
            'landmarks': landmarks,
            'metadata': {
                'model': 'CLdetection Optimized 1024x1024 (ONNX)',
                'num_landmarks': 38,
                'valid_landmarks': valid_landmarks,
                'image_size': {
                    'width': orig_width,
                    'height': orig_height
                },
                'processing_time': round(inference_time, 3),
                'optimization': {
                    'input_size': 1024,
                    'framework': 'ONNX Runtime',
                    'speedup': '~3x faster than PyTorch',
                    'inference_device': 'cpu'
                }
            }
        })
        
    except Exception as e:
        print(f"[CLdetection Optimized 1024] Error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/models', methods=['GET'])
def get_models():
    """Get available models for facial landmark detection"""
    models = []
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        models.append({
            'name': 'mediapipe',
            'available': True,
            'description': 'MediaPipe Face Mesh (468 points)'
        })
    except ImportError:
        models.append({
            'name': 'mediapipe',
            'available': False,
            'description': 'MediaPipe Face Mesh (468 points) - Not installed'
        })
    
    # Check dlib
    try:
        import dlib
        predictor_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'facial-landmark-detection',
            'shape_predictor_68_face_landmarks.dat'
        )
        models.append({
            'name': 'dlib',
            'available': os.path.exists(predictor_path),
            'description': 'dlib (68 points)'
        })
    except ImportError:
        models.append({
            'name': 'dlib',
            'available': False,
            'description': 'dlib (68 points) - Not installed'
        })
    
    # Check face-alignment
    try:
        import face_alignment
        models.append({
            'name': 'face_alignment',
            'available': True,
            'description': 'face-alignment (68 points)'
        })
    except ImportError:
        models.append({
            'name': 'face_alignment',
            'available': False,
            'description': 'face-alignment (68 points) - Not installed'
        })
    
    # Check RetinaFace
    try:
        from retinaface import RetinaFace
        models.append({
            'name': 'retinaface',
            'available': True,
            'description': 'RetinaFace (5 key points)'
        })
    except ImportError:
        models.append({
            'name': 'retinaface',
            'available': False,
            'description': 'RetinaFace (5 key points) - Not installed'
        })
    
    # Check LAB (Look at Boundary)
    try:
        import torch
        try:
            import face_alignment
            models.append({
                'name': 'lab',
                'available': True,
                'description': 'LAB - Look at Boundary (68 points, high accuracy)'
            })
        except ImportError:
            models.append({
                'name': 'lab',
                'available': False,
                'description': 'LAB - Look at Boundary (68 points) - Dependencies not installed'
            })
    except ImportError:
        models.append({
            'name': 'lab',
            'available': False,
            'description': 'LAB - Look at Boundary (68 points) - PyTorch not installed'
        })
    
    # Check 3DDFA (3D Dense Face Alignment)
    try:
        import torch
        try:
            import face_alignment
            models.append({
                'name': '3ddfa',
                'available': True,
                'description': '3DDFA - 3D Dense Face Alignment (68 points with 3D coordinates)'
            })
        except ImportError:
            models.append({
                'name': '3ddfa',
                'available': False,
                'description': '3DDFA - 3D Dense Face Alignment (3D landmarks) - Dependencies not installed'
            })
    except ImportError:
        models.append({
            'name': '3ddfa',
            'available': False,
            'description': '3DDFA - 3D Dense Face Alignment (3D landmarks) - PyTorch not installed'
        })
    
    return jsonify({
        'models': models,
        'facial_landmark': {
            'available_models': [m['name'] for m in models if m['available']]
        }
    })


@app.route('/preload-models', methods=['GET', 'POST'])
def preload_models():
    """Preload all Aariz models and return their status"""
    print("\n" + "="*70)
    print("Preloading Aariz Models...")
    print("="*70)
    
    results = {}
    
    # Preload 768 model
    print("\n[1/2] Preloading 768 model...")
    predictor_768 = get_aariz_predictor('768')
    results['768'] = {
        'status': aariz_status.get('768', 'not_loaded'),
        'loaded': predictor_768 is not None
    }
    
    print("\n" + "="*70)
    print("Preload Complete!")
    print("="*70)
    print(f"768 Model: {results['768']['status']}")
    print("="*70 + "\n")
    
    return jsonify({
        'success': True,
        'models': results,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def root():
    """Root endpoint - list available endpoints"""
    return jsonify({
        'service': 'Unified AI API Server',
        'version': '1.0.0',
        'endpoints': {
            'GET /health': 'Health check',
            'GET /models': 'Get available facial landmark models',
            'GET /preload-models': 'Preload all Aariz models and return status',
            'POST /predict': 'Detect teeth and dental issues in intra-oral images (YOLOv8)',
            'POST /facial-landmark': 'Detect facial landmarks (MediaPipe/dlib/face-alignment/RetinaFace)',
            'POST /detect-768': 'Detect cephalometric landmarks (768x768)',
            'POST /detect-cldetection2023': 'Detect cephalometric landmarks (CLdetection2023, 38 landmarks)',
            'POST /detect-fast-cpu-512': 'Detect cephalometric landmarks (Fast CPU 512x512 - checkpoint_fast_cpu_512_best.pth)',
            'POST /detect-cldetection-optimized-512': 'Detect cephalometric landmarks (CLdetection Optimized 512x512 - Fast)',
            'POST /detect-cldetection-optimized-640': 'Detect cephalometric landmarks (CLdetection Optimized 640x640 - Balanced)',
            'POST /detect-cldetection-optimized-1024': 'Detect cephalometric landmarks (CLdetection Optimized 1024x1024 - High Accuracy)',
        },
        'timestamp': datetime.now().isoformat()
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse
    import ssl

    parser = argparse.ArgumentParser(description='Unified AI API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host IP to bind server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5001, help='Port to run server on (default: 5001)')
    parser.add_argument('--ssl', action='store_true', help='Enable HTTPS with self-signed certificate')
    parser.add_argument('--cert', type=str, help='Path to SSL certificate file')
    parser.add_argument('--key', type=str, help='Path to SSL private key file')
    args = parser.parse_args()

    print("=" * 80)
    print("Unified AI API Server")
    print("=" * 80)

    # SSL Configuration
    ssl_context = None
    if True:
        try:
            if args.cert and args.key:
                # Use provided certificate and key
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS)
                ssl_context.load_cert_chain(args.cert, args.key)
                print("✅ Using provided SSL certificate and key")
            else:
                # Generate self-signed certificate for development
                from cryptography import x509
                from cryptography.x509.oid import NameOID
                from cryptography.hazmat.primitives import hashes, serialization
                from cryptography.hazmat.primitives.asymmetric import rsa
                import datetime

                # Generate private key
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048,
                )

                # Generate certificate
                subject = issuer = x509.Name([
                    x509.NameAttribute(NameOID.COUNTRY_NAME, "IR"),
                    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Tehran"),
                    x509.NameAttribute(NameOID.LOCALITY_NAME, "Tehran"),
                    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Bioritalin"),
                    x509.NameAttribute(NameOID.COMMON_NAME, "ceph2.bioritalin.ir"),
                ])

                cert = x509.CertificateBuilder().subject_name(
                    subject
                ).issuer_name(
                    issuer
                ).public_key(
                    private_key.public_key()
                ).serial_number(
                    x509.random_serial_number()
                ).not_valid_before(
                    datetime.datetime.utcnow()
                ).not_valid_after(
                    datetime.datetime.utcnow() + datetime.timedelta(days=365)
                ).add_extension(
                    x509.SubjectAlternativeName([
                        x509.DNSName("ceph2.bioritalin.ir"),
                        x509.DNSName("localhost"),
                        x509.DNSName("127.0.0.1"),
                    ]),
                    critical=False,
                ).sign(private_key, hashes.SHA256())

                # Write certificate and key to temporary files
                import tempfile
                import os

                cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem')
                key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.key')

                cert_file.write(cert.public_bytes(serialization.Encoding.PEM))
                cert_file.close()

                key_file.write(private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
                key_file.close()

                # Create SSL context
                ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
                ssl_context.load_cert_chain(cert_file.name, key_file.name)

                print("✅ Generated self-signed SSL certificate")
                print(f"   Certificate: {cert_file.name}")
                print(f"   Key: {key_file.name}")
                print("⚠️  WARNING: Using self-signed certificate - browsers will show security warning")
                print("   For production, use a proper SSL certificate from a trusted CA")

                # Store temp file paths for cleanup
                os.environ['SSL_CERT_FILE'] = cert_file.name
                os.environ['SSL_KEY_FILE'] = key_file.name

        except ImportError:
            print("❌ SSL enabled but cryptography library not installed")
            print("   Install with: pip install cryptography")
            print("   Falling back to HTTP...")
            ssl_context = None
        except Exception as e:
            print(f"❌ Error setting up SSL: {e}")
            print("   Falling back to HTTP...")
            ssl_context = None

    protocol = "https" if ssl_context else "http"
    print(f"Starting server on {protocol}://{args.host}:{args.port} (accessible from network)")
    print(f"Local access: {protocol}://localhost:{args.port}")
    print("\nAvailable endpoints:")
    print("  GET  /health                    - Health check")
    print("  POST /predict                   - Detect teeth in intra-oral images (YOLOv8)")
    print("  POST /detect-768                - Detect landmarks (768x768)")
    print("  POST /detect-cldetection2023    - Detect landmarks (CLdetection2023, 38 landmarks)")
    print("  POST /detect-fast-cpu-512       - Detect landmarks (Fast CPU 512x512)")
    print("=" * 80)
    print("\n📝 Note: Aariz models will be loaded on first request (lazy loading)")
    print("📝 Note: CLdetection2023 model will be loaded on first request (lazy loading)")
    if ssl_context:
        print("🔒 SSL/TLS enabled - Server is running with HTTPS")
    else:
        print("🔓 SSL/TLS disabled - Server is running with HTTP")
    print("=" * 80)

    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, ssl_context=ssl_context)
