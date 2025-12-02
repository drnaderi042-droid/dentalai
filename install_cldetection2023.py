#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Installation script for CLdetection2023 model dependencies
This script installs MMPose and its dependencies required for CLdetection2023 model
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"\n{'='*60}")
    print(f"📦 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    print()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        print(f"✅ {description} - Success")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Failed")
        print(f"Error: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ is recommended for MMPose")
        return False
    return True

def main():
    """Main installation function"""
    print("="*60)
    print("🔧 CLdetection2023 Model Dependencies Installation")
    print("="*60)
    print()
    
    # Check Python version
    if not check_python_version():
        print("⚠️  Continuing anyway, but you may encounter issues...")
        print()
    
    # Get base directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cldetection_dir = os.path.join(base_dir, 'CLdetection2023')
    mmpose_dir = os.path.join(cldetection_dir, 'mmpose_package', 'mmpose')
    
    # Check if CLdetection2023 directory exists
    if not os.path.exists(cldetection_dir):
        print(f"❌ Error: CLdetection2023 directory not found: {cldetection_dir}")
        print("   Make sure CLdetection2023 folder exists in the project root")
        return False
    
    # Check if mmpose_package exists
    if not os.path.exists(mmpose_dir):
        print(f"❌ Error: MMPose package directory not found: {mmpose_dir}")
        print("   Make sure CLdetection2023/mmpose_package/mmpose exists")
        return False
    
    # Installation steps
    steps = [
        # Step 1: Install openmim
        ("pip install -U openmim", "Installing openmim (OpenMMLab package manager)"),
        
        # Step 2: Install mmengine (compatible version)
        ("mim install \"mmengine>=0.6.0,<1.0.0\"", "Installing mmengine (MMPose core dependency)"),
        
        # Step 3: Uninstall existing mmcv if incompatible
        ("pip uninstall mmcv mmcv-full -y", "Removing existing mmcv (if incompatible)"),
        
        # Step 4: Install compatible mmcv version
        ("mim install \"mmcv>=2.0.0rc4,<=2.1.0\"", "Installing mmcv (Computer Vision library) - Compatible version"),
        
        # Step 5: Install MMPose in development mode
        (f"cd {mmpose_dir} && pip install -e .", "Installing MMPose (in development mode)"),
        
        # Step 6: Upgrade numpy (required for MMPose)
        ("pip install --upgrade numpy", "Upgrading numpy (required for MMPose)"),
    ]
    
    # Run installation steps
    success_count = 0
    for command, description in steps:
        if run_command(command, description):
            success_count += 1
        else:
            print(f"⚠️  Warning: {description} failed, but continuing...")
            print()
    
    # Verify installation
    print(f"\n{'='*60}")
    print("🔍 Verifying Installation")
    print(f"{'='*60}")
    
    try:
        import mmengine
        print(f"✅ mmengine installed: {mmengine.__version__}")
    except ImportError:
        print("❌ mmengine not installed")
        return False
    
    try:
        import mmcv
        mmcv_version = mmcv.__version__
        print(f"✅ mmcv installed: {mmcv_version}")
        # Check if version is compatible
        from mmengine.utils import digit_version
        min_version = digit_version('2.0.0rc4')
        max_version = digit_version('2.1.0')
        current_version = digit_version(mmcv_version)
        if current_version < min_version or current_version > max_version:
            print(f"⚠️  Warning: mmcv version {mmcv_version} may not be compatible (requires >=2.0.0rc4, <=2.1.0)")
    except ImportError:
        print("❌ mmcv not installed")
        return False
    
    # SimpleITK is optional - only needed for loading .mha training files
    # For inference, we use a pure numpy implementation
    try:
        import SimpleITK
        print(f"✅ SimpleITK installed: {SimpleITK.__version__} (optional)")
    except ImportError:
        print("ℹ️  SimpleITK not installed (optional - only needed for training data)")
        print("   Inference will work without SimpleITK using pure numpy implementation")
    
    try:
        # Add mmpose to path for import
        sys.path.insert(0, mmpose_dir)
        import mmpose
        print(f"✅ mmpose installed: {mmpose.__version__}")
    except ImportError as e:
        print(f"❌ mmpose not installed: {e}")
        print("   Try running: cd CLdetection2023/mmpose_package/mmpose && pip install -e .")
        return False
    
    try:
        from mmpose.apis import init_model
        print("✅ mmpose.apis imported successfully")
    except ImportError as e:
        print(f"❌ mmpose.apis import failed: {e}")
        return False
    
    print(f"\n{'='*60}")
    print("✅ Installation Complete!")
    print(f"{'='*60}")
    print("\n📝 Next steps:")
    print("   1. Restart the unified_ai_api_server.py")
    print("   2. Test the CLdetection2023 model endpoint")
    print("   3. Check the server logs for any errors")
    print()
    
    return True

if __name__ == '__main__':
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

