"""
اسکریپت نصب Dependency های لازم برای بهینه‌سازی CLdetection2023
Install script for CLdetection2023 optimization dependencies
"""

import subprocess
import sys
import os

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*70}")
    print(f"📦 {description}")
    print(f"{'='*70}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ Success!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"\n🐍 Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("⚠️  Warning: Python 3.8+ is recommended")
        return False
    return True

def check_torch_version():
    """Check current PyTorch version"""
    try:
        import torch
        version = torch.__version__
        print(f"\n🔥 PyTorch version: {version}")
        
        # Check if version is 2.0+
        major_version = int(version.split('.')[0])
        minor_version = int(version.split('.')[1])
        
        if major_version >= 2:
            print("✅ PyTorch 2.0+ detected - torch.compile will be available")
            return True
        else:
            print("⚠️  PyTorch < 2.0 detected - torch.compile will not be available")
            print("   Consider upgrading: pip install torch>=2.0.0")
            return False
    except ImportError:
        print("⚠️  PyTorch not installed")
        return False

def install_required_packages():
    """Install required packages for optimizations"""
    print("\n" + "="*70)
    print("🚀 Installing Required Packages for CLdetection Optimizations")
    print("="*70)
    
    packages = [
        # Core optimizations
        ("torch>=2.0.0", "PyTorch 2.0+ for torch.compile support", True),
        ("torchvision>=0.15.0", "TorchVision for image processing", True),
        
        # Optional but recommended (may fail on some systems)
        ("mkl", "Intel MKL for faster CPU operations", False),
        ("mkl-service", "MKL service for thread management", False),
    ]
    
    success_count = 0
    failed_packages = []
    optional_failed = []
    
    for item in packages:
        if len(item) == 3:
            package, description, required = item
        else:
            package, description = item
            required = True
        print(f"\n📦 Installing: {package}")
        print(f"   Purpose: {description}")
        
        # Skip if already installed and up to date
        try:
            if package.startswith("torch"):
                import torch
                if "torch" in package.lower():
                    print(f"   ⏭️  PyTorch already installed: {torch.__version__}")
                    # Check if upgrade needed
                    if ">=2.0.0" in package:
                        major = int(torch.__version__.split('.')[0])
                        if major >= 2:
                            print(f"   ✅ Version is compatible, skipping...")
                            success_count += 1
                            continue
        except ImportError:
            pass
        
        if run_command(f"{sys.executable} -m pip install --upgrade {package}", f"Installing {package}"):
            success_count += 1
        else:
            if required:
                failed_packages.append(package)
            else:
                optional_failed.append(package)
                print(f"   ⚠️  Optional package failed (not critical)")
    
    return success_count, failed_packages, optional_failed

def install_optional_packages():
    """Install optional packages for advanced optimizations"""
    print("\n" + "="*70)
    print("📦 Optional Packages (for advanced optimizations)")
    print("="*70)
    
    optional_packages = [
        ("onnx", "ONNX for model conversion (30-50% speedup)"),
        ("onnxruntime", "ONNX Runtime for faster inference"),
        ("numba", "Numba JIT for preprocessing (10-20% speedup)"),
    ]
    
    print("\n❓ Optional packages (for advanced optimizations):")
    print("   These can provide additional speed improvements:")
    for pkg, desc in optional_packages:
        print(f"   - {pkg}: {desc}")
    
    try:
        response = input("\n   Install optional packages? (y/n): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\n⏭️  Skipping optional packages (non-interactive mode)")
        return
    
    if response == 'y':
        success_count = 0
        for package, description in optional_packages:
            print(f"\n📦 Installing: {package}")
            if run_command(f"{sys.executable} -m pip install {package}", f"Installing {package}"):
                success_count += 1
        print(f"\n✅ Installed {success_count}/{len(optional_packages)} optional packages")
    else:
        print("\n⏭️  Skipping optional packages")

def verify_installations():
    """Verify that installations are working"""
    print("\n" + "="*70)
    print("🔍 Verifying Installations")
    print("="*70)
    
    checks = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
    ]
    
    optional_checks = [
        ("onnx", "ONNX"),
        ("onnxruntime", "ONNX Runtime"),
        ("numba", "Numba"),
    ]
    
    print("\n✅ Required packages:")
    for module, name in checks:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ❌ {name}: Not installed")
    
    print("\n📦 Optional packages:")
    for module, name in optional_checks:
        try:
            mod = __import__(module)
            version = getattr(mod, '__version__', 'unknown')
            print(f"   ✅ {name}: {version}")
        except ImportError:
            print(f"   ⏭️  {name}: Not installed (optional)")
    
    # Check torch.compile availability
    print("\n🚀 Optimization features:")
    try:
        import torch
        if hasattr(torch, 'compile'):
            print("   ✅ torch.compile: Available")
        else:
            print("   ⚠️  torch.compile: Not available (PyTorch < 2.0)")
    except ImportError:
        print("   ❌ torch.compile: PyTorch not installed")

def main():
    """Main installation function"""
    print("="*70)
    print("🚀 CLdetection2023 Optimization Dependencies Installer")
    print("="*70)
    
    # Check Python version
    if not check_python_version():
        print("\n⚠️  Warning: Python version may not be compatible")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            print("Installation cancelled.")
            return
    
    # Check current PyTorch version
    check_torch_version()
    
    # Install required packages
    success_count, failed_packages, optional_failed = install_required_packages()
    
    print("\n" + "="*70)
    print("📊 Installation Summary")
    print("="*70)
    print(f"✅ Successfully installed: {success_count} packages")
    if failed_packages:
        print(f"❌ Failed required packages: {', '.join(failed_packages)}")
        print("\n⚠️  You may need to install these manually:")
        for pkg in failed_packages:
            print(f"   pip install {pkg}")
    if optional_failed:
        print(f"⚠️  Optional packages failed (not critical): {', '.join(optional_failed)}")
        print("   These are optional and won't affect core functionality")
    
    # Install optional packages (skip if non-interactive)
    try:
        install_optional_packages()
    except (EOFError, KeyboardInterrupt):
        print("\n⏭️  Skipping optional packages (non-interactive mode)")
    
    # Verify installations
    verify_installations()
    
    print("\n" + "="*70)
    print("✅ Installation Complete!")
    print("="*70)
    print("\n📝 Next steps:")
    print("   1. Restart your Python server")
    print("   2. Test the optimizations with a CLdetection request")
    print("   3. Check logs for 'torch.compile enabled' message")
    print("\n💡 Tips:")
    print("   - If torch.compile is not available, upgrade PyTorch: pip install torch>=2.0.0")
    print("   - For maximum speed, consider ONNX conversion (see ADVANCED_OPTIMIZATIONS_CLDETECTION.md)")

if __name__ == "__main__":
    main()

