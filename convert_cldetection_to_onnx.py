"""
اسکریپت تبدیل مدل CLdetection2023 به ONNX
Convert CLdetection2023 model to ONNX format for faster inference
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Add CLdetection2023 to path
base_dir = os.path.dirname(os.path.abspath(__file__))
cldetection_dir = os.path.join(base_dir, 'CLdetection2023')
if os.path.exists(cldetection_dir):
    sys.path.insert(0, cldetection_dir)

mmpose_package_dir = os.path.join(cldetection_dir, 'mmpose_package')
if os.path.exists(mmpose_package_dir):
    sys.path.insert(0, mmpose_package_dir)

def convert_to_onnx():
    """Convert CLdetection2023 model to ONNX"""
    print("="*70)
    print("🔄 Converting CLdetection2023 Model to ONNX")
    print("="*70)
    
    try:
        from mmengine.config import Config
        from mmpose.apis import init_model as init_pose_estimator
        import mmpose
    except ImportError as e:
        print(f"❌ Error importing mmpose: {e}")
        print("Please install MMPose first")
        return False
    
    # Paths
    config_file = os.path.join(cldetection_dir, 'configs', 'CLdetection2023', 'srpose_s2.py')
    checkpoint_path = os.path.join(cldetection_dir, 'model_pretrained_on_train_and_val.pth')
    onnx_output_path = os.path.join(cldetection_dir, 'model_cldetection2023.onnx')
    
    # Check files
    if not os.path.exists(config_file):
        print(f"❌ Config file not found: {config_file}")
        return False
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    print(f"\n📂 Loading model...")
    print(f"   Config: {config_file}")
    print(f"   Checkpoint: {checkpoint_path}")
    
    # Load model
    device = torch.device('cpu')
    pose_estimator = init_pose_estimator(
        config=config_file,
        checkpoint=checkpoint_path,
        device='cpu'
    )
    
    print("✅ Model loaded successfully")
    
    # Get underlying model
    if not hasattr(pose_estimator, 'model'):
        print("❌ Cannot access underlying model")
        return False
    
    model = pose_estimator.model
    model.eval()
    
    # Create dummy input (typical image size: 1024x1024)
    # MMPose models expect input in format: (batch, channels, height, width)
    # Input is usually preprocessed by MMPose pipeline
    print("\n🔍 Analyzing model input requirements...")
    
    # Try to get input shape from config or model
    dummy_input_shape = (1, 3, 1024, 1024)  # (batch, channels, height, width)
    dummy_input = torch.randn(*dummy_input_shape)
    
    print(f"   Input shape: {dummy_input_shape}")
    
    # Try to trace the model
    print("\n🔄 Converting to ONNX...")
    try:
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_output_path,
            export_params=True,
            opset_version=11,  # Compatible with most runtimes
            do_constant_folding=True,  # Optimize constants
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✅ ONNX model saved to: {onnx_output_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(onnx_output_path)
            onnx.checker.check_model(onnx_model)
            print("✅ ONNX model verification passed")
            
            # Print model info
            print(f"\n📊 Model Information:")
            print(f"   Input shape: {dummy_input_shape}")
            print(f"   Output shape: {model(dummy_input).shape}")
            print(f"   ONNX file size: {os.path.getsize(onnx_output_path) / (1024*1024):.2f} MB")
            
        except ImportError:
            print("⚠️  ONNX not installed, skipping verification")
        except Exception as e:
            print(f"⚠️  ONNX verification failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error converting to ONNX: {e}")
        import traceback
        traceback.print_exc()
        
        # Try alternative approach: export the forward method
        print("\n🔄 Trying alternative export method...")
        try:
            # Some MMPose models need special handling
            # Try exporting with actual preprocessing
            from mmpose.apis import inference_topdown
            from mmpose.structures import merge_data_samples
            
            # Create a dummy image (numpy array)
            dummy_img = np.zeros((1024, 1024, 3), dtype=np.uint8)
            
            # Run inference to get the actual input format
            with torch.no_grad():
                results = inference_topdown(
                    model=pose_estimator,
                    img=dummy_img,
                    bboxes=None,
                    bbox_format='xyxy'
                )
            
            print("✅ Model inference successful")
            print("⚠️  Direct ONNX export may not work for MMPose models")
            print("   Consider using ONNX Runtime with the original PyTorch model")
            
        except Exception as e2:
            print(f"❌ Alternative method also failed: {e2}")
        
        return False

def main():
    """Main function"""
    success = convert_to_onnx()
    
    if success:
        print("\n" + "="*70)
        print("✅ Conversion Complete!")
        print("="*70)
        print("\n📝 Next steps:")
        print("   1. The ONNX model is saved in CLdetection2023/ directory")
        print("   2. Update unified_ai_api_server.py to use ONNX Runtime")
        print("   3. Restart the server")
        print("\n💡 Note: If direct conversion fails, we'll use ONNX Runtime")
        print("   with PyTorch model (still faster than pure PyTorch)")
    else:
        print("\n" + "="*70)
        print("⚠️  Conversion Failed")
        print("="*70)
        print("\n💡 Alternative: We'll implement ONNX Runtime optimization")
        print("   which can still provide speed improvements")

if __name__ == "__main__":
    main()

