"""
Script to inspect model checkpoints and determine number of landmarks
"""
import torch
import sys
import os

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file to determine model structure and landmarks"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {checkpoint_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: File not found: {checkpoint_path}")
        return None
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Print checkpoint keys
        print(f"\nCheckpoint keys: {list(checkpoint.keys())}")
        
        # Check for model state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Find output layer (usually last conv layer)
        output_channels = None
        landmark_count = None
        
        # Look for common output layer names
        for key in state_dict.keys():
            if 'heatmap' in key.lower() or 'output' in key.lower() or 'final' in key.lower():
                if 'weight' in key:
                    weight = state_dict[key]
                    if len(weight.shape) == 4:  # Conv2d weight
                        output_channels = weight.shape[0]
                        print(f"\nFound output layer: {key}")
                        print(f"  Shape: {weight.shape}")
                        print(f"  Output channels (landmarks): {output_channels}")
                        landmark_count = output_channels
                        break
        
        # Check for cls_layer, x_layer, y_layer (common in detection models)
        if 'cls_layer.weight' in state_dict:
            cls_weight = state_dict['cls_layer.weight']
            if len(cls_weight.shape) == 2:
                num_classes = cls_weight.shape[0]
                print(f"\nFound cls_layer:")
                print(f"  Shape: {cls_weight.shape}")
                print(f"  Number of classes/landmarks: {num_classes}")
                landmark_count = num_classes
        
        if 'x_layer.weight' in state_dict and 'y_layer.weight' in state_dict:
            x_weight = state_dict['x_layer.weight']
            y_weight = state_dict['y_layer.weight']
            if len(x_weight.shape) == 2:
                num_landmarks = x_weight.shape[0]
                print(f"\nFound coordinate layers:")
                print(f"  x_layer shape: {x_weight.shape}")
                print(f"  y_layer shape: {y_weight.shape}")
                print(f"  Number of landmarks: {num_landmarks}")
                if landmark_count is None:
                    landmark_count = num_landmarks
        
        # If not found, try to infer from model structure
        if landmark_count is None:
            # Look for any conv layer with small kernel size (likely output)
            for key in sorted(state_dict.keys()):
                if 'conv' in key.lower() and 'weight' in key:
                    weight = state_dict[key]
                    if len(weight.shape) == 4:
                        # Check if it's likely an output layer (small kernel, reasonable channels)
                        kernel_size = weight.shape[2] * weight.shape[3]
                        if kernel_size <= 9 and weight.shape[0] < 100:  # Likely output layer
                            print(f"\nPossible output layer: {key}")
                            print(f"  Shape: {weight.shape}")
                            print(f"  Output channels (possible landmarks): {weight.shape[0]}")
                            if landmark_count is None:
                                landmark_count = weight.shape[0]
        
        # Check for metadata
        metadata_keys = ['num_landmarks', 'landmarks', 'landmark_names', 'landmark_symbols', 
                         'config', 'args', 'model_config']
        found_metadata = {}
        for key in metadata_keys:
            if key in checkpoint:
                found_metadata[key] = checkpoint[key]
        
        if found_metadata:
            print(f"\nMetadata found:")
            for key, value in found_metadata.items():
                if isinstance(value, (list, tuple)) and len(value) > 0:
                    print(f"  {key}: {value[:10]}..." if len(value) > 10 else f"  {key}: {value}")
                else:
                    print(f"  {key}: {value}")
        
        # Check epoch info
        if 'epoch' in checkpoint:
            print(f"\nTraining info:")
            print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
            if 'best_mre' in checkpoint:
                print(f"  Best MRE: {checkpoint.get('best_mre', 'N/A')}")
        
        result = {
            'path': checkpoint_path,
            'landmark_count': landmark_count,
            'metadata': found_metadata,
            'checkpoint_keys': list(checkpoint.keys())
        }
        
        return result
        
    except Exception as e:
        print(f"ERROR loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    # Check models in Aariz folder
    aariz_path = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '.'
    
    models_to_check = [
        os.path.join(aariz_path, 'lung_uda.pth'),
        os.path.join(aariz_path, 'head_uda.pth')
    ]
    
    results = []
    for model_path in models_to_check:
        result = inspect_checkpoint(model_path)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        model_name = os.path.basename(result['path'])
        landmark_count = result['landmark_count']
        print(f"\n{model_name}:")
        if landmark_count:
            print(f"  Number of landmarks: {landmark_count}")
        else:
            print(f"  Number of landmarks: Could not determine")
        
        if result['metadata']:
            if 'landmark_names' in result['metadata']:
                print(f"  Landmark names: {result['metadata']['landmark_names']}")
            elif 'landmark_symbols' in result['metadata']:
                print(f"  Landmark symbols: {result['metadata']['landmark_symbols']}")
            elif 'num_landmarks' in result['metadata']:
                print(f"  Num landmarks (from metadata): {result['metadata']['num_landmarks']}")


if __name__ == "__main__":
    main()

