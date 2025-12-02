"""
بررسی اینکه آیا وزن‌های P1/P2 برای همه لایه‌ها کپی شده‌اند
"""
import torch

p1p2 = torch.load('models/hrnet_p1p2_heatmap_best.pth', map_location='cpu', weights_only=False)
combined = torch.load('checkpoint_best_768_combined_31.pth', map_location='cpu', weights_only=False)

p1p2_sd = p1p2.get('model_state_dict', p1p2)
combined_sd = combined.get('model_state_dict', combined)

p1p2_weight = p1p2_sd['heatmap_decoder.16.weight']
p1p2_bias = p1p2_sd['heatmap_decoder.16.bias']

print("="*80)
print("Checking P1/P2 weights in all final_layers")
print("="*80)

for i in range(4):
    if i == 0:
        w_key = 'final_layers.0.3.weight'
        b_key = 'final_layers.0.3.bias'
    else:
        w_key = f'final_layers.{i}.4.weight'
        b_key = f'final_layers.{i}.4.bias'
    
    w = combined_sd[w_key][29:31]
    b = combined_sd[b_key][29:31]
    
    w_match = torch.equal(w, p1p2_weight)
    b_match = torch.equal(b, p1p2_bias)
    
    print(f"Layer {i}:")
    print(f"  Weight match: {w_match}")
    print(f"  Bias match: {b_match}")
    if not w_match or not b_match:
        print(f"  [WARNING] P1/P2 weights not copied for layer {i}!")

print("\n" + "="*80)
print("Summary: All 4 layers should have P1/P2 weights copied")
print("="*80)




