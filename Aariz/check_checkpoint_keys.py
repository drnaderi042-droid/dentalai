"""
بررسی کلیدهای موجود در checkpoint
"""
import torch
from pathlib import Path

checkpoint_path = 'models/hrnet_p1p2_heatmap_best.pth'
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

state_dict = checkpoint['model_state_dict']

print(f"Total keys in checkpoint: {len(state_dict.keys())}")
print(f"\nFirst 20 keys:")
for i, k in enumerate(list(state_dict.keys())[:20]):
    print(f"  {i+1}. {k}")

print(f"\nKeys by prefix:")
backbone_keys = [k for k in state_dict.keys() if k.startswith('backbone')]
heatmap_keys = [k for k in state_dict.keys() if k.startswith('heatmap_decoder')]

print(f"  backbone keys: {len(backbone_keys)}")
print(f"  heatmap_decoder keys: {len(heatmap_keys)}")

if backbone_keys:
    print(f"\nFirst 5 backbone keys:")
    for k in backbone_keys[:5]:
        print(f"    {k}")
else:
    print("\n[WARNING] No backbone keys in checkpoint!")

if heatmap_keys:
    print(f"\nFirst 5 heatmap_decoder keys:")
    for k in heatmap_keys[:5]:
        print(f"    {k}")






