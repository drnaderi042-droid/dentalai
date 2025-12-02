"""
بررسی ساختار مدل P1/P2
"""
import torch

ckpt = torch.load('models/hrnet_p1p2_heatmap_best.pth', map_location='cpu', weights_only=False)
sd = ckpt.get('model_state_dict', ckpt)

print("="*80)
print("P1/P2 Model Structure")
print("="*80)

decoder_keys = [k for k in sd.keys() if 'heatmap_decoder' in k]
print(f"\nDecoder keys ({len(decoder_keys)}):")
for k in sorted(decoder_keys):
    print(f"  {k}: {sd[k].shape}")

# پیدا کردن آخرین Conv2d (خروجی)
output_keys = [k for k in decoder_keys if 'heatmap_decoder.16' in k or 'heatmap_decoder.17' in k]
print(f"\nOutput layer keys (last Conv2d):")
for k in sorted(output_keys):
    print(f"  {k}: {sd[k].shape}")




