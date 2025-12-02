"""
مقایسه سریع نتایج Frontend با Ground Truth
"""

import json

# نتایج Frontend
frontend = {
    "A": {"x": 311.34375, "y": 1116.845703125},
    "ANS": {"x": 1233.84375, "y": 599.70703125},
    "N": {"x": 1329.9375, "y": 1712.20703125},
    "S": {"x": 987.84375, "y": 1499.267578125},
    "B": {"x": 1329.9375, "y": 1725.244140625},
    "Pog": {"x": 1329.9375, "y": 1725.244140625},
    "Me": {"x": 1283.8125, "y": 1729.58984375},
    "Or": {"x": 388.21875, "y": 1273.291015625},
    "Po": {"x": 1356.84375, "y": 1290.673828125},
    "PNS": {"x": 453.5625, "y": 1290.673828125},
}

# Ground Truth از JSON
ground_truth = {
    "A": {"x": 1315, "y": 1086},
    "ANS": {"x": 1338, "y": 1048},
    "N": {"x": 1183, "y": 508},
    "S": {"x": 499, "y": 758},
    "B": {"x": 1333, "y": 1564},
    "Pog": {"x": 1348, "y": 1663},
    "Me": {"x": 1297, "y": 1733},
    "Or": {"x": 1112, "y": 790},
    "Po": {"x": 291, "y": 958},
    "PNS": {"x": 793, "y": 1138},
}

print("="*80)
print("🔍 مقایسه سریع Frontend vs Ground Truth")
print("="*80)
print("\nتصویر: cks2ip8fq29yq0yufc4scftj8.png (1968 × 2225)\n")

print(f"{'Landmark':<10} {'Frontend':<25} {'Ground Truth':<25} {'Error (px)':<15} {'Error (mm)':<15}")
print("-"*90)

errors = []
for lm in sorted(frontend.keys()):
    if lm not in ground_truth:
        continue
    
    f = frontend[lm]
    gt = ground_truth[lm]
    
    error_px = ((f['x'] - gt['x'])**2 + (f['y'] - gt['y'])**2)**0.5
    error_mm = error_px * 0.1  # pixel_size = 0.1
    
    errors.append((lm, error_px, error_mm, f, gt))
    
    print(f"{lm:<10} ({f['x']:>7.1f}, {f['y']:>7.1f})  ({gt['x']:>7.1f}, {gt['y']:>7.1f})  {error_px:>10.1f} px  {error_mm:>10.2f} mm")

errors.sort(key=lambda x: x[1], reverse=True)

print("\n" + "="*80)
print("📊 خلاصه:")
print("="*80)
mre_px = sum(e[1] for e in errors) / len(errors)
mre_mm = sum(e[2] for e in errors) / len(errors)

print(f"\n✅ MRE: {mre_px:.2f} پیکسل = {mre_mm:.4f} mm")
print(f"\n⚠️  بیشترین خطا: {errors[0][0]} - {errors[0][1]:.1f}px ({errors[0][2]:.2f}mm)")
print(f"✅ کمترین خطا: {errors[-1][0]} - {errors[-1][1]:.1f}px ({errors[-1][2]:.2f}mm)")

