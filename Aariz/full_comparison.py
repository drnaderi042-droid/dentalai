"""
مقایسه کامل نتایج Frontend با Ground Truth
"""

import json
import numpy as np

# نتایج Frontend (از console logs)
frontend = {
    "A": {"x": 311.34375, "y": 1116.845703125},
    "ANS": {"x": 1233.84375, "y": 599.70703125},
    "Ar": {"x": 311.34375, "y": 1082.080078125},
    "B": {"x": 1329.9375, "y": 1725.244140625},
    "Co": {"x": 434.34375, "y": 1325.439453125},
    "Gn": {"x": 280.59375, "y": 925.634765625},
    "Go": {"x": 295.96875, "y": 1377.587890625},
    "LIA": {"x": 1310.71875, "y": 1707.861328125},
    "LIT": {"x": 295.96875, "y": 995.166015625},
    "LMT": {"x": 1249.21875, "y": 1603.564453125},
    "LPM": {"x": 388.21875, "y": 1029.931640625},
    "Li": {"x": 1499.0625, "y": 1208.10546875},
    "Ls": {"x": 1264.59375, "y": 425.87890625},
    "Me": {"x": 1283.8125, "y": 1729.58984375},
    "N": {"x": 1329.9375, "y": 1712.20703125},
    "N`": {"x": 295.96875, "y": 1520.99609375},
    "Or": {"x": 388.21875, "y": 1273.291015625},
    "PNS": {"x": 453.5625, "y": 1290.673828125},
    "Pn": {"x": 1499.0625, "y": 1208.10546875},
    "Po": {"x": 1356.84375, "y": 1290.673828125},
    "Pog": {"x": 1329.9375, "y": 1725.244140625},
    "Pog`": {"x": 219.09375, "y": 856.103515625},
    "R": {"x": 588.09375, "y": 1186.376953125},
    "S": {"x": 987.84375, "y": 1499.267578125},
    "Sn": {"x": 1422.1875, "y": 1273.291015625},
    "UIA": {"x": 311.34375, "y": 1103.80859375},
    "UIT": {"x": 238.3125, "y": 995.166015625},
    "UMT": {"x": 418.96875, "y": 1034.27734375},
    "UPM": {"x": 1249.21875, "y": 1625.29296875}
}

# Ground Truth از JSON
gt_raw = {
    "A": {"x": 1315, "y": 1086},
    "ANS": {"x": 1338, "y": 1048},
    "B": {"x": 1333, "y": 1564},
    "Me": {"x": 1297, "y": 1733},
    "N": {"x": 1183, "y": 508},
    "Or": {"x": 1112, "y": 790},
    "Pog": {"x": 1348, "y": 1663},
    "PNS": {"x": 793, "y": 1138},
    "Pn": {"x": 1585, "y": 946},
    "R": {"x": 523, "y": 1265},
    "S": {"x": 499, "y": 758},
    "Ar": {"x": 445, "y": 1061},
    "Co": {"x": 449, "y": 980},
    "Gn": {"x": 1336, "y": 1707},
    "Go": {"x": 593, "y": 1496},
    "Po": {"x": 291, "y": 958},
    "LPM": {"x": 1189, "y": 1329},
    "LIT": {"x": 1371, "y": 1308},
    "LMT": {"x": 1147, "y": 1334},
    "UPM": {"x": 1177, "y": 1339},
    "UIA": {"x": 1288, "y": 1106},
    "UIT": {"x": 1400, "y": 1334},
    "UMT": {"x": 1119, "y": 1339},
    "LIA": {"x": 1288, "y": 1556},
    "Li": {"x": 1501, "y": 1369},
    "Ls": {"x": 1508, "y": 1212},
    "N`": {"x": 1240, "y": 557},
    "Pog`": {"x": 1488, "y": 1608},
    "Sn": {"x": 1458, "y": 1067}
}

pixel_size = 0.1  # mm/pixel
image_width = 1968
image_height = 2225

print("="*100)
print("🔍 مقایسه کامل Frontend vs Ground Truth")
print("="*100)
print(f"\nتصویر: cks2ip8fq29yq0yufc4scftj8.png")
print(f"اندازه: {image_width} × {image_height} پیکسل")
print(f"Pixel Size: {pixel_size} mm/pixel\n")

# محاسبه خطاها
errors = []
common_landmarks = set(frontend.keys()) & set(gt_raw.keys())

for lm_name in sorted(common_landmarks):
    f = frontend[lm_name]
    gt = gt_raw[lm_name]
    
    error_px = np.sqrt((f['x'] - gt['x'])**2 + (f['y'] - gt['y'])**2)
    error_mm = error_px * pixel_size
    
    errors.append({
        'name': lm_name,
        'error_px': error_px,
        'error_mm': error_mm,
        'frontend': f,
        'gt': gt,
        'diff_x': f['x'] - gt['x'],
        'diff_y': f['y'] - gt['y']
    })

errors.sort(key=lambda x: x['error_px'], reverse=True)

# نمایش جدول
print(f"{'Landmark':<8} {'Frontend X':<12} {'Frontend Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<10} {'Diff Y':<10} {'Error (px)':<12} {'Error (mm)':<12}")
print("-"*100)

for err in errors:
    f = err['frontend']
    gt = err['gt']
    print(f"{err['name']:<8} {f['x']:<12.2f} {f['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {err['diff_x']:<10.2f} {err['diff_y']:<10.2f} {err['error_px']:<12.2f} {err['error_mm']:<12.4f}")

# آمار
error_values_px = [e['error_px'] for e in errors]
error_values_mm = [e['error_mm'] for e in errors]

print("\n" + "="*100)
print("📊 آمار خطاها")
print("="*100)
print(f"\n✅ تعداد لندمارک‌ها: {len(errors)}")
print(f"\n📊 خطا بر حسب پیکسل:")
print(f"   میانگین (MRE): {np.mean(error_values_px):.2f} پیکسل")
print(f"   میانه: {np.median(error_values_px):.2f} پیکسل")
print(f"   کمینه: {np.min(error_values_px):.2f} پیکسل")
print(f"   بیشینه: {np.max(error_values_px):.2f} پیکسل")
print(f"   انحراف معیار: {np.std(error_values_px):.2f} پیکسل")

print(f"\n📊 خطا بر حسب میلی‌متر:")
print(f"   میانگین (MRE): {np.mean(error_values_mm):.4f} mm")
print(f"   میانه: {np.median(error_values_mm):.4f} mm")
print(f"   کمینه: {np.min(error_values_mm):.4f} mm")
print(f"   بیشینه: {np.max(error_values_mm):.4f} mm")
print(f"   انحراف معیار: {np.std(error_values_mm):.4f} mm")

# لندمارک‌های با بیشترین خطا
print("\n" + "="*100)
print("⚠️  لندمارک‌های با بیشترین خطا (10 تا اول)")
print("="*100)
for i, err in enumerate(errors[:10], 1):
    print(f"\n{i}. {err['name']}:")
    print(f"   خطا: {err['error_mm']:.4f} mm ({err['error_px']:.2f} پیکسل)")
    print(f"   Frontend: ({err['frontend']['x']:.2f}, {err['frontend']['y']:.2f})")
    print(f"   Ground Truth: ({err['gt']['x']:.0f}, {err['gt']['y']:.0f})")
    print(f"   تفاوت: X={err['diff_x']:.2f}px, Y={err['diff_y']:.2f}px")

# SDR
thresholds_mm = [1.0, 2.0, 2.5, 3.0, 4.0]
print("\n" + "="*100)
print("📊 Success Detection Rate (SDR)")
print("="*100)
for threshold_mm in thresholds_mm:
    success = sum(1 for e_mm in error_values_mm if e_mm <= threshold_mm)
    sdr = (success / len(error_values_mm)) * 100
    print(f"   SDR @ {threshold_mm}mm: {sdr:.2f}% ({success}/{len(error_values_mm)})")

print("\n" + "="*100)

