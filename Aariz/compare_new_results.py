"""
مقایسه نتایج جدید (بعد از رفع مشکل 256×256) با Ground Truth
"""

import json
import os
import numpy as np

# نتایج جدید از API/Direct (بعد از رفع مشکل)
new_results = {
    "A": {"x": 1311.79, "y": 1096.74},
    "ANS": {"x": 1343.66, "y": 1055.62},
    "Ar": {"x": 455.86, "y": 1097.25},
    "B": {"x": 1319.44, "y": 1530.52},
    "Co": {"x": 452.87, "y": 984.46},
    "Gn": {"x": 1330.72, "y": 1702.61},
    "Go": {"x": 592.70, "y": 1481.22},
    "LIA": {"x": 1274.30, "y": 1526.62},
    "LIT": {"x": 1362.97, "y": 1305.52},
    "LMT": {"x": 1120.89, "y": 1345.42},
    "LPM": {"x": 1177.63, "y": 1342.41},
    "Li": {"x": 1490.80, "y": 1353.22},
    "Ls": {"x": 1500.72, "y": 1202.83},
    "Me": {"x": 1294.08, "y": 1721.39},
    "N": {"x": 1172.49, "y": 524.15},
    "N`": {"x": 1229.54, "y": 560.03},
    "Or": {"x": 1103.48, "y": 811.08},
    "PNS": {"x": 806.68, "y": 1144.36},
    "Pn": {"x": 1577.09, "y": 935.09},
    "Po": {"x": 346.49, "y": 1017.58},
    "Pog": {"x": 1342.96, "y": 1647.08},
    "Pog`": {"x": 1480.05, "y": 1611.64},
    "R": {"x": 535.06, "y": 1270.67},
    "S": {"x": 523.44, "y": 747.98},
    "Sn": {"x": 1455.72, "y": 1084.01},
    "UIA": {"x": 1270.18, "y": 1123.53},
    "UIT": {"x": 1400.21, "y": 1330.91},
    "UMT": {"x": 1114.69, "y": 1329.79},
    "UPM": {"x": 1174.63, "y": 1334.04}
}

# Ground Truth از JSON
ground_truth = {
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

print("="*110)
print("🔍 مقایسه نتایج جدید (256×256) با Ground Truth")
print("="*110)
print(f"\nتصویر: cks2ip8fq29yq0yufc4scftj8.png")
print(f"اندازه: {image_width} × {image_height} پیکسل")
print(f"Pixel Size: {pixel_size} mm/pixel")
print(f"✅ Model input size: 256×256 (fixed!)")
print()

# محاسبه خطاها
errors = []
common_landmarks = set(new_results.keys()) & set(ground_truth.keys())

for lm_name in sorted(common_landmarks):
    pred = new_results[lm_name]
    gt = ground_truth[lm_name]
    
    error_px = np.sqrt((pred['x'] - gt['x'])**2 + (pred['y'] - gt['y'])**2)
    error_mm = error_px * pixel_size
    
    errors.append({
        'name': lm_name,
        'error_px': error_px,
        'error_mm': error_mm,
        'pred': pred,
        'gt': gt,
        'diff_x': pred['x'] - gt['x'],
        'diff_y': pred['y'] - gt['y']
    })

errors.sort(key=lambda x: x['error_px'], reverse=True)

# نمایش جدول
print(f"{'Landmark':<8} {'Pred X':<12} {'Pred Y':<12} {'GT X':<10} {'GT Y':<10} {'Diff X':<10} {'Diff Y':<10} {'Error (px)':<12} {'Error (mm)':<12}")
print("-"*110)

for err in errors:
    pred = err['pred']
    gt = err['gt']
    print(f"{err['name']:<8} {pred['x']:<12.2f} {pred['y']:<12.2f} {gt['x']:<10.0f} {gt['y']:<10.0f} {err['diff_x']:<10.2f} {err['diff_y']:<10.2f} {err['error_px']:<12.2f} {err['error_mm']:<12.4f}")

# آمار
error_values_px = [e['error_px'] for e in errors]
error_values_mm = [e['error_mm'] for e in errors]

print("\n" + "="*110)
print("📊 آمار خطاها")
print("="*110)
print(f"\n✅ تعداد لندمارک‌ها: {len(errors)}/29")
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

# مقایسه با قبل
print("\n" + "="*110)
print("📈 مقایسه با نتایج قبلی (512×256 - نادرست)")
print("="*110)
print(f"   قبل: MRE = 540.97 px (54.10 mm)")
print(f"   حالا: MRE = {np.mean(error_values_px):.2f} px ({np.mean(error_values_mm):.4f} mm)")
print(f"   بهبود: {((540.97 - np.mean(error_values_px)) / 540.97 * 100):.1f}% کاهش در خطای پیکسل")
print(f"   بهبود: {((54.10 - np.mean(error_values_mm)) / 54.10 * 100):.1f}% کاهش در خطای میلی‌متر")

# لندمارک‌های با بیشترین خطا
print("\n" + "="*110)
print("⚠️  لندمارک‌های با بیشترین خطا (10 تا اول)")
print("="*110)
for i, err in enumerate(errors[:10], 1):
    print(f"\n{i}. {err['name']}:")
    print(f"   خطا: {err['error_mm']:.4f} mm ({err['error_px']:.2f} پیکسل)")
    print(f"   Prediction: ({err['pred']['x']:.2f}, {err['pred']['y']:.2f})")
    print(f"   Ground Truth: ({err['gt']['x']:.0f}, {err['gt']['y']:.0f})")
    print(f"   تفاوت: X={err['diff_x']:.2f}px, Y={err['diff_y']:.2f}px")

# لندمارک‌های با کمترین خطا
print("\n" + "="*110)
print("✅ لندمارک‌های با کمترین خطا (10 تا اول)")
print("="*110)
for i, err in enumerate(errors[-10:], 1):
    print(f"\n{i}. {err['name']}:")
    print(f"   خطا: {err['error_mm']:.4f} mm ({err['error_px']:.2f} پیکسل)")
    print(f"   Prediction: ({err['pred']['x']:.2f}, {err['pred']['y']:.2f})")
    print(f"   Ground Truth: ({err['gt']['x']:.0f}, {err['gt']['y']:.0f})")

# SDR calculation
thresholds_mm = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
print("\n" + "="*110)
print("📊 Success Detection Rate (SDR)")
print("="*110)
for threshold_mm in thresholds_mm:
    success = sum(1 for e_mm in error_values_mm if e_mm <= threshold_mm)
    sdr = (success / len(error_values_mm)) * 100
    print(f"   SDR @ {threshold_mm}mm: {sdr:.2f}% ({success}/{len(error_values_mm)})")

# مقایسه SDR با قبل
print("\n" + "="*110)
print("📊 مقایسه SDR")
print("="*110)
print(f"   SDR @ 2mm - قبل: ~10% (با 512×512)")
print(f"   SDR @ 2mm - حالا: {(sum(1 for e_mm in error_values_mm if e_mm <= 2.0) / len(error_values_mm) * 100):.2f}%")
print(f"   SDR مورد انتظار (از آموزش): 72%")

# تحلیل
print("\n" + "="*110)
print("📋 تحلیل")
print("="*110)
mre_mm = np.mean(error_values_mm)
if mre_mm < 2.0:
    print(f"✅ عالی! MRE = {mre_mm:.4f}mm کمتر از 2mm است (آستانه قابل قبول بالینی)")
elif mre_mm < 3.0:
    print(f"✅ خوب! MRE = {mre_mm:.4f}mm در محدوده قابل قبول (کمتر از 3mm)")
else:
    print(f"⚠️  MRE = {mre_mm:.4f}mm بالاتر از حد مطلوب است")

sdr_2mm = sum(1 for e_mm in error_values_mm if e_mm <= 2.0) / len(error_values_mm) * 100
if sdr_2mm >= 70:
    print(f"✅ عالی! SDR @ 2mm = {sdr_2mm:.2f}% بسیار خوب است")
elif sdr_2mm >= 50:
    print(f"✅ قابل قبول! SDR @ 2mm = {sdr_2mm:.2f}%")
else:
    print(f"⚠️  SDR @ 2mm = {sdr_2mm:.2f}% می‌تواند بهتر شود")

print("\n" + "="*110)

# ذخیره نتایج
output = {
    'image_id': 'cks2ip8fq29yq0yufc4scftj8',
    'pixel_size': pixel_size,
    'image_size': {'width': image_width, 'height': image_height},
    'model_input_size': '256×256',
    'mre_pixels': float(np.mean(error_values_px)),
    'mre_mm': float(np.mean(error_values_mm)),
    'median_error_mm': float(np.median(error_values_mm)),
    'min_error_mm': float(np.min(error_values_mm)),
    'max_error_mm': float(np.max(error_values_mm)),
    'sdr_2mm': float(sdr_2mm),
    'errors': {k: {'pixels': float(v['error_px']), 'mm': float(v['error_mm'])} 
              for k, v in zip([e['name'] for e in errors], errors)}
}

with open('new_comparison_result.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"💾 نتایج در new_comparison_result.json ذخیره شد")
print("="*110)

