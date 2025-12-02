#!/usr/bin/env python3
"""
محاسبه ضریب scaling برای تصحیح مختصات landmarks

این اسکریپت:
1. ابعاد تصویر اصلی را می‌خواند
2. نسبت scaling را محاسبه می‌کند
3. مختصات را تصحیح می‌کند
"""

import json
import sys
from PIL import Image
from pathlib import Path

def get_image_dimensions(image_path):
    """دریافت ابعاد تصویر"""
    try:
        with Image.open(image_path) as img:
            return img.size  # (width, height)
    except Exception as e:
        print(f"❌ خطا در خواندن تصویر: {e}")
        return None

def calculate_scaling_factor(landmarks_json, image_path):
    """
    محاسبه ضریب scaling با استفاده از محدوده landmarks
    
    رویکرد 1: محاسبه از محدوده landmarks
    رویکرد 2: استفاده از ابعاد استاندارد
    """
    # دریافت ابعاد واقعی تصویر
    dimensions = get_image_dimensions(image_path)
    if not dimensions:
        return None
    
    actual_width, actual_height = dimensions
    
    print(f"📏 ابعاد واقعی تصویر: {actual_width} × {actual_height} پیکسل")
    
    # استخراج محدوده landmarks
    landmarks = landmarks_json.get('landmarks', {})
    if not landmarks:
        print("❌ landmarks یافت نشد")
        return None
    
    # پیدا کردن min/max مختصات
    x_coords = [lm['x'] for lm in landmarks.values()]
    y_coords = [lm['y'] for lm in landmarks.values()]
    
    landmark_width = max(x_coords) - min(x_coords)
    landmark_height = max(y_coords) - min(y_coords)
    
    print(f"📍 محدوده landmarks: {landmark_width:.0f} × {landmark_height:.0f}")
    print(f"   X: {min(x_coords):.0f} - {max(x_coords):.0f}")
    print(f"   Y: {min(y_coords):.0f} - {max(y_coords):.0f}")
    
    # محاسبه ضریب scaling
    # معمولاً تصاویر cephalometric در محدوده 600-2000 پیکسل هستند
    # و AI مدل‌ها معمولاً به 512 یا 1024 scale می‌کنند
    
    # روش 1: نسبت مستقیم بر اساس max مختصات
    max_landmark_coord = max(max(x_coords), max(y_coords))
    max_actual_dim = max(actual_width, actual_height)
    
    scaling_factor_1 = max_actual_dim / max_landmark_coord
    
    # روش 2: نسبت بر اساس width
    scaling_factor_2 = actual_width / max(x_coords)
    
    # روش 3: نسبت بر اساس height
    scaling_factor_3 = actual_height / max(y_coords)
    
    print(f"\n🔢 ضرایب محاسبه شده:")
    print(f"   روش 1 (max dimension): {scaling_factor_1:.4f}")
    print(f"   روش 2 (width based):   {scaling_factor_2:.4f}")
    print(f"   روش 3 (height based):  {scaling_factor_3:.4f}")
    
    # استفاده از میانگین روش 2 و 3 (معمولاً دقیق‌تر است)
    recommended_scaling = (scaling_factor_2 + scaling_factor_3) / 2
    
    print(f"\n✅ ضریب توصیه شده: {recommended_scaling:.4f}")
    
    return {
        'scaling_factor': recommended_scaling,
        'method_1': scaling_factor_1,
        'method_2': scaling_factor_2,
        'method_3': scaling_factor_3,
        'actual_dimensions': (actual_width, actual_height),
        'landmark_range': (landmark_width, landmark_height)
    }

def scale_landmarks(landmarks_json, scaling_factor):
    """اعمال ضریب scaling به landmarks"""
    landmarks = landmarks_json.get('landmarks', {})
    
    scaled_landmarks = {}
    for name, coords in landmarks.items():
        scaled_landmarks[name] = {
            'x': round(coords['x'] * scaling_factor, 2),
            'y': round(coords['y'] * scaling_factor, 2)
        }
    
    return {
        'landmarks': scaled_landmarks,
        'confidence': landmarks_json.get('confidence', 0),
        'notes': landmarks_json.get('notes', ''),
        'scaling_info': {
            'factor': scaling_factor,
            'applied': True
        }
    }

def main():
    print("🔍 محاسبه ضریب Scaling برای Landmarks")
    print("=" * 60)
    
    # دریافت ورودی‌ها
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("\n📷 مسیر تصویر: ").strip()
    
    if len(sys.argv) > 2:
        json_path = sys.argv[2]
    else:
        json_path = input("📄 مسیر فایل JSON (یا Enter برای ورود دستی): ").strip()
    
    # بررسی تصویر
    if not Path(image_path).exists():
        print(f"❌ تصویر یافت نشد: {image_path}")
        return
    
    # خواندن JSON
    if json_path and Path(json_path).exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            landmarks_json = json.load(f)
    else:
        print("\n📝 لطفاً JSON landmarks را وارد کنید:")
        print("(می‌توانید مستقیماً paste کنید و Enter بزنید)")
        json_text = input()
        try:
            landmarks_json = json.loads(json_text)
        except json.JSONDecodeError:
            print("❌ JSON نامعتبر است")
            return
    
    # محاسبه scaling
    print("\n" + "=" * 60)
    scaling_info = calculate_scaling_factor(landmarks_json, image_path)
    
    if not scaling_info:
        return
    
    # اعمال scaling
    scaling_factor = scaling_info['scaling_factor']
    scaled_result = scale_landmarks(landmarks_json, scaling_factor)
    
    # نمایش نتایج
    print("\n" + "=" * 60)
    print("📊 نتایج Scaled:")
    print("=" * 60)
    print(json.dumps(scaled_result, indent=2, ensure_ascii=False))
    
    # ذخیره در فایل
    output_file = "landmarks_scaled.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scaled_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در {output_file} ذخیره شد")
    
    # نمایش چند نمونه
    print("\n📍 مقایسه چند نقطه:")
    print("-" * 60)
    for name in ['S', 'N', 'A', 'B']:
        if name in landmarks_json['landmarks']:
            original = landmarks_json['landmarks'][name]
            scaled = scaled_result['landmarks'][name]
            print(f"{name:3s}: ({original['x']:6.1f}, {original['y']:6.1f}) → "
                  f"({scaled['x']:7.1f}, {scaled['y']:7.1f})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  توسط کاربر متوقف شد")
    except Exception as e:
        print(f"\n❌ خطا: {e}")

