"""
محاسبه خطای پیکسل از روی خطای میلی‌متر
"""

import pandas as pd
import numpy as np
import os

def calculate_mm_to_pixels(error_mm, pixel_size_mm):
    """
    تبدیل خطای میلی‌متر به پیکسل
    
    Args:
        error_mm: خطا بر حسب میلی‌متر
        pixel_size_mm: اندازه هر پیکسل بر حسب میلی‌متر
    
    Returns:
        خطا بر حسب پیکسل
    """
    return error_mm / pixel_size_mm

def get_pixel_size_statistics(dataset_path="Aariz"):
    """دریافت آمار pixel size از dataset"""
    csv_path = os.path.join(dataset_path, "cephalogram_machine_mappings.csv")
    
    if not os.path.exists(csv_path):
        print(f"⚠️  CSV file not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if 'pixel_size' not in df.columns:
            print("⚠️  'pixel_size' column not found in CSV")
            return None
        
        pixel_sizes = df['pixel_size'].values
        
        return {
            'mean': np.mean(pixel_sizes),
            'median': np.median(pixel_sizes),
            'min': np.min(pixel_sizes),
            'max': np.max(pixel_sizes),
            'std': np.std(pixel_sizes),
            'all': pixel_sizes
        }
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

def main():
    print("="*80)
    print("🔢 محاسبه خطای پیکسل از روی خطای میلی‌متر")
    print("="*80)
    
    # دریافت آمار pixel size از dataset
    stats = get_pixel_size_statistics()
    
    if stats:
        print("\n📊 آمار Pixel Size در Dataset:")
        print(f"   میانگین: {stats['mean']:.4f} mm/pixel")
        print(f"   میانه: {stats['median']:.4f} mm/pixel")
        print(f"   کمینه: {stats['min']:.4f} mm/pixel")
        print(f"   بیشینه: {stats['max']:.4f} mm/pixel")
        print(f"   انحراف معیار: {stats['std']:.4f} mm/pixel")
    else:
        # استفاده از مقادیر پیش‌فرض (معمول در رادیولوژی)
        print("\n⚠️  استفاده از مقادیر پیش‌فرض (چون CSV یافت نشد)")
        print("   معمولاً pixel size در رادیولوژی بین 0.1 تا 0.2 mm/pixel است")
    
    # محاسبه برای خطای 3mm
    error_mm = 3.0
    image_size = 2000  # پیکسل
    
    print("\n" + "="*80)
    print(f"📏 تبدیل {error_mm}mm خطا به پیکسل برای تصویر {image_size}×{image_size}")
    print("="*80)
    
    if stats:
        # محاسبه برای pixel size های مختلف
        pixel_sizes_to_check = [
            stats['mean'],
            stats['median'],
            stats['min'],
            stats['max']
        ]
        
        print(f"\n{'Pixel Size (mm/pixel)':<25} {'3mm = پیکسل':<20} {'% از تصویر':<15}")
        print("-"*60)
        
        for ps in pixel_sizes_to_check:
            pixels = calculate_mm_to_pixels(error_mm, ps)
            percentage = (pixels / image_size) * 100
            label = f"{'میانگین' if ps == stats['mean'] else 'میانه' if ps == stats['median'] else 'کمینه' if ps == stats['min'] else 'بیشینه'}"
            print(f"{ps:.4f} ({label}):{'':<10} {pixels:.2f} px{'':<10} {percentage:.2f}%")
        
        # محاسبه برای تمام pixel sizes
        all_errors_pixels = [calculate_mm_to_pixels(error_mm, ps) for ps in stats['all']]
        print(f"\n📈 برای تمام تصاویر در dataset:")
        print(f"   میانگین: {np.mean(all_errors_pixels):.2f} پیکسل")
        print(f"   میانه: {np.median(all_errors_pixels):.2f} پیکسل")
        print(f"   کمینه: {np.min(all_errors_pixels):.2f} پیکسل")
        print(f"   بیشینه: {np.max(all_errors_pixels):.2f} پیکسل")
    else:
        # محاسبه با مقادیر پیش‌فرض
        typical_pixel_sizes = [0.1, 0.15, 0.2, 0.25]
        
        print(f"\n{'Pixel Size (mm/pixel)':<25} {'3mm = پیکسل':<20} {'% از تصویر':<15}")
        print("-"*60)
        
        for ps in typical_pixel_sizes:
            pixels = calculate_mm_to_pixels(error_mm, ps)
            percentage = (pixels / image_size) * 100
            print(f"{ps:.2f}:{'':<20} {pixels:.2f} px{'':<10} {percentage:.2f}%")
    
    # محاسبه برای خطاهای مختلف
    print("\n" + "="*80)
    print("📊 جدول تبدیل خطاهای مختلف (با استفاده از pixel size میانگین)")
    print("="*80)
    
    if stats:
        avg_pixel_size = stats['mean']
    else:
        avg_pixel_size = 0.15  # پیش‌فرض
    
    errors_mm = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    
    print(f"\nPixel Size استفاده شده: {avg_pixel_size:.4f} mm/pixel")
    print(f"\n{'خطا (mm)':<15} {'خطا (پیکسل)':<20} {'% از تصویر 2000px':<20}")
    print("-"*55)
    
    for err_mm in errors_mm:
        pixels = calculate_mm_to_pixels(err_mm, avg_pixel_size)
        percentage = (pixels / image_size) * 100
        print(f"{err_mm:<15} {pixels:<20.2f} {percentage:<20.2f}")
    
    # تفسیر
    print("\n" + "="*80)
    print("💡 تفسیر:")
    print("="*80)
    
    error_3mm_pixels = calculate_mm_to_pixels(3.0, avg_pixel_size if stats else 0.15)
    percentage_3mm = (error_3mm_pixels / image_size) * 100
    
    print(f"\n✅ خطای {error_mm}mm در یک تصویر {image_size}×{image_size}:")
    print(f"   = {error_3mm_pixels:.2f} پیکسل")
    print(f"   = {percentage_3mm:.2f}% از عرض/ارتفاع تصویر")
    
    if error_3mm_pixels > 50:
        print(f"\n⚠️  هشدار: این خطا نسبتاً بزرگ است!")
        print(f"   پیشنهاد می‌شود که:")
        print(f"   - مدل را fine-tune کنید")
        print(f"   - تنظیمات آموزش را بهبود دهید")
        print(f"   - داده‌های آموزش را بررسی کنید")
    elif error_3mm_pixels > 20:
        print(f"\n⚠️  توجه: این خطا قابل قبول است اما قابل بهبود")
    else:
        print(f"\n✅ این خطا نسبتاً کوچک است")
    
    print("\n" + "="*80)

if __name__ == '__main__':
    main()

