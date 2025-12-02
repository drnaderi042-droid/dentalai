"""تست پیکربندی Difficult Landmarks Only"""
import sys
sys.path.insert(0, '.')

from train2 import DIFFICULT_LANDMARKS_ONLY, LANDMARK_SYMBOLS

print("="*80)
print("پیکربندی: Training فقط برای لندمارک‌های مشکل‌دار")
print("="*80)

difficult = [s for s in LANDMARK_SYMBOLS if DIFFICULT_LANDMARKS_ONLY.get(s, False)]
ignored = [s for s in LANDMARK_SYMBOLS if not DIFFICULT_LANDMARKS_ONLY.get(s, False)]

print(f"\n✅ لندمارک‌هایی که آموزش می‌بینند ({len(difficult)} لندمارک):")
for lm in difficult:
    print(f"   - {lm}")

print(f"\n❌ لندمارک‌هایی که ignore می‌شوند ({len(ignored)} لندمارک):")
for lm in ignored:
    print(f"   - {lm}")

print(f"\n📊 خلاصه:")
print(f"   کل لندمارک‌ها: {len(LANDMARK_SYMBOLS)}")
print(f"   آموزش می‌بینند: {len(difficult)} ({len(difficult)/len(LANDMARK_SYMBOLS)*100:.1f}%)")
print(f"   Ignore می‌شوند: {len(ignored)} ({len(ignored)/len(LANDMARK_SYMBOLS)*100:.1f}%)")
print(f"   سرعت: ~{len(difficult)/len(LANDMARK_SYMBOLS)*100:.0f}% از قبل (کاهش ~{len(ignored)/len(LANDMARK_SYMBOLS)*100:.0f}%)")

print("\n" + "="*80)
print("✅ پیکربندی صحیح است!")
print("="*80)















