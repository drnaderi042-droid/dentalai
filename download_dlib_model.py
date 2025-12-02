"""
اسکریپت دانلود خودکار فایل shape predictor برای dlib
"""

import os
import sys
import urllib.request
import bz2
from pathlib import Path

# Fix encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# مسیر فایل
DLIB_MODEL_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
TARGET_DIR = "facial-landmark-detection"
TARGET_FILE = os.path.join(TARGET_DIR, "shape_predictor_68_face_landmarks.dat")
COMPRESSED_FILE = os.path.join(TARGET_DIR, "shape_predictor_68_face_landmarks.dat.bz2")

def download_dlib_model():
    """دانلود و extract فایل shape predictor"""
    
    # ایجاد پوشه اگر وجود ندارد
    Path(TARGET_DIR).mkdir(parents=True, exist_ok=True)
    
    # بررسی وجود فایل
    if os.path.exists(TARGET_FILE):
        print(f"✅ فایل از قبل وجود دارد: {TARGET_FILE}")
        return True
    
    print("=" * 60)
    print("📥 دانلود فایل shape predictor برای dlib...")
    print("=" * 60)
    print(f"URL: {DLIB_MODEL_URL}")
    print(f"مسیر نهایی: {TARGET_FILE}")
    print()
    
    try:
        # دانلود فایل فشرده
        print("⏳ در حال دانلود...")
        urllib.request.urlretrieve(DLIB_MODEL_URL, COMPRESSED_FILE)
        print(f"✅ دانلود کامل شد: {COMPRESSED_FILE}")
        
        # Extract فایل
        print("⏳ در حال extract...")
        with bz2.open(COMPRESSED_FILE, 'rb') as f_in:
            with open(TARGET_FILE, 'wb') as f_out:
                f_out.write(f_in.read())
        
        print(f"✅ Extract کامل شد: {TARGET_FILE}")
        
        # حذف فایل فشرده
        if os.path.exists(COMPRESSED_FILE):
            os.remove(COMPRESSED_FILE)
            print(f"🗑️  فایل فشرده حذف شد")
        
        # بررسی اندازه فایل
        file_size = os.path.getsize(TARGET_FILE) / (1024 * 1024)  # MB
        print(f"📊 اندازه فایل: {file_size:.2f} MB")
        
        print("=" * 60)
        print("✅ دانلود و نصب با موفقیت انجام شد!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"❌ خطا در دانلود: {e}")
        print()
        print("💡 راهنمای دستی:")
        print("1. فایل را از این آدرس دانلود کنید:")
        print(f"   {DLIB_MODEL_URL}")
        print("2. فایل را extract کنید (با WinRAR یا 7-Zip)")
        print(f"3. فایل shape_predictor_68_face_landmarks.dat را در پوشه {TARGET_DIR} قرار دهید")
        return False

if __name__ == "__main__":
    success = download_dlib_model()
    if success:
        print("\n✅ آماده استفاده است!")
    else:
        print("\n⚠️ لطفاً به صورت دستی دانلود کنید.")

