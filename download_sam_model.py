"""
Download SAM model .pth file
"""
import urllib.request
import os
import sys

def download_sam_model():
    url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    target = "TeethDreamer/ckpt/sam_vit_b_01ec64.pth"
    
    # ایجاد پوشه
    os.makedirs(os.path.dirname(target), exist_ok=True)
    
    # بررسی وجود فایل
    if os.path.exists(target):
        size_mb = os.path.getsize(target) / (1024 * 1024)
        print(f"File already exists: {target}")
        print(f"Size: {size_mb:.2f} MB")
        response = input("Do you want to download again? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download.")
            return True
    
    try:
        print(f"Downloading SAM model from: {url}")
        print(f"Target: {target}")
        print("This may take a few minutes (375 MB)...")
        
        # دانلود با progress bar
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")
        
        urllib.request.urlretrieve(url, target, show_progress)
        print()  # New line after progress
        
        # بررسی
        if os.path.exists(target):
            size_mb = os.path.getsize(target) / (1024 * 1024)
            print(f"Download completed successfully!")
            print(f"File size: {size_mb:.2f} MB")
            return True
        else:
            print("Error: File was not downloaded!")
            return False
            
    except Exception as e:
        print(f"Error during download: {e}")
        print("\nAlternative: Download manually from:")
        print(f"   URL: {url}")
        print(f"   Save to: {target}")
        return False

if __name__ == "__main__":
    success = download_sam_model()
    sys.exit(0 if success else 1)













