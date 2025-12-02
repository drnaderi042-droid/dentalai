"""
اسکریپت برای اضافه کردن Weighted Loss به train2.py

این اسکریپت تغییرات لازم را برای اعمال weighted loss به train2.py اعمال می‌کند.
"""

import os
import re

def apply_weighted_loss_changes():
    """اعمال تغییرات برای weighted loss"""
    
    train2_path = os.path.join(os.path.dirname(__file__), 'train2.py')
    
    if not os.path.exists(train2_path):
        print(f"[ERROR] فایل train2.py یافت نشد: {train2_path}")
        return
    
    # خواندن فایل
    with open(train2_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # بررسی اینکه آیا قبلاً weighted loss اضافه شده
    if 'DIFFICULT_LANDMARK_WEIGHTS' in content:
        print("[INFO] Weighted loss قبلاً اضافه شده است!")
        return
    
    # اضافه کردن وزن‌ها بعد از imports
    weights_code = """
# Weighted Loss برای لندمارک‌های مشکل‌دار
DIFFICULT_LANDMARK_WEIGHTS = {
    'UMT': 2.5,   # Upper Molar Tip - بیشترین خطا
    'UPM': 2.5,   # Upper Premolar
    'R': 2.0,     # Ramus point
    'Ar': 1.8,    # Articulare
    'Go': 1.8,    # Gonion
    'LMT': 1.6,   # Lower Molar Tip
    'LPM': 1.4,   # Lower Premolar
    'Or': 1.3,    # Orbitale
    'Co': 1.2,    # Condylion
    'PNS': 1.2,   # Posterior Nasal Spine
}

LANDMARK_SYMBOLS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

def calculate_weighted_loss(outputs, targets, landmark_symbols, base_criterion, device):
    \"\"\"
    محاسبه weighted loss برای لندمارک‌های مشکل‌دار
    \"\"\"
    batch_size = outputs.shape[0]
    num_landmarks = outputs.shape[1]
    
    total_loss = 0.0
    total_weight = 0.0
    
    for i in range(num_landmarks):
        if i >= len(landmark_symbols):
            weight = 1.0
        else:
            landmark_name = landmark_symbols[i]
            weight = DIFFICULT_LANDMARK_WEIGHTS.get(landmark_name, 1.0)
        
        landmark_output = outputs[:, i:i+1, :, :]
        landmark_target = targets[:, i:i+1, :, :]
        
        landmark_loss = base_criterion(landmark_output, landmark_target)
        
        total_loss += weight * landmark_loss
        total_weight += weight
    
    weighted_loss = total_loss / total_weight if total_weight > 0 else total_loss / num_landmarks
    
    return weighted_loss


"""
    
    # پیدا کردن جای مناسب برای اضافه کردن (بعد از imports و قبل از classes)
    import_end_pattern = r'(from.*\n|import.*\n)'
    imports_match = list(re.finditer(import_end_pattern, content))
    
    if imports_match:
        last_import_end = imports_match[-1].end()
        # اضافه کردن کد بعد از imports
        content = content[:last_import_end] + weights_code + content[last_import_end:]
    
    # جایگزین کردن loss calculation در train_epoch
    # پیدا کردن بخش loss calculation
    loss_pattern1 = r'if use_adaptive_wing:\s+loss = criterion\(outputs_resized, targets\)'
    loss_pattern2 = r'if use_adaptive_wing:\s+loss = criterion\(outputs, targets\)'
    
    replacement1 = """if use_adaptive_wing:
                    # استفاده از weighted loss برای لندمارک‌های مشکل‌دار
                    loss = calculate_weighted_loss(
                        outputs_resized, targets,
                        LANDMARK_SYMBOLS,
                        criterion,
                        device
                    )"""
    
    replacement2 = """if use_adaptive_wing:
                # استفاده از weighted loss برای لندمارک‌های مشکل‌دار
                loss = calculate_weighted_loss(
                    outputs, targets,
                    LANDMARK_SYMBOLS,
                    criterion,
                    device
                )"""
    
    # جایگزینی
    content = re.sub(loss_pattern1, replacement1, content, flags=re.MULTILINE)
    content = re.sub(loss_pattern2, replacement2, content, flags=re.MULTILINE)
    
    # ذخیره فایل
    backup_path = train2_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(open(train2_path, 'r', encoding='utf-8').read())
        print(f"[OK] Backup ایجاد شد: {backup_path}")
    
    with open(train2_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("[OK] تغییرات اعمال شد!")
    print("\nتغییرات اعمال شده:")
    print("  1. اضافه شدن DIFFICULT_LANDMARK_WEIGHTS")
    print("  2. اضافه شدن تابع calculate_weighted_loss")
    print("  3. جایگزینی loss calculation در train_epoch")
    print("\nبرای بازگشت به نسخه قبلی، از backup استفاده کنید:")
    print(f"  copy {backup_path} {train2_path}")

if __name__ == "__main__":
    print("="*80)
    print("اعمال Weighted Loss به train2.py")
    print("="*80)
    apply_weighted_loss_changes()















