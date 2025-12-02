"""
بررسی لندمارک‌های مفقود برای آنالیزهای سفالومتری
"""
import sys
import codecs
import json

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# لندمارک‌های موجود در مدل Aariz (29 لندمارک)
AARIZ_LANDMARKS = [
    "A", "ANS", "B", "Me", "N", "Or", "Pog", "PNS", "Pn", "R",
    "S", "Ar", "Co", "Gn", "Go", "Po", "LPM", "LIT", "LMT", "UPM",
    "UIA", "UIT", "UMT", "LIA", "Li", "Ls", "N`", "Pog`", "Sn"
]

# لندمارک‌های مورد نیاز برای هر آنالیز (از کد calculateMeasurements)
REQUIRED_LANDMARKS = {
    "Steiner": {
        "SNA": ["S", "N", "A"],
        "SNB": ["S", "N", "B"],
        "ANB": ["S", "N", "A", "B"],
        "GoGn-SN": ["S", "N", "Go", "Gn"],
        "U1-SN": ["S", "N", "U1"],
        "L1-MP": ["Go", "Me", "L1"],
    },
    "Ricketts": {
        "Facial Axis": ["Ba", "Na", "Pt", "Gn"],
        "Facial Depth": ["N", "Pog", "Or", "Po"],
        "Lower Face Height": ["ANS", "Me", "N"],
        "Mandibular Plane": ["Go", "Me", "Or", "Po"],
        "Convexity": ["A", "N", "Pog"],
        "Upper Incisor": ["U1", "A", "Pog"],
        "Lower Incisor": ["L1", "A", "Pog"],
    },
    "McNamara": {
        "N-A-Pog": ["N", "A", "Pog"],
        "Co-A": ["Co", "A"],
        "Co-Gn": ["Co", "Gn"],
        "Wits Appraisal": ["A", "B"],
        "Lower Face Height": ["ANS", "Me"],
        "Upper Face Height": ["N", "ANS"],
        "Facial Height Ratio": ["N", "ANS", "Me"],
    },
    "Wits": {
        "AO-BO": ["A", "B"],
        "PP/Go-Gn": ["ANS", "PNS", "Go", "Gn"],
        "S-Go": ["S", "Go"],
    },
    "Tweed": {
        "FMA": ["Or", "Po", "Go", "Me"],
        "FMIA": ["Or", "Po", "L1", "Me"],
        "IMPA": ["Go", "Me", "LIA", "LIT"],
    },
    "Bjork": {
        "S-Ar/Go-Gn Ratio": ["S", "Ar", "Go", "Gn"],
        "Ar-Go-N/Go-Me Ratio": ["Ar", "Go", "N", "Me"],
        "S-Go/Go-Me Ratio": ["S", "Go", "Me"],
        "NS-Gn Angle": ["N", "S", "Gn"],
    },
    "Jarabak": {
        "S-Go/Ar-Go Ratio": ["S", "Go", "Ar"],
        "Ar-Go/N-Go Ratio": ["Ar", "Go", "N"],
        "Co-Gn/Ar-Go Ratio": ["Co", "Gn", "Ar", "Go"],
        "S-Ar/Go-Gn Ratio": ["S", "Ar", "Go", "Gn"],
    },
    "Sassouni": {
        "N-S-Ar": ["N", "S", "Ar"],
        "N-Ar-Go": ["N", "Ar", "Go"],
        "Go-Co//N-S": ["Go", "Co", "N", "S"],
        "Go-Co/Go-Gn": ["Go", "Co", "Gn"],
        "N-Co//Go-Co": ["N", "Co", "Go"],
        "Ar-Co//Co-Gn": ["Ar", "Co", "Gn"],
    },
}

def find_missing_landmarks():
    """پیدا کردن لندمارک‌های مفقود"""
    all_required = set()
    
    # جمع‌آوری تمام لندمارک‌های مورد نیاز
    for analysis_name, parameters in REQUIRED_LANDMARKS.items():
        for param_name, landmarks in parameters.items():
            all_required.update(landmarks)
    
    # لندمارک‌های موجود در مدل
    aariz_set = set(AARIZ_LANDMARKS)
    
    # لندمارک‌های مفقود
    missing = all_required - aariz_set
    
    # لندمارک‌هایی که ممکن است با نام‌های مختلف وجود داشته باشند
    potential_matches = {
        "Na": "N",  # Na ممکن است همان N باشد
        "U1": ["UIA", "UIT", "UMT"],  # U1 ممکن است از UIA, UIT, UMT استخراج شود
        "L1": ["LIA", "LIT"],  # L1 ممکن است از LIA, LIT استخراج شود
        "U6": ["UPM"],  # U6 ممکن است از UPM استخراج شود
        "L6": ["LPM"],  # L6 ممکن است از LPM استخراج شود
        "U1A": ["UIA"],  # U1A ممکن است همان UIA باشد
        "L1A": ["LIA"],  # L1A ممکن است همان LIA باشد
    }
    
    # بررسی تطابق‌های احتمالی
    truly_missing = []
    can_be_approximated = {}
    
    for landmark in missing:
        if landmark in potential_matches:
            matches = potential_matches[landmark]
            if isinstance(matches, str):
                if matches in aariz_set:
                    can_be_approximated[landmark] = matches
                    continue
            else:
                # بررسی آیا هر کدام از matches در aariz_set وجود دارد
                found = [m for m in matches if m in aariz_set]
                if found:
                    can_be_approximated[landmark] = found
                    continue
        
        truly_missing.append(landmark)
    
    return {
        "all_required": sorted(all_required),
        "aariz_landmarks": sorted(AARIZ_LANDMARKS),
        "missing": sorted(truly_missing),
        "can_be_approximated": can_be_approximated,
        "missing_by_analysis": {}
    }

def analyze_by_analysis():
    """تحلیل لندمارک‌های مفقود برای هر آنالیز"""
    result = {}
    aariz_set = set(AARIZ_LANDMARKS)
    
    for analysis_name, parameters in REQUIRED_LANDMARKS.items():
        missing_for_analysis = set()
        
        for param_name, landmarks in parameters.items():
            for landmark in landmarks:
                # بررسی تطابق‌های احتمالی
                if landmark == "Na" and "N" in aariz_set:
                    continue
                if landmark == "U1" and any(x in aariz_set for x in ["UIA", "UIT", "UMT"]):
                    continue
                if landmark == "L1" and any(x in aariz_set for x in ["LIA", "LIT"]):
                    continue
                if landmark == "U6" and "UPM" in aariz_set:
                    continue
                if landmark == "L6" and "LPM" in aariz_set:
                    continue
                if landmark == "U1A" and "UIA" in aariz_set:
                    continue
                if landmark == "L1A" and "LIA" in aariz_set:
                    continue
                
                if landmark not in aariz_set:
                    missing_for_analysis.add(landmark)
        
        result[analysis_name] = sorted(missing_for_analysis)
    
    return result

if __name__ == "__main__":
    print("=" * 80)
    print("تحلیل لندمارک‌های مفقود برای آنالیزهای سفالومتری")
    print("=" * 80)
    print()
    
    # لندمارک‌های مدل Aariz
    print("📋 لندمارک‌های موجود در مدل Aariz (29 لندمارک):")
    print(f"   {', '.join(AARIZ_LANDMARKS)}")
    print()
    
    # تحلیل کلی
    analysis = find_missing_landmarks()
    
    print("=" * 80)
    print("لندمارک‌های مورد نیاز برای همه آنالیزها:")
    print(f"   {', '.join(analysis['all_required'])}")
    print(f"   تعداد: {len(analysis['all_required'])}")
    print()
    
    print("=" * 80)
    print("لندمارک‌های مفقود (نیاز به اضافه کردن به مدل):")
    if analysis['missing']:
        for landmark in analysis['missing']:
            print(f"   ❌ {landmark}")
    else:
        print("   ✅ همه لندمارک‌ها موجود هستند!")
    print()
    
    print("=" * 80)
    print("لندمارک‌هایی که می‌توانند تقریب زده شوند:")
    if analysis['can_be_approximated']:
        for landmark, approximation in analysis['can_be_approximated'].items():
            if isinstance(approximation, list):
                print(f"   ⚠️  {landmark} → می‌توان از {', '.join(approximation)} استفاده کرد")
            else:
                print(f"   ⚠️  {landmark} → می‌توان از {approximation} استفاده کرد")
    else:
        print("   هیچ لندمارکی قابل تقریب نیست")
    print()
    
    # تحلیل بر اساس هر آنالیز
    print("=" * 80)
    print("لندمارک‌های مفقود برای هر آنالیز:")
    print("=" * 80)
    
    missing_by_analysis = analyze_by_analysis()
    for analysis_name, missing_landmarks in missing_by_analysis.items():
        print(f"\n📊 {analysis_name}:")
        if missing_landmarks:
            for landmark in missing_landmarks:
                print(f"   ❌ {landmark}")
        else:
            print("   ✅ همه لندمارک‌های مورد نیاز موجود هستند")
    
    print()
    print("=" * 80)
    print("خلاصه:")
    print("=" * 80)
    
    total_missing = set()
    for missing_list in missing_by_analysis.values():
        total_missing.update(missing_list)
    
    print(f"تعداد لندمارک‌های مفقود کل: {len(total_missing)}")
    if total_missing:
        print(f"لیست: {', '.join(sorted(total_missing))}")
    
    # ذخیره نتایج
    output = {
        "aariz_landmarks": AARIZ_LANDMARKS,
        "missing_landmarks": sorted(total_missing),
        "can_be_approximated": analysis['can_be_approximated'],
        "missing_by_analysis": missing_by_analysis
    }
    
    with open("missing_landmarks_analysis.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 نتایج در فایل missing_landmarks_analysis.json ذخیره شد")

