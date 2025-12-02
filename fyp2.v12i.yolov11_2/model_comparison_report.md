# 📊 گزارش مقایسه مدل‌های FYP2 و Lateral ORTHO
تاریخ: 2025-11-07 22:28:33
---

## 📝 توضیحات

- **FYP2 Models**: مدل‌های 18 کلاس (canine/molar Class I/II/III با subdivisions)
- **Lateral Models**: مدل‌های 11 کلاس (Class I/II/III + مشکلات دندانی)
- هر مدل فقط با dataset مربوط به خودش مقایسه شده است

---

## 📈 جدول مقایسه مدل‌های FYP2 (18 کلاس)

| Model | TTA | Precision | Recall | F1 Score | Avg Time (s) | Avg Confidence |
|-------|-----|-----------|--------|----------|--------------|----------------|
| train (no TTA) | ❌ | 0.2831 | 0.8275 | 0.4218 | 0.021 | 0.3227 |
| train (TTA) | ✅ | 0.2750 | 0.8690 | 0.4178 | 0.161 | 0.3103 |
| phase1_run_weights (no TTA) | ❌ | 0.2809 | 0.7476 | 0.4084 | 0.025 | 0.3164 |
| phase1_run_weights (TTA) | ✅ | 0.3024 | 0.8339 | 0.4439 | 0.208 | 0.3013 |
| phase2_run_weights (no TTA) | ❌ | 0.3850 | 0.7380 | 0.5060 | 0.021 | 0.4245 |
| phase2_run_weights (TTA) | ✅ | 0.3386 | 0.8275 | 0.4805 | 0.193 | 0.3608 |

---

## 📈 جدول مقایسه مدل‌های Lateral ORTHO (11 کلاس)

| Model | TTA | Precision | Recall | F1 Score | Avg Time (s) | Avg Confidence |
|-------|-----|-----------|--------|----------|--------------|----------------|
| lateral_fyp2_train2 (no TTA) | ❌ | 0.0264 | 0.7742 | 0.0510 | 0.113 | 0.0343 |
| lateral_fyp2_train2 (TTA) | ✅ | 0.1250 | 0.3871 | 0.1890 | 1.091 | 0.0211 |
| lateral_fyp2_run1 (no TTA) | ❌ | 0.1660 | 0.6774 | 0.2667 | 0.057 | 0.2223 |
| lateral_fyp2_run1 (TTA) | ✅ | 0.2632 | 0.4839 | 0.3409 | 0.314 | 0.2064 |

---

## 🔍 جزئیات هر مدل

### train (FYP2, no TTA) (without TTA)

- **Precision**: 0.2831
- **Recall**: 0.8275
- **F1 Score**: 0.4218
- **True Positives**: 259
- **False Positives**: 656
- **False Negatives**: 54
- **Average Processing Time**: 0.021s
- **Average Confidence**: 0.3227
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 915

---

### train (FYP2, TTA) (with TTA)

- **Precision**: 0.2750
- **Recall**: 0.8690
- **F1 Score**: 0.4178
- **True Positives**: 272
- **False Positives**: 717
- **False Negatives**: 41
- **Average Processing Time**: 0.161s
- **Average Confidence**: 0.3103
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 989

---

### phase1_run_weights (FYP2, no TTA) (without TTA)

- **Precision**: 0.2809
- **Recall**: 0.7476
- **F1 Score**: 0.4084
- **True Positives**: 234
- **False Positives**: 599
- **False Negatives**: 79
- **Average Processing Time**: 0.025s
- **Average Confidence**: 0.3164
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 833

---

### phase1_run_weights (FYP2, TTA) (with TTA)

- **Precision**: 0.3024
- **Recall**: 0.8339
- **F1 Score**: 0.4439
- **True Positives**: 261
- **False Positives**: 602
- **False Negatives**: 52
- **Average Processing Time**: 0.208s
- **Average Confidence**: 0.3013
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 863

---

### phase2_run_weights (FYP2, no TTA) (without TTA)

- **Precision**: 0.3850
- **Recall**: 0.7380
- **F1 Score**: 0.5060
- **True Positives**: 231
- **False Positives**: 369
- **False Negatives**: 82
- **Average Processing Time**: 0.021s
- **Average Confidence**: 0.4245
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 600

---

### phase2_run_weights (FYP2, TTA) (with TTA)

- **Precision**: 0.3386
- **Recall**: 0.8275
- **F1 Score**: 0.4805
- **True Positives**: 259
- **False Positives**: 506
- **False Negatives**: 54
- **Average Processing Time**: 0.193s
- **Average Confidence**: 0.3608
- **Total Images**: 165
- **Total GT Boxes**: 313
- **Total Pred Boxes**: 765

---

### lateral_fyp2_train2 (Lateral, no TTA) (without TTA)

- **Precision**: 0.0264
- **Recall**: 0.7742
- **F1 Score**: 0.0510
- **True Positives**: 48
- **False Positives**: 1773
- **False Negatives**: 14
- **Average Processing Time**: 0.113s
- **Average Confidence**: 0.0343
- **Total Images**: 18
- **Total GT Boxes**: 62
- **Total Pred Boxes**: 1821

---

### lateral_fyp2_train2 (Lateral, TTA) (with TTA)

- **Precision**: 0.1250
- **Recall**: 0.3871
- **F1 Score**: 0.1890
- **True Positives**: 24
- **False Positives**: 168
- **False Negatives**: 38
- **Average Processing Time**: 1.091s
- **Average Confidence**: 0.0211
- **Total Images**: 18
- **Total GT Boxes**: 62
- **Total Pred Boxes**: 192

---

### lateral_fyp2_run1 (Lateral, no TTA) (without TTA)

- **Precision**: 0.1660
- **Recall**: 0.6774
- **F1 Score**: 0.2667
- **True Positives**: 42
- **False Positives**: 211
- **False Negatives**: 20
- **Average Processing Time**: 0.057s
- **Average Confidence**: 0.2223
- **Total Images**: 18
- **Total GT Boxes**: 62
- **Total Pred Boxes**: 253

---

### lateral_fyp2_run1 (Lateral, TTA) (with TTA)

- **Precision**: 0.2632
- **Recall**: 0.4839
- **F1 Score**: 0.3409
- **True Positives**: 30
- **False Positives**: 84
- **False Negatives**: 32
- **Average Processing Time**: 0.314s
- **Average Confidence**: 0.2064
- **Total Images**: 18
- **Total GT Boxes**: 62
- **Total Pred Boxes**: 114

---

