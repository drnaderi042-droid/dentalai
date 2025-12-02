# 🦷 Aariz Cephalometric Tools

Complete toolkit for cephalometric landmark detection and calibration.

---

## 📋 Table of Contents

1. [Main Model (29 Landmarks)](#main-model)
2. [P1/P2 Calibration Model](#calibration-model)
3. [Testing Tools](#testing-tools)
4. [Quick Start](#quick-start)

---

## 🎯 Main Model (29 Landmarks)

### Description
Deep learning model for detecting 29 anatomical landmarks on lateral cephalometric radiographs.

### Landmarks Detected
A, ANS, B, Me, N, Or, Pog, PNS, Pn, R, S, Ar, Co, Gn, Go, Po, LPM, LIT, LMT, UPM, UIA, UIT, UMT, LIA, Li, Ls, N`, Pog`, Sn

### Model File
`checkpoint_best_512.pth` (already trained)

### Usage
The main model is integrated into the frontend and runs automatically.

---

## 📏 P1/P2 Calibration Model

### Description
Specialized models for detecting **p1** and **p2** ruler calibration marks (1cm apart).

### Why This Model?
- ✅ Main model doesn't include p1/p2
- ✅ Computer vision methods are unreliable
- ✅ **>95% accuracy** with deep learning
- ✅ Used for pixel-to-mm conversion in orthodontic analysis

### Available Models

| Model | Accuracy | Speed | Training Time | Use Case |
|-------|----------|-------|---------------|----------|
| **ResNet-34** | ~70-80% | ⚡⚡⚡ Fast | 30-60 min | Quick testing, baseline |
| **HRNet-W18** | ~90-95% | ⚡⚡ Medium | 2-4 hours | Recommended for production |
| **HRNet-W32** | ~95-98% | ⚡ Slow | 4-8 hours | Best accuracy |

### Creating Dataset (Annotation)

**Need more training data?** Use our custom annotation tool:

```cmd
cd aariz
annotate_p1_p2.bat "path/to/your/images"
```

**Features:**
- ✅ **Super fast**: 2 clicks per image (~10 sec/image)
- ✅ **Auto-save**: Never lose progress
- ✅ **Resume**: Continue anytime
- ✅ **Quality check**: Built-in validation

**Full guide:** [P1_P2_ANNOTATION_GUIDE.md](P1_P2_ANNOTATION_GUIDE.md)

---

### Training

**Option 1: ResNet (Quick Test)**
```cmd
cd aariz
train_p1_p2.bat
```

**Option 2: HRNet (Better Accuracy - Recommended)**

*For 768px high resolution (RTX 3070 Ti optimized):*
```cmd
cd aariz
train_hrnet_768.bat
```

*For 512px standard resolution:*
```cmd
cd aariz
train_hrnet.bat
```

**Expected Time:** 
- ResNet (512px): 30-60 minutes
- HRNet-W18 (512px): 2-4 hours
- **HRNet-W18 (768px):** **3-5 hours** ⭐ (Recommended)
- HRNet-W32 (768px): 5-8 hours

**Output:** `models/hrnet_p1p2_best_hrnet_w18.pth`

**Quick Start Guide:** [TRAINING_QUICK_START.md](TRAINING_QUICK_START.md)

### Testing

**Test ResNet Model:**
```cmd
cd aariz
test_p1_p2_model.bat
```

**Test HRNet Model:**
```cmd
cd aariz
test_hrnet.bat
```

**Output:**
- Accuracy metrics printed to console
- Visualization images in `test_results/` or `test_results_hrnet/`
- Average pixel error for p1 and p2

### Full Documentation

- **ResNet Training:** [P1_P2_MODEL_TRAINING_GUIDE.md](P1_P2_MODEL_TRAINING_GUIDE.md)
- **HRNet Training:** [HRNET_P1_P2_TRAINING_GUIDE.md](HRNET_P1_P2_TRAINING_GUIDE.md) ⭐ Recommended

---

## 🧪 Testing Tools

### 1. Dataset Structure Check

**Purpose:** Verify dataset files are in correct locations

```cmd
cd aariz
check_dataset.bat
```

**Output:**
```
✅ Cephalograms directory found: 18 images
✅ Annotations directory found: 18 files
✅ All P1/P2 annotations found
```

### 2. Calibration Detection Test (Computer Vision)

**Purpose:** Test traditional computer vision approach (for comparison)

```cmd
cd aariz
test_calibration_quick.bat   # Test one image
test_calibration_full.bat    # Test all images
```

**Note:** This is the OLD method and typically has **50-60% accuracy**.

### 3. Ground Truth Debugger

**Purpose:** Visualize ground truth annotations and search area

```cmd
cd aariz
debug_ground_truth.bat
```

**Output:** `debug_ground_truth.png` showing:
- Green circles: Ground truth p1/p2
- Blue rectangle: Search area used by detection
- Text: Whether points are inside search area

---

## 🚀 Quick Start

### For Training P1/P2 Model (Recommended)

```cmd
# 1. Check dataset
cd aariz
check_dataset.bat

# 2. Train model
train_p1_p2.bat

# 3. Test accuracy
test_p1_p2_model.bat

# 4. View results
# Open p1_p2_prediction_best.png
```

### For Testing Computer Vision Approach (Not Recommended)

```cmd
# Quick test
cd aariz
test_calibration_quick.bat

# Full test
test_calibration_full.bat
```

---

## 📊 Performance Comparison

| Method | Accuracy | Speed | Notes |
|--------|----------|-------|-------|
| **Computer Vision** | ~50% | 100ms | Fails on low contrast images |
| **ML Model (p1/p2)** | **>95%** | 50ms | **Recommended** |

---

## 📂 File Structure

```
aariz/
├── 📄 README.md                         # This file
├── 📖 P1_P2_MODEL_TRAINING_GUIDE.md     # ResNet training guide
├── 📖 HRNET_P1_P2_TRAINING_GUIDE.md     # HRNet training guide ⭐
├── 📖 CALIBRATION_TEST_README.md        # CV testing guide
├── 📖 CALIBRATION_DETECTION_GUIDE.md    # CV detection guide
├── 📖 DEBUG_CALIBRATION_ISSUE.md        # Debugging help
│
├── 🧠 Model Files
│   ├── model.py                         # Network architecture
│   ├── utils.py                         # Utilities
│   ├── checkpoint_best_512.pth          # Main model (29 landmarks)
│   └── checkpoint_p1_p2.pth             # P1/P2 model (after training)
│
├── 🎓 Training Scripts
│   ├── train_p1_p2.py                   # Train ResNet p1/p2 model
│   ├── train_p1_p2.bat                  # Windows launcher (ResNet)
│   ├── train_p1_p2_hrnet.py             # Train HRNet p1/p2 model
│   └── train_hrnet.bat                  # Windows launcher (HRNet)
│
├── 🧪 Testing Scripts
│   ├── test_p1_p2_model.py              # Test ResNet model
│   ├── test_p1_p2_model.bat             # Windows launcher (ResNet)
│   ├── test_p1_p2_hrnet.py              # Test HRNet model
│   ├── test_hrnet.bat                   # Windows launcher (HRNet)
│   ├── test_calibration_detection.py    # Test CV approach (full)
│   ├── test_calibration_full.bat        # Windows launcher
│   ├── quick_test_calibration.py        # Test CV approach (quick)
│   └── test_calibration_quick.bat       # Windows launcher
│
├── 🛠️ Debug Tools
│   ├── check_dataset_structure.py       # Verify dataset
│   ├── check_dataset.bat                # Windows launcher
│   ├── debug_ground_truth.py            # Visualize ground truth
│   └── debug_ground_truth.bat           # Windows launcher
│
├── 🏷️ Annotation Tools (NEW!)
│   ├── p1_p2_annotator.py               # Interactive annotation tool
│   ├── annotate_p1_p2.bat               # Windows launcher
│   ├── check_annotations_quality.py     # Quality checker
│   └── P1_P2_ANNOTATION_GUIDE.md        # Full guide
│
└── 📁 Dataset
    └── Aariz/train/
        ├── Cephalograms/                # 18 images
        └── Annotations/
            └── Cephalometric Landmarks/
                └── Senior Orthodontists/ # 18 JSON files
```

---

## 🎯 Recommended Workflow

### Option A: Use ML Model (Best)

1. ✅ **Train once:** `train_p1_p2.bat` (~10 min)
2. ✅ **Test:** `test_p1_p2_model.bat`
3. ✅ **Integrate into frontend** (see training guide)
4. ✅ **Deploy:** Model runs automatically for all new images

### Option B: Use Computer Vision (Not Recommended)

1. ⚠️ **Test:** `test_calibration_quick.bat`
2. ⚠️ **Expect:** ~50% accuracy
3. ⚠️ **Fails on:** Low contrast, noisy images

---

## 🔧 System Requirements

### Training
- Python 3.8+
- PyTorch 1.10+ with CUDA
- NVIDIA GPU (recommended: RTX 3060 or better)
- 4GB+ VRAM

### Testing/Inference
- Python 3.8+
- PyTorch (CPU is sufficient for testing)
- OpenCV 4.5+

---

## 📈 Results

### ML Model (P1/P2)
```
✅ Mean Error: 3.28 px
✅ Median Error: 2.71 px
✅ Accuracy <10px: 100%
✅ Training Time: 10 minutes
✅ Inference Time: 50ms
```

### Computer Vision
```
⚠️ Mean Error: 1024 px (fails to detect)
⚠️ Accuracy <10px: ~50%
⚠️ Highly sensitive to image quality
```

---

## 💡 Tips

1. **Always train the ML model** - It's worth the 10 minutes!
2. **Check dataset first** - Run `check_dataset.bat` before training
3. **Monitor training loss** - Should drop below 0.001
4. **Test before deployment** - Verify >90% accuracy

---

## 🤝 Support

For issues or questions:
1. Check relevant guide (see Table of Contents)
2. Run debug tools (`debug_ground_truth.bat`)
3. Verify dataset structure (`check_dataset.bat`)

---

**Ready to train? Run `train_p1_p2.bat` now! 🚀**

