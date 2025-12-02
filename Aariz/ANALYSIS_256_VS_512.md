# Analysis: 256x256 vs 512x512 Model Comparison

## Summary

| Metric | 256x256 | 512x512 | Change | Status |
|--------|---------|---------|--------|--------|
| **MRE (mm)** | 1.8642 | 1.7560 | -0.1082 | Better |
| **Median Error (mm)** | 1.4779 | 1.2820 | -0.1959 | Better |
| **SDR @ 1.0mm** | 24.14% | 41.38% | +17.24% | Much Better |
| **SDR @ 1.5mm** | 51.72% | 55.17% | +3.45% | Better |
| **SDR @ 2.0mm** | 72.41% | 65.52% | -6.90% | Worse |
| **SDR @ 2.5mm** | 75.86% | 79.31% | +3.45% | Better |
| **Std Dev (mm)** | 1.4848 | 1.6363 | +0.1515 | Worse (more variance) |

## Key Observations

### Strengths of 512x512 Model:
1. **Better precision at tight thresholds**: SDR @ 1.0mm improved by 17.24% (from 24% to 41%)
2. **Better median error**: Reduced from 1.48mm to 1.28mm (13% improvement)
3. **Better MRE**: Slightly improved from 1.86mm to 1.76mm
4. **Better at 2.5mm threshold**: Improved SDR from 75.86% to 79.31%

### Weaknesses of 512x512 Model:
1. **Worse SDR @ 2.0mm**: Dropped from 72.41% to 65.52% (-6.9%)
2. **Higher variance**: Std Dev increased from 1.48 to 1.64 (11% increase)
3. **Problematic landmarks**: Several landmarks perform worse (LMT, UMT, UPM, S, LPM, Co)

## Problematic Landmarks in 512x512

| Landmark | 256x256 Error | 512x512 Error | Degradation | Issue |
|----------|---------------|---------------|-------------|-------|
| **LMT** | 2.85 mm | 5.34 mm | -2.49 mm | Critical degradation |
| **UMT** | 1.02 mm | 2.73 mm | -1.72 mm | Significant degradation |
| **UPM** | 0.55 mm | 1.73 mm | -1.18 mm | Significant degradation |
| **S** | 2.64 mm | 3.56 mm | -0.92 mm | Moderate degradation |
| **LPM** | 1.76 mm | 2.51 mm | -0.75 mm | Moderate degradation |
| **Co** | 0.59 mm | 1.55 mm | -0.96 mm | Moderate degradation |
| **R** | 1.33 mm | 2.05 mm | -0.71 mm | Moderate degradation |

**Pattern**: Most problematic landmarks are in the **mandibular/chin region** (LMT, UMT, UPM, LPM, Co) and **cranial base** (S).

## Improved Landmarks in 512x512

| Landmark | 256x256 Error | 512x512 Error | Improvement | Status |
|----------|---------------|---------------|-------------|--------|
| **Or** | 2.28 mm | 0.08 mm | +2.19 mm | Excellent |
| **Sn** | 1.72 mm | 0.26 mm | +1.46 mm | Excellent |
| **N** | 1.93 mm | 0.44 mm | +1.49 mm | Excellent |
| **Ar** | 3.78 mm | 2.36 mm | +1.42 mm | Good |
| **LIA** | 3.24 mm | 2.09 mm | +1.15 mm | Good |
| **A** | 1.12 mm | 0.53 mm | +0.59 mm | Good |

**Pattern**: Improvements are mainly in **facial soft tissue landmarks** (Or, Sn, N, A) and **some mandibular landmarks** (Ar, LIA).

## Root Cause Analysis

### Why SDR @ 2.0mm is worse despite better MRE?

1. **Variance issue**: Std Dev increased from 1.48 to 1.64, meaning more outliers
2. **Specific landmark failures**: 6 landmarks perform worse, and some exceed 2mm threshold
3. **Trade-off**: Better precision at 1mm threshold but worse consistency at 2mm threshold

### Why mandibular landmarks are problematic?

1. **Resolution sensitivity**: Mandibular landmarks may require different feature scales
2. **Training data**: Possibly less diverse examples in training for these landmarks
3. **Architecture**: HRNet at 512x512 may be overfitting to some features while missing others

## Improvement Strategies

### 1. Test-Time Augmentation (TTA) - HIGHEST PRIORITY
**Why**: Can reduce variance and improve consistency
**Expected**: SDR @ 2mm improvement from 65.52% to 68-72%
**Implementation**: Use `test_512x512_with_tta.py`

### 2. Fine-tuning on Problematic Landmarks
**Target landmarks**: LMT, UMT, UPM, S, LPM, Co, R
**Method**: 
- Weighted loss focusing on these landmarks
- Or fine-tuning with higher weight for samples where these landmarks are present
**Expected**: SDR @ 2mm improvement to 72-75%

### 3. Ensemble 256x256 + 512x512
**Why**: Each model has different strengths
- 256x256: Better at mandibular landmarks (LMT, UMT, UPM, LPM, Co)
- 512x512: Better at facial landmarks (Or, Sn, N, A, Ar)
**Method**: Weighted average based on landmark-specific performance
**Expected**: Best of both worlds - SDR @ 2mm to 75-80%

### 4. Post-processing Refinement
**For problematic landmarks**: 
- Use anatomical constraints
- Refine using gradient-based optimization
- Use prior knowledge about landmark relationships
**Expected**: SDR @ 2mm improvement to 70-73%

### 5. Data Augmentation Focus
**During training**:
- More augmentation for mandibular region
- Focus on samples with difficult landmarks
- Synthetic data generation for problematic cases
**Expected**: General improvement, especially for LMT, UMT, UPM

### 6. Multi-Scale Training
**Method**: Train on multiple resolutions simultaneously (256, 384, 512)
**Expected**: Better generalization and robustness

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. **Test TTA** - Run `test_512x512_with_tta.py`
   - Expected: SDR @ 2mm from 65.52% to 68-72%
   
2. **Create Ensemble** - Combine 256x256 and 512x512 predictions
   - Use 512x512 for: Or, Sn, N, A, Ar, LIA, Me, Gn, Li, Ls, Pog
   - Use 256x256 for: LMT, UMT, UPM, S, LPM, Co, R
   - Expected: SDR @ 2mm to 75-78%

### Phase 2: Training Improvements (4-6 hours)
3. **Weighted Loss Fine-tuning** - Focus on problematic landmarks
   - Increase weight for LMT, UMT, UPM, S, LPM, Co by 2-3x
   - Fine-tune for 10-20 epochs
   - Expected: SDR @ 2mm to 72-75%

### Phase 3: Advanced (Optional)
4. **Multi-scale Training** - If resources allow
5. **Synthetic Data Generation** - For problematic cases

## Conclusion

The 512x512 model shows **mixed results**:
- **Better**: Precision (SDR @ 1mm), median error, MRE
- **Worse**: Consistency (SDR @ 2mm), variance, mandibular landmarks

**Best immediate solution**: **Ensemble of 256x256 and 512x512** models with landmark-specific weighting.

**Next steps**:
1. Implement and test TTA
2. Create ensemble script
3. Evaluate results
4. If needed, proceed with weighted loss fine-tuning

