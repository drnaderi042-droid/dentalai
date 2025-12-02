# Final Results Analysis - Multiple Models Comparison

## Test Results Summary (20 images)

| Method | Mean MRE (mm) | Median MRE (mm) | Mean SDR @ 2mm (%) | Median SDR @ 2mm (%) | Status |
|--------|---------------|-----------------|-------------------|---------------------|--------|
| **256x256** | 2.0853 | 1.9089 | 61.55 | 62.07 | Baseline |
| **512x512** | 1.6039 | 1.5510 | 73.45 | 75.86 | Better |
| **512x512 + TTA** | 1.5421 | 1.4650 | **74.83** | **79.31** | **BEST** |
| **Ensemble** | 1.7694 | 1.6189 | 71.90 | 72.41 | Worse than 512x512 |
| **Ensemble + TTA** | 1.7278 | 1.5597 | 72.41 | 75.86 | Similar to Ensemble |

## Key Findings

### 1. **512x512 is clearly better than 256x256**
- MRE improvement: 0.48mm (2.09 → 1.60mm)
- SDR improvement: +11.9% (61.55% → 73.45%)

### 2. **TTA provides consistent improvement**
- On 512x512: +1.38% SDR (73.45% → 74.83%)
- MRE reduction: 0.06mm (1.60 → 1.54mm)
- **Median SDR improvement: +3.45%** (75.86% → 79.31%)

### 3. **Ensemble did not help as expected**
- Ensemble alone: Worse than 512x512 (-1.55% SDR)
- Ensemble + TTA: Still worse than 512x512 + TTA alone
- **Conclusion**: 512x512 model is strong enough; ensemble adds complexity without benefit

### 4. **Best Configuration: 512x512 + TTA**
- **Mean SDR @ 2mm: 74.83%**
- **Median SDR @ 2mm: 79.31%**
- **Mean MRE: 1.54mm**
- **Median MRE: 1.47mm**

## Per-Image Analysis

### TTA Improvement Variability
- Some images show +10.34% improvement (e.g., `cks2ip8fx2aj00yuf7h0kh519`)
- Some show 0% improvement
- **Average improvement: +1.38%**
- **Median improvement: +3.45%** (better indicator of typical case)

### Why Ensemble Failed?
1. **512x512 is already very good**: Outperforms 256x256 significantly
2. **Landmark preference may not be optimal**: Static preferences based on single image may not generalize
3. **Ensemble overhead**: Combining predictions may introduce errors for some landmarks

## Recommendations

### For Production:
1. **Primary Model: 512x512 + TTA**
   - Best overall performance
   - Consistent improvement across images
   - Acceptable processing time (2x inference)

2. **Alternative: 512x512 (no TTA)**
   - If processing speed is critical
   - Still good performance (73.45% SDR)
   - 2x faster than TTA version

3. **Not Recommended:**
   - Ensemble (adds complexity, no benefit)
   - 256x256 (significantly worse)

### For Frontend:
- Provide all variants for user choice
- Default to **512x512 + TTA** (best accuracy)
- Show performance metrics in UI
- Allow users to select based on speed vs accuracy trade-off

## Performance Metrics

### Speed (estimated):
- 256x256: ~0.5s per image
- 512x512: ~1.0s per image
- 512x512 + TTA: ~2.0s per image (2x inference)
- Ensemble: ~1.5s per image (2 models)
- Ensemble + TTA: ~3.0s per image (4 inferences)

### Accuracy Ranking:
1. **512x512 + TTA**: 74.83% SDR
2. **512x512**: 73.45% SDR
3. **Ensemble + TTA**: 72.41% SDR
4. **Ensemble**: 71.90% SDR
5. **256x256**: 61.55% SDR

