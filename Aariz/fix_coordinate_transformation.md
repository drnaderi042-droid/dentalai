# Fix for P1/P2 Coordinate Transformation

## Problem Analysis

The model was trained with an inconsistency:
1. Annotations have coordinates normalized [0,1] relative to **ORIGINAL** image
2. Image is resized to 1024x1024 (STRETCHED)
3. Heatmap is generated from coordinates [0,1] relative to **ORIGINAL** image
4. But in validation: `pred_px = pred_coords * 1024` treats coords as relative to **1024x1024**

## Solution

The validation code shows that `extract_coordinates` returns [0,1] relative to the **resized image (1024x1024)**, not the original.

So the transformation should be:
```
x_original = coords_normalized * orig_width
y_original = coords_normalized * orig_height
```

This is what we're doing now. The model learned to predict in the resized space, so we scale back to original.

