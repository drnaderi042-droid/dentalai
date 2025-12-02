# Frontend Integration Summary

## Backend API Endpoints Added

All endpoints available on `http://localhost:5001`:

1. **POST /detect** - 512x512 model (default, same as before)
2. **POST /detect-256** - 256x256 model
3. **POST /detect-512** - 512x512 model (same as /detect)
4. **POST /detect-512-tta** - 512x512 with Test-Time Augmentation
5. **POST /detect-ensemble** - Ensemble of 256x256 + 512x512
6. **POST /detect-ensemble-tta** - Ensemble with TTA

## Frontend Models Added

1. **Aariz 256x256 (Local)**
   - ID: `local/aariz-256`
   - Description: Fast (SDR: 61.55%)
   - Endpoint: `/detect-256`

2. **Aariz 512x512 (Local)**
   - ID: `local/aariz-512`
   - Description: Balanced (SDR: 73.45%)
   - Endpoint: `/detect-512`

3. **Aariz 512x512 + TTA (Local)** ⭐ BEST
   - ID: `local/aariz-512-tta`
   - Description: Best accuracy (SDR: 74.83%)
   - Endpoint: `/detect-512-tta`

4. **Aariz Ensemble (Local)**
   - ID: `local/aariz-ensemble`
   - Description: Ensemble 256+512 (SDR: 71.90%)
   - Endpoint: `/detect-ensemble`

5. **Aariz Ensemble + TTA (Local)**
   - ID: `local/aariz-ensemble-tta`
   - Description: Ensemble 256+512 with TTA (SDR: 72.41%)
   - Endpoint: `/detect-ensemble-tta`

## Usage

All models are now available in the dropdown on `/dashboard/ai-model-test` page.

**Recommended**: Use **Aariz 512x512 + TTA** for best accuracy (74.83% SDR @ 2mm).

## Backend Requirements

- 512x512 checkpoint: `Aariz/checkpoints/checkpoint_best.pth` (required)
- 256x256 checkpoint: `Aariz/checkpoint_best_256.pth` (required for ensemble models)

If 256x256 checkpoint is missing, ensemble endpoints will return an error.

