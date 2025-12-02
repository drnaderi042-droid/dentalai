@echo off
REM Run Unified AI API Server on port 5001 for Aariz model compatibility
echo Starting Unified AI API Server on port 5001...
echo.
echo All Aariz endpoints are now included by default:
echo   - /detect (512x512)
echo   - /detect-512 (512x512)
echo   - /detect-512-tta (512x512 + TTA)
echo   - /detect-768 (768x768)
echo   - /detect-768-tta (768x768 + TTA)
echo.
python unified_ai_api_server.py --port 5001
pause

