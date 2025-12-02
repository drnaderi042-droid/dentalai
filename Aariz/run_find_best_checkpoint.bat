@echo off
echo ========================================
echo Finding Best Checkpoint
echo ========================================
echo.

cd /d "%~dp0"

REM Check if venv exists in parent directory
if exist "..\cephx_service\venv\Scripts\python.exe" (
    echo Using Python from cephx_service venv...
    ..\cephx_service\venv\Scripts\python.exe find_best_checkpoint.py
) else if exist "venv\Scripts\python.exe" (
    echo Using Python from local venv...
    venv\Scripts\python.exe find_best_checkpoint.py
) else (
    echo Using system Python...
    python find_best_checkpoint.py
)

pause

