@echo off
REM Activate virtual environment and run TeethDreamer generation
cd /d "%~dp0TeethDreamer"
call ..\venv_teethdreamer\Scripts\activate.bat
python TeethDreamer.py -b configs/TeethDreamer.yaml --gpus 0 --test ckpt/TeethDreamer.ckpt --output ../output/generation data.params.test_dir=../output/segmented
pause













