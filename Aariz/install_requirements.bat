@echo off
REM اسکریپت نصب پکیج‌ها برای Windows
REM Script to install requirements on Windows

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch (CPU version - if you have CUDA, install GPU version separately)...
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo.
echo Installing other requirements...
python -m pip install numpy "numpy<2.0.0"
python -m pip install Pillow
python -m pip install pandas
python -m pip install opencv-python
python -m pip install scikit-image
python -m pip install scipy
python -m pip install tqdm
python -m pip install tensorboard
python -m pip install matplotlib

echo.
echo Installation completed!
pause

