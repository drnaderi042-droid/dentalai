@echo off
REM اسکریپت نصب PyTorch با پشتیبانی CUDA
REM Script to install PyTorch with CUDA support

echo Upgrading pip...
python -m pip install --upgrade pip

echo.
echo Installing PyTorch with CUDA support...
echo Please visit https://pytorch.org/get-started/locally/ to get the correct command for your CUDA version
echo.
echo For CUDA 11.8:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
echo For CUDA 12.1:
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

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

