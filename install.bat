@echo off
echo ========================================
echo 安装 EnCodec + QR Code 项目依赖
echo ========================================

echo [1/5] 安装 PyTorch（CPU版）...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

echo [2/5] 安装 EnCodec...
pip install encodec -i https://mirrors.aliyun.com/pypi/simple/

echo [3/5] 安装音频处理库...
pip install numpy librosa soundfile scipy -i https://mirrors.aliyun.com/pypi/simple/

echo [4/5] 安装二维码库...
pip install qrcode pillow opencv-python pyzbar -i https://mirrors.aliyun.com/pypi/simple/

echo [5/5] 安装 PDF 排版库...
pip install reportlab -i https://mirrors.aliyun.com/pypi/simple/

echo ========================================
echo ✅ 所有依赖安装完成！
echo ========================================
pause
