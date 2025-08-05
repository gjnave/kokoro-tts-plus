@echo off
:: Check if the script is run as Administrator
net session >nul 2>&1
if %errorlevel% neq 0 (
    echo This script requires administrator privileges.
    echo Please run this script as an administrator.
    pause
    exit /b
)

cd /d %~dp0
IF EXIST "disclaimer.md" (
    TYPE "disclaimer.md"
    pause
)

IF EXIST "type about.nfo" TYPE type about.nfo

echo.
:: Check if conda is installed
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda is not installed or not found in PATH.
    echo Please install Anaconda/Miniconda and ensure it's added to PATH.
    pause
    exit /b
)

:: Check if conda environment 'kokoro' exists
call conda env list | findstr /C:"kokoro " >nul
if %errorlevel% equ 0 (
    echo Conda environment 'kokoro' already exists.
    set /p replace_env="Do you want to replace it? (y/n): "
    if /i "!replace_env!"=="y" (
        echo Removing existing 'kokoro' environment...
        call conda deactivate
        call conda env remove --name kokoro
        echo Creating new 'kokoro' environment...
        call conda create --name kokoro python=3.12 -y
    ) else (
        echo Using existing 'kokoro' environment...
    )
) else (
    echo Creating new 'kokoro' environment...
    call conda create --name kokoro python=3.12 -y
)
call conda activate kokoro
git clone https://github.com/gjnave/kokoro-tts-plus
cd kokoro-tts-plus
git config --system --add safe.directory "kokoro-tts-plus"
cd kokoro-tts-plus
curl -LO https://raw.githubusercontent.com/gjnave/cogni-scripts/refs/heads/main/koko/app.py
curl -LO https://raw.githubusercontent.com/gjnave/cogni-scripts/refs/heads/main/koko/en.txt
curl -LO https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/voices-v1.0.bin
curl -LO https://github.com/nazdridoy/kokoro-tts/releases/download/v1.0.0/kokoro-v1.0.onnx
REM call conda install nvidia/label/cuda-12.6.3::cuda-toolkit -y
pip install -r requirements.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.2.0-windows.post10/triton-3.2.0-cp312-cp312-win_amd64.whl
pip install sageattention
pip install "https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-win_amd64.whl"

pip install kokoro
pip install ebooklib
pip install PyMuPDF
pip install pymupdf4llm
pip install beautifulsoup4
pip install gradio
echo Installation Complete...
pause
