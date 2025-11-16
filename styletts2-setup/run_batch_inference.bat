@echo off
REM Batch inference launcher for all StyleTTS2 checkpoints
REM Tests all 50 epochs and generates 350 audio samples

echo ===================================
echo StyleTTS2 Batch Inference Launcher
echo ===================================
echo.

REM Change to script directory
cd /d "%~dp0"

echo Activating Python environment...
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo ✓ Activated .venv
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo ✓ Activated venv
) else (
    echo WARNING: No venv found, using system Python
    echo Create venv with: python -m venv .venv
    pause
    exit /b 1
)

REM Configure espeak-ng for phonemizer (update path if needed)
where espeak-ng >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: espeak-ng not found in PATH
    echo Install from: https://github.com/espeak-ng/espeak-ng/releases
)

echo.
echo Running batch inference on all 50 epochs...
echo This will take 1-2 hours - generating 350 audio samples total
echo Output will be saved to: inference_outputs\
echo.

python batch_inference_epochs.py --start-epoch 0 --end-epoch 49 --diffusion-steps 5 --embedding-scale 1.0

echo.
echo ===================================
echo Batch inference complete!
echo ===================================
echo Check the inference_outputs folder for results
pause
