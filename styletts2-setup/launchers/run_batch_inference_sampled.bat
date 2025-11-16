@echo off
REM Quick test - samples every 5th epoch for faster evaluation
REM Tests epochs: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49

echo ================================================
echo StyleTTS2 Batch Inference Launcher (SAMPLED)
echo Testing every 5th epoch for quick comparison
echo ================================================
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
    pause
    exit /b 1
)

REM Configure espeak-ng for phonemizer
where espeak-ng >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: espeak-ng not found in PATH
)

echo.
echo Running sampled batch inference (10 checkpoints)...
echo Testing epochs: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50
echo Output will be saved to: inference_outputs\
echo.

python ..\scripts\batch_inference_epochs.py --start-epoch 5 --end-epoch 50 --step 5 --diffusion-steps 5 --embedding-scale 1.0

echo.
echo ===================================
echo Sampled inference complete!
echo ===================================
echo Check the inference_outputs folder for results
pause
