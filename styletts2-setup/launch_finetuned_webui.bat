@echo off
REM Launch Gradio Web UI for Fine-Tuned Model
REM EDIT finetuned_webui.py to set BEST_EPOCH first!

echo ================================================
echo Fine-Tuned StyleTTS2 Web UI
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
    echo WARNING: No venv found
    echo Create with: python -m venv .venv
    pause
    exit /b 1
)

REM Check for espeak-ng
where espeak-ng >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: espeak-ng not found in PATH
    echo Install from: https://github.com/espeak-ng/espeak-ng/releases
)

echo.
echo REMINDER: Edit BEST_EPOCH in finetuned_webui.py before first use!
echo.
echo Starting web UI on http://127.0.0.1:7861
echo.

python finetuned_webui.py

pause
