@echo off
REM Interactive voice generation using your best checkpoint
REM EDIT inference_single_checkpoint.py to set BEST_EPOCH first!

echo ================================================
echo StyleTTS2 Interactive Voice Generation
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
    pause
    exit /b 1
)

REM Configure espeak-ng for phonemizer
where espeak-ng >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: espeak-ng not found in PATH
)

echo.
echo Starting interactive generation...
echo Edit BEST_EPOCH in inference_single_checkpoint.py to use your preferred checkpoint
echo.

python inference_single_checkpoint.py

pause
