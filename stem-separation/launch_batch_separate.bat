@echo off
REM Batch Stem Separation Launcher

echo ========================================
echo Batch Stem Separation - Vocal Extraction
echo ========================================
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo WARNING: Virtual environment not found. Using system Python.
    echo.
) else (
    REM Activate local virtual environment
    call venv\Scripts\activate.bat
)

REM Prompt for input folder
set /p INPUT_FOLDER="Enter input folder path (containing audio files): "

if not exist "%INPUT_FOLDER%" (
    echo.
    echo ERROR: Input folder does not exist!
    pause
    exit /b 1
)

echo.
echo Input folder: %INPUT_FOLDER%
echo Output folder: %INPUT_FOLDER%\vocals_only
echo Model: htdemucs_ft (high quality)
echo.
echo This will process all audio files (MP3/WAV) in the folder.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

python batch_separate.py --input "%INPUT_FOLDER%" --model htdemucs_ft

if errorlevel 1 (
    echo.
    echo ERROR: Processing failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Processing complete! Check the output folder for vocals.
echo ======================================================================
pause
