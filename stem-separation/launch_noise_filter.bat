@echo off
echo ======================================================================
echo Batch Vocal Noise Filtering
echo ======================================================================
echo.
echo This will apply noise reduction to remove:
echo   - Birds chirping
echo   - Menu sounds / UI beeps
echo   - Background ambient noise
echo   - Low-frequency rumble
echo.

cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found. Using system Python.
    echo.
)

REM Prompt for input folder
set /p INPUT_FOLDER="Enter input folder path (containing WAV files): "

if not exist "%INPUT_FOLDER%" (
    echo.
    echo ERROR: Input folder does not exist!
    pause
    exit /b 1
)

echo.
echo Input folder: %INPUT_FOLDER%
echo Output folder: %INPUT_FOLDER%_filtered
echo.
echo Settings:
echo   - Noise Reduction Strength: 0.5 (default)
echo   - Spectral Gate Threshold: -40 dB
echo   - Highpass Filter: 80 Hz
echo.
echo Press Ctrl+C to cancel, or any key to continue...
pause >nul

REM Run the noise filtering script
python batch_noise_filter.py --input "%INPUT_FOLDER%"

if errorlevel 1 (
    echo.
    echo ERROR: Processing failed. Check the output above.
    pause
    exit /b 1
)

echo.
echo ======================================================================
echo Processing complete! Check the output folder for filtered vocals.
echo ======================================================================
pause
