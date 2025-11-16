@echo off
REM StyleTTS2 Fine-tuning Launcher with Safety Measures
REM Includes CUDA debugging flags and working directory fix

echo ====================================
echo StyleTTS2 Training Script (Safe Mode)
echo ====================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Set CUDA environment variables for debugging and memory management
set CUDA_LAUNCH_BLOCKING=1
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

echo CUDA safety flags enabled:
echo   - CUDA_LAUNCH_BLOCKING=1 (synchronous errors)
echo   - Memory allocation: max_split_size_mb=128
echo.

REM Activate virtual environment
echo Activating virtual environment...
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

REM Check monotonic_align installation
echo.
echo Checking monotonic_align installation...
pip show monotonic_align >nul 2>&1
if errorlevel 1 (
    echo WARNING: monotonic_align not found
    echo Install with: pip install git+https://github.com/resemble-ai/monotonic_align.git
    echo Or run: python install_monotonic_align.py
    set /p CONTINUE="Continue anyway? (y/n): "
    if /i not "%CONTINUE%"=="y" exit /b 1
) else (
    echo ✓ monotonic_align installed
)

REM CRITICAL: Change to StyleTTS2 directory before running training
REM This ensures relative paths in config_ft.yml resolve correctly
echo.
echo Changing to StyleTTS2 directory...
if not exist "StyleTTS2" (
    echo ERROR: StyleTTS2 directory not found
    echo Expected location: %CD%\StyleTTS2
    echo.
    echo Please ensure StyleTTS2 is cloned and patches are applied:
    echo   1. git clone https://github.com/yl4579/StyleTTS2.git
    echo   2. .\apply_patches.ps1
    pause
    exit /b 1
)

pushd StyleTTS2
echo ✓ Working directory: %CD%
echo.

REM Verify config exists
if not exist "Configs\config_ft.yml" (
    echo ERROR: config_ft.yml not found
    echo Expected location: %CD%\Configs\config_ft.yml
    popd
    pause
    exit /b 1
)

REM Start training
echo Starting training...
echo Config: Configs\config_ft.yml
echo.
echo NOTE: Initial imports may take 1-2 minutes (transformers, scipy, sklearn)
echo Please wait without interruption...
echo.

python train_finetune.py -p Configs\config_ft.yml

REM Return to original directory
popd

echo.
echo ====================================
echo Training completed or stopped
echo ====================================
echo.
echo Check logs above for any errors.
echo Checkpoints saved to: StyleTTS2\Models\LJSpeech\
echo.
pause
