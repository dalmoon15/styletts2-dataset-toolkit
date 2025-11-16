@echo off
REM Stem Separation Launcher
REM 
REM Configuration:
REM - Set STEM_SEPARATION_PATH environment variable to override default path
REM - Set FFMPEG_PATH environment variable to override FFmpeg location
REM - Set CACHE_DIR environment variable to override cache location

echo ========================================
echo Stem Separation Web Interface
echo ========================================
echo.

cd /d "%~dp0"

REM Check if venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ❌ Virtual environment not found!
    echo.
    echo Please run install.ps1 or install_stem_separation.bat first.
    pause
    exit /b 1
)

REM Activate local virtual environment
call venv\Scripts\activate.bat

REM Add FFmpeg to PATH (check multiple locations)
where ffmpeg >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo FFmpeg found in system PATH
) else (
    REM Try common locations
    if not "%FFMPEG_PATH%"=="" (
        if exist "%FFMPEG_PATH%\ffmpeg.exe" (
            set PATH=%FFMPEG_PATH%;%PATH%
            echo FFmpeg found at: %FFMPEG_PATH%
        )
    ) else (
        if exist "C:\ffmpeg\bin\ffmpeg.exe" (
            set PATH=C:\ffmpeg\bin;%PATH%
            echo FFmpeg found at: C:\ffmpeg\bin
        ) else (
            echo ⚠️  FFmpeg not found. Some features may not work.
            echo    Install FFmpeg or set FFMPEG_PATH environment variable.
        )
    )
)

REM Set cache directories (configurable via CACHE_DIR environment variable)
if not "%CACHE_DIR%"=="" (
    set PIP_CACHE_DIR=%CACHE_DIR%\pip
    set HF_HOME=%CACHE_DIR%\huggingface
    set TORCH_HOME=%CACHE_DIR%\torch
    set XDG_CACHE_HOME=%CACHE_DIR%
    echo Cache directory: %CACHE_DIR%
) else (
    REM Use default system cache locations (typically %USERPROFILE%\.cache)
    echo Using default cache locations
)

echo.
echo Starting Stem Separation Web Interface...
echo Web interface will open at: http://127.0.0.1:7861
echo.
echo Press Ctrl+C to stop the server
echo.

python stem_separation_webui.py

if errorlevel 1 (
    echo.
    echo An error occurred. Check the output above.
    pause
)
