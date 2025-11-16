# StyleTTS2 Training Script
# This script activates the virtual environment and runs the fine-tuning training

$ErrorActionPreference = "Stop"

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "       StyleTTS2 Fine-Tuning Training" -ForegroundColor Cyan
Write-Host "===============================================" -ForegroundColor Cyan
Write-Host ""

# Set paths (relative to script directory)
# Script is now in launchers/ subfolder, go up one level to styletts2-setup/
$SCRIPT_DIR = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VENV_PATH = "$SCRIPT_DIR\.venv"
$TRAIN_SCRIPT = "$SCRIPT_DIR\StyleTTS2\train_finetune.py"
$CONFIG_PATH = "$SCRIPT_DIR\StyleTTS2\Configs\config_ft.yml"

# Check if virtual environment exists
if (-not (Test-Path "$VENV_PATH\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found at $VENV_PATH" -ForegroundColor Red
    Write-Host "Please run the installation script first." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if training script exists
if (-not (Test-Path $TRAIN_SCRIPT)) {
    Write-Host "ERROR: Training script not found at $TRAIN_SCRIPT" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if config exists
if (-not (Test-Path $CONFIG_PATH)) {
    Write-Host "ERROR: Config file not found at $CONFIG_PATH" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Using virtual environment at: $VENV_PATH" -ForegroundColor Cyan
Write-Host "Starting StyleTTS2 fine-tuning training..." -ForegroundColor Green
Write-Host "Config: $CONFIG_PATH" -ForegroundColor Yellow
Write-Host ""
Write-Host "This may take a long time depending on your dataset size and GPU." -ForegroundColor Yellow
Write-Host "Training logs will be saved to the Models directory." -ForegroundColor Yellow
Write-Host ""

# Change to the StyleTTS2 directory
Set-Location "$SCRIPT_DIR\StyleTTS2"

# Run training with the virtual environment's Python
& "$VENV_PATH\Scripts\python.exe" $TRAIN_SCRIPT --config_path $CONFIG_PATH

Write-Host ""
Write-Host "Training script has finished." -ForegroundColor Yellow
Read-Host "Press Enter to exit"
