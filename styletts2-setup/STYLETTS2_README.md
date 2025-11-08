# StyleTTS2 Web UI

A clean installation of StyleTTS2 with its own isolated virtual environment.

## 📁 Installation Location

**Default Location:** `E:\AI\tts-webui\styletts2\`

**Note:** You can override the installation path by setting the `STYLETTS2_PATH` environment variable.

- **Root Directory**: `E:\AI\tts-webui\styletts2\` (or `%STYLETTS2_PATH%`)
- **Virtual Environment**: `.venv\`
- **Models**: `models\`
- **Outputs**: `outputs\`
- **Voice Samples**: `voice-samples\`

## 🚀 Quick Start

### Launch the Web UI

**Option 1: PowerShell Script (Recommended)**
```powershell
.\launch_styletts2.ps1
```

**Option 2: Batch File**
```batch
launch_styletts2.bat
```

**Option 3: Direct Python**
```powershell
# Set path if using non-default location
$env:STYLETTS2_PATH = "E:\AI\tts-webui\styletts2"
cd $env:STYLETTS2_PATH
& .venv\Scripts\Activate.ps1
python styletts2_webui.py --server_port 7860
```

The web UI will open automatically in your browser at: `http://localhost:7860`

## 🎯 Features

### Text-to-Speech
- High-quality speech synthesis
- Adjustable timbre and prosody controls
- Variable diffusion steps for quality/speed tradeoff

### Voice Cloning
- Clone any voice from a reference audio sample
- Upload WAV files as reference
- Style transfer capabilities

### Advanced Controls
- **Alpha (Timbre)**: Controls voice characteristics (0-1)
- **Beta (Prosody)**: Controls speech rhythm and intonation (0-1)
- **Diffusion Steps**: Quality vs speed (10-20 recommended)
- **Embedding Scale**: Style intensity control (0.5-2.0)

## 📦 Installed Packages

Key dependencies in the isolated environment:
- PyTorch 2.5.1 (CUDA 12.1)
- StyleTTS2 0.1.6
- Gradio 5.7.1
- Transformers 4.40.2
- librosa 0.10.2
- phonemizer 3.3.0
- And many more...

## 🔧 Environment Isolation

This installation is **completely isolated** from:
- Stable Diffusion WebUI Forge
- Stem Separation
- XTTS-v2
- Any other Python projects

Each has its own virtual environment to prevent dependency conflicts.

## 📝 Usage Tips

1. **First Run**: The model will automatically download on first use (~1-2 GB)
2. **Voice Samples**: Place reference audio files in `voice-samples/` directory
3. **Outputs**: Generated audio is saved to `outputs/` with timestamps
4. **GPU**: CUDA is automatically used if available, falls back to CPU

## 🛠️ Troubleshooting

### Model Loading Issues
If the model fails to load, try:
```powershell
# Set path if using non-default location
$env:STYLETTS2_PATH = "E:\AI\tts-webui\styletts2"
cd $env:STYLETTS2_PATH
& .venv\Scripts\pip.exe install styletts2 --force-reinstall
```

### Dependency Conflicts
To reinstall dependencies:
```powershell
$env:STYLETTS2_PATH = "E:\AI\tts-webui\styletts2"
cd $env:STYLETTS2_PATH
& .venv\Scripts\pip.exe install -r requirements.txt --force-reinstall
```

### Virtual Environment Issues
To recreate the virtual environment:
```powershell
$env:STYLETTS2_PATH = "E:\AI\tts-webui\styletts2"
cd $env:STYLETTS2_PATH
Remove-Item -Path ".venv" -Recurse -Force
python -m venv .venv
& .venv\Scripts\pip.exe install -r requirements.txt
```

## 📊 System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: ~5GB for models and dependencies

## 🔗 Links

- [StyleTTS2 GitHub](https://github.com/yl4579/StyleTTS2)
- [Gradio Documentation](https://gradio.app/)
- [PyTorch Installation](https://pytorch.org/)

## 📅 Installation Date

Installed: November 3, 2025

---

**Note**: This is a fresh installation with all dependencies in an isolated environment on the E drive.
