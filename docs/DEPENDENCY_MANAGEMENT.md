# Dependency Management Guide

Complete guide for managing dependencies in the StyleTTS2 Dataset Toolkit.

---

## 📋 Quick Reference

### Tested Configuration

**Platform:** Windows 10/11  
**Python:** 3.10.11  
**GPU:** NVIDIA RTX 3060 12GB  
**CUDA:** 12.1

**Core Versions (Battle-Tested):**
```
torch==2.5.1+cu121
torchaudio==2.5.1+cu121
transformers==4.40.2
gradio==4.44.1
huggingface-hub==0.19.4
num2words==0.5.14 (CRITICAL)
```

---

## 🚀 Clean Installation

### Step-by-Step Process

```powershell
# 1. Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Upgrade pip
python -m pip install --upgrade pip

# 3. Install PyTorch with CUDA (FIRST!)
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# 4. Verify CUDA works
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# 5. Install main dependencies
pip install -r requirements.txt

# 6. Install monotonic_align from GitHub (for training)
pip install git+https://github.com/resemble-ai/monotonic_align.git

# 7. OPTIONAL: For WebUI pretrained model support
pip install styletts2==0.1.6 --no-deps

# 8. Verify installation
python -c "import librosa, gradio, whisper, num2words; print('All imports OK')"
```

---

## ⚠️ Known Conflicts

### 1. huggingface-hub Version Conflict

**Problem:**
```
styletts2 package pins: huggingface-hub<0.20
gradio 4.44+ requires: huggingface-hub>=0.20
transformers 4.40+ requires: huggingface-hub>=0.19
```

**Symptoms:**
- Pip downgrades huggingface-hub to 0.19
- Gradio import warnings or failures
- Transformers download issues

**Solution:**
```powershell
# Install styletts2 with --no-deps to prevent downgrades
pip install styletts2==0.1.6 --no-deps
```

**Why it works:**
- styletts2 package is only needed for pretrained model inference
- For fine-tuning, we use StyleTTS2 training scripts directly
- Training scripts don't need the styletts2 package

**Confidence:** HIGHEST (tested over 7 debugging sessions)

---

### 2. langchain Version Lock

**Problem:**
```
langchain>=0.1.0 removed text_splitter module
styletts2 package expects: langchain.text_splitter
```

**Symptoms:**
```python
ModuleNotFoundError: No module named 'langchain.text_splitter'
```

**Solution:**
```powershell
# Option A: Use old langchain (if using styletts2 package)
pip install "langchain<0.1.0"
pip uninstall langchain-text-splitters -y

# Option B: Don't use styletts2 package (recommended for training)
# Just skip installing it entirely
```

**Confidence:** HIGH (documented in Session 4.5)

---

### 3. monotonic_align PyPI Issue

**Problem:**
- PyPI version is incomplete/broken
- Missing Cython build artifacts

**Symptoms:**
```python
ImportError: cannot import name 'maximum_path_c' from 'monotonic_align'
```

**Solution:**
```powershell
# Always install from GitHub
pip install git+https://github.com/resemble-ai/monotonic_align.git
```

**Requirements:**
- Microsoft C++ Build Tools (Windows)
- Cython (auto-installed as dependency)

**Confidence:** HIGHEST (required for all training)

---

### 4. PyTorch Version Compatibility

**Current (Tested):**
```
torch==2.5.1+cu121
transformers==4.40.2
```

**Newer (Available but untested):**
```
torch==2.6.0+cu124  # Includes CVE-2025-32434 security fix
transformers==4.57.1+  # Requires torch 2.6+
```

**Migration Path:**
```powershell
# Upgrade both together
pip install torch==2.6.0+cu124 torchaudio==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.57.1
```

**Test thoroughly before production!**

**Confidence:** MED (newer versions available but not battle-tested)

---

### 5. Windows DataLoader Fork Bomb

**Problem:**
```yaml
# In config_ft.yml:
loader_params:
  train_num_workers: 4  # CRASHES ON WINDOWS
```

**Symptoms:**
- Training prints transcripts infinitely
- Multiple Python processes spawn
- System becomes unresponsive

**Solution:**
```yaml
# Force to 0 on Windows
loader_params:
  train_num_workers: 0
  val_num_workers: 0
```

**Already Fixed:**
Our `patches/meldataset.py` auto-detects Windows and forces `num_workers=0`

**Confidence:** HIGHEST (Session 3 fix, tested extensively)

---

### 6. espeak-ng System Dependency

**Problem:**
```python
phonemizer requires espeak-ng system binary
```

**Symptoms:**
```
RuntimeError: espeak-ng not found
```

**Solution:**

**Windows:**
1. Download: https://github.com/espeak-ng/espeak-ng/releases
2. Install to: `C:\Program Files\eSpeak NG\`
3. Add to PATH or set environment variable:
   ```powershell
   $env:PHONEMIZER_ESPEAK_LIBRARY = "C:\Program Files\eSpeak NG\libespeak-ng.dll"
   ```

**Verify:**
```powershell
espeak-ng --version
```

**Confidence:** HIGHEST (required for all text processing)

---

## 🔄 Updating Dependencies

### Safe Update Process

```powershell
# 1. Backup current environment
pip freeze > requirements_backup.txt

# 2. Check for outdated packages
pip list --outdated

# 3. Update conservatively (one at a time)
pip install --upgrade package_name

# 4. Test thoroughly
python styletts2_webui.py  # Test WebUI
python batch_inference_epochs.py --start-epoch 0 --end-epoch 0  # Test inference

# 5. If issues, rollback
pip install -r requirements_backup.txt
```

### What to Update

**Safe to update:**
- matplotlib, pandas, numpy (minor versions)
- tqdm, click (utilities)
- tensorboard (monitoring)

**Update with caution:**
- gradio (UI may change)
- transformers (model compatibility)
- librosa, soundfile (audio processing)

**DO NOT update without testing:**
- torch, torchaudio (breaks everything)
- huggingface-hub (conflicts)
- styletts2 package (frozen at 0.1.6)

---

## 🧪 Dependency Testing

### Automated Checks

```powershell
# Run from styletts2-setup/

# 1. Import test
python -c "
import torch
import librosa
import gradio
import whisper
import num2words
import transformers
print('✓ All core imports successful')
print(f'✓ CUDA Available: {torch.cuda.is_available()}')
"

# 2. Version check
python -c "
import torch, gradio, transformers
print(f'torch: {torch.__version__}')
print(f'gradio: {gradio.__version__}')
print(f'transformers: {transformers.__version__}')
"

# 3. Full integration test
pytest tests/test_dependencies.py  # If test suite exists
```

---

## 📦 requirements.txt Structure

**Main requirements.txt:**
- Core dependencies with tested versions
- Installation order documentation
- Known conflict resolutions
- Memory estimates

**requirements-dev.txt:**
- Testing frameworks (pytest)
- Code quality tools (ruff, mypy)
- Documentation tools (sphinx)
- Profiling utilities

**requirements_freeze.txt (local repo only):**
- Complete pip freeze output
- Includes all transitive dependencies
- Reference for exact working state

---

## 🔧 Troubleshooting Installation

### "No matching distribution found"

**Cause:** PyTorch CUDA index not specified

**Fix:**
```powershell
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu121
```

### "Could not build wheels for monotonic_align"

**Cause:** Missing C++ compiler

**Fix:**
1. Install Microsoft C++ Build Tools
2. Restart terminal
3. Retry: `pip install git+https://github.com/resemble-ai/monotonic_align.git`

### "ImportError: DLL load failed"

**Cause:** Missing CUDA runtime or wrong CUDA version

**Fix:**
1. Check CUDA version: `nvidia-smi`
2. Install matching PyTorch version
3. Verify: `python -c "import torch; print(torch.cuda.is_available())"`

### Dependency conflict loop

**Cause:** Competing version requirements

**Fix:**
```powershell
# Nuclear option: Fresh venv
deactivate
Remove-Item .venv -Recurse -Force
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Follow clean installation steps
```

---

## 📊 Memory Requirements

### Installation Size

**Minimum:** ~8 GB
- PyTorch + CUDA: ~5 GB
- Other packages: ~1-2 GB
- Models (on first use): ~2 GB

**Recommended:** ~15 GB (includes cache)

### Runtime Memory

**VRAM (RTX 3060 12GB):**
- Training (batch_size=8): ~10 GB
- Inference: ~2 GB
- WebUI idle: ~1 GB
- Whisper transcription: ~1-5 GB (model dependent)

**System RAM:**
- Minimum: 8 GB
- Recommended: 16 GB
- Training large datasets: 32 GB

---

## 🔐 Security Considerations

### CVE-2025-32434 (PyTorch)

**Issue:** Malicious pickle files in torch.load()

**Affected:** torch < 2.6.0

**Mitigation:**
1. Upgrade to torch 2.6.0+ when available
2. Only load checkpoints from trusted sources
3. Use `weights_only=True` when possible

### Dependency Vulnerabilities

**Check regularly:**
```powershell
pip-audit  # Requires: pip install pip-audit
```

**Update security-critical packages:**
```powershell
pip install --upgrade requests urllib3 certifi
```

---

## 📝 Best Practices

### 1. Pin Versions in Production

Always use exact versions in requirements.txt:
```
✓ gradio==4.44.1
✗ gradio>=4.0.0
```

### 2. Document Why

Add comments for non-obvious pins:
```
num2words==0.5.14  # CRITICAL: StyleTTS2 vocab requires this for digit conversion
```

### 3. Test After Updates

Don't update in production without testing:
```powershell
# Dev environment first
pip install --upgrade package_name
# Run full test suite
pytest tests/
# Then deploy to production
```

### 4. Keep Freeze File

Generate after stable configuration:
```powershell
pip freeze > requirements_freeze_$(Get-Date -Format "yyyyMMdd").txt
```

### 5. Use Virtual Environments

Never install globally:
```powershell
python -m venv .venv  # Always use venv
```

---

## 📚 Additional Resources

- [STYLETTS2_INSTALLATION.md](STYLETTS2_INSTALLATION.md) - Complete setup guide
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - Common issues
- [PRETRAINED_MODELS.md](PRETRAINED_MODELS.md) - Model downloads
- [requirements.txt](../styletts2-setup/requirements.txt) - Main dependencies
- [requirements-dev.txt](../styletts2-setup/requirements-dev.txt) - Development tools

---

**Last Updated:** November 2025  
**Based on:** 7 debugging sessions, 50+ epoch training runs, production deployment
