# Windows Training Issues & Solutions

Comprehensive guide to Windows-specific issues when training StyleTTS2.

**Based on:** 7 debugging sessions, 50-epoch training runs, production deployment

---

## 🔥 Critical Issues (MUST FIX)

### 1. Windows DataLoader Fork Bomb ⚠️

**Confidence:** HIGHEST (Session 3 fix, extensively tested)

**Problem:**
```yaml
# In config_ft.yml:
train_params:
  loader_params:
    train_num_workers: 4  # CRASHES ON WINDOWS
```

**Symptoms:**
- Training prints transcripts infinitely
- Multiple Python processes spawn uncontrollably
- System becomes unresponsive
- CPU usage spikes to 100%
- Must force-kill Python processes

**Root Cause:**
Windows doesn't support `fork()` like Linux. DataLoader with `num_workers>0` tries to spawn worker processes using `spawn()`, but each spawned process re-imports the main script, creating infinite recursion.

**Solution:**
```yaml
# Force to 0 on Windows
loader_params:
  train_num_workers: 0
  val_num_workers: 0
```

**Already Fixed:**
Our `patches/meldataset.py` auto-detects Windows:
```python
if platform.system() == 'Windows':
    num_workers = 0  # Force single-threaded on Windows
```

**Apply patches:**
```powershell
.\apply_patches.ps1
```

---

### 2. Working Directory Path Resolution

**Confidence:** HIGHEST (Session 5 fix)

**Problem:**
```batch
REM Running from debugging_utils/ directory
cd debugging_utils
python ..\StyleTTS2\train_finetune.py
```

Relative paths in `config_ft.yml` resolve from `debugging_utils/` instead of `StyleTTS2/`:
```yaml
# These paths break:
train_data: "../../datasets/train_list.txt"  # Wrong base directory
log_dir: "./Models/LJSpeech"  # Saves to debugging_utils/Models/
```

**Symptoms:**
- Checkpoints save to wrong directory
- Can't find pretrained models
- Training data not found
- Config paths fail to resolve

**Solution:**
**ALWAYS** `cd` into `StyleTTS2/` directory before running training:

```batch
REM WRONG
python StyleTTS2\train_finetune.py

REM CORRECT
cd StyleTTS2
python train_finetune.py
```

**Already Fixed:**
`run_finetune_safe.bat` includes:
```batch
pushd StyleTTS2
echo Working directory: %CD%
python train_finetune.py -p Configs\config_ft.yml
popd
```

---

### 3. CUDA Kernel Silent Crashes

**Confidence:** HIGH (Session 7 diagnosis)

**Problem:**
Training exits silently after batch 0 with no error messages:
```
Epoch [0] | Step 0 | Loss: 5.234 | Grad Norm: 2.1
[Script exits with no error]
```

**Root Cause:**
CUDA kernel crashes in `monotonic_align.maximum_path` when processing:
- Very short audio samples (<5s, <500 mel frames)
- Extreme length ratios between text and audio
- Out-of-range indices in alignment kernels

**Symptoms:**
- No Python exception raised
- No traceback in logs
- Silent exit after first batch
- CUDA error in `nvidia-smi` (sometimes)

**Detection:**
Enable synchronous CUDA errors:
```batch
set CUDA_LAUNCH_BLOCKING=1
```

This forces CUDA to run synchronously, showing actual error location.

**Solution A: Filter Dataset (Preventive)**

Remove short audio during export or in dataset class:
```python
# In meldataset.py __getitem__
if mel.shape[1] < 500:  # Skip if <500 frames (~2 seconds)
    return self.__getitem__(random.randint(0, len(self)-1))
```

**Solution B: Add Validation (Defensive)**

Validate tensors before CUDA operations:
```python
# Before maximum_path call
if text_len < 5 or mel_len < 50:
    logger.warning(f"Skipping short sample: text={text_len}, mel={mel_len}")
    continue
```

**Already Fixed:**
- `run_finetune_safe.bat` sets `CUDA_LAUNCH_BLOCKING=1`
- `patches/train_finetune.py` adds validation checks
- `validate_dataset.py` detects problematic samples

---

### 4. monotonic_align Installation

**Confidence:** HIGHEST (Required for training)

**Problem:**
PyPI version is incomplete/broken:
```powershell
pip install monotonic_align  # WRONG - doesn't work
```

**Symptoms:**
```python
ImportError: cannot import name 'maximum_path_c' from 'monotonic_align'
```

**Requirements:**
1. Microsoft C++ Build Tools (for Cython compilation)
2. Git (for cloning repository)

**Solution:**
```powershell
# Always install from GitHub
pip install git+https://github.com/resemble-ai/monotonic_align.git
```

**Automated installer provided:**
```powershell
python install_monotonic_align.py
```

This script:
- Checks prerequisites
- Clones repository
- Compiles Cython extensions
- Validates installation
- Logs everything

---

## ⚠️ Common Issues (Frequent)

### 5. espeak-ng Not Found

**Confidence:** HIGH

**Problem:**
```python
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
# Should output version number
```

**In batch scripts:**
```batch
set "PATH=C:\Program Files\eSpeak NG;%PATH%"
set "PHONEMIZER_ESPEAK_LIBRARY=C:\Program Files\eSpeak NG\libespeak-ng.dll"
```

---

### 6. Import Delays (1-2 Minutes)

**Confidence:** MED (Session 4 observation)

**Problem:**
Training script appears frozen after "Using device: cuda"

**Root Cause:**
First import of large packages (transformers, scipy, sklearn) takes time:
```python
from transformers import AutoModelForMaskedLM  # 30-60 seconds
from scipy.io.wavfile import read              # 10-20 seconds
from sklearn.preprocessing import StandardScaler  # 10-20 seconds
```

**Solution:**
**DO NOT interrupt!** Wait 1-2 minutes without touching keyboard/mouse.

**Normal behavior:**
```
Using device: cuda
[Wait 1-2 minutes - this is normal]
Epoch [0] | Step 0 | ...
```

**Add to scripts:**
```batch
echo NOTE: Initial imports may take 1-2 minutes
echo Please wait without interruption...
```

---

### 7. Venv Path Portability

**Confidence:** HIGHEST (Session 4.5 fix)

**Problem:**
Virtual environment created with C-drive Python breaks when Python is deleted:
```
Fatal Error: Unable to find python.exe at C:\Users\...\python.exe
```

**Root Cause:**
Venvs store absolute paths to original Python interpreter. Not portable.

**Solution:**
Use a stable Python installation that won't be deleted:

```powershell
# AVOID - breaks if Python is uninstalled later
C:\Users\YourName\AppData\Local\Programs\Python\Python310\python.exe -m venv .venv

# BETTER - use system-wide or dedicated Python installation
# Option 1: System-wide Python
C:\Python310\python.exe -m venv .venv

# Option 2: Dedicated project Python on separate drive
D:\python\Python310\python.exe -m venv .venv
```

**Best Practices:**
- Install Python to a stable location you won't delete
- Avoid user-specific AppData locations for shared projects
- Consider using a dedicated drive for development tools
- Document Python path in project README

**Recovery if broken:**
```powershell
# Remove broken venv
Remove-Item .venv -Recurse -Force

# Recreate with stable Python installation
path\to\stable\python.exe -m venv .venv

# Re-install dependencies
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🔧 Debugging Tools

### Enable Verbose CUDA Errors

```batch
set CUDA_LAUNCH_BLOCKING=1
set TORCH_USE_CUDA_DSA=1
```

### Check CUDA Status

```powershell
nvidia-smi
# Look for Python processes and memory usage
```

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"Device Count: {torch.cuda.device_count()}")
print(f"Current Device: {torch.cuda.current_device()}")
print(f"Device Name: {torch.cuda.get_device_name(0)}")
```

### Memory Debugging

```batch
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True
```

### Process Monitoring

```powershell
# Watch for runaway processes
Get-Process python | Select-Object Id, CPU, WorkingSet
```

---

## 📋 Pre-Training Checklist

Before starting training, verify:

- [ ] **Working directory:** `cd StyleTTS2` before running
- [ ] **num_workers:** Set to 0 in config_ft.yml
- [ ] **monotonic_align:** Installed from GitHub
- [ ] **espeak-ng:** Installed and in PATH
- [ ] **CUDA:** `torch.cuda.is_available()` returns True
- [ ] **Dataset:** Validated with `validate_dataset.py`
- [ ] **Config paths:** All paths relative to StyleTTS2/ directory
- [ ] **Venv:** Created with E-drive Python
- [ ] **Patches:** Applied with `apply_patches.ps1`

**Quick verification:**
```powershell
# From styletts2-setup/
python -c "
import os, torch, platform
print('✓ Python OK')
print(f'✓ CUDA: {torch.cuda.is_available()}')
print(f'✓ Platform: {platform.system()}')
assert os.path.exists('StyleTTS2'), 'StyleTTS2/ not found'
print('✓ StyleTTS2 directory exists')
"
```

---

## 🚀 Recommended Training Launch

Use the safe mode launcher:

```batch
cd styletts2-setup
run_finetune_safe.bat
```

This script:
- ✅ Sets CUDA debugging flags
- ✅ Activates correct venv
- ✅ Checks monotonic_align installation
- ✅ Changes to StyleTTS2 directory
- ✅ Validates config exists
- ✅ Provides clear error messages

---

## 📊 Known Performance Impact

**Windows vs Linux:**

| Metric | Linux | Windows | Difference |
|--------|-------|---------|------------|
| DataLoader | Multi-process | Single-thread | ~20% slower |
| CUDA Kernels | Same | Same | No difference |
| File I/O | Fast | Slightly slower | ~5-10% slower |
| Overall Training | Baseline | ~25% slower | Acceptable |

**RTX 3060 12GB (batch_size=8):**
- **Windows:** ~4 hours per 50 epochs (tested)
- **Linux:** ~3 hours per 50 epochs (estimated)

**Trade-off acceptable:** Stability > speed for Windows users.

---

## 🐛 Error Messages & Fixes

### "CUDA error: device-side assert triggered"

**Cause:** Out-of-range tensor indices  
**Fix:** Enable `CUDA_LAUNCH_BLOCKING=1`, check dataset for invalid samples

### "RuntimeError: DataLoader worker exited unexpectedly"

**Cause:** num_workers > 0 on Windows  
**Fix:** Set num_workers=0 in config

### "FileNotFoundError: train_list.txt"

**Cause:** Running from wrong directory  
**Fix:** `cd StyleTTS2` before training

### "ModuleNotFoundError: No module named 'monotonic_align'"

**Cause:** Not installed or PyPI version  
**Fix:** `pip install git+https://github.com/resemble-ai/monotonic_align.git`

### Training freezes at "Using device: cuda"

**Cause:** Normal - importing transformers  
**Fix:** Wait 1-2 minutes without interruption

---

## 📚 Related Documentation

- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) - General issues
- [DEPENDENCY_MANAGEMENT.md](DEPENDENCY_MANAGEMENT.md) - Package conflicts
- [STYLETTS2_INSTALLATION.md](STYLETTS2_INSTALLATION.md) - Full setup guide
- [patches/README.md](../styletts2-setup/patches/README.md) - Code patch details

---

**Last Updated:** November 2025  
**Tested On:** Windows 10/11, Python 3.10, RTX 3060 12GB, CUDA 12.1  
**Confidence:** HIGHEST - All issues encountered and resolved over 7 debugging sessions
