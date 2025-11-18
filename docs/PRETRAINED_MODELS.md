# Pretrained Models Guide

This guide covers all the pretrained models required for StyleTTS2 fine-tuning and where to obtain them.

---

## Overview

StyleTTS2 fine-tuning requires **4 types of pretrained models**:

1. **LibriTTS Base Model** - The main checkpoint to fine-tune from
2. **ASR Model** - Automatic Speech Recognition for phoneme alignment
3. **F0 Model** - Fundamental frequency (pitch) estimation
4. **PLBERT Model** - Phoneme-level BERT for text representation

**Total download size**: ~2.5 GB  
**Required disk space**: ~5 GB (with checkpoints during training)

---

## ⚠️ IMPORTANT: Choosing the Right Base Model (American vs British)

### Accent Contamination Issue

The pretrained model you choose will **significantly impact your fine-tuned voice's accent**, especially in longer generations. There are two primary options:

| Model | Accent | Pros | Cons | Best For |
|-------|--------|------|------|----------|
| **LibriTTS** | British/Mixed | Well-tested, widely used | Can cause British accent bleed in fine-tuned models | European English projects |
| **LJSpeech** | American English | Clean American accent base, no contamination | Less commonly documented | American English voice cloning |

### Recommendation

**If you want American English voices**, use the **LJSpeech** pretrained model instead of LibriTTS. This prevents accent contamination where British pronunciation bleeds through during longer speech generation.

### Quick Setup for American English

**Download LJSpeech Model:**
```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "StyleTTS2/Models/LJSpeech_American"

# Download model (750 MB)
# Model: https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/epoch_2nd_00100.pth
# Config: https://huggingface.co/yl4579/StyleTTS2-LJSpeech/resolve/main/Models/LJSpeech/config.yml
```

**Update your config_ft.yml:**
```yaml
pretrained_model: "Models/LJSpeech_American/epoch_2nd_00100.pth"
second_stage_load_pretrained: true
```

**LJSpeech Dataset Details:**
- **Duration**: 24 hours of speech
- **Speaker**: Single female (Linda Johnson)
- **Quality**: Studio recordings
- **Accent**: Standard American English (General American)

### How Fine-tuning Works

When you fine-tune:
1. **Base model** provides accent foundation (American or British)
2. **Your training data** adds voice characteristics (timbre, style, speaking pattern)
3. **Result**: Your custom voice with the base model's accent

Even though LJSpeech is single-speaker, StyleTTS2's architecture **supports multispeaker training** when you fine-tune with proper speaker IDs.

---

## 1. LibriTTS Pretrained Model (British/Mixed Accent)

### Purpose
This is the main StyleTTS2 model checkpoint trained on the LibriTTS dataset. You will fine-tune this model on your custom voice data.

**⚠️ Note**: This model contains British English accent bias. For American English voices, see the LJSpeech option above.

### Files Needed
- **File**: `epochs_2nd_00020.pth`
- **Size**: ~400 MB
- **Location**: `StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth`

### Download

**Primary Source: Google Drive**
1. Visit: https://drive.google.com/drive/folders/1CJ8jbIT0qGfaIBTgb1U6RG4OI8VKXV05?usp=sharing
2. Download `epochs_2nd_00020.pth` (or latest checkpoint)
3. Save to: `StyleTTS2/Models/LibriTTS/`

**Using PowerShell:**
```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "C:\Projects\StyleTTS2\Models\LibriTTS"

# After manual download, move file:
Move-Item "Downloads\epochs_2nd_00020.pth" "C:\Projects\StyleTTS2\Models\LibriTTS\"
```

### Verification
```powershell
python -c "import torch; checkpoint = torch.load('Models/LibriTTS/epochs_2nd_00020.pth', map_location='cpu'); print(f'Loaded checkpoint with {len(checkpoint)} keys'); print('✓ LibriTTS model OK')"
```

Expected output:
```
Loaded checkpoint with X keys
✓ LibriTTS model OK
```

---

## 2. ASR Model (Automatic Speech Recognition)

### Purpose
The ASR model is used during training to extract phoneme-level alignments from your audio. This ensures the model learns accurate pronunciation timing.

### Files Needed
- **File**: `epoch_00080.pth`
- **Size**: ~150 MB
- **Location**: `StyleTTS2/Utils/ASR/epoch_00080.pth`

### Download

**Primary Source: Google Drive**
1. Visit: Same Google Drive link as LibriTTS model
2. Download `epoch_00080.pth`
3. Save to: `StyleTTS2/Utils/ASR/`

**Using PowerShell:**
```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "C:\Projects\StyleTTS2\Utils\ASR"

# After manual download:
Move-Item "Downloads\epoch_00080.pth" "C:\Projects\StyleTTS2\Utils\ASR\"
```

### Additional File
The ASR model also needs a config file:
- **File**: `config.yml`
- **Location**: `StyleTTS2/Utils/ASR/config.yml`
- **Source**: Included in StyleTTS2 repository (already present after cloning)

### Verification
```powershell
python -c "import torch; checkpoint = torch.load('Utils/ASR/epoch_00080.pth', map_location='cpu'); print('✓ ASR model OK')"
```

---

## 3. F0 Model (Pitch Estimator)

### Purpose
The F0 (fundamental frequency) model extracts pitch information from audio. This helps StyleTTS2 learn voice prosody and intonation.

### Files Needed
- **File**: `bst.t7` (JDC model)
- **Size**: ~50 MB
- **Location**: `StyleTTS2/Utils/JDC/bst.t7`

### Download

**Primary Source: Google Drive**
1. Visit: Same Google Drive link as other models
2. Download `bst.t7`
3. Save to: `StyleTTS2/Utils/JDC/`

**Using PowerShell:**
```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "C:\Projects\StyleTTS2\Utils\JDC"

# After manual download:
Move-Item "Downloads\bst.t7" "C:\Projects\StyleTTS2\Utils\JDC\"
```

### Verification
```powershell
python -c "import torch; model = torch.load('Utils/JDC/bst.t7', map_location='cpu'); print('✓ F0 model OK')"
```

---

## 4. PLBERT Model (Phoneme-Level BERT)

### Purpose
PLBERT provides rich phoneme-level text representations that help StyleTTS2 understand linguistic context during training.

### Files Needed
Multiple files from HuggingFace repository:
- `config.json`
- `pytorch_model.bin` (~400 MB)
- `step_2000000.t7` (~400 MB)
- Additional config files

**Total Size**: ~1.5 GB  
**Location**: `StyleTTS2/Utils/PLBERT/`

### Download

**Option 1: Using Git LFS (Recommended)**

```powershell
# Install git-lfs if not already installed
# Download from: https://git-lfs.github.com/

# Initialize git-lfs
git lfs install

# Create directory
New-Item -ItemType Directory -Force -Path "C:\Projects\StyleTTS2\Utils"
cd "C:\Projects\StyleTTS2\Utils"

# Clone the PLBERT repository
git clone https://huggingface.co/yl4579/StyleTTS2-LibriTTS PLBERT
```

**Option 2: Manual Download**

1. Visit: https://huggingface.co/yl4579/StyleTTS2-LibriTTS/tree/main
2. Click "Files and versions" tab
3. Download all files:
   - `config.json`
   - `pytorch_model.bin`
   - `step_2000000.t7`
   - `vocab.txt`
   - Other config files
4. Save all to: `StyleTTS2/Utils/PLBERT/`

**Using PowerShell:**
```powershell
# Create directory
New-Item -ItemType Directory -Force -Path "C:\Projects\StyleTTS2\Utils\PLBERT"

# After manual downloads, move files:
Move-Item "Downloads\config.json" "C:\Projects\StyleTTS2\Utils\PLBERT\"
Move-Item "Downloads\pytorch_model.bin" "C:\Projects\StyleTTS2\Utils\PLBERT\"
Move-Item "Downloads\step_2000000.t7" "C:\Projects\StyleTTS2\Utils\PLBERT\"
# ... repeat for other files
```

### Verification
```powershell
python -c "import os; files = ['config.json', 'pytorch_model.bin', 'step_2000000.t7']; missing = [f for f in files if not os.path.exists(f'Utils/PLBERT/{f}')]; print('✓ PLBERT model OK' if not missing else f'Missing: {missing}')"
```

---

## Complete Directory Structure

After downloading all models, your StyleTTS2 directory should look like this:

```
StyleTTS2/
├── Models/
│   └── LibriTTS/
│       └── epochs_2nd_00020.pth          (Main checkpoint - 400MB)
│
├── Utils/
│   ├── ASR/
│   │   ├── config.yml                    (Config file - included in repo)
│   │   └── epoch_00080.pth               (ASR model - 150MB)
│   │
│   ├── JDC/
│   │   └── bst.t7                        (F0 model - 50MB)
│   │
│   └── PLBERT/
│       ├── config.json
│       ├── pytorch_model.bin             (BERT weights - 400MB)
│       ├── step_2000000.t7               (PLBERT checkpoint - 400MB)
│       ├── vocab.txt
│       └── [other config files]
│
└── Configs/
    └── config_ft.yml                      (Training config)
```

---

## Verify All Models

Run this comprehensive check:

```powershell
# Check all required files exist
$models = @{
    "LibriTTS" = "Models/LibriTTS/epochs_2nd_00020.pth"
    "ASR" = "Utils/ASR/epoch_00080.pth"
    "F0" = "Utils/JDC/bst.t7"
    "PLBERT config" = "Utils/PLBERT/config.json"
    "PLBERT model" = "Utils/PLBERT/pytorch_model.bin"
    "PLBERT checkpoint" = "Utils/PLBERT/step_2000000.t7"
}

foreach ($name in $models.Keys) {
    $path = $models[$name]
    if (Test-Path $path) {
        $size = (Get-Item $path).Length / 1MB
        Write-Host "✓ $name - $([math]::Round($size, 1)) MB" -ForegroundColor Green
    } else {
        Write-Host "✗ $name - MISSING" -ForegroundColor Red
    }
}
```

Expected output (all green checkmarks):
```
✓ LibriTTS - 402.3 MB
✓ ASR - 151.8 MB
✓ F0 - 49.2 MB
✓ PLBERT config - 0.5 KB
✓ PLBERT model - 418.7 MB
✓ PLBERT checkpoint - 411.2 MB
```

---

## Config File Setup

After downloading models, update `Configs/config_ft.yml` to point to them:

```yaml
# Pretrained model paths
pretrained_model: Models/LibriTTS/epochs_2nd_00020.pth

# ASR model
ASR_config: Utils/ASR/config.yml
ASR_path: Utils/ASR/epoch_00080.pth

# F0 model
F0_path: Utils/JDC/bst.t7

# PLBERT model
PLBERT_dir: Utils/PLBERT/
```

**Important**: Use forward slashes `/` even on Windows, or absolute paths:
```yaml
# Alternative: absolute paths (use your actual path)
pretrained_model: C:/Projects/StyleTTS2/Models/LibriTTS/epochs_2nd_00020.pth
```

---

## Troubleshooting

### Downloads are slow
- **Solution**: Use a download manager (Free Download Manager, IDM)
- **Alternative**: Download during off-peak hours
- **VPN**: Try different VPN servers if restricted

### Git LFS fails
```
Error downloading object: ...
```
**Solution**: Download files manually from HuggingFace web interface instead.

### "File not found" during training
```
FileNotFoundError: [Errno 2] No such file or directory: 'Models/LibriTTS/...'
```
**Solution**: 
1. Check paths in `config_ft.yml` match actual file locations
2. Use absolute paths if relative paths fail
3. Verify files exist with `Test-Path` command

### Checkpoint loading errors
```
RuntimeError: Error(s) in loading state_dict...
```
**Solution**:
1. Re-download the checkpoint (may be corrupted)
2. Verify file size matches expected size
3. Check that you're using compatible versions

### Disk space issues
```
OSError: [Errno 28] No space left on device
```
**Solution**:
1. Clear space on drive (need ~5GB free)
2. Move models to larger drive
3. Delete old checkpoints from previous training runs

---

## Alternative Model Sources

If Google Drive is inaccessible:

### HuggingFace Mirror
Some models may be available on HuggingFace:
- https://huggingface.co/yl4579
- Search for "StyleTTS2" models

### GitHub Releases
Check StyleTTS2 repository releases:
- https://github.com/yl4579/StyleTTS2/releases

### Community Mirrors
- Academic mirrors (if you have access)
- Community-hosted mirrors (check StyleTTS2 discussions)

---

## Using Custom Pretrained Models

If you want to start from a different base model:

1. **Ensure vocabulary compatibility**: Model must use same 178-token vocabulary
2. **Check architecture compatibility**: Must be StyleTTS2 architecture
3. **Update config paths**: Point to your custom model in `config_ft.yml`

```yaml
pretrained_model: path/to/your/custom_model.pth
```

**Note**: Starting from the official LibriTTS model is recommended for best results.

---

## Model Licensing

**LibriTTS Model**:
- Based on LibriTTS dataset (CC BY 4.0)
- Free for research and commercial use with attribution

**PLBERT Model**:
- Check HuggingFace repository for specific license
- Generally permissive for research

**ASR & F0 Models**:
- Part of StyleTTS2 project (MIT License)

Always verify current licensing terms before commercial use.

---

## Next Steps

After setting up all pretrained models:

1. ✅ Verify all models load correctly
2. ✅ Update `config_ft.yml` with correct paths
3. ✅ Prepare your custom dataset
4. ✅ Validate dataset with toolkit scripts
5. ✅ Apply code patches from this toolkit
6. 🚀 Start training!

See [STYLETTS2_INSTALLATION.md](STYLETTS2_INSTALLATION.md) for complete setup guide.

---

*Last updated: November 2025*
