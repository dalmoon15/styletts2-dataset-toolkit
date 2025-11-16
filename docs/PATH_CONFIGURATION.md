# 🔧 Path Configuration Guide

This guide explains how to configure paths for the StyleTTS2 Dataset Toolkit.

---

## 📍 Understanding the Structure

The toolkit references actual implementations located in separate directories:

- **Stem Separation**: Located at `E:\AI\stem-separation-webui\` (default)
- **StyleTTS2**: Located at `E:\AI\tts-webui\styletts2\` (default)

The repository (`E:\styletts2-dataset-toolkit\`) contains launchers and documentation that reference these implementations.

---

## 🔧 Configuring Paths

### Environment Variables

You can override default paths using environment variables:

#### StyleTTS2 Path
```powershell
# Set StyleTTS2 installation path
$env:STYLETTS2_PATH = "C:\MyProjects\StyleTTS2"
```

#### FFmpeg Path
```powershell
# Set FFmpeg installation path
$env:FFMPEG_PATH = "C:\ffmpeg\bin"
```

#### Cache Directory
```powershell
# Set cache directory (for models, downloads, etc.)
$env:CACHE_DIR = "D:\ML_Cache"
```

#### Stem Separation Path (if needed)
```powershell
# Set stem separation installation path
$env:STEM_SEPARATION_PATH = "C:\MyProjects\stem-separation"
```

---

## 🚀 Launcher Behavior

### StyleTTS2 Launcher (`launch_styletts2.ps1`)

**Path Resolution Priority:**
1. `STYLETTS2_PATH` environment variable
2. Default location: `C:\StyleTTS2` (configurable in script)
3. Current directory (fallback for portable installations)

**FFmpeg Resolution Priority:**
1. System PATH
2. `FFMPEG_PATH` environment variable
3. Common installation locations:
   - `C:\ffmpeg\bin`
   - `%ProgramFiles%\ffmpeg\bin`
   - `%ProgramFiles(x86)%\ffmpeg\bin`
6. `%ProgramFiles(x86)%\ffmpeg\bin`

### Stem Separation Launcher (`launch_stem_separation.bat`)

**Path Resolution:**
- Uses current directory (where the launcher is located)
- Assumes virtual environment is in `venv\` subdirectory

**FFmpeg Resolution Priority:**
1. System PATH
2. `FFMPEG_PATH` environment variable
3. `E:\AI\tools\ffmpeg\bin`
4. `C:\ffmpeg\bin`

**Cache Resolution Priority:**
1. `CACHE_DIR` environment variable
2. `E:\.cache\` (if E: drive exists)
3. Default system cache locations (AppData)

---

## 💡 Examples

### Example 1: Custom StyleTTS2 Location

If you installed StyleTTS2 in a different location:

```powershell
# Set environment variable
$env:STYLETTS2_PATH = "D:\Projects\StyleTTS2"

# Launch
cd E:\styletts2-dataset-toolkit\styletts2-setup
.\launch_styletts2.ps1
```

### Example 2: Custom FFmpeg Location

If FFmpeg is installed in a custom location:

```powershell
# Set environment variable
$env:FFMPEG_PATH = "C:\Tools\ffmpeg\bin"

# Launch
cd E:\styletts2-dataset-toolkit\stem-separation
.\launch_stem_separation.bat
```

### Example 3: Custom Cache Location

To use a different drive for cache:

```powershell
# Set environment variable
$env:CACHE_DIR = "D:\ML_Cache"

# Launch
cd E:\styletts2-dataset-toolkit\stem-separation
.\launch_stem_separation.bat
```

### Example 4: Permanent Environment Variables

To set environment variables permanently (Windows):

1. **Via System Properties:**
   - Right-click "This PC" → Properties
   - Advanced system settings → Environment Variables
   - Add new User/System variables

2. **Via PowerShell (User-level):**
   ```powershell
   [System.Environment]::SetEnvironmentVariable("STYLETTS2_PATH", "E:\AI\tts-webui\styletts2", "User")
   [System.Environment]::SetEnvironmentVariable("FFMPEG_PATH", "E:\AI\tools\ffmpeg\bin", "User")
   [System.Environment]::SetEnvironmentVariable("CACHE_DIR", "E:\.cache", "User")
   ```

3. **Via Command Prompt:**
   ```cmd
   setx STYLETTS2_PATH "E:\AI\tts-webui\styletts2"
   setx FFMPEG_PATH "E:\AI\tools\ffmpeg\bin"
   setx CACHE_DIR "E:\.cache"
   ```

**Note:** After setting permanent environment variables, restart your terminal/PowerShell for changes to take effect.

---

## 🔍 Troubleshooting

### "Virtual environment not found"

**Problem:** Launcher can't find the virtual environment.

**Solutions:**
1. Check if `STYLETTS2_PATH` is set correctly
2. Verify the path exists: `Test-Path "C:\path\to\styletts2\.venv"`
3. Run installation script if venv doesn't exist

### "FFmpeg not found"

**Problem:** FFmpeg is not detected.

**Solutions:**
1. Add FFmpeg to system PATH, or
2. Set `FFMPEG_PATH` environment variable, or
3. Install FFmpeg in one of the common locations

### "Web UI script not found"

**Problem:** `styletts2_webui.py` is not found.

**Solutions:**
1. Verify `STYLETTS2_PATH` points to correct directory
2. Check if `styletts2_webui.py` exists in that directory
3. Ensure StyleTTS2 is installed correctly

### Cache filling C: drive

**Problem:** Cache is using C: drive instead of E: drive.

**Solutions:**
1. Set `CACHE_DIR` environment variable to desired location
2. Or ensure E: drive exists (launcher will use it automatically)

---

## 📝 Best Practices

1. **Use Environment Variables** for custom paths instead of editing launcher scripts
2. **Set Permanent Variables** if you always use custom locations
3. **Document Your Setup** - note your custom paths for future reference
4. **Test After Changes** - verify launchers work after changing paths

---

## 🔗 Related Documentation

- [Installation Guide](INSTALLATION.md)
- [Troubleshooting Guide](TROUBLESHOOTING.md)
- [Workflow Guide](WORKFLOW_GUIDE.md)

---

**Last Updated:** 2025-01-27

