# Changelog

All notable changes to the stem-separation toolkit will be documented in this file.

## [Enhanced] - November 2025

### Added
- **🧹 Noise Reduction Tool** - `batch_noise_filter.py`
  - Batch processing for cleaning background noise from vocals
  - Granular controls for noise reduction strength, gate threshold, highpass filter
  - Command-line interface with customizable parameters
  - Multi-stage filtering: highpass + spectral gate + normalization
  - Removes birds chirping, menu sounds, ambient noise, low-frequency rumble
  
- **🔥 Ultra-Quality Support** - Research-grade separation settings
  - overlap=0.99 (99% context overlap)
  - shifts=5 (five augmentation passes)
  - Maximizes RTX 3060 12GB VRAM
  - Eliminates stubborn music bleed
  - Documented in QUALITY_PRESETS.md
  
- **📦 Enhanced Batch Processing**
  - Command-line arguments for `batch_separate.py`
  - Support for both MP3 and WAV input files
  - Flexible input/output folder specification
  - Model selection via CLI
  
- **🚀 Windows Launcher Scripts**
  - `launch_noise_filter.bat` - Easy noise filtering launcher
  - `launch_batch_separate.bat` - Batch stem separation launcher
  - User-friendly prompts for folder paths
  - Automatic virtual environment activation

### Enhanced
- **Quality Documentation**
  - Added Ultra preset to QUALITY_PRESETS.md
  - Documented noise filtering parameters and workflow
  - Added performance benchmarks with music bleed metrics
  - Two-stage workflow guide (separate → filter)
  
- **Batch Script Improvements**
  - Added command-line argument parsing
  - Better error handling and validation
  - File size reporting
  - Progress tracking with percentages
  - VRAM usage monitoring
  - Comprehensive failure reporting

### Performance
- Ultra quality settings optimized for RTX 3060 12GB
- Memory management improvements
- Cache clearing between files
- Smooth processing for any file length

### Documentation
- Updated QUALITY_PRESETS.md with Ultra settings and noise filtering
- Added usage examples for two-stage workflow
- Enhanced tips section with best practices
- Created comprehensive CHANGELOG.md

## [1.0.0] - Initial Release

### Features
- Demucs v4 stem separation
- UVR model support
- Gradio web interface
- GPU acceleration
- Batch processing capability
- Model caching
- Quality presets (Fast, Balanced, High Quality, Maximum)
- Windows-optimized launcher scripts
