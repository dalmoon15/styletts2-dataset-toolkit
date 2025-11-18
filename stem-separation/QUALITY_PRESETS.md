# 🎛️ Stem Separation Quality Presets

## Quality Settings Explained

### Fast
- **Overlap:** 0.1
- **Split:** True (chunked processing)
- **Shifts:** 0 (no augmentation)
- **Speed:** ~15-20 seconds per minute of audio
- **Best for:** Quick testing, previewing results

### Balanced (Default)
- **Overlap:** 0.25
- **Split:** True
- **Shifts:** 0
- **Speed:** ~30-40 seconds per minute of audio
- **Best for:** General purpose separation

### High Quality
- **Overlap:** 0.5
- **Split:** True
- **Shifts:** 1 (test-time augmentation)
- **Speed:** ~60-90 seconds per minute of audio
- **Best for:** High-quality vocal isolation

### Maximum (Slow) ⭐ RECOMMENDED FOR VOICE CLONING
- **Overlap:** 0.75 (maximum overlap between processing windows)
- **Split:** False (processes entire file at once)
- **Shifts:** 2 (multiple test-time augmentations)
- **Speed:** ~2-3 minutes per minute of audio
- **Best for:** Maximum quality vocal extraction for voice cloning datasets

### Ultra (Research Grade) 🔥 ULTIMATE QUALITY
- **Overlap:** 0.99 (99% overlap - research-grade quality)
- **Split:** True (handles any file length)
- **Shifts:** 5 (five augmentation passes)
- **Speed:** ~6-10 minutes per minute of audio
- **VRAM:** Maximizes RTX 3060 12GB
- **Best for:** Eliminating stubborn music bleed, archival-quality vocal isolation
- **Note:** Available via `batch_separate.py` command-line script

## Parameter Details

### Overlap
Controls how much processing windows overlap. Higher values mean better continuity and fewer artifacts at boundaries.
- 0.1 = minimal overlap (fast, possible artifacts)
- 0.75 = maximum overlap (slow, cleanest results)

### Split
- **True:** Processes audio in chunks (memory efficient, slight quality trade-off)
- **False:** Processes entire file at once (uses more VRAM, best quality)

### Shifts
Test-time augmentation that processes the audio multiple times with random shifts and averages results.
- 0 = no augmentation (fast)
- 1 = one augmentation pass (~2x slower, better)
- 2 = two augmentation passes (~3x slower, best quality)

## Recommendations by Use Case

### Voice Cloning Dataset Preparation
**Use:** Maximum (Slow)
- Clean vocals are critical for training quality
- Processing time is worth the quality improvement
- Reduces artifacts that could affect voice model

### Music Production
**Use:** High Quality or Maximum
- Professional results need high separation quality
- Time investment pays off in final mix

### Quick Preview/Testing
**Use:** Fast
- Check if source material is suitable
- Test different models quickly

### Batch Processing Large Libraries
**Use:** Balanced or High Quality
- Good balance for processing many files
- Maximum may be too slow for hundreds of files

## Performance Impact (RTX 3060, 3-minute song)

| Preset | Processing Time | VRAM Usage | Quality Score | Music Bleed |
|--------|----------------|------------|---------------|-------------|
| Fast | ~45 seconds | 2-3 GB | Good | Moderate |
| Balanced | ~1.5 minutes | 2-3 GB | Very Good | Low |
| High Quality | ~3-4 minutes | 3-4 GB | Excellent | Very Low |
| Maximum (Slow) | ~6-9 minutes | 4-6 GB | Outstanding | Minimal |
| **Ultra (Batch)** | **~18-30 minutes** | **6-9 GB** | **Research-Grade** | **Nearly Zero** |

## Noise Filtering (Post-Processing) 🧹

After stem separation, use the **batch_noise_filter.py** tool to clean up:
- Birds chirping
- Menu sounds / UI beeps
- Outdoor ambient noise
- Low-frequency rumble
- Non-vocal background sounds

### Noise Filtering Parameters

**Reduction Strength** (0.0 - 1.0)
- 0.3-0.5: Gentle cleanup, natural sound
- 0.6-0.8: Aggressive removal for noisy sources
- 1.0: Maximum reduction (may affect voice quality)

**Gate Threshold** (-60 to -20 dB)
- -40 dB: Standard (recommended)
- -30 dB: More aggressive background removal
- -50 dB: Gentler, preserves more ambience

**Highpass Filter** (50-150 Hz)
- 80 Hz: Standard for vocals (default)
- 100-120 Hz: Aggressive rumble removal
- 50-70 Hz: Preserve deep male voices

### Two-Stage Workflow

1. **First:** Stem separation (eliminates music)
   - Use Maximum or Ultra quality for best results
   - Focus: Remove instrumental content

2. **Then:** Noise filtering (removes ambient sounds)
   - Run `batch_noise_filter.py` on separated vocals
   - Focus: Clean up environmental noise

**Usage Example:**
```powershell
# Stage 1: Separate vocals with Ultra quality
python batch_separate.py --input "C:\audio\songs" --model htdemucs_ft

# Stage 2: Apply noise filtering
python batch_noise_filter.py --input "C:\audio\songs\vocals_only" --strength 0.5
```

## Tips for Best Results

1. **Use Ultra quality for voice cloning** - eliminates stubborn music bleed (batch script only)
2. **Start with Fast preset** to test if a song will separate well before committing to Ultra
3. **Apply noise filtering** if vocals have birds, menu sounds, or ambient noise
4. **Monitor VRAM** - Ultra quality maximizes 12GB but won't overflow
5. **Two-stage approach**: First separate (Ultra) → Then filter noise (batch_noise_filter.py)
6. **WAV format** preserves maximum quality (no compression artifacts)
7. **Source quality matters** - clean recordings separate better than compressed/low-quality audio

## Technical Background

The quality improvements come from:
- **Higher overlap** reduces boundary artifacts where processing windows meet
- **No splitting** allows the model to see the entire context, improving coherence
- **Shifts augmentation** creates ensemble predictions that average out errors

These settings essentially trade compute time for separation quality by making the model work harder and smarter on your audio.
