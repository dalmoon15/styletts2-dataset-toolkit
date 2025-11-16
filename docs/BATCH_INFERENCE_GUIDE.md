# StyleTTS2 Batch Inference System

Complete system for testing all checkpoint epochs and identifying your best model.

---

## 📦 What's Included

### Core Scripts
- **`batch_inference_epochs.py`** - Main inference engine that tests all checkpoints
- **`analyze_inference_results.py`** - Statistical analysis and plot generation
- **`inference_single_checkpoint.py`** - Interactive CLI for single checkpoint testing

### Launcher Scripts
- **`run_batch_inference.bat`** - Test all 50 epochs (~1-2 hours)
- **`run_batch_inference_sampled.bat`** - Quick test every 5th epoch (~15-20 min)
- **`run_interactive_inference.bat`** - Interactive generation with your best checkpoint

---

## 🚀 Quick Start

### Option 1: Quick Sampled Test (Recommended First)
```batch
cd styletts2-setup
run_batch_inference_sampled.bat
```
Tests epochs: 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49  
**Time:** ~15-20 minutes  
**Output:** ~70 audio samples

### Option 2: Full Test (All Epochs)
```batch
cd styletts2-setup
run_batch_inference.bat
```
Tests all 50 epochs  
**Time:** 1-2 hours  
**Output:** 350 audio samples (7 sentences × 50 epochs)

### Option 3: Custom Range
```batch
python batch_inference_epochs.py --start-epoch 20 --end-epoch 30
```

---

## 📊 Understanding Results

### Output Structure
```
inference_outputs/
└── batch_YYYYMMDD_HHMMSS/
    ├── inference_results.csv      # Performance metrics
    ├── epoch_00000/
    │   ├── neutral_short.wav
    │   ├── neutral_medium.wav
    │   ├── emotional_happy.wav
    │   ├── emotional_sad.wav
    │   ├── technical.wav
    │   ├── conversational.wav
    │   └── question.wav
    ├── epoch_00001/
    │   └── ... (same 7 files)
    └── ... (50 epoch folders total)
```

### Test Sentences (7 per epoch)
1. **neutral_short** - "Hello, this is a test of the fine-tuned model."
2. **neutral_medium** - "The quick brown fox jumps over the lazy dog near the riverbank."
3. **emotional_happy** - "I'm absolutely thrilled to announce we've achieved incredible results!"
4. **emotional_sad** - "Unfortunately, we must face the difficult truth that things didn't work out."
5. **technical** - "The neural network architecture utilizes attention mechanisms for improved performance."
6. **conversational** - "So anyway, I was thinking we could grab some coffee later if you're free?"
7. **question** - "Have you ever wondered what lies beyond the stars in the vast universe?"

---

## 📈 Analyzing Results

### Auto-analyze Most Recent Batch
```batch
python analyze_inference_results.py
```

### Analyze Specific Batch
```batch
python analyze_inference_results.py inference_outputs\batch_20241115_220000
```

### Compare All Batches
```batch
python analyze_inference_results.py --compare-all
```

### What You Get
1. **Overall Statistics** - Total samples, average RTF
2. **Per-Epoch Analysis** - Performance breakdown by epoch
3. **Top 5 Best Epochs** - Fastest inference times
4. **Per-Sentence Analysis** - Which sentences are hardest
5. **Plots** (saved to `analysis_plots/`)
   - RTF vs Epoch
   - RTF by Sentence Type
   - RTF Distribution
6. **Recommendations** - Which checkpoint to use for production

---

## 🎯 Understanding RTF (Real-Time Factor)

**RTF = Inference Time / Audio Duration**

- **RTF < 1.0** = Faster than realtime ✅ (e.g., 0.5 = 2x faster)
- **RTF = 1.0** = Realtime speed
- **RTF > 1.0** = Slower than realtime ⚠️

**Example:** RTF of 0.3 means generating 10 seconds of audio takes 3 seconds

---

## 🎨 Production Usage

### After Finding Your Best Checkpoint

Edit `inference_single_checkpoint.py`:
```python
BEST_EPOCH = 25  # Change to your best epoch number
```

Then launch interactive generation:
```batch
run_interactive_inference.bat
```

Type text to generate speech in real-time!

---

## ⚙️ Advanced Options

### Custom Diffusion Steps
```batch
python batch_inference_epochs.py --diffusion-steps 10
```
- 5 = fast (default)
- 10 = better quality
- 20 = highest quality (slower)

### Enhanced Emotion
```batch
python batch_inference_epochs.py --embedding-scale 2.0
```
- 1.0 = neutral (default)
- 1.5 = more expressive
- 2.0 = very emotional

### Sample Every N Epochs
```batch
python batch_inference_epochs.py --sample-every 10
```

---

## 🔧 Troubleshooting

### "No module named 'models'"
- Ensure StyleTTS2/ folder exists in styletts2-setup/
- Check you're running from the correct directory

### CUDA Out of Memory
- Close other GPU applications
- Reduce batch size in config
- Try one epoch at a time:
  ```batch
  python batch_inference_epochs.py --start-epoch 0 --end-epoch 0
  ```

### Missing Dependencies
```batch
pip install pandas matplotlib torch torchaudio phonemizer nltk librosa
```

### espeak-ng Not Found
- Windows: Download from https://github.com/espeak-ng/espeak-ng/releases
- Add to PATH or set PHONEMIZER_ESPEAK_LIBRARY environment variable

---

## 💡 Best Practices

### 1. Start with Sampled Test
Run sampled test first to verify setup before committing to full test.

### 2. Listen to Early/Middle/Late
Compare epochs 5, 25, 45 to see training progression.

### 3. RTF ≠ Quality
Fastest checkpoint isn't always the best. Listen to audio quality!

### 4. Check for Overfitting
If later epochs sound worse, you may have overtrained.

### 5. Use Interactive Mode
Once you know your best epoch, use `inference_single_checkpoint.py` for production.

---

## 📊 Expected Performance

**RTX 3060 12GB:**
- Per-epoch time: ~1-2 minutes (7 sentences)
- Full 50 epochs: ~60-90 minutes
- Sampled (10 epochs): ~15-20 minutes
- Expected RTF: 0.3-0.8 (faster than realtime)

**CPU (no GPU):**
- Per-epoch time: ~10-15 minutes
- Full run: 8-12 hours
- Use sampled mode!

---

## 🎊 What's Next?

1. ✅ Run sampled inference
2. ✅ Listen to outputs
3. ✅ Analyze results
4. ✅ Find best checkpoint
5. ✅ Configure `inference_single_checkpoint.py`
6. ✅ Generate speech for your application!

---

**Happy inferencing! 🎤✨**
