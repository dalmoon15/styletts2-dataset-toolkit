# Fine-Tuned Model Deployment Guide

Complete guide for using your trained StyleTTS2 model in production.

---

## 🎯 Overview

After training your model for 50 epochs, you have three options for deployment:

1. **Dedicated Web UI** (Recommended) - Simple Gradio interface
2. **Interactive CLI** - Terminal-based generation
3. **Integrate into Main WebUI** - Add as separate tab

This guide covers all three approaches.

---

## 🚀 Option 1: Dedicated Web UI (Recommended)

### Quick Start

1. **Find Your Best Checkpoint**
   ```batch
   cd styletts2-setup
   run_batch_inference_sampled.bat
   python analyze_inference_results.py
   ```
   → Note the best epoch number (e.g., epoch 25)

2. **Configure Finetuned WebUI**
   
   Edit `finetuned_webui.py` line 32:
   ```python
   BEST_EPOCH = 25  # ← Change to your best epoch
   ```

3. **Launch WebUI**
   ```batch
   launch_finetuned_webui.bat
   ```
   → Opens at http://127.0.0.1:7861

### Features

- ✅ Simple, focused interface for your trained voice
- ✅ Adjustable quality and emotion settings
- ✅ Seed control for reproducible outputs
- ✅ Automatic file saving with timestamps
- ✅ VRAM auto-cleanup after generation
- ✅ No voice cloning needed (model is already trained)

### Settings Guide

**Quality (Diffusion Steps):**
- 3-5: Fast, good quality (recommended for testing)
- 10: Balanced
- 15-20: Highest quality (slower)

**Emotion Scale:**
- 0.8-1.0: Neutral delivery
- 1.2-1.5: More expressive
- 1.6-2.0: Very emotional (may sound exaggerated)

**Seed:**
- -1: Random variation each time
- 42, 123, etc.: Reproducible (same text + seed = identical output)

---

## 🖥️ Option 2: Interactive CLI

### Quick Start

1. **Configure Script**
   
   Edit `inference_single_checkpoint.py` line 30:
   ```python
   BEST_EPOCH = 25  # Change to your best epoch
   ```

2. **Launch**
   ```batch
   run_interactive_inference.bat
   ```

### Usage

```
Enter text: Hello, this is my custom voice!
Generating... (steps=5, scale=1.0)
✓ Generated 3.2s audio
✓ Saved: generated_001.wav

Enter text: settings
Current settings:
  Diffusion steps: 5
  Embedding scale: 1.0
Diffusion steps [5]: 10
Embedding scale [1.0]: 1.5
✓ Updated

Enter text: Another sentence with new settings
...

Enter text: quit
Goodbye!
```

### Commands

- Type text → generates audio
- `settings` → adjust quality/emotion
- `quit` or `exit` → close

---

## 🔗 Option 3: Integrate into Main WebUI

### Overview

Add your fine-tuned model as a new tab in the existing StyleTTS2 WebUI, alongside the pretrained model.

### Benefits

- Switch between pretrained and fine-tuned models
- Compare outputs side-by-side
- Keep all functionality in one place

### Implementation Steps

1. **Backup Main WebUI**
   ```batch
   copy styletts2_webui.py styletts2_webui.backup.py
   ```

2. **Add Fine-Tuned Tab**
   
   Edit `styletts2_webui.py`:
   
   **Step A: Add imports at top**
   ```python
   # After existing imports, add:
   FINETUNED_MODEL = None
   FINETUNED_SAMPLER = None
   ```

   **Step B: Add load function**
   ```python
   def load_finetuned_model(epoch_num):
       """Load fine-tuned checkpoint"""
       global FINETUNED_MODEL, FINETUNED_SAMPLER
       
       checkpoint_path = Path(f"StyleTTS2/Models/LJSpeech/epoch_2nd_{epoch_num:05d}.pth")
       config_path = Path("StyleTTS2/Models/LJSpeech/config_ft.yml")
       
       # ... (copy logic from finetuned_webui.py load_models function)
       
       return f"✅ Loaded epoch {epoch_num}"
   ```

   **Step C: Add generation function**
   ```python
   def generate_finetuned(text, diffusion_steps, embedding_scale):
       """Generate with fine-tuned model"""
       global FINETUNED_MODEL, FINETUNED_SAMPLER
       
       if FINETUNED_MODEL is None:
           return None, "❌ Fine-tuned model not loaded"
       
       # ... (copy logic from finetuned_webui.py generate_audio function)
   ```

   **Step D: Add tab to UI**
   ```python
   # Inside gr.Blocks() with statement, after existing tabs:
   
   with gr.Tab("🎯 Fine-Tuned Model"):
       gr.Markdown("### Your Trained Voice")
       
       with gr.Row():
           epoch_selector = gr.Number(value=25, label="Epoch", precision=0)
           load_ft_btn = gr.Button("Load Model")
       
       ft_status = gr.Textbox(label="Status", interactive=False)
       
       with gr.Row():
           ft_text = gr.Textbox(label="Text", lines=5)
           ft_steps = gr.Slider(3, 20, 5, label="Quality")
           ft_scale = gr.Slider(0.5, 2.0, 1.0, label="Emotion")
       
       ft_generate_btn = gr.Button("Generate")
       ft_audio_out = gr.Audio(label="Output")
       
       # Wire up events
       load_ft_btn.click(
           fn=load_finetuned_model,
           inputs=[epoch_selector],
           outputs=[ft_status]
       )
       
       ft_generate_btn.click(
           fn=generate_finetuned,
           inputs=[ft_text, ft_steps, ft_scale],
           outputs=[ft_audio_out, ft_status]
       )
   ```

3. **Test Integration**
   ```batch
   launch_styletts2.bat
   ```
   → New "Fine-Tuned Model" tab should appear

### Notes

- Dedicated UI (Option 1) is simpler and less error-prone
- Integration (Option 3) is more advanced but provides unified interface
- You can use both approaches simultaneously

---

## 📊 Comparing Models

### Pretrained vs Fine-Tuned

| Feature | Pretrained | Fine-Tuned |
|---------|-----------|------------|
| **Voice** | Any (via reference audio) | Your specific speaker |
| **Quality** | Good, generic | Excellent, speaker-specific |
| **Setup** | Pip install | Train 50 epochs |
| **Speed** | Fast | Fast (same architecture) |
| **Use Case** | Voice cloning experiments | Production custom voice |

### A/B Testing Workflow

1. Generate with pretrained model
2. Generate with fine-tuned model (same text)
3. Compare:
   - Voice accuracy
   - Prosody naturalness
   - Pronunciation clarity
   - Emotional expressiveness

---

## 🔧 Troubleshooting

### "Model not loaded"
- Check BEST_EPOCH matches your checkpoint file name
- Verify checkpoint exists in `StyleTTS2/Models/LJSpeech/`
- Check console for load errors

### "No module named 'models'"
- Ensure StyleTTS2/ folder exists
- Check sys.path includes StyleTTS2 directory
- Run from correct directory (styletts2-setup/)

### CUDA Out of Memory
- Close other GPU applications
- Reduce diffusion_steps to 5
- Use CPU mode: change DEVICE to 'cpu' in script

### Poor Audio Quality
- Try different epochs (early vs late training)
- Adjust diffusion_steps (higher = better)
- Check if training converged properly
- Validate dataset quality was good

### Robotic/Unnatural Voice
- Lower embedding_scale (try 0.8-1.0)
- Check training didn't overfit
- Ensure dataset had prosody variety
- Try earlier checkpoints (20-30 instead of 45-49)

---

## 💡 Production Tips

### Performance Optimization

1. **Batch Generation**
   - Generate multiple texts in sequence
   - Model stays loaded (faster)

2. **GPU Utilization**
   - Use diffusion_steps=5 for realtime
   - Monitor VRAM usage
   - Clear cache between long sessions

3. **File Management**
   - Generated files go to `generated_audio/`
   - Auto-timestamped filenames
   - Clean up old files periodically

### Quality vs Speed

**Realtime (RTF < 1.0):**
- diffusion_steps=5
- embedding_scale=1.0
- Expected RTF: 0.3-0.5

**High Quality (Non-realtime):**
- diffusion_steps=15-20
- embedding_scale=1.0-1.5
- Expected RTF: 1.0-2.0

### Reproducibility

Use seeds for consistent outputs:
```python
# Same text + same seed = identical audio
seed = 42
```

Useful for:
- Creating sample libraries
- A/B testing changes
- Debugging generation issues

---

## 📝 Next Steps

1. ✅ Find best checkpoint with batch inference
2. ✅ Configure finetuned_webui.py with best epoch
3. ✅ Launch and test generation
4. ✅ Compare with pretrained model
5. ✅ Deploy to your application!

---

**Congratulations on training your custom voice model!** 🎊
