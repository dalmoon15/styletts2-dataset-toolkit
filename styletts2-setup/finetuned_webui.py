"""
Simple Gradio Web UI for Fine-Tuned StyleTTS2 Model
Dedicated interface for your trained voice model
"""

import os
import sys
import yaml
import torch
import gradio as gr
import torchaudio
import numpy as np
from pathlib import Path
from datetime import datetime
from nltk.tokenize import word_tokenize
import phonemizer

# Add StyleTTS2 to path (relative to script)
styletts2_path = Path(__file__).parent / "StyleTTS2"
sys.path.insert(0, str(styletts2_path))

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# ============================================
# CONFIGURATION
# ============================================
BEST_EPOCH = 25  # ← CHANGE THIS to your best epoch
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_PATH = SCRIPT_DIR / "StyleTTS2" / "Models" / "LJSpeech" / f"epoch_2nd_{BEST_EPOCH:05d}.pth"
CONFIG_PATH = SCRIPT_DIR / "StyleTTS2" / "Models" / "LJSpeech" / "config_ft.yml"
OUTPUT_DIR = SCRIPT_DIR / "generated_audio"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ============================================

OUTPUT_DIR.mkdir(exist_ok=True)

# Global model
model = None
sampler = None
global_phonemizer = None
textcleaner = None

print("🚀 Initializing Fine-Tuned StyleTTS2...")
print(f"   Checkpoint: epoch {BEST_EPOCH:05d}")
print(f"   Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask

def load_models():
    """Load model and checkpoint once at startup"""
    global model, sampler, global_phonemizer, textcleaner
    
    print("\n📦 Loading model...")
    
    # Setup phonemizer
    global_phonemizer = phonemizer.backend.EspeakBackend(
        language='en-us', 
        preserve_punctuation=True, 
        with_stress=True
    )
    textcleaner = TextCleaner()
    
    # Load config
    config = yaml.safe_load(open(CONFIG_PATH))
    
    # Load supporting models
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    from Utils.PLBERT.util import load_plbert
    BERT_path = config.get('PLBERT_dir', False)
    plbert = load_plbert(BERT_path)
    
    # Build model
    model = build_model(
        recursive_munch(config['model_params']), 
        text_aligner, 
        pitch_extractor, 
        plbert
    )
    
    # Load checkpoint
    print(f"📂 Loading checkpoint: {CHECKPOINT_PATH.name}")
    params_whole = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    params = params_whole['net']
    
    for key in model:
        if key in params:
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                from collections import OrderedDict
                state_dict = params[key]
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:] if k.startswith('module.') else k
                    new_state_dict[name] = v
                model[key].load_state_dict(new_state_dict, strict=False)
    
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(DEVICE) for key in model]
    
    # Setup sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    print("✅ Model loaded successfully!\n")
    return f"✅ Model loaded: Epoch {BEST_EPOCH}"

def generate_audio(text, diffusion_steps, embedding_scale, seed):
    """Generate speech from text"""
    global model, sampler, global_phonemizer, textcleaner
    
    if model is None:
        return None, "❌ Model not loaded. Please restart the app."
    
    if not text or len(text.strip()) == 0:
        return None, "❌ Please enter text to synthesize"
    
    try:
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Process text
        text = text.strip().replace('"', '')
        ps = global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        
        tokens = textcleaner(ps)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(DEVICE).unsqueeze(0)
        
        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(tokens.device)
            text_mask = length_to_mask(input_lengths).to(tokens.device)
            
            t_en = model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
            d_en = model.bert_encoder(bert_dur).transpose(-1, -2)
            
            # Generate style
            noise = torch.randn(1, 1, 256).to(DEVICE)
            ref_noise = torch.randn(1, 256).to(DEVICE)
            
            s_pred = sampler(
                noise,
                embedding=bert_dur[0].unsqueeze(0),
                num_steps=diffusion_steps,
                embedding_scale=embedding_scale,
                features=ref_noise
            ).squeeze(0)
            
            s = s_pred[:, 128:]
            ref = s_pred[:, :128]
            
            d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            
            x, _ = model.predictor.lstm(d)
            duration = model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            
            pred_dur[-1] += 5
            
            pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
                c_frame += int(pred_dur[i].data)
            
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(DEVICE))
            F0_pred, N_pred = model.predictor.F0Ntrain(en, s)
            out = model.decoder(
                (t_en @ pred_aln_trg.unsqueeze(0).to(DEVICE)),
                F0_pred, N_pred, ref.squeeze().unsqueeze(0)
            )
        
        wav = out.squeeze().cpu().numpy()
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"finetuned_{timestamp}.wav"
        output_path = OUTPUT_DIR / filename
        
        torchaudio.save(
            str(output_path),
            torch.from_numpy(wav).unsqueeze(0),
            24000
        )
        
        duration = len(wav) / 24000
        message = f"✅ Generated {duration:.2f}s audio\n📁 Saved: {filename}"
        
        print(message)
        
        # Clear VRAM
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return str(output_path), message
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, error_msg

# Load model at startup
startup_status = load_models()

# Build Gradio UI
with gr.Blocks(title="Fine-Tuned StyleTTS2", theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"""
    # 🎙️ Fine-Tuned StyleTTS2 Voice Model
    
    **Model:** Epoch {BEST_EPOCH:05d}  
    **Speaker:** Custom Fine-tuned Voice  
    **Device:** {DEVICE.upper()}
    
    {startup_status}
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter your text here...",
                lines=5,
                max_lines=10
            )
            
            with gr.Row():
                diffusion_steps = gr.Slider(
                    minimum=3,
                    maximum=20,
                    value=5,
                    step=1,
                    label="Quality (Diffusion Steps)",
                    info="Higher = better quality, slower (5=fast, 10=balanced, 20=best)"
                )
                
                embedding_scale = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Emotion Scale",
                    info="Higher = more expressive (1.0=neutral, 1.5=emotional)"
                )
            
            with gr.Row():
                seed = gr.Number(
                    value=-1,
                    label="Seed",
                    info="-1 for random, >= 0 for reproducible results",
                    precision=0
                )
            
            generate_btn = gr.Button("🎤 Generate Speech", variant="primary", size="lg")
        
        with gr.Column(scale=1):
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath"
            )
            status_output = gr.Textbox(
                label="Status",
                lines=3,
                interactive=False
            )
    
    # Example texts
    gr.Markdown("### 📝 Example Texts")
    with gr.Row():
        gr.Examples(
            examples=[
                ["Hello, this is a test of my fine-tuned voice model."],
                ["The quick brown fox jumps over the lazy dog."],
                ["I'm absolutely thrilled to announce we've achieved incredible results!"],
                ["The neural network architecture utilizes attention mechanisms for improved performance."],
                ["Have you ever wondered what lies beyond the stars in the vast universe?"]
            ],
            inputs=text_input,
            label="Click to try"
        )
    
    # Settings info
    with gr.Accordion("ℹ️ Settings Guide", open=False):
        gr.Markdown("""
        ### Quality (Diffusion Steps)
        - **3-5:** Fast generation, good quality (recommended for testing)
        - **10:** Balanced quality and speed
        - **15-20:** Highest quality, slower
        
        ### Emotion Scale
        - **0.8-1.0:** Neutral, stable delivery
        - **1.2-1.5:** More expressive, emotional
        - **1.6-2.0:** Very expressive (may sound exaggerated)
        
        ### Seed
        - **-1:** Random variation each time
        - **42, 123, etc.:** Same text + same seed = identical output
        
        ### Output Location
        All generated files are saved to: `generated_audio/` (relative to this script)
        """)
    
    # Event handlers
    generate_btn.click(
        fn=generate_audio,
        inputs=[text_input, diffusion_steps, embedding_scale, seed],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port from main webui (7860)
        share=False,
        show_error=True
    )
