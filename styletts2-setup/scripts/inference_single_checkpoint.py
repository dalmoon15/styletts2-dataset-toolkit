"""
Single Checkpoint Inference Script
Use your best epoch checkpoint for production voice generation
"""

import torch
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import sys
import yaml
from pathlib import Path
import torchaudio
from nltk.tokenize import word_tokenize
import phonemizer

# Add StyleTTS2 to path (relative to this script)
styletts2_path = Path(__file__).parent / "StyleTTS2"
sys.path.insert(0, str(styletts2_path))

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# ============================================
# CONFIGURATION - EDIT THESE
# ============================================
BEST_EPOCH = 20  # CHANGE THIS to your best epoch number
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_PATH = SCRIPT_DIR / "StyleTTS2" / "Models" / "LJSpeech" / f"epoch_2nd_{BEST_EPOCH:05d}.pth"
CONFIG_PATH = SCRIPT_DIR / "StyleTTS2" / "Models" / "LJSpeech" / "config_ft.yml"
OUTPUT_DIR = SCRIPT_DIR / "generated_audio"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Generation settings
DIFFUSION_STEPS = 5  # 5=fast, 10=better quality, 20=highest quality
EMBEDDING_SCALE = 1.0  # 1.0=neutral, 1.5=more emotion
# ============================================

# Setup
OUTPUT_DIR.mkdir(exist_ok=True)
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', 
    preserve_punctuation=True, 
    with_stress=True
)
textcleaner = TextCleaner()
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def load_model():
    """Load model and checkpoint"""
    print(f"Loading config from {CONFIG_PATH}...")
    config = yaml.safe_load(open(CONFIG_PATH))
    
    # Load ASR model
    ASR_config = config.get('ASR_config', False)
    ASR_path = config.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config)
    
    # Load F0 model
    F0_path = config.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path)
    
    # Load PLBERT
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
    print(f"Loading checkpoint: {CHECKPOINT_PATH.name}")
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
    
    # Setup diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    print(f"✓ Model loaded successfully on {DEVICE}")
    return model, sampler


def generate_speech(text, model, sampler, output_filename=None, 
                   diffusion_steps=DIFFUSION_STEPS, 
                   embedding_scale=EMBEDDING_SCALE):
    """Generate speech from text"""
    
    # Clean and phonemize text
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
        ref_noise = torch.randn(1, 256).to(DEVICE)  # Reference style for multispeaker
        
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
    
    # Save if filename provided
    if output_filename:
        output_path = OUTPUT_DIR / output_filename
        torchaudio.save(
            str(output_path),
            torch.from_numpy(wav).unsqueeze(0),
            24000
        )
        print(f"✓ Saved: {output_path}")
        return wav, output_path
    
    return wav


def main():
    """Interactive generation"""
    print("\n" + "="*70)
    print("StyleTTS2 Interactive Voice Generation")
    print(f"Using checkpoint: epoch_{BEST_EPOCH:05d}")
    print("="*70 + "\n")
    
    # Load model once
    model, sampler = load_model()
    
    print("\nReady to generate speech!")
    print("Commands:")
    print("  - Type text to generate speech")
    print("  - Type 'quit' or 'exit' to stop")
    print("  - Type 'settings' to adjust generation parameters")
    print()
    
    global DIFFUSION_STEPS, EMBEDDING_SCALE
    counter = 1
    
    while True:
        text = input("Enter text: ").strip()
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if text.lower() == 'settings':
            print(f"\nCurrent settings:")
            print(f"  Diffusion steps: {DIFFUSION_STEPS} (5=fast, 10=balanced, 20=quality)")
            print(f"  Embedding scale: {EMBEDDING_SCALE} (1.0=neutral, 1.5=emotional)")
            
            try:
                new_steps = input(f"Diffusion steps [{DIFFUSION_STEPS}]: ").strip()
                if new_steps:
                    DIFFUSION_STEPS = int(new_steps)
                
                new_scale = input(f"Embedding scale [{EMBEDDING_SCALE}]: ").strip()
                if new_scale:
                    EMBEDDING_SCALE = float(new_scale)
                
                print(f"✓ Updated: diffusion_steps={DIFFUSION_STEPS}, embedding_scale={EMBEDDING_SCALE}\n")
            except ValueError:
                print("Invalid input, keeping previous settings\n")
            continue
        
        if not text:
            continue
        
        try:
            filename = f"generated_{counter:03d}.wav"
            print(f"Generating... (steps={DIFFUSION_STEPS}, scale={EMBEDDING_SCALE})")
            
            wav, path = generate_speech(
                text, model, sampler, 
                output_filename=filename,
                diffusion_steps=DIFFUSION_STEPS,
                embedding_scale=EMBEDDING_SCALE
            )
            
            duration = len(wav) / 24000
            print(f"✓ Generated {duration:.2f}s audio\n")
            counter += 1
            
        except Exception as e:
            print(f"✗ Error: {e}\n")


if __name__ == "__main__":
    main()
