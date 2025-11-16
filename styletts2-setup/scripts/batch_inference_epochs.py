"""
Batch Inference for StyleTTS2 Checkpoints
Tests all epoch checkpoints and generates comparison samples

Note: For multispeaker models fine-tuned on single speaker, 
this script generates a consistent random reference style per epoch
to satisfy the context_features requirement in the diffusion model.
"""

import torch
torch.manual_seed(42)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

import random
random.seed(42)
import numpy as np
np.random.seed(42)

import sys
import time
import yaml
from pathlib import Path
import torchaudio
from nltk.tokenize import word_tokenize
import pandas as pd
from datetime import datetime
import phonemizer

# Add StyleTTS2 to path (relative to this script)
styletts2_path = Path(__file__).parent / "StyleTTS2"
sys.path.insert(0, str(styletts2_path))

from models import *
from utils import *
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

# Configuration - paths relative to script location
SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "StyleTTS2" / "Models" / "LJSpeech"
CONFIG_PATH = CHECKPOINT_DIR / "config_ft.yml"
OUTPUT_BASE = SCRIPT_DIR / "inference_outputs"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Test sentences - diverse phonetics and emotions
TEST_SENTENCES = {
    "neutral_short": "Hello, this is a test of the fine-tuned model.",
    "neutral_medium": "The quick brown fox jumps over the lazy dog near the riverbank.",
    "emotional_happy": "I'm absolutely thrilled to announce we've achieved incredible results!",
    "emotional_sad": "Unfortunately, we must face the difficult truth that things didn't work out.",
    "technical": "The neural network architecture utilizes attention mechanisms for improved performance.",
    "conversational": "So anyway, I was thinking we could grab some coffee later if you're free?",
    "question": "Have you ever wondered what lies beyond the stars in the vast universe?"
}

# Global setup
global_phonemizer = phonemizer.backend.EspeakBackend(
    language='en-us', 
    preserve_punctuation=True, 
    with_stress=True
)
textcleaner = TextCleaner()

# Mel spectrogram setup
to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=80, n_fft=2048, win_length=1200, hop_length=300
)
mean, std = -4, 4


def length_to_mask(lengths):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def load_models_and_config():
    """Load config and initialize models (without checkpoint weights)"""
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
    
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(DEVICE) for key in model]
    
    # Setup diffusion sampler
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False
    )
    
    return model, sampler, config


def load_checkpoint(model, checkpoint_path):
    """Load checkpoint weights into model"""
    print(f"  Loading checkpoint: {checkpoint_path.name}")
    try:
        params_whole = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
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
        return True
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return False


def inference(text, model, sampler, noise, diffusion_steps=5, embedding_scale=1, ref_s=None):
    """Run inference on a single text"""
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
        
        # For multispeaker models, provide reference style features
        # Use provided ref_s or generate from first inference
        s_pred = sampler(
            noise,
            embedding=bert_dur[0].unsqueeze(0),
            num_steps=diffusion_steps,
            embedding_scale=embedding_scale,
            features=ref_s  # Pass reference style for multispeaker models
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
    
    return out.squeeze().cpu().numpy()


def batch_inference_checkpoints(
    start_epoch=0, 
    end_epoch=49, 
    diffusion_steps=5,
    embedding_scale=1.0
):
    """Run inference on all checkpoints"""
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_BASE / f"batch_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"BATCH INFERENCE - Epochs {start_epoch} to {end_epoch}")
    print(f"Output directory: {output_dir}")
    print(f"Diffusion steps: {diffusion_steps}, Embedding scale: {embedding_scale}")
    print(f"{'='*70}\n")
    
    # Load models once
    print("Loading base models...")
    model, sampler, config = load_models_and_config()
    print("✓ Models loaded\n")
    
    # Get checkpoint files
    checkpoint_files = sorted(
        CHECKPOINT_DIR.glob("epoch_2nd_*.pth"),
        key=lambda x: int(x.stem.split('_')[-1])
    )
    checkpoint_files = [
        f for f in checkpoint_files 
        if start_epoch <= int(f.stem.split('_')[-1]) <= end_epoch
    ]
    
    print(f"Found {len(checkpoint_files)} checkpoints to process\n")
    
    # Results tracking
    results = []
    
    # Process each checkpoint
    for ckpt_path in checkpoint_files:
        epoch_num = int(ckpt_path.stem.split('_')[-1])
        print(f"\n{'─'*70}")
        print(f"Processing Epoch {epoch_num:05d}")
        print(f"{'─'*70}")
        
        # Create epoch output directory
        epoch_dir = output_dir / f"epoch_{epoch_num:05d}"
        epoch_dir.mkdir(exist_ok=True)
        
        # Load checkpoint
        if not load_checkpoint(model, ckpt_path):
            print(f"  ⚠ Skipping epoch {epoch_num} due to load error")
            continue
        
        # Generate reference style once per checkpoint (for multispeaker model with single speaker fine-tune)
        # Use consistent random style for reproducibility across samples
        torch.manual_seed(42 + epoch_num)  # Epoch-specific but consistent
        ref_noise = torch.randn(1, 256).to(DEVICE)  # Shape: (batch=1, features=256) - full style vector
        
        # Generate samples for each test sentence
        for sent_name, sent_text in TEST_SENTENCES.items():
            print(f"  Generating: {sent_name}")
            
            try:
                # Run inference
                start_time = time.time()
                noise = torch.randn(1, 1, 256).to(DEVICE)
                wav = inference(
                    sent_text, model, sampler, noise,
                    diffusion_steps=diffusion_steps,
                    embedding_scale=embedding_scale,
                    ref_s=ref_noise  # Pass reference style for multispeaker models
                )
                inference_time = time.time() - start_time
                
                # Calculate RTF (Real-Time Factor)
                audio_duration = len(wav) / 24000
                rtf = inference_time / audio_duration
                
                # Save audio
                output_path = epoch_dir / f"{sent_name}.wav"
                torchaudio.save(
                    str(output_path),
                    torch.from_numpy(wav).unsqueeze(0),
                    24000
                )
                
                # Record results
                results.append({
                    'epoch': epoch_num,
                    'sentence': sent_name,
                    'text': sent_text,
                    'inference_time': inference_time,
                    'audio_duration': audio_duration,
                    'rtf': rtf,
                    'diffusion_steps': diffusion_steps,
                    'embedding_scale': embedding_scale,
                    'output_path': str(output_path.relative_to(output_dir))
                })
                
                print(f"    ✓ RTF: {rtf:.4f}, Duration: {audio_duration:.2f}s")
                
            except Exception as e:
                print(f"    ✗ Error: {e}")
                results.append({
                    'epoch': epoch_num,
                    'sentence': sent_name,
                    'text': sent_text,
                    'error': str(e)
                })
        
        # Memory cleanup
        torch.cuda.empty_cache()
    
    # Save results to CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "inference_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{'='*70}")
    print(f"✓ Batch inference complete!")
    print(f"  Results saved to: {csv_path}")
    print(f"  Audio files in: {output_dir}")
    
    # Print summary statistics
    if 'rtf' in df.columns:
        print(f"\n📊 Summary Statistics:")
        print(f"  Average RTF: {df['rtf'].mean():.4f}")
        print(f"  Min RTF: {df['rtf'].min():.4f} (Epoch {df.loc[df['rtf'].idxmin(), 'epoch']:.0f})")
        print(f"  Max RTF: {df['rtf'].max():.4f} (Epoch {df.loc[df['rtf'].idxmax(), 'epoch']:.0f})")
        print(f"  Total audio generated: {df['audio_duration'].sum():.2f}s")
    
    print(f"{'='*70}\n")
    
    return df, output_dir


def main():
    """Main entry point with configurable parameters"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch inference for StyleTTS2 checkpoints')
    parser.add_argument('--start-epoch', type=int, default=0, help='Starting epoch number')
    parser.add_argument('--end-epoch', type=int, default=49, help='Ending epoch number')
    parser.add_argument('--diffusion-steps', type=int, default=5, help='Number of diffusion steps')
    parser.add_argument('--embedding-scale', type=float, default=1.0, help='Embedding scale for emotion')
    parser.add_argument('--sample-every', type=int, default=1, help='Sample every N epochs (1=all)')
    
    args = parser.parse_args()
    
    # Adjust epoch range if sampling
    if args.sample_every > 1:
        epochs_to_test = range(args.start_epoch, args.end_epoch + 1, args.sample_every)
        print(f"Sampling every {args.sample_every} epochs: {list(epochs_to_test)}")
    
    # Run batch inference
    results_df, output_dir = batch_inference_checkpoints(
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        diffusion_steps=args.diffusion_steps,
        embedding_scale=args.embedding_scale
    )
    
    print(f"✨ All done! Check {output_dir} for results")


if __name__ == "__main__":
    main()
