"""
Batch Noise Filtering Tool
Apply noise reduction to vocal files to remove birds, menu sounds, ambient noise, etc.
"""

import os
import sys
import torch
import torchaudio
from pathlib import Path
import time
import numpy as np
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Batch noise filtering tool for cleaning vocal recordings"
)
parser.add_argument(
    '--input',
    type=str,
    required=True,
    help='Input folder path containing WAV files to process'
)
parser.add_argument(
    '--output',
    type=str,
    required=False,
    help='Output folder path (default: input_folder_filtered)'
)
parser.add_argument(
    '--strength',
    type=float,
    default=0.5,
    help='Noise reduction strength 0.0-1.0 (default: 0.5)'
)
parser.add_argument(
    '--threshold',
    type=int,
    default=-40,
    help='Spectral gate threshold in dB (default: -40)'
)
parser.add_argument(
    '--highpass',
    type=int,
    default=80,
    help='Highpass filter cutoff frequency in Hz (default: 80)'
)
args = parser.parse_args()

# Configuration
INPUT_FOLDER = args.input
OUTPUT_FOLDER = args.output if args.output else str(Path(INPUT_FOLDER).parent / (Path(INPUT_FOLDER).name + "_filtered"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Noise reduction settings
NOISE_REDUCE_STRENGTH = args.strength
SPECTRAL_GATE_THRESHOLD = args.threshold
HIGHPASS_CUTOFF = args.highpass

print("="*70)
print("🎤 Batch Vocal Noise Filtering")
print("="*70)
print(f"Device: {DEVICE}")
print(f"Input: {INPUT_FOLDER}")
print(f"Output: {OUTPUT_FOLDER}")
print(f"Noise Reduction Strength: {NOISE_REDUCE_STRENGTH}")
print(f"Spectral Gate Threshold: {SPECTRAL_GATE_THRESHOLD} dB")
print(f"Highpass Filter Cutoff: {HIGHPASS_CUTOFF} Hz")
print("="*70 + "\n")

def apply_spectral_gate(audio, threshold_db=-40, sr=44100):
    """
    Apply spectral gating to remove low-level background noise
    Works by zeroing out frequency bins below threshold
    """
    # Convert to numpy for processing
    audio_np = audio.numpy()
    
    # Compute STFT
    n_fft = 2048
    hop_length = 512
    window = torch.hann_window(n_fft).numpy()
    
    # Process each channel
    filtered = np.zeros_like(audio_np)
    for ch in range(audio_np.shape[0]):
        # STFT
        stft = np.abs(np.array([
            np.fft.rfft(audio_np[ch, i:i+n_fft] * window)
            for i in range(0, len(audio_np[ch]) - n_fft, hop_length)
        ]))
        
        # Convert to dB
        stft_db = 20 * np.log10(stft + 1e-10)
        
        # Create mask (1 for keep, 0 for remove)
        mask = (stft_db > threshold_db).astype(float)
        
        # Apply mask with smooth edges
        from scipy.ndimage import gaussian_filter
        mask = gaussian_filter(mask, sigma=1.0)
        
        # Reconstruct with mask
        phase = np.angle(np.array([
            np.fft.rfft(audio_np[ch, i:i+n_fft] * window)
            for i in range(0, len(audio_np[ch]) - n_fft, hop_length)
        ]))
        
        stft_filtered = stft * mask * np.exp(1j * phase)
        
        # ISTFT (overlap-add)
        output = np.zeros(len(audio_np[ch]))
        for i, frame_idx in enumerate(range(0, len(audio_np[ch]) - n_fft, hop_length)):
            frame = np.fft.irfft(stft_filtered[i])
            output[frame_idx:frame_idx+n_fft] += frame * window
        
        filtered[ch] = output
    
    return torch.from_numpy(filtered).float()


def apply_highpass_filter(audio, cutoff_freq=80, sr=44100):
    """
    Apply highpass filter to remove low-frequency rumble/noise
    Human voice starts around 80-100 Hz
    """
    # Simple first-order highpass filter
    from scipy import signal
    nyquist = sr / 2
    normalized_cutoff = cutoff_freq / nyquist
    
    b, a = signal.butter(4, normalized_cutoff, btype='high')
    
    audio_np = audio.numpy()
    filtered = np.zeros_like(audio_np)
    
    for ch in range(audio_np.shape[0]):
        filtered[ch] = signal.filtfilt(b, a, audio_np[ch])
    
    return torch.from_numpy(filtered).float()


def reduce_noise(audio, sr=44100, strength=0.5, threshold=-40, highpass_cutoff=80):
    """
    Multi-stage noise reduction:
    1. Highpass filter (remove low rumble)
    2. Spectral gating (remove background noise)
    3. Gentle compression (even out levels)
    """
    # Stage 1: Highpass filter
    audio = apply_highpass_filter(audio, cutoff_freq=highpass_cutoff, sr=sr)
    
    # Stage 2: Spectral gate
    audio = apply_spectral_gate(audio, threshold_db=threshold, sr=sr)
    
    # Stage 3: Normalize
    max_val = torch.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95  # Leave headroom
    
    return audio


def process_batch():
    """Process all WAV files in the input folder"""
    
    # Create output directory
    Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)
    
    # Find all WAV files
    input_path = Path(INPUT_FOLDER)
    wav_files = list(input_path.glob("*.wav"))
    
    if not wav_files:
        print("❌ No WAV files found!")
        return
    
    print(f"📁 Found {len(wav_files)} WAV files to process\n")
    
    # Check dependencies
    try:
        import scipy  # noqa: F401
    except ImportError:
        print("❌ scipy not installed. Installing...")
        os.system(f"{sys.executable} -m pip install scipy")
    
    successful = 0
    failed = []
    start_time = time.time()
    
    for idx, audio_file in enumerate(wav_files, 1):
        try:
            progress_pct = (idx-1) / len(wav_files) * 100
            print(f"[{idx}/{len(wav_files)} - {progress_pct:.0f}%] Processing: {audio_file.name}")
            file_start = time.time()
            
            # Load audio
            audio, sr = torchaudio.load(str(audio_file))
            
            print(f"   🎵 Sample rate: {sr} Hz, Duration: {audio.shape[1]/sr:.1f}s")
            
            # Apply noise reduction
            print("   🧹 Applying noise filtering...")
            filtered = reduce_noise(
                audio, 
                sr=sr, 
                strength=NOISE_REDUCE_STRENGTH,
                threshold=SPECTRAL_GATE_THRESHOLD,
                highpass_cutoff=HIGHPASS_CUTOFF
            )
            
            # Save filtered audio
            output_file = Path(OUTPUT_FOLDER) / audio_file.name
            torchaudio.save(str(output_file), filtered, sr)
            
            elapsed = time.time() - file_start
            print(f"   ✅ Saved: {output_file.name} ({elapsed:.1f}s)\n")
            
            successful += 1
            
        except Exception as e:
            print(f"   ❌ Failed: {e}\n")
            import traceback
            traceback.print_exc()
            failed.append(audio_file.name)
            continue
    
    # Summary
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "="*70)
    print("🎉 Noise Filtering Complete!")
    print("="*70)
    print(f"✅ Successful: {successful}/{len(wav_files)}")
    if failed:
        print(f"❌ Failed: {len(failed)}")
        for name in failed[:10]:
            print(f"   • {name}")
        if len(failed) > 10:
            print(f"   ... and {len(failed)-10} more")
    print(f"⏱️  Total time: {minutes}m {seconds}s")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print("="*70)


if __name__ == "__main__":
    try:
        process_batch()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
