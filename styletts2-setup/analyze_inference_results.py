"""
Analyze and compare batch inference results across epochs
Generates comparison reports and identifies best checkpoints
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def analyze_results(results_dir):
    """Analyze inference results from a batch run"""
    results_dir = Path(results_dir)
    csv_path = results_dir / "inference_results.csv"
    
    if not csv_path.exists():
        print(f"❌ No results found at {csv_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"ANALYZING RESULTS: {results_dir.name}")
    print(f"{'='*70}\n")
    
    # Load results
    df = pd.read_csv(csv_path)
    
    # Basic statistics
    print("📊 **OVERALL STATISTICS**\n")
    print(f"  Total samples: {len(df)}")
    print(f"  Epochs tested: {df['epoch'].nunique()}")
    print(f"  Sentences per epoch: {len(df) // df['epoch'].nunique()}")
    
    if 'rtf' in df.columns:
        print(f"\n  RTF Statistics:")
        print(f"    Mean: {df['rtf'].mean():.4f}")
        print(f"    Std:  {df['rtf'].std():.4f}")
        print(f"    Min:  {df['rtf'].min():.4f}")
        print(f"    Max:  {df['rtf'].max():.4f}")
        
        total_audio = df['audio_duration'].sum()
        total_inference = df['inference_time'].sum()
        print(f"\n  Total audio generated: {total_audio:.2f}s ({total_audio/60:.2f} min)")
        print(f"  Total inference time: {total_inference:.2f}s ({total_inference/60:.2f} min)")
        print(f"  Overall speedup: {total_audio/total_inference:.2f}x realtime")
    
    # Per-epoch analysis
    print(f"\n{'─'*70}")
    print("📈 **PER-EPOCH RTF ANALYSIS**\n")
    
    epoch_stats = df.groupby('epoch').agg({
        'rtf': ['mean', 'std', 'min', 'max'],
        'audio_duration': 'sum',
        'inference_time': 'sum'
    }).round(4)
    
    print(epoch_stats.to_string())
    
    # Find best epochs
    print(f"\n{'─'*70}")
    print("🏆 **TOP 5 FASTEST EPOCHS (by mean RTF)**\n")
    
    best_epochs = df.groupby('epoch')['rtf'].mean().sort_values().head(5)
    for idx, (epoch, rtf) in enumerate(best_epochs.items(), 1):
        print(f"  {idx}. Epoch {int(epoch):05d} - RTF: {rtf:.4f}")
    
    # Per-sentence analysis
    print(f"\n{'─'*70}")
    print("📝 **PER-SENTENCE RTF ANALYSIS**\n")
    
    sentence_stats = df.groupby('sentence').agg({
        'rtf': ['mean', 'std', 'min', 'max'],
        'audio_duration': 'mean'
    }).round(4)
    
    print(sentence_stats.to_string())
    
    # Identify potential issues
    print(f"\n{'─'*70}")
    print("⚠️  **POTENTIAL ISSUES**\n")
    
    if 'error' in df.columns:
        errors = df[df['error'].notna()]
        if len(errors) > 0:
            print(f"  Found {len(errors)} errors:")
            for _, row in errors.iterrows():
                print(f"    Epoch {int(row['epoch']):05d}, {row['sentence']}: {row['error']}")
        else:
            print("  ✓ No errors detected")
    
    # RTF outliers
    if 'rtf' in df.columns:
        rtf_mean = df['rtf'].mean()
        rtf_std = df['rtf'].std()
        outliers = df[np.abs(df['rtf'] - rtf_mean) > 2 * rtf_std]
        
        if len(outliers) > 0:
            print(f"\n  RTF outliers (>2σ from mean):")
            for _, row in outliers.iterrows():
                print(f"    Epoch {int(row['epoch']):05d}, {row['sentence']}: RTF={row['rtf']:.4f}")
        else:
            print("\n  ✓ No significant RTF outliers")
    
    # Generate plots
    print(f"\n{'─'*70}")
    print("📊 **GENERATING PLOTS**\n")
    
    plot_dir = results_dir / "analysis_plots"
    plot_dir.mkdir(exist_ok=True)
    
    try:
        # Plot 1: RTF over epochs
        fig, ax = plt.subplots(figsize=(12, 6))
        epoch_rtf = df.groupby('epoch')['rtf'].mean()
        ax.plot(epoch_rtf.index, epoch_rtf.values, marker='o', linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean RTF', fontsize=12)
        ax.set_title('Real-Time Factor vs Training Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / 'rtf_vs_epoch.png', dpi=150)
        plt.close()
        print("  ✓ Saved: rtf_vs_epoch.png")
        
        # Plot 2: RTF by sentence type
        fig, ax = plt.subplots(figsize=(10, 6))
        sentence_rtf = df.groupby('sentence')['rtf'].mean().sort_values()
        ax.barh(range(len(sentence_rtf)), sentence_rtf.values)
        ax.set_yticks(range(len(sentence_rtf)))
        ax.set_yticklabels(sentence_rtf.index)
        ax.set_xlabel('Mean RTF', fontsize=12)
        ax.set_title('RTF by Sentence Type', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / 'rtf_by_sentence.png', dpi=150)
        plt.close()
        print("  ✓ Saved: rtf_by_sentence.png")
        
        # Plot 3: RTF distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df['rtf'], bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(df['rtf'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["rtf"].mean():.4f}')
        ax.set_xlabel('RTF', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('RTF Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_dir / 'rtf_distribution.png', dpi=150)
        plt.close()
        print("  ✓ Saved: rtf_distribution.png")
        
        print(f"\n  All plots saved to: {plot_dir}")
        
    except Exception as e:
        print(f"  ⚠️  Error generating plots: {e}")
    
    # Generate recommendations
    print(f"\n{'='*70}")
    print("💡 **RECOMMENDATIONS**\n")
    
    best_epoch = int(df.groupby('epoch')['rtf'].mean().idxmin())
    best_rtf = df.groupby('epoch')['rtf'].mean().min()
    
    print(f"  🎯 Best checkpoint: epoch_2nd_{best_epoch:05d}.pth (RTF: {best_rtf:.4f})")
    
    # Check if later epochs are better
    epochs = sorted(df['epoch'].unique())
    if len(epochs) >= 3:
        early_rtf = df[df['epoch'].isin(epochs[:len(epochs)//3])]['rtf'].mean()
        late_rtf = df[df['epoch'].isin(epochs[-len(epochs)//3:])]['rtf'].mean()
        
        if late_rtf < early_rtf:
            improvement = ((early_rtf - late_rtf) / early_rtf) * 100
            print(f"\n  📈 Training improved RTF by {improvement:.1f}% from early to late epochs")
        else:
            print(f"\n  ⚠️  Later epochs show slower inference - consider early stopping")
    
    print(f"\n{'='*70}\n")
    
    return df


def compare_multiple_runs(results_base_dir):
    """Compare multiple batch inference runs"""
    results_base_dir = Path(results_base_dir)
    
    batch_dirs = sorted([d for d in results_base_dir.iterdir() if d.is_dir() and d.name.startswith('batch_')])
    
    if len(batch_dirs) == 0:
        print(f"No batch result directories found in {results_base_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"COMPARING {len(batch_dirs)} BATCH RUNS")
    print(f"{'='*70}\n")
    
    for batch_dir in batch_dirs:
        csv_path = batch_dir / "inference_results.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            mean_rtf = df['rtf'].mean() if 'rtf' in df.columns else None
            print(f"  {batch_dir.name}")
            print(f"    Epochs: {df['epoch'].min()}-{df['epoch'].max()}")
            print(f"    Mean RTF: {mean_rtf:.4f}" if mean_rtf else "    No RTF data")
            print()


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze batch inference results')
    parser.add_argument('results_dir', nargs='?', help='Path to batch results directory')
    parser.add_argument('--compare-all', action='store_true', help='Compare all batch runs')
    
    args = parser.parse_args()
    
    # Determine output base directory (relative to script location)
    script_dir = Path(__file__).parent
    output_base = script_dir / "inference_outputs"
    
    if args.compare_all:
        compare_multiple_runs(output_base)
    elif args.results_dir:
        analyze_results(args.results_dir)
    else:
        # Find most recent batch
        if not output_base.exists():
            print(f"❌ No inference outputs directory found at {output_base}")
            return
        
        batch_dirs = sorted(
            [d for d in output_base.iterdir() if d.is_dir() and d.name.startswith('batch_')],
            key=lambda x: x.name,
            reverse=True
        )
        
        if len(batch_dirs) == 0:
            print(f"❌ No batch results found in {output_base}")
            return
        
        print(f"Analyzing most recent batch: {batch_dirs[0].name}\n")
        analyze_results(batch_dirs[0])


if __name__ == "__main__":
    main()
