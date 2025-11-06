"""
Summarize training experiment results.

Reads training logs and generates a comprehensive summary.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml


def summarize_experiment(experiment_dir: str):
    """Summarize a training experiment."""
    exp_path = Path(experiment_dir)
    
    if not exp_path.exists():
        print(f"❌ Experiment directory not found: {experiment_dir}")
        return
    
    print("=" * 70)
    print(f"EXPERIMENT SUMMARY: {exp_path.name}")
    print("=" * 70)
    print()
    
    # Load arguments
    args_file = exp_path / "args.yaml"
    if args_file.exists():
        with open(args_file) as f:
            args = yaml.safe_load(f)
        
        print("📋 **Configuration**:")
        print(f"  Model: {args.get('model', 'N/A')}")
        print(f"  Dataset: {args.get('data', 'N/A')}")
        print(f"  Epochs: {args.get('epochs', 'N/A')}")
        print(f"  Batch size: {args.get('batch', 'N/A')}")
        print(f"  Image size: {args.get('imgsz', 'N/A')}")
        print(f"  Device: {args.get('device', 'N/A')}")
        print(f"  Optimizer: {args.get('optimizer', 'N/A')}")
        print(f"  Learning rate: {args.get('lr0', 'N/A')}")
        print()
    
    # Load results
    results_file = exp_path / "results.csv"
    if results_file.exists():
        df = pd.read_csv(results_file)
        df.columns = df.columns.str.strip()  # Remove whitespace
        
        print("📊 **Training Results**:")
        print(f"  Total epochs trained: {len(df)}")
        print()
        
        # Final epoch metrics
        final = df.iloc[-1]
        print("  Final Metrics:")
        
        # Box loss
        if 'train/box_loss' in df.columns:
            print(f"    Box loss: {final['train/box_loss']:.4f}")
        
        # Classification loss
        if 'train/cls_loss' in df.columns:
            print(f"    Classification loss: {final['train/cls_loss']:.4f}")
        
        # DFL loss
        if 'train/dfl_loss' in df.columns:
            print(f"    DFL loss: {final['train/dfl_loss']:.4f}")
        
        print()
        
        # Validation metrics
        if 'metrics/mAP50(B)' in df.columns:
            best_map = df['metrics/mAP50(B)'].max()
            best_epoch = df['metrics/mAP50(B)'].idxmax() + 1
            final_map = final['metrics/mAP50(B)']
            
            print("  Validation Performance:")
            print(f"    Best mAP@0.5: {best_map:.4f} (epoch {best_epoch})")
            print(f"    Final mAP@0.5: {final_map:.4f}")
            
            if 'metrics/mAP50-95(B)' in df.columns:
                best_map_95 = df['metrics/mAP50-95(B)'].max()
                print(f"    Best mAP@0.5:0.95: {best_map_95:.4f}")
        
        print()
        
        # Loss trends
        print("  Loss Trends:")
        if 'train/box_loss' in df.columns:
            initial_box = df['train/box_loss'].iloc[0]
            final_box = final['train/box_loss']
            improvement = ((initial_box - final_box) / initial_box) * 100
            trend = "↓" if improvement > 0 else "↑"
            print(f"    Box loss: {initial_box:.4f} → {final_box:.4f} ({trend} {abs(improvement):.1f}%)")
        
        if 'train/cls_loss' in df.columns:
            initial_cls = df['train/cls_loss'].iloc[0]
            final_cls = final['train/cls_loss']
            improvement = ((initial_cls - final_cls) / initial_cls) * 100
            trend = "↓" if improvement > 0 else "↑"
            print(f"    Classification loss: {initial_cls:.4f} → {final_cls:.4f} ({trend} {abs(improvement):.1f}%)")
    
    print()
    
    # Check output files
    print("📁 **Output Files**:")
    weights_dir = exp_path / "weights"
    if weights_dir.exists():
        print(f"  ✓ Checkpoints: {weights_dir}")
        for weight_file in weights_dir.glob("*.pt"):
            size_mb = weight_file.stat().st_size / (1024 * 1024)
            print(f"    - {weight_file.name} ({size_mb:.1f} MB)")
    
    if (exp_path / "results.png").exists():
        print(f"  ✓ Training curves: results.png")
    
    if (exp_path / "confusion_matrix.png").exists():
        print(f"  ✓ Confusion matrix: confusion_matrix.png")
    
    if (exp_path / "labels.jpg").exists():
        print(f"  ✓ Label distribution: labels.jpg")
    
    print()
    print("=" * 70)
    print("✅ Summary Complete")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize training experiment")
    parser.add_argument(
        "experiment_dir",
        type=str,
        help="Path to experiment directory (e.g., runs/train/quick_start_experiment)"
    )
    
    args = parser.parse_args()
    summarize_experiment(args.experiment_dir)
