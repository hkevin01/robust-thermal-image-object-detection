# Experiment Log

This document tracks all training experiments and their results.

## Experiment 1: Quick Start Validation

**Date**: November 5, 2025  
**Status**: ✅ **COMPLETED** - Pipeline Validated  
**Goal**: Validate training pipeline with synthetic dataset

### Configuration

- **Model**: YOLOv8n (Nano - 3M parameters)
- **Dataset**: Synthetic thermal images
  - Train: 20 images
  - Val: 5 images
  - Test: 5 images
- **Training Settings**:
  - Epochs: 10
  - Batch size: 4
  - Image size: 640×640
  - Device: CPU
  - Optimizer: AdamW (lr=0.001)
  - Workers: 2

### Results

| Metric | Value | Notes |
|--------|-------|-------|
| Final mAP@0.5 | 0.0 | Expected - dataset too small |
| Training time | 0.01 hours (~36 seconds) | On CPU |
| Box loss | 3.070 (final) | Decreasing trend |
| Classification loss | 4.457 (final) | Decreasing trend |
| DFL loss | 2.938 (final) | Decreasing trend |

**Per-class Results**:
- Person: mAP=0.0
- Bicycle: mAP=0.0
- Motorcycle: mAP=0.0
- Vehicle: mAP=0.0

### Analysis

#### ✅ Successes
1. **Pipeline Validation**: Complete training pipeline works end-to-end
2. **Data Loading**: Synthetic dataset successfully loaded
3. **Model Training**: YOLOv8 trained without errors
4. **Loss Convergence**: Losses showed decreasing trend
5. **Checkpointing**: Weights saved correctly
6. **Validation**: Automatic validation executed

#### ⚠️ Limitations
1. **Dataset Size**: Only 20 training images insufficient for learning
2. **CPU Training**: Very slow compared to GPU
3. **Zero mAP**: Expected due to tiny dataset and short training
4. **Synthetic Data**: Not representative of real thermal imagery

### Conclusions

- ✅ Training infrastructure is working correctly
- ✅ All components (data, model, training, validation) integrated properly
- ✅ Ready for experiments with real LTDv2 dataset
- 📝 Next: Download full LTDv2 dataset and train on GPU

### Output Files

```
runs/train/quick_start_experiment/
├── weights/
│   ├── best.pt    # Best model checkpoint
│   └── last.pt    # Last epoch checkpoint
├── labels.jpg     # Label distribution visualization
├── results.csv    # Training metrics
└── args.yaml      # Training arguments
```

---

## Next Experiments

### Experiment 2: Baseline on LTDv2 (Planned)

**Goal**: Establish baseline performance on full dataset

**Configuration**:
- Model: YOLOv8m (Medium)
- Dataset: Full LTDv2 (1M+ images)
- Epochs: 100
- Batch size: 16
- Device: GPU (CUDA)
- Config: `configs/baseline.yaml`

**Expected Results**:
- mAP@0.5: 0.55-0.65
- Training time: ~24-48 hours on single GPU

### Experiment 3: Weather-Conditioned Training (Planned)

**Goal**: Test metadata fusion approach

**Configuration**:
- Model: YOLOv8m
- Dataset: LTDv2 with metadata
- Epochs: 150
- Config: `configs/weather_conditioned.yaml`

**Expected Improvements**:
- Better temporal consistency (lower CoV)
- Improved performance in challenging weather

### Experiment 4: Domain Adaptation (Planned)

**Goal**: Minimize performance drift across seasons

**Configuration**:
- Model: YOLOv8l (Large)
- Dataset: LTDv2 with temporal splits
- Epochs: 200
- Config: `configs/domain_adaptation.yaml`

**Expected Results**:
- CoV < 0.15
- Challenge score > 0.60

---

## Experiment Tracking

### Metrics to Track

1. **Detection Performance**:
   - Global mAP@0.5
   - Per-class AP
   - mAP@[0.5:0.95]

2. **Temporal Consistency**:
   - Monthly mAP
   - Coefficient of Variation (CoV)
   - Challenge Score = mAP × (1 - CoV)

3. **Training Efficiency**:
   - Training time
   - GPU memory usage
   - Convergence speed

4. **Robustness**:
   - Performance across weather conditions
   - Day vs night performance
   - Season-specific performance

### Tools

- **Weights & Biases**: Experiment tracking and visualization
- **Local Logs**: CSV files in runs/train/
- **TensorBoard**: Alternative visualization (optional)

---

## Best Practices

1. **Always use GPU** for real experiments (100x faster than CPU)
2. **Track everything** with W&B or similar tools
3. **Save checkpoints** every 5-10 epochs
4. **Validate frequently** to catch overfitting early
5. **Use larger datasets** (>1000 images minimum)
6. **Monitor losses** - should steadily decrease
7. **Test early stopping** with patience=20-30
8. **Run multiple seeds** for statistical significance

---

**Last Updated**: November 5, 2025  
**Next Action**: Download LTDv2 dataset and run Experiment 2
