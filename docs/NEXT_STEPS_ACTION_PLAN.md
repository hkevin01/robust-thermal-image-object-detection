# 🎯 Next Steps Action Plan

## ✅ Completed Tasks

- [x] Download LTDv2 dataset (1M+ images, 48GB)
- [x] Extract dataset (1,069,247 images)
- [x] Convert COCO to YOLO format (329,299 train, 41,226 val, 46,884 test)
- [x] Update .gitignore (reduced from 793K to 14 files)
- [x] Configure Copilot (prevent VS Code crashes)
- [x] Update baseline config (point to ltdv2_full)
- [x] Create data converters (4 versions)
- [x] Document project status

## 🚀 Immediate Next Steps

### 1. Commit Current Work ⏭️
```bash
# Already staged, ready to commit
git commit -m "feat: add LTDv2 dataset support and Copilot configuration

- Update .gitignore to exclude data directories (reduce 793K files)
- Add Copilot instructions to prevent VS Code crashes
- Create efficient LTDv2 COCO→YOLO converters
- Update baseline config to use ltdv2_full dataset
- Add comprehensive documentation
"

# Optional: Push to remote
git push origin main
```

### 2. Verify Dataset Integrity 🔍
```bash
# Check actual converted dataset
cd data/ltdv2_full

# Count images
find images/train -type l | wc -l  # Should be ~329K
find images/val -type l | wc -l    # Should be ~41K  
find images/test -type l | wc -l   # Should be ~47K

# Count labels
find labels/train -name "*.txt" | wc -l  # Should match train images
find labels/val -name "*.txt" | wc -l    # Should match val images

# Check data.yaml
cat data.yaml
```

### 3. Test Data Loading 📊
```bash
# Quick test of data loader
./venv/bin/python -c "
from ultralytics import YOLO
from pathlib import Path

data_yaml = 'data/ltdv2_full/data.yaml'
print(f'Testing data loading from: {data_yaml}')

# Load a small batch to verify
model = YOLO('yolov8n.pt')
results = model.val(
    data=data_yaml,
    split='val',
    batch=4,
    imgsz=640,
    max_det=100,
    save=False,
    plots=False,
    verbose=False
)
print(f'✓ Data loading successful!')
print(f'  mAP50: {results.box.map50:.4f}')
print(f'  Images: {len(results.speed)}')
"
```

### 4. Check GPU Availability 🖥️
```bash
# Check CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# If GPU available, verify it works with YOLO
./venv/bin/python -c "
from ultralytics import YOLO
import torch
print(f'PyTorch CUDA: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

## 🎓 Training Phase

### Option A: Start Small (Recommended First) 🏃

**Quick validation run** (30 minutes):
```bash
./venv/bin/python src/training/train.py \
  --config configs/quick_start.yaml \
  --data data/ltdv2_full/data.yaml \
  --epochs 5 \
  --batch 16 \
  --device 0 \
  --project runs/validation
```

**Expected outcome**: 
- Verify data loading works
- Check GPU utilization
- Validate training loop
- Confirm checkpoints save correctly

### Option B: Baseline Training 🎯

**Full baseline** (2-7 days on single GPU):
```bash
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --data data/ltdv2_full/data.yaml \
  --epochs 100 \
  --batch 32 \
  --device 0 \
  --project runs/baseline \
  --name ltdv2_yolov8m_baseline
```

**Expected metrics**:
- Target mAP@0.5: 0.55-0.65
- Target CoV: < 0.30
- Challenge Score: 0.40-0.50

### Option C: Smaller Subset First 🔬

If GPU memory limited or want faster iteration:
```bash
# Create 10K subset
./venv/bin/python scripts/data/download_ltdv2.py \
  --output data/ltdv2_10k \
  --mode subset \
  --subset-train 8000 \
  --subset-val 2000

# Train on subset
./venv/bin/python src/training/train.py \
  --config configs/baseline.yaml \
  --data data/ltdv2_10k/data.yaml \
  --epochs 50 \
  --device 0
```

## 📈 Monitoring Training

### Real-time Monitoring
```bash
# Option 1: Watch training log
tail -f runs/baseline/ltdv2_yolov8m_baseline/train.log

# Option 2: TensorBoard (if configured)
tensorboard --logdir runs/baseline

# Option 3: W&B (if API key set)
# Training will auto-log to https://wandb.ai/your-username/thermal-detection-ltdv2
```

### Check Progress
```bash
# View results
cat runs/baseline/ltdv2_yolov8m_baseline/results.csv

# Summarize experiment
./venv/bin/python scripts/training/summarize_experiment.py \
  runs/baseline/ltdv2_yolov8m_baseline
```

## 🔧 Optimization Phase

### After Baseline Complete:

1. **Analyze Results**:
   - Check mAP@0.5 per class
   - Calculate temporal consistency (CoV)
   - Compute challenge score
   - Identify failure modes

2. **Advanced Training** (choose based on results):

   **Weather-Conditioned**:
   ```bash
   ./venv/bin/python src/training/train.py \
     --config configs/weather_conditioned.yaml \
     --data data/ltdv2_full/data.yaml
   ```

   **Domain Adaptation**:
   ```bash
   ./venv/bin/python src/training/train.py \
     --config configs/domain_adaptation.yaml \
     --data data/ltdv2_full/data.yaml
   ```

   **Hyperparameter Sweep**:
   ```bash
   # Requires W&B account
   ./venv/bin/python src/training/train.py \
     --config configs/wandb_sweep.yaml \
     --sweep
   ```

3. **Ensemble Models** (if time permits):
   - Train YOLOv8x for higher accuracy
   - Train YOLOv8s for speed
   - Combine predictions

## 📝 Evaluation & Submission

### Generate Predictions
```bash
# Run on test set
./venv/bin/python src/inference/predict.py \
  --model runs/baseline/ltdv2_yolov8m_baseline/weights/best.pt \
  --data data/ltdv2_full/data.yaml \
  --split test \
  --output submissions/baseline
```

### Calculate Challenge Score
```bash
# Compute mAP and CoV
./venv/bin/python scripts/evaluation/compute_challenge_score.py \
  --predictions submissions/baseline/predictions.json \
  --ground-truth data/ltdv2_full/Valid.json
```

### Prepare Submission
```bash
# Format according to competition requirements
# (Check WACV 2026 RWS Challenge submission guidelines)
```

## 🚨 Troubleshooting

### If Training Crashes

**Out of Memory**:
```bash
# Reduce batch size in config
# Or use gradient accumulation:
--batch 16 --accumulate 2  # Effective batch size 32
```

**Data Loading Slow**:
```bash
# Increase workers in config
workers: 8  # Adjust based on CPU cores
```

**Disk Space Issues**:
```bash
# Check space
df -h

# Clean old runs
rm -rf runs/old_experiments/

# Disable checkpoint saving for epochs
save_period: 10  # Only save every 10 epochs
```

### If GPU Not Available

**CPU Training** (very slow):
```bash
# Use tiny model and small subset
./venv/bin/python src/training/train.py \
  --config configs/quick_start.yaml \
  --data data/ltdv2_10k/data.yaml \
  --device cpu \
  --workers 4
```

**Cloud GPU** (recommended):
- Google Colab Pro (~$10/month)
- Paperspace Gradient
- AWS/GCP/Azure GPU instances

## 📊 Success Criteria

| Metric | Baseline Target | Competitive Target |
|--------|----------------|-------------------|
| mAP@0.5 | 0.55+ | 0.65+ |
| CoV | < 0.30 | < 0.20 |
| Challenge Score | 0.40+ | 0.52+ |
| Inference Speed | > 10 FPS | > 20 FPS |

## 🎯 Priority Order

1. **HIGH**: Verify dataset (step 2)
2. **HIGH**: Test data loading (step 3)
3. **HIGH**: Check GPU (step 4)
4. **MEDIUM**: Quick validation run (Option A)
5. **MEDIUM**: Baseline training (Option B)
6. **LOW**: Advanced experiments
7. **LOW**: Ensemble methods

## 📚 Resources

**Documentation**:
- `.github/COPILOT_GUIDE.md` - Copilot usage
- `DATASET_READY.md` - Dataset status
- `STATUS_AND_NEXT_STEPS.md` - Project overview
- `docs/experiment-log.md` - Training log

**Configs**:
- `configs/baseline.yaml` - Main training config
- `configs/quick_start.yaml` - Fast validation
- `configs/weather_conditioned.yaml` - Weather metadata
- `configs/domain_adaptation.yaml` - Temporal consistency

**Scripts**:
- `src/training/train.py` - Main training script
- `scripts/training/summarize_experiment.py` - Results analysis
- `scripts/data/convert_ltdv2_efficient.py` - Dataset converter

---

**Current State**: Ready to train! Dataset converted, configs updated, git cleaned.
**Next Action**: Run validation (Option A) or start baseline (Option B)
**ETA**: Validation ~30min, Baseline ~2-7 days

🚀 **Let's compete!**
