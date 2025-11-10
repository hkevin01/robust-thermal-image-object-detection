# YOLOv8 Training Monitoring Guide - ROCm 5.2.0 RDNA1

**Training Start Time**: November 9, 2024, 18:47  
**Configuration**: YOLOv8n, 100 epochs, batch=16, amp=False, ROCm 5.2.0 + PyTorch 1.13.1
**Dataset**: LTDv2 (329,299 train, 43,850 val images)

---

## �� Quick Status Check

```bash
# Check if training is still running
ps aux | grep "yolo.*train" | grep -v grep

# View last 50 lines of training log
tail -50 training.log

# Watch log in real-time (Ctrl+C to exit)
tail -f training.log

# Check training progress (look for epoch numbers)
grep "Epoch" training.log | tail -10

# Check for any errors
grep -i "error\|fail\|crash" training.log
```

## 📊 Training Performance Metrics

### Monitor GPU Usage (using htop for CPU/RAM)
```bash
# Install htop if not available
sudo apt install htop

# Run htop and look for python3.10 processes
htop
# Main process should show 100%+ CPU
# 8 dataloader workers should each show ~30-50% CPU
```

### Check Training Speed
```bash
# Extract epoch timing information
grep "Epoch" training.log | tail -20
```

**Expected Performance**:
- **First Epoch**: 30-60 minutes (MIOpen kernel compilation)
- **Subsequent Epochs**: 15-25 minutes (using cached kernels)
- **Total Training Time**: ~24-48 hours for 100 epochs

### Check Memory Usage
```bash
# Check system RAM usage
free -h

# Check process memory
ps aux | grep yolo | awk '{print $6, $11}' | grep python
```

**Expected Memory Usage**:
- Main process: 3-4 GB RAM
- 8 dataloader workers: ~1 GB each (~8 GB total)
- Total system RAM: 12-15 GB used (out of 31 GB)

## 📈 Training Progress Files

### Key Output Files
```
runs/detect/production_yolov8n_rocm522/
├── weights/
│   ├── best.pt          # Best model weights (highest mAP)
│   └── last.pt          # Latest model weights
├── results.csv          # Training metrics (loss, mAP, etc.)
├── results.png          # Training curves visualization
├── labels.jpg           # Dataset label distribution
├── train_batch*.jpg     # Training batch visualizations
└── val_batch*.jpg       # Validation batch visualizations
```

### View Training Metrics
```bash
# Check results file
head -20 runs/detect/production_yolov8n_rocm522/results.csv

# View specific metrics (if results.csv exists)
tail -10 runs/detect/production_yolov8n_rocm522/results.csv | column -t -s,
```

### Metrics to Watch:
- **box_loss**: Bounding box regression loss (should decrease)
- **cls_loss**: Classification loss (should decrease)
- **dfl_loss**: Distribution focal loss (should decrease)
- **metrics/mAP50**: Mean Average Precision at IoU=0.50 (should increase, target >0.7)
- **metrics/mAP50-95**: mAP averaged over IoU 0.50-0.95 (should increase, target >0.5)

## ⚠️ Troubleshooting

### If Training Crashes or Hangs

1. **Check for errors in log**:
   ```bash
   grep -i "error\|exception\|traceback" training.log | tail -20
   ```

2. **Check if process is still running**:
   ```bash
   ps aux | grep 407777  # Use actual PID
   ```

3. **If process stopped unexpectedly**:
   ```bash
   # Check system logs
   sudo dmesg | tail -50
   
   # Check for out-of-memory killer
   grep "Out of memory" /var/log/syslog
   ```

### Common Issues & Solutions

#### 1. Training Hangs (No Progress for >10 minutes)
**Symptom**: Log stops updating, CPU drops to 0%  
**Solution**:
```bash
# Kill training
kill -9 407777  # Use actual PID

# Restart with same config (will resume if checkpoint exists)
source venv/bin/activate
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  batch=16 \
  device=0 \
  imgsz=640 \
  amp=False \
  name=production_yolov8n_rocm52 \
  project=runs/detect \
  resume=True
```

#### 2. Out of Memory Error
**Symptom**: "RuntimeError: CUDA out of memory" in log  
**Solution**:
```bash
# Reduce batch size and restart
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=runs/detect/production_yolov8n_rocm522/weights/last.pt \
  epochs=100 \
  batch=12 \  # Reduced from 16
  device=0 \
  imgsz=640 \
  amp=False \
  name=production_yolov8n_rocm52_resume \
  project=runs/detect
```

#### 3. HSA Memory Aperture Violation (RDNA1 Bug Returns)
**Symptom**: "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in log  
**Solution**:
```bash
# Verify environment variables are set
echo $MIOPEN_DEBUG_CONV_IMPLICIT_GEMM  # Should be 1
echo $HSA_OVERRIDE_GFX_VERSION  # Should be 10.3.0

# If not set, source environment file
source /etc/profile.d/rocm-rdna1-52.sh

# Restart training
source venv/bin/activate
yolo detect train [same arguments as before]
```

## 📊 Progress Checkpoints

### At Epoch 10 (~3-5 hours):
- [ ] Check first 10 epochs completed without crashes
- [ ] Verify losses are decreasing
- [ ] Confirm results.csv exists and has data
- [ ] Check mAP50 starting to show meaningful values (>0.1)

### At Epoch 30 (~10-12 hours):
- [ ] Training speed stabilized (epochs taking consistent time)
- [ ] Losses significantly lower than epoch 1
- [ ] mAP50 > 0.3
- [ ] No memory leaks (RAM usage stable)

### At Epoch 50 (Halfway, ~24 hours):
- [ ] mAP50 > 0.5
- [ ] Losses plateauing or slowly decreasing
- [ ] best.pt file exists with good performance
- [ ] Consider whether to continue to 100 or stop early

### At Epoch 100 (Complete, ~48 hours):
- [ ] Training completed successfully
- [ ] Final mAP50 documented
- [ ] Best model weights saved
- [ ] Results documented in ROCM_52_TRAINING_SUCCESS.md

## �� After Training Completes

### Validate Final Model
```bash
source venv/bin/activate
yolo detect val \
  model=runs/detect/production_yolov8n_rocm522/weights/best.pt \
  data=data/ltdv2_full/data.yaml \
  device=0
```

### Test on Sample Images
```bash
yolo detect predict \
  model=runs/detect/production_yolov8n_rocm522/weights/best.pt \
  source=data/ltdv2_full/images/test \
  device=0 \
  save=True \
  conf=0.25
```

### Backup Results
```bash
# Create timestamped backup
timestamp=$(date +%Y%m%d_%H%M%S)
tar -czf training_backup_$timestamp.tar.gz \
  runs/detect/production_yolov8n_rocm522 \
  training.log \
  ROCM_52_TRAINING_SUCCESS.md
```

## 📝 Documentation Template

When training completes, create `ROCM_52_TRAINING_SUCCESS.md`:
```markdown
# ROCm 5.2.0 RDNA1 Training Success Report

## Configuration
- **GPU**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)
- **Software**: ROCm 5.2.0, PyTorch 1.13.1+rocm5.2, Python 3.10.19
- **Model**: YOLOv8n (3M parameters)
- **Dataset**: LTDv2 (420,033 images, 5 classes)
- **Training**: 100 epochs, batch=16, amp=False
- **Algorithm**: MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1

## Results
- **Final mAP50**: [X.XXX]
- **Final mAP50-95**: [X.XXX]
- **Training Time**: [XX hours]
- **Best Epoch**: [XX]
- **Stability**: [No crashes/hangs]

## Key Achievements
- ✅ First successful GPU training on RDNA1 hardware
- ✅ IMPLICIT_GEMM algorithm bypassed Conv2d hardware bug
- ✅ Training completed without HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
- ✅ Production-ready model for thermal image object detection

## Files
- Best weights: `runs/detect/production_yolov8n_rocm522/weights/best.pt`
- Results: `runs/detect/production_yolov8n_rocm522/results.csv`
- Visualizations: `runs/detect/production_yolov8n_rocm522/*.png`
```

---

**Last Updated**: November 9, 2024, 18:50  
**Training PID**: 407777  
**Status**: 🟢 Running (First epoch, kernel compilation phase)
