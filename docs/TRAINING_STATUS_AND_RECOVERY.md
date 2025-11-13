# Training Status and Recovery Guide

**Date**: November 10, 2025  
**Status**: ✅ TRAINING ACTIVE (9+ hours runtime)

## Current Training Status

### Active Training
- **Process**: `train_patched.py` (PID 229139)
- **Runtime**: 9+ hours (started 10:33 AM)
- **Processes**: 21 Python processes
- **GPU Usage**: 99% (actively training)
- **Temperature**: 38-39°C edge, 52°C memory
- **Fan Speed**: 70% (smart curve active)

### Issue Identified
⚠️ **No checkpoints being saved yet** - Current train_patched.py doesn't have `save_period=1`

## VS Code Independent Training

### New Standalone Training Script Created
**File**: `train_standalone.py`

**Features**:
- ✅ Saves checkpoint **every epoch** (`save_period=1`)
- ✅ Automatic resume from `last.pt`
- ✅ Signal handlers for graceful shutdown
- ✅ MIOpen bypass included
- ✅ Comprehensive error handling
- ✅ Independent of VS Code

### Launch Scripts

#### 1. Screen Session Launcher
**File**: `start_training_standalone.sh`

```bash
./start_training_standalone.sh
```

**What it does**:
- Starts training in detached screen session
- Logs to `logs/training_YYYYMMDD_HHMMSS.log`
- Survives system crashes, reboots, VS Code closes
- Can reconnect anytime with `screen -r yolo_training`

**Screen Commands**:
```bash
# Attach to training session
screen -r yolo_training

# Detach (leave running)
Ctrl+A then D

# List sessions
screen -ls

# Kill session
screen -X -S yolo_training quit
```

#### 2. Status Monitoring
**File**: `check_training_status.sh`

```bash
./check_training_status.sh
```

**Shows**:
- Running processes
- GPU usage and temperature
- Latest training runs
- Checkpoint status
- Screen sessions

## Checkpoint Strategy

### Current Training (train2)
- **Location**: `runs/detect/train2/`
- **Status**: Running but no checkpoints yet (needs save_period=1)
- **Risk**: If crashes before epoch 1 completes, need to restart

### Recommended Action

**Option 1: Let Current Training Continue**
- Wait for it to complete epoch 1 (may take ~4-6 more hours)
- Will save checkpoint at epoch 1
- Can then resume if crashes

**Option 2: Switch to Standalone Now**
```bash
# 1. Stop current training
pkill -f "python train_patched.py"

# 2. Wait for processes to stop
sleep 5

# 3. Start standalone training
./start_training_standalone.sh

# 4. Monitor
screen -r yolo_training
```

## System Crash Recovery

### If System Crashes Mid-Training

#### Check for Checkpoints
```bash
ls -lh runs/detect/train_standalone/weights/last.pt
```

#### Resume Training
```bash
# If checkpoint exists, just restart
./start_training_standalone.sh
# Script will auto-detect and resume

# Or manually:
screen -dmS yolo_training bash -c "
    cd ~/Projects/robust-thermal-image-object-detection
    source venv-py310-rocm52/bin/activate
    python train_standalone.py
"
```

#### If No Checkpoint
Training will start from scratch. To minimize risk:
- Current setup saves **every epoch**
- Each epoch ~4-6 hours
- Maximum loss: 1 epoch of work

## Monitoring During Training

### GPU Status
```bash
watch -n 5 rocm-smi --showuse --showtemp
```

### Training Log (if in screen)
```bash
screen -r yolo_training
# Ctrl+A then D to detach
```

### Check Progress
```bash
# View results
tail -f runs/detect/train_standalone/results.csv

# Check latest checkpoint
ls -lh runs/detect/train_standalone/weights/
```

## Training Configuration

### Standalone Training Settings
- **Epochs**: 50
- **Batch Size**: 4
- **Image Size**: 640×640
- **Workers**: 8
- **Optimizer**: SGD (momentum=0.937)
- **Learning Rate**: 0.01 (cosine decay)
- **AMP**: Disabled (stability)
- **Checkpointing**: Every epoch (`save_period=1`)

### Expected Timeline
- **Per Epoch**: ~4-6 hours
- **Total Training**: ~8-12 days
- **Checkpoint Size**: ~6.4 MB per checkpoint

## Fan Control

### Current Settings
- **Baseline**: 70% (quiet, consistent)
- **Smart Curve**: Ramps up with temperature
  - <55°C: 70%
  - 55-64°C: 75%
  - 65-69°C: 80%
  - 70-74°C: 90%
  - ≥75°C: 100%

### System-Wide Service
```bash
# Check status
sudo systemctl status amdgpu-fan-curve.service

# Restart if needed
sudo systemctl restart amdgpu-fan-curve.service
```

## Quick Commands Reference

```bash
# Check training status
./check_training_status.sh

# Start standalone training
./start_training_standalone.sh

# Attach to training session
screen -r yolo_training

# Check GPU
rocm-smi --showuse --showtemp

# View latest results
tail -f runs/detect/train_standalone/results.csv

# Check checkpoints
ls -lh runs/detect/train_standalone/weights/

# Stop training
screen -X -S yolo_training quit
```

## Next Steps After Training

1. **Generate Submission**
   ```bash
   python scripts/generate_submission.py \
       --model runs/detect/train_standalone/weights/best.pt \
       --test-dir data/ltdv2_full/valid/images \
       --output submission_dev.json
   ```

2. **Validate Submission**
   ```bash
   python scripts/pre_submission_check.py submission_dev.json
   ```

3. **Upload to Codabench**
   - Development Phase: Until Nov 30, 2025
   - Submit at: https://www.codabench.org/competitions/10954/

## Troubleshooting

### Training Not Progressing
```bash
# Check if stuck
ps aux | grep python | grep train

# Check GPU usage
rocm-smi --showuse

# If <90% usage for >5 minutes, restart
```

### Out of Memory
```bash
# Reduce batch size in train_standalone.py
# Change: batch=4 → batch=2
```

### MIOpen Errors
```bash
# Verify patch is applied
grep "APPLYING CONV2D PATCH" logs/training_*.log

# If not, check train_standalone.py imports
```

---

**Remember**: Standalone training with `screen` survives:
- ✅ VS Code crashes
- ✅ Terminal closes  
- ✅ SSH disconnects
- ✅ System reboots (with autostart)

Use `screen -r yolo_training` to reconnect anytime! 🚀
