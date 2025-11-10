# 🔧 Training Recovery Checklist

**Date**: November 9, 2025, 22:00  
**Status**: System recovered from crash, ready to restart

---

## ✅ Completed Recovery Steps

### 1. System Health Verified
- [x] GPU driver loaded (amdgpu)
- [x] PyTorch GPU detection working
- [x] ROCm 5.2.0 functional
- [x] GPU memory available: 5.98 GB
- [x] System memory healthy: 31 GB total
- [x] Disk space adequate: 166 GB free

### 2. Monitoring Tools Installed
- [x] `lm-sensors` installed and configured
- [x] `rocm-smi` installed and working
- [x] GPU temperature sensors active
- [x] CPU temperature sensors active

### 3. Current System Status
```
GPU (RX 5600 XT):
  Temperature: 49°C (edge), 49°C (junction), 62°C (memory)
  Power: 21W (idle, max 160W)
  VRAM: 16% used (idle)
  Utilization: 3% (idle)
  Fan: 0 RPM (auto mode)

CPU (Ryzen 5 3600):
  Temperature: 40°C
  System Load: Low

Memory:
  Used: 9.7 GB / 31 GB
  Swap: 0 / 8 GB
  Status: Healthy
```

### 4. Crash Analysis Completed
- [x] Incident report created: `CRASH_INCIDENT_REPORT.md`
- [x] Root cause identified: GPU hang (RDNA1 stability issue)
- [x] Lost progress documented: 31% of epoch 1 (~3 hours)
- [x] Mitigation strategies defined

### 5. Monitoring Scripts Created
- [x] `gpu_watchdog.sh` - Real-time GPU monitoring
- [x] `monitor_training_alerts.sh` - Epoch milestone alerts
- [x] `analyze_training_results.sh` - Metrics analysis

---

## 📋 Pre-Flight Checklist (Before Restart)

### System Health
- [x] GPU temperature < 50°C ✅ (49°C)
- [x] GPU driver loaded ✅
- [x] ROCm functional ✅
- [x] Disk space > 50 GB ✅ (166 GB)
- [x] RAM usage < 10 GB idle ✅ (9.7 GB)

### Training Configuration Changes
- [ ] **CRITICAL**: Reduce batch size from 16 to 12
- [ ] Verify dataset accessible
- [ ] Check virtual environment
- [ ] Review training arguments
- [ ] Clean old training run directory

### Monitoring Setup
- [ ] Start GPU watchdog (`./gpu_watchdog.sh`)
- [ ] Test GPU monitoring working
- [ ] Start training alerts monitor
- [ ] Verify notification system
- [ ] Set up temperature alerts

### Safety Measures (Recommended)
- [ ] Test with 5-epoch trial run first
- [ ] Monitor GPU temps during first hour
- [ ] Check for stability issues
- [ ] Verify checkpointing works

---

## 🚀 Restart Procedure

### Step 1: Clean Old Training Run
```bash
# Backup old run (optional)
mv runs/detect/production_yolov8n_rocm522 runs/detect/production_yolov8n_rocm522_crashed_$(date +%Y%m%d_%H%M)

# Or just remove it
rm -rf runs/detect/production_yolov8n_rocm522
```

### Step 2: Start GPU Watchdog
```bash
# Run in background with nohup
nohup ./gpu_watchdog.sh > gpu_watchdog_output.log 2>&1 &
echo $! > .gpu_watchdog_pid

# Verify it's running
sleep 2 && tail -10 gpu_watchdog.log
```

### Step 3: Start Training (REDUCED BATCH SIZE)
```bash
# Activate virtual environment
source venv-py310-rocm52/bin/activate

# Start training with batch size 12 (reduced from 16)
nohup yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=100 \
  batch=12 \
  device=0 \
  imgsz=640 \
  amp=False \
  name=production_yolov8n_rocm52_v2 \
  project=runs/detect \
  > training.log 2>&1 &

# Save training PID
echo $! > .training_pid

# Display PID
cat .training_pid
```

### Step 4: Start Training Monitor
```bash
# Clean old monitor state
rm -f .training_state monitor.log training_alerts.log

# Start monitor in background
nohup ./monitor_training_alerts.sh > monitor.log 2>&1 &
echo $! > .monitor_pid

# Verify running
sleep 2 && tail -10 monitor.log
```

### Step 5: Verify All Systems Running
```bash
# Check all processes
echo "Training PID: $(cat .training_pid)" && ps -p $(cat .training_pid) | grep -v PID
echo "GPU Watchdog PID: $(cat .gpu_watchdog_pid)" && ps -p $(cat .gpu_watchdog_pid) | grep -v PID
echo "Monitor PID: $(cat .monitor_pid)" && ps -p $(cat .monitor_pid) | grep -v PID
```

---

## 🔍 Critical Monitoring (First Hour)

### Every 5 Minutes (First 30 Minutes)
```bash
# Quick GPU check
sensors amdgpu-pci-2d00 | grep -E "edge|junction|mem"

# Should see:
# edge: < 85°C (warning at 85°C, critical at 100°C)
# junction: < 90°C (warning at 90°C, critical at 110°C)
# mem: < 95°C (warning at 95°C, critical at 105°C)
```

### Every 15 Minutes (First Hour)
```bash
# Full system status
./check_training_progress.sh

# GPU watchdog log
tail -20 gpu_watchdog.log

# Training progress
tail -100 training.log | grep "1/100" | tail -1
```

### Warning Signs to Watch For
- 🔴 **GPU temp > 85°C** - Reduce batch size or stop
- 🔴 **Memory > 95%** - Possible OOM incoming
- 🔴 **Training log not updating for 5+ min** - Possible hang
- 🔴 **GPU util = 0% for 2+ min** - GPU driver issue
- 🟡 **Fan speed = 0 RPM with high temp** - Fan control issue

---

## 🧪 Optional: 5-Epoch Test Run (Recommended)

Before committing to full 100-epoch run, test with 5 epochs:

```bash
# Stop current training if running
kill $(cat .training_pid)

# Clean directory
rm -rf runs/detect/production_yolov8n_rocm52_test

# Run 5-epoch test
source venv-py310-rocm52/bin/activate
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=5 \
  batch=12 \
  device=0 \
  imgsz=640 \
  amp=False \
  name=production_yolov8n_rocm52_test \
  project=runs/detect

# Monitor closely
# If completes without crash, proceed with full training
# Estimated time: 1-2 hours
```

---

## 📊 Success Criteria

### Training Launch Success
- ✅ All 3 processes running (training, watchdog, monitor)
- ✅ Training log updating regularly
- ✅ GPU temperature stable < 80°C
- ✅ VRAM usage stable < 90%
- ✅ No HSA errors in log

### First Hour Success
- ✅ GPU temperature stays < 85°C
- ✅ Training completes 3-4% of epoch 1
- ✅ No GPU driver errors
- ✅ No system hangs
- ✅ Watchdog not reporting alerts

### Epoch 1 Success (Target: ~4 hours)
- ✅ Epoch 1 completes without crash
- ✅ Checkpoint saved (best.pt, last.pt)
- ✅ results.csv created
- ✅ Alert notification received
- ✅ Analysis report generated

---

## 🛑 Emergency Procedures

### If GPU Temperature > 90°C
```bash
# Immediately stop training
kill $(cat .training_pid)

# Check GPU health
rocm-smi --showtemp
sensors amdgpu-pci-2d00

# Wait for cooldown (< 50°C)
# Restart with batch size 8
```

### If System Becomes Unresponsive
```bash
# From SSH or TTY (Ctrl+Alt+F2):
# Find training process
ps aux | grep "yolo detect train"

# Kill it
kill -9 <PID>

# Check GPU status
rocm-smi

# If GPU hung, may need reboot
sudo reboot
```

### If Training Crashes
```bash
# Check last error
tail -50 training.log

# Check GPU errors
grep -i "error\|hsa" training.log | tail -20

# Check watchdog alerts
cat gpu_alerts.log

# Review and adjust batch size down
```

---

## 📈 Expected Timeline (With Batch Size 12)

**Batch Size 12 Impact**:
- Slower per-epoch time: ~4-5 hours (vs 3-4 with batch 16)
- Lower VRAM usage: ~4.5 GB (vs 5.7 GB)
- Better thermal management: More headroom
- **Total training time**: 16-20 days (vs 12-16 days)

### Wait, that's too long!

**Alternative: Reduce Epochs to 50**
```bash
# 50 epochs with batch 12: 8-10 days
# mAP@0.5 target: > 0.5 achievable in 30-50 epochs
```

### Milestone Timeline (100 Epochs)
- **Epoch 1**: Tonight (4-5 hours)
- **Epoch 10**: Tomorrow afternoon
- **Epoch 30**: 5-6 days
- **Epoch 50**: 8-10 days
- **Epoch 100**: 16-20 days

---

## 🎯 Decision Point

### Option A: Full 100 Epochs (Recommended if time allows)
- **Pros**: Best possible accuracy, complete training
- **Cons**: 16-20 days, higher crash risk
- **Use case**: Production model, research

### Option B: 50 Epochs (Recommended for stability testing)
- **Pros**: 8-10 days, still achieves good accuracy
- **Cons**: May not reach maximum performance
- **Use case**: First successful RDNA1 training

### Option C: 30 Epochs (Quick validation)
- **Pros**: 5-6 days, proves stability
- **Cons**: Lower accuracy
- **Use case**: Proof of concept, testing

**Recommendation**: Start with 50 epochs to prove stability, then extend if needed.

---

## 📝 Next Steps

1. [ ] Review this checklist
2. [ ] Decide on epoch count (100, 50, or 30)
3. [ ] Execute restart procedure
4. [ ] Monitor first hour closely
5. [ ] Verify epoch 1 completion
6. [ ] Document success/issues

---

**Status**: 🟡 **READY TO RESTART**  
**Safety Level**: 🟢 **HIGH** (monitoring in place, batch size reduced)  
**Crash Risk**: 🟡 **MEDIUM** (RDNA1 stability still a concern)

*Checklist created: November 9, 2025, 22:00*  
*Next action: Review and execute restart procedure*
