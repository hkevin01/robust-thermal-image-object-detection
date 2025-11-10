# 🚀 Quick Reference - Training Monitoring

## One-Line Status Checks

```bash
# Full dashboard
./training_dashboard.sh

# Quick status
./check_status.sh

# Extract latest metrics
./extract_metrics.sh

# GPU temperature
rocm-smi --showtemp

# Live training log
tail -f training_production.log

# Live monitor log  
tail -f training_monitor.log

# Fan curve log
sudo tail -f /var/log/amdgpu-fan-curve.log

# Real-time GPU watch
watch -n1 'rocm-smi --showtemp --showfan --showuse'
```

## Critical Commands

### Check if Training Running
```bash
ps aux | grep "[p]ython.*train_patched"
```

### Check GPU Health
```bash
rocm-smi --showtemp --showfan --showuse --showmeminfo vram
```

### Check Fan Curve Service
```bash
sudo systemctl status amdgpu-fan-curve.service
```

### View Latest Progress
```bash
grep "1/50" training_production.log | tail -1
```

## Temperature Thresholds

| Sensor | Safe | Warning | Critical |
|--------|------|---------|----------|
| Edge | < 70°C | 70-85°C | > 85°C |
| Junction | < 80°C | 80-95°C | > 95°C |
| Memory | < 85°C | 85-95°C | > 95°C |

**Current**: 48°C edge, 56°C junction ✅ **EXCELLENT**

## Fan Speed Guide

| Speed | Usage | GPU Temp | Notes |
|-------|-------|----------|-------|
| 70% | Minimum (training) | 48-60°C | Optimal balance |
| 75-85% | Heavy load | 60-70°C | Normal training |
| 90-95% | Very heavy | 70-80°C | Peak performance |
| 100% | Critical | > 80°C | Emergency cooling |

**Current**: 74% ✅ **OPTIMAL**

## Training Files

| File | Purpose |
|------|---------|
| `train_patched.py` | Training script (MIOpen bypass) |
| `training_production.log` | Main training log |
| `training_monitor.log` | 5-minute health checks |
| `training_metrics.csv` | Extracted metrics (run `./extract_metrics.sh`) |
| `runs/detect/train2/` | Training outputs |
| `runs/detect/train2/weights/last.pt` | Latest checkpoint |
| `runs/detect/train2/weights/best.pt` | Best model |
| `runs/detect/train2/results.csv` | Per-epoch metrics |

## Expected Timeline

```
Epoch 1:    ~4.5 hours  (in progress)
Epoch 10:   ~2 days
Epoch 25:   ~5 days  
Epoch 50:   ~10 days (complete)
```

## Emergency Commands

### If GPU Too Hot (> 90°C)
```bash
# Force 100% fan immediately
echo 255 | sudo tee /sys/class/hwmon/hwmon3/pwm1
```

### If Training Crashes
```bash
# Check last 50 lines of log
tail -50 training_production.log

# Check for errors
grep -i "error\|exception\|killed" training_production.log | tail -10

# Resume from checkpoint (if needed)
source venv/bin/activate
python << 'PYTHON'
from ultralytics import YOLO
model = YOLO('runs/detect/train2/weights/last.pt')
model.train(resume=True)
PYTHON
```

### If System Freezes
```bash
# Before reboot: Save PID for investigation
echo $(ps aux | grep "[p]ython.*train" | awk '{print $2}') > crashed_pid.txt

# After reboot: Check what happened
dmesg | grep -i "killed\|oom\|gpu"
```

## Success Indicators

✅ **Training is HEALTHY if**:
- GPU temperature < 70°C
- GPU utilization > 80%
- Loss values decreasing
- No "error" in recent logs
- Process still running
- Fan speed 70-90%

❌ **INVESTIGATE if**:
- GPU temperature > 85°C
- GPU utilization < 50%
- Loss values increasing
- Errors in logs
- Process died
- Fan speed < 50%

## Current Status

```
Training:  ✅ RUNNING (19+ minutes)
GPU Temp:  ✅ 48°C edge, 56°C junction
GPU Util:  ✅ 82-99%
MIOpen:    ✅ No errors (bypassed)
Fan Speed: ✅ 74% (70% minimum)
Losses:    ✅ Decreasing
```

**Everything looks GREAT! 🚀**

## Contact Info

If you need to stop training:
```bash
# Find PID
cat .training_pid

# Stop gracefully (allows checkpoint save)
kill -SIGINT $(cat .training_pid)

# Force stop (only if frozen)
kill -9 $(cat .training_pid)
```

---

**Last Updated**: November 10, 2025, 10:52 AM  
**Status**: All systems operational ✅
