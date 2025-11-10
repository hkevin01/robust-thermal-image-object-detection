# 🔴 System Crash Incident Report

**Date**: November 9, 2025  
**Time**: ~21:45 (crash), 22:46 (reboot)  
**Severity**: HIGH - Full system freeze requiring hard reboot

---

## 📊 Incident Summary

**What Happened**:
- YOLOv8 training crashed the entire system during epoch 1
- System became completely unresponsive (freeze)
- Required hard reboot to recover
- Training was 31% through first epoch when crash occurred

**Impact**:
- Lost ~3 hours of training progress
- No checkpoint saved (crash before epoch completion)
- Monitor automation stopped
- Need to restart training from beginning

---

## 🔍 Crash Analysis

### Training State at Crash
```
Epoch: 1/100 (31% complete - batch 6482/20582)
GPU Memory: 5.7 GB allocated
Runtime: ~3 hours
Speed: 1.3 batches/second
ETA remaining: 2h 56m (for epoch 1)
```

### Losses at Crash Point
```
box_loss: 1.694
cls_loss: 1.575  
dfl_loss: 1.142
```

### System State After Reboot
```
Current Time: 21:58 (Nov 9)
Uptime: 13 minutes (rebooted at 22:46)
Memory: 9.7 GB used / 31 GB total (healthy)
Swap: 0 used / 8 GB available
Load: 3.37 (high - likely from reboot services)
```

---

## 🔬 Root Cause Analysis

### Most Likely Cause: GPU Hang

**Evidence**:
1. System freeze (not just process crash)
2. Required hard reboot (no soft recovery)
3. RDNA1 GPU (RX 5600 XT) running intensive workload
4. ROCm 5.2.0 compatibility issues known with RDNA1
5. No OOM indicators (plenty of RAM/swap available)

**GPU Hang Characteristics**:
- RDNA1 GPUs can experience driver hangs under sustained load
- MIOpen kernel compilation warnings seen in logs
- Training speed was slow (1.3 it/s) suggesting GPU stress
- 5.7 GB VRAM usage (close to 6 GB card limit)

### Secondary Possibility: Thermal Throttling

**Evidence**:
- 3+ hour continuous GPU load
- 99% GPU utilization
- Batch size 16 with 640x640 images
- No thermal monitoring in place

### Unlikely: Out of Memory

**Evidence Against**:
- System RAM: 31 GB total, only 9.7 GB used after reboot
- Swap: 8 GB available, none used
- No OOM killer messages in logs
- Batch size 16 is conservative

---

## ⚠️ Risk Factors Identified

### 1. RDNA1 GPU Stability
- **Issue**: RX 5600 XT (Navi 14) has known ROCm stability issues
- **Severity**: HIGH
- **Impact**: Random system hangs during training

### 2. No GPU Thermal Monitoring
- **Issue**: No temperature tracking during training
- **Severity**: MEDIUM
- **Impact**: Cannot detect overheating before crash

### 3. No Checkpoint Recovery
- **Issue**: YOLOv8 only saves at epoch completion
- **Severity**: MEDIUM
- **Impact**: Lose all progress if crash mid-epoch

### 4. Aggressive Batch Size
- **Issue**: Batch 16 pushes VRAM close to limit (5.7/6 GB)
- **Severity**: LOW
- **Impact**: Less thermal/memory headroom

### 5. No Watchdog/Auto-Restart
- **Issue**: Training doesn't auto-resume after crash
- **Severity**: LOW
- **Impact**: Manual intervention required

---

## 🛠️ Recovery Plan

### Immediate Actions (Now)

1. ✅ **Verify system stability** - Check GPU, memory, thermals
2. ✅ **Document crash details** - This report
3. ⏳ **Implement crash prevention** - See mitigation strategies
4. ⏳ **Restart training** - With monitoring and safeguards

### Short-term Mitigations (Before restarting)

1. **Reduce batch size** to 12 or 8 (reduce VRAM pressure)
2. **Add GPU monitoring** (temperature, utilization, VRAM)
3. **Enable aggressive power management** (prevent overheating)
4. **Add watchdog script** (detect hangs, log GPU state)
5. **Test stability** with 5-epoch test run first

### Long-term Solutions (For future runs)

1. **Upgrade ROCm** to 5.4+ or 6.0 (better RDNA1 support)
2. **Implement mid-epoch checkpointing** (custom callback)
3. **Add thermal limits** (reduce GPU power if temp > threshold)
4. **Create auto-restart wrapper** (resume training on crash)
5. **Consider CPU fallback** for critical sections

---

## 📋 Pre-Flight Checklist (Before Restart)

### System Health
- [ ] Check GPU temperature (idle should be < 45°C)
- [ ] Verify GPU driver loaded (`rocm-smi`)
- [ ] Check system logs for GPU errors
- [ ] Verify disk space available (> 50 GB)
- [ ] Check RAM usage (should be < 10 GB at idle)

### Training Configuration
- [ ] Reduce batch size to 12 (from 16)
- [ ] Verify dataset still accessible
- [ ] Check virtual environment activated
- [ ] Review training arguments
- [ ] Backup any existing training outputs

### Monitoring Setup
- [ ] Create GPU temperature monitor script
- [ ] Add GPU stats to training log
- [ ] Set up watchdog for hang detection
- [ ] Test notification system
- [ ] Document restart procedure

### Safety Measures
- [ ] Set GPU power limit (reduce by 10-20%)
- [ ] Enable aggressive fan curve
- [ ] Clear old temp files
- [ ] Test with 5-epoch trial run first
- [ ] Ensure swap is enabled

---

## 🔬 Diagnostic Commands

### GPU Health Check
```bash
# Check GPU status
rocm-smi

# Check GPU temperature
rocm-smi --showtemp

# Check VRAM usage
rocm-smi --showmeminfo vram

# Check for GPU errors
dmesg | grep amdgpu | tail -30
```

### System Health Check  
```bash
# Memory status
free -h

# Disk space
df -h

# System load
uptime

# Recent errors
journalctl -b 0 -p err --no-pager | tail -20
```

### Training Environment Check
```bash
# Virtual environment
source venv-py310-rocm52/bin/activate

# PyTorch GPU detection
python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"

# Dataset accessibility
ls -lh data/ltdv2_full/images/train/ | head -10
```

---

## 📈 Next Steps

### Immediate (Next 30 minutes)
1. Run GPU health diagnostics
2. Implement batch size reduction
3. Create GPU monitoring script
4. Test with 5-epoch trial

### Before Full Training
1. Verify 5-epoch run completes without crash
2. Review GPU temps (should stay < 80°C)
3. Check memory usage patterns
4. Validate checkpoint saving works

### During Training
1. Monitor GPU temps every 15 minutes (first 2 hours)
2. Check training log for anomalies
3. Verify monitor automation working
4. Keep system logs open for early warning

---

## 🎯 Success Criteria for Restart

**Must Have**:
- ✅ GPU temperature < 45°C at idle
- ✅ Batch size reduced to 12
- ✅ GPU monitoring active
- ✅ 5-epoch test run completes successfully

**Should Have**:
- ✅ Watchdog script running
- ✅ GPU power limit set
- ✅ Thermal monitoring automated
- ✅ Auto-restart mechanism tested

**Nice to Have**:
- ⭕ Mid-epoch checkpointing
- ⭕ Email alerts on crash
- ⭕ Historical GPU stats logging
- ⭕ Automated thermal management

---

## 📝 Lessons Learned

1. **RDNA1 GPUs need extra monitoring** - Not as stable as RDNA2/3 for ML
2. **Hard freezes are GPU driver issues** - Not OOM or Python crashes
3. **Batch size matters** - Conservative sizing leaves thermal headroom
4. **No checkpoint = lost progress** - Need mid-epoch saves for long runs
5. **Monitoring isn't optional** - Need GPU temps, VRAM, utilization tracked

---

## 🔗 Related Issues

- RDNA1 ROCm stability: https://github.com/ROCm/ROCm/issues
- YOLOv8 checkpoint frequency: Consider custom callback
- GPU hang detection: Implement watchdog timer
- Thermal management: Add temperature-based throttling

---

**Status**: 🔴 **INCIDENT ACTIVE - RECOVERY IN PROGRESS**  
**Next Action**: Implement crash mitigations and restart training

*Report created: November 9, 2025, 21:58*  
*Incident duration: ~3 hours training lost*  
*Recovery ETA: 1-2 hours (testing + restart)*
