# ROCm 5.7.1 + PyTorch 2.2.2 Downgrade - SUCCESS

**Date**: January 8, 2025  
**Status**: ✅ COMPLETE  
**GPU Detection**: ✅ WORKING

---

## Summary

Successfully downgraded from **ROCm 6.2 + PyTorch 2.5.1** to **ROCm 5.7.1 + PyTorch 2.2.2** to resolve the `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` bug affecting RDNA1 GPUs (RX 5600 XT).

---

## Installed Versions

### Software Stack
- **ROCm**: 5.7.1 (from October 13, 2023)
- **PyTorch**: 2.2.2+rocm5.7
- **TorchVision**: 0.17.2+rocm5.7
- **TorchAudio**: 2.2.2+rocm5.7
- **NumPy**: 1.26.4 (downgraded from 2.2.6 for compatibility)
- **Ultralytics**: 8.3.225 (unchanged)

### Hardware
- **GPU**: AMD Radeon RX 5600 XT (gfx1010, RDNA1)
- **VRAM**: 6.43 GB
- **CPU**: AMD Ryzen 5 3600 6-Core Processor
- **RAM**: 31.3 GB
- **OS**: Ubuntu 24.04.3 LTS (noble), kernel 6.14.0-35-generic

---

## Installation Details

### Repository
- **Source**: https://repo.radeon.com/rocm/apt/5.7.1
- **Ubuntu Base**: jammy (22.04) - used on noble (24.04) via compatibility
- **GPG Key**: /usr/share/keyrings/rocm.gpg

### Packages Installed
54 ROCm packages installed including:
- rocm-hip-sdk
- rocm-libs
- rocm-llvm (17.0.0)
- comgr, hsa-rocr, hipcc, rocblas, miopen-hip, rccl
- Development headers and libraries

---

## Verification Results

### GPU Detection Test
```bash
$ python3 -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
2.2.2+rocm5.7
True
AMD Radeon RX 5600 XT
```

### ROCm Info
```bash
$ rocminfo | grep "Marketing Name"
  Marketing Name:          AMD Ryzen 5 3600 6-Core Processor  
  Marketing Name:          AMD Radeon RX 5600 XT
```

✅ **Both CPU and GPU agents detected**

---

## Known Issues & Workarounds

### 1. NumPy Compatibility Warning
**Issue**: PyTorch 2.2.2 was compiled against NumPy 1.x, incompatible with NumPy 2.x  
**Solution**: Downgraded to numpy<2 (1.26.4)  
**Status**: RESOLVED

### 2. Dependency Conflicts (Non-blocking)
- `hyper-connections 0.2.1` requires torch>=2.3 (we have 2.2.2)
- `monai 1.5.0` requires torch<2.7.0,>=2.4.1 (we have 2.2.2)
- **Impact**: These packages won't work, but don't affect YOLO training
- **Action**: No action needed unless those packages are required

### 3. Ubuntu 24.04 Compatibility
**Issue**: ROCm 5.7.1 was built for Ubuntu 22.04 (jammy), not 24.04 (noble)  
**Workaround**: Using jammy repository on noble system  
**Status**: WORKING (no issues observed)

---

## Next Steps

### 1. Test GPU Training (REQUIRED)
Before starting full training, test with a 1-epoch run:

```bash
cd /home/kevin/Projects/robust-thermal-image-object-detection
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=1 \
  batch=4 \
  device=0 \
  name=rocm57_test
```

**Expected Outcome**: Training should complete without memory access violations

### 2. Start Baseline Training
If test succeeds, start full training:

```bash
nohup yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8m.pt \
  epochs=100 \
  batch=8 \
  device=0 \
  name=baseline_yolov8m_rocm57 \
  > training_rocm57.log 2>&1 &
```

### 3. Monitor Training
```bash
# Check process
ps aux | grep yolo

# Watch log
tail -f training_rocm57.log

# Check GPU usage
watch -n 1 rocm-smi
```

---

## Rollback Instructions

If ROCm 5.7.1 doesn't work, rollback to previous state:

```bash
# Remove ROCm 5.7.1
sudo apt remove rocm-* hip-* hsa-* -y
sudo apt autoremove -y
sudo rm /etc/apt/sources.list.d/rocm.list

# Reinstall PyTorch 2.5.1+rocm6.2
pip3 uninstall torch torchvision torchaudio -y --break-system-packages
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --break-system-packages
```

---

## Troubleshooting

### GPU Not Detected After Reboot
```bash
# Check ROCm installation
rocminfo | grep "Marketing Name"

# Verify PyTorch can see GPU
python3 -c "import torch; print(torch.cuda.is_available())"

# Check kernel module
lsmod | grep amdgpu

# Reload amdgpu if needed
sudo modprobe -r amdgpu
sudo modprobe amdgpu
```

### Training Still Crashes
If crashes persist with ROCm 5.7.1:

1. Try smaller batch sizes: 8 → 4 → 2 → 1
2. Try YOLOv8n instead of YOLOv8m
3. Check system logs: `journalctl -f | grep -i amd`
4. Consider cloud GPU alternative (if local GPU truly incompatible)

---

## Files Modified/Created

### Created:
- `downgrade_rocm57.sh` - Automated downgrade script
- `ROCM_DOWNGRADE_SUCCESS.md` - This document

### Modified:
- Python packages: torch, torchvision, torchaudio, numpy
- System packages: 54 ROCm packages installed
- APT sources: `/etc/apt/sources.list.d/rocm.list`

### Preserved:
- Dataset: `data/ltdv2_full/` (420,033 images)
- Configuration: `configs/baseline.yaml`
- Git repository: Clean, no changes
- Previous training runs: `runs/detect/*`

---

## Performance Expectations

### Training Speed Estimates (YOLOv8m, batch=8)
Based on successful RDNA1 users with ROCm 5.7:

- **Iterations/sec**: ~2-4 it/s (GPU)
- **Time/epoch**: ~45-90 minutes
- **Total time (100 epochs)**: ~3-6 days
- **VRAM usage**: ~5.5 GB (within 6.43 GB limit)

Compare to previous attempts:
- ROCm 6.2: Immediate crash ❌
- CPU: 0.6 it/s (~8 days for 10 epochs) ❌
- ROCm 5.7: TBD (expecting success ✅)

---

## References

### ROCm 5.7.1
- **Release Date**: October 13, 2023
- **Repository**: https://repo.radeon.com/rocm/apt/5.7.1
- **Supported Ubuntu**: 20.04 (focal), 22.04 (jammy)
- **Status**: Last stable ROCm for RDNA1

### PyTorch 2.2.2
- **Release Date**: March 21, 2024
- **ROCm Support**: 5.6, 5.7
- **Wheels**: https://download.pytorch.org/whl/rocm5.7/
- **Compatibility**: Python 3.8-3.12

### Known Bug (ROCm 6.2+)
- **GitHub Issue**: ROCm/ROCm#5051
- **Affects**: RDNA1/2 GPUs (RX 5000/6000 series)
- **Error**: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
- **Root Cause**: Hardware SVM coherency bug
- **Status**: Under investigation, no ETA for fix

---

## Success Criteria

### ✅ Installation Complete
- ROCm 5.7.1 installed
- PyTorch 2.2.2+rocm5.7 installed
- GPU detected by PyTorch
- rocminfo shows GPU

### ⏳ Pending Validation
- [ ] 1-epoch GPU test completes
- [ ] Full training runs without crashes
- [ ] Final model achieves acceptable mAP

---

## Contact & Support

### Project Repository
- **GitHub**: github.com/hkevin01/robust-thermal-image-object-detection
- **Branch**: main
- **Status**: Clean, up to date

### ROCm Resources
- **Documentation**: https://rocm.docs.amd.com/
- **GitHub**: https://github.com/ROCm/ROCm
- **Community**: https://github.com/orgs/ROCm/discussions

### PyTorch Resources
- **Documentation**: https://pytorch.org/docs/
- **ROCm Support**: https://pytorch.org/get-started/locally/
- **Forums**: https://discuss.pytorch.org/

---

**End of Report**

Generated: January 8, 2025  
System: Ubuntu 24.04.3 LTS  
User: kevin  
Project: robust-thermal-image-object-detection
