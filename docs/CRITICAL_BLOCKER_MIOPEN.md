# 🔴 CRITICAL BLOCKER: MIOpen Convolution Error

**Date**: November 9, 2025, 22:20  
**Status**: ❌ **TRAINING BLOCKED**  
**Severity**: CRITICAL

---

## 🔴 Issue Summary

Training fails immediately with MIOpen convolution error on RDNA1 GPU (RX 5600 XT) with ROCm 5.2.0 and PyTorch 1.13.1.

### Error Message
```
MIOpen Error: /MIOpen/src/ocl/convolutionocl.cpp:518: 
Forward Convolution cannot be executed due to incorrect params
RuntimeError: miopenStatusUnknownError
```

### Configuration Attempted
1. ✅ Batch size 16 - FAILED (system hang + crash)
2. ❌ Batch size 12 - FAILED (MIOpen error)
3. ❌ Batch size 8 - FAILED (MIOpen error)
4. ❌ Image size 416 - FAILED (MIOpen error)
5. ❌ MIOPEN_FIND_MODE=1 - FAILED
6. ❌ MIOPEN_FIND_ENFORCE=3 - FAILED

---

## 🔬 Root Cause Analysis

### Hardware/Software Stack
```
GPU: AMD Radeon RX 5600 XT (Navi 10, RDNA1)
ROCm: 5.2.0
PyTorch: 1.13.1+rocm5.2
MIOpen: ROCm 5.2 bundled
Python: 3.10.19
```

### Known Issues
1. **RDNA1 Limited Support**: RX 5600 XT (gfx1030) has incomplete ROCm support
2. **Missing MIOpen Kernels**: Warning seen: "Missing system database file: gfx1030_18.kdb"
3. **ROCm 5.2 Age**: Released 2022, newer versions (5.4+, 6.0+) have better RDN A1 support

---

## 🛠️ Attempted Solutions

### 1. MIOpen Find Mode
```bash
export MIOPEN_FIND_MODE=1
export MIOPEN_DEBUG_DISABLE_FIND_DB=1
```
**Result**: FAILED - Same error

### 2. Immediate Mode
```bash
export MIOPEN_FIND_ENFORCE=3
```
**Result**: FAILED - Same error

### 3. Smaller Batch/Image
```bash
batch=8, imgsz=416
```
**Result**: FAILED - Same error

---

## 🎯 Possible Solutions

### Option A: Upgrade ROCm (RECOMMENDED)
**Upgrade to ROCm 5.4+ or 6.0+**

**Pros**:
- Better RDNA1 support
- More complete MIOpen kernels
- Bug fixes for gfx1030

**Cons**:
- Requires system-wide upgrade
- May need PyTorch rebuild
- Compatibility risks
- ~2-4 hours work

**Steps**:
```bash
# 1. Remove ROCm 5.2
sudo apt remove rocm-*

# 2. Add ROCm 6.0 repository
wget https://repo.radeon.com/rocm/rocm.gpg.key -O - | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/6.0 ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# 3. Install ROCm 6.0
sudo apt update
sudo apt install rocm-hip-sdk

# 4. Rebuild venv with PyTorch for ROCm 6.0
python3.10 -m venv venv-py310-rocm60
source venv-py310-rocm60/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.0
pip install ultralytics
```

### Option B: Precompile MIOpen Kernels
**Manually build kernel database**

**Pros**:
- Keeps current ROCm version
- Targeted fix

**Cons**:
- Complex process
- May not resolve all issues
- Time-consuming (~1-2 hours)

**Steps**:
```bash
# 1. Install MIOpen development tools
sudo apt install miopen-hip-dev

# 2. Compile kernels for gfx1030
# (Complex - requires MIOpen source compilation)
```

### Option C: Use CPU Training (TEMPORARY)
**Train on CPU to verify pipeline**

**Pros**:
- Immediate solution
- Verifies data/model
- ~24 hours for 50 epochs estimate

**Cons**:
- VERY slow (50x slower)
- Defeats purpose of GPU training
- Not viable long-term

**Steps**:
```bash
yolo detect train \
  data=data/ltdv2_full/data.yaml \
  model=yolov8n.pt \
  epochs=5 \
  batch=4 \
  device=cpu \
  imgsz=640
```

### Option D: Use Different GPU
**Switch to RDNA2/3 or NVIDIA**

**Pros**:
- Guaranteed to work
- Better ML support

**Cons**:
- Requires hardware
- Not solving RDNA1 problem

---

## 📊 Recommendation Matrix

| Solution | Time | Risk | Success Rate | Recommended |
|----------|------|------|--------------|-------------|
| Upgrade ROCm 6.0 | 2-4h | Medium | 70% | ⭐⭐⭐ YES |
| Precompile Kernels | 1-2h | High | 40% | ⭕ Maybe |
| CPU Training | 0h | Low | 100% (slow) | ⭕ Temporary |
| New GPU | N/A | Low | 100% | ❌ Not feasible |

**RECOMMENDED**: **Option A - Upgrade to ROCm 6.0**

---

## 🚀 Next Steps (Recommended Path)

### Immediate (Tonight)
1. [ ] Decision: Upgrade ROCm or try CPU training?
2. [ ] If upgrade: Back up current environment
3. [ ] If CPU: Start 5-epoch CPU test

### If Upgrading ROCm 6.0 (Tomorrow)
1. [ ] Create system restore point
2. [ ] Back up /opt/rocm and venv
3. [ ] Remove ROCm 5.2
4. [ ] Install ROCm 6.0
5. [ ] Create new venv with PyTorch ROCm 6.0
6. [ ] Test GPU detection
7. [ ] Test simple PyTorch model
8. [ ] Retry YOLOv8 training

---

## 📝 Historical Context

**What Worked Before (Nov 6-7)**:
- ✅ Test runs with 3 epochs
- ✅ Data pipeline validation
- ✅ Model loading
- ✅ Brief training runs

**What Changed**:
- Longer training duration attempt
- Different batch sizes
- System under sustained load

**Hypothesis**: Brief test runs succeeded because they didn't stress MIOpen enough to trigger the kernel compilation/execution error. Longer runs expose the incomplete gfx1030 support.

---

## 🔗 References

- ROCm RDNA1 Support: https://github.com/ROCmSoftwarePlatform/ROCm/issues
- MIOpen gfx1030: https://github.com/ROCmSoftwarePlatform/MIOpen/issues
- PyTorch ROCm: https://pytorch.org/get-started/locally/

---

**Status**: 🔴 **BLOCKED - AWAITING DECISION**  
**Blocker**: MIOpen convolution error on RDNA1  
**Impact**: Cannot train YOLOv8 on current GPU/ROCm setup  
**Severity**: Critical - Prevents all training

*Created: November 9, 2025, 22:20*  
*Owner: System Administrator*  
*Priority: P0 - Critical Blocker*
