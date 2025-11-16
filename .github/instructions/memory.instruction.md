---
applyTo: '**'
---

# Project Organization Rules

## File Organization Principles

**ALWAYS keep the root directory clean and organized.**

### Directory Structure

1. **Training Scripts** → `scripts/training/`
   - Current active: Keep in root OR scripts/
   - Old versions: Move to `archive/training_scripts/`

2. **Logs** → `logs/`
   - Current training log: `logs/` (with symlink from root)
   - Old logs: `logs/archive/`

3. **Documentation** → `docs/`
   - User docs: `docs/`
   - Dev notes/analysis: `docs/development/`
   - API docs: `docs/api/`

4. **Shell Scripts** → `scripts/`
   - Training scripts: `scripts/training/`
   - Utility scripts: `scripts/utils/`
   - Monitoring: `scripts/monitoring/`

5. **Models** → `models/`
   - Pretrained: `models/pretrained/`
   - Checkpoints: `runs/` (Ultralytics default)

6. **Configuration** → `configs/`
   - Training configs: `configs/training/`
   - Data configs: `configs/data/`

7. **Source Code** → `src/`
   - Core modules: `src/core/`
   - Utilities: `src/utils/`
   - Data processing: `src/data/`

8. **Patches/Fixes** → `patches/`
   - ROCm fixes: `patches/rocm_fix/`
   - Custom implementations: `patches/`

9. **Tests** → `tests/`
   - Unit tests: `tests/unit/`
   - Integration tests: `tests/integration/`

10. **Archive** → `archive/`
    - Old scripts: `archive/training_scripts/`
    - Old docs: `archive/docs/`
    - Old logs: Move to `logs/archive/`

### Root Directory - What Should Be There

**Keep in root:**
- `README.md` - Main project documentation
- `requirements.txt` - Python dependencies
- `setup.py` - Package setup
- `training.pid` - Current process ID
- `current_training.log` - Symlink to active log
- `train_v7_final_working.py` - Current stable training script (or move to scripts/)
- Core directories: `src/`, `tests/`, `docs/`, `scripts/`, `configs/`, etc.

**DO NOT keep in root:**
- ❌ Multiple versions of scripts (train_v1.py, train_v2.py, etc.)
- ❌ Old log files (training_*.log)
- ❌ Temporary files (.pid, .log except current)
- ❌ Backup files (*.backup, *.old)
- ❌ Analysis documents (move to docs/)

## Training Script Workflow

### Current Training Setup
- **Active Script**: `train_v7_final_working.py`
- **Configuration**: workers=0, AMP disabled, Conv2d patched
- **Log**: `logs/archive/training_WORKERS0_20251116_153618.log`
- **Checkpoint**: `runs/detect/train_optimized_v5_multiprocess/weights/last.pt`

### Key Fixes Implemented
1. **Workers=0**: Force single-threaded DataLoader (ROCm compatibility)
   - Patched `BaseTrainer.__init__` to override checkpoint config
   - Patched `build_dataloader` to force workers=0
2. **AMP Check Bypass**: Patched `checks.check_amp()` to return False
3. **Conv2d Patches**: Custom im2col+rocBLAS implementation for MIOpen compatibility

### When Creating New Files

1. **New training script**: 
   - If experimental: `scripts/training/experimental/`
   - If stable version: `scripts/training/` or root
   - Archive old version: `archive/training_scripts/`

2. **New documentation**:
   - User-facing: `docs/`
   - Development notes: `docs/development/`
   - Research/analysis: `docs/research/`

3. **New logs**:
   - Always in `logs/`
   - Archive old ones: `logs/archive/`

4. **New scripts**:
   - Shell scripts: `scripts/utils/` or `scripts/training/`
   - Python utilities: `src/utils/`

## Cleanup Checklist

Before committing changes, ensure:
- [ ] No loose files in root (except allowed ones)
- [ ] All logs in `logs/` or `logs/archive/`
- [ ] Old script versions in `archive/training_scripts/`
- [ ] Documentation in `docs/`
- [ ] Shell scripts in `scripts/`
- [ ] Model files in `models/`

## Hardware Configuration

- **GPU**: AMD Radeon RX 5600 XT (gfx1010)
- **ROCm**: 5.2
- **PyTorch**: 1.13.1+rocm5.2
- **Python**: 3.10.19 (in `./venv/bin/python`)
- **Issue**: MIOpen kernel database missing → custom Conv2d implementation
- **Solution**: workers=0 for DataLoader stability

## Training Timeline

- **Deadline**: November 30, 2025
- **Current**: Epoch 2/50 (49% complete as of Nov 16, 3:40 PM)
- **Expected Completion**: Nov 24-25
- **Buffer**: 5-6 days
