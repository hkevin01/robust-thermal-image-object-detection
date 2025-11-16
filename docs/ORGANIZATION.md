# Project Organization

## Recent Cleanup (Nov 16, 2025)

### Changes Made

1. **Created Archive Structure**
   - `archive/training_scripts/` - Old training script versions
   - `archive/docs/` - Old documentation
   - `archive/pid_files/` - Old process IDs

2. **Moved Files**
   - Training scripts (v1-v7) → `archive/training_scripts/`
   - Logs → `logs/archive/`
   - Documentation → `docs/`
   - Shell scripts → `scripts/`
   - Model files → `models/`
   - Backup files → `archive/docs/`

3. **Current Active Files**
   - `train_v7_final_working.py` - Working training script
   - `training.pid` - Active process ID
   - `current_training.log` - Symlink to active log
   - `logs/archive/training_WORKERS0_20251116_153618.log` - Current log

## Directory Structure

```
/
├── archive/          # Historical files
│   ├── training_scripts/
│   ├── docs/
│   └── pid_files/
├── configs/          # Configuration files
├── data/            # Dataset
├── docs/            # Documentation
├── logs/            # Training logs
│   └── archive/     # Old logs
├── models/          # Model files
│   └── pretrained/  # Pretrained weights
├── patches/         # Custom patches
├── scripts/         # Shell/utility scripts
├── src/             # Source code
├── tests/           # Unit tests
└── runs/            # Training outputs
```

## File Naming Conventions

### Training Scripts
- Current stable: `train_<name>.py` (root or scripts/)
- Versions: Archive immediately after superseding
- Experimental: `scripts/training/experimental/`

### Logs
- Format: `training_<NAME>_YYYYMMDD_HHMMSS.log`
- Location: `logs/`
- Archive old logs to: `logs/archive/`

### Documentation
- User docs: `docs/<topic>.md`
- Development: `docs/development/<topic>.md`
- Research: `docs/research/<topic>.md`

## Maintenance

### Weekly Cleanup
1. Archive old training logs
2. Remove temporary files
3. Update documentation
4. Check for unused scripts

### Before Commits
1. Verify root directory is clean
2. Ensure all files in correct subdirectories
3. Update relevant documentation
4. Remove debugging artifacts
