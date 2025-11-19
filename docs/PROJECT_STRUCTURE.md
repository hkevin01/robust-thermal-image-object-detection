# 🏗️ Project Structure

## Root Directory (Clean & Organized)

```
robust-thermal-image-object-detection/
│
├── README.md                          # Main project documentation
├── LICENSE                            # Project license
├── requirements.txt                   # Python dependencies
├── setup.py                          # Package setup configuration
├── train_v7_final_working.py         # Main training script (STRENGTHENED v2)
│
├── check_training.sh                 # Symlink → scripts/check_fresh_training.sh
├── monitor.sh                        # Symlink → scripts/monitor_training.sh
│
├── .github/                          # GitHub configuration
│   ├── copilot-instructions.md       # AI assistant instructions
│   └── workflows/                    # CI/CD pipelines
│
├── archive/                          # Archived/backup files
│   └── venv_backups/                # Old virtual environments
│
├── assets/                           # Project assets
│   └── models/                      # Pre-trained model weights
│
├── configs/                          # Configuration files
│   ├── training_config.yaml         # Training parameters
│   └── model_config.yaml            # Model architecture
│
├── data/                             # Dataset location (gitignored)
│   └── ltdv2_full/                  # LTDv2 dataset
│
├── docs/                             # 📚 ALL DOCUMENTATION HERE
│   ├── README.md                    # Documentation hub
│   ├── PROJECT_STRUCTURE.md         # This file
│   ├── STATUS_FILES_INDEX.md        # Historical status files
│   │
│   ├── COMPETITION_SUBMISSION_GUIDE.md
│   ├── SUBMISSION_WORKFLOW.md
│   ├── SUBMISSION_CHECKLIST.md
│   ├── MEMORY_BANK.md
│   ├── QUICK_REFERENCE.md
│   │
│   └── [Historical status files...]
│
├── logs/                             # Training logs (gitignored)
│   ├── training_FRESH_START_*.log   # Current training logs
│   ├── QUICK_STATUS.txt             # Quick status snapshots
│   └── training.pid                 # Process ID files
│
├── memory-bank/                      # Long-term memory/notes
│
├── models/                           # Trained model outputs
│
├── patches/                          # Custom code patches
│   ├── conv2d_optimized.py          # ROCm Conv2d optimization
│   └── rocm_fix/                    # ROCm-specific fixes
│
├── runs/                             # Training run outputs (gitignored)
│   └── detect/
│       └── train_v7_final_working/  # Current training run
│           └── weights/             # Model checkpoints
│
├── scripts/                          # 🔧 ALL SCRIPTS HERE
│   ├── check_fresh_training.sh      # Check training status
│   ├── monitor_training.sh          # Monitor training progress
│   ├── monitor_epoch4.sh            # Epoch-specific monitoring
│   └── START_TRAINING.sh            # Start training wrapper
│
├── src/                              # Source code
│   ├── data/                        # Data loading utilities
│   ├── models/                      # Model definitions
│   ├── training/                    # Training utilities
│   └── utils/                       # Helper functions
│
├── tests/                            # Unit tests
│   ├── test_data.py
│   ├── test_model.py
│   └── test_training.py
│
├── venv/                             # Virtual environment (gitignored, symlink)
│
└── YOLOv8/                           # YOLOv8 repository (if cloned)
```

## Directory Purposes

### 📂 Configuration & Core
- **Root**: Only essential files (README, requirements, main scripts)
- **.github/**: GitHub-specific configuration
- **configs/**: YAML/JSON configuration files

### 📚 Documentation
- **docs/**: ALL documentation, guides, and historical records
  - Active guides and references
  - Historical status files
  - API documentation
  - Troubleshooting guides

### 💻 Code
- **src/**: All source code, organized by function
- **patches/**: Custom patches for ROCm/GPU compatibility
- **tests/**: Unit and integration tests

### 🔧 Utilities
- **scripts/**: ALL executable scripts (.sh files)
  - Training scripts
  - Monitoring scripts
  - Utility scripts
  - Symlinks in root for convenience

### 📊 Data & Outputs
- **data/**: Datasets (gitignored, large)
- **logs/**: Training logs and status files
- **runs/**: Training outputs and checkpoints
- **models/**: Final trained models

### 🗄️ Storage
- **archive/**: Old files, backups, deprecated code
- **memory-bank/**: Long-term notes and knowledge base

## File Naming Conventions

### Documentation (docs/)
- `UPPERCASE_WITH_UNDERSCORES.md` - Major documents
- `lowercase-with-dashes.md` - Specific guides
- Descriptive names (not generic)

### Scripts (scripts/)
- `action_description.sh` - Shell scripts
- `verb_noun.py` - Python scripts
- Executable permissions: `chmod +x`

### Logs (logs/)
- `training_DESCRIPTION_TIMESTAMP.log` - Training logs
- `COMPONENT_status.txt` - Status snapshots
- `*.pid` - Process ID files

## Best Practices

### ✅ DO:
- Create new docs in `docs/`
- Create new scripts in `scripts/`
- Use symlinks in root for frequently used scripts
- Keep root directory minimal and clean
- Archive old files instead of deleting
- Use descriptive names

### ❌ DON'T:
- Create `.md` files in root (except README.md)
- Create `.sh` files in root
- Leave temporary files in root
- Commit large datasets or model weights
- Duplicate documentation
- Use generic names (STATUS.md, temp.txt)

## Quick Commands

```bash
# Check what's in root
ls -1 | grep -v "^\."

# Find all markdown files
find . -name "*.md" -type f

# Find all scripts
find scripts/ -name "*.sh" -type f

# Check for root clutter
ls *.md *.sh *.txt *.pid 2>/dev/null

# Clean up (if needed)
# Move docs: mv *.md docs/
# Move scripts: mv *.sh scripts/
# Clean temp: rm -f *.pid *.txt
```

## Maintenance

### Weekly
- Review new files in root
- Move misplaced files to correct locations
- Clean up temporary files

### Monthly
- Archive old logs
- Review docs for updates needed
- Check for duplicate content

### Quarterly
- Full structure audit
- Update this document
- Archive inactive experiments

---

**Last Updated**: November 18, 2025
**Maintained By**: Project team
**Questions?**: See [docs/README.md](README.md)
