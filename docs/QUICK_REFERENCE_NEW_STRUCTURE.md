# 🗂️ Quick Reference: New Project Structure

## Quick Commands

### Check Training
```bash
./check_training.sh
# or
scripts/check_fresh_training.sh
```

### Monitor Training
```bash
./monitor.sh
# or
scripts/monitor_training.sh
```

### View Logs
```bash
tail -f logs/training_FRESH_START_*.log
```

## Where to Put Files

| File Type | Location | Example |
|-----------|----------|---------|
| **Documentation** | `docs/` | `docs/MY_GUIDE.md` |
| **Scripts** | `scripts/` | `scripts/my_script.sh` |
| **Status Updates** | README.md Changelog | Edit main README |
| **Temp/Log files** | `logs/` | `logs/status.txt` |
| **Source code** | `src/` | `src/utils/helper.py` |
| **Tests** | `tests/` | `tests/test_model.py` |
| **Configs** | `configs/` | `configs/train.yaml` |
| **Old files** | `archive/` | `archive/old_script.sh` |

## File Naming Conventions

```bash
# Documentation (in docs/)
docs/MAJOR_DOCUMENT.md           # Major guides
docs/specific-guide.md           # Specific guides
docs/troubleshooting-gpu.md      # Descriptive names

# Scripts (in scripts/)
scripts/action_target.sh         # verb_noun format
scripts/monitor_training.sh      # What it does
scripts/check_status.sh          # Clear purpose

# Logs (in logs/)
logs/training_TIMESTAMP.log      # Training logs
logs/component_status.txt        # Status snapshots
```

## Common Tasks

### Create New Documentation
```bash
# Create in docs/
nano docs/NEW_GUIDE.md

# Update docs index
nano docs/README.md

# Link from main README if important
nano README.md
```

### Create New Script
```bash
# Create in scripts/
nano scripts/new_task.sh
chmod +x scripts/new_task.sh

# Test it
scripts/new_task.sh

# Optional: Create symlink for convenience
ln -sf scripts/new_task.sh ./new_task.sh
```

### Add Status Update
```bash
# DON'T create new .md file
# Instead, update README Changelog
nano README.md
# Find: ## 📝 Changelog & Development Timeline
# Add entry under appropriate date
```

### Check for Clutter
```bash
# Should only show symlinks
ls *.sh

# Should be empty (no .md except README)
ls *.md | grep -v README

# Should be empty (no temp files)
ls *.pid *.txt 2>/dev/null
```

## Directory Quick Reference

```
Root/
├── docs/              # 📚 All documentation
├── scripts/           # 🔧 All scripts
├── logs/              # 📝 All logs
├── src/               # 💻 Source code
├── tests/             # 🧪 Tests
├── configs/           # ⚙️ Configurations
├── archive/           # 📦 Old files
├── data/              # 📊 Datasets
└── runs/              # 🏃 Training outputs
```

## Help

- **Full structure**: See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)
- **Documentation hub**: See [docs/README.md](README.md)
- **Cleanup details**: See [CLEANUP_SUMMARY_20251118.md](CLEANUP_SUMMARY_20251118.md)
- **Main README**: See [../README.md](../README.md)

---

**Pro Tip**: Use tab completion with the symlinks:
```bash
./che<TAB>    # Expands to ./check_training.sh
./mon<TAB>    # Expands to ./monitor.sh
```
