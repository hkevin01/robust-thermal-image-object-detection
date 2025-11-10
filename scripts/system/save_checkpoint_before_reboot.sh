#!/bin/bash
# Checkpoint Save Script - Before Reboot
# Date: $(date)

cd ~/Projects/robust-thermal-image-object-detection

echo "=== Saving Training Checkpoint Before Reboot ==="
echo "Date: $(date)"
echo ""

# Create checkpoint backup directory
CHECKPOINT_DIR="checkpoints/pre_reboot_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$CHECKPOINT_DIR"

# 1. Check for running processes
echo "1. Checking for running training processes..."
RUNNING_PROCS=$(ps aux | grep -E "yolo|train|python.*train" | grep -v grep)
if [ -n "$RUNNING_PROCS" ]; then
    echo "⚠️  WARNING: Training processes detected!"
    echo "$RUNNING_PROCS"
    echo ""
    echo "Attempting graceful shutdown..."
    pkill -SIGTERM -f "yolo.*train"
    pkill -SIGTERM -f "python.*train"
    sleep 5
    
    # Force kill if still running
    pkill -SIGKILL -f "yolo.*train"
    pkill -SIGKILL -f "python.*train"
    echo "✓ Processes terminated"
else
    echo "✓ No training processes running"
fi
echo ""

# 2. Copy all model checkpoints
echo "2. Backing up model checkpoints..."
if [ -d "runs/train" ]; then
    cp -r runs/train "$CHECKPOINT_DIR/"
    echo "✓ Copied runs/train/ to $CHECKPOINT_DIR/"
fi

# Copy pretrained models
cp -f yolov8*.pt "$CHECKPOINT_DIR/" 2>/dev/null || true
echo "✓ Backed up pretrained models"
echo ""

# 3. Save training logs
echo "3. Backing up training logs..."
cp -f *.log "$CHECKPOINT_DIR/" 2>/dev/null || true
echo "✓ Backed up log files"
echo ""

# 4. Save ROCm patches
echo "4. Backing up ROCm patches..."
if [ -d "patches" ]; then
    cp -r patches "$CHECKPOINT_DIR/"
    echo "✓ Backed up patches directory"
fi
echo ""

# 5. Save current git state
echo "5. Saving git state..."
git status > "$CHECKPOINT_DIR/git_status.txt" 2>&1
git log --oneline -10 > "$CHECKPOINT_DIR/git_log.txt" 2>&1
git diff > "$CHECKPOINT_DIR/git_diff.txt" 2>&1
echo "✓ Saved git state"
echo ""

# 6. Save system info
echo "6. Saving system information..."
cat > "$CHECKPOINT_DIR/system_info.txt" << SYSINFO
=== System Information ===
Date: $(date)
Hostname: $(hostname)
Kernel: $(uname -r)
ROCm: $(rocminfo --version 2>/dev/null || echo "N/A")
GPU: $(rocm-smi --showproductname 2>/dev/null || echo "N/A")
PyTorch: $(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "N/A")
CUDA Available: $(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "N/A")
Disk Usage:
$(df -h .)
Memory:
$(free -h)
SYSINFO
echo "✓ Saved system info"
echo ""

# 7. Create checkpoint manifest
echo "7. Creating checkpoint manifest..."
cat > "$CHECKPOINT_DIR/MANIFEST.md" << MANIFEST
# Training Checkpoint - Pre-Reboot

**Created**: $(date)
**Location**: $CHECKPOINT_DIR

## Contents

1. **Model Checkpoints**: runs/train/ directory
2. **Training Logs**: All .log files
3. **ROCm Patches**: patches/ directory
4. **Git State**: status, log, diff
5. **System Info**: ROCm, PyTorch, GPU details

## Restoration

To restore after reboot:
\`\`\`bash
cd ~/Projects/robust-thermal-image-object-detection
cp -r $CHECKPOINT_DIR/runs/train/* runs/train/
cp -r $CHECKPOINT_DIR/patches/* patches/ 2>/dev/null || true
\`\`\`

## Training Resume

To continue training from last checkpoint:
\`\`\`bash
# Find last checkpoint
LAST_CHECKPOINT=\$(find runs/train -name "last.pt" | head -1)
echo "Last checkpoint: \$LAST_CHECKPOINT"

# Resume training
yolo detect train \\
  data=data/ltdv2_full/data.yaml \\
  model=\$LAST_CHECKPOINT \\
  resume=True \\
  device=0
\`\`\`

MANIFEST
echo "✓ Created manifest"
echo ""

# 8. Create tarball backup
echo "8. Creating compressed backup..."
tar -czf "$CHECKPOINT_DIR.tar.gz" -C checkpoints "$(basename $CHECKPOINT_DIR)"
BACKUP_SIZE=$(du -sh "$CHECKPOINT_DIR.tar.gz" | cut -f1)
echo "✓ Created backup: $CHECKPOINT_DIR.tar.gz ($BACKUP_SIZE)"
echo ""

# Summary
echo "=========================================="
echo "✅ CHECKPOINT SAVED SUCCESSFULLY"
echo "=========================================="
echo ""
echo "Backup Location: $CHECKPOINT_DIR"
echo "Compressed Backup: $CHECKPOINT_DIR.tar.gz"
echo "Backup Size: $BACKUP_SIZE"
echo ""
echo "Safe to reboot now!"
echo ""
echo "After reboot, view restore instructions:"
echo "cat $CHECKPOINT_DIR/MANIFEST.md"
echo ""

# List checkpoint contents
ls -lh "$CHECKPOINT_DIR/"
