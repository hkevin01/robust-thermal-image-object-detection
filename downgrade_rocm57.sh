#!/bin/bash
set -e

echo "========================================="
echo "ROCm 5.7.1 + PyTorch 2.2.2 Downgrade Script"
echo "========================================="
echo ""

# Step 1: Remove current PyTorch
echo "[1/6] Removing current PyTorch..."
pip3 uninstall torch torchvision torchaudio -y --break-system-packages 2>/dev/null || echo "PyTorch packages not found or already removed"

# Step 2: Remove ROCm 6.2 (if installed via apt)
echo "[2/6] Checking for ROCm 6.x packages..."
if dpkg -l 2>/dev/null | grep -q rocm; then
    echo "Found ROCm packages, removing..."
    sudo apt remove rocm-* hip-* hsa-* -y 2>/dev/null || true
    sudo apt autoremove -y
else
    echo "No ROCm packages found via apt"
fi

# Step 3: Add ROCm 5.7.1 repository
echo "[3/6] Adding ROCm 5.7.1 repository..."
# Remove old ROCm repositories
sudo rm -f /etc/apt/sources.list.d/rocm.list /etc/apt/sources.list.d/amdgpu.list

# Add ROCm 5.7.1 GPG key (modern method)
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/rocm.gpg

# Add ROCm 5.7.1 repository (using jammy for Ubuntu 24.04 compatibility)
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/5.7.1 jammy main" | sudo tee /etc/apt/sources.list.d/rocm.list

# Update package lists
sudo apt update

# Step 4: Install ROCm 5.7.1
echo "[4/6] Installing ROCm 5.7.1..."
sudo apt install rocm-hip-sdk rocm-libs -y

# Step 5: Install PyTorch 2.2.2 with ROCm 5.7 support
echo "[5/6] Installing PyTorch 2.2.2+rocm5.7..."
pip3 install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/rocm5.7 --break-system-packages

# Step 6: Verify installation
echo "[6/6] Verifying installation..."
echo ""
echo "========================================="
echo "Verification:"
echo "========================================="

# Check ROCm version
if command -v rocminfo &> /dev/null; then
    echo "ROCm Info:"
    rocminfo | grep -E "Agent|Marketing Name" | head -5
else
    echo "Warning: rocminfo not found"
fi

# Check PyTorch version and GPU detection
python3 << PYTHON
import torch
print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"GPU Count: {torch.cuda.device_count()}")
else:
    print("WARNING: GPU not detected!")
PYTHON

echo ""
echo "========================================="
echo "Downgrade Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Reboot system: sudo reboot"
echo "2. After reboot, test GPU with: python3 -c 'import torch; print(torch.cuda.is_available())'"
echo "3. Test GPU training with: yolo detect train data=data/ltdv2_full/data.yaml model=yolov8n.pt epochs=1 batch=4 device=0"
