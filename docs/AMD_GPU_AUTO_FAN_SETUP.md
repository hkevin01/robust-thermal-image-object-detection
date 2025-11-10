# AMD GPU Automatic Fan Control Setup

## Problem
By default, the AMD GPU fan was in **manual mode** (pwm1_enable=1), which meant it wouldn't automatically increase speed when temperatures rose. This caused the GPU to reach dangerous temperatures (104°C junction) during training.

## Solution
Configured the system to automatically enable AMD GPU automatic fan control on boot.

## Components

### 1. Script: `/usr/local/bin/amdgpu-fan-auto.sh`
- Dynamically finds the amdgpu hwmon device
- Sets `pwm1_enable=2` (automatic mode)
- Works regardless of hwmon number changes

### 2. Systemd Service: `/etc/systemd/system/amdgpu-fan-auto.service`
- Runs on boot after multi-user.target
- Executes the fan auto script
- Enabled system-wide

## Verification

Check fan mode:
```bash
cat /sys/class/hwmon/hwmon*/pwm1_enable
# Should output: 2 (automatic mode)
```

Check service status:
```bash
sudo systemctl status amdgpu-fan-auto.service
```

Check GPU temperatures and fan:
```bash
rocm-smi --showtemp --showfan
```

## Fan Modes
- **0**: No fan control
- **1**: Manual fan control (fixed speed)
- **2**: Automatic fan control (adjusts based on temperature) ✅

## Temperature Thresholds (Typical)
- **Edge**: < 85°C normal, 95°C max
- **Junction**: < 95°C normal, 110°C max  
- **Memory**: < 90°C normal, 100°C max

## Results
With automatic fan control enabled:
- **Before**: 96°C edge, 104°C junction at 33% fan speed ❌
- **After**: 54°C edge, 62°C junction with automatic adjustment ✅

The system now automatically increases fan speed as temperature rises, preventing thermal throttling and potential GPU damage.

## Manual Override (if needed)
To manually set fan speed (disables automatic):
```bash
# Set to manual mode
echo 1 | sudo tee /sys/class/hwmon/hwmon3/pwm1_enable

# Set fan to 80% (204 out of 255)
echo 204 | sudo tee /sys/class/hwmon/hwmon3/pwm1

# Return to automatic
echo 2 | sudo tee /sys/class/hwmon/hwmon3/pwm1_enable
```

## On Reboot
The service automatically runs and enables automatic fan control. No manual intervention needed.
