# AMD GPU Fan Curve Configuration

## Problem Solved
The default AMD GPU automatic fan control was too conservative, running at only 33-37% during heavy training loads, causing temperatures to reach 104°C junction. The fan needs to run more aggressively during sustained GPU workloads.

## Solution: Custom Fan Curve with 50% Minimum

### Fan Curve Temperature Targets
```
Junction Temp    Fan Speed    Purpose
─────────────────────────────────────────
< 65°C          50%          Minimum (idle/light load)
65-75°C         60-70%       Active training
75-85°C         70-85%       Heavy load
85-95°C         85-100%      High temperature
≥ 95°C          100%         Critical (max cooling)
```

### Components

**1. Script:** `/usr/local/bin/amdgpu-fan-curve.sh`
- Monitors GPU temperature every 5 seconds
- Adjusts fan speed based on junction temperature
- Maintains minimum 50% fan speed during any GPU activity
- Logs temperature and fan speed to systemd journal

**2. Service:** `/etc/systemd/system/amdgpu-fan-curve.service`
- Runs continuously as a system service
- Auto-starts on boot
- Auto-restarts if crashed
- Logs to system journal

## GPU Fan Longevity

### How Long Can Fans Run at 100%?

**Short Answer:** Modern GPU fans are designed to run at 100% continuously for years.

**Long Answer:**
- **Design Life**: GPU fans are rated for 50,000-100,000 hours at full speed
- **100% Continuous**: At 100% speed 24/7, fan would last **5-11 years**
- **50-60% Speed**: At our minimum 50% speed, fan will last **10+ years**
- **Duty Cycle**: Training sessions are typically intermittent, extending life further

### Fan Bearing Types
- **Sleeve Bearing**: 30,000-50,000 hours (3-5 years continuous)
- **Ball Bearing**: 50,000-80,000 hours (5-9 years continuous) - Most common
- **Fluid Dynamic Bearing (FDB)**: 100,000+ hours (11+ years) - Premium

### AMD RX 5600 XT Specifications
- **Fan Type**: Dual ball bearing fans
- **Expected Life**: 50,000+ hours at full load
- **Our Usage**: ~50-70% average speed during training
- **Realistic Lifespan**: 10+ years with proper maintenance

### Thermal Impact on GPU
**Running GPU hot is MUCH worse than running fans hard:**
- **Fan replacement**: $20-50 and 30 minutes of work
- **GPU damage from heat**: Permanent, card replacement $300+
- **Thermal throttling**: Reduces performance, wastes training time
- **Junction > 100°C**: Accelerates silicon degradation

**Verdict**: Keep fans running aggressively. It's worth it!

## Monitoring Commands

View live fan curve adjustments:
```bash
sudo journalctl -u amdgpu-fan-curve.service -f
```

Check current status:
```bash
sudo systemctl status amdgpu-fan-curve.service
```

View recent adjustments:
```bash
sudo journalctl -u amdgpu-fan-curve.service -n 50 --no-pager
```

Check GPU temps and fan:
```bash
rocm-smi --showtemp --showfan
```

## Configuration Options

To change minimum fan speed, edit: `/usr/local/bin/amdgpu-fan-curve.sh`
```bash
MIN_FAN_PERCENT=50    # Change this value (30-80 recommended)
```

Then restart the service:
```bash
sudo systemctl restart amdgpu-fan-curve.service
```

## Temperature Curves Explained

### Conservative (Default AMD - NOT RECOMMENDED)
- Idle: 20-30%
- Load: 40-60%
- Problem: GPU reaches 90-100°C

### Balanced (Our 50% minimum - RECOMMENDED)
- Idle: 50%
- Load: 60-80%
- Critical: 100%
- Result: GPU stays 60-75°C

### Aggressive (60% minimum)
- Idle: 60%
- Load: 70-90%
- Critical: 100%
- Result: GPU stays 55-65°C
- Trade-off: Louder, but coolest

## Why 50% Minimum?

1. **Thermal Reserve**: Keeps temps low before aggressive workloads
2. **Fan Responsiveness**: Easier to ramp up from 50% than 30%
3. **Silence**: 50% is quiet enough for most users
4. **Longevity**: Still well below fan's capability
5. **Power**: Minimal power draw difference vs 30%

## Results

**Before (Auto Fan):**
- Idle: 30-35% fan
- Training: 33-40% fan
- Temperature: 96°C edge, 104°C junction ❌
- **DANGEROUS TEMPERATURES**

**After (50% Minimum Curve):**
- Idle: 50% fan
- Training: 60-70% fan  
- Temperature: 54°C edge, 64°C junction ✅
- **SAFE AND OPTIMAL**

## Fan Maintenance

To maximize fan life:
1. **Clean dust every 3-6 months** (compressed air)
2. **Monitor bearing noise** (grinding = replacement needed)
3. **Keep case ventilation clear**
4. **Room temperature** < 25°C ideal

## Emergency: Manual Fan Control

If service fails, manually set fan:
```bash
# Stop service
sudo systemctl stop amdgpu-fan-curve.service

# Set to manual mode
echo 1 | sudo tee /sys/class/hwmon/hwmon3/pwm1_enable

# Set fan to 80% (204 out of 255)
echo 204 | sudo tee /sys/class/hwmon/hwmon3/pwm1

# Restart service
sudo systemctl start amdgpu-fan-curve.service
```

## Summary

✅ **Fan can run at 100% for years** - don't worry about fan wear
✅ **50% minimum prevents thermal issues** - maintains safe temps
✅ **Automatic adjustment** - scales up when needed
✅ **System-wide** - persists across reboots
✅ **Logged** - trackable via journalctl

**Bottom Line**: Aggressive fan curves protect expensive GPU hardware at the cost of cheap, replaceable fan bearings. Always choose cooling over silence for compute workloads.
