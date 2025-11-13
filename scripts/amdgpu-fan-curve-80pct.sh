#!/bin/bash
# AMD GPU Smart Fan Curve with 80% Minimum for Training
# Maintains aggressive cooling to maximize GPU performance and lifespan

# Find amdgpu hwmon device
AMDGPU_HWMON=""
for hwmon in /sys/class/hwmon/hwmon*/name; do
    if [ "$(cat "$hwmon")" = "amdgpu" ]; then
        AMDGPU_HWMON=$(dirname "$hwmon")
        break
    fi
done

if [ -z "$AMDGPU_HWMON" ]; then
    echo "Error: AMD GPU hwmon device not found"
    exit 1
fi

PWM_FILE="$AMDGPU_HWMON/pwm1"
PWM_ENABLE="$AMDGPU_HWMON/pwm1_enable"
TEMP_INPUT="$AMDGPU_HWMON/temp1_input"

# Set to manual mode for custom curve
echo 1 > "$PWM_ENABLE"

LOG_FILE="/var/log/amdgpu-fan-curve.log"
echo "=== AMD GPU Fan Curve Started: $(date) ===" >> "$LOG_FILE"
echo "Minimum fan speed: 80% (optimal for long training runs)" >> "$LOG_FILE"

while true; do
    # Read temperature (in millidegrees)
    TEMP=$(cat "$TEMP_INPUT")
    TEMP_C=$((TEMP / 1000))
    
    # Fan curve with 80% minimum (204/255)
    # Optimized for 24/7 training workloads
    if [ "$TEMP_C" -lt 50 ]; then
        # Idle/Light load: 80%
        FAN_PWM=204
        FAN_PERCENT=80
    elif [ "$TEMP_C" -lt 60 ]; then
        # Moderate load: 85%
        FAN_PWM=217
        FAN_PERCENT=85
    elif [ "$TEMP_C" -lt 70 ]; then
        # Heavy load: 90%
        FAN_PWM=230
        FAN_PERCENT=90
    elif [ "$TEMP_C" -lt 75 ]; then
        # Very heavy load: 95%
        FAN_PWM=242
        FAN_PERCENT=95
    else
        # Hot/Critical: 100%
        FAN_PWM=255
        FAN_PERCENT=100
    fi
    
    # Apply fan speed
    echo "$FAN_PWM" > "$PWM_FILE"
    
    # Log every 30 seconds
    if [ $(($(date +%s) % 30)) -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Temp: ${TEMP_C}°C | Fan: ${FAN_PERCENT}% (${FAN_PWM}/255)" >> "$LOG_FILE"
    fi
    
    sleep 2
done
