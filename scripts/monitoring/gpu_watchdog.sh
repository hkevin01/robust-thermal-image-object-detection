#!/bin/bash

# GPU Watchdog and Monitoring Script
# Monitors GPU health, detects hangs, logs stats

TRAINING_PID=""
WATCH_INTERVAL=30  # Check every 30 seconds
TEMP_THRESHOLD=85  # Celsius - warning threshold
VRAM_THRESHOLD=90  # Percent - warning threshold
HANG_THRESHOLD=300 # Seconds - if no log update, possible hang

LOG_FILE="gpu_watchdog.log"
GPU_STATS_FILE="gpu_stats.csv"
ALERT_FILE="gpu_alerts.log"

# Initialize CSV header
if [ ! -f "$GPU_STATS_FILE" ]; then
    echo "timestamp,temp_c,vram_used_mb,vram_total_mb,vram_percent,gpu_util_percent,power_w" > "$GPU_STATS_FILE"
fi

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  ALERT: $message" | tee -a "$ALERT_FILE"
    
    # Desktop notification if available
    if command -v notify-send &> /dev/null; then
        notify-send "GPU Watchdog Alert" "$message" --urgency=critical
    fi
    
    # Sound alert if available
    if command -v paplay &> /dev/null && [ -f /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga ]; then
        paplay /usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga &
    fi
}

get_gpu_stats() {
    # Get GPU temperature (edge)
    local temp=$(rocm-smi --showtemp 2>/dev/null | grep "edge" | awk -F': ' '{print $2}' | awk '{print $1}' | head -1)
    temp=${temp:-0}
    
    # Get VRAM usage via rocm-smi basic output
    local vram_percent=$(rocm-smi 2>/dev/null | grep "GPU" | awk '{print $10}' | tr -d '%' | head -1)
    vram_percent=${vram_percent:-0}
    
    # Calculate VRAM in MB (6 GB = 6144 MB)
    local vram_total=6144
    local vram_used=$(awk "BEGIN {printf \"%.0f\", $vram_total * $vram_percent / 100}")
    
    # Get GPU utilization
    local gpu_util=$(rocm-smi --showuse 2>/dev/null | grep "GPU use" | awk -F': ' '{print $2}' | awk '{print $1}' | head -1)
    gpu_util=${gpu_util:-0}
    
    # Get power usage from rocm-smi basic output
    local power=$(rocm-smi 2>/dev/null | grep "GPU" | awk '{print $3}' | tr -d 'W' | head -1)
    power=${power:-0}
    
    # Calculate VRAM percent
    local vram_percent=0
    if [ "$vram_total" -gt 0 ]; then
        vram_percent=$(awk "BEGIN {printf \"%.1f\", ($vram_used/$vram_total)*100}")
    fi
    
    # Log to CSV
    echo "$(date '+%Y-%m-%d %H:%M:%S'),$temp,$vram_used,$vram_total,$vram_percent,$gpu_util,$power" >> "$GPU_STATS_FILE"
    
    # Return stats as space-separated string
    echo "$temp $vram_used $vram_total $vram_percent $gpu_util $power"
}

check_gpu_health() {
    local stats=$(get_gpu_stats)
    read temp vram_used vram_total vram_percent gpu_util power <<< "$stats"
    
    log_message "GPU Stats - Temp: ${temp}°C | VRAM: ${vram_percent}% (${vram_used}/${vram_total} MB) | Util: ${gpu_util}% | Power: ${power}W"
    
    # Check temperature
    if [ "$temp" -ge "$TEMP_THRESHOLD" ]; then
        alert "High GPU temperature: ${temp}°C (threshold: ${TEMP_THRESHOLD}°C)"
    fi
    
    # Check VRAM usage
    if awk "BEGIN {exit !($vram_percent >= $VRAM_THRESHOLD)}"; then
        alert "High VRAM usage: ${vram_percent}% (threshold: ${VRAM_THRESHOLD}%)"
    fi
    
    # Check if GPU is responsive
    if ! rocm-smi &>/dev/null; then
        alert "GPU not responding to rocm-smi! Possible driver hang!"
        return 1
    fi
    
    return 0
}

check_training_alive() {
    if [ -z "$TRAINING_PID" ]; then
        return 0
    fi
    
    # Check if process exists
    if ! ps -p "$TRAINING_PID" > /dev/null 2>&1; then
        alert "Training process (PID $TRAINING_PID) has stopped!"
        return 1
    fi
    
    # Check if training log is being updated
    if [ -f "training.log" ]; then
        local log_age=$(stat -c %Y training.log)
        local now=$(date +%s)
        local age_seconds=$((now - log_age))
        
        if [ "$age_seconds" -gt "$HANG_THRESHOLD" ]; then
            alert "Training log not updated for ${age_seconds}s! Possible hang!"
            return 1
        fi
    fi
    
    return 0
}

find_training_process() {
    local pid=$(ps aux | grep "yolo detect train" | grep -v grep | awk '{print $2}' | head -1)
    if [ -n "$pid" ]; then
        TRAINING_PID="$pid"
        log_message "Found training process: PID $TRAINING_PID"
    fi
}

print_summary() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "           GPU Watchdog - Real-time Monitoring"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Temperature Threshold: ${TEMP_THRESHOLD}°C"
    echo "  VRAM Threshold: ${VRAM_THRESHOLD}%"
    echo "  Hang Detection: ${HANG_THRESHOLD}s"
    echo "  Check Interval: ${WATCH_INTERVAL}s"
    echo ""
    echo "  Logs: $LOG_FILE"
    echo "  Stats CSV: $GPU_STATS_FILE"
    echo "  Alerts: $ALERT_FILE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
}

main() {
    print_summary
    log_message "GPU Watchdog started"
    
    # Try to find training process
    find_training_process
    
    while true; do
        check_gpu_health
        
        if [ -n "$TRAINING_PID" ]; then
            check_training_alive
        else
            # Periodically search for training process
            find_training_process
        fi
        
        sleep "$WATCH_INTERVAL"
    done
}

# Handle Ctrl+C gracefully
trap 'log_message "GPU Watchdog stopped"; exit 0' INT TERM

main
