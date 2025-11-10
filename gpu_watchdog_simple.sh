#!/bin/bash

# Simple GPU Watchdog using sensors
WATCH_INTERVAL=30
LOG_FILE="gpu_watchdog.log"
TEMP_THRESHOLD=85

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alert() {
    local message="$1"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️  ALERT: $message" | tee -a "$LOG_FILE"
    if command -v notify-send &> /dev/null; then
        notify-send "GPU Alert" "$message" --urgency=critical
    fi
}

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "     GPU Watchdog - Temperature Monitor"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Check Interval: ${WATCH_INTERVAL}s"
echo "  Temp Threshold: ${TEMP_THRESHOLD}°C"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

log_message "GPU Watchdog started"

while true; do
    # Get GPU temps from sensors
    temp_edge=$(sensors amdgpu-pci-2d00 2>/dev/null | grep "edge:" | awk '{print $2}' | tr -d '+°C')
    temp_junction=$(sensors amdgpu-pci-2d00 2>/dev/null | grep "junction:" | awk '{print $2}' | tr -d '+°C')
    temp_mem=$(sensors amdgpu-pci-2d00 2>/dev/null | grep "mem:" | awk '{print $2}' | tr -d '+°C')
    power=$(sensors amdgpu-pci-2d00 2>/dev/null | grep "PPT:" | awk '{print $2}' | tr -d 'W')
    
    temp_edge=${temp_edge:-0}
    temp_junction=${temp_junction:-0}
    temp_mem=${temp_mem:-0}
    power=${power:-0}
    
    log_message "GPU: Edge ${temp_edge}°C | Junction ${temp_junction}°C | Memory ${temp_mem}°C | Power ${power}W"
    
    # Check thresholds
    if awk "BEGIN {exit !($temp_edge >= $TEMP_THRESHOLD)}"; then
        alert "High edge temperature: ${temp_edge}°C"
    fi
    
    if awk "BEGIN {exit !($temp_junction >= 95)}"; then
        alert "High junction temperature: ${temp_junction}°C"
    fi
    
    if awk "BEGIN {exit !($temp_mem >= 95)}"; then
        alert "High memory temperature: ${temp_mem}°C"
    fi
    
    sleep "$WATCH_INTERVAL"
done
