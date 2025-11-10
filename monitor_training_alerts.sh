#!/bin/bash

# Automated Training Monitor with Milestone Alerts
# Runs continuously and alerts on epoch milestones

TRAINING_PID=407777
ALERT_FILE="training_alerts.log"
STATE_FILE=".training_state"

# Initialize state file if doesn't exist
if [ ! -f "$STATE_FILE" ]; then
    echo "last_epoch=0" > "$STATE_FILE"
    echo "first_epoch_alerted=false" >> "$STATE_FILE"
    echo "epoch_10_alerted=false" >> "$STATE_FILE"
    echo "epoch_30_alerted=false" >> "$STATE_FILE"
    echo "epoch_50_alerted=false" >> "$STATE_FILE"
    echo "epoch_100_alerted=false" >> "$STATE_FILE"
fi

# Function to send alert
alert() {
    local title="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Log to file
    echo "[$timestamp] $title: $message" >> "$ALERT_FILE"
    
    # Desktop notification (if available)
    if command -v notify-send &> /dev/null; then
        notify-send -u normal "YOLOv8 Training" "$title\n$message"
    fi
    
    # Console output with color
    echo -e "\n🔔 \033[1;32mALERT\033[0m [$timestamp]"
    echo -e "   \033[1;36m$title\033[0m"
    echo -e "   $message\n"
    
    # System beep (if available)
    if command -v paplay &> /dev/null && [ -f /usr/share/sounds/freedesktop/stereo/complete.oga ]; then
        paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null &
    fi
}

# Check if training is running
check_process() {
    if ! ps -p $TRAINING_PID > /dev/null 2>&1; then
        alert "⚠️ Training Stopped" "Process $TRAINING_PID is no longer running! Check training.log for errors."
        exit 1
    fi
}

# Get current epoch count
get_epoch_count() {
    local count=$(grep -c "Validating:" training.log 2>/dev/null || echo "0")
    echo "$count"
}

# Load state
load_state() {
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
    fi
}

# Monitor loop
echo "=========================================="
echo "Training Monitor Started"
echo "=========================================="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "PID: $TRAINING_PID"
echo "Monitoring for epoch milestones..."
echo ""
echo "Press Ctrl+C to stop monitoring"
echo "=========================================="
echo ""

while true; do
    check_process
    load_state
    
    current_epoch=$(get_epoch_count)
    
    # Update last_epoch
    if [ "$current_epoch" != "$last_epoch" ]; then
        sed -i "s/^last_epoch=.*/last_epoch=$current_epoch/" "$STATE_FILE"
        load_state
    fi
    
    # Check milestones
    if [ "$current_epoch" -ge 1 ] && [ "$first_epoch_alerted" = "false" ]; then
        runtime=$(ps -p $TRAINING_PID -o etime= | tr -d ' ')
        alert "✅ First Epoch Complete!" "Epoch 1/100 finished in $runtime. Training speed will now accelerate!"
        sed -i "s/^first_epoch_alerted=.*/first_epoch_alerted=true/" "$STATE_FILE"
        
        # Trigger detailed analysis
        if [ -f "./analyze_training_results.sh" ]; then
            ./analyze_training_results.sh > "analysis_epoch_1.txt" 2>&1
            alert "📊 Analysis Generated" "Detailed analysis saved to analysis_epoch_1.txt"
        fi
    fi
    
    if [ "$current_epoch" -ge 10 ] && [ "$epoch_10_alerted" = "false" ]; then
        alert "🎯 Checkpoint: 10 Epochs" "10% complete! Review analysis_epoch_10.txt for metrics."
        sed -i "s/^epoch_10_alerted=.*/epoch_10_alerted=true/" "$STATE_FILE"
        
        if [ -f "./analyze_training_results.sh" ]; then
            ./analyze_training_results.sh > "analysis_epoch_10.txt" 2>&1
        fi
    fi
    
    if [ "$current_epoch" -ge 30 ] && [ "$epoch_30_alerted" = "false" ]; then
        alert "🎯 Checkpoint: 30 Epochs" "30% complete! Training progressing well."
        sed -i "s/^epoch_30_alerted=.*/epoch_30_alerted=true/" "$STATE_FILE"
        
        if [ -f "./analyze_training_results.sh" ]; then
            ./analyze_training_results.sh > "analysis_epoch_30.txt" 2>&1
        fi
    fi
    
    if [ "$current_epoch" -ge 50 ] && [ "$epoch_50_alerted" = "false" ]; then
        alert "🏁 Halfway Point!" "50 epochs complete! Check mAP50 metrics in analysis_epoch_50.txt"
        sed -i "s/^epoch_50_alerted=.*/epoch_50_alerted=true/" "$STATE_FILE"
        
        if [ -f "./analyze_training_results.sh" ]; then
            ./analyze_training_results.sh > "analysis_epoch_50.txt" 2>&1
        fi
    fi
    
    if [ "$current_epoch" -ge 100 ] && [ "$epoch_100_alerted" = "false" ]; then
        alert "🎉 TRAINING COMPLETE!" "All 100 epochs finished! Final model ready for validation."
        sed -i "s/^epoch_100_alerted=.*/epoch_100_alerted=true/" "$STATE_FILE"
        
        if [ -f "./analyze_training_results.sh" ]; then
            ./analyze_training_results.sh > "analysis_epoch_100_FINAL.txt" 2>&1
            alert "📊 Final Analysis" "Complete training analysis saved to analysis_epoch_100_FINAL.txt"
        fi
        
        echo ""
        echo "Training completed successfully!"
        echo "Check analysis_epoch_100_FINAL.txt for detailed results."
        exit 0
    fi
    
    # Sleep for 60 seconds before next check
    sleep 60
done
