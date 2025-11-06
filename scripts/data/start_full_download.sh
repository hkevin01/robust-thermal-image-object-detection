#!/bin/bash
# Start LTDv2 full dataset download in the background
# This will download ~49 GB and take 2-4 hours

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/ltdv2_full"
LOG_FILE="$PROJECT_ROOT/data/ltdv2_download.log"

echo "=========================================="
echo "LTDv2 Full Dataset Download"
echo "=========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "This will download ~49 GB (1M+ images)"
echo "Estimated time: 2-4 hours depending on connection"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$(dirname "$LOG_FILE")"

# Start download in background
echo "Starting download in background..."
nohup "$PROJECT_ROOT/venv/bin/python" "$SCRIPT_DIR/download_ltdv2.py" \
    --output "$OUTPUT_DIR" \
    --mode full \
    > "$LOG_FILE" 2>&1 &

DOWNLOAD_PID=$!
echo "Download started with PID: $DOWNLOAD_PID"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check if running:"
echo "  ps aux | grep $DOWNLOAD_PID"
echo ""
echo "Stop download:"
echo "  kill $DOWNLOAD_PID"
echo ""
echo "=========================================="
