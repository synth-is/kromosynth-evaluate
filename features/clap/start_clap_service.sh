#!/bin/bash

# Start CLAP feature extraction WebSocket service
#
# Usage:
#   ./start_clap_service.sh [options]
#
# Options:
#   --port PORT          Port to run on (default: 32051)
#   --device DEVICE      Device to use: cuda or cpu (default: auto-detect)
#   --checkpoint PATH    Path to CLAP checkpoint (default: auto-download)
#
# Environment variables:
#   CLAP_CHECKPOINT_PATH   Path to CLAP checkpoint
#   CLAP_DEVICE            Device to use (cuda/cpu)
#
# First run will download the default CLAP checkpoint (~500MB)
# Default checkpoint: music_audioset_epoch_15_esc_90.14.pt

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please create and activate a virtual environment first."
    exit 1
fi

# Default values
PORT=32051
DEVICE=${CLAP_DEVICE:-}
CHECKPOINT=${CLAP_CHECKPOINT_PATH:-}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--device DEVICE] [--checkpoint PATH]"
            exit 1
            ;;
    esac
done

echo "Starting CLAP Feature Extraction Service..."
echo "Port: $PORT"
echo "Device: ${DEVICE:-auto-detect}"
echo "Checkpoint: ${CHECKPOINT:-default (will download)}"
echo ""

# Build command
CMD="python -m features.clap.ws_clap_service --port $PORT"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint-path $CHECKPOINT"
fi

# Run service
cd "$PROJECT_ROOT"
exec $CMD
