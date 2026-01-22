#!/bin/bash
#
# Start QDHF projection WebSocket service
#
# Usage:
#   ./start_projection_service.sh --model models/projection/projection_v1.pt --port 32053
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Default arguments
MODEL_PATH=""
PORT=32053
HOST="127.0.0.1"
DEVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --model MODEL_PATH [--port PORT] [--host HOST] [--device DEVICE]"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model argument is required"
    echo "Usage: $0 --model MODEL_PATH [--port PORT] [--host HOST] [--device DEVICE]"
    exit 1
fi

# Build command
CMD="python -m projection.qdhf.ws_projection_service --model $MODEL_PATH --host $HOST --port $PORT"

if [ -n "$DEVICE" ]; then
    CMD="$CMD --device $DEVICE"
fi

echo "Starting QDHF projection service..."
echo "Model: $MODEL_PATH"
echo "Endpoint: ws://$HOST:$PORT/project"
echo ""

exec $CMD
