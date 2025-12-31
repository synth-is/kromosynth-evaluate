#!/bin/bash

# Start pyribs QD Service
#
# Usage:
#   ./start_pyribs_service.sh [options]
#
# Options:
#   --port PORT          Port to run on (default: 32052)
#   --debug              Run in debug mode
#
# Environment variables:
#   PYRIBS_PORT          Port to use (default: 32052)

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Activate virtual environment
if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    source "$PROJECT_ROOT/.venv/bin/activate"
else
    echo "Error: Virtual environment not found at $PROJECT_ROOT/.venv"
    echo "Please create and activate a virtual environment first."
    exit 1
fi

# Default values
PORT=${PYRIBS_PORT:-32052}
DEBUG_FLAG=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--port PORT] [--debug]"
            exit 1
            ;;
    esac
done

echo "Starting pyribs QD Service..."
echo "Port: $PORT"
echo "Debug: ${DEBUG_FLAG:-false}"
echo ""

# Build command
CMD="python -m qd.pyribs_service --port $PORT $DEBUG_FLAG"

# Run service
cd "$PROJECT_ROOT"
exec $CMD
