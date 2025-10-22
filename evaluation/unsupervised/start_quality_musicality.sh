#!/bin/bash

# Quality Musicality Service Startup Script
# 
# This script starts the quality_musicality evaluation service
# that will be used by your QD search for musical quality filtering.

echo "=========================================="
echo "Quality Musicality Service Startup"
echo "=========================================="

# Configuration
SAMPLE_RATE=16000
PORT=32060
HOST="127.0.0.1"

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port $PORT is already in use!"
    echo "Please stop the existing service or change the port."
    exit 1
fi

# Navigate to the evaluation directory
cd /Users/bjornpjo/Developer/apps/kromosynth-evaluate/evaluation/unsupervised

echo ""
echo "Starting quality_musicality service..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Sample Rate: $SAMPLE_RATE Hz"
echo ""
echo "Available endpoints:"
echo "  ws://$HOST:$PORT/musicality?config_preset=noise_only"
echo "  ws://$HOST:$PORT/musicality?config_preset=spectral_clarity"
echo "  ws://$HOST:$PORT/musicality?config_preset=vi_focused"
echo ""
echo "Press Ctrl+C to stop the service"
echo "=========================================="
echo ""

# Start the service
/Users/bjornpjo/Developer/apps/kromosynth-evaluate/.venv/bin/python3 quality_musicality.py \
    --host $HOST \
    --port $PORT \
    --sample-rate $SAMPLE_RATE \
    --process-title "quality_musicality_$PORT"
