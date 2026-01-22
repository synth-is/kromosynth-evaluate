#!/bin/bash

# Startup script for Audiobox Aesthetics Evaluation Service
# This service evaluates audio using the audiobox-aesthetics model
# Returns 4 aesthetic dimensions: CE, CU, PC, PQ

# Default configuration
HOST="localhost"
PORT=32080
SAMPLE_RATE=48000
PROCESS_TITLE="quality_aesthetics"
CHECKPOINT_PATH=""  # Leave empty to use HuggingFace default

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --host)
      HOST="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --sample-rate)
      SAMPLE_RATE="$2"
      shift 2
      ;;
    --checkpoint-path)
      CHECKPOINT_PATH="$2"
      shift 2
      ;;
    --process-title)
      PROCESS_TITLE="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --host HOST              Host to bind to (default: localhost)"
      echo "  --port PORT              Port to listen on (default: 32080)"
      echo "  --sample-rate RATE       Audio sample rate (default: 48000)"
      echo "  --checkpoint-path PATH   Path to model checkpoint (default: HuggingFace)"
      echo "  --process-title TITLE    Process title (default: quality_aesthetics)"
      echo "  --help                   Show this help message"
      echo ""
      echo "Examples:"
      echo "  $0                                          # Start with defaults"
      echo "  $0 --port 32081                             # Custom port"
      echo "  $0 --checkpoint-path /path/to/model.pth     # Custom model"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build the command
CMD="/Users/bjornpjo/Developer/apps/synth.is/kromosynth-evaluate/.venv/bin/python3 quality_aesthetics.py \
  --host $HOST \
  --port $PORT \
  --sample-rate $SAMPLE_RATE \
  --process-title $PROCESS_TITLE"

# Add checkpoint path if provided
if [ -n "$CHECKPOINT_PATH" ]; then
  CMD="$CMD --checkpoint-path $CHECKPOINT_PATH"
fi

# Display configuration
echo "========================================"
echo "Audiobox Aesthetics Evaluation Service"
echo "========================================"
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Sample Rate: ${SAMPLE_RATE}Hz"
echo "  Process Title: $PROCESS_TITLE"
if [ -n "$CHECKPOINT_PATH" ]; then
  echo "  Checkpoint: $CHECKPOINT_PATH"
else
  echo "  Checkpoint: HuggingFace default (facebook/audiobox-aesthetics)"
fi
echo "========================================"
echo ""
echo "WebSocket URL: ws://$HOST:$PORT"
echo ""
echo "Output modes:"
echo "  - ?output_mode=all (default): Returns all 4 dimension scores"
echo "  - ?output_mode=top          : Returns highest-scoring dimension only"
echo ""
echo "Dimension codes (no underscores to avoid parsing issues):"
echo "  - CE = Content Enjoyment"
echo "  - CU = Content Usefulness"
echo "  - PC = Production Complexity"
echo "  - PQ = Production Quality"
echo ""
echo "========================================"
echo ""

# Activate conda environment if needed
# Uncomment and modify if using conda:
# echo "Activating conda environment..."
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate audiobox-aesthetics

# Run the service
echo "Starting service..."
echo "Command: $CMD"
echo ""
exec $CMD
