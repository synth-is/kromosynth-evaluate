#!/bin/bash
# Start surrogate quality prediction service
#
# Usage:
#   ./start_surrogate_service.sh                    # Fresh model, default port
#   ./start_surrogate_service.sh --port 32070       # Custom port
#   ./start_surrogate_service.sh --model path.pt    # Load pre-trained model

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Default arguments
PORT=${PORT:-32070}
INPUT_DIM=${INPUT_DIM:-64}
N_MEMBERS=${N_MEMBERS:-5}

echo "Starting Surrogate Quality Prediction Service..."
echo "  Port: $PORT"
echo "  Input dim: $INPUT_DIM"
echo "  Ensemble members: $N_MEMBERS"
echo ""

/Users/bjornpjo/Developer/apps/synth.is/kromosynth-evaluate/.venv/bin/python3 projection/surrogate/ws_surrogate_service.py \
    --port $PORT \
    --input-dim $INPUT_DIM \
    --n-members $N_MEMBERS \
    "$@"
