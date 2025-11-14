#!/bin/bash

# Start User Preferences Dimensions WebSocket Service
# This service provides user-based behavior space dimensions for QD search

# Default configuration
HOST=${HOST:-localhost}
PORT=${PORT:-32070}
RECOMMEND_SERVICE_URL=${RECOMMEND_SERVICE_URL:-http://localhost:3004}
SIMILARITY_METRIC=${SIMILARITY_METRIC:-cosine}
NUM_USERS=${NUM_USERS:-10}
USE_KNN=${USE_KNN:-true}
SOUNDS_PER_USER=${SOUNDS_PER_USER:-10}
SOUND_SELECTION=${SOUND_SELECTION:-random}

echo "Starting User Preferences Dimensions Service..."
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Recommend Service: $RECOMMEND_SERVICE_URL"
echo "  Classification Mode: $([ "$USE_KNN" = "true" ] && echo "k-NN" || echo "embedding")"
echo "  Similarity Metric: $SIMILARITY_METRIC"
echo "  Number of Users: $NUM_USERS"
if [ "$USE_KNN" = "true" ]; then
  echo "  Sounds per User: $SOUNDS_PER_USER ($SOUND_SELECTION strategy)"
fi

# Build command with optional k-NN parameters
CMD="/Users/bjornpjo/Developer/apps/synth.is/kromosynth-evaluate/.venv/bin/python3 user_preferences_dimensions.py \
  --host \"$HOST\" \
  --port \"$PORT\" \
  --recommend-service-url \"$RECOMMEND_SERVICE_URL\" \
  --similarity-metric \"$SIMILARITY_METRIC\" \
  --num-users \"$NUM_USERS\" \
  --process-title \"user_prefs_dims\""

if [ "$USE_KNN" = "true" ]; then
  CMD="$CMD --use-knn --sounds-per-user \"$SOUNDS_PER_USER\" --sound-selection \"$SOUND_SELECTION\""
fi

eval $CMD
