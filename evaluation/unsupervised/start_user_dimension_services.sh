#!/bin/bash

# Startup script for user and reference sounds dimension services
# This script starts both websocket services needed for user-based QD behavior dimensions

echo "🚀 Starting User Dimension Services for QD Search"

# Set base directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Default configuration
RECOMMEND_SERVICE_URL=${RECOMMEND_SERVICE_URL:-"http://localhost:3004"}
REFERENCE_SOUNDS_FOLDER=${REFERENCE_SOUNDS_FOLDER:-"./reference_sounds"}
USER_PREFS_PORT=${USER_PREFS_PORT:-8080}
REF_SOUNDS_PORT=${REF_SOUNDS_PORT:-8081}

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        echo "⚠ Port $port is already in use"
        return 1
    fi
    return 0
}

# Function to start user preferences service
start_user_preferences_service() {
    echo "📊 Starting User Preferences Dimensions service on port $USER_PREFS_PORT..."
    
    if ! check_port $USER_PREFS_PORT; then
        echo "❌ Cannot start user preferences service - port $USER_PREFS_PORT in use"
        return 1
    fi
    
    # Start in embedding mode (default)
    /Users/bjornpjo/Developer/apps/synth.is/kromosynth-evaluate/.venv/bin/python3 user_preferences_dimensions.py \
        --port $USER_PREFS_PORT \
        --recommend-service-url $RECOMMEND_SERVICE_URL \
        --similarity-metric cosine \
        --num-users 10 \
        --process-title "user_preferences_dimensions_embedding" &
    
    USER_PREFS_PID=$!
    echo "✓ User Preferences service started (PID: $USER_PREFS_PID)"
    return 0
}

# Function to start user preferences service in k-NN mode
start_user_preferences_knn_service() {
    local port=$(($USER_PREFS_PORT + 1))
    echo "📊 Starting User Preferences Dimensions service (k-NN mode) on port $port..."
    
    if ! check_port $port; then
        echo "⚠ Cannot start user preferences k-NN service - port $port in use"
        return 1
    fi
    
    # Start in k-NN mode
    python user_preferences_dimensions.py \
        --port $port \
        --recommend-service-url $RECOMMEND_SERVICE_URL \
        --similarity-metric cosine \
        --num-users 10 \
        --use-knn \
        --knn-k 7 \
        --process-title "user_preferences_dimensions_knn" &
    
    USER_PREFS_KNN_PID=$!
    echo "✓ User Preferences k-NN service started (PID: $USER_PREFS_KNN_PID)"
    return 0
}

# Function to start reference sounds service
start_reference_sounds_service() {
    echo "🎵 Starting Reference Sounds Dimensions service on port $REF_SOUNDS_PORT..."
    
    if ! check_port $REF_SOUNDS_PORT; then
        echo "❌ Cannot start reference sounds service - port $REF_SOUNDS_PORT in use"
        return 1
    fi
    
    if [ ! -d "$REFERENCE_SOUNDS_FOLDER" ]; then
        echo "❌ Reference sounds folder not found: $REFERENCE_SOUNDS_FOLDER"
        echo "   Please create the folder and add reference audio files, or set REFERENCE_SOUNDS_FOLDER environment variable"
        return 1
    fi
    
    python user_references_sound_dimensions.py \
        --port $REF_SOUNDS_PORT \
        --reference-sounds-folder "$REFERENCE_SOUNDS_FOLDER" \
        --similarity-metric cosine \
        --mfcc-dimensions 96 \
        --sample-rate 22050 \
        --process-title "reference_sounds_dimensions" &
    
    REF_SOUNDS_PID=$!
    echo "✓ Reference Sounds service started (PID: $REF_SOUNDS_PID)"
    return 0
}

# Function to cleanup background processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$USER_PREFS_PID" ]; then
        kill $USER_PREFS_PID 2>/dev/null
        echo "✓ Stopped User Preferences service (PID: $USER_PREFS_PID)"
    fi
    
    if [ ! -z "$USER_PREFS_KNN_PID" ]; then
        kill $USER_PREFS_KNN_PID 2>/dev/null
        echo "✓ Stopped User Preferences k-NN service (PID: $USER_PREFS_KNN_PID)"
    fi
    
    if [ ! -z "$REF_SOUNDS_PID" ]; then
        kill $REF_SOUNDS_PID 2>/dev/null
        echo "✓ Stopped Reference Sounds service (PID: $REF_SOUNDS_PID)"
    fi
    
    echo "✅ All services stopped"
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

echo "🔧 Configuration:"
echo "   Recommend service: $RECOMMEND_SERVICE_URL"
echo "   Reference sounds: $REFERENCE_SOUNDS_FOLDER"
echo "   User preferences port: $USER_PREFS_PORT"
echo "   Reference sounds port: $REF_SOUNDS_PORT"
echo ""

# Check if kromosynth-recommend is running
echo "🔍 Checking kromosynth-recommend service..."
if ! curl -s "$RECOMMEND_SERVICE_URL/health" >/dev/null 2>&1; then
    echo "⚠ Warning: kromosynth-recommend service not accessible at $RECOMMEND_SERVICE_URL"
    echo "   User preferences services may not work correctly"
fi

# Start services
start_user_preferences_service
start_user_preferences_knn_service  # Optional k-NN variant
start_reference_sounds_service

echo ""
echo "🎉 All services started successfully!"
echo ""
echo "📋 Service URLs for QD configuration:"
echo "   User Preferences (embedding): ws://localhost:$USER_PREFS_PORT"
echo "   User Preferences (k-NN):      ws://localhost:$(($USER_PREFS_PORT + 1))"
echo "   Reference Sounds:             ws://localhost:$REF_SOUNDS_PORT"
echo ""
echo "⚡ To use in QD search, add to classifiers array in your config:"
echo '   "classifiers": ['
echo '     [50, 50],'
echo "     \"ws://localhost:$USER_PREFS_PORT\","
echo '     [50, 50, "USER_DIMENSIONS"]'
echo '   ]'
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for all background processes
wait