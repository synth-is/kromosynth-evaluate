#!/usr/bin/env python3
"""
PCA Projection with User Classification Service

This service combines PCA dimensionality reduction with user preference classification.
It returns feature_map with 3 dimensions: [pca_x, pca_y, user_id]

Usage:
    python projection_with_user_classification.py --port 33051 --user-classifier-url ws://localhost:9011
"""

import asyncio
import websockets
import json
import argparse
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from setproctitle import setproctitle
import os
import sys

sys.path.append('../..')
from evaluation.util import filepath_to_port

# Global state
pca_models = {}  # Dict mapping request_path -> PCA model
scalers = {}  # Dict mapping request_path -> StandardScaler
cell_range_min_for_projection = {}
cell_range_max_for_projection = {}
USER_CLASSIFIER_URL = None

async def classify_features_via_websocket(features_list, user_classifier_url):
    """
    Send features to user classification websocket and get user IDs back.
    
    Args:
        features_list: List of feature vectors (numpy arrays)
        user_classifier_url: WebSocket URL for user classification service
    
    Returns:
        List of user IDs (strings)
    """
    if not user_classifier_url:
        # No user classification - return None for each feature
        return [None] * len(features_list)
    
    try:
        async with websockets.connect(user_classifier_url, max_size=100*1024*1024) as ws:
            user_ids = []
            
            for features in features_list:
                # Send features as bytes
                await ws.send(features.astype(np.float32).tobytes())
                
                # Receive classification response
                response_str = await ws.recv()
                response = json.loads(response_str)
                
                # Extract user ID from taggedPredictions
                if 'taggedPredictions' in response:
                    # Find user with highest prediction
                    tagged_preds = response['taggedPredictions']
                    if tagged_preds:
                        best_user = max(tagged_preds, key=tagged_preds.get)
                        # Extract user ID from tag like "USER_username"
                        user_id = best_user.replace('USER_', '')
                        user_ids.append(user_id)
                    else:
                        user_ids.append(None)
                else:
                    user_ids.append(None)
            
            return user_ids
            
    except Exception as e:
        print(f"⚠ Error classifying features via websocket: {e}")
        # Return None for all if classification fails
        return [None] * len(features_list)

async def socket_server(websocket, path):
    """
    WebSocket server handler for PCA projection with user classification.
    
    Accepts requests with feature_vectors and returns feature_map with [x, y, user_id].
    """
    global pca_models, scalers, cell_range_min_for_projection, cell_range_max_for_projection
    global USER_CLASSIFIER_URL
    
    try:
        message = await websocket.recv()
        jsonData = json.loads(message)
        
        request_path = path if path else '/pca'
        
        # Extract parameters
        feature_vectors = np.array(jsonData['feature_vectors'])
        fitness_values = jsonData.get('fitness_values', [])
        should_fit = jsonData.get('should_fit', False)
        pca_components = jsonData.get('pca_components', 2)
        
        print(f'✓ Received {len(feature_vectors)} feature vectors (shape: {feature_vectors.shape})')
        
        # Initialize or retrieve PCA model
        if should_fit or request_path not in pca_models:
            print(f'  ↻ Fitting new PCA model with {pca_components} components...')
            
            # Standardize features
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(feature_vectors)
            
            # Fit PCA
            pca = PCA(n_components=pca_components)
            pca.fit(scaled_features)
            
            # Store models
            pca_models[request_path] = pca
            scalers[request_path] = scaler
            
            print(f'  ✓ PCA model fitted. Explained variance: {pca.explained_variance_ratio_}')
        else:
            pca = pca_models[request_path]
            scaler = scalers[request_path]
        
        # Transform features
        scaled_features = scaler.transform(feature_vectors)
        pca_projection = pca.transform(scaled_features)
        
        # Discretize to grid
        if request_path in cell_range_min_for_projection and request_path in cell_range_max_for_projection:
            projection_min = cell_range_min_for_projection[request_path]
            projection_max = cell_range_max_for_projection[request_path]
        else:
            projection_min = np.min(pca_projection, axis=0)
            projection_max = np.max(pca_projection, axis=0)
            cell_range_min_for_projection[request_path] = projection_min
            cell_range_max_for_projection[request_path] = projection_max
        
        # Get grid dimensions from classification_dimensions if provided
        classification_dimensions = jsonData.get('classification_dimensions', [10, 10])
        
        # Extract just the numeric dimensions (ignore websocket URLs)
        grid_dimensions = []
        for dim in classification_dimensions:
            if isinstance(dim, (int, float)):
                grid_dimensions.append(int(dim))
            elif isinstance(dim, str) and not dim.startswith('ws://'):
                break  # Stop at first non-numeric dimension
        
        if len(grid_dimensions) < pca_components:
            grid_dimensions = [10] * pca_components
        
        # Discretize PCA projection to grid
        discretised_projection = []
        for projection in pca_projection:
            discretised = []
            for i, coord in enumerate(projection):
                if i >= len(grid_dimensions):
                    break
                # Normalize to [0, grid_size-1]
                normalized = (coord - projection_min[i]) / (projection_max[i] - projection_min[i] + 1e-10)
                grid_index = int(normalized * grid_dimensions[i])
                grid_index = max(0, min(grid_dimensions[i] - 1, grid_index))
                discretised.append(grid_index)
            discretised_projection.append(discretised)
        
        # Classify features to users if user classifier URL is provided
        user_ids = await classify_features_via_websocket(feature_vectors, USER_CLASSIFIER_URL)
        
        # Combine PCA coordinates with user IDs
        feature_map = []
        for pca_coords, user_id in zip(discretised_projection, user_ids):
            if user_id is not None:
                # Add user ID as third dimension
                feature_map.append(pca_coords + [user_id])
            else:
                # No user classification - just PCA coordinates
                feature_map.append(pca_coords)
        
        print(f'  ✓ Projected to {len(feature_map)} cells with dimensions: {[len(fm) for fm in feature_map[:3]]}...')
        if user_ids[0] is not None:
            print(f'    Example cell keys: {feature_map[0]}, {feature_map[1] if len(feature_map) > 1 else "..."}')
        
        # Prepare response
        response = {
            'status': 'OK',
            'feature_map': feature_map,
            'surprise_scores': None,
            'novelty_scores': None
        }
        
        await websocket.send(json.dumps(response))
        
    except Exception as e:
        print(f'✗ Exception in socket_server: {e}')
        import traceback
        traceback.print_exc()
        
        response = {'status': f'ERROR: {str(e)}'}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run PCA Projection with User Classification WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified.')
parser.add_argument('--port', type=int, default=33051, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='projection_user_classification', help='Process title.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file.')
parser.add_argument('--user-classifier-url', type=str, default='', 
                    help='WebSocket URL for user classification service (e.g., ws://localhost:9011)')
args = parser.parse_args()

# Set global user classifier URL
USER_CLASSIFIER_URL = args.user_classifier_url if args.user_classifier_url else None

# Set process title
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# Set PORT
PORT = int(os.environ.get('PORT', args.port))
HOST = args.host

# Handle host-info-file
if args.host_info_file:
    if not args.force_host:
        HOST = os.uname().nodename
    PORT = filepath_to_port(args.host_info_file)
    with open(args.host_info_file, 'w') as f:
        f.write(f'{HOST}:{PORT}')

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print(f'🚀 Starting PCA Projection with User Classification WebSocket server at ws://{HOST}:{PORT}')
if USER_CLASSIFIER_URL:
    print(f'   User classifier: {USER_CLASSIFIER_URL}')
else:
    print(f'   ⚠  No user classifier configured - will only return PCA coordinates')

# Start the WebSocket server
start_server = websockets.serve(
    socket_server, 
    HOST, 
    PORT,
    max_size=MAX_MESSAGE_SIZE,
    ping_timeout=None,
    ping_interval=None
)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
