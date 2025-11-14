# User Preferences Dimensions Evaluation Service
# Websocket service that:
# 1. On startup, fetches n most active users from kromosynth-recommend REST API
# 2. On evaluation, classifies audio features to the nearest user's preference space
# 3. Returns user ID as the classification dimension

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
import requests
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional
from scipy.spatial.distance import cosine, euclidean

import sys
sys.path.append('../..')
from evaluation.util import filepath_to_port

# Global state
USER_DIMENSIONS = {}  # Dict mapping user_id -> preference embedding
USER_LIKED_SOUNDS = {}  # Dict mapping user_id -> list of sound features (for k-NN mode)
FIXED_USER_IDS = []  # Fixed list of user IDs (set once at startup, never changes)
RECOMMEND_SERVICE_URL = None
SIMILARITY_METRIC = "cosine"  # or "euclidean"
USE_KNN_CLASSIFICATION = False  # Use k-NN instead of embeddings
KNN_K = 5  # Number of neighbors for k-NN classification
SOUNDS_PER_USER = None  # Limit sounds per user for balanced classification
SOUND_SELECTION_STRATEGY = 'recent'  # Strategy for selecting sounds
NUM_USERS = 10  # Number of most active users to fetch
REFRESH_INTERVAL = 300  # Refresh feature vectors every N seconds (default: 5 minutes)

def fetch_user_dimensions_from_rest_api():
    """
    Fetch n most active users from the kromosynth-recommend REST API.
    Returns a dict mapping user_id -> user info (including preference embeddings)
    Also fetches liked sounds for k-NN mode if enabled.
    
    On first call: establishes the fixed list of user IDs (FIXED_USER_IDS)
    On subsequent calls: only refreshes feature vectors for those same users
    """
    global USER_DIMENSIONS, USER_LIKED_SOUNDS, RECOMMEND_SERVICE_URL, USE_KNN_CLASSIFICATION
    global SOUNDS_PER_USER, SOUND_SELECTION_STRATEGY, NUM_USERS, FIXED_USER_IDS
    
    is_initial_fetch = len(FIXED_USER_IDS) == 0
    
    try:
        if is_initial_fetch:
            # Initial fetch: get n most active users
            mode = 'knn' if USE_KNN_CLASSIFICATION else 'embedding'
            response = requests.get(
                f"{RECOMMEND_SERVICE_URL}/api/evolution/user-dimensions",
                params={'n': NUM_USERS, 'mode': mode},
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            users = data.get('users', [])
            
            # Set the fixed list of user IDs (will never change)
            FIXED_USER_IDS = [user['user_id'] for user in users]
            print(f"✓ Established fixed user dimension list: {FIXED_USER_IDS}")
        else:
            # Refresh: use the fixed list of user IDs
            users = []
            for user_id in FIXED_USER_IDS:
                users.append({'user_id': user_id})
            print(f"🔄 Refreshing feature vectors for {len(FIXED_USER_IDS)} fixed users...")
        
        # On initial fetch: clear and rebuild USER_DIMENSIONS
        # On refresh: preserve existing USER_DIMENSIONS, only update sounds
        if is_initial_fetch:
            USER_DIMENSIONS = {}
            USER_LIKED_SOUNDS = {}
        else:
            # Keep USER_DIMENSIONS intact, only clear/update USER_LIKED_SOUNDS
            USER_LIKED_SOUNDS = {}
        
        for user in users:
            user_id = user['user_id']
            
            # On initial fetch, get user metadata from API response
            # On refresh, we already have the user_id and just need to fetch sounds
            if is_initial_fetch:
                embedding = np.array(user.get('embedding', []))
                USER_DIMENSIONS[user_id] = {
                    'user_id': user_id,
                    'username': user.get('username', f'user_{user_id}'),
                    'preference_embedding': embedding,
                    'embedding_dimension': user.get('embedding_dimension'),
                    'activity_score': user.get('activity_metrics', {}).get('activity_count', 0)
                }
            else:
                # Verify user still exists in our fixed list
                if user_id not in USER_DIMENSIONS:
                    # User might have been deleted, skip
                    print(f"⚠ User {user_id} no longer exists in USER_DIMENSIONS, skipping")
                    continue
            
            # Fetch liked sounds for this user (on both initial and refresh)
            if USE_KNN_CLASSIFICATION:
                try:
                    # Build URL with optional limit and strategy parameters
                    params = {}
                    if SOUNDS_PER_USER is not None:
                        params['limit'] = SOUNDS_PER_USER
                    if SOUND_SELECTION_STRATEGY:
                        params['strategy'] = SOUND_SELECTION_STRATEGY
                    
                    liked_response = requests.get(
                        f"{RECOMMEND_SERVICE_URL}/api/evolution/user-liked-sounds/{user_id}",
                        params=params,
                        timeout=10
                    )
                    if liked_response.status_code == 200:
                        liked_data = liked_response.json()
                        # Support dimension-agnostic audio_features
                        USER_LIKED_SOUNDS[user_id] = [
                            np.array(sound['audio_features']) 
                            for sound in liked_data.get('sounds', [])
                            if 'audio_features' in sound and sound['audio_features']
                        ]
                        if USER_LIKED_SOUNDS[user_id]:
                            feature_dim = liked_data.get('sounds', [{}])[0].get('feature_dimension', 'unknown')
                            total = liked_data.get('total', len(USER_LIKED_SOUNDS[user_id]))
                            action = "Loaded" if is_initial_fetch else "Refreshed"
                            print(f"  ✓ {action} {len(USER_LIKED_SOUNDS[user_id])}/{total} sounds for user {user_id} (dim: {feature_dim})")
                    else:
                        USER_LIKED_SOUNDS[user_id] = []
                except requests.exceptions.RequestException:
                    print(f"⚠ Could not fetch liked sounds for user {user_id}")
                    USER_LIKED_SOUNDS[user_id] = []
        
        mode_str = "k-NN" if USE_KNN_CLASSIFICATION else "embedding"
        action_str = "Fetched" if is_initial_fetch else "Refreshed"
        print(f"✓ {action_str} {len(USER_DIMENSIONS)} user dimensions ({mode_str} mode) from {RECOMMEND_SERVICE_URL}")
        if not is_initial_fetch:
            print(f"  User list remains fixed: {FIXED_USER_IDS}")
        if USE_KNN_CLASSIFICATION:
            total_liked = sum(len(sounds) for sounds in USER_LIKED_SOUNDS.values())
            print(f"  Total liked sounds for k-NN: {total_liked}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching user dimensions from REST API: {e}")
        return False

def classify_features_to_user(features: np.ndarray) -> Dict:
    """
    Classify the given audio features to the nearest user's preference space.
    Uses either embedding-based or k-NN classification based on USE_KNN_CLASSIFICATION.
    Returns the user_id, similarity score, and user info.
    """
    global USER_DIMENSIONS, USER_LIKED_SOUNDS, SIMILARITY_METRIC, USE_KNN_CLASSIFICATION, KNN_K
    
    if not USER_DIMENSIONS:
        return {
            'user_id': None,
            'username': 'no_users',
            'similarity': 0.0,
            'error': 'No user dimensions available'
        }
    
    if USE_KNN_CLASSIFICATION:
        return classify_features_to_user_knn(features)
    else:
        return classify_features_to_user_embedding(features)

def classify_features_to_user_embedding(features: np.ndarray) -> Dict:
    """
    Classify using pre-computed user preference embeddings.
    """
    global USER_DIMENSIONS, SIMILARITY_METRIC
    
    best_user_id = None
    best_similarity = -1
    
    for user_id, user_info in USER_DIMENSIONS.items():
        preference_embedding = user_info['preference_embedding']
        
        if preference_embedding.size == 0:
            continue
        
        # Ensure features and embeddings have the same dimensionality
        if features.shape[0] != preference_embedding.shape[0]:
            expected_dim = user_info.get('embedding_dimension', preference_embedding.shape[0])
            print(f"⚠ Dimension mismatch: features {features.shape[0]}D vs user {user_id} embedding {expected_dim}D")
            continue
        
        # Calculate similarity
        if SIMILARITY_METRIC == "cosine":
            # Cosine similarity: 1 - cosine_distance
            similarity = 1 - cosine(features, preference_embedding)
        else:  # euclidean
            # Convert Euclidean distance to similarity (closer = higher similarity)
            distance = euclidean(features, preference_embedding)
            max_distance = np.sqrt(len(features))  # Theoretical maximum
            similarity = 1 - (distance / max_distance)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_user_id = user_id
    
    if best_user_id is None:
        return {
            'user_id': None,
            'username': 'no_match',
            'similarity': 0.0,
            'error': 'No valid user match found'
        }
    
    return {
        'user_id': best_user_id,
        'username': USER_DIMENSIONS[best_user_id]['username'],
        'similarity': float(best_similarity),
        'activity_score': USER_DIMENSIONS[best_user_id]['activity_score']
    }

def classify_features_to_user_knn(features: np.ndarray) -> Dict:
    """
    Classify using k-NN among all users' liked sounds.
    Find k nearest neighbors across all users, then classify based on which user dominates.
    Only compares against sounds with matching feature dimensions.
    """
    global USER_LIKED_SOUNDS, USER_DIMENSIONS, SIMILARITY_METRIC, KNN_K
    
    feature_dim = features.shape[0]
    
    # Collect all liked sounds with user labels (only matching dimensions)
    all_liked_sounds = []
    for user_id, sounds in USER_LIKED_SOUNDS.items():
        for sound_features in sounds:
            if sound_features.shape[0] == feature_dim:
                all_liked_sounds.append((user_id, sound_features))
    
    if len(all_liked_sounds) < KNN_K:
        print(f"⚠ Not enough liked sounds ({len(all_liked_sounds)}) with {feature_dim}D features for k-NN classification (k={KNN_K})")
        # Fall back to embedding method if available
        if any(USER_DIMENSIONS[uid]['preference_embedding'].size > 0 for uid in USER_DIMENSIONS):
            return classify_features_to_user_embedding(features)
        else:
            return {
                'user_id': None,
                'username': 'insufficient_data',
                'similarity': 0.0,
                'error': f'Not enough liked sounds with {feature_dim}D features for k-NN (need {KNN_K}, have {len(all_liked_sounds)})'
            }
    
    # Calculate similarities to all liked sounds
    similarities = []
    for user_id, sound_features in all_liked_sounds:
        if SIMILARITY_METRIC == "cosine":
            similarity = 1 - cosine(features, sound_features)
        else:  # euclidean
            distance = euclidean(features, sound_features)
            max_distance = np.sqrt(len(features))
            similarity = 1 - (distance / max_distance)
        
        similarities.append((user_id, similarity))
    
    # Get k nearest neighbors
    similarities.sort(key=lambda x: x[1], reverse=True)
    k_neighbors = similarities[:KNN_K]
    
    # Count votes for each user
    user_votes = {}
    total_similarity = 0
    for user_id, similarity in k_neighbors:
        if user_id not in user_votes:
            user_votes[user_id] = {'count': 0, 'total_similarity': 0}
        user_votes[user_id]['count'] += 1
        user_votes[user_id]['total_similarity'] += similarity
        total_similarity += similarity
    
    # Find user with most votes (tie-break by average similarity)
    best_user_id = None
    best_score = 0
    best_avg_similarity = 0
    
    for user_id, votes in user_votes.items():
        vote_ratio = votes['count'] / KNN_K
        avg_similarity = votes['total_similarity'] / votes['count']
        
        # Score combines vote ratio and average similarity
        score = vote_ratio * 0.7 + avg_similarity * 0.3
        
        if score > best_score:
            best_score = score
            best_user_id = user_id
            best_avg_similarity = avg_similarity
    
    if best_user_id is None:
        return {
            'user_id': None,
            'username': 'no_match',
            'similarity': 0.0,
            'error': 'k-NN classification failed'
        }
    
    return {
        'user_id': best_user_id,
        'username': USER_DIMENSIONS[best_user_id]['username'],
        'similarity': float(best_avg_similarity),
        'activity_score': USER_DIMENSIONS[best_user_id]['activity_score'],
        'knn_votes': user_votes[best_user_id]['count'],
        'knn_total_votes': KNN_K
    }

async def socket_server(websocket, path):
    """
    Websocket server handler.
    Accepts requests with:
    1. {'getKeys': True} - returns list of user IDs (dimensions)
    2. Audio features (numpy array) - returns classification to nearest user
    """
    global USER_DIMENSIONS
    
    url_components = urlparse(path)
    request_path = url_components.path
    query_params = parse_qs(url_components.query)
    
    try:
        message = await websocket.recv()
        features = None
        
        # Handle request for dimension keys (user IDs)
        if isinstance(message, str):
            try:
                request_data = json.loads(message)
                
                # Handle getKeys request (returns dict)
                if isinstance(request_data, dict) and request_data.get('getKeys') is True:
                    # Return list of user IDs as the dimension keys
                    # Format matches existing services - send array directly
                    user_ids = list(USER_DIMENSIONS.keys())
                    await websocket.send(json.dumps(user_ids))
                    print(f"✓ Returned {len(user_ids)} user dimension keys: {user_ids}")
                    return
                
                # Handle feature array sent as JSON (returns list)
                elif isinstance(request_data, list):
                    # Convert JSON array to numpy array
                    features = np.array(request_data, dtype=np.float32)
                    print(f'✓ Received {len(features)}-dimensional audio features (JSON) for user classification')
                else:
                    print(f"⚠ Unexpected JSON data type: {type(request_data)}")
                    await websocket.send(json.dumps({'status': 'ERROR: Unexpected data format'}))
                    return
            except json.JSONDecodeError:
                pass
        
        # Handle audio features for classification (binary format)
        elif isinstance(message, bytes):
            # Convert bytes to numpy array
            features = np.frombuffer(message, dtype=np.float32)
            print(f'✓ Received {len(features)}-dimensional audio features (binary) for user classification')
        
        # If we have features (from either JSON or binary), classify them
        if features is not None:
            start = time.time()
            
            # Classify to nearest user
            classification_result = classify_features_to_user(features)
            
            end = time.time()
            print(f'✓ Classified to user: {classification_result["username"]} (similarity: {classification_result["similarity"]:.3f}) in {end - start:.3f}s')
            
            # Determine response format based on request type
            # JSON array requests (from quality evaluation) expect 'fitness' field with top_score/top_score_class
            # Binary requests (from classification) expect 'taggedPredictions' field
            is_quality_request = isinstance(message, str)
            
            if is_quality_request:
                # Quality evaluation mode - return top_score and top_score_class (like instrumentation service)
                fitness_value = {
                    'top_score': classification_result['similarity'],
                    'top_score_class': classification_result['user_id']  # User ID (e.g., zm5slkizn7ibmbs) as class
                }
                response = {
                    'status': 'received standalone audio features',
                    'fitness': fitness_value
                }
            else:
                # Classification mode - return taggedPredictions for all users
                tagged_predictions = {}
                if classification_result['user_id']:
                    # Create predictions for all users, with the best match getting highest score
                    for user_id in USER_DIMENSIONS.keys():
                        if user_id == classification_result['user_id']:
                            tagged_predictions[f"USER_{classification_result['username']}"] = classification_result['similarity']
                        else:
                            # Give other users a small base score
                            tagged_predictions[f"USER_{USER_DIMENSIONS[user_id]['username']}"] = 0.01
                else:
                    # If no match, give all users equal low scores
                    for user_id, user_info in USER_DIMENSIONS.items():
                        tagged_predictions[f"USER_{user_info['username']}"] = 0.01
                
                response = {
                    'status': 'received standalone audio features',
                    'taggedPredictions': tagged_predictions
                }
            
            await websocket.send(json.dumps(response))
    
    except Exception as e:
        print(f'✗ Exception in socket_server: {e}')
        import traceback
        traceback.print_exc()
        
        response = {'status': f'ERROR: {str(e)}'}
        await websocket.send(json.dumps(response))

async def periodic_refresh_task():
    """
    Background task that periodically refreshes feature vectors for the fixed set of users.
    This allows the service to adapt to users' continued activity without changing the
    behavior space dimensions (which are determined by the initial user list).
    """
    global REFRESH_INTERVAL
    
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        print(f"\n⏰ Periodic refresh triggered (interval: {REFRESH_INTERVAL}s)")
        try:
            success = fetch_user_dimensions_from_rest_api()
            if success:
                print(f"✅ Periodic refresh completed successfully")
            else:
                print(f"⚠️ Periodic refresh failed, will retry in {REFRESH_INTERVAL}s")
        except Exception as e:
            print(f"❌ Error during periodic refresh: {e}")
            import traceback
            traceback.print_exc()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run User Preferences Dimensions WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='user_preferences_dimensions', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
parser.add_argument('--recommend-service-url', type=str, default='http://localhost:3002', 
                    help='URL of kromosynth-recommend REST service')
parser.add_argument('--similarity-metric', type=str, default='cosine', choices=['cosine', 'euclidean'],
                    help='Similarity metric to use for user classification')
parser.add_argument('--num-users', type=int, default=10, help='Number of most active users to fetch')
parser.add_argument('--sounds-per-user', type=int, default=None, 
                    help='Maximum sounds per user for balanced k-NN (default: None = all sounds)')
parser.add_argument('--sound-selection', type=str, default='recent', choices=['recent', 'random', 'oldest'],
                    help='Strategy for selecting sounds per user')
parser.add_argument('--use-knn', action='store_true', 
                    help='Use k-NN classification instead of embeddings')
parser.add_argument('--knn-k', type=int, default=5, 
                    help='Number of neighbors for k-NN classification')
parser.add_argument('--refresh-interval', type=int, default=300,
                    help='Interval in seconds to refresh feature vectors for fixed users (default: 300 = 5 minutes, 0 = disable)')
args = parser.parse_args()

# Set global configuration
RECOMMEND_SERVICE_URL = args.recommend_service_url
SIMILARITY_METRIC = args.similarity_metric
USE_KNN_CLASSIFICATION = args.use_knn
KNN_K = args.knn_k
SOUNDS_PER_USER = args.sounds_per_user
SOUND_SELECTION_STRATEGY = args.sound_selection
NUM_USERS = args.num_users
REFRESH_INTERVAL = args.refresh_interval

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

# Fetch user dimensions on startup
print(f"🔄 Fetching user dimensions from {RECOMMEND_SERVICE_URL}...")
if not fetch_user_dimensions_from_rest_api():
    print("⚠ WARNING: Could not fetch user dimensions. Service will start but won't classify correctly.")
    print("   Make sure kromosynth-recommend is running and accessible.")

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print(f'🚀 Starting User Preferences Dimensions WebSocket server at ws://{HOST}:{PORT}')
print(f'   Recommend service: {RECOMMEND_SERVICE_URL}')
print(f'   Classification mode: {"k-NN" if USE_KNN_CLASSIFICATION else "embedding"}')
print(f'   Similarity metric: {SIMILARITY_METRIC}')
if USE_KNN_CLASSIFICATION:
    print(f'   k-NN neighbors: {KNN_K}')
    if SOUNDS_PER_USER:
        print(f'   Sounds per user: {SOUNDS_PER_USER} ({SOUND_SELECTION_STRATEGY})')
    else:
        print(f'   Sounds per user: all ({SOUND_SELECTION_STRATEGY})')
print(f'   Active users: {len(USER_DIMENSIONS)}')
if REFRESH_INTERVAL > 0:
    print(f'   Refresh interval: {REFRESH_INTERVAL}s ({REFRESH_INTERVAL/60:.1f} minutes)')
else:
    print(f'   Refresh interval: disabled')

# Start the WebSocket server
start_server = websockets.serve(
    socket_server, 
    HOST, 
    PORT,
    max_size=MAX_MESSAGE_SIZE,
    ping_timeout=None,
    ping_interval=None
)

# Start background refresh task if enabled
async def start_services():
    await start_server
    if REFRESH_INTERVAL > 0:
        print(f"🔄 Starting periodic refresh task (every {REFRESH_INTERVAL}s)")
        asyncio.create_task(periodic_refresh_task())
    else:
        print(f"⏸️  Periodic refresh disabled")

asyncio.get_event_loop().run_until_complete(start_services())
asyncio.get_event_loop().run_forever()
