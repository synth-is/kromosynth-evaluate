# similarity / distance measurements for embeddings / sets of feature vectors

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
from scipy.spatial.distance import cosine, euclidean
from sklearn.preprocessing import StandardScaler

from typing import Dict, Tuple, List
import glob
import pathlib

import sys
sys.path.append('../..')
from evaluation.util import filepath_to_port
import gzip

def str_to_bool(s):
    return s.lower() in ['true', '1', 't', 'y', 'yes']

reference_embeddings = {}

### Z-score normalization - begin ###
normalization_stats: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # (mean, std) pairs

def load_features_from_file(path: str, feature_key: str) -> List[np.ndarray]:
    """Load all feature sets from a single JSON or JSON.GZ file."""
    open_func = gzip.open if path.endswith('.gz') else open
    with open_func(path, 'rt') as f:
        feature_sets = json.load(f)
    features = []
    for feature_set in feature_sets:
        try:
            features.append(np.array(feature_set[feature_key]))
        except KeyError:
            print(f"KeyError: '{feature_key}' not found in feature set from file {path}")
    return features

def load_features_from_directory(directory: str, feature_key: str) -> List[np.ndarray]:
    """Recursively load features from all JSON files in directory."""
    features = []
    for json_file in glob.glob(f"{directory}/**/*.json", recursive=True):
        with open(json_file, 'r') as f:
            features_dict = json.load(f)
            try:
                features.append(np.array(features_dict[feature_key]))
            except KeyError:
                print(f"KeyError: '{feature_key}' not found in feature set from file {json_file}")
    return features

def compute_global_statistics(ref_paths: List[str], train_file_path: str, feature_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute global mean and std from reference directories and training file
    """
    all_features = []
    
    # Load features from reference paths
    for ref_path in ref_paths:
        all_features.extend(load_features_from_directory(ref_path.strip(), feature_key))
    
    # Load features from training file
    if train_file_path:
        all_features.extend(load_features_from_file(train_file_path, feature_key))
    
    # Convert to numpy array for calculations
    features_array = np.vstack(all_features)
    
    # Compute statistics
    global_mean = np.mean(features_array, axis=0)
    global_std = np.std(features_array, axis=0)
    global_std[global_std == 0] = 1.0  # Prevent division by zero
    
    return global_mean, global_std

def get_normalization_key(ref_paths: str, train_file_path: str) -> str:
    """Create unique key for normalization statistics."""
    paths = sorted([p.strip() for p in ref_paths.split(',')]) if ref_paths else []
    if train_file_path:
        paths.append(train_file_path)
    return ','.join(paths)

def z_score_normalize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply Z-score normalization to features."""
    return (features - mean) / std
### Z-score normalization - end ###


def cosine_similarity(query_embedding, reference_embedding):
    cosine_dissimilarity = cosine(query_embedding.flatten(), reference_embedding.flatten())
    return 1 - (cosine_dissimilarity / 2)

def improved_cosine_similarity(query_embedding, reference_embedding):
    # Check for zero vector
    if np.all(query_embedding == 0) or np.all(reference_embedding == 0):
        return 0
        
    # Normalize vectors - epsilon (1e-8) to avoid division by zero during normalization
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    reference_norm = reference_embedding / (np.linalg.norm(reference_embedding) + 1e-8)
    
    # Compute cosine similarity
    similarity = 1 - cosine(query_norm, reference_norm)
    
    # Ensure similarity is within [-1, 1] range
    similarity = np.clip(similarity, -1, 1)
    
    # Convert to [0, 1] range
    similarity = (similarity + 1) / 2
    
    return similarity

def euclidean_distance(query_embedding, reference_embedding):
    scaler = StandardScaler()
    combined = np.vstack((query_embedding, reference_embedding))
    combined_scaled = scaler.fit_transform(combined)
    query_scaled, reference_scaled = combined_scaled[0], combined_scaled[1]
    
    distance = euclidean(query_scaled, reference_scaled)
    max_distance = np.sqrt(len(query_scaled))  # Maximum possible distance in normalized space
    return 1 - (distance / max_distance)

# alternative to euclidean_distance (euclidean in effect)
def low_dimensional_similarity(query_features, reference_features):
    # Ensure inputs are numpy arrays
    query_features = np.array(query_features)
    reference_features = np.array(reference_features)
    
    # Normalize features to [0, 1] range
    min_vals = np.minimum(query_features.min(axis=0), reference_features.min(axis=0))
    max_vals = np.maximum(query_features.max(axis=0), reference_features.max(axis=0))
    query_norm = (query_features - min_vals) / (max_vals - min_vals)
    reference_norm = (reference_features - min_vals) / (max_vals - min_vals)
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(query_norm - reference_norm)
    
    # Convert distance to similarity score
    max_distance = np.sqrt(len(query_features))  # Maximum possible distance in normalized space
    similarity = 1 - (distance / max_distance)
    
    return similarity

# def adaptive_similarity(query_embedding, reference_embedding):
#     dim = len(query_embedding)
#     if dim <= 2:
#         return euclidean_distance(query_embedding, reference_embedding)
#     else:
#         return cosine_similarity(query_embedding, reference_embedding)

def adaptive_similarity(query_features, reference_features, transformation_power=None):
    print('transformation_power:', transformation_power)
    dim = len(query_features)
    if dim <= 3:
        return low_dimensional_similarity(query_features, reference_features)
    elif dim <= 50:
        similarity = improved_cosine_similarity(query_features, reference_features)
        power = transformation_power if transformation_power is not None else 1.5
        return similarity ** power
    else:
        similarity = improved_cosine_similarity(query_features, reference_features)
        power = transformation_power if transformation_power is not None else 2
        return similarity ** power

# "To account for the difference in dimensionality between MFCC and VGGish features, we could apply a scaling factor based on the number of dimensions."
# TODO not yet used
def get_similarity(query_embedding, reference_embedding):
    similarity = improved_cosine_similarity(query_embedding, reference_embedding)
    
    # Optional: Apply a scaling factor based on dimensionality
    dim = len(query_embedding)
    scaling_factor = np.log(dim) / np.log(128)  # Normalize to VGGish dimensionality
    
    return similarity * scaling_factor

async def socket_server(websocket, path):
    global reference_embeddings, normalization_stats
    url_components = urlparse(path)
    request_path = url_components.path
    query_params = parse_qs(url_components.query)
    
    try:
        message = await websocket.recv()
        start = time.time()
        
        reference_embedding_path = query_params.get('reference_embedding_path', [None])[0]
        reference_embedding_key = query_params.get('reference_embedding_key', [None])[0]
        transformation_power = query_params.get('transformation_power', [None])[0]
        
        z_score_normalisation_reference_features_paths = query_params.get('zScoreNormalisationReferenceFeaturesPaths', [None])[0]
        z_score_normalisation_train_features_file_path = query_params.get('zScoreNormalisationTrainFeaturesPath', [None])[0]

        dynamic_components = query_params.get('dynamicComponents', [False])[0]
        feature_indices = query_params.get('featureIndices', [None])[0]

        if transformation_power is not None:
            transformation_power = float(transformation_power)
        
        # Load reference embeddings
        if reference_embedding_path is not None and os.path.exists(reference_embedding_path) and reference_embedding_path not in reference_embeddings:
            print(f"Loading reference embeddings from {reference_embedding_path}")
            with open(reference_embedding_path, 'r') as f:
                reference_embeddings[reference_embedding_path] = json.load(f)
        
        query_embedding = np.array(json.loads(message))
        print(f"embeddings shape: {query_embedding.shape}")
        
        if ',' in reference_embedding_key:
            reference_embedding_keys = reference_embedding_key.split(',')
            reference_embedding = np.concatenate([reference_embeddings[reference_embedding_path][key] for key in reference_embedding_keys], axis=0)
        else:
            reference_embedding = np.array(reference_embeddings[reference_embedding_path][reference_embedding_key])
        
        # Apply Z-score normalization if paths are provided
        if z_score_normalisation_reference_features_paths or z_score_normalisation_train_features_file_path:
            norm_key = get_normalization_key(
                z_score_normalisation_reference_features_paths,
                z_score_normalisation_train_features_file_path
            )
            
            if norm_key not in normalization_stats:
                print(f"Computing global statistics for {norm_key}")
                ref_paths = z_score_normalisation_reference_features_paths.split(',') if z_score_normalisation_reference_features_paths else []
                mean, std = compute_global_statistics(
                    ref_paths,
                    z_score_normalisation_train_features_file_path,
                    reference_embedding_key
                )
                normalization_stats[norm_key] = (mean, std)
            
            # Apply normalization
            mean, std = normalization_stats[norm_key]
            query_embedding = z_score_normalize(query_embedding, mean, std)
            reference_embedding = z_score_normalize(reference_embedding, mean, std)
        
        if dynamic_components and feature_indices is not None:
            feature_indices = [int(i) for i in feature_indices.split(',')]
            query_embedding = query_embedding[feature_indices]
            reference_embedding = reference_embedding[feature_indices]

        # Compute similarity using existing methods
        if request_path == '/cosine':
            fitness = cosine_similarity(query_embedding, reference_embedding)
        elif request_path == '/improved_cosine':
            fitness = improved_cosine_similarity(query_embedding, reference_embedding)
        elif request_path == '/euclidean':
            fitness = euclidean_distance(query_embedding, reference_embedding)
        elif request_path == '/adaptive':
            fitness = adaptive_similarity(query_embedding, reference_embedding, transformation_power)
        else:
            raise ValueError(f"Unknown endpoint: {request_path}")
        
        fitness = np.clip(fitness, 0, 1)  # Ensure fitness is between 0 and 1
        print('fitness:', fitness)
        
        response = {'status': 'received standalone audio', 'fitness': float(fitness)}
        await websocket.send(json.dumps(response))
        
        end = time.time()
        print('Time taken to evaluate fitness:', end - start)
        
    except Exception as e:
        print('quality: Exception', e)
        response = {'status': 'ERROR', 'error': str(e)}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run a WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.') # e.g for the ROBIN-HPC
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='quality_ref_features', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
args = parser.parse_args()

# set PROCESS_TITLE as either the environment variable or the default value
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# set PORT as either the environment variable or the default value
PORT = int(os.environ.get('PORT', args.port))

HOST = args.host

# if the host-info-file is not empty
if args.host_info_file:
    # automatically assign the host IP from the machine's hostname
    if not args.force_host:
        HOST = os.uname().nodename
    # HOST = socket.gethostname()
    print("HOST", HOST)
    # the host-info-file name ends with "host-" and an index number: host-0, host-1, etc.
    # - for each comonent of that index number, add that number plus 1 to PORT and assign to the variable PORT

    PORT = filepath_to_port(args.host_info_file)

    # write the host IP and port to the host-info-file
    with open(args.host_info_file, 'w') as f:
        f.write('{}:{}'.format(HOST, PORT))


MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting quality ref features WebSocket server at ws://{}:{}'.format(HOST, PORT))

# Start the WebSocket server with supplied command line arguments
start_server = websockets.serve(socket_server, 
                                HOST, 
                                PORT,
                                max_size=MAX_MESSAGE_SIZE,
                                ping_timeout=None,
                                ping_interval=None)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()