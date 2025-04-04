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

def get_mfcc_focus_indices(feature_type, focus_type):
    """
    Get the appropriate feature indices for different MFCC focus types and feature extraction methods.
    
    Parameters:
    -----------
    feature_type : str
        The type of MFCC feature extraction used ('mfcc', 'mfcc-sans0', 'mfcc-statistics', 'mfcc-sans0-statistics')
    focus_type : str
        Type of focus: 'timbre', 'spectral', 'energy', 'temporal', 'full'
    
    Returns:
    --------
    indices : list
        List of indices to select from the feature vector
    """
    # Determine if we're using statistics-based features
    is_statistics = 'statistics' in feature_type
    # Determine if we're using sans0 features
    is_sans0 = 'sans0' in feature_type
    
    # For statistical features, the format is:
    # [means(13), std_devs(13), mins(13), maxes(13), der_means(13), der_std_devs(13), der_mins(13), der_maxes(13)]
    # For sans0, we have 12 instead of 13 coefficients
    
    # Regular MFCC features (non-statistics)
    if not is_statistics:
        # For regular features with means, std_devs, first-order differences
        # Format: [means(13), std_devs(13), first_order_diffs(13)]
        # For sans0, we have 12 instead of 13 coefficients
        n_coeffs = 12 if is_sans0 else 13
        offset = 0 if not is_sans0 else 0  # No offset needed since first coeff is already removed
        
        if focus_type == 'timbre':
            # Coefficients 1-5 (indices 1-5 or 0-4 for sans0)
            start_idx = 1 if not is_sans0 else 0
            indices = list(range(start_idx, start_idx + 5))
            # Add corresponding std_devs and first_order_diffs
            indices += [i + n_coeffs for i in indices]
            indices += [i + 2*n_coeffs for i in indices[:n_coeffs]]
            return indices
            
        elif focus_type == 'spectral':
            # Coefficients 6-12 (indices 6-12 or 5-11 for sans0)
            start_idx = 6 if not is_sans0 else 5
            end_idx = 13 if not is_sans0 else 12
            indices = list(range(start_idx, end_idx))
            # Add corresponding std_devs and first_order_diffs
            indices += [i + n_coeffs for i in indices]
            indices += [i + 2*n_coeffs for i in indices[:n_coeffs]]
            return indices
            
        elif focus_type == 'energy':
            if is_sans0:
                raise ValueError("Energy focus is not compatible with sans0 features as they exclude coefficient 0")
            # Coefficient 0 (energy)
            indices = [0]
            # Add corresponding std_dev and first_order_diff
            indices += [0 + n_coeffs]
            indices += [0 + 2*n_coeffs]
            return indices
            
        elif focus_type == 'temporal':
            # For temporal focus, we'll use all coefficients but focus on derivatives
            # In regular features, these are the first_order_diffs
            return list(range(2*n_coeffs, 3*n_coeffs))
            
        elif focus_type == 'full':
            # Use all coefficients
            return list(range(3*n_coeffs))
            
    else:
        # For statistical features (mfcc-statistics or mfcc-sans0-statistics)
        # Format: [means(13), std_devs(13), mins(13), maxes(13), der_means(13), der_std_devs(13), der_mins(13), der_maxes(13)]
        n_coeffs = 12 if is_sans0 else 13
        
        if focus_type == 'timbre':
            # Coefficients 1-5 (indices 1-5 or 0-4 for sans0)
            start_idx = 1 if not is_sans0 else 0
            timbre_indices = list(range(start_idx, start_idx + 5))
            # Gather all related statistics for these coefficients
            indices = []
            for i in timbre_indices:
                # Add mean, std_dev, min, max
                indices.append(i)  # mean
                indices.append(i + n_coeffs)  # std_dev
                indices.append(i + 2*n_coeffs)  # min
                indices.append(i + 3*n_coeffs)  # max
                # Add der_mean, der_std_dev, der_min, der_max
                indices.append(i + 4*n_coeffs)  # der_mean
                indices.append(i + 5*n_coeffs)  # der_std_dev
                indices.append(i + 6*n_coeffs)  # der_min
                indices.append(i + 7*n_coeffs)  # der_max
            return indices
            
        elif focus_type == 'spectral':
            # Coefficients 6-12 (indices 6-12 or 5-11 for sans0)
            start_idx = 6 if not is_sans0 else 5
            end_idx = 13 if not is_sans0 else 12
            spectral_indices = list(range(start_idx, end_idx))
            # Gather all related statistics for these coefficients
            indices = []
            for i in spectral_indices:
                # Add mean, std_dev, min, max
                indices.append(i)  # mean
                indices.append(i + n_coeffs)  # std_dev
                indices.append(i + 2*n_coeffs)  # min
                indices.append(i + 3*n_coeffs)  # max
                # Add der_mean, der_std_dev, der_min, der_max
                indices.append(i + 4*n_coeffs)  # der_mean
                indices.append(i + 5*n_coeffs)  # der_std_dev
                indices.append(i + 6*n_coeffs)  # der_min
                indices.append(i + 7*n_coeffs)  # der_max
            return indices
            
        elif focus_type == 'energy':
            if is_sans0:
                raise ValueError("Energy focus is not compatible with sans0 features as they exclude coefficient 0")
            # Coefficient 0 (energy)
            i = 0
            indices = [
                i, i + n_coeffs, i + 2*n_coeffs, i + 3*n_coeffs,  # mean, std_dev, min, max
                i + 4*n_coeffs, i + 5*n_coeffs, i + 6*n_coeffs, i + 7*n_coeffs  # der_mean, der_std_dev, der_min, der_max
            ]
            return indices
            
        elif focus_type == 'temporal':
            # For temporal focus, we'll focus on derivatives
            # Get all derivative statistics
            indices = list(range(4*n_coeffs, 8*n_coeffs))
            return indices
            
        elif focus_type == 'full':
            # Use all coefficients and statistics
            return list(range(8*n_coeffs))

    # Default case - return all indices
    return list(range(100))  # Large enough for all feature types

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

        dynamic_components = str_to_bool(query_params.get('dynamicComponents', ['False'])[0])
        feature_indices = query_params.get('featureIndices', [None])[0]
        
        # Get MFCC focus parameter
        mfcc_focus = query_params.get('mfcc_focus', [None])[0]
        
        if transformation_power is not None:
            transformation_power = float(transformation_power)
        
        # Load reference embeddings
        if reference_embedding_path is not None and os.path.exists(reference_embedding_path) and reference_embedding_path not in reference_embeddings:
            print(f"Loading reference embeddings from {reference_embedding_path}")
            with open(reference_embedding_path, 'r') as f:
                reference_embeddings[reference_embedding_path] = json.load(f)
        
        query_embedding = np.array(json.loads(message))
        print(f"embeddings shape: {query_embedding.shape}")
        
        # Handle reference embedding extraction
        if ',' in reference_embedding_key:
            reference_embedding_keys = reference_embedding_key.split(',')
            reference_embedding = np.concatenate([reference_embeddings[reference_embedding_path][key] for key in reference_embedding_keys], axis=0)
        else:
            reference_embedding = np.array(reference_embeddings[reference_embedding_path][reference_embedding_key])
        
        # If MFCC focus is specified, override feature_indices
        if mfcc_focus:
            print(f"Applying MFCC focus: {mfcc_focus}")
            feature_indices = get_mfcc_focus_indices(reference_embedding_key, mfcc_focus)
            feature_indices = ','.join(map(str, feature_indices))
            dynamic_components = True
            print(f"Generated feature indices for {mfcc_focus} focus: {feature_indices}")
        
        # Apply Z-score normalization if paths are provided
        if z_score_normalisation_reference_features_paths or z_score_normalisation_train_features_file_path:
            # Create a unique normalization key
            focus_suffix = f"-{mfcc_focus}" if mfcc_focus else ""
            norm_key = get_normalization_key(
                z_score_normalisation_reference_features_paths,
                z_score_normalisation_train_features_file_path
            ) + f"-{reference_embedding_key}" + focus_suffix
            
            if norm_key not in normalization_stats:
                print(f"Computing global statistics for {norm_key}")
                ref_paths = z_score_normalisation_reference_features_paths.split(',') if z_score_normalisation_reference_features_paths else []
                
                # Use standard statistics computation
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
        
        # Apply dynamic component selection if specified
        if dynamic_components and feature_indices is not None:
            if isinstance(feature_indices, str):
                feature_indices = [int(i) for i in feature_indices.split(',')]
            
            # Make sure indices are within bounds
            valid_indices = [i for i in feature_indices if i < len(query_embedding) and i < len(reference_embedding)]
            if len(valid_indices) != len(feature_indices):
                print(f"Warning: Some feature indices were out of bounds. Using {len(valid_indices)} valid indices.")
                feature_indices = valid_indices
                
            if len(feature_indices) == 0:
                raise ValueError("No valid feature indices after filtering")
                
            query_embedding = query_embedding[feature_indices]
            reference_embedding = reference_embedding[feature_indices]
            print(f"After applying feature indices, embeddings shape: {query_embedding.shape}")

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
        import traceback
        traceback.print_exc()
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