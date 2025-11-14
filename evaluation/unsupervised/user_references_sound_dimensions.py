# User Reference Sounds Dimensions Evaluation Service
# Websocket service that:
# 1. On startup, scans a folder of reference sounds and extracts MFCC features
# 2. Populates an HNSW vector database with the features
# 3. On evaluation, classifies audio buffers to the nearest reference sound
# 4. Returns reference sound ID as the classification dimension

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
import sys
import librosa
import hnswlib
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import hashlib

sys.path.append('../..')
from evaluation.util import filepath_to_port

# Global state
REFERENCE_SOUNDS = {}  # Dict mapping ref_id -> sound info
HNSW_INDEX = None
REF_SOUND_FOLDER = None
MFCC_DIMENSIONS = 96  # Configurable MFCC feature dimensions
SIMILARITY_METRIC = "cosine"  # or "l2"
SAMPLE_RATE = 22050
SOUND_EXTENSIONS = ['.wav', '.mp3', '.flac', '.aiff', '.ogg']

def extract_mfcc_features(audio_path: str, n_mfcc: int = 96) -> np.ndarray:
    """
    Extract MFCC features from an audio file.
    Returns a 96-dimensional feature vector.
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
        
        # Extract MFCC features
        # We want 96 dimensions total, so we'll compute 12 MFCCs with 8 time frames
        n_mfcc_base = 12
        hop_length = len(y) // 8  # 8 time frames
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_base, hop_length=hop_length)
        
        # Flatten and ensure we have exactly n_mfcc dimensions
        mfcc_flat = mfccs.flatten()
        
        if len(mfcc_flat) >= n_mfcc:
            # Take first n_mfcc values
            features = mfcc_flat[:n_mfcc]
        else:
            # Pad with zeros if needed
            features = np.pad(mfcc_flat, (0, n_mfcc - len(mfcc_flat)), mode='constant')
        
        return features.astype(np.float32)
    except Exception as e:
        print(f"✗ Error extracting MFCC from {audio_path}: {e}")
        return np.zeros(n_mfcc, dtype=np.float32)

def scan_reference_sounds_folder():
    """
    Scan the reference sounds folder and extract MFCC features from all audio files.
    Populate the HNSW index with the features.
    """
    global REFERENCE_SOUNDS, HNSW_INDEX, REF_SOUND_FOLDER, MFCC_DIMENSIONS, SIMILARITY_METRIC
    
    if not REF_SOUND_FOLDER or not os.path.exists(REF_SOUND_FOLDER):
        print(f"✗ Reference sounds folder not found: {REF_SOUND_FOLDER}")
        return False
    
    print(f"🔍 Scanning reference sounds folder: {REF_SOUND_FOLDER}")
    
    # Find all audio files
    audio_files = []
    for ext in SOUND_EXTENSIONS:
        audio_files.extend(Path(REF_SOUND_FOLDER).rglob(f'*{ext}'))
    
    if not audio_files:
        print(f"✗ No audio files found in {REF_SOUND_FOLDER}")
        return False
    
    print(f"📁 Found {len(audio_files)} audio files")
    
    # Extract features from all files
    REFERENCE_SOUNDS = {}
    features_list = []
    valid_files = []
    
    for i, audio_path in enumerate(audio_files):
        print(f"🎵 Processing {i+1}/{len(audio_files)}: {audio_path.name}")
        
        features = extract_mfcc_features(str(audio_path), MFCC_DIMENSIONS)
        
        # Create reference ID from file path (relative to reference folder)
        rel_path = audio_path.relative_to(Path(REF_SOUND_FOLDER))
        ref_id = str(rel_path).replace(os.sep, '_').replace('.', '_')
        
        REFERENCE_SOUNDS[ref_id] = {
            'ref_id': ref_id,
            'filename': audio_path.name,
            'relative_path': str(rel_path),
            'full_path': str(audio_path),
            'features': features
        }
        
        features_list.append(features)
        valid_files.append(ref_id)
    
    if not features_list:
        print("✗ No valid features extracted")
        return False
    
    # Create HNSW index
    features_array = np.array(features_list)
    num_elements = len(features_list)
    
    # Initialize HNSW index
    space = 'cosine' if SIMILARITY_METRIC == 'cosine' else 'l2'
    HNSW_INDEX = hnswlib.Index(space=space, dim=MFCC_DIMENSIONS)
    HNSW_INDEX.init_index(max_elements=num_elements, ef_construction=200, M=16)
    
    # Add all features to index
    for i, features in enumerate(features_list):
        HNSW_INDEX.add_items(features, i)
    
    # Set query parameters
    HNSW_INDEX.set_ef(50)
    
    print(f"✓ Created HNSW index with {num_elements} reference sounds")
    print(f"  Reference IDs: {list(REFERENCE_SOUNDS.keys())[:5]}{'...' if len(REFERENCE_SOUNDS) > 5 else ''}")
    
    return True

def classify_audio_buffer_to_reference(audio_buffer: np.ndarray, sample_rate: int = None) -> Dict:
    """
    Classify an audio buffer to the nearest reference sound using HNSW.
    
    Args:
        audio_buffer: Raw audio samples
        sample_rate: Sample rate of the audio (optional, will resample if needed)
    
    Returns:
        Classification result with reference sound info and similarity
    """
    global REFERENCE_SOUNDS, HNSW_INDEX, MFCC_DIMENSIONS
    
    if HNSW_INDEX is None or not REFERENCE_SOUNDS:
        return {
            'ref_id': None,
            'filename': 'no_references',
            'similarity': 0.0,
            'error': 'No reference sounds loaded'
        }
    
    try:
        # Resample if needed
        if sample_rate and sample_rate != SAMPLE_RATE:
            audio_buffer = librosa.resample(audio_buffer, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        
        # Extract MFCC features from the audio buffer
        if len(audio_buffer) < 1024:  # Minimum length for MFCC
            return {
                'ref_id': None,
                'filename': 'audio_too_short',
                'similarity': 0.0,
                'error': f'Audio buffer too short ({len(audio_buffer)} samples)'
            }
        
        # Extract MFCCs using the same method as reference sounds
        n_mfcc_base = 12
        hop_length = len(audio_buffer) // 8
        if hop_length < 1:
            hop_length = len(audio_buffer) // 4  # Fallback for very short audio
        
        mfccs = librosa.feature.mfcc(y=audio_buffer, sr=SAMPLE_RATE, n_mfcc=n_mfcc_base, hop_length=hop_length)
        mfcc_flat = mfccs.flatten()
        
        if len(mfcc_flat) >= MFCC_DIMENSIONS:
            query_features = mfcc_flat[:MFCC_DIMENSIONS]
        else:
            query_features = np.pad(mfcc_flat, (0, MFCC_DIMENSIONS - len(mfcc_flat)), mode='constant')
        
        query_features = query_features.astype(np.float32)
        
        # Query HNSW index
        labels, distances = HNSW_INDEX.knn_query(query_features.reshape(1, -1), k=1)
        
        # Get the best match
        best_idx = labels[0][0]
        best_distance = distances[0][0]
        
        # Convert distance to similarity (closer = higher similarity)
        if SIMILARITY_METRIC == 'cosine':
            similarity = 1.0 - best_distance  # Cosine distance -> similarity
        else:  # l2
            # For L2 distance, convert to similarity
            max_possible_distance = np.sqrt(MFCC_DIMENSIONS * 4)  # Rough estimate
            similarity = max(0.0, 1.0 - (best_distance / max_possible_distance))
        
        # Get reference sound info
        ref_ids = list(REFERENCE_SOUNDS.keys())
        if best_idx < len(ref_ids):
            best_ref_id = ref_ids[best_idx]
            ref_info = REFERENCE_SOUNDS[best_ref_id]
            
            return {
                'ref_id': best_ref_id,
                'filename': ref_info['filename'],
                'relative_path': ref_info['relative_path'],
                'similarity': float(similarity),
                'distance': float(best_distance),
                'hnsw_index': int(best_idx)
            }
        else:
            return {
                'ref_id': None,
                'filename': 'index_error',
                'similarity': 0.0,
                'error': f'HNSW index out of range: {best_idx}/{len(ref_ids)}'
            }
    
    except Exception as e:
        return {
            'ref_id': None,
            'filename': 'classification_error',
            'similarity': 0.0,
            'error': f'Classification failed: {str(e)}'
        }

async def socket_server(websocket, path):
    """
    Websocket server handler.
    Accepts requests with:
    1. {'getKeys': True} - returns list of reference sound IDs (dimensions)
    2. Audio buffer (numpy array) - returns classification to nearest reference sound
    """
    global REFERENCE_SOUNDS
    
    url_components = urlparse(path)
    request_path = url_components.path
    query_params = parse_qs(url_components.query)
    
    try:
        message = await websocket.recv()
        
        # Handle request for dimension keys (reference sound IDs)
        if isinstance(message, str):
            try:
                request_data = json.loads(message)
                if request_data.get('getKeys') is True:
                    # Return list of reference sound IDs as the dimension keys
                    ref_ids = list(REFERENCE_SOUNDS.keys())
                    await websocket.send(json.dumps(ref_ids))
                    print(f"✓ Returned {len(ref_ids)} reference sound dimension keys: {ref_ids[:3]}{'...' if len(ref_ids) > 3 else ''}")
                    return
            except json.JSONDecodeError:
                pass
        
        # Handle audio buffer for classification
        if isinstance(message, bytes):
            start = time.time()
            
            # Convert bytes to numpy array (assuming float32)
            audio_buffer = np.frombuffer(message, dtype=np.float32)
            print(f'✓ Received audio buffer with {len(audio_buffer)} samples for reference classification')
            
            # Classify to nearest reference sound
            classification_result = classify_audio_buffer_to_reference(audio_buffer)
            
            # Format response with taggedPredictions for QD search compatibility
            tagged_predictions = {}
            if classification_result['ref_id']:
                # Create predictions for all reference sounds, with the best match getting highest score
                for ref_id, ref_info in REFERENCE_SOUNDS.items():
                    if ref_id == classification_result['ref_id']:
                        tagged_predictions[f"REF_{ref_info['filename']}"] = classification_result['similarity']
                    else:
                        # Give other references a small base score
                        tagged_predictions[f"REF_{ref_info['filename']}"] = 0.01
            else:
                # If no match, give all references equal low scores
                for ref_id, ref_info in REFERENCE_SOUNDS.items():
                    tagged_predictions[f"REF_{ref_info['filename']}"] = 0.01
            
            end = time.time()
            print(f'✓ Classified to reference: {classification_result["filename"]} (similarity: {classification_result["similarity"]:.3f}) in {end - start:.3f}s')
            
            response = {
                'status': 'received standalone audio buffer',
                'taggedPredictions': tagged_predictions
            }
            await websocket.send(json.dumps(response))
    
    except Exception as e:
        print(f'✗ Exception in socket_server: {e}')
        import traceback
        traceback.print_exc()
        
        response = {'status': f'ERROR: {str(e)}'}
        await websocket.send(json.dumps(response))

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run User Reference Sounds Dimensions WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified in the host argument.')
parser.add_argument('--port', type=int, default=8080, help='Port number to run the WebSocket server on.')
parser.add_argument('--process-title', type=str, default='user_references_sound_dimensions', help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file to use.')
parser.add_argument('--reference-sounds-folder', type=str, required=True,
                    help='Path to folder containing reference sounds')
parser.add_argument('--similarity-metric', type=str, default='cosine', choices=['cosine', 'l2'],
                    help='Similarity metric to use for HNSW index')
parser.add_argument('--mfcc-dimensions', type=int, default=96, 
                    help='Number of MFCC dimensions to extract')
parser.add_argument('--sample-rate', type=int, default=22050,
                    help='Sample rate for audio processing')
args = parser.parse_args()

# Set global configuration
REF_SOUND_FOLDER = args.reference_sounds_folder
SIMILARITY_METRIC = args.similarity_metric
MFCC_DIMENSIONS = args.mfcc_dimensions
SAMPLE_RATE = args.sample_rate

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

# Scan and index reference sounds on startup
print(f"🔄 Scanning reference sounds from {REF_SOUND_FOLDER}...")
if not scan_reference_sounds_folder():
    print("⚠ WARNING: Could not load reference sounds. Service will start but won't classify correctly.")
    print("   Check the reference sounds folder path and ensure it contains audio files.")

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print(f'🚀 Starting Reference Sounds Dimensions WebSocket server at ws://{HOST}:{PORT}')
print(f'   Reference sounds folder: {REF_SOUND_FOLDER}')
print(f'   Similarity metric: {SIMILARITY_METRIC}')
print(f'   MFCC dimensions: {MFCC_DIMENSIONS}')
print(f'   Sample rate: {SAMPLE_RATE}')
print(f'   Reference sounds loaded: {len(REFERENCE_SOUNDS)}')

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