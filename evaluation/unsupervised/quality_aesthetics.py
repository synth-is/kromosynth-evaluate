# Audiobox Aesthetics Evaluation Service
# Websocket service that evaluates audio using the audiobox-aesthetics model
# Returns 4 aesthetic dimensions: CE, CU, PC, PQ
# - CE: Content Enjoyment
# - CU: Content Usefulness
# - PC: Production Complexity
# - PQ: Production Quality

import asyncio
import websockets
import websockets.exceptions
import json
import argparse
import numpy as np
import os
from setproctitle import setproctitle
import time
from urllib.parse import urlparse, parse_qs
import torch
import torchaudio

import sys
sys.path.append('../..')
from evaluation.util import filepath_to_port

# Global model predictor
PREDICTOR = None
SAMPLE_RATE = 48000

def initialize_model(checkpoint_path=None):
    """
    Initialize the audiobox-aesthetics model predictor.

    Args:
        checkpoint_path: Optional path to model checkpoint. If None, uses default from HuggingFace.

    Returns:
        Initialized predictor
    """
    try:
        from audiobox_aesthetics.infer import initialize_predictor

        if checkpoint_path:
            print(f"Loading audiobox-aesthetics model from checkpoint: {checkpoint_path}")
            # The parameter name is 'ckpt', not 'checkpoint_path'
            predictor = initialize_predictor(ckpt=checkpoint_path)
        else:
            print("Loading audiobox-aesthetics model from HuggingFace (facebook/audiobox-aesthetics)")
            predictor = initialize_predictor()

        # Log device information
        if hasattr(predictor, 'model') and hasattr(predictor.model, 'device'):
            device = predictor.model.device
            print(f"✓ Model loaded on device: {device}")
        elif torch.backends.mps.is_available():
            print("✓ MPS (Apple Silicon GPU) is available and will be used")

        print("✓ Audiobox-aesthetics model loaded successfully")
        return predictor
    except ImportError as e:
        print(f"✗ Error: audiobox_aesthetics not installed. Run: pip install audiobox_aesthetics")
        raise e
    except Exception as e:
        print(f"✗ Error loading audiobox-aesthetics model: {e}")
        raise e

def evaluate_audio_aesthetics(audio_data, sample_rate):
    """
    Evaluate audio aesthetics using the audiobox-aesthetics model.

    Args:
        audio_data: numpy array of audio samples (float32)
        sample_rate: audio sample rate

    Returns:
        dict with keys: CE, CU, PC, PQ
    """
    global PREDICTOR

    if PREDICTOR is None:
        raise RuntimeError("Model predictor not initialized")

    # Convert numpy array to torch tensor
    wav_tensor = torch.from_numpy(audio_data).float()

    # Ensure tensor has correct shape (channels, samples)
    # If 1D, add channel dimension
    if wav_tensor.ndim == 1:
        wav_tensor = wav_tensor.unsqueeze(0)  # Shape: (1, samples)
    elif wav_tensor.ndim == 2:
        # If already 2D, ensure it's (channels, samples) not (samples, channels)
        if wav_tensor.shape[0] > wav_tensor.shape[1]:
            wav_tensor = wav_tensor.transpose(0, 1)

    # Run inference using torch tensor input
    # Format: [{"path": tensor, "sample_rate": sr}]
    result = PREDICTOR.forward([{"path": wav_tensor, "sample_rate": sample_rate}])

    # Result is a list, get first element (our single audio input)
    if isinstance(result, list) and len(result) > 0:
        result = result[0]

    return result

async def socket_server(websocket, path):
    """
    Websocket server handler for audiobox-aesthetics evaluation.

    Supports two output modes via query parameter 'output_mode':
    - 'all' (default): Returns all 4 dimension scores as taggedPredictions
    - 'top': Returns only the highest scoring dimension as fitness

    Query parameters:
    - output_mode: 'all' or 'top' (default: 'all')
    """
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    print(f'🔌 New WebSocket connection from {client_id}, path: {path}')

    try:
        # Parse query parameters
        url_components = urlparse(path)
        query_params = parse_qs(url_components.query)
        output_mode = query_params.get('output_mode', ['all'])[0]

        # Wait for message
        print(f'⏳ [{client_id}] Waiting for message...')
        message = await websocket.recv()
        print(f'📨 [{client_id}] Received message type: {type(message)}, size: {len(message) if isinstance(message, (str, bytes)) else "N/A"}')

        # Handle text message (JSON)
        if isinstance(message, str):
            try:
                data = json.loads(message)

                # Handle getKeys request (returns dimension keys for classification)
                if isinstance(data, dict) and data.get('getKeys') is True:
                    # Return dimension keys (no underscores to avoid parsing issues)
                    # CE = Content Enjoyment, CU = Content Usefulness
                    # PC = Production Complexity, PQ = Production Quality
                    keys = ['CE', 'CU', 'PC', 'PQ']
                    await websocket.send(json.dumps(keys))
                    print(f"✓ Returned {len(keys)} dimension keys: {keys}")
                    return
                else:
                    print(f"⚠ Unexpected JSON data: {data}")
                    await websocket.send(json.dumps({'status': 'ERROR: Expected binary audio data or getKeys request'}))
                    return
            except json.JSONDecodeError:
                pass

        # Handle binary message (audio buffer)
        if isinstance(message, bytes):
            start = time.time()

            # Received binary message (audio buffer)
            audio_data = message
            print(f'Audio data received for aesthetics evaluation (output_mode={output_mode})')

            # Convert to numpy array
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            print(f'  Audio buffer size: {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.2f}s at {SAMPLE_RATE}Hz)')

            # Evaluate aesthetics
            result = evaluate_audio_aesthetics(audio_data, SAMPLE_RATE)

            # Extract scores
            ce_score = float(result.get('CE', 0))
            cu_score = float(result.get('CU', 0))
            pc_score = float(result.get('PC', 0))
            pq_score = float(result.get('PQ', 0))

            end = time.time()
            print(f'Aesthetics scores - CE: {ce_score:.3f}, CU: {cu_score:.3f}, PC: {pc_score:.3f}, PQ: {pq_score:.3f}')
            print(f'Time taken to evaluate: {end - start:.3f}s')

            # Prepare response based on output mode
            if output_mode == 'top':
                # Return only the top scoring dimension (fitness mode)
                scores = {
                    'CE': ce_score,
                    'CU': cu_score,
                    'PC': pc_score,
                    'PQ': pq_score
                }

                # Find dimension with highest score
                top_dimension = max(scores, key=scores.get)
                top_score = scores[top_dimension]
                dimension_index = list(scores.keys()).index(top_dimension)

                # Map dimension codes to full names (for logging only)
                dimension_names = {
                    'CE': 'Content Enjoyment',
                    'CU': 'Content Usefulness',
                    'PC': 'Production Complexity',
                    'PQ': 'Production Quality'
                }

                fitness_value = {
                    'top_score': top_score,
                    'index': dimension_index,
                    'top_score_class': top_dimension  # Return simple code (CE, CU, PC, PQ)
                }

                response = {
                    'status': 'received standalone audio',
                    'fitness': fitness_value
                }
                print(f'  → Returning top score: {top_dimension} ({dimension_names[top_dimension]}) = {top_score:.3f}')

            else:  # output_mode == 'all' (default)
                # Return all dimension scores (classification mode)
                # Use simple keys without underscores to avoid parsing issues in cell keys
                tagged_predictions = {
                    'CE': ce_score,
                    'CU': cu_score,
                    'PC': pc_score,
                    'PQ': pq_score
                }

                response = {
                    'status': 'received standalone audio',
                    'taggedPredictions': tagged_predictions
                }
                print(f'  → Returning all 4 dimension scores')

            await websocket.send(json.dumps(response))

        else:
            print(f'⚠ Unexpected message type: {type(message)}')
            response = {'status': 'ERROR: Expected binary audio data'}
            await websocket.send(json.dumps(response))

    except websockets.exceptions.ConnectionClosedOK:
        # Connection closed normally - this is fine, just log it
        print(f'ℹ WebSocket connection closed normally')
    except websockets.exceptions.ConnectionClosedError as e:
        # Connection closed with error - log but don't try to send response
        print(f'✗ WebSocket connection closed with error: {e}')
    except Exception as e:
        print(f'✗ Exception in audiobox-aesthetics evaluation: {e}')
        import traceback
        traceback.print_exc()

        # Only try to send error response if websocket is still open
        try:
            if websocket.open:
                response = {'status': f'ERROR: {str(e)}'}
                await websocket.send(json.dumps(response))
        except:
            # If sending fails, just log it
            print(f'✗ Could not send error response (connection already closed)')

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run Audiobox Aesthetics WebSocket server.')
parser.add_argument('--host', type=str, default='localhost',
                    help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False,
                    help='Force the host to be the one specified in the host argument.')
parser.add_argument('--port', type=int, default=8080,
                    help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=48000,
                    help='Sample rate of the audio data.')
parser.add_argument('--checkpoint-path', type=str, default=None,
                    help='Path to audiobox-aesthetics model checkpoint (optional, uses HuggingFace default if not specified).')
parser.add_argument('--process-title', type=str, default='quality_aesthetics',
                    help='Process title to use.')
parser.add_argument('--host-info-file', type=str, default='',
                    help='Host information file to use.')
args = parser.parse_args()

# Set global sample rate
SAMPLE_RATE = args.sample_rate

# Handle checkpoint path with SLURM localscratch support
CHECKPOINT_PATH = args.checkpoint_path
if CHECKPOINT_PATH and '/localscratch/' in CHECKPOINT_PATH:
    CHECKPOINT_PATH = CHECKPOINT_PATH.replace('/localscratch/<job-ID>',
                                              '/localscratch/' + os.environ.get('SLURM_JOB_ID', ''))

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

# Initialize model
print('=' * 80)
print('Audiobox Aesthetics Evaluation Service')
print('=' * 80)
print(f'Configuration:')
print(f'  Sample rate: {SAMPLE_RATE}Hz')
print(f'  Checkpoint: {CHECKPOINT_PATH if CHECKPOINT_PATH else "HuggingFace default"}')
print(f'  Process title: {PROCESS_TITLE}')
print('-' * 80)

PREDICTOR = initialize_model(CHECKPOINT_PATH)

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('=' * 80)
print(f'🚀 Starting Audiobox Aesthetics WebSocket server at ws://{HOST}:{PORT}')
print(f'   Output modes:')
print(f'     - ?output_mode=all  : Returns all 4 dimension scores (default)')
print(f'     - ?output_mode=top  : Returns only highest scoring dimension')
print(f'   Dimensions:')
print(f'     - CE (Content Enjoyment)')
print(f'     - CU (Content Usefulness)')
print(f'     - PC (Production Complexity)')
print(f'     - PQ (Production Quality)')
print('=' * 80)

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
