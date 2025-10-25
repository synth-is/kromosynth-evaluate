"""
WebSocket server for musical quality evaluation.

Provides configurable quality assessment including:
- Phase 1: Noise filtering (SNR, spectral entropy, etc.)
- Phase 2: Spectral clarity (harmonic content, stability, etc.)
- Phase 3: VI coherence (multi-pitch consistency) [Future]

Configuration can be passed via websocket URL query parameters or JSON message.
"""

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
import base64
from setproctitle import setproctitle
from urllib.parse import urlparse, parse_qs
import time

import sys
sys.path.append('../..')
from measurements.quality.noise_detection import calculate_noise_filter_scores
from measurements.quality.spectral_clarity import calculate_spectral_clarity_scores
from measurements.quality.vi_coherence import (
    calculate_vi_coherence_scores,
    requires_multi_pitch_evaluation
)
from evaluation.util import filepath_to_port


# Default configuration presets
CONFIG_PRESETS = {
    "noise_only": {
        "noise_filters": {
            "enabled": True,
            "check_snr": True,
            "snr_threshold": 6.0,
            "snr_weight": 0.3,
            "check_spectral_entropy": True,
            "spectral_entropy_max": 0.95,
            "spectral_entropy_weight": 0.25,
            "require_attack": True,
            "attack_weight": 0.25,
            "require_spectral_peaks": True,
            "spectral_peaks_weight": 0.2,
            "require_periodicity": False
        },
        "spectral_clarity": {
            "enabled": False
        },
        "vi_coherence": {
            "enabled": False
        },
        "score_aggregation": "weighted_average"
    },
    
    "spectral_clarity": {
        "noise_filters": {
            "enabled": True,
            "check_snr": True,
            "snr_threshold": 6.0,
            "snr_weight": 0.2,
            "check_spectral_entropy": True,
            "spectral_entropy_max": 0.95,
            "spectral_entropy_weight": 0.1,
            "require_attack": True,
            "attack_weight": 0.1,
            "require_spectral_peaks": True,
            "spectral_peaks_weight": 0.1
        },
        "spectral_clarity": {
            "enabled": True,
            "weight": 0.5,
            "measure_concentration": True,
            "concentration_weight": 0.3,
            "measure_hnr": True,
            "hnr_weight": 0.3,
            "measure_stability": True,
            "stability_weight": 0.2,
            "measure_harmonicity": True,
            "harmonicity_weight": 0.2
        },
        "vi_coherence": {
            "enabled": False
        },
        "score_aggregation": "weighted_average"
    },
    
    "vi_focused": {
        "noise_filters": {
            "enabled": True,
            "check_snr": True,
            "snr_threshold": 8.0,  # Higher for VI
            "snr_weight": 0.15,
            "check_spectral_entropy": True,
            "spectral_entropy_max": 0.9,
            "spectral_entropy_weight": 0.1,
            "require_attack": True,
            "attack_weight": 0.1,
            "require_spectral_peaks": True,
            "spectral_peaks_weight": 0.15
        },
        "spectral_clarity": {
            "enabled": True,
            "weight": 0.3,
            "measure_concentration": True,
            "concentration_weight": 0.25,
            "measure_hnr": True,
            "hnr_weight": 0.4,
            "measure_stability": True,
            "stability_weight": 0.25,
            "measure_harmonicity": True,
            "harmonicity_weight": 0.1
        },
        "vi_coherence": {
            "enabled": True,
            "weight": 0.5,
            "test_pitches": [-12, 0, 12],  # Octave below, center, octave above
            "measure_pitch_coherence": True,
            "coherence_weight": 0.5,
            "measure_attack_consistency": True,
            "attack_consistency_weight": 0.3,
            "measure_spectral_stability": True,
            "spectral_stability_weight": 0.2,
            "overall_score_weight": 0.0,  # Usually don't use overall score separately
            "min_snr_for_vi": 8.0,
            "min_clarity_for_vi": 0.6
        },
        "score_aggregation": "weighted_average"
    }
}


def parse_config_from_url(path):
    """
    Parse configuration from websocket URL query parameters.
    
    Examples:
        /musicality?config_preset=noise_only
        /musicality?noise_filters.snr_threshold=8.0&spectral_clarity.enabled=true
    """
    parsed = urlparse(path)
    params = parse_qs(parsed.query)
    
    config = None
    
    # Check for preset
    if 'config_preset' in params:
        preset_name = params['config_preset'][0]
        if preset_name in CONFIG_PRESETS:
            config = CONFIG_PRESETS[preset_name].copy()
    
    # If no preset, start with default (noise_only)
    if config is None:
        config = CONFIG_PRESETS["noise_only"].copy()
    
    # Override with individual parameters
    for key, values in params.items():
        if key == 'config_preset':
            continue
        
        # Parse nested config keys like "noise_filters.snr_threshold"
        parts = key.split('.')
        if len(parts) == 2:
            section, param = parts
            value = values[0]
            
            # Convert to appropriate type
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            else:
                try:
                    value = float(value)
                except:
                    pass
            
            if section in config:
                config[section][param] = value
    
    return config


def aggregate_scores(scores, weights, aggregation_method):
    """
    Combine multiple scores into a single fitness value.
    
    Args:
        scores: Dict of score name to value
        weights: Dict of score name to weight
        aggregation_method: 'weighted_average', 'geometric_mean', or 'minimum'
        
    Returns:
        float: Aggregated fitness score (0-1)
    """
    if not scores:
        return 0.0
    
    if aggregation_method == "weighted_average":
        total_weight = sum(weights.get(k, 1.0) for k in scores)
        if total_weight == 0:
            return 0.0
        fitness = sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_weight
        
    elif aggregation_method == "geometric_mean":
        total_weight = sum(weights.get(k, 1.0) for k in scores)
        if total_weight == 0:
            return 0.0
        fitness = np.prod([scores[k] ** weights.get(k, 1.0) for k in scores]) ** (1/total_weight)
        
    elif aggregation_method == "minimum":
        fitness = min(scores.values())
        
    else:
        # Default to weighted average
        total_weight = sum(weights.get(k, 1.0) for k in scores)
        if total_weight == 0:
            return 0.0
        fitness = sum(scores[k] * weights.get(k, 1.0) for k in scores) / total_weight
    
    return fitness


def evaluate_musicality(audio_data, sample_rate, config):
    """
    Main evaluation function for musical quality.
    
    Args:
        audio_data: Audio samples as numpy array (or dict for multi-pitch)
        sample_rate: Sample rate in Hz
        config: Configuration dictionary
        
    Returns:
        dict: Evaluation results with fitness score and metadata
    """
    start_time = time.time()
    
    all_scores = {}
    all_weights = {}
    vi_metadata = {}  # Store VI metadata separately from numeric scores
    
    # Handle single audio buffer or multi-pitch dict
    is_multi_pitch = isinstance(audio_data, dict)
    single_audio = audio_data if not is_multi_pitch else audio_data.get(0, list(audio_data.values())[0])
    
    # Phase 1: Noise Filtering
    if config["noise_filters"]["enabled"]:
        noise_config = config["noise_filters"]
        noise_result = calculate_noise_filter_scores(single_audio, sample_rate, noise_config)
        
        # Check for rejection
        if noise_result.get("rejected", False):
            elapsed = time.time() - start_time
            return {
                "fitness": {
                    "top_score": 0.0,
                    "top_score_class": "oneShot"
                },
                "rejected": True,
                "reason": noise_result["reason"],
                "metadata": noise_result,
                "evaluation_time": elapsed
            }
        
        # Add noise filter scores
        noise_scores = noise_result.get("scores", {})
        for key, value in noise_scores.items():
            score_name = f"noise_{key}"
            all_scores[score_name] = value
            all_weights[score_name] = noise_config.get(f"{key}_weight", 1.0)
    
    # Phase 2: Spectral Clarity
    if config["spectral_clarity"]["enabled"]:
        clarity_config = config["spectral_clarity"]
        clarity_scores = calculate_spectral_clarity_scores(single_audio, sample_rate, clarity_config)
        
        for key, value in clarity_scores.items():
            score_name = f"clarity_{key}"
            all_scores[score_name] = value
            all_weights[score_name] = clarity_config.get(f"{key}_weight", 1.0) * clarity_config["weight"]
    
    # Phase 3: VI Coherence (if enabled and multi-pitch data provided)
    if config["vi_coherence"]["enabled"]:
        vi_config = config["vi_coherence"]
        
        if is_multi_pitch:
            # We have multi-pitch data, evaluate VI coherence
            vi_scores = calculate_vi_coherence_scores(audio_data, sample_rate, vi_config)
            
            # Get overall VI weight (defaults to 0.5 if not specified)
            vi_overall_weight = vi_config.get("weight", 0.5)
            
            # Map returned keys to config weight keys
            # vi_coherence.py returns keys like 'vi_pitch_coherence', 'vi_attack_consistency', etc.
            # Only numeric scores should be added to all_scores for fitness calculation
            weight_mapping = {
                'vi_pitch_coherence': 'coherence_weight',
                'vi_attack_consistency': 'attack_consistency_weight', 
                'vi_spectral_stability': 'spectral_stability_weight',
                'vi_overall_score': 'overall_score_weight',  # Include overall score
            }
            
            # Store metadata separately (don't add to scores for fitness calc)
            # vi_metadata is defined at function level
            
            for key, value in vi_scores.items():
                # Only add numeric scores to all_scores
                if key in weight_mapping:
                    all_scores[key] = value
                    weight_key = weight_mapping[key]
                    metric_weight = vi_config.get(weight_key, 1.0)
                    all_weights[key] = metric_weight * vi_overall_weight
                else:
                    # Store metadata for later use
                    vi_metadata[key] = value
        else:
            # Check if this sound warrants multi-pitch evaluation
            requires_vi_eval = requires_multi_pitch_evaluation(all_scores, vi_config)
            
            if requires_vi_eval:
                elapsed = time.time() - start_time
                return {
                    "fitness": {
                        "top_score": None,
                        "top_score_class": "oneShot"
                    },
                    "requires_multi_pitch_evaluation": True,
                    "test_pitches": vi_config["test_pitches"],
                    "preliminary_scores": all_scores,
                    "evaluation_time": elapsed
                }
    
    # Aggregate all scores
    aggregation_method = config.get("score_aggregation", "weighted_average")
    fitness = aggregate_scores(all_scores, all_weights, aggregation_method)
    
    # Determine sound type classification
    sound_type = "oneShot"  # Default classification
    confidence = 0.5
    
    # Use VI metadata if available (from Phase 3)
    if vi_metadata and 'sound_type' in vi_metadata:
        sound_type = vi_metadata['sound_type']
        confidence = vi_metadata.get('sound_type_confidence', 0.5)
    elif "vi_pitch_coherence" in all_scores:
        # Fallback: infer from VI coherence score
        vi_score = all_scores.get("vi_pitch_coherence", 0)
        if vi_score > 0.7:
            sound_type = "VIworthy"
            confidence = 0.9
    elif "clarity_hnr" in all_scores:
        # Fallback: use spectral clarity scores
        clarity_score = all_scores.get("clarity_hnr", 0)
        snr_score = all_scores.get("noise_snr", 0)
        if clarity_score > 0.6 and snr_score > 0.8:
            sound_type = "oneShot"
            confidence = 0.8
    
    elapsed = time.time() - start_time
    
    return {
        "fitness": {
            "top_score": fitness,
            "top_score_class": sound_type
        },
        "rejected": False,
        "scores": all_scores,
        "sound_type_confidence": confidence,
        "evaluation_time": elapsed
    }


async def socket_server(websocket, path):
    """WebSocket handler for musicality evaluation."""
    try:
        # Parse config from URL
        config = parse_config_from_url(path)
        
        # Wait for message
        message = await websocket.recv()
        
        if isinstance(message, bytes):
            # Single audio buffer (Phase 1 & 2 only)
            audio_data = np.frombuffer(message, dtype=np.float32)
            
            result = evaluate_musicality(audio_data, sample_rate, config)
            await websocket.send(json.dumps(result))
            
        elif isinstance(message, str):
            # JSON message - could contain config or multi-pitch data
            data = json.loads(message)
            
            if "audio_buffers" in data:
                # Multi-pitch evaluation (Phase 3)
                # Decode base64 audio buffers
                audio_data = {}
                for pitch_str, b64_data in data["audio_buffers"].items():
                    pitch = int(pitch_str)
                    buffer_bytes = base64.b64decode(b64_data)
                    audio_data[pitch] = np.frombuffer(buffer_bytes, dtype=np.float32)
                
                # Merge config if provided (override specific keys while keeping URL preset)
                if "config" in data and data["config"]:
                    config_override = data["config"]
                    # Deep merge: update nested dictionaries
                    for section_key, section_values in config_override.items():
                        if section_key in config and isinstance(section_values, dict):
                            config[section_key].update(section_values)
                        else:
                            config[section_key] = section_values
                    print(f"Config merged with override: {json.dumps(config, indent=2)}")
                
                result = evaluate_musicality(audio_data, sample_rate, config)
                await websocket.send(json.dumps(result))
                
            elif "config" in data:
                # Config update only
                config = data["config"]
                response = {"status": "config_updated"}
                await websocket.send(json.dumps(response))
            else:
                response = {"status": "ERROR", "message": "Invalid message format"}
                await websocket.send(json.dumps(response))
    
    except Exception as e:
        print(f'Error in socket_server: {e}')
        import traceback
        traceback.print_exc()
        response = {'status': 'ERROR', 'message': str(e)}
        await websocket.send(json.dumps(response))


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run musicality evaluation WebSocket server.')
parser.add_argument('--host', type=str, default='localhost', help='Host to run the WebSocket server on.')
parser.add_argument('--force-host', type=bool, default=False, help='Force the host to be the one specified.')
parser.add_argument('--port', type=int, default=32051, help='Port number to run the WebSocket server on.')
parser.add_argument('--sample-rate', type=int, default=16000, help='Sample rate of the audio data.')
parser.add_argument('--process-title', type=str, default='quality_musicality', help='Process title.')
parser.add_argument('--host-info-file', type=str, default='', help='Host information file.')
args = parser.parse_args()

sample_rate = args.sample_rate

# Set process title
PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)
setproctitle(PROCESS_TITLE)

# Set port
PORT = int(os.environ.get('PORT', args.port))
HOST = args.host

# Handle host info file
if args.host_info_file:
    if not args.force_host:
        HOST = os.uname().nodename
    
    PORT = filepath_to_port(args.host_info_file)
    
    with open(args.host_info_file, 'w') as f:
        f.write('{}:{}'.format(HOST, PORT))

MAX_MESSAGE_SIZE = 100 * 1024 * 1024  # 100MB

print('Starting musicality evaluation WebSocket server at ws://{}:{}'.format(HOST, PORT))
print('Available config presets:', list(CONFIG_PRESETS.keys()))
print('Example URLs:')
print('  ws://{}:{}/musicality?config_preset=noise_only'.format(HOST, PORT))
print('  ws://{}:{}/musicality?config_preset=spectral_clarity'.format(HOST, PORT))
print('  ws://{}:{}/musicality?config_preset=vi_focused'.format(HOST, PORT))

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
