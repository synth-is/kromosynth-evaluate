"""
WebSocket server for CLAP feature extraction.

Provides CLAP embedding extraction via WebSocket.
Endpoint: ws://localhost:32051/clap

Request formats:
1. Binary: Raw audio buffer (float32)
2. JSON: {"audio_buffer": "<base64>", "sample_rate": 16000}

Response format:
{
    "embedding": [512 floats],
    "extraction_time_ms": 45
}
"""

import asyncio
import websockets
import json
import argparse
import numpy as np
import os
import base64
from setproctitle import setproctitle
from urllib.parse import urlparse
import time
import sys

# Handling path for module execution vs script execution
try:
    from features.clap.clap_extractor import CLAPExtractor
except ImportError:
    # If running as script from features/clap/ directory
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from features.clap.clap_extractor import CLAPExtractor

# Global extractor instance (loaded once)
clap_extractor = None


async def socket_server(websocket, path):
    """WebSocket handler for CLAP extraction."""
    global clap_extractor

    try:
        # Wait for message
        message = await websocket.recv()

        start_time = time.time()

        # Parse message
        if isinstance(message, bytes):
            # Binary audio buffer (float32)
            audio_data = np.frombuffer(message, dtype=np.float32)
            current_sample_rate = sample_rate  # Use global default

        elif isinstance(message, str):
            # JSON message
            data = json.loads(message)

            if "audio_buffer" not in data:
                response = {
                    "status": "ERROR",
                    "message": "Missing audio_buffer in request"
                }
                await websocket.send(json.dumps(response))
                return

            # Decode base64 audio buffer
            buffer_bytes = base64.b64decode(data["audio_buffer"])
            audio_data = np.frombuffer(buffer_bytes, dtype=np.float32)
            current_sample_rate = data.get("sample_rate", sample_rate)

        else:
            response = {
                "status": "ERROR",
                "message": "Invalid message format"
            }
            await websocket.send(json.dumps(response))
            return

        # Extract CLAP embedding
        embedding = clap_extractor.extract_embedding(
            audio_data,
            current_sample_rate
        )

        elapsed_ms = (time.time() - start_time) * 1000

        # Send response - align format with features.py
        response = {
            "status": "received standalone audio",
            "features": embedding.tolist(),  # CLAP embedding as features
            "embedding": embedding.tolist(),  # Also keep as embedding for compatibility
            "type": "clap",
            "time": elapsed_ms / 1000.0,  # Convert to seconds to match features.py
            "extraction_time_ms": round(elapsed_ms, 2)  # Keep for backward compatibility
        }
        await websocket.send(json.dumps(response))

    except Exception as e:
        print(f"Error in socket_server: {e}")
        import traceback
        traceback.print_exc()
        response = {
            "status": "ERROR",
            "message": str(e)
        }
        await websocket.send(json.dumps(response))


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run CLAP feature extraction WebSocket server."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host to run the WebSocket server on."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=32051,
        help="Port number to run the WebSocket server on."
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Default sample rate of the audio data."
    )
    parser.add_argument(
        "--process-title",
        type=str,
        default="clap_service",
        help="Process title."
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to CLAP checkpoint file. If not provided, uses default."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use (cuda/cpu). Auto-detects if not specified."
    )

    args = parser.parse_args()

    # Configuration from args and environment
    sample_rate = args.sample_rate
    checkpoint_path = args.checkpoint_path or os.environ.get("CLAP_CHECKPOINT_PATH")
    device = args.device or os.environ.get("CLAP_DEVICE")

    # Set PORT as either the environment variable or the default value
    PORT = int(os.environ.get('PORT', args.port))

    # Set PROCESS_TITLE as either the environment variable or the default value
    PROCESS_TITLE = os.environ.get('PROCESS_TITLE', args.process_title)

    # Set process title
    if PROCESS_TITLE:
        setproctitle(PROCESS_TITLE)

    print("=" * 60)
    print("CLAP Feature Extraction Service")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {PORT}")
    print(f"Sample rate: {sample_rate}")
    print(f"Checkpoint: {checkpoint_path or 'default (will download)'}")
    print(f"Device: {device or 'auto-detect'}")
    print("=" * 60)

    # Initialize CLAP extractor
    print("\nInitializing CLAP extractor...")
    clap_extractor = CLAPExtractor(
        checkpoint_path=checkpoint_path,
        device=device
    )
    print("CLAP extractor ready!")

    # Start WebSocket server
    print(f"\nStarting WebSocket server on ws://{args.host}:{PORT}/clap")
    print("Waiting for connections...")

    start_server = websockets.serve(
        socket_server,
        args.host,
        PORT
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()

