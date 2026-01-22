#!/usr/bin/env python3
"""
WebSocket service for QDHF projection inference and incremental training.

Provides real-time behavior descriptor prediction from CLAP embeddings
using a trained projection network. Supports incremental retraining during
evolution runs.

Usage:
    python projection/qdhf/ws_projection_service.py \
        --model models/projection/projection_v1.pt \
        --port 32053

WebSocket endpoint: ws://localhost:32053/project

Message formats:

    === INFERENCE (Single) ===
    Request (JSON):
        {
            "embedding": [512 floats],  # CLAP embedding
            "sound_id": "optional_id"   # Optional identifier
        }

    Response (JSON):
        {
            "behavior_descriptor": [6 floats],  # Predicted BD in [0, 1]
            "sound_id": "optional_id",
            "inference_time_ms": 1.23
        }

    === INFERENCE (Batch) ===
    Request (JSON):
        {
            "embeddings": [[512 floats], ...],  # Multiple CLAP embeddings
            "sound_ids": ["id1", "id2", ...]     # Optional identifiers
        }

    Response (JSON):
        {
            "behavior_descriptors": [[6 floats], ...],
            "sound_ids": ["id1", "id2", ...],
            "inference_time_ms": 12.34,
            "count": 10
        }

    === INCREMENTAL TRAINING ===
    Request (JSON):
        {
            "type": "train",
            "embeddings": [[512 floats], ...],  # Archive CLAP embeddings
            "epochs": 10,                       # Optional (default: 10)
            "triplets_per_epoch": 5000,         # Optional (default: 5000)
            "learning_rate": 1e-4,              # Optional (default: 1e-4)
            "k_neighbors": 10,                  # Optional (default: 10)
            "margin": 1.0                       # Optional (default: 1.0)
        }

    Response (JSON):
        {
            "type": "train_complete",
            "epochs": 10,
            "training_time": 45.2,
            "final_val_loss": 0.234,
            "final_val_accuracy": 0.892,
            "num_embeddings": 1234
        }
"""

import asyncio
import websockets
import json
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from projection.qdhf.projection_network import ProjectionNetwork
from projection.qdhf.proxy_triplet_generator import ProxyTripletGenerator
from projection.qdhf.triplet_trainer import TripletTrainer


class ProjectionService:
    """WebSocket service for projection inference and incremental training."""

    def __init__(self, model_path: Optional[str] = None, device: Optional[str] = None):
        """
        Initialize projection service.

        Args:
            model_path: Path to trained model checkpoint (.pt file). If None, initializes with random weights.
            device: Device to run inference on ('cuda', 'cpu', or None for auto)
        """
        self.model_path = model_path

        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Device: {self.device}")

        # Load checkpoint or initialize with default config
        if model_path and Path(model_path).exists():
            print(f"Loading projection model from: {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)

            # Extract config
            config = checkpoint['config']
            print(f"Model configuration:")
            print(f"  Input: {config['input_dim']}D")
            print(f"  Hidden: {config['hidden_dim']}D × {config['num_hidden_layers']} layers")
            print(f"  Output: {config['output_dim']}D")
            print(f"  Dropout: {config['dropout']}")

            # Create model
            self.model = ProjectionNetwork(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_hidden_layers=config['num_hidden_layers'],
                dropout=config['dropout']
            )

            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully")

            # Training history (optional)
            if 'history' in checkpoint:
                history = checkpoint['history']
                print(f"Model training info:")
                print(f"  Epochs: {history['epochs']}")
                print(f"  Final val accuracy: {history['val_accuracy'][-1]:.2%}")
                print(f"  Best val accuracy: {max(history['val_accuracy']):.2%}")
        else:
            # Initialize with default config (for dynamic training)
            print("No pre-trained model provided - initializing with random weights")
            print("Model will be trained incrementally during evolution runs")
            config = {
                'input_dim': 512,  # CLAP embedding dimension
                'hidden_dim': 256,
                'output_dim': 6,   # QDHF projection dimension
                'num_hidden_layers': 3,
                'dropout': 0.1
            }
            print(f"Default configuration:")
            print(f"  Input: {config['input_dim']}D")
            print(f"  Hidden: {config['hidden_dim']}D × {config['num_hidden_layers']} layers")
            print(f"  Output: {config['output_dim']}D")
            print(f"  Dropout: {config['dropout']}")

            # Create model with random weights
            self.model = ProjectionNetwork(
                input_dim=config['input_dim'],
                hidden_dim=config['hidden_dim'],
                output_dim=config['output_dim'],
                num_hidden_layers=config['num_hidden_layers'],
                dropout=config['dropout']
            )
            print("Model initialized with random weights")

        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Cache config
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']

        print(f"  Parameters: {self.model.get_num_parameters():,}")

        print()

    def project_single(self, embedding: np.ndarray) -> np.ndarray:
        """
        Project single CLAP embedding to behavior descriptor.

        Args:
            embedding: CLAP embedding (512D)

        Returns:
            Behavior descriptor (6D) in [0, 1]
        """
        # Validate input
        if embedding.shape != (self.input_dim,):
            raise ValueError(
                f"Invalid embedding shape: {embedding.shape}, "
                f"expected ({self.input_dim},)"
            )

        # Convert to tensor
        x = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)  # [1, 512]

        # Inference
        with torch.no_grad():
            bd = self.model(x)  # [1, 6]

        # Convert to numpy
        return bd.cpu().numpy()[0]

    def project_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project batch of CLAP embeddings to behavior descriptors.

        Args:
            embeddings: CLAP embeddings (N, 512)

        Returns:
            Behavior descriptors (N, 6) in [0, 1]
        """
        # Validate input
        if embeddings.ndim != 2 or embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Invalid embeddings shape: {embeddings.shape}, "
                f"expected (N, {self.input_dim})"
            )

        # Convert to tensor
        x = torch.FloatTensor(embeddings).to(self.device)  # [N, 512]

        # Inference
        with torch.no_grad():
            bds = self.model(x)  # [N, 6]

            # TEMPORARY: Add noise for early exploration (remove after diversity established)
            # Add uniform noise in [-0.2, 0.2] to help escape local optima
            # This helps bootstrap diversity when projection is untrained
            noise = torch.rand_like(bds) * 0.4 - 0.2  # Uniform in [-0.2, 0.2]
            bds = torch.clamp(bds + noise, 0.0, 1.0)  # Keep in [0, 1] range

        # Convert to numpy
        return bds.cpu().numpy()

    def train_incremental(
        self,
        embeddings: np.ndarray,
        epochs: int = 10,
        triplets_per_epoch: int = 5000,
        learning_rate: float = 1e-4,
        k_neighbors: int = 10,
        margin: float = 1.0
    ) -> dict:
        """
        Incrementally train the projection model on new embeddings.

        Args:
            embeddings: CLAP embeddings from archive (N, 512)
            epochs: Number of training epochs
            triplets_per_epoch: Triplets per epoch
            learning_rate: Learning rate for fine-tuning
            k_neighbors: K for positive triplet sampling
            margin: Triplet loss margin

        Returns:
            Training statistics dict
        """
        print(f"\nIncremental training with {len(embeddings)} embeddings...")
        print(f"  Epochs: {epochs}")
        print(f"  Triplets/epoch: {triplets_per_epoch}")
        print(f"  Learning rate: {learning_rate}")

        # Create triplet generator
        generator = ProxyTripletGenerator(
            clap_embeddings=embeddings,
            k_neighbors=k_neighbors
        )

        # Create trainer
        trainer = TripletTrainer(
            model=self.model,
            triplet_generator=generator,
            margin=margin,
            learning_rate=learning_rate,
            device=self.device
        )

        # Train
        start_time = time.time()

        for epoch in range(epochs):
            train_loss = trainer.train_epoch(
                num_triplets=triplets_per_epoch,
                batch_size=64
            )

            # Evaluate every few epochs
            if epoch % max(1, epochs // 3) == 0 or epoch == epochs - 1:
                val_loss, val_accuracy = trainer.evaluate_triplet_accuracy(
                    num_triplets=1000
                )
                print(f"  Epoch {epoch+1}/{epochs}: "
                      f"train_loss={train_loss:.4f}, "
                      f"val_loss={val_loss:.4f}, "
                      f"val_acc={val_accuracy:.2%}")

        training_time = time.time() - start_time

        # Final validation
        final_val_loss, final_val_accuracy = trainer.evaluate_triplet_accuracy(
            num_triplets=1000
        )

        print(f"Incremental training complete in {training_time:.1f}s")
        print(f"  Final validation: loss={final_val_loss:.4f}, "
              f"accuracy={final_val_accuracy:.2%}")

        # Return to evaluation mode
        self.model.eval()

        return {
            'epochs': epochs,
            'training_time': training_time,
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_accuracy,
            'num_embeddings': len(embeddings)
        }

    async def handle_message(self, websocket, path):
        """
        Handle incoming WebSocket messages.

        Args:
            websocket: WebSocket connection
            path: WebSocket path
        """
        try:
            # Receive message
            message = await websocket.recv()

            # Parse JSON
            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                error_response = {
                    'error': f'Invalid JSON: {str(e)}'
                }
                await websocket.send(json.dumps(error_response))
                return

            start_time = time.time()

            # Check message type
            message_type = data.get('type', 'inference')

            if message_type == 'train':
                # Incremental training request
                embeddings = np.array(data['embeddings'], dtype=np.float32)
                epochs = data.get('epochs', 10)
                triplets_per_epoch = data.get('triplets_per_epoch', 5000)
                learning_rate = data.get('learning_rate', 1e-4)
                k_neighbors = data.get('k_neighbors', 10)
                margin = data.get('margin', 1.0)

                # Train
                stats = self.train_incremental(
                    embeddings=embeddings,
                    epochs=epochs,
                    triplets_per_epoch=triplets_per_epoch,
                    learning_rate=learning_rate,
                    k_neighbors=k_neighbors,
                    margin=margin
                )

                # Response
                response = {
                    'type': 'train_complete',
                    **stats
                }

            elif 'embeddings' in data:
                # Batch inference
                embeddings = np.array(data['embeddings'], dtype=np.float32)
                sound_ids = data.get('sound_ids', None)

                # Project
                bds = self.project_batch(embeddings)

                # Response
                response = {
                    'behavior_descriptors': bds.tolist(),
                    'count': len(bds),
                    'inference_time_ms': (time.time() - start_time) * 1000
                }

                if sound_ids is not None:
                    response['sound_ids'] = sound_ids

            else:
                # Single inference
                embedding = np.array(data['embedding'], dtype=np.float32)
                sound_id = data.get('sound_id', None)

                # Project
                bd = self.project_single(embedding)

                # Response
                response = {
                    'behavior_descriptor': bd.tolist(),
                    'inference_time_ms': (time.time() - start_time) * 1000
                }

                if sound_id is not None:
                    response['sound_id'] = sound_id

            # Send response
            await websocket.send(json.dumps(response))

        except Exception as e:
            # Error handling
            error_response = {
                'error': str(e),
                'type': type(e).__name__
            }
            await websocket.send(json.dumps(error_response))

    async def start_server(self, host: str, port: int):
        """
        Start WebSocket server.

        Args:
            host: Host to bind to
            port: Port to bind to
        """
        print(f"Starting projection service on ws://{host}:{port}/project")
        print("Ready to accept connections...")
        print()

        # Increase max_size to handle large training requests with many embeddings
        # Default is 1MB, but training requests can be larger
        # With compression, decompressed size can be much larger than compressed
        # Set to 100MB to handle large archives (e.g., 500+ elites with 512D embeddings)
        # Disable compression to avoid decompression size limits
        async with websockets.serve(
            self.handle_message,
            host,
            port,
            max_size=300 * 1024 * 1024,  # 300MB
            compression=None  # Disable compression to avoid decompression limits
        ):
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket service for QDHF projection inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained projection model (.pt file). If not provided, starts with random weights for dynamic training.'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='127.0.0.1',
        help='Host to bind to'
    )

    parser.add_argument(
        '--port',
        type=int,
        default=32053,
        help='Port to bind to'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to run inference on (None=auto)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("QDHF Projection Service")
    print("=" * 80)
    print()

    # Validate model path if provided
    if args.model and not Path(args.model).exists():
        print(f"Warning: Model file not found: {args.model}")
        print(f"Starting with random weights for dynamic training")
        args.model = None

    # Create service
    service = ProjectionService(
        model_path=args.model,
        device=args.device
    )

    # Start server
    try:
        asyncio.run(service.start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
