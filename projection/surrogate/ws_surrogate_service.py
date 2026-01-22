#!/usr/bin/env python3
"""
WebSocket service for surrogate quality prediction.

Predicts genome quality BEFORE audio rendering, enabling selective evaluation
in QD loops. Supports both genome feature vectors and raw genomes.

Usage:
    python projection/surrogate/ws_surrogate_service.py \
        --model models/surrogate/surrogate_v1.pt \
        --port 32070

    # Or start fresh (untrained):
    python projection/surrogate/ws_surrogate_service.py \
        --port 32070 --input-dim 64

WebSocket endpoint: ws://localhost:32070/predict

Message formats:

    === INFERENCE (Single Genome) ===
    Request (JSON):
        {
            "genome": {...},              # Raw CPPN+DSP genome
            "genome_id": "optional_id"    # Optional identifier
        }

    Response (JSON):
        {
            "quality": 0.734,             # Predicted quality [0, 1]
            "uncertainty": 0.089,         # Epistemic uncertainty (std)
            "genome_id": "optional_id",
            "inference_time_ms": 2.34
        }

    === INFERENCE (Single Features) ===
    Request (JSON):
        {
            "features": [64 floats],      # Pre-extracted features
            "genome_id": "optional_id"
        }

    Response (JSON):
        {
            "quality": 0.734,
            "uncertainty": 0.089,
            "genome_id": "optional_id",
            "inference_time_ms": 1.12
        }

    === INFERENCE (Batch) ===
    Request (JSON):
        {
            "genomes": [{...}, ...],      # Multiple genomes
            "genome_ids": ["id1", ...]    # Optional identifiers
        }
    
    OR:
        {
            "features_batch": [[64 floats], ...],
            "genome_ids": ["id1", ...]
        }

    Response (JSON):
        {
            "qualities": [0.734, 0.512, ...],
            "uncertainties": [0.089, 0.124, ...],
            "genome_ids": ["id1", ...],
            "inference_time_ms": 15.6,
            "count": 50
        }

    === ONLINE TRAINING ===
    Request (JSON):
        {
            "type": "train",
            "genomes": [{...}, ...],       # Genomes OR features
            "features_batch": [[...], ...],
            "quality_scores": [0.8, 0.2, ...],  # Ground truth quality
            "epochs": 10,                   # Optional (default: 10)
            "learning_rate": 1e-3           # Optional (default: 1e-3)
        }

    Response (JSON):
        {
            "type": "train_complete",
            "n_samples": 100,
            "epochs": 10,
            "ensemble_val_loss": 0.0234,
            "is_trained": true
        }

    === STATUS ===
    Request (JSON):
        {
            "type": "status"
        }

    Response (JSON):
        {
            "type": "status",
            "is_trained": true,
            "n_training_samples": 1234,
            "input_dim": 64,
            "n_members": 5
        }

    === SAVE MODEL ===
    Request (JSON):
        {
            "type": "save",
            "path": "models/surrogate/checkpoint.pt"
        }

    Response (JSON):
        {
            "type": "save_complete",
            "path": "models/surrogate/checkpoint.pt"
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

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from projection.surrogate.surrogate_network import SurrogateEnsemble
from qd.genome_features import GenomeFeatureExtractor


class SurrogateService:
    """WebSocket service for surrogate quality prediction."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 64,
        n_members: int = 5,
        device: Optional[str] = None
    ):
        """
        Initialize surrogate service.

        Args:
            model_path: Path to pre-trained model (None to start fresh)
            input_dim: Input feature dimension (used if no model)
            n_members: Number of ensemble members (used if no model)
            device: Device for inference ('cuda', 'cpu', or None for auto)
        """
        # Device selection
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Device: {self.device}")

        # Feature extractor
        self.feature_extractor = GenomeFeatureExtractor(feature_dim=input_dim)
        print(f"Feature extractor initialized (output dim: {input_dim})")

        # Load or create model
        if model_path and Path(model_path).exists():
            print(f"Loading surrogate model from: {model_path}")
            self.model = SurrogateEnsemble.load(model_path, device=self.device)
            self.input_dim = self.model.input_dim
        else:
            if model_path:
                print(f"Model not found at {model_path}, creating new ensemble")
            else:
                print("Creating new surrogate ensemble")
            
            self.model = SurrogateEnsemble(
                input_dim=input_dim,
                n_members=n_members
            )
            self.model.to(self.device)
            self.input_dim = input_dim

        print(f"Model ready:")
        print(f"  Input dim: {self.model.input_dim}")
        print(f"  Ensemble members: {self.model.n_members}")
        print(f"  Trained: {self.model.is_trained}")
        print(f"  Training samples: {self.model.n_training_samples}")
        print()

    def extract_features(self, genome: Dict) -> np.ndarray:
        """
        Extract features from genome using GenomeFeatureExtractor.

        Args:
            genome: CPPN+DSP genome dict

        Returns:
            Feature vector [input_dim]
        """
        return self.feature_extractor.extract(genome)

    def predict_single(
        self,
        features: Optional[np.ndarray] = None,
        genome: Optional[Dict] = None
    ) -> tuple:
        """
        Predict quality for single genome/features.

        Args:
            features: Pre-extracted features [input_dim]
            genome: Raw genome dict (features extracted if not provided)

        Returns:
            (quality, uncertainty) tuple
        """
        if features is None:
            if genome is None:
                raise ValueError("Either features or genome must be provided")
            features = self.extract_features(genome)

        return self.model.predict_with_uncertainty(features, device=self.device)

    def predict_batch(
        self,
        features_batch: Optional[np.ndarray] = None,
        genomes: Optional[List[Dict]] = None
    ) -> tuple:
        """
        Batch prediction.

        Args:
            features_batch: Pre-extracted features [batch_size, input_dim]
            genomes: List of genome dicts

        Returns:
            (qualities, uncertainties) arrays
        """
        if features_batch is None:
            if genomes is None:
                raise ValueError("Either features_batch or genomes must be provided")
            features_batch = np.array([
                self.extract_features(g) for g in genomes
            ])

        return self.model.predict_batch(features_batch, device=self.device)

    def train(
        self,
        features_batch: Optional[np.ndarray] = None,
        genomes: Optional[List[Dict]] = None,
        quality_scores: np.ndarray = None,
        epochs: int = 10,
        learning_rate: float = 1e-3
    ) -> Dict:
        """
        Train on experience buffer.

        Args:
            features_batch: Pre-extracted features
            genomes: Genome dicts (features extracted if needed)
            quality_scores: Ground truth quality scores
            epochs: Training epochs
            learning_rate: Learning rate

        Returns:
            Training statistics dict
        """
        if quality_scores is None:
            raise ValueError("quality_scores must be provided")

        if features_batch is None:
            if genomes is None:
                raise ValueError("Either features_batch or genomes must be provided")
            features_batch = np.array([
                self.extract_features(g) for g in genomes
            ])

        stats = self.model.train_on_buffer(
            features=features_batch,
            quality_scores=np.array(quality_scores),
            epochs=epochs,
            learning_rate=learning_rate,
            device=self.device
        )

        return stats

    async def handle_message(self, websocket, path):
        """
        Handle incoming WebSocket messages.

        Args:
            websocket: WebSocket connection
            path: WebSocket path
        """
        try:
            message = await websocket.recv()

            try:
                data = json.loads(message)
            except json.JSONDecodeError as e:
                error_response = {'error': f'Invalid JSON: {str(e)}'}
                await websocket.send(json.dumps(error_response))
                return

            start_time = time.time()
            message_type = data.get('type', 'inference')

            if message_type == 'status':
                response = {
                    'type': 'status',
                    'is_trained': self.model.is_trained,
                    'n_training_samples': self.model.n_training_samples,
                    'input_dim': self.model.input_dim,
                    'n_members': self.model.n_members
                }

            elif message_type == 'save':
                save_path = data.get('path', 'models/surrogate/checkpoint.pt')
                self.model.save(save_path)
                response = {
                    'type': 'save_complete',
                    'path': save_path
                }

            elif message_type == 'train':
                features_batch = data.get('features_batch')
                genomes = data.get('genomes')
                quality_scores = data.get('quality_scores')
                epochs = data.get('epochs', 10)
                learning_rate = data.get('learning_rate', 1e-3)

                if features_batch is not None:
                    features_batch = np.array(features_batch, dtype=np.float32)

                stats = self.train(
                    features_batch=features_batch,
                    genomes=genomes,
                    quality_scores=quality_scores,
                    epochs=epochs,
                    learning_rate=learning_rate
                )

                response = {
                    'type': 'train_complete',
                    'is_trained': self.model.is_trained,
                    **stats
                }

            elif 'genomes' in data or 'features_batch' in data:
                # Batch inference
                features_batch = data.get('features_batch')
                genomes = data.get('genomes')
                genome_ids = data.get('genome_ids')

                if features_batch is not None:
                    features_batch = np.array(features_batch, dtype=np.float32)

                qualities, uncertainties = self.predict_batch(
                    features_batch=features_batch,
                    genomes=genomes
                )

                response = {
                    'qualities': qualities.tolist(),
                    'uncertainties': uncertainties.tolist(),
                    'count': len(qualities),
                    'inference_time_ms': (time.time() - start_time) * 1000
                }

                if genome_ids is not None:
                    response['genome_ids'] = genome_ids

            else:
                # Single inference
                features = data.get('features')
                genome = data.get('genome')
                genome_id = data.get('genome_id')

                if features is not None:
                    features = np.array(features, dtype=np.float32)

                quality, uncertainty = self.predict_single(
                    features=features,
                    genome=genome
                )

                response = {
                    'quality': quality,
                    'uncertainty': uncertainty,
                    'inference_time_ms': (time.time() - start_time) * 1000
                }

                if genome_id is not None:
                    response['genome_id'] = genome_id

            await websocket.send(json.dumps(response))

        except Exception as e:
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
        print(f"Starting surrogate service on ws://{host}:{port}/predict")
        print("Ready to accept connections...")
        print()

        async with websockets.serve(self.handle_message, host, port):
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser(
        description="WebSocket service for surrogate quality prediction",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained surrogate model (.pt file)'
    )

    parser.add_argument(
        '--input-dim',
        type=int,
        default=64,
        help='Input feature dimension (genome features)'
    )

    parser.add_argument(
        '--n-members',
        type=int,
        default=5,
        help='Number of ensemble members'
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
        default=32070,
        help='Port to bind to'
    )

    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device for inference (None=auto)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Surrogate Quality Prediction Service")
    print("=" * 80)
    print()

    service = SurrogateService(
        model_path=args.model,
        input_dim=args.input_dim,
        n_members=args.n_members,
        device=args.device
    )

    try:
        asyncio.run(service.start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == '__main__':
    main()
