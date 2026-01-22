#!/usr/bin/env python3
"""
Training script for QDHF Projection Network.

Trains a projection network to map CLAP embeddings (512D) to behavior
descriptors (6D) using triplet loss and proxy similarity judgments.

Usage:
    python scripts/train_projection.py \
        --embeddings /path/to/clap_embeddings.npy \
        --sound-ids /path/to/sound_ids.json \
        --output models/projection/projection_v1.pt \
        --triplets-per-epoch 50000 \
        --epochs 100 \
        --bd-dim 6

Requirements:
    - CLAP embeddings extracted from training sounds
    - Sound IDs matching the embeddings (optional but recommended)
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from projection.qdhf.projection_network import (
    ProjectionNetwork,
    create_projection_network,
    small_projection_network,
    standard_projection_network,
    large_projection_network,
    deep_projection_network
)
from projection.qdhf.proxy_triplet_generator import ProxyTripletGenerator
from projection.qdhf.triplet_trainer import TripletTrainer


def load_embeddings(embeddings_path: str) -> np.ndarray:
    """Load CLAP embeddings from .npy file."""
    print(f"Loading embeddings from: {embeddings_path}")
    embeddings = np.load(embeddings_path)
    print(f"  Shape: {embeddings.shape}")
    print(f"  Dtype: {embeddings.dtype}")
    return embeddings


def load_sound_ids(sound_ids_path: str) -> list:
    """Load sound IDs from JSON file."""
    if not sound_ids_path:
        return None

    print(f"Loading sound IDs from: {sound_ids_path}")
    with open(sound_ids_path, 'r') as f:
        sound_ids = json.load(f)
    print(f"  Count: {len(sound_ids)}")
    return sound_ids


def create_network(args) -> ProjectionNetwork:
    """Create projection network from arguments."""

    # Use predefined architecture if specified
    if args.architecture:
        arch_map = {
            'small': small_projection_network,
            'standard': standard_projection_network,
            'large': large_projection_network,
            'deep': deep_projection_network
        }
        if args.architecture not in arch_map:
            raise ValueError(f"Unknown architecture: {args.architecture}")

        print(f"Using predefined architecture: {args.architecture}")
        return arch_map[args.architecture]()

    # Create custom network from config
    config = {
        'input_dim': args.input_dim,
        'hidden_dim': args.hidden_dim,
        'output_dim': args.bd_dim,
        'num_hidden_layers': args.num_hidden_layers,
        'dropout': args.dropout,
        'activation': args.activation
    }

    print(f"Creating custom network:")
    print(f"  Input: {config['input_dim']}D")
    print(f"  Hidden: {config['hidden_dim']}D × {config['num_hidden_layers']} layers")
    print(f"  Output: {config['output_dim']}D")
    print(f"  Dropout: {config['dropout']}")
    print(f"  Activation: {config['activation']}")

    return create_projection_network(config)


def main():
    parser = argparse.ArgumentParser(
        description="Train QDHF projection network",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--embeddings',
        type=str,
        required=True,
        help='Path to CLAP embeddings (.npy file, shape: [N, 512])'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for trained model (.pt file)'
    )

    # Optional data arguments
    parser.add_argument(
        '--sound-ids',
        type=str,
        default=None,
        help='Path to sound IDs JSON file (optional)'
    )

    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )

    parser.add_argument(
        '--triplets-per-epoch',
        type=int,
        default=50000,
        help='Number of triplets per epoch'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training'
    )

    parser.add_argument(
        '--val-triplets',
        type=int,
        default=1000,
        help='Number of triplets for validation'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='L2 regularization weight'
    )

    parser.add_argument(
        '--margin',
        type=float,
        default=1.0,
        help='Triplet loss margin'
    )

    parser.add_argument(
        '--early-stopping-patience',
        type=int,
        default=None,
        help='Early stopping patience (epochs, None=disabled)'
    )

    # Network architecture arguments
    parser.add_argument(
        '--architecture',
        type=str,
        default='standard',
        choices=['small', 'standard', 'large', 'deep', 'custom'],
        help='Predefined architecture (use "custom" for manual config)'
    )

    parser.add_argument(
        '--input-dim',
        type=int,
        default=512,
        help='Input dimension (CLAP embedding size)'
    )

    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden layer dimension (custom architecture only)'
    )

    parser.add_argument(
        '--bd-dim',
        type=int,
        default=6,
        help='Behavior descriptor dimension (output size)'
    )

    parser.add_argument(
        '--num-hidden-layers',
        type=int,
        default=2,
        help='Number of hidden layers (custom architecture only)'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.0,
        help='Dropout probability (custom architecture only)'
    )

    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'tanh', 'leaky_relu'],
        help='Activation function (custom architecture only)'
    )

    # Triplet generation arguments
    parser.add_argument(
        '--k-neighbors',
        type=int,
        default=10,
        help='Number of nearest neighbors for positive sampling'
    )

    parser.add_argument(
        '--distance-threshold',
        type=float,
        default=None,
        help='Distance threshold for negative sampling (None=auto)'
    )

    # Checkpointing arguments
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default=None,
        help='Directory to save checkpoints (None=no checkpoints)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume training from'
    )

    # Device arguments
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu'],
        help='Device to train on (None=auto)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Print training progress'
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        np.random.seed(args.seed)
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)

    print("=" * 80)
    print("QDHF Projection Network Training")
    print("=" * 80)
    print()

    # Load data
    embeddings = load_embeddings(args.embeddings)
    sound_ids = load_sound_ids(args.sound_ids)

    # Validate embeddings
    if embeddings.shape[1] != args.input_dim:
        raise ValueError(
            f"Embedding dimension mismatch: got {embeddings.shape[1]}, "
            f"expected {args.input_dim}"
        )

    if sound_ids is not None and len(sound_ids) != len(embeddings):
        raise ValueError(
            f"Sound IDs count mismatch: got {len(sound_ids)}, "
            f"expected {len(embeddings)}"
        )

    print(f"Loaded {len(embeddings)} embeddings")
    print()

    # Create triplet generator
    print("Creating triplet generator...")
    generator = ProxyTripletGenerator(
        clap_embeddings=embeddings,
        sound_ids=sound_ids,
        k_neighbors=args.k_neighbors,
        distance_threshold=args.distance_threshold,
        seed=args.seed
    )
    print()

    # Create network
    if args.architecture == 'custom':
        args.architecture = None  # Use custom config
    network = create_network(args)
    print()

    # Create trainer
    trainer = TripletTrainer(
        model=network,
        triplet_generator=generator,
        margin=args.margin,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )
    print()

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
        print()

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Train
    start_time = time.time()

    history = trainer.train(
        epochs=args.epochs,
        triplets_per_epoch=args.triplets_per_epoch,
        batch_size=args.batch_size,
        val_triplets=args.val_triplets,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        verbose=args.verbose
    )

    training_time = time.time() - start_time

    print()
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Total training time: {training_time:.1f}s ({training_time/60:.1f} minutes)")
    print(f"Epochs trained: {history['epochs']}")
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    print(f"Final val accuracy: {history['val_accuracy'][-1]:.2%}")
    print(f"Best val loss: {min(history['val_loss']):.4f}")
    print(f"Best val accuracy: {max(history['val_accuracy']):.2%}")
    print()

    # Save final model
    print(f"Saving final model to: {args.output}")
    trainer.save_checkpoint(str(output_path))

    # Save training history
    history_path = output_path.parent / f"{output_path.stem}_history.json"
    print(f"Saving training history to: {history_path}")
    trainer.save_history(str(history_path))

    print()
    print("Done!")


if __name__ == '__main__':
    main()
