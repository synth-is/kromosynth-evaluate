"""
Triplet Trainer for QDHF Projection Network.

Trains the projection network using triplet loss to preserve
perceptual similarity relationships in the projected behavior space.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
import time

from .projection_network import ProjectionNetwork
from .proxy_triplet_generator import ProxyTripletGenerator


class TripletTrainer:
    """
    Trainer for projection network using triplet loss.

    Trains the network to preserve similarity relationships:
    - Similar sounds should be close in BD space
    - Dissimilar sounds should be far in BD space
    """

    def __init__(
        self,
        model: ProjectionNetwork,
        triplet_generator: ProxyTripletGenerator,
        margin: float = 1.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        device: Optional[str] = None
    ):
        """
        Initialize trainer.

        Args:
            model: ProjectionNetwork to train
            triplet_generator: Generator for training triplets
            margin: Margin for triplet loss
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.model = model
        self.generator = triplet_generator
        self.margin = margin

        # Device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model.to(self.device)

        # Loss and optimizer
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)  # Euclidean distance
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'epochs': 0
        }

        print(f"TripletTrainer initialized:")
        print(f"  Device: {self.device}")
        print(f"  Margin: {margin}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Model parameters: {model.get_num_parameters():,}")

    def train_epoch(
        self,
        num_triplets: int,
        batch_size: int = 64
    ) -> float:
        """
        Train one epoch.

        Args:
            num_triplets: Number of triplets to train on
            batch_size: Batch size for training

        Returns:
            Average loss for the epoch
        """
        self.model.train()

        # Generate triplets
        triplet_indices = self.generator.generate_batch(num_triplets)
        anchors, positives, negatives = self.generator.get_embeddings_for_triplets(triplet_indices)

        # Convert to tensors
        anchors_t = torch.FloatTensor(anchors).to(self.device)
        positives_t = torch.FloatTensor(positives).to(self.device)
        negatives_t = torch.FloatTensor(negatives).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(anchors_t, positives_t, negatives_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_loss = 0.0
        num_batches = 0

        for anchor_batch, positive_batch, negative_batch in dataloader:
            # Forward pass
            anchor_proj = self.model(anchor_batch)
            positive_proj = self.model(positive_batch)
            negative_proj = self.model(negative_batch)

            # Compute loss
            loss = self.criterion(anchor_proj, positive_proj, negative_proj)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def evaluate_triplet_accuracy(
        self,
        num_triplets: int = 1000,
        batch_size: int = 64
    ) -> Tuple[float, float]:
        """
        Evaluate triplet accuracy on validation set.

        Accuracy = fraction of triplets where:
        d(anchor, positive) < d(anchor, negative) in projected space

        Args:
            num_triplets: Number of triplets to evaluate
            batch_size: Batch size for evaluation

        Returns:
            (loss, accuracy) tuple
        """
        self.model.eval()

        # Generate validation triplets
        triplet_indices = self.generator.generate_batch(num_triplets)
        anchors, positives, negatives = self.generator.get_embeddings_for_triplets(triplet_indices)

        # Convert to tensors
        anchors_t = torch.FloatTensor(anchors).to(self.device)
        positives_t = torch.FloatTensor(positives).to(self.device)
        negatives_t = torch.FloatTensor(negatives).to(self.device)

        # Create dataset and dataloader
        dataset = TensorDataset(anchors_t, positives_t, negatives_t)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for anchor_batch, positive_batch, negative_batch in dataloader:
                # Forward pass
                anchor_proj = self.model(anchor_batch)
                positive_proj = self.model(positive_batch)
                negative_proj = self.model(negative_batch)

                # Compute loss
                loss = self.criterion(anchor_proj, positive_proj, negative_proj)
                total_loss += loss.item()

                # Compute accuracy
                dist_pos = torch.norm(anchor_proj - positive_proj, p=2, dim=1)
                dist_neg = torch.norm(anchor_proj - negative_proj, p=2, dim=1)

                correct += (dist_pos < dist_neg).sum().item()
                total += len(anchor_batch)

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total

        return avg_loss, accuracy

    def train(
        self,
        epochs: int,
        triplets_per_epoch: int,
        batch_size: int = 64,
        val_triplets: int = 1000,
        early_stopping_patience: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with validation and early stopping.

        Args:
            epochs: Number of epochs to train
            triplets_per_epoch: Triplets per epoch
            batch_size: Batch size
            val_triplets: Number of triplets for validation
            early_stopping_patience: Stop if no improvement for N epochs (None = no early stopping)
            checkpoint_dir: Directory to save checkpoints (None = no checkpoints)
            verbose: Print progress

        Returns:
            Training history dict
        """
        best_val_loss = float('inf')
        patience_counter = 0

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training:")
        print(f"  Epochs: {epochs}")
        print(f"  Triplets per epoch: {triplets_per_epoch}")
        print(f"  Batch size: {batch_size}")
        print(f"  Validation triplets: {val_triplets}")
        print(f"  Early stopping patience: {early_stopping_patience}")
        print()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(triplets_per_epoch, batch_size)

            # Validate
            val_loss, val_accuracy = self.evaluate_triplet_accuracy(val_triplets, batch_size)

            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['epochs'] += 1

            epoch_time = time.time() - epoch_start

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s):")
                print(f"  Train loss: {train_loss:.4f}")
                print(f"  Val loss: {val_loss:.4f}")
                print(f"  Val accuracy: {val_accuracy:.2%}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                # Save best model
                if checkpoint_dir:
                    best_path = Path(checkpoint_dir) / "best_model.pt"
                    self.save_checkpoint(str(best_path))
                    if verbose:
                        print(f"  ✓ Best model saved")

            else:
                patience_counter += 1
                if early_stopping_patience and patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"\nEarly stopping at epoch {epoch+1}")
                    break

            # Save periodic checkpoints
            if checkpoint_dir and (epoch + 1) % 10 == 0:
                checkpoint_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(str(checkpoint_path))

        if verbose:
            print(f"\nTraining complete!")
            print(f"  Final train loss: {self.history['train_loss'][-1]:.4f}")
            print(f"  Final val loss: {self.history['val_loss'][-1]:.4f}")
            print(f"  Final val accuracy: {self.history['val_accuracy'][-1]:.2%}")
            print(f"  Best val loss: {best_val_loss:.4f}")

        return self.history

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'input_dim': self.model.input_dim,
                'hidden_dim': self.model.hidden_dim,
                'output_dim': self.model.output_dim,
                'num_hidden_layers': self.model.num_hidden_layers,
                'dropout': self.model.dropout,
                'margin': self.margin
            }
        }, path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']

        print(f"Checkpoint loaded from {path}")
        print(f"  Epochs trained: {self.history['epochs']}")

    def save_history(self, path: str):
        """Save training history to JSON."""
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def set_learning_rate(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def get_current_stats(self) -> Dict:
        """Get current training statistics."""
        if not self.history['train_loss']:
            return {'status': 'not_trained'}

        return {
            'epochs_trained': self.history['epochs'],
            'last_train_loss': self.history['train_loss'][-1],
            'last_val_loss': self.history['val_loss'][-1],
            'last_val_accuracy': self.history['val_accuracy'][-1],
            'best_val_loss': min(self.history['val_loss']),
            'best_val_accuracy': max(self.history['val_accuracy'])
        }
