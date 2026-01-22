"""
Deep Ensemble Surrogate Model for Quality Prediction.

Uses an ensemble of neural networks to predict quality scores with
uncertainty quantification. Uncertainty is derived from disagreement
between ensemble members.

## Architecture

Each ensemble member is an MLP with:
- Input: Genome features (64D from GenomeFeatureExtractor or configurable)
- Hidden: Configurable depth and width
- Output: Quality score in [0, 1]

Ensemble prediction:
- Mean across members → predicted quality
- Std across members → epistemic uncertainty

## Usage

    from surrogate_network import SurrogateEnsemble
    
    model = SurrogateEnsemble(input_dim=64, n_members=5)
    
    # Single prediction with uncertainty
    mean, std = model.predict_with_uncertainty(features)
    
    # Batch prediction
    means, stds = model.predict_batch(features_batch)
    
    # Train on experience buffer
    model.train_on_buffer(features, quality_scores, epochs=10)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, List, Optional, Dict
from pathlib import Path


class SurrogateMember(nn.Module):
    """
    Single member of the surrogate ensemble.
    
    MLP architecture with dropout for regularization.
    Outputs quality score in [0, 1] via sigmoid.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize surrogate member.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer width
            num_hidden_layers: Number of hidden layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        layers.append(nn.Sigmoid())  # Quality in [0, 1]
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Quality predictions [batch_size, 1]
        """
        return self.network(x)
    
    def get_num_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SurrogateEnsemble(nn.Module):
    """
    Ensemble of surrogate models for uncertainty-aware quality prediction.
    
    Uses ensemble disagreement for epistemic uncertainty estimation.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_members: int = 5,
        hidden_dim: int = 128,
        num_hidden_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize surrogate ensemble.
        
        Args:
            input_dim: Input feature dimension
            n_members: Number of ensemble members
            hidden_dim: Hidden layer width per member
            num_hidden_layers: Number of hidden layers per member
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        
        # Create ensemble members
        self.members = nn.ModuleList([
            SurrogateMember(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_hidden_layers=num_hidden_layers,
                dropout=dropout
            )
            for _ in range(n_members)
        ])
        
        # Training state
        self.is_trained = False
        self.n_training_samples = 0
        
        print(f"Initialized SurrogateEnsemble:")
        print(f"  Input: {input_dim}D")
        print(f"  Members: {n_members}")
        print(f"  Hidden: {hidden_dim}D × {num_hidden_layers} layers")
        print(f"  Parameters per member: {self.members[0].get_num_parameters():,}")
        print(f"  Total parameters: {self.get_num_parameters():,}")
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            mean: Mean prediction [batch_size, 1]
            std: Standard deviation [batch_size, 1]
        """
        # Get predictions from all members
        predictions = torch.stack([
            member(x) for member in self.members
        ], dim=0)  # [n_members, batch_size, 1]
        
        # Compute mean and std across ensemble
        mean = predictions.mean(dim=0)  # [batch_size, 1]
        std = predictions.std(dim=0)    # [batch_size, 1]
        
        return mean, std
    
    def predict_with_uncertainty(
        self,
        features: np.ndarray,
        device: str = 'cpu'
    ) -> Tuple[float, float]:
        """
        Predict quality with uncertainty for single sample.
        
        Args:
            features: Feature vector [input_dim]
            device: Device for inference
            
        Returns:
            mean: Predicted quality
            std: Epistemic uncertainty
        """
        self.eval()
        x = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, std = self.forward(x)
        
        return mean.item(), std.item()
    
    def predict_batch(
        self,
        features: np.ndarray,
        device: str = 'cpu'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction with uncertainty.
        
        Args:
            features: Feature matrix [batch_size, input_dim]
            device: Device for inference
            
        Returns:
            means: Predicted qualities [batch_size]
            stds: Uncertainties [batch_size]
        """
        self.eval()
        x = torch.FloatTensor(features).to(device)
        
        with torch.no_grad():
            means, stds = self.forward(x)
        
        return means.cpu().numpy().squeeze(-1), stds.cpu().numpy().squeeze(-1)
    
    def train_on_buffer(
        self,
        features: np.ndarray,
        quality_scores: np.ndarray,
        epochs: int = 10,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        val_split: float = 0.1,
        device: str = 'cpu'
    ) -> Dict:
        """
        Train ensemble on experience buffer.
        
        Each member is trained on bootstrap samples for diversity.
        
        Args:
            features: Feature matrix [n_samples, input_dim]
            quality_scores: Target quality scores [n_samples]
            epochs: Training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
            device: Device for training
            
        Returns:
            Training statistics dict
        """
        n_samples = len(features)
        n_val = int(n_samples * val_split)
        
        # Validation set (same for all members)
        val_indices = np.random.choice(n_samples, n_val, replace=False)
        val_mask = np.zeros(n_samples, dtype=bool)
        val_mask[val_indices] = True
        
        X_val = torch.FloatTensor(features[val_mask]).to(device)
        y_val = torch.FloatTensor(quality_scores[val_mask]).unsqueeze(-1).to(device)
        
        train_features = features[~val_mask]
        train_scores = quality_scores[~val_mask]
        n_train = len(train_features)
        
        # Track training stats
        member_losses = []
        
        for member_idx, member in enumerate(self.members):
            member.train()
            member.to(device)
            
            # Bootstrap sample for this member
            bootstrap_idx = np.random.choice(n_train, n_train, replace=True)
            X_train = torch.FloatTensor(train_features[bootstrap_idx]).to(device)
            y_train = torch.FloatTensor(train_scores[bootstrap_idx]).unsqueeze(-1).to(device)
            
            # Optimizer
            optimizer = optim.Adam(member.parameters(), lr=learning_rate)
            criterion = nn.MSELoss()
            
            # Training loop
            best_val_loss = float('inf')
            
            for epoch in range(epochs):
                # Shuffle
                perm = torch.randperm(n_train)
                X_train = X_train[perm]
                y_train = y_train[perm]
                
                epoch_loss = 0.0
                n_batches = 0
                
                for i in range(0, n_train, batch_size):
                    batch_x = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    pred = member(batch_x)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    n_batches += 1
                
                avg_train_loss = epoch_loss / max(n_batches, 1)
                
                # Validation
                member.eval()
                with torch.no_grad():
                    val_pred = member(X_val)
                    val_loss = criterion(val_pred, y_val).item()
                member.train()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
            
            member_losses.append(best_val_loss)
            member.eval()
        
        self.is_trained = True
        self.n_training_samples = n_samples
        
        # Ensemble validation loss
        self.eval()
        with torch.no_grad():
            mean_pred, _ = self.forward(X_val)
            ensemble_val_loss = nn.MSELoss()(mean_pred, y_val).item()
        
        return {
            'n_samples': n_samples,
            'n_train': n_train,
            'n_val': n_val,
            'epochs': epochs,
            'member_val_losses': member_losses,
            'ensemble_val_loss': ensemble_val_loss,
            'mean_member_loss': np.mean(member_losses)
        }
    
    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def save(self, path: str):
        """
        Save ensemble checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            'config': {
                'input_dim': self.input_dim,
                'n_members': self.n_members,
                'hidden_dim': self.hidden_dim,
                'num_hidden_layers': self.num_hidden_layers,
                'dropout': self.dropout
            },
            'model_state_dict': self.state_dict(),
            'is_trained': self.is_trained,
            'n_training_samples': self.n_training_samples
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        print(f"Saved ensemble to {path}")
    
    @classmethod
    def load(cls, path: str, device: str = 'cpu') -> 'SurrogateEnsemble':
        """
        Load ensemble from checkpoint.
        
        Args:
            path: Path to checkpoint
            device: Device to load to
            
        Returns:
            Loaded SurrogateEnsemble
        """
        checkpoint = torch.load(path, map_location=device)
        config = checkpoint['config']
        
        model = cls(
            input_dim=config['input_dim'],
            n_members=config['n_members'],
            hidden_dim=config['hidden_dim'],
            num_hidden_layers=config['num_hidden_layers'],
            dropout=config['dropout']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.is_trained = checkpoint.get('is_trained', False)
        model.n_training_samples = checkpoint.get('n_training_samples', 0)
        model.to(device)
        model.eval()
        
        print(f"Loaded ensemble from {path}")
        return model


def create_surrogate(
    input_dim: int = 64,
    n_members: int = 5,
    **kwargs
) -> SurrogateEnsemble:
    """
    Factory function to create surrogate ensemble.
    
    Args:
        input_dim: Input feature dimension (default 64 for genome features)
        n_members: Number of ensemble members
        **kwargs: Additional config options
        
    Returns:
        SurrogateEnsemble instance
    """
    return SurrogateEnsemble(input_dim=input_dim, n_members=n_members, **kwargs)


if __name__ == "__main__":
    # Test ensemble
    print("Testing SurrogateEnsemble...\n")
    
    # Create ensemble
    model = SurrogateEnsemble(input_dim=64, n_members=5, hidden_dim=128)
    
    # Test inference
    features = np.random.rand(64).astype(np.float32)
    mean, std = model.predict_with_uncertainty(features)
    print(f"Single prediction: mean={mean:.4f}, std={std:.4f}")
    
    # Test batch
    batch = np.random.rand(10, 64).astype(np.float32)
    means, stds = model.predict_batch(batch)
    print(f"Batch prediction: {len(means)} samples")
    print(f"  Means: [{means.min():.3f}, {means.max():.3f}]")
    print(f"  Stds:  [{stds.min():.3f}, {stds.max():.3f}]")
    
    # Test training
    print("\nTraining on synthetic data...")
    X = np.random.rand(200, 64).astype(np.float32)
    y = np.random.rand(200).astype(np.float32)
    
    stats = model.train_on_buffer(X, y, epochs=5)
    print(f"Training complete:")
    print(f"  Samples: {stats['n_samples']}")
    print(f"  Ensemble val loss: {stats['ensemble_val_loss']:.4f}")
    
    # Test after training
    mean, std = model.predict_with_uncertainty(features)
    print(f"\nPost-training prediction: mean={mean:.4f}, std={std:.4f}")
    
    print("\nAll tests passed!")
