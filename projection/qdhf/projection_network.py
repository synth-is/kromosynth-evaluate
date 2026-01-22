"""
Projection Network for QDHF.

Maps CLAP embeddings (512D) to behavior descriptors (6D) using learned
triplet-based projection that preserves perceptual similarity.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class ProjectionNetwork(nn.Module):
    """
    MLP that projects CLAP embeddings to behavior descriptors.

    Architecture:
    - Input: 512D CLAP embedding
    - Hidden layers: 2 layers with ReLU
    - Output: 6D behavior descriptor (sigmoid to [0, 1])

    The network is trained with triplet loss to preserve perceptual
    similarity relationships from CLAP space into the BD space.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        output_dim: int = 6,
        num_hidden_layers: int = 2,
        dropout: float = 0.0,
        activation: str = "relu"
    ):
        """
        Initialize projection network.

        Args:
            input_dim: CLAP embedding dimension (512)
            hidden_dim: Hidden layer width
            output_dim: Behavior descriptor dimension (6)
            num_hidden_layers: Number of hidden layers (default: 2)
            dropout: Dropout probability (0 = no dropout)
            activation: Activation function ("relu", "tanh", "leaky_relu")
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout

        # Select activation
        if activation == "relu":
            act_fn = nn.ReLU
        elif activation == "tanh":
            act_fn = nn.Tanh
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(act_fn())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Output in [0, 1] for archive ranges

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project CLAP embedding to behavior descriptor.

        Args:
            x: CLAP embeddings (batch_size, 512)

        Returns:
            Behavior descriptors (batch_size, 6) in [0, 1]
        """
        return self.net(x)

    def project_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for batch projection.

        Args:
            embeddings: (N, 512) CLAP embeddings

        Returns:
            (N, 6) behavior descriptors
        """
        return self.forward(embeddings)

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_projection_network(
    config: Optional[dict] = None
) -> ProjectionNetwork:
    """
    Factory function to create projection network from config.

    Args:
        config: Configuration dict with keys:
            - input_dim (default: 512)
            - hidden_dim (default: 128)
            - output_dim (default: 6)
            - num_hidden_layers (default: 2)
            - dropout (default: 0.0)
            - activation (default: "relu")

    Returns:
        ProjectionNetwork instance
    """
    if config is None:
        config = {}

    return ProjectionNetwork(
        input_dim=config.get('input_dim', 512),
        hidden_dim=config.get('hidden_dim', 128),
        output_dim=config.get('output_dim', 6),
        num_hidden_layers=config.get('num_hidden_layers', 2),
        dropout=config.get('dropout', 0.0),
        activation=config.get('activation', 'relu')
    )


# Predefined architectures for experimentation

def small_projection_network() -> ProjectionNetwork:
    """Small network (fewer parameters, faster)."""
    return ProjectionNetwork(
        input_dim=512,
        hidden_dim=64,
        output_dim=6,
        num_hidden_layers=2
    )


def standard_projection_network() -> ProjectionNetwork:
    """Standard network (recommended)."""
    return ProjectionNetwork(
        input_dim=512,
        hidden_dim=128,
        output_dim=6,
        num_hidden_layers=2
    )


def large_projection_network() -> ProjectionNetwork:
    """Large network (more capacity)."""
    return ProjectionNetwork(
        input_dim=512,
        hidden_dim=256,
        output_dim=6,
        num_hidden_layers=3
    )


def deep_projection_network() -> ProjectionNetwork:
    """Deep network (more layers)."""
    return ProjectionNetwork(
        input_dim=512,
        hidden_dim=128,
        output_dim=6,
        num_hidden_layers=4
    )
