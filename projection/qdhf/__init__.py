"""
QDHF (Quality Diversity through Human Feedback) Projection.

Learned projection from CLAP embeddings (512D) to behavior descriptors (6D)
using triplet loss and proxy similarity judgments.
"""

from .projection_network import ProjectionNetwork
from .proxy_triplet_generator import ProxyTripletGenerator
from .triplet_trainer import TripletTrainer

__all__ = [
    'ProjectionNetwork',
    'ProxyTripletGenerator',
    'TripletTrainer'
]
