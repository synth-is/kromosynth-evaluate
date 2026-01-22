"""
CLAP (Contrastive Language-Audio Pretraining) feature extraction module.

This module provides CLAP embedding extraction for audio, which can be used
as perceptual features for Quality Diversity search and learned behavior descriptors.
"""

from .clap_extractor import CLAPExtractor

__all__ = ['CLAPExtractor']
