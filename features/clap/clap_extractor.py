"""
CLAP (Contrastive Language-Audio Pretraining) feature extractor.

Extracts 512-dimensional embeddings from audio using LAION-CLAP model.
These embeddings capture perceptual audio characteristics and can be used
as input for learned behavior descriptors in QD search.
"""

import numpy as np
import torch
import laion_clap
from typing import List, Optional, Union
import warnings


class CLAPExtractor:
    """Extract CLAP embeddings from audio buffers."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        enable_fusion: bool = False,
        amodel: str = 'HTSAT-tiny'
    ):
        """
        Initialize CLAP extractor with music checkpoint.

        Args:
            checkpoint_path: Path to CLAP checkpoint file. If None, will download default.
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            enable_fusion: Whether to enable fusion model (default: False for audio-only)
            amodel: Audio model architecture (default: 'HTSAT-tiny')
        """
        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Initializing CLAP extractor on {self.device}...")

        # Initialize CLAP model
        self.model = laion_clap.CLAP_Module(
            enable_fusion=enable_fusion,
            device=self.device,
            amodel=amodel
        )

        # Load checkpoint
        if checkpoint_path:
            print(f"Loading checkpoint from {checkpoint_path}")
            self.model.load_ckpt(checkpoint_path)
        else:
            # Use default music checkpoint
            print("Loading default music checkpoint (music_audioset_epoch_15_esc_90.14.pt)")
            self.model.load_ckpt()  # Downloads default if not cached

        # Set model to eval mode
        self.model.model.eval()

        # Expected sample rate for CLAP
        self.target_sample_rate = 48000

        print("CLAP extractor initialized successfully")

    def _preprocess_audio(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Preprocess audio to CLAP requirements.

        Args:
            audio_buffer: Audio samples as numpy array
            sample_rate: Sample rate of input audio

        Returns:
            Preprocessed audio ready for CLAP
        """
        # Ensure numpy array
        if not isinstance(audio_buffer, np.ndarray):
            audio_buffer = np.array(audio_buffer)

        # Handle stereo -> mono conversion
        if len(audio_buffer.shape) > 1:
            if audio_buffer.shape[0] == 2:  # (2, samples) format
                audio_buffer = np.mean(audio_buffer, axis=0)
            elif audio_buffer.shape[1] == 2:  # (samples, 2) format
                audio_buffer = np.mean(audio_buffer, axis=1)

        # Ensure float32
        if audio_buffer.dtype != np.float32:
            audio_buffer = audio_buffer.astype(np.float32)

        # Resample if needed
        if sample_rate != self.target_sample_rate:
            try:
                import resampy
                audio_buffer = resampy.resample(
                    audio_buffer,
                    sample_rate,
                    self.target_sample_rate,
                    filter='kaiser_best'
                )
            except ImportError:
                warnings.warn(
                    f"resampy not available, using scipy for resampling. "
                    f"Install resampy for better quality."
                )
                from scipy import signal
                num_samples = int(len(audio_buffer) * self.target_sample_rate / sample_rate)
                audio_buffer = signal.resample(audio_buffer, num_samples)

        # Normalize to [-1, 1] range
        max_val = np.abs(audio_buffer).max()
        if max_val > 0:
            audio_buffer = audio_buffer / max_val

        return audio_buffer

    def extract_embedding(
        self,
        audio_buffer: np.ndarray,
        sample_rate: int
    ) -> np.ndarray:
        """
        Extract 512D CLAP embedding from audio.

        Args:
            audio_buffer: Audio samples as numpy array
            sample_rate: Sample rate of input audio

        Returns:
            512-dimensional embedding vector
        """
        # Preprocess
        audio = self._preprocess_audio(audio_buffer, sample_rate)

        # Add batch dimension if needed (CLAP expects list of arrays)
        audio_list = [audio]

        # Extract embedding
        with torch.no_grad():
            embeddings = self.model.get_audio_embedding_from_data(
                x=audio_list,
                use_tensor=False  # Return numpy array
            )

        # Return first (and only) embedding
        return embeddings[0]

    def extract_batch(
        self,
        audio_buffers: List[np.ndarray],
        sample_rate: int
    ) -> np.ndarray:
        """
        Batch extraction for efficiency.

        Args:
            audio_buffers: List of audio samples as numpy arrays
            sample_rate: Sample rate of input audio (same for all)

        Returns:
            (N, 512) array of embeddings
        """
        # Preprocess all buffers
        audio_list = [
            self._preprocess_audio(buffer, sample_rate)
            for buffer in audio_buffers
        ]

        # Extract embeddings in batch
        with torch.no_grad():
            embeddings = self.model.get_audio_embedding_from_data(
                x=audio_list,
                use_tensor=False
            )

        return embeddings

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding (512D)
            embedding2: Second embedding (512D)

        Returns:
            Cosine similarity in [0, 1] range
        """
        # Normalize
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

        # Clamp to [0, 1] (sometimes numerical errors can cause slight overflow)
        similarity = np.clip(similarity, 0.0, 1.0)

        return float(similarity)

    def compute_distance(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """
        Compute distance between two embeddings (1 - similarity).

        Args:
            embedding1: First embedding (512D)
            embedding2: Second embedding (512D)

        Returns:
            Distance in [0, 1] range
        """
        return 1.0 - self.compute_similarity(embedding1, embedding2)
