"""
Proxy Triplet Generator for QDHF cold-start.

Generates training triplets (anchor, similar, dissimilar) using CLAP
embedding distances as proxy for human perceptual similarity judgments.

This enables training the projection network before collecting actual
human similarity data.
"""

import numpy as np
from typing import List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances


class ProxyTripletGenerator:
    """
    Generate triplets from CLAP embeddings for cold-start training.

    Uses CLAP embedding distances as proxy for perceptual similarity:
    - Anchor: randomly selected sound
    - Similar (positive): nearby sound in CLAP space
    - Dissimilar (negative): distant sound in CLAP space

    This assumes CLAP embeddings capture perceptual similarity,
    which is reasonable given CLAP's training on audio-text pairs.
    """

    def __init__(
        self,
        clap_embeddings: np.ndarray,
        sound_ids: Optional[List[str]] = None,
        k_neighbors: int = 10,
        distance_threshold_percentile: float = 50.0,
        precompute_distances: bool = True
    ):
        """
        Initialize with corpus of CLAP embeddings.

        Args:
            clap_embeddings: (N, 512) array of CLAP embeddings
            sound_ids: Corresponding sound identifiers (optional)
            k_neighbors: Number of nearest neighbors to consider for positive
            distance_threshold_percentile: Percentile for dissimilar threshold
            precompute_distances: Whether to precompute distance matrix (faster but uses memory)
        """
        self.embeddings = clap_embeddings
        self.sound_ids = sound_ids or [f"sound_{i}" for i in range(len(clap_embeddings))]
        self.k_neighbors = k_neighbors
        self.distance_threshold_percentile = distance_threshold_percentile
        self.num_sounds = len(clap_embeddings)

        # Precompute distance matrix if requested
        self.distance_matrix = None
        if precompute_distances:
            print(f"Precomputing distance matrix for {self.num_sounds} sounds...")
            self.distance_matrix = self._compute_distances()

            # Compute distance threshold for dissimilar sounds
            triu_indices = np.triu_indices(self.num_sounds, k=1)
            all_distances = self.distance_matrix[triu_indices]
            self.distance_threshold = np.percentile(
                all_distances,
                distance_threshold_percentile
            )
            print(f"Distance threshold ({distance_threshold_percentile}th percentile): {self.distance_threshold:.4f}")

    def _compute_distances(self) -> np.ndarray:
        """
        Compute pairwise cosine distances.

        Returns:
            (N, N) distance matrix
        """
        return cosine_distances(self.embeddings)

    def _get_distances_to(self, anchor_idx: int) -> np.ndarray:
        """
        Get distances from anchor to all other sounds.

        Args:
            anchor_idx: Index of anchor sound

        Returns:
            (N,) array of distances
        """
        if self.distance_matrix is not None:
            return self.distance_matrix[anchor_idx]
        else:
            # Compute on-the-fly
            anchor_emb = self.embeddings[anchor_idx:anchor_idx+1]
            return cosine_distances(anchor_emb, self.embeddings)[0]

    def generate_triplet(self) -> Tuple[int, int, int]:
        """
        Generate one (anchor, positive, negative) triplet.

        Strategy:
        - Anchor: random sound
        - Positive: one of k-nearest neighbors (excluding self)
        - Negative: random sound from distant sounds (distance > threshold)

        Returns:
            (anchor_idx, positive_idx, negative_idx)
        """
        # Random anchor
        anchor_idx = np.random.randint(0, self.num_sounds)

        # Get distances to all other sounds
        distances = self._get_distances_to(anchor_idx)

        # Find k nearest neighbors (excluding self)
        distances_copy = distances.copy()
        distances_copy[anchor_idx] = np.inf  # Exclude self
        k_nearest_indices = np.argpartition(distances_copy, self.k_neighbors)[:self.k_neighbors]

        # Random positive from k-nearest
        positive_idx = np.random.choice(k_nearest_indices)

        # Find distant sounds
        if self.distance_matrix is not None:
            # Use precomputed threshold
            distant_mask = (distances > self.distance_threshold) & (np.arange(self.num_sounds) != anchor_idx)
        else:
            # Compute threshold on-the-fly
            threshold = np.median(distances)
            distant_mask = (distances > threshold) & (np.arange(self.num_sounds) != anchor_idx)

        distant_indices = np.where(distant_mask)[0]

        if len(distant_indices) == 0:
            # Fallback: use furthest sound
            negative_idx = np.argmax(distances)
        else:
            # Random negative from distant sounds
            negative_idx = np.random.choice(distant_indices)

        return anchor_idx, positive_idx, negative_idx

    def generate_batch(self, n: int) -> np.ndarray:
        """
        Generate batch of triplet indices.

        Args:
            n: Number of triplets to generate

        Returns:
            (n, 3) array of triplet indices
        """
        triplets = np.zeros((n, 3), dtype=np.int64)
        for i in range(n):
            triplets[i] = self.generate_triplet()
        return triplets

    def get_embeddings_for_triplets(
        self,
        triplet_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get embedding tensors for anchor, positive, negative.

        Args:
            triplet_indices: (n, 3) array of triplet indices

        Returns:
            (anchors, positives, negatives) each (n, 512)
        """
        anchors = self.embeddings[triplet_indices[:, 0]]
        positives = self.embeddings[triplet_indices[:, 1]]
        negatives = self.embeddings[triplet_indices[:, 2]]

        return anchors, positives, negatives

    def validate_triplet(
        self,
        anchor_idx: int,
        positive_idx: int,
        negative_idx: int
    ) -> bool:
        """
        Validate that triplet satisfies similarity constraint.

        Returns True if distance(anchor, positive) < distance(anchor, negative)
        """
        distances = self._get_distances_to(anchor_idx)
        return distances[positive_idx] < distances[negative_idx]

    def get_statistics(self) -> dict:
        """
        Get statistics about the embedding corpus.

        Returns:
            Dict with corpus statistics
        """
        if self.distance_matrix is None:
            self.distance_matrix = self._compute_distances()

        # Get upper triangle (avoid duplicates and self-distances)
        triu_indices = np.triu_indices(self.num_sounds, k=1)
        all_distances = self.distance_matrix[triu_indices]

        return {
            'num_sounds': self.num_sounds,
            'embedding_dim': self.embeddings.shape[1],
            'distance_min': float(np.min(all_distances)),
            'distance_max': float(np.max(all_distances)),
            'distance_mean': float(np.mean(all_distances)),
            'distance_median': float(np.median(all_distances)),
            'distance_std': float(np.std(all_distances)),
            'distance_threshold': float(self.distance_threshold) if hasattr(self, 'distance_threshold') else None
        }


class HardNegativeTripletGenerator(ProxyTripletGenerator):
    """
    Triplet generator with hard negative mining.

    Instead of sampling random distant sounds, samples the closest
    distant sound (hardest negative) to make training more challenging.
    """

    def generate_triplet(self) -> Tuple[int, int, int]:
        """
        Generate triplet with hard negative.

        Negative is the closest sound among distant sounds (semi-hard mining).
        """
        # Random anchor
        anchor_idx = np.random.randint(0, self.num_sounds)

        # Get distances
        distances = self._get_distances_to(anchor_idx)

        # Find k nearest for positive
        distances_copy = distances.copy()
        distances_copy[anchor_idx] = np.inf
        k_nearest_indices = np.argpartition(distances_copy, self.k_neighbors)[:self.k_neighbors]
        positive_idx = np.random.choice(k_nearest_indices)

        # Get positive distance
        positive_distance = distances[positive_idx]

        # Find hard negative: closest sound that's further than positive
        # (semi-hard negative mining)
        candidate_mask = (distances > positive_distance) & (np.arange(self.num_sounds) != anchor_idx)
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            # Fallback: use furthest sound
            negative_idx = np.argmax(distances)
        else:
            # Use closest among candidates (hardest negative)
            negative_idx = candidate_indices[np.argmin(distances[candidate_indices])]

        return anchor_idx, positive_idx, negative_idx
