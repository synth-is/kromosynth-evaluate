"""
Genome Codec for kromosynth genomes.

## Purpose

Bridges the gap between kromosynth's variable-structure CPPN genomes
and pyribs' requirement for fixed-length solution vectors.

## The Problem

- **kromosynth genomes**: Variable-structure graphs (different numbers of
  nodes/connections), hierarchical dictionaries, can grow/shrink during evolution
- **pyribs CVTArchive**: Requires fixed-length numpy arrays for all solutions

## The Solution

GenomeCodec acts as a translator, converting complex genomes to/from
simple vectors that pyribs can store and manipulate.

## Strategies Provided

A) **GenomeIDCodec** (recommended): Store genome externally, use ID as solution
   - Handles any genome complexity
   - No size limits
   - Preserves all structure

B) **ParameterVectorCodec**: Flatten parameters to vector (requires fixed structure)
   - Enables CMA-ES parameter optimization
   - Compact representation
   - Only for fixed-topology genomes

For detailed explanation, see GENOME_CODEC_EXPLAINED.md
"""

import numpy as np
import json
from typing import Dict, Any, Union, List, Optional, Tuple
from pathlib import Path


class GenomeCodec:
    """
    Base class for genome encoding/decoding.

    Subclasses implement specific encoding strategies.
    """

    def __init__(self, solution_dim: int):
        """
        Initialize codec.

        Args:
            solution_dim: Dimensionality of solution vectors
        """
        self.solution_dim = solution_dim

    def encode(self, genome: Union[Dict, str]) -> np.ndarray:
        """
        Convert genome to solution vector.

        Args:
            genome: Kromosynth genome (dict or JSON string)

        Returns:
            Solution vector (1D numpy array)
        """
        raise NotImplementedError

    def decode(self, solution: np.ndarray) -> Dict:
        """
        Convert solution vector back to genome.

        Args:
            solution: Solution vector

        Returns:
            Kromosynth genome dict
        """
        raise NotImplementedError


class GenomeIDCodec(GenomeCodec):
    """
    ID-based reference codec.

    Stores genomes externally and uses genome ID as the solution vector.
    This allows variable-structure genomes without size constraints.

    The solution vector is simply: [genome_id, padding...]

    Genomes are stored in a directory with filenames: genome_{id}.json
    """

    def __init__(
        self,
        solution_dim: int = 1,
        genome_dir: Optional[str] = None
    ):
        """
        Initialize ID-based codec.

        Args:
            solution_dim: Must be >= 1 (just needs to store ID)
            genome_dir: Directory to store genomes (defaults to ./genomes)
        """
        if solution_dim < 1:
            raise ValueError("solution_dim must be >= 1 for ID codec")

        super().__init__(solution_dim)

        if genome_dir is None:
            genome_dir = "./genomes"

        self.genome_dir = Path(genome_dir)
        self.genome_dir.mkdir(parents=True, exist_ok=True)

        # Counter for generating unique IDs
        self.next_id = self._get_next_id()

    def _get_next_id(self) -> int:
        """Get next available genome ID."""
        existing = list(self.genome_dir.glob("genome_*.json"))
        if not existing:
            return 0

        # Extract IDs and find max
        ids = []
        for path in existing:
            try:
                id_str = path.stem.split('_')[1]
                ids.append(int(id_str))
            except (IndexError, ValueError):
                continue

        return max(ids) + 1 if ids else 0

    def encode(self, genome: Union[Dict, str]) -> np.ndarray:
        """
        Encode genome by storing it and returning its ID.

        Args:
            genome: Genome dict or JSON string

        Returns:
            Solution vector with genome ID
        """
        # Parse if string
        if isinstance(genome, str):
            genome = json.loads(genome)

        # Check if genome already has an ID
        if '_codec_id' in genome:
            genome_id = genome['_codec_id']
        else:
            # Assign new ID
            genome_id = self.next_id
            self.next_id += 1
            genome['_codec_id'] = genome_id

        # Save genome
        genome_path = self.genome_dir / f"genome_{genome_id}.json"
        with open(genome_path, 'w') as f:
            json.dump(genome, f, indent=2)

        # Create solution vector
        solution = np.zeros(self.solution_dim)
        solution[0] = float(genome_id)

        return solution

    def decode(self, solution: np.ndarray) -> Dict:
        """
        Decode genome by loading from ID.

        Args:
            solution: Solution vector containing genome ID

        Returns:
            Genome dict
        """
        genome_id = int(solution[0])
        genome_path = self.genome_dir / f"genome_{genome_id}.json"

        if not genome_path.exists():
            raise ValueError(f"Genome {genome_id} not found at {genome_path}")

        with open(genome_path, 'r') as f:
            genome = json.load(f)

        return genome

    def clear_genomes(self):
        """Delete all stored genomes."""
        for path in self.genome_dir.glob("genome_*.json"):
            path.unlink()
        self.next_id = 0


class ParameterVectorCodec(GenomeCodec):
    """
    Parameter-only codec for fixed-structure genomes.

    Assumes genomes have fixed structure (same nodes/connections)
    and only parameters vary. Flattens all numeric parameters into
    a fixed-length vector.

    This is simpler but less flexible - requires knowing the genome
    structure in advance.
    """

    def __init__(
        self,
        solution_dim: int,
        parameter_ranges: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Initialize parameter vector codec.

        Args:
            solution_dim: Number of parameters in genome
            parameter_ranges: Optional bounds for each parameter
        """
        super().__init__(solution_dim)
        self.parameter_ranges = parameter_ranges

    def encode(self, genome: Union[Dict, str]) -> np.ndarray:
        """
        Flatten genome parameters to vector.

        This is a placeholder - actual implementation depends on
        kromosynth genome structure.

        Args:
            genome: Genome dict

        Returns:
            Parameter vector
        """
        if isinstance(genome, str):
            genome = json.loads(genome)

        # Extract numeric parameters
        # This would need to be customized based on actual genome structure
        # For now, return random vector as placeholder
        solution = np.random.uniform(0, 1, size=self.solution_dim)

        # Store original genome for reconstruction
        # (This is a hack - in real implementation, we'd properly extract/reconstruct)
        if not hasattr(self, '_genome_cache'):
            self._genome_cache = {}

        # Use hash of solution as cache key
        key = hash(solution.tobytes())
        self._genome_cache[key] = genome

        return solution

    def decode(self, solution: np.ndarray) -> Dict:
        """
        Reconstruct genome from parameter vector.

        This is a placeholder - actual implementation depends on
        kromosynth genome structure.

        Args:
            solution: Parameter vector

        Returns:
            Genome dict
        """
        # Try to retrieve cached genome
        key = hash(solution.tobytes())
        if hasattr(self, '_genome_cache') and key in self._genome_cache:
            return self._genome_cache[key]

        # Otherwise, construct minimal genome
        # This would need real implementation
        genome = {
            "parameters": solution.tolist(),
            "_note": "Placeholder genome - needs proper implementation"
        }

        return genome


def create_codec(
    codec_type: str = "id",
    solution_dim: int = 1,
    **kwargs
) -> GenomeCodec:
    """
    Factory function to create genome codec.

    Args:
        codec_type: "id" or "parameter"
        solution_dim: Solution dimensionality
        **kwargs: Additional codec-specific arguments

    Returns:
        GenomeCodec instance
    """
    if codec_type == "id":
        return GenomeIDCodec(solution_dim=solution_dim, **kwargs)
    elif codec_type == "parameter":
        return ParameterVectorCodec(solution_dim=solution_dim, **kwargs)
    else:
        raise ValueError(f"Unknown codec type: {codec_type}")
