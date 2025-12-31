"""
Archive Manager for pyribs CVT-MAP-Elites.

Wraps ribs.archives.CVTArchive with application-specific logic for:
- Genome storage and retrieval
- Archive statistics and monitoring
- Persistence (save/load)
- Remapping when behavior descriptors change
"""

import numpy as np
import pickle
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ribs.archives import CVTArchive


class ArchiveManager:
    """
    Manages CVT archive for Quality Diversity search.

    The archive stores elite solutions (genomes) organized by their
    behavior descriptors in a Voronoi tessellation of the behavior space.
    """

    def __init__(
        self,
        solution_dim: int,
        bd_dim: int = 6,
        num_cells: int = 10000,
        ranges: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None,
        use_list: bool = False
    ):
        """
        Initialize CVT archive.

        Args:
            solution_dim: Dimensionality of solution vectors (genome encoding)
            bd_dim: Behavior descriptor dimensionality
            num_cells: Number of CVT cells/niches
            ranges: BD range per dimension, defaults to [(0,1)] * bd_dim
            seed: Random seed for reproducibility
            use_list: Whether to use list-based storage (for variable-length genomes)
        """
        self.solution_dim = solution_dim
        self.bd_dim = bd_dim
        self.num_cells = num_cells
        self.seed = seed
        self.use_list = use_list

        # Default ranges: [0, 1] for each dimension
        if ranges is None:
            ranges = [(0.0, 1.0)] * bd_dim
        self.ranges = ranges

        # Create CVT archive
        self.archive = CVTArchive(
            solution_dim=solution_dim,
            cells=num_cells,
            ranges=ranges,
            seed=seed,
            dtype=np.float64 if not use_list else object  # object dtype for lists
        )

        print(f"CVTArchive initialized:")
        print(f"  Solution dim: {solution_dim}")
        print(f"  BD dim: {bd_dim}")
        print(f"  Cells: {num_cells}")
        print(f"  Ranges: {ranges}")
        print(f"  Use list: {use_list}")

    def add(
        self,
        solution: np.ndarray,
        objective: float,
        behavior_descriptor: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[int]]:
        """
        Add solution to archive.

        Args:
            solution: Solution vector (genome encoding)
            objective: Fitness/quality score
            behavior_descriptor: BD vector
            metadata: Optional metadata to store with solution

        Returns:
            (was_added, cell_index) tuple where:
            - was_added: True if solution was added/improved cell
            - cell_index: Index of cell that was updated (or None)
        """
        # pyribs expects 2D arrays (batch dimension)
        solutions = solution.reshape(1, -1) if not self.use_list else [solution]
        objectives = np.array([objective])
        bds = behavior_descriptor.reshape(1, -1)

        # Add to archive
        status_batch, value_batch = self.archive.add(
            solutions,
            objectives,
            bds,
            metadata=[metadata] if metadata else None
        )

        # Extract single result
        was_added = status_batch[0] > 0  # status > 0 means added or improved
        cell_index = value_batch[0] if was_added else None

        return was_added, cell_index

    def add_batch(
        self,
        solutions: np.ndarray,
        objectives: np.ndarray,
        behavior_descriptors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Batch add solutions to archive.

        Args:
            solutions: (N, solution_dim) array of solutions
            objectives: (N,) array of objectives
            behavior_descriptors: (N, bd_dim) array of BDs
            metadata: Optional list of metadata dicts

        Returns:
            Statistics dict with:
            - num_added: Number of solutions added
            - num_improved: Number of cells improved
            - indices: List of cell indices updated
        """
        # Add to archive
        add_kwargs = {
            "solution": solutions,
            "objective": objectives,
            "measures": behavior_descriptors,
        }
        if metadata is not None:
             add_kwargs["metadata"] = metadata

        result = self.archive.add(**add_kwargs)
        if isinstance(result, tuple):
             status_batch, value_batch = result
        elif isinstance(result, dict):
             status_batch = result.get("status")
             value_batch = result.get("value")
             if status_batch is None:
                 print(f"ERROR: archive.add returned dict keys: {result.keys()}")
                 raise ValueError("Unexpected dict from archive.add")
        else:
             print(f"ERROR: archive.add returned {type(result)}")
             raise ValueError("Unexpected type from archive.add")
        # Analyze results
        # status: 0 = not added, 1 = new cell, 2 = improved cell
        num_new = np.sum(status_batch == 1)
        num_improved = np.sum(status_batch == 2)
        num_added = num_new + num_improved

        indices = value_batch[status_batch > 0].tolist()

        return {
            'num_added': int(num_added),
            'num_new': int(num_new),
            'num_improved': int(num_improved),
            'indices': indices
        }

    def sample_elites(
        self,
        n: int,
        return_metadata: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Sample n elites from archive for parent selection.

        Args:
            n: Number of elites to sample
            return_metadata: Whether to include metadata in results

        Returns:
            List of dicts with keys:
            - solution: Solution vector
            - objective: Fitness score
            - behavior_descriptor: BD vector
            - index: Archive cell index
            - metadata: (if return_metadata=True)
        """
        # Get elite data
        data = self.archive.data(return_type="dict")

        if len(data["solution"]) == 0:
            return []

        # Sample random indices
        num_elites = len(data["solution"])
        if n > num_elites:
            n = num_elites

        sample_indices = np.random.choice(num_elites, size=n, replace=False)

        # Build result
        elites = []
        for idx in sample_indices:
            elite = {
                'solution': data["solution"][idx],
                'objective': data["objective"][idx],
                'behavior_descriptor': data["measures"][idx],
                'index': data["index"][idx]
            }
            if return_metadata and data.get("metadata") is not None:
                elite['metadata'] = data["metadata"][idx]
            elites.append(elite)

        return elites

    def get_stats(self) -> Dict[str, Any]:
        """
        Return archive statistics.

        Returns:
            {
                "qd_score": float,
                "coverage": float,
                "max_fitness": float,
                "mean_fitness": float,
                "num_elites": int,
                "cells_filled": int,
                "cells_total": int
            }
        """
        stats = self.archive.stats
        data = self.archive.data(return_type="dict")

        num_elites = len(data["solution"])
        max_fitness = float(np.max(data["objective"])) if num_elites > 0 else 0.0
        mean_fitness = float(np.mean(data["objective"])) if num_elites > 0 else 0.0

        return {
            "qd_score": float(stats.qd_score),
            "coverage": float(stats.coverage),
            "max_fitness": max_fitness,
            "mean_fitness": mean_fitness,
            "num_elites": num_elites,
            "cells_filled": num_elites,
            "cells_total": self.num_cells
        }

    def save(self, path: str):
        """
        Save archive state to disk.

        Args:
            path: File path for saving (will create parent dirs if needed)
        """
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        # Save archive using pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'archive': self.archive,
                'solution_dim': self.solution_dim,
                'bd_dim': self.bd_dim,
                'num_cells': self.num_cells,
                'ranges': self.ranges,
                'seed': self.seed,
                'use_list': self.use_list
            }, f)

        print(f"Archive saved to {path}")
        print(f"  Elites: {len(self.archive.data(return_type='dict')['solution'])}")

    def load(self, path: str):
        """
        Load archive state from disk.

        Args:
            path: File path to load from
        """
        with open(path, 'rb') as f:
            data = pickle.load(f)

        # Restore state
        self.archive = data['archive']
        self.solution_dim = data['solution_dim']
        self.bd_dim = data['bd_dim']
        self.num_cells = data['num_cells']
        self.ranges = data['ranges']
        self.seed = data.get('seed')
        self.use_list = data.get('use_list', False)

        num_elites = len(self.archive.data(return_type='dict')['solution'])
        print(f"Archive loaded from {path}")
        print(f"  Elites: {num_elites}")

    def remap(
        self,
        new_bds: np.ndarray,
        new_ranges: Optional[List[Tuple[float, float]]] = None
    ):
        """
        Remap all elites to new behavior descriptors.

        Used when projection model is updated and we need to re-project
        all solutions with the new model.

        Args:
            new_bds: (N, bd_dim) array of new BDs for all elites
            new_ranges: Optional new BD ranges (defaults to current ranges)

        Note:
            The number of BDs must match the current number of elites.
            Order should match the order from archive.data().
        """
        # Get current elite data
        data = self.archive.data(return_type="dict")
        num_elites = len(data["solution"])

        if len(new_bds) != num_elites:
            raise ValueError(
                f"Number of new BDs ({len(new_bds)}) doesn't match "
                f"number of elites ({num_elites})"
            )

        # Create new archive with potentially new ranges
        if new_ranges is None:
            new_ranges = self.ranges

        new_archive = CVTArchive(
            solution_dim=self.solution_dim,
            cells=self.num_cells,
            ranges=new_ranges,
            seed=self.seed,
            dtype=np.float64 if not self.use_list else object
        )

        # Re-add all solutions with new BDs
        metadata = data.get("metadata")
        new_archive.add(
            data["solution"],
            data["objective"],
            new_bds,
            metadata=metadata
        )

        # Replace current archive
        self.archive = new_archive
        self.ranges = new_ranges

        new_num_elites = len(new_archive.data(return_type='dict')['solution'])
        print(f"Archive remapped:")
        print(f"  Old elites: {num_elites}")
        print(f"  New elites: {new_num_elites}")
        print(f"  Lost in remapping: {num_elites - new_num_elites}")

    def get_elite_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """
        Get elite at specific archive cell index.

        Args:
            index: Archive cell index

        Returns:
            Elite dict or None if cell is empty
        """
        data = self.archive.data(return_type="dict")

        # Find elite at this index
        for i, idx in enumerate(data["index"]):
            if idx == index:
                elite = {
                    'solution': data["solution"][i],
                    'objective': data["objective"][i],
                    'behavior_descriptor': data["measures"][i],
                    'index': idx
                }
                if data.get("metadata"):
                    elite['metadata'] = data["metadata"][i]
                return elite

        return None

    def clear(self):
        """Clear all elites from archive (keeps structure)."""
        self.archive.clear()
        print("Archive cleared")
