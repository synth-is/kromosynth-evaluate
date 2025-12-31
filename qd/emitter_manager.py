"""
Emitter Manager for CMA-MAE (CMA-ES with MAP-Elites Archive).

Manages multiple Evolution Strategy emitters that generate candidate
solutions using CMA-ES optimization guided by archive performance.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any

from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from .archive_manager import ArchiveManager


class EmitterManager:
    """
    Manages CMA-MAE emitters for Quality Diversity search.

    Uses multiple parallel Evolution Strategy emitters to efficiently
    explore and optimize across the behavior space.
    """

    def __init__(
        self,
        archive: ArchiveManager,
        num_emitters: int = 5,
        sigma0: float = 0.5,
        batch_size: int = 36,
        ranker: str = "imp",
        selection_rule: str = "filter",
        restart_rule: str = "basic",
        bounds: Optional[List[Tuple[float, float]]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize CMA-MAE emitters.

        Args:
            archive: ArchiveManager instance
            num_emitters: Number of parallel emitters
            sigma0: Initial step size for CMA-ES
            batch_size: Solutions per emitter per ask()
            ranker: Ranking method ("imp" for improvement, "obj" for objective)
            selection_rule: Parent selection ("filter" or "mu")
            restart_rule: Restart strategy ("basic" or "no_improvement")
            bounds: Solution space bounds (defaults to None = unbounded)
            seed: Random seed for reproducibility
        """
        self.archive_manager = archive
        self.num_emitters = num_emitters
        self.sigma0 = sigma0
        self.batch_size = batch_size
        self.ranker = ranker
        self.selection_rule = selection_rule
        self.restart_rule = restart_rule
        self.bounds = bounds
        self.seed = seed

        # Generate initial solutions (random or from archive)
        initial_solutions = self._generate_initial_solutions()

        # Create emitters
        emitters = []
        for i in range(num_emitters):
            emitter = EvolutionStrategyEmitter(
                archive=self.archive_manager.archive,
                x0=initial_solutions[i],
                sigma0=sigma0,
                ranker=ranker,
                selection_rule=selection_rule,
                restart_rule=restart_rule,
                bounds=bounds,
                batch_size=batch_size,
                seed=seed + i if seed is not None else None
            )
            emitters.append(emitter)

        # Create scheduler
        self.scheduler = Scheduler(
            archive=self.archive_manager.archive,
            emitters=emitters
        )

        print(f"EmitterManager initialized:")
        print(f"  Num emitters: {num_emitters}")
        print(f"  Sigma0: {sigma0}")
        print(f"  Batch size: {batch_size}")
        print(f"  Ranker: {ranker}")
        print(f"  Total solutions per ask: {num_emitters * batch_size}")

    def _generate_initial_solutions(self) -> List[np.ndarray]:
        """
        Generate initial solutions for emitters.

        Strategy:
        1. Try to sample from archive if it has elites
        2. Otherwise generate random solutions

        Returns:
            List of initial solution vectors
        """
        solution_dim = self.archive_manager.solution_dim

        # Try to get elites from archive
        elites = self.archive_manager.sample_elites(
            n=self.num_emitters,
            return_metadata=False
        )

        initial_solutions = []

        if len(elites) >= self.num_emitters:
            # Use elite solutions
            for elite in elites[:self.num_emitters]:
                initial_solutions.append(elite['solution'])
        else:
            # Generate random solutions
            for i in range(self.num_emitters):
                if i < len(elites):
                    # Use available elite
                    initial_solutions.append(elites[i]['solution'])
                else:
                    # Generate random solution in [0, 1] range
                    # This assumes solution space is normalized
                    x0 = np.random.uniform(0.0, 1.0, size=solution_dim)

                    # If bounds are specified, scale to bounds
                    if self.bounds is not None:
                        for dim, (low, high) in enumerate(self.bounds):
                            x0[dim] = low + x0[dim] * (high - low)

                    initial_solutions.append(x0)

        return initial_solutions

    def ask(self) -> Tuple[np.ndarray, List[int]]:
        """
        Get candidate solutions from emitters.

        Returns:
            (solutions, emitter_ids) tuple where:
            - solutions: (N, solution_dim) array of candidate solutions
            - emitter_ids: List of emitter IDs for each solution
        """
        # Ask scheduler for solutions
        solutions = self.scheduler.ask()

        # Get emitter IDs (which emitter generated each solution)
        emitter_ids = []
        for i in range(self.num_emitters):
            emitter_ids.extend([i] * self.batch_size)

        return solutions, emitter_ids

    def tell(
        self,
        solutions: np.ndarray,
        objectives: np.ndarray,
        behavior_descriptors: np.ndarray,
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Report evaluation results to emitters.

        Args:
            solutions: (N, solution_dim) array of solutions
            objectives: (N,) array of objectives
            behavior_descriptors: (N, bd_dim) array of BDs
            metadata: Optional list of metadata dicts

        Returns:
            Statistics about archive updates:
            {
                "num_added": int,
                "num_new": int,
                "num_improved": int,
                "indices": list
            }
        """
        # Tell scheduler about results
        # Tell scheduler about results
        tell_kwargs = {
            "objective": objectives,
            "measures": behavior_descriptors,
        }
        if metadata is not None:
            tell_kwargs["metadata"] = metadata

        self.scheduler.tell(**tell_kwargs)
        # Get archive update statistics
        # Note: We need to track this separately as scheduler.tell()
        # doesn't return statistics directly
        # For now, we'll call archive.add_batch again to get stats
        # (pyribs scheduler already added them, but this is idempotent)
        stats = self.archive_manager.add_batch(
            solutions=solutions,
            objectives=objectives,
            behavior_descriptors=behavior_descriptors,
            metadata=metadata
        )

        return stats

    def ask_tell(
        self,
        objective_fn,
        bd_fn,
        metadata_fn = None
    ) -> Dict[str, Any]:
        """
        Convenience method: ask, evaluate, and tell in one step.

        Args:
            objective_fn: Function that takes solutions array and returns objectives
            bd_fn: Function that takes solutions array and returns BDs
            metadata_fn: Optional function that takes solutions and returns metadata list

        Returns:
            Statistics from tell()
        """
        # Ask
        solutions, emitter_ids = self.ask()

        # Evaluate
        objectives = objective_fn(solutions)
        bds = bd_fn(solutions)
        metadata = metadata_fn(solutions) if metadata_fn else None

        # Tell
        stats = self.tell(solutions, objectives, bds, metadata)

        return stats

    def get_stats(self) -> Dict[str, Any]:
        """
        Get emitter and archive statistics.

        Returns:
            Combined statistics dict
        """
        archive_stats = self.archive_manager.get_stats()

        return {
            **archive_stats,
            'num_emitters': self.num_emitters,
            'batch_size': self.batch_size,
            'total_batch_size': self.num_emitters * self.batch_size
        }

    def reset_emitters(self):
        """
        Reset all emitters (e.g., after remapping).

        This reinitializes emitters with new solutions from the archive.
        """
        # Generate new initial solutions
        initial_solutions = self._generate_initial_solutions()

        # Recreate emitters
        emitters = []
        for i in range(self.num_emitters):
            emitter = EvolutionStrategyEmitter(
                archive=self.archive_manager.archive,
                x0=initial_solutions[i],
                sigma0=self.sigma0,
                ranker=self.ranker,
                selection_rule=self.selection_rule,
                restart_rule=self.restart_rule,
                bounds=self.bounds,
                batch_size=self.batch_size,
                seed=self.seed + i if self.seed is not None else None
            )
            emitters.append(emitter)

        # Recreate scheduler
        self.scheduler = Scheduler(
            archive=self.archive_manager.archive,
            emitters=emitters
        )

        print("Emitters reset")
