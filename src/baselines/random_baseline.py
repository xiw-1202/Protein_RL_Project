"""
Random sampling baseline
Simply samples random k-point mutants uniformly
"""

import numpy as np
from typing import Dict


class RandomBaseline:
    """
    Random sampling baseline

    Samples random k-point mutants uniformly until budget exhausted
    """

    def __init__(self, oracle, k=1, seed=42):
        """
        Initialize random baseline

        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            seed: Random seed
        """
        self.oracle = oracle
        self.k = k
        self.rng = np.random.RandomState(seed)

    def optimize(self, wt_sequence, budget=500):
        """
        Run random search

        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries

        Returns:
            Dict with results
        """
        from src.utils.mutations import get_random_mutant

        print(f"Random Baseline (k={self.k}, budget={budget})")
        print("-" * 70)

        # Track best
        best_seq = wt_sequence
        best_fitness = self.oracle.score_sequence(wt_sequence)

        # History: [(sequence, fitness, mutation_desc)]
        history = [(wt_sequence, best_fitness, "WT")]

        # Reset oracle query count
        self.oracle.reset_query_count()

        # Random search
        for i in range(budget - 1):  # -1 because we already scored WT
            # Generate random mutant
            mutant_seq, mutation_desc = get_random_mutant(
                wt_sequence, k=self.k, rng=self.rng
            )

            # Score
            fitness = self.oracle.score_sequence(mutant_seq)

            # Track
            history.append((mutant_seq, fitness, mutation_desc))

            # Update best
            if fitness > best_fitness:
                best_fitness = fitness
                best_seq = mutant_seq

            # Progress
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{budget-1}: Best fitness = {best_fitness:.4f}")

        queries_used = self.oracle.get_query_count()

        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")

        return {
            "method": "Random",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
        }
