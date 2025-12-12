"""
Greedy hill-climbing baseline
Always picks the best neighbor, no backtracking
"""

import numpy as np


class GreedyBaseline:
    """
    Greedy hill-climbing

    At each step:
    1. Generate all k-point neighbors
    2. Score all neighbors
    3. Pick best neighbor
    4. If better than current, move there
    5. Repeat until no improvement or budget exhausted
    """

    def __init__(self, oracle, k=1):
        """
        Initialize greedy baseline

        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
        """
        self.oracle = oracle
        self.k = k

    def optimize(self, wt_sequence, budget=500):
        """
        Run greedy hill-climbing

        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries

        Returns:
            Dict with results
        """
        from src.utils.mutations import get_single_mutants

        print(f"Greedy Hill-Climbing (k={self.k}, budget={budget})")
        print("-" * 70)

        # Start with WT
        current_seq = wt_sequence
        current_fitness = self.oracle.score_sequence(current_seq)

        best_seq = current_seq
        best_fitness = current_fitness

        # History
        history = [(current_seq, current_fitness, "WT")]

        # Reset oracle query count
        self.oracle.reset_query_count()
        queries_used = 1  # Counted WT

        step = 0

        # MODIFIED: Sample neighbors instead of scoring all
        max_neighbors = 500  # Maximum neighbors to check per step

        while queries_used < budget:
            step += 1

            # Generate all single-point neighbors
            if self.k == 1:
                neighbors = get_single_mutants(current_seq)
            else:
                # For k>1, would need to sample neighbors
                raise NotImplementedError(f"k={self.k} not yet supported")

            # Score all neighbors (up to budget)
            best_neighbor_seq = None
            best_neighbor_fitness = current_fitness
            best_neighbor_desc = None

            for neighbor_seq, mutation_desc in neighbors:
                if queries_used >= budget:
                    break

                fitness = self.oracle.score_sequence(neighbor_seq)
                queries_used += 1

                history.append((neighbor_seq, fitness, mutation_desc))

                # Track best neighbor
                if fitness > best_neighbor_fitness:
                    best_neighbor_fitness = fitness
                    best_neighbor_seq = neighbor_seq
                    best_neighbor_desc = mutation_desc

            # If found better neighbor, move there
            if best_neighbor_seq is not None:
                improvement = best_neighbor_fitness - current_fitness
                print(
                    f"  Step {step}: {best_neighbor_desc}, fitness={best_neighbor_fitness:.4f} (+{improvement:.4f})"
                )

                current_seq = best_neighbor_seq
                current_fitness = best_neighbor_fitness

                # Update global best
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_seq = current_seq
            else:
                # No improvement, stuck at local optimum
                print(f"  Step {step}: No improvement found, stopping")
                break

        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Steps taken: {step}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")

        return {
            "method": "Greedy",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
            "steps": step,
        }
