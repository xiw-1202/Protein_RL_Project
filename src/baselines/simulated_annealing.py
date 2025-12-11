"""
Simulated Annealing baseline
Probabilistically accepts worse solutions to escape local optima
"""

import numpy as np


class SimulatedAnnealingBaseline:
    """
    Simulated Annealing

    Accepts worse solutions with probability exp(-ΔE/T)
    Temperature T decreases over time (annealing schedule)
    """

    def __init__(self, oracle, k=1, T_start=10.0, T_end=0.1, seed=42):
        """
        Initialize simulated annealing

        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            T_start: Starting temperature (higher = more exploration)
            T_end: Ending temperature (lower = more exploitation)
            seed: Random seed
        """
        self.oracle = oracle
        self.k = k
        self.T_start = T_start
        self.T_end = T_end
        self.rng = np.random.RandomState(seed)

    def optimize(self, wt_sequence, budget=500):
        """
        Run simulated annealing

        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries

        Returns:
            Dict with results
        """
        from src.utils.mutations import get_random_mutant

        print(f"Simulated Annealing (k={self.k}, budget={budget})")
        print(f"  Temperature: {self.T_start:.2f} → {self.T_end:.2f}")
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
        queries_used = 1

        # Annealing loop
        for i in range(budget - 1):
            # Temperature decay (linear)
            progress = i / (budget - 1)
            T = self.T_start * (1 - progress) + self.T_end * progress

            # Generate random neighbor
            neighbor_seq, mutation_desc = get_random_mutant(
                current_seq, k=self.k, rng=self.rng
            )

            # Score neighbor
            neighbor_fitness = self.oracle.score_sequence(neighbor_seq)
            queries_used += 1

            history.append((neighbor_seq, neighbor_fitness, mutation_desc))

            # Acceptance criterion
            delta = neighbor_fitness - current_fitness

            if delta > 0:
                # Always accept improvements
                accept = True
            else:
                # Accept worse solutions with probability exp(delta/T)
                accept_prob = np.exp(delta / T)
                accept = self.rng.random() < accept_prob

            if accept:
                current_seq = neighbor_seq
                current_fitness = neighbor_fitness

            # Update global best
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_seq = current_seq

            # Progress
            if (i + 1) % 100 == 0:
                print(
                    f"  {i+1}/{budget-1}: T={T:.3f}, Best={best_fitness:.4f}, Current={current_fitness:.4f}"
                )

        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")

        return {
            "method": "SimulatedAnnealing",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
        }
