"""
Contextual Bandit with Thompson Sampling
Learns which mutations are most promising given sequence context
"""

import numpy as np
from collections import defaultdict


class ContextualBandit:
    """
    Contextual Bandit using Thompson Sampling

    For each mutation position, maintains a Beta distribution of success rates
    Samples from these distributions to balance exploration and exploitation
    """

    def __init__(self, oracle, k=1, seed=42, alpha_prior=1.0, beta_prior=1.0):
        """
        Initialize Contextual Bandit

        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            seed: Random seed
            alpha_prior: Prior successes (higher = more optimistic)
            beta_prior: Prior failures (higher = more conservative)
        """
        self.oracle = oracle
        self.k = k
        self.rng = np.random.RandomState(seed)

        # Thompson Sampling parameters
        # For each position, track successes and failures
        self.alpha = defaultdict(lambda: alpha_prior)  # Successes
        self.beta = defaultdict(lambda: beta_prior)  # Failures

        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

    def select_mutation(self, current_seq, current_fitness):
        """
        Select mutation using Thompson Sampling

        Samples success probability for each position from Beta distribution
        Picks position with highest sampled probability

        Args:
            current_seq: Current sequence
            current_fitness: Current fitness

        Returns:
            (position, from_aa, to_aa) mutation
        """
        from src.utils.mutations import AMINO_ACIDS

        # Sample success probability for each position
        position_scores = []

        for pos in range(len(current_seq)):
            # Get Beta parameters for this position
            alpha = self.alpha[pos]
            beta = self.beta[pos]

            # Sample from Beta distribution
            success_prob = self.rng.beta(alpha, beta)
            position_scores.append((success_prob, pos))

        # Sort positions by sampled probability
        position_scores.sort(reverse=True)

        # Try top positions until we find valid mutation
        for _, pos in position_scores[:10]:  # Try top 10
            wt_aa = current_seq[pos]

            # Pick random different amino acid
            possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            to_aa = self.rng.choice(possible_aas)

            return (pos, wt_aa, to_aa)

        # Fallback: random mutation
        pos = self.rng.randint(len(current_seq))
        wt_aa = current_seq[pos]
        possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        to_aa = self.rng.choice(possible_aas)

        return (pos, wt_aa, to_aa)

    def update(self, position, improved):
        """
        Update Beta distribution for a position

        Args:
            position: Mutated position
            improved: Whether mutation improved fitness
        """
        if improved:
            self.alpha[position] += 1
        else:
            self.beta[position] += 1

    def optimize(self, wt_sequence, budget=500):
        """
        Run Contextual Bandit optimization

        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries

        Returns:
            Dict with results
        """
        from src.utils.mutations import apply_mutations

        print(f"Contextual Bandit (Thompson Sampling, k={self.k}, budget={budget})")
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

        improvements = 0

        # Optimization loop
        for i in range(budget - 1):
            # Select mutation using Thompson Sampling
            mutation = self.select_mutation(current_seq, current_fitness)
            pos, from_aa, to_aa = mutation

            # Apply mutation
            mutant_seq = apply_mutations(current_seq, [mutation])
            mutation_desc = f"{from_aa}{pos+1}{to_aa}"

            # Score mutant
            mutant_fitness = self.oracle.score_sequence(mutant_seq)
            queries_used += 1

            history.append((mutant_seq, mutant_fitness, mutation_desc))

            # Check if improved
            improved = mutant_fitness > current_fitness

            # Update Thompson Sampling statistics
            self.update(pos, improved)

            # Accept if improved
            if improved:
                improvements += 1
                current_seq = mutant_seq
                current_fitness = mutant_fitness

            # Update global best
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_seq = current_seq

            # Progress
            if (i + 1) % 100 == 0:
                print(
                    f"  {i+1}/{budget-1}: Best={best_fitness:.4f}, Improvements={improvements}"
                )

        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Improvements: {improvements}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")

        return {
            "method": "ContextualBandit",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
            "improvements": improvements,
        }
