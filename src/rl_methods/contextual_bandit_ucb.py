"""
Contextual Bandit with Upper Confidence Bound (UCB) - IMPROVED VERSION

Improvements over Thompson Sampling:
1. Uses UCB for exploration-exploitation tradeoff (theoretically grounded)
2. Deterministic selection (no sampling variance)
3. Explicit exploration bonus that decreases over time
"""

import numpy as np


class ContextualBanditUCB:
    """
    Contextual Bandit using Upper Confidence Bound (UCB)
    
    Key Differences from Thompson Sampling:
    - Thompson: Sample from Beta(alpha, beta) distributions
    - UCB: Deterministic selection based on mean + confidence interval
    
    UCB Formula: score = mean_reward + c * sqrt(ln(t) / n_trials)
    where:
    - mean_reward: estimated success probability
    - c: exploration coefficient (default 2.0)
    - t: total number of trials
    - n_trials: number of trials for this position
    """
    
    def __init__(self, oracle, k=1, seed=42, ucb_c=2.0):
        """
        Initialize UCB bandit
        
        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            seed: Random seed
            ucb_c: UCB exploration coefficient (default 2.0)
                  Higher = more exploration
                  Lower = more exploitation
        """
        self.oracle = oracle
        self.k = k
        self.ucb_c = ucb_c
        
        # Random state for tie-breaking
        np.random.seed(seed)
        self.rng = np.random.RandomState(seed)
        
        # Will be initialized when we know sequence length
        self.seq_length = None
        
        # UCB statistics for each position
        self.successes = None  # Number of successful mutations at each position
        self.trials = None     # Total mutations tried at each position
        self.total_trials = 0  # Global trial counter for UCB
    
    def _initialize_stats(self, seq_length):
        """Initialize success/trial counters"""
        if self.seq_length is None:
            self.seq_length = seq_length
            
            # Initialize with uniform prior (1 success, 1 failure per position)
            # This prevents division by zero and gives equal initial scores
            self.successes = np.ones(seq_length)
            self.trials = np.ones(seq_length) * 2
            
            print(f"  Initialized UCB bandit (seq_length={seq_length}, c={self.ucb_c})")
    
    def _compute_ucb_scores(self):
        """
        Compute UCB scores for all positions
        
        Returns:
            np.array: UCB score for each position
        """
        # Mean reward (success probability)
        mean_reward = self.successes / self.trials
        
        # Confidence interval (exploration bonus)
        # Decreases as trials increase
        if self.total_trials > 0:
            confidence = self.ucb_c * np.sqrt(
                np.log(self.total_trials + 1) / self.trials
            )
        else:
            confidence = np.ones(self.seq_length) * self.ucb_c
        
        # UCB score = exploitation + exploration
        ucb_scores = mean_reward + confidence
        
        return ucb_scores
    
    def select_mutations(self, current_seq, current_fitness):
        """
        Select k mutations using UCB
        
        Args:
            current_seq: Current sequence
            current_fitness: Current fitness (not used in UCB, but kept for API consistency)
        
        Returns:
            List of (position, from_aa, to_aa) tuples
        """
        from src.utils.mutations import AMINO_ACIDS
        
        # Initialize if first call
        self._initialize_stats(len(current_seq))
        
        # Compute UCB scores for all positions
        ucb_scores = self._compute_ucb_scores()
        
        # Select top k positions by UCB score
        # Use argsort to get indices sorted by score (descending)
        sorted_indices = np.argsort(ucb_scores)[::-1]
        
        # Take top k positions
        selected_positions = sorted_indices[:self.k].tolist()
        
        # For tie-breaking, shuffle positions with equal scores
        # This adds some randomness while maintaining UCB selection
        top_k_scores = ucb_scores[selected_positions]
        unique_scores = np.unique(top_k_scores)
        
        if len(unique_scores) < self.k:
            # There are ties - shuffle tied positions
            self.rng.shuffle(selected_positions)
        
        # Create mutations for selected positions
        mutations = []
        for pos in selected_positions:
            wt_aa = current_seq[pos]
            
            # Select random amino acid (exclude wild-type)
            possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            to_aa = self.rng.choice(possible_aas)
            
            mutations.append((pos, wt_aa, to_aa))
        
        return mutations
    
    def update(self, mutations, improved):
        """
        Update UCB statistics based on outcome
        
        Args:
            mutations: List of (position, from_aa, to_aa) mutations
            improved: Boolean - did fitness improve?
        """
        self.total_trials += 1
        
        for mutation in mutations:
            pos = mutation[0]
            
            # Update trial count
            self.trials[pos] += 1
            
            # Update success count if improved
            if improved:
                self.successes[pos] += 1
    
    def get_position_stats(self):
        """
        Get current statistics for all positions
        
        Returns:
            Dict with UCB scores, mean rewards, and confidence intervals
        """
        ucb_scores = self._compute_ucb_scores()
        mean_rewards = self.successes / self.trials
        
        if self.total_trials > 0:
            confidence = self.ucb_c * np.sqrt(
                np.log(self.total_trials + 1) / self.trials
            )
        else:
            confidence = np.ones(self.seq_length) * self.ucb_c
        
        return {
            'ucb_scores': ucb_scores,
            'mean_rewards': mean_rewards,
            'confidence': confidence,
            'trials': self.trials,
            'successes': self.successes
        }
    
    def optimize(self, wt_sequence, budget=500):
        """
        Run UCB bandit optimization
        
        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries
        
        Returns:
            Dict with results
        """
        from src.utils.mutations import apply_mutations
        
        print(f"UCB Bandit (k={self.k}, budget={budget}, c={self.ucb_c})")
        print("  Method: Upper Confidence Bound (deterministic, theoretically grounded)")
        print("-" * 70)
        
        # Initialize
        self._initialize_stats(len(wt_sequence))
        
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
            # Select k mutations using UCB
            mutations = self.select_mutations(current_seq, current_fitness)
            
            # Apply mutations
            mutant_seq = apply_mutations(current_seq, mutations)
            
            # Create mutation description
            mutation_strs = [f"{m[1]}{m[0]+1}{m[2]}" for m in mutations]
            mutation_desc = ":".join(mutation_strs)
            
            # Score mutant
            mutant_fitness = self.oracle.score_sequence(mutant_seq)
            queries_used += 1
            
            history.append((mutant_seq, mutant_fitness, mutation_desc))
            
            # Check if improved
            improved = mutant_fitness > current_fitness
            
            # Update UCB statistics
            self.update(mutations, improved)
            
            # Accept if improved (greedy)
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
                stats = self.get_position_stats()
                avg_ucb = stats['ucb_scores'].mean()
                avg_confidence = stats['confidence'].mean()
                
                print(
                    f"  {i+1}/{budget-1}: Best={best_fitness:.4f}, "
                    f"Improvements={improvements}, "
                    f"Avg UCB={avg_ucb:.3f}, Avg Conf={avg_confidence:.3f}"
                )
        
        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Improvements: {improvements}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")
        
        # Get final statistics
        final_stats = self.get_position_stats()
        
        return {
            "method": "UCB_Bandit",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
            "improvements": improvements,
            "ucb_stats": final_stats,  # Additional diagnostic info
        }


class ContextualBanditUCB1(ContextualBanditUCB):
    """
    UCB1 variant - simplified version with c=sqrt(2)
    
    This is the classic UCB1 algorithm from Auer et al. (2002)
    """
    
    def __init__(self, oracle, k=1, seed=42):
        super().__init__(oracle, k, seed, ucb_c=np.sqrt(2))
        print("  Using UCB1 variant (c=sqrt(2))")


class ContextualBanditUCBTuned(ContextualBanditUCB):
    """
    UCB-Tuned variant - uses variance estimates for tighter bounds
    
    More sophisticated than UCB1, adapts exploration bonus based on
    observed variance in rewards.
    """
    
    def __init__(self, oracle, k=1, seed=42):
        super().__init__(oracle, k, seed, ucb_c=1.0)
        
        # Additional statistics for variance estimation
        self.squared_rewards = None
    
    def _initialize_stats(self, seq_length):
        """Initialize with variance tracking"""
        super()._initialize_stats(seq_length)
        
        if self.squared_rewards is None:
            self.squared_rewards = np.ones(seq_length) * 0.25  # Initial variance estimate
    
    def _compute_ucb_scores(self):
        """
        Compute UCB-Tuned scores using variance estimates
        
        UCB-Tuned: score = mean + sqrt( (ln(t) / n) * min(1/4, V(n)) )
        where V(n) is the sample variance
        """
        mean_reward = self.successes / self.trials
        
        # Compute sample variance
        variance = (self.squared_rewards / self.trials) - (mean_reward ** 2)
        variance = np.maximum(variance, 0)  # Ensure non-negative
        
        # Upper bound on variance (for bounded rewards [0,1])
        variance_bound = np.minimum(0.25, variance + np.sqrt(2 * np.log(self.total_trials + 1) / self.trials))
        
        # UCB-Tuned exploration bonus
        if self.total_trials > 0:
            confidence = np.sqrt(
                (np.log(self.total_trials + 1) / self.trials) * variance_bound
            )
        else:
            confidence = np.ones(self.seq_length) * 0.5
        
        ucb_scores = mean_reward + confidence
        
        return ucb_scores
    
    def update(self, mutations, improved):
        """Update statistics including variance estimates"""
        super().update(mutations, improved)
        
        reward = 1.0 if improved else 0.0
        
        for mutation in mutations:
            pos = mutation[0]
            self.squared_rewards[pos] += reward ** 2
