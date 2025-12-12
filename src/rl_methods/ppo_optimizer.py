"""
PPO (Proximal Policy Optimization) for protein sequence optimization
Uses policy and value networks to learn optimal mutation strategies
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class PolicyValueNetwork(nn.Module):
    """
    Combined policy and value network

    Input: One-hot encoded sequence
    Output:
        - Policy: probability distribution over mutations
        - Value: estimated fitness value
    """

    def __init__(self, seq_length, hidden_dim=128):
        super().__init__()

        # Input: one-hot encoded sequence (seq_length × 20 amino acids)
        input_dim = seq_length * 20

        # Shared feature extraction
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Policy head: outputs logits for each position
        self.policy = nn.Linear(hidden_dim, seq_length)

        # Value head: outputs estimated fitness
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass

        Args:
            x: One-hot encoded sequence [batch, seq_length * 20]

        Returns:
            policy_logits: [batch, seq_length]
            value: [batch, 1]
        """
        features = self.shared(x)
        policy_logits = self.policy(features)
        value = self.value(features)

        return policy_logits, value


class PPOOptimizer:
    """
    PPO for protein optimization

    Learns a policy network that selects which positions to mutate
    Uses advantage estimation to update policy
    """

    def __init__(
        self,
        oracle,
        k=1,
        seed=42,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        update_epochs=4,
        batch_size=32,
    ):
        """
        Initialize PPO optimizer

        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            seed: Random seed
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            update_epochs: Number of optimization epochs per update
            batch_size: Batch size for updates
        """
        self.oracle = oracle
        self.k = k

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.rng = np.random.RandomState(seed)

        # PPO hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size

        # Will be initialized when we know sequence length
        self.network = None
        self.optimizer = None
        self.lr = lr

        # Experience buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []

    def _encode_sequence(self, sequence):
        """
        One-hot encode protein sequence

        Args:
            sequence: Protein sequence string

        Returns:
            torch.Tensor: One-hot encoded [seq_length * 20]
        """
        from src.utils.mutations import AMINO_ACIDS

        aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

        # One-hot encoding
        encoding = np.zeros((len(sequence), 20))
        for i, aa in enumerate(sequence):
            if aa in aa_to_idx:
                encoding[i, aa_to_idx[aa]] = 1.0

        # Flatten
        encoding = encoding.flatten()

        return torch.FloatTensor(encoding)

    def _initialize_network(self, seq_length):
        """Initialize policy-value network"""
        if self.network is None:
            self.network = PolicyValueNetwork(seq_length, hidden_dim=128)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            print(f"  Initialized PPO network (seq_length={seq_length})")

    def select_actions(self, sequence):
        """
        Select k mutation positions using policy network

        Args:
            sequence: Current sequence

        Returns:
            mutations: List of (position, from_aa, to_aa)
            log_probs: List of log probabilities for each position
            value: Estimated value
        """
        from src.utils.mutations import AMINO_ACIDS

        # Encode sequence
        state = self._encode_sequence(sequence).unsqueeze(0)

        # Get policy and value
        with torch.no_grad():
            policy_logits, value = self.network(state)

        # Sample k positions from policy (without replacement)
        policy_probs = torch.softmax(policy_logits, dim=-1).squeeze()

        # Sample k positions without replacement
        selected_positions = []
        log_probs_list = []

        for _ in range(self.k):
            # Create mask for already-selected positions (avoid in-place ops)
            mask = torch.ones_like(policy_probs)
            for pos in selected_positions:
                mask[pos] = 0.0
            
            # Apply mask and renormalize
            masked_probs = policy_probs * mask
            masked_probs = masked_probs / masked_probs.sum()

            # Sample position
            position_dist = torch.distributions.Categorical(masked_probs)
            position = position_dist.sample().item()
            log_prob = position_dist.log_prob(torch.tensor(position)).item()

            selected_positions.append(position)
            log_probs_list.append(log_prob)

        # Create mutations for selected positions
        mutations = []
        for position in selected_positions:
            wt_aa = sequence[position]
            possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            to_aa = self.rng.choice(possible_aas)
            mutations.append((position, wt_aa, to_aa))

        # Sum log probs (since positions are independent)
        total_log_prob = sum(log_probs_list)

        return mutations, total_log_prob, value.item()

    def store_transition(self, state, actions, reward, log_prob, value):
        """Store experience"""
        self.states.append(state)
        self.actions.append(actions)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def compute_returns(self):
        """Compute discounted returns"""
        returns = []
        R = 0

        for reward in reversed(self.rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)

        return torch.FloatTensor(returns)

    def update_policy(self):
        """Update policy using PPO"""
        if len(self.states) < self.batch_size:
            return  # Not enough data

        # Compute returns and advantages
        returns = self.compute_returns()
        values = torch.FloatTensor(self.values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Prepare data
        states = torch.stack([self._encode_sequence(s) for s in self.states])
        old_log_probs = torch.FloatTensor(self.log_probs)

        # Get positions from actions (first element of each mutation tuple)
        all_positions = []
        for actions in self.actions:
            positions = [a[0] for a in actions]
            all_positions.append(positions)

        # PPO update
        for _ in range(self.update_epochs):
            # Forward pass
            policy_logits, values_pred = self.network(states)

            # Reconstruct log probs for selected actions
            new_log_probs_list = []
            for i, positions in enumerate(all_positions):
                policy_probs = torch.softmax(policy_logits[i], dim=-1)

                # Compute log prob for this sequence of k positions
                total_log_prob = 0.0

                for j, pos in enumerate(positions):
                    # Create mask for already-selected positions (avoid in-place ops)
                    mask = torch.ones_like(policy_probs)
                    for prev_idx in range(j):
                        mask[positions[prev_idx]] = 0.0
                    
                    # Apply mask and renormalize
                    masked_probs = policy_probs * mask
                    masked_probs_norm = masked_probs / masked_probs.sum()

                    # Get log prob for this position
                    position_dist = torch.distributions.Categorical(masked_probs_norm)
                    total_log_prob += position_dist.log_prob(torch.tensor(pos))

                new_log_probs_list.append(total_log_prob)

            new_log_probs = torch.stack(new_log_probs_list)

            # PPO objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()

            # Value loss
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns)

            # Total loss
            loss = policy_loss + 0.5 * value_loss

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()

    def optimize(self, wt_sequence, budget=500):
        """
        Run PPO optimization

        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries

        Returns:
            Dict with results
        """
        from src.utils.mutations import apply_mutations

        print(f"PPO Optimizer (k={self.k}, budget={budget})")
        print("-" * 70)

        # Initialize network
        self._initialize_network(len(wt_sequence))

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
        updates = 0

        # Optimization loop
        for i in range(budget - 1):
            # Select k actions
            mutations, log_prob, value = self.select_actions(current_seq)

            # Apply mutations
            mutant_seq = apply_mutations(current_seq, mutations)

            # Create mutation description
            mutation_strs = [f"{m[1]}{m[0]+1}{m[2]}" for m in mutations]
            mutation_desc = ":".join(mutation_strs)

            # Score mutant
            mutant_fitness = self.oracle.score_sequence(mutant_seq)
            queries_used += 1

            history.append((mutant_seq, mutant_fitness, mutation_desc))

            # Compute reward (fitness improvement)
            reward = mutant_fitness - current_fitness

            # Store transition
            self.store_transition(current_seq, mutations, reward, log_prob, value)

            # Accept if improved
            improved = mutant_fitness > current_fitness
            if improved:
                improvements += 1
                current_seq = mutant_seq
                current_fitness = mutant_fitness

            # Update global best
            if current_fitness > best_fitness:
                best_fitness = current_fitness
                best_seq = current_seq

            # Update policy periodically
            if (i + 1) % self.batch_size == 0:
                self.update_policy()
                updates += 1

            # Progress
            if (i + 1) % 100 == 0:
                print(
                    f"  {i+1}/{budget-1}: Best={best_fitness:.4f}, Improvements={improvements}, Updates={updates}"
                )

        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Improvements: {improvements}")
        print(f"  Policy updates: {updates}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")

        return {
            "method": "PPO",
            "k": self.k,
            "best_sequence": best_seq,
            "best_fitness": best_fitness,
            "wt_fitness": history[0][1],
            "improvement": best_fitness - history[0][1],
            "history": history,
            "queries_used": queries_used,
            "improvements": improvements,
            "updates": updates,
        }
