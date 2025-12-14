"""
PPO v2 (Proximal Policy Optimization) - IMPROVED VERSION
Improvements over v1:
1. Uses ESM-2 embeddings instead of one-hot encoding (richer representations)
2. Adds entropy bonus for better exploration
3. Learns amino acid selection per position (not just positions)

FIXED ISSUES:
- AA policy now position-dependent (Critical Fix #1)
- Handles batch_converter properly (Critical Fix #2)
- Caches embeddings to avoid re-encoding (Performance Fix #3)
- Consistent AA masking in training and inference (Fix #4)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class PolicyValueNetwork(nn.Module):
    """
    Combined policy and value network using ESM-2 embeddings
    
    Input: ESM-2 sequence embeddings (seq_length × 1280)
    Output:
        - Policy: probability distribution over positions AND amino acids
        - Value: estimated fitness value
    """
    
    def __init__(self, seq_length, embed_dim=1280, hidden_dim=256):
        super().__init__()
        
        self.seq_length = seq_length
        
        # Shared feature extraction from ESM-2 embeddings
        self.shared = nn.Sequential(
            nn.Linear(seq_length * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        
        # Policy head for position selection
        self.policy_position = nn.Linear(hidden_dim, seq_length)
        
        # Policy head for amino acid selection (POSITION-DEPENDENT!)
        # Output: [seq_length * 20] then reshape to [seq_length, 20]
        self.policy_amino_acid = nn.Linear(hidden_dim, seq_length * 20)
        
        # Value head: outputs estimated fitness
        self.value = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: ESM-2 embeddings [batch, seq_length * embed_dim]
        
        Returns:
            position_logits: [batch, seq_length]
            aa_logits: [batch, seq_length, 20]
            value: [batch, 1]
        """
        features = self.shared(x)
        position_logits = self.policy_position(features)
        
        # Reshape AA logits to be position-dependent
        aa_logits = self.policy_amino_acid(features).view(-1, self.seq_length, 20)
        
        value = self.value(features)
        
        return position_logits, aa_logits, value


class PPOOptimizerV2:
    """
    PPO v2 for protein optimization - IMPROVED VERSION
    
    Improvements:
    1. Uses ESM-2 embeddings for state representation
    2. Adds entropy bonus for exploration
    3. Learns which amino acids to select (position-dependent)
    """
    
    def __init__(
        self,
        oracle,
        k=1,
        seed=42,
        lr=3e-4,
        gamma=0.99,
        epsilon=0.2,
        entropy_coef=0.01,
        update_epochs=4,
        batch_size=32,
    ):
        """
        Initialize PPO v2 optimizer
        
        Args:
            oracle: ESM2Oracle instance
            k: Number of simultaneous mutations
            seed: Random seed
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clipping parameter
            entropy_coef: Entropy bonus coefficient (0.01 recommended)
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
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Will be initialized when we know sequence length
        self.network = None
        self.optimizer = None
        self.lr = lr
        
        # Experience buffer - store embeddings AND sequences
        self.state_embeddings = []  # ESM-2 embeddings
        self.state_sequences = []   # Sequence strings (for AA masking)
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
        # Amino acid mappings (created once)
        from src.utils.mutations import AMINO_ACIDS
        self.aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}
        
        # Setup batch converter
        self._setup_batch_converter()
    
    def _setup_batch_converter(self):
        """Setup ESM-2 batch converter (Fix #2)"""
        if not hasattr(self.oracle, 'batch_converter'):
            # Try to create it from oracle's alphabet
            if hasattr(self.oracle, 'alphabet'):
                self.batch_converter = self.oracle.alphabet.get_batch_converter()
                print("  Created batch_converter from oracle.alphabet")
            else:
                # Load alphabet ourselves
                import esm
                model_name = getattr(self.oracle, 'model_name', 'esm2_t33_650M_UR50D')
                _, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
                self.batch_converter = alphabet.get_batch_converter()
                print(f"  Loaded batch_converter for {model_name}")
        else:
            self.batch_converter = self.oracle.batch_converter
            print("  Using oracle.batch_converter")
    
    def _encode_sequence(self, sequence):
        """
        Encode protein sequence using ESM-2 embeddings
        
        Args:
            sequence: Protein sequence string
        
        Returns:
            torch.Tensor: ESM-2 embeddings [seq_length * 1280]
        """
        # Get ESM-2 embeddings from oracle
        with torch.no_grad():
            # Prepare input
            data = [("protein", sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            
            # Move to same device as model
            batch_tokens = batch_tokens.to(next(self.oracle.model.parameters()).device)
            
            # Get representations from last layer
            results = self.oracle.model(
                batch_tokens, 
                repr_layers=[self.oracle.model.num_layers],
                return_contacts=False
            )
            
            # Extract embeddings (remove BOS/EOS tokens)
            embeddings = results["representations"][self.oracle.model.num_layers]
            embeddings = embeddings[0, 1:len(sequence)+1, :]  # [L, 1280]
            
            # Flatten to 1D
            embeddings = embeddings.flatten()  # [L * 1280]
        
        return embeddings.cpu()
    
    def _initialize_network(self, seq_length):
        """Initialize policy-value network"""
        if self.network is None:
            self.network = PolicyValueNetwork(seq_length, embed_dim=1280, hidden_dim=256)
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.lr)
            print(f"  Initialized PPO v2 network (seq_length={seq_length}, ESM-2 embeddings)")
    
    def select_actions(self, sequence):
        """
        Select k mutations using policy network
        Learns BOTH positions AND amino acids (position-dependent)
        
        Args:
            sequence: Current sequence
        
        Returns:
            mutations: List of (position, from_aa, to_aa)
            log_prob: Total log probability
            value: Estimated value
        """
        # Encode sequence with ESM-2
        state = self._encode_sequence(sequence).unsqueeze(0)
        
        # Get policy and value
        with torch.no_grad():
            position_logits, aa_logits, value = self.network(state)
        
        # Convert to probabilities
        position_probs = torch.softmax(position_logits, dim=-1).squeeze()
        # aa_logits is [1, seq_length, 20]
        
        # Sample k positions without replacement
        selected_positions = []
        selected_aas = []
        log_probs_list = []
        
        for _ in range(self.k):
            # Mask already-selected positions
            mask = torch.ones_like(position_probs)
            for pos in selected_positions:
                mask[pos] = 0.0
            
            masked_probs = position_probs * mask
            masked_probs = masked_probs / masked_probs.sum()
            
            # Sample position
            position_dist = torch.distributions.Categorical(masked_probs)
            position = position_dist.sample().item()
            pos_log_prob = position_dist.log_prob(
                torch.tensor(position, device=masked_probs.device)
            ).item()
            
            # Get AA logits for THIS specific position (Fix #1)
            aa_logits_for_pos = aa_logits[0, position, :]  # [20]
            aa_probs_for_pos = torch.softmax(aa_logits_for_pos, dim=-1)
            
            # Sample amino acid (exclude wild-type)
            wt_aa = sequence[position]
            wt_idx = self.aa_to_idx[wt_aa]
            
            aa_mask = torch.ones_like(aa_probs_for_pos)
            aa_mask[wt_idx] = 0.0  # Can't mutate to same AA
            
            masked_aa_probs = aa_probs_for_pos * aa_mask
            masked_aa_probs = masked_aa_probs / masked_aa_probs.sum()
            
            aa_dist = torch.distributions.Categorical(masked_aa_probs)
            aa_idx = aa_dist.sample().item()
            aa_log_prob = aa_dist.log_prob(
                torch.tensor(aa_idx, device=masked_aa_probs.device)
            ).item()
            
            selected_positions.append(position)
            selected_aas.append(aa_idx)
            log_probs_list.append(pos_log_prob + aa_log_prob)
        
        # Create mutations
        mutations = []
        for position, aa_idx in zip(selected_positions, selected_aas):
            wt_aa = sequence[position]
            to_aa = self.idx_to_aa[aa_idx]
            mutations.append((position, wt_aa, to_aa))
        
        total_log_prob = sum(log_probs_list)
        
        return mutations, total_log_prob, value.item()
    
    def store_transition(self, state_embedding, state_sequence, actions, reward, log_prob, value):
        """Store experience (Fix #3: cache embeddings)"""
        self.state_embeddings.append(state_embedding)
        self.state_sequences.append(state_sequence)
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
        """Update policy using PPO with entropy bonus"""
        if len(self.state_embeddings) < self.batch_size:
            return  # Not enough data
        
        # Compute returns and advantages
        returns = self.compute_returns()
        values = torch.FloatTensor(self.values)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Prepare data (Fix #3: use cached embeddings, no re-encoding!)
        states = torch.stack(self.state_embeddings)
        old_log_probs = torch.FloatTensor(self.log_probs)
        
        # Get actions
        all_actions = []
        for actions in self.actions:
            positions = [a[0] for a in actions]
            aas = [a[2] for a in actions]  # to_aa
            all_actions.append((positions, aas))
        
        # PPO update
        for _ in range(self.update_epochs):
            # Forward pass
            position_logits, aa_logits, values_pred = self.network(states)
            
            # Reconstruct log probs for selected actions
            new_log_probs_list = []
            
            for i, (positions, aas) in enumerate(all_actions):
                pos_probs = torch.softmax(position_logits[i], dim=-1)
                # aa_logits[i] is [seq_length, 20]
                
                total_log_prob = 0.0
                
                for j, (pos, to_aa) in enumerate(zip(positions, aas)):
                    # Position log prob with masking
                    mask = torch.ones_like(pos_probs)
                    for prev_idx in range(j):
                        mask[positions[prev_idx]] = 0.0
                    
                    masked_pos = pos_probs * mask
                    masked_pos = masked_pos / masked_pos.sum()
                    
                    pos_dist = torch.distributions.Categorical(masked_pos)
                    total_log_prob += pos_dist.log_prob(
                        torch.tensor(pos, device=masked_pos.device)
                    )
                    
                    # Amino acid log prob FOR THIS POSITION (Fix #1)
                    aa_logits_for_pos = aa_logits[i, pos, :]  # [20]
                    aa_probs_for_pos = torch.softmax(aa_logits_for_pos, dim=-1)
                    
                    # Apply wild-type masking (Fix #4)
                    wt_aa = self.state_sequences[i][pos]
                    wt_idx = self.aa_to_idx[wt_aa]
                    
                    aa_mask = torch.ones_like(aa_probs_for_pos)
                    aa_mask[wt_idx] = 0.0
                    
                    masked_aa_probs = aa_probs_for_pos * aa_mask
                    masked_aa_probs = masked_aa_probs / masked_aa_probs.sum()
                    
                    aa_dist = torch.distributions.Categorical(masked_aa_probs)
                    to_aa_idx = self.aa_to_idx[to_aa]
                    
                    total_log_prob += aa_dist.log_prob(
                        torch.tensor(to_aa_idx, device=masked_aa_probs.device)
                    )
                
                new_log_probs_list.append(total_log_prob)
            
            new_log_probs = torch.stack(new_log_probs_list)
            
            # Compute entropy (BOTH position and AA distributions)
            pos_probs_all = torch.softmax(position_logits, dim=-1)
            aa_probs_all = torch.softmax(aa_logits, dim=-1)
            
            pos_entropy = -(pos_probs_all * torch.log(pos_probs_all + 1e-10)).sum(dim=-1).mean()
            aa_entropy = -(aa_probs_all * torch.log(aa_probs_all + 1e-10)).sum(dim=-1).mean()
            total_entropy = pos_entropy + aa_entropy
            
            # PPO objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
            
            policy_loss = -torch.min(
                ratio * advantages, clipped_ratio * advantages
            ).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns)
            
            # Total loss WITH ENTROPY BONUS
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * total_entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear buffer
        self.state_embeddings.clear()
        self.state_sequences.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def optimize(self, wt_sequence, budget=500):
        """
        Run PPO v2 optimization
        
        Args:
            wt_sequence: Wild-type starting sequence
            budget: Number of oracle queries
        
        Returns:
            Dict with results
        """
        from src.utils.mutations import apply_mutations
        
        print(f"PPO v2 Optimizer (k={self.k}, budget={budget}, entropy_coef={self.entropy_coef})")
        print("  Improvements: ESM-2 embeddings + Entropy bonus + Position-dependent AA learning")
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
            
            # Cache embedding for this state (Fix #3)
            state_embedding = self._encode_sequence(current_seq)
            
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
            
            # Store transition (with cached embedding)
            self.store_transition(state_embedding, current_seq, mutations, reward, log_prob, value)
            
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
                    f"  {i+1}/{budget-1}: Best={best_fitness:.4f}, "
                    f"Improvements={improvements}, Updates={updates}"
                )
        
        print(f"\n✓ Complete!")
        print(f"  Queries used: {queries_used}")
        print(f"  Improvements: {improvements}")
        print(f"  Policy updates: {updates}")
        print(f"  Best fitness: {best_fitness:.4f}")
        print(f"  Improvement: {best_fitness - history[0][1]:.4f}")
        
        return {
            "method": "PPO_v2",
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
