"""
ESM-2 Oracle for protein fitness prediction
Uses pseudo-log-likelihood scoring
"""

import torch
import numpy as np
from esm import pretrained


class ESM2Oracle:
    """
    ESM-2 oracle for scoring protein sequences

    Uses pseudo-log-likelihood: mask each position, predict AA, sum log probs
    Higher score = better predicted fitness
    """

    def __init__(
        self, model_name="esm2_t12_35M_UR50D", device="auto", cache_scores=True
    ):
        """
        Initialize ESM-2 oracle

        Args:
            model_name: ESM-2 model variant
                - esm2_t12_35M_UR50D: Fast (recommended)
                - esm2_t33_650M_UR50D: More accurate but slower
            device: 'auto', 'mps', 'cuda', or 'cpu'
            cache_scores: Cache sequence scores to avoid recomputation
        """
        print(f"Initializing ESM-2 Oracle...")
        print(f"  Model: {model_name}")

        # Device selection
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        print(f"  Device: {self.device}")

        # Load model
        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.mask_idx = self.alphabet.mask_idx

        # Score cache
        self.cache_scores = cache_scores
        self.score_cache = {}
        self.query_count = 0

        print(f"✓ Oracle ready\n")

    def score_sequence(self, sequence):
        """
        Score a single sequence using pseudo-log-likelihood

        Args:
            sequence: Protein sequence (string of amino acids)

        Returns:
            float: Fitness score (higher = better)
        """
        # Check cache
        if self.cache_scores and sequence in self.score_cache:
            return self.score_cache[sequence]

        # Increment query count
        self.query_count += 1

        # Prepare data
        data = [("protein", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)

        log_probs = []

        with torch.no_grad():
            # Mask each position and predict
            for i in range(1, len(sequence) + 1):  # 1-indexed (skip BOS token)
                # Create masked version
                masked_tokens = tokens.clone()
                masked_tokens[0, i] = self.mask_idx

                # Get predictions
                results = self.model(masked_tokens, repr_layers=[])
                logits = results["logits"]

                # Get log prob of true amino acid
                true_token_idx = tokens[0, i]
                log_prob = torch.log_softmax(logits[0, i], dim=-1)[true_token_idx]
                log_probs.append(log_prob.item())

        # Sum log probabilities
        score = sum(log_probs)

        # Cache result
        if self.cache_scores:
            self.score_cache[sequence] = score

        return score

    def score_batch(self, sequences):
        """
        Score multiple sequences (more efficient than one-by-one)

        Args:
            sequences: List of protein sequences

        Returns:
            np.ndarray: Fitness scores
        """
        scores = []
        for seq in sequences:
            scores.append(self.score_sequence(seq))
        return np.array(scores)

    def get_query_count(self):
        """Get number of oracle queries made"""
        return self.query_count

    def reset_query_count(self):
        """Reset query counter"""
        self.query_count = 0

    def clear_cache(self):
        """Clear score cache"""
        self.score_cache.clear()
