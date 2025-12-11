"""
Mutation utilities for protein sequence optimization
"""

import numpy as np
from typing import List, Tuple
import itertools


# Standard amino acids
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def parse_mutation(mutation_str):
    """
    Parse mutation string into components

    Args:
        mutation_str: Format "A2V" or "A2V:K5R:D10E"

    Returns:
        List of (position, from_aa, to_aa) tuples (0-indexed positions)

    Example:
        parse_mutation("A2V:K5R") → [(1, 'A', 'V'), (4, 'K', 'R')]
    """
    if mutation_str in ["WT", "wt", "", "wild_type"]:
        return []

    mutations = []

    for mut in mutation_str.split(":"):
        mut = mut.strip()
        if len(mut) < 3:
            continue

        try:
            from_aa = mut[0]
            to_aa = mut[-1]
            position = int(mut[1:-1]) - 1  # Convert to 0-indexed

            mutations.append((position, from_aa, to_aa))
        except:
            continue

    return mutations


def apply_mutations(sequence, mutations):
    """
    Apply mutations to a sequence

    Args:
        sequence: Wild-type sequence (string)
        mutations: List of (position, from_aa, to_aa) or mutation string

    Returns:
        str: Mutated sequence

    Example:
        apply_mutations("MKTAY", [(1, 'K', 'A')]) → "MATAY"
        apply_mutations("MKTAY", "K2A") → "MATAY"
    """
    # Handle string input
    if isinstance(mutations, str):
        mutations = parse_mutation(mutations)

    # Convert to list for mutation
    seq_list = list(sequence)

    for position, from_aa, to_aa in mutations:
        # Validate position
        if position < 0 or position >= len(seq_list):
            raise ValueError(
                f"Position {position} out of bounds for sequence length {len(seq_list)}"
            )

        # Validate wild-type matches
        if seq_list[position] != from_aa:
            raise ValueError(
                f"Expected {from_aa} at position {position}, found {seq_list[position]}"
            )

        # Apply mutation
        seq_list[position] = to_aa

    return "".join(seq_list)


def get_single_mutants(sequence, positions=None):
    """
    Generate all single-point mutants

    Args:
        sequence: Wild-type sequence
        positions: Specific positions to mutate (None = all positions)

    Returns:
        List of (mutant_sequence, mutation_description) tuples

    Example:
        get_single_mutants("MAK") → [
            ("AAK", "M1A"), ("CAK", "M1C"), ...,
            ("MKK", "A2K"), ...
        ]
    """
    if positions is None:
        positions = range(len(sequence))

    mutants = []

    for pos in positions:
        wt_aa = sequence[pos]

        for new_aa in AMINO_ACIDS:
            if new_aa == wt_aa:
                continue  # Skip wild-type

            # Apply mutation
            mutant = list(sequence)
            mutant[pos] = new_aa
            mutant_seq = "".join(mutant)

            # Create mutation description (1-indexed)
            mutation_desc = f"{wt_aa}{pos+1}{new_aa}"

            mutants.append((mutant_seq, mutation_desc))

    return mutants


def get_random_mutant(sequence, k=1, rng=None):
    """
    Generate random k-point mutant

    Args:
        sequence: Wild-type sequence
        k: Number of simultaneous mutations
        rng: Random number generator (numpy.random.RandomState)

    Returns:
        Tuple of (mutant_sequence, mutation_description)

    Example:
        get_random_mutant("MKTAY", k=2) → ("MATVY", "K2A:A4V")
    """
    if rng is None:
        rng = np.random.RandomState()

    # Select k random positions
    positions = rng.choice(len(sequence), size=k, replace=False)

    mutations = []
    mutant = list(sequence)

    for pos in sorted(positions):
        wt_aa = sequence[pos]

        # Select random different amino acid
        possible_aas = [aa for aa in AMINO_ACIDS if aa != wt_aa]
        new_aa = rng.choice(possible_aas)

        mutant[pos] = new_aa
        mutations.append(f"{wt_aa}{pos+1}{new_aa}")

    mutant_seq = "".join(mutant)
    mutation_desc = ":".join(mutations)

    return mutant_seq, mutation_desc


def get_k_mutants(sequence, k, max_samples=None, rng=None):
    """
    Generate k-point mutants (all or sampled)

    Args:
        sequence: Wild-type sequence
        k: Number of simultaneous mutations
        max_samples: If not None, randomly sample this many mutants
        rng: Random number generator

    Returns:
        List of (mutant_sequence, mutation_description) tuples

    Note:
        For k=1, generates all 19*L mutants (~5K for 300 AA protein)
        For k=3, would be C(L,3) * 19^3 (~billions) - must sample!
    """
    L = len(sequence)

    # Estimate total number of mutants
    from math import comb

    total_mutants = comb(L, k) * (19**k)

    # If total is reasonable and no sampling requested, generate all
    if max_samples is None and total_mutants < 10000:
        return _generate_all_k_mutants(sequence, k)

    # Otherwise, sample random mutants
    if rng is None:
        rng = np.random.RandomState()

    n_samples = max_samples if max_samples is not None else min(10000, total_mutants)

    mutants = []
    seen = set()

    while len(mutants) < n_samples:
        mutant_seq, mutation_desc = get_random_mutant(sequence, k, rng)

        if mutant_seq not in seen:
            seen.add(mutant_seq)
            mutants.append((mutant_seq, mutation_desc))

    return mutants


def _generate_all_k_mutants(sequence, k):
    """Generate all possible k-point mutants (use with caution!)"""
    L = len(sequence)
    mutants = []

    # All combinations of k positions
    for positions in itertools.combinations(range(L), k):
        # All combinations of amino acids for these positions
        wt_aas = [sequence[pos] for pos in positions]

        # For each position, all amino acids except WT
        aa_choices = []
        for pos in positions:
            wt_aa = sequence[pos]
            choices = [aa for aa in AMINO_ACIDS if aa != wt_aa]
            aa_choices.append(choices)

        # Generate all combinations
        for new_aas in itertools.product(*aa_choices):
            mutant = list(sequence)
            mutations = []

            for pos, new_aa in zip(positions, new_aas):
                wt_aa = sequence[pos]
                mutant[pos] = new_aa
                mutations.append(f"{wt_aa}{pos+1}{new_aa}")

            mutant_seq = "".join(mutant)
            mutation_desc = ":".join(mutations)
            mutants.append((mutant_seq, mutation_desc))

    return mutants
