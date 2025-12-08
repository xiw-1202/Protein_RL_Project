"""
Validate ESM-2 as oracle by computing correlation with experimental fitness
FIXED: Properly handles mutation application
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
from esm import pretrained


class ESM2Oracle:
    """ESM-2 oracle for scoring protein sequences"""

    def __init__(self, model_name="esm2_t33_650M_UR50D"):
        """
        Initialize ESM-2

        Model options:
        - esm2_t12_35M_UR50D (fast, for quick testing)
        - esm2_t33_650M_UR50D (recommended - better accuracy)
        """

        print(f"Loading ESM-2 model: {model_name}")

        if torch.backends.mps.is_available():
            self.device = torch.device("mps")  # Use Apple GPU!
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        self.model, self.alphabet = pretrained.load_model_and_alphabet(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.batch_converter = self.alphabet.get_batch_converter()
        self.mask_idx = self.alphabet.mask_idx

        print("✓ Model loaded\n")

    def score_sequence(self, sequence):
        """
        Score a sequence using pseudo-log-likelihood (as per proposal)
        Higher score = better predicted fitness
        """

        data = [("protein", sequence)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        log_probs = []

        with torch.no_grad():
            for i in range(1, len(sequence) + 1):
                # Mask position i
                masked_tokens = batch_tokens.clone()
                masked_tokens[0, i] = self.mask_idx

                # Get prediction
                results = self.model(masked_tokens, repr_layers=[])
                logits = results["logits"]

                # Log prob of true amino acid
                true_token_idx = batch_tokens[0, i]
                log_prob = torch.log_softmax(logits[0, i], dim=-1)[true_token_idx]
                log_probs.append(log_prob.item())

        return sum(log_probs)


def apply_mutations(wt_seq, mutation_str):
    """
    Apply mutations to wild-type sequence

    Args:
        wt_seq: Wild-type sequence
        mutation_str: Mutation string (e.g., "A2V" or "A2V:K5R")

    Returns:
        Mutated sequence
    """

    if mutation_str in ["WT", "wt", "", "wild_type"]:
        return wt_seq

    # Handle multiple mutations separated by ':'
    mutations = mutation_str.split(":")

    seq = list(wt_seq)

    for mut in mutations:
        mut = mut.strip()

        if len(mut) < 3:
            continue

        try:
            # Parse mutation: A2V means position 2 (1-indexed), A->V
            wt_aa = mut[0]
            position = int(mut[1:-1]) - 1  # Convert to 0-indexed
            new_aa = mut[-1]

            # Verify wild-type matches
            if position < len(seq) and seq[position] == wt_aa:
                seq[position] = new_aa
            else:
                # Mismatch - skip this mutation
                continue
        except:
            continue

    return "".join(seq)


def validate_dataset(dms_id, oracle, n_samples=500):
    """Validate oracle on one dataset"""

    print(f"Validating: {dms_id}")
    print("-" * 70)

    # Load DMS data
    dms_path = Path(f"data/raw/dms_datasets/{dms_id}.csv")
    df = pd.read_csv(dms_path)

    # Load wild-type
    wt_path = Path(f"data/raw/wild_types/{dms_id}.fasta")
    with open(wt_path, "r") as f:
        lines = f.readlines()
        wt_seq = "".join([line.strip() for line in lines if not line.startswith(">")])

    print(f"Dataset size: {len(df)} variants")
    print(f"Wild-type length: {len(wt_seq)} AA")

    # Check required columns
    if "mutant" not in df.columns or "DMS_score" not in df.columns:
        print("✗ Missing required columns (mutant, DMS_score)")
        return None

    # Sample if too large
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
        print(f"Sampling {n_samples} variants for speed")

    # Score all variants
    print("Scoring variants with ESM-2...")

    esm_scores = []
    dms_scores = []
    valid_count = 0

    for idx, row in df.iterrows():
        try:
            # Get sequence
            if "mutated_sequence" in df.columns and pd.notna(row["mutated_sequence"]):
                seq = row["mutated_sequence"]
            else:
                # Apply mutations from 'mutant' column
                seq = apply_mutations(wt_seq, row["mutant"])

            # Skip if invalid
            if len(seq) == 0 or len(seq) != len(wt_seq):
                continue

            # Score with ESM-2
            esm_score = oracle.score_sequence(seq)
            dms_score = row["DMS_score"]

            if pd.notna(dms_score):
                esm_scores.append(esm_score)
                dms_scores.append(dms_score)
                valid_count += 1

        except Exception as e:
            # Skip problematic variants
            continue

        if (valid_count) % 50 == 0 and valid_count > 0:
            print(f"  Progress: {valid_count}/{len(df)}")

    print(f"  Scored {valid_count} valid variants")

    if len(esm_scores) < 10:
        print("✗ Too few valid variants")
        return None

    # Compute correlations
    esm_scores = np.array(esm_scores)
    dms_scores = np.array(dms_scores)

    spearman_rho, spearman_p = spearmanr(esm_scores, dms_scores)
    pearson_r, pearson_p = pearsonr(esm_scores, dms_scores)

    # Determine quality tier (as per proposal)
    if spearman_rho >= 0.6:
        tier = "HIGH"
    elif spearman_rho >= 0.4:
        tier = "MEDIUM"
    else:
        tier = "LOW"

    print(f"\nResults:")
    print(f"  Spearman ρ:  {spearman_rho:.4f} (p={spearman_p:.2e})")
    print(f"  Pearson r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Oracle Tier: {tier}")
    print()

    return {
        "dms_id": dms_id,
        "n_variants": valid_count,
        "spearman_rho": spearman_rho,
        "pearson_r": pearson_r,
        "oracle_tier": tier,
        "esm_scores": esm_scores,
        "dms_scores": dms_scores,
    }


def plot_correlation(results, output_dir):
    """Plot ESM-2 vs DMS scores"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        dms_id = result["dms_id"]
        esm_scores = result["esm_scores"]
        dms_scores = result["dms_scores"]
        rho = result["spearman_rho"]
        tier = result["oracle_tier"]

        plt.figure(figsize=(8, 6))

        plt.scatter(esm_scores, dms_scores, alpha=0.5, s=10)

        # Add trend line
        z = np.polyfit(esm_scores, dms_scores, 1)
        p = np.poly1d(z)
        plt.plot(esm_scores, p(esm_scores), "r--", alpha=0.8, linewidth=2)

        plt.xlabel("ESM-2 Score (Pseudo-Log-Likelihood)", fontsize=12)
        plt.ylabel("DMS Score (Experimental Fitness)", fontsize=12)
        plt.title(
            f"{dms_id}\nSpearman ρ = {rho:.3f} ({tier} Quality)",
            fontsize=14,
            fontweight="bold",
        )

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(output_dir / f"{dms_id}_correlation.png", dpi=150)
        plt.close()


def main():
    """Validate oracle on all datasets"""

    print("=" * 70)
    print("ORACLE VALIDATION: ESM-2 vs EXPERIMENTAL FITNESS")
    print("=" * 70)
    print("\nAs per proposal:")
    print("  - High quality:   Spearman ρ ≥ 0.6")
    print("  - Medium quality: 0.4 ≤ ρ < 0.6")
    print("  - Low quality:    ρ < 0.4")
    print("\nUsing ESM-2 pseudo-log-likelihood as fitness oracle")
    print("\nNote: This may take 10-30 minutes depending on hardware")
    print("=" * 70)
    print()

    # Find datasets
    data_dir = Path("data/raw/dms_datasets")
    csv_files = [f for f in data_dir.glob("*.csv") if not f.name.startswith("_")]

    if len(csv_files) == 0:
        print("✗ No datasets found")
        return

    print(f"✓ Found {len(csv_files)} dataset(s)\n")

    # Initialize oracle
    # For testing: use esm2_t12_35M_UR50D
    # For final: use esm2_t33_650M_UR50D (matches proposal)
    print("Model choice:")
    print("  1. esm2_t12_35M_UR50D (fast, for testing)")
    print("  2. esm2_t33_650M_UR50D (recommended, better accuracy)")

    choice = input("\nEnter choice (1 or 2, or press Enter for default=1): ").strip()

    if choice == "2":
        model_name = "esm2_t33_650M_UR50D"
    else:
        model_name = "esm2_t12_35M_UR50D"

    print()
    oracle = ESM2Oracle(model_name=model_name)

    # Validate each dataset
    results = []

    for i, csv_file in enumerate(sorted(csv_files), 1):
        dms_id = csv_file.stem

        print(f"\n[{i}/{len(csv_files)}] " + "=" * 50)

        result = validate_dataset(dms_id, oracle, n_samples=500)

        if result is not None:
            results.append(result)

    if len(results) == 0:
        print("\n✗ No valid results")
        return

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    # Group by tier
    high = [r for r in results if r["oracle_tier"] == "HIGH"]
    medium = [r for r in results if r["oracle_tier"] == "MEDIUM"]
    low = [r for r in results if r["oracle_tier"] == "LOW"]

    print(f"\nOracle Quality Distribution:")
    print(f"  HIGH (ρ ≥ 0.6):     {len(high)} datasets")
    print(f"  MEDIUM (0.4-0.6):   {len(medium)} datasets")
    print(f"  LOW (< 0.4):        {len(low)} datasets")

    print("\nDetailed Results:")
    for result in results:
        tier = result["oracle_tier"]
        rho = result["spearman_rho"]
        dms_id = result["dms_id"]
        print(f"  [{tier:6s}] {dms_id:45s} ρ = {rho:.3f}")

    # Check if we have good distribution
    print("\n" + "=" * 70)
    if len(high) >= 2 and len(medium) >= 2 and len(low) >= 2:
        print("✓ EXCELLENT: Good distribution across all quality tiers!")
    elif len(high) >= 1 and len(medium) >= 1 and len(low) >= 1:
        print("✓ GOOD: Have at least one dataset per tier")
    else:
        print("⚠ WARNING: Uneven distribution across tiers")
        print("  Consider re-running Step 1 to select different datasets")
    print("=" * 70)

    # Save results
    output_dir = Path("experiments/oracle_validation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics
    summary_df = pd.DataFrame(
        [
            {
                "dms_id": r["dms_id"],
                "n_variants": r["n_variants"],
                "spearman_rho": r["spearman_rho"],
                "pearson_r": r["pearson_r"],
                "oracle_tier": r["oracle_tier"],
            }
            for r in results
        ]
    )

    summary_path = output_dir / "oracle_validation_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n✓ Summary saved: {summary_path}")

    # Plot correlations
    print("\nGenerating correlation plots...")
    plot_correlation(results, output_dir / "plots")
    print(f"✓ Plots saved: {output_dir}/plots/")

    print("\n" + "=" * 70)
    print("✓✓✓ ORACLE VALIDATION COMPLETE! ✓✓✓")
    print("=" * 70)

    print("\n✓ Validated oracle quality for all datasets")
    print("✓ This matches the proposal's experimental setup")
    print("\nNext steps (as per proposal):")
    print("  1. Implement baselines:")
    print("     - Random sampling")
    print("     - Greedy hill-climbing")
    print("     - Simulated annealing")
    print("  2. Implement RL methods:")
    print("     - Contextual Bandits (primary)")
    print("     - PPO (secondary)")
    print("  3. Run experiments with fixed budget (500 queries)")


if __name__ == "__main__":
    main()
