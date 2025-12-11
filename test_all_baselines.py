"""
Test all three baselines on one dataset
"""

import sys

sys.path.append("src")

from models.esm_oracle import ESM2Oracle
from baselines.random_baseline import RandomBaseline
from baselines.greedy_baseline import GreedyBaseline
from baselines.simulated_annealing import SimulatedAnnealingBaseline
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def load_wild_type(dms_id):
    """Load wild-type sequence from FASTA"""
    fasta_path = Path(f"data/raw/wild_types/{dms_id}.fasta")

    with open(fasta_path, "r") as f:
        lines = f.readlines()
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])

    return sequence


def main():
    print("=" * 70)
    print("TESTING ALL BASELINES")
    print("=" * 70)

    # Use smallest dataset for quick test
    dms_id = "SAV1_MOUSE_Tsuboyama_2023_2YSB"
    budget = 100  # Small budget for testing
    k = 1

    print(f"\nDataset: {dms_id}")
    print(f"Budget: {budget} queries")
    print(f"k: {k} mutations\n")

    # Load wild-type
    wt_seq = load_wild_type(dms_id)
    print(f"Wild-type: {len(wt_seq)} AA\n")

    # Initialize oracle
    oracle = ESM2Oracle(model_name="esm2_t12_35M_UR50D", device="auto")

    # Run all baselines
    results = {}

    print("\n" + "=" * 70)
    results["Random"] = RandomBaseline(oracle, k=k, seed=42).optimize(wt_seq, budget)

    print("\n" + "=" * 70)
    oracle.clear_cache()  # Clear cache between methods
    results["Greedy"] = GreedyBaseline(oracle, k=k).optimize(wt_seq, budget)

    print("\n" + "=" * 70)
    oracle.clear_cache()
    results["SA"] = SimulatedAnnealingBaseline(oracle, k=k, seed=42).optimize(
        wt_seq, budget
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for method, res in results.items():
        print(f"\n{method}:")
        print(f"  Best fitness: {res['best_fitness']:.4f}")
        print(f"  Improvement:  {res['improvement']:.4f}")
        print(f"  Queries used: {res['queries_used']}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Best-so-far curves
    ax = axes[0]
    for method, res in results.items():
        fitnesses = [h[1] for h in res["history"]]
        best_so_far = np.maximum.accumulate(fitnesses)
        ax.plot(best_so_far, label=method, linewidth=2)

    ax.axhline(
        results["Random"]["wt_fitness"],
        color="red",
        linestyle="--",
        alpha=0.5,
        label="Wild-type",
    )
    ax.set_xlabel("Query")
    ax.set_ylabel("Best Fitness Found")
    ax.set_title("Sample Efficiency Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Right: Final improvements
    ax = axes[1]
    methods = list(results.keys())
    improvements = [results[m]["improvement"] for m in methods]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    bars = ax.bar(methods, improvements, color=colors, alpha=0.7)
    ax.set_ylabel("Fitness Improvement")
    ax.set_title("Final Improvement Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{imp:.2f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("baseline_comparison.png", dpi=150)
    print("\n✓ Plot saved: baseline_comparison.png")


if __name__ == "__main__":
    main()
