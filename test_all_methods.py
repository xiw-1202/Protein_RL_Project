"""
Test ALL methods: 3 baselines + 2 RL
"""

import sys

sys.path.append("src")

from src.models.esm_oracle import ESM2Oracle
from src.baselines.random_baseline import RandomBaseline
from src.baselines.greedy_baseline import GreedyBaseline
from src.baselines.simulated_annealing import SimulatedAnnealingBaseline
from src.rl_methods.contextual_bandit import ContextualBandit
from src.rl_methods.ppo_optimizer import PPOOptimizer
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
    print("COMPLETE METHOD COMPARISON: 3 Baselines + 2 RL Methods")
    print("=" * 70)

    # Use smallest dataset
    dms_id = "SAV1_MOUSE_Tsuboyama_2023_2YSB"
    budget = 150
    k = 1

    print(f"\nDataset: {dms_id}")
    print(f"Budget: {budget} queries")
    print(f"k: {k} mutations\n")

    # Load wild-type
    wt_seq = load_wild_type(dms_id)
    print(f"Wild-type: {len(wt_seq)} AA\n")

    # Initialize oracle (small model for speed)
    oracle = ESM2Oracle(model_name="esm2_t12_35M_UR50D", device="auto")

    # Run all methods
    results = {}

    print("\n" + "=" * 70)
    print("BASELINES")
    print("=" * 70)

    results["Random"] = RandomBaseline(oracle, k=k, seed=42).optimize(wt_seq, budget)

    print("\n" + "=" * 70)
    oracle.clear_cache()
    results["Greedy"] = GreedyBaseline(oracle, k=k).optimize(wt_seq, budget)

    print("\n" + "=" * 70)
    oracle.clear_cache()
    results["SA"] = SimulatedAnnealingBaseline(oracle, k=k, seed=42).optimize(
        wt_seq, budget
    )

    print("\n" + "=" * 70)
    print("RL METHODS")
    print("=" * 70)

    oracle.clear_cache()
    results["Bandit"] = ContextualBandit(oracle, k=k, seed=42).optimize(wt_seq, budget)

    print("\n" + "=" * 70)
    oracle.clear_cache()
    results["PPO"] = PPOOptimizer(oracle, k=k, seed=42).optimize(wt_seq, budget)

    # Compare results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    for method, res in results.items():
        print(f"\n{method}:")
        print(f"  Best fitness: {res['best_fitness']:.4f}")
        print(f"  Improvement:  {res['improvement']:.4f}")
        print(f"  Queries used: {res['queries_used']}")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Sample efficiency curves
    ax = axes[0]
    colors = {
        "Random": "#1f77b4",
        "Greedy": "#ff7f0e",
        "SA": "#2ca02c",
        "Bandit": "#d62728",
        "PPO": "#9467bd",
    }

    for method, res in results.items():
        fitnesses = [h[1] for h in res["history"]]
        best_so_far = np.maximum.accumulate(fitnesses)
        ax.plot(best_so_far, label=method, linewidth=2, color=colors[method])

    ax.axhline(
        results["Random"]["wt_fitness"],
        color="gray",
        linestyle="--",
        alpha=0.5,
        label="Wild-type",
    )
    ax.set_xlabel("Query", fontsize=12)
    ax.set_ylabel("Best Fitness Found", fontsize=12)
    ax.set_title("Sample Efficiency: All Methods", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right: Final improvements
    ax = axes[1]
    methods = ["Random", "Greedy", "SA", "Bandit", "PPO"]
    improvements = [results[m]["improvement"] for m in methods]
    bar_colors = [colors[m] for m in methods]

    bars = ax.bar(methods, improvements, color=bar_colors, alpha=0.7, edgecolor="black")
    ax.set_ylabel("Fitness Improvement", fontsize=12)
    ax.set_title("Final Improvement Comparison", fontsize=14, fontweight="bold")
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
            fontweight="bold",
        )

    # Highlight RL methods
    ax.axvline(2.5, color="red", linestyle="--", alpha=0.3, linewidth=2)
    ax.text(
        1,
        max(improvements) * 0.9,
        "Baselines",
        ha="center",
        fontsize=10,
        style="italic",
    )
    ax.text(
        3.5,
        max(improvements) * 0.9,
        "RL Methods",
        ha="center",
        fontsize=10,
        style="italic",
        color="red",
    )

    plt.tight_layout()
    plt.savefig("all_methods_comparison.png", dpi=150, bbox_inches="tight")
    print("\n✓ Plot saved: all_methods_comparison.png")

    # Winner
    winner = max(results.items(), key=lambda x: x[1]["improvement"])
    print(f"\n🏆 WINNER: {winner[0]} (+{winner[1]['improvement']:.2f})")

    # RL vs Baseline improvement
    rl_best = max(results["Bandit"]["improvement"], results["PPO"]["improvement"])
    baseline_best = max(
        results["Random"]["improvement"],
        results["Greedy"]["improvement"],
        results["SA"]["improvement"],
    )

    print(f"\nRL Advantage: {rl_best/baseline_best:.2f}x better than best baseline")


if __name__ == "__main__":
    main()
