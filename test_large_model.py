"""
Test large model (650M) vs small model (35M)
Compare speed and accuracy on one dataset
"""

import sys

sys.path.append("src")

# FIX: Add src. prefix
from src.models.esm_oracle import ESM2Oracle
from src.baselines.random_baseline import RandomBaseline
from pathlib import Path
import time


def load_wild_type(dms_id):
    """Load wild-type sequence from FASTA"""
    fasta_path = Path(f"data/raw/wild_types/{dms_id}.fasta")

    with open(fasta_path, "r") as f:
        lines = f.readlines()
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])

    return sequence


def test_model(model_name, wt_seq, budget=20):
    """Test a model"""
    print(f"\nTesting {model_name}")
    print("-" * 70)

    # Initialize oracle
    start_time = time.time()
    oracle = ESM2Oracle(model_name=model_name, device="auto")
    init_time = time.time() - start_time

    print(f"  Initialization: {init_time:.2f}s")

    # Run random baseline
    baseline = RandomBaseline(oracle, k=1, seed=42)

    start_time = time.time()
    results = baseline.optimize(wt_seq, budget=budget)
    run_time = time.time() - start_time

    print(f"  Runtime: {run_time:.2f}s")
    print(f"  Time per query: {run_time/budget:.3f}s")
    print(f"  Best fitness: {results['best_fitness']:.4f}")
    print(f"  Improvement: {results['improvement']:.4f}")

    return {
        "model": model_name,
        "init_time": init_time,
        "run_time": run_time,
        "time_per_query": run_time / budget,
        "best_fitness": results["best_fitness"],
        "improvement": results["improvement"],
    }


def main():
    print("=" * 70)
    print("LARGE MODEL (650M) vs SMALL MODEL (35M) COMPARISON")
    print("=" * 70)

    # Use smallest dataset
    dms_id = "SAV1_MOUSE_Tsuboyama_2023_2YSB"
    wt_seq = load_wild_type(dms_id)

    print(f"\nDataset: {dms_id}")
    print(f"Sequence length: {len(wt_seq)} AA")

    budget = 20  # Small budget to compare speed

    # Test both models
    results = {}

    # Small model
    results["small"] = test_model("esm2_t12_35M_UR50D", wt_seq, budget)

    # Large model
    results["large"] = test_model("esm2_t33_650M_UR50D", wt_seq, budget)

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    speedup = results["large"]["time_per_query"] / results["small"]["time_per_query"]

    print(f"\nSpeed:")
    print(f"  Small (35M):  {results['small']['time_per_query']:.3f}s per query")
    print(f"  Large (650M): {results['large']['time_per_query']:.3f}s per query")
    print(f"  → Large is {speedup:.1f}x slower")

    print(f"\nAccuracy (fitness scores):")
    print(f"  Small: {results['small']['best_fitness']:.4f}")
    print(f"  Large: {results['large']['best_fitness']:.4f}")
    print(
        f"  → Difference: {abs(results['large']['best_fitness'] - results['small']['best_fitness']):.4f}"
    )

    print(f"\nFor 500-query experiment:")
    print(f"  Small: ~{results['small']['time_per_query']*500/60:.1f} minutes")
    print(f"  Large: ~{results['large']['time_per_query']*500/60:.1f} minutes")

    print(f"\nFor full experiments (600 runs × 500 queries):")
    total_queries = 600 * 500
    print(
        f"  Small: ~{results['small']['time_per_query']*total_queries/3600:.1f} hours"
    )
    print(
        f"  Large: ~{results['large']['time_per_query']*total_queries/3600:.1f} hours"
    )


if __name__ == "__main__":
    main()
