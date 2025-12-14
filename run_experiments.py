"""
Run full experiments across all datasets, methods, seeds, and k-values
Saves results incrementally to avoid data loss

UPDATED: Now includes improved RL methods:
- ppo_v2: PPO with ESM-2 embeddings + entropy bonus
- ucb: UCB bandit (standard)
- ucb1: UCB1 bandit (c=sqrt(2))
- ucb_tuned: UCB-Tuned bandit (variance-based)
"""

import sys

sys.path.append("src")

from src.models.esm_oracle import ESM2Oracle
from src.baselines.random_baseline import RandomBaseline
from src.baselines.greedy_baseline import GreedyBaseline
from src.baselines.simulated_annealing import SimulatedAnnealingBaseline

# Original RL methods
from src.rl_methods.contextual_bandit import ContextualBandit
from src.rl_methods.ppo_optimizer import PPOOptimizer

# Improved RL methods
from src.rl_methods.ppo_optimizer_v2 import PPOOptimizerV2
from src.rl_methods.contextual_bandit_ucb import (
    ContextualBanditUCB,
    ContextualBanditUCB1,
    ContextualBanditUCBTuned,
)

from pathlib import Path
import pandas as pd
import pickle
import json
import time
from datetime import datetime


def load_wild_type(dms_id):
    """Load wild-type sequence from FASTA"""
    fasta_path = Path(f"data/raw/wild_types/{dms_id}.fasta")

    with open(fasta_path, "r") as f:
        lines = f.readlines()
        sequence = "".join([line.strip() for line in lines if not line.startswith(">")])

    return sequence


def load_datasets():
    """Load dataset metadata"""
    df = pd.read_csv("data/raw/balanced_datasets.csv")
    return df


def get_method(method_name, oracle, k, seed):
    """
    Get method instance
    
    Available methods:
    - Baselines: random, greedy, sa
    - Original RL: bandit (Thompson Sampling), ppo (PPO v1)
    - Improved RL: ppo_v2, ucb, ucb1, ucb_tuned
    """
    # Baselines
    if method_name == "random":
        return RandomBaseline(oracle, k=k, seed=seed)
    elif method_name == "greedy":
        return GreedyBaseline(oracle, k=k)
    elif method_name == "sa":
        return SimulatedAnnealingBaseline(oracle, k=k, seed=seed)
    
    # Original RL methods
    elif method_name == "bandit":
        return ContextualBandit(oracle, k=k, seed=seed)
    elif method_name == "ppo":
        return PPOOptimizer(oracle, k=k, seed=seed)
    
    # Improved RL methods
    elif method_name == "ppo_v2":
        return PPOOptimizerV2(oracle, k=k, seed=seed, entropy_coef=0.01)
    elif method_name == "ucb":
        return ContextualBanditUCB(oracle, k=k, seed=seed, ucb_c=2.0)
    elif method_name == "ucb1":
        return ContextualBanditUCB1(oracle, k=k, seed=seed)
    elif method_name == "ucb_tuned":
        return ContextualBanditUCBTuned(oracle, k=k, seed=seed)
    
    else:
        raise ValueError(
            f"Unknown method: {method_name}\n"
            f"Available methods:\n"
            f"  Baselines: random, greedy, sa\n"
            f"  Original RL: bandit, ppo\n"
            f"  Improved RL: ppo_v2, ucb, ucb1, ucb_tuned"
        )


def run_single_experiment(dataset_id, method_name, k, seed, budget, model_name, device):
    """
    Run a single experiment

    Returns:
        Dict with experiment results
    """
    print(f"\n{'='*70}")
    print(f"Experiment: {dataset_id} | {method_name} | k={k} | seed={seed}")
    print(f"{'='*70}")

    # Load wild-type
    wt_seq = load_wild_type(dataset_id)
    print(f"Sequence length: {len(wt_seq)} AA")

    # Initialize oracle
    oracle = ESM2Oracle(model_name=model_name, device=device)

    # Get method
    method = get_method(method_name, oracle, k, seed)

    # Run optimization
    start_time = time.time()
    results = method.optimize(wt_seq, budget=budget)
    runtime = time.time() - start_time

    # Add metadata
    results["dataset_id"] = dataset_id
    results["method"] = method_name
    results["k"] = k
    results["seed"] = seed
    results["budget"] = budget
    results["model_name"] = model_name
    results["runtime"] = runtime
    results["timestamp"] = datetime.now().isoformat()

    print(f"\n✓ Complete in {runtime/60:.1f} minutes")
    print(f"  Best fitness: {results['best_fitness']:.4f}")
    print(f"  Improvement: {results['improvement']:.4f}")

    return results


def save_results(results, output_dir):
    """Save results to disk"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results (with history)
    pickle_path = (
        output_dir
        / f"result_{results['dataset_id']}_{results['method']}_k{results['k']}_seed{results['seed']}_{timestamp}.pkl"
    )
    with open(pickle_path, "wb") as f:
        pickle.dump(results, f)

    # Save summary (without history)
    summary = {k: v for k, v in results.items() if k != "history"}
    json_path = (
        output_dir
        / f"summary_{results['dataset_id']}_{results['method']}_k{results['k']}_seed{results['seed']}_{timestamp}.json"
    )
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Saved: {pickle_path.name}")

    return pickle_path, json_path


def run_experiments(
    methods,
    datasets,
    k_values,
    seeds,
    budget,
    model_name,
    device,
    output_dir,
    resume=False,
):
    """
    Run all experiments

    Args:
        methods: List of method names
        datasets: List of dataset IDs
        k_values: List of k values
        seeds: List of random seeds
        budget: Oracle query budget
        model_name: ESM-2 model name
        device: Device to use
        output_dir: Output directory
        resume: If True, skip already completed experiments
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate total experiments
    total = len(methods) * len(datasets) * len(k_values) * len(seeds)

    print(f"\n{'='*70}")
    print(f"EXPERIMENT CONFIGURATION")
    print(f"{'='*70}")
    print(f"Methods: {methods}")
    print(f"Datasets: {datasets}")
    print(f"k-values: {k_values}")
    print(f"Seeds: {seeds}")
    print(f"Budget: {budget}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"\nTotal experiments: {total}")
    print(f"Output: {output_dir}")
    print(f"Resume: {resume}")
    print(f"{'='*70}\n")

    # Track progress
    completed = 0
    failed = []

    # Run experiments
    for dataset_id in datasets:
        for method in methods:
            for k in k_values:
                for seed in seeds:
                    # Check if already completed (if resuming)
                    if resume:
                        existing = list(
                            output_dir.glob(
                                f"result_{dataset_id}_{method}_k{k}_seed{seed}_*.pkl"
                            )
                        )
                        if existing:
                            print(
                                f"⏭️  Skipping (already completed): {dataset_id} | {method} | k={k} | seed={seed}"
                            )
                            completed += 1
                            continue

                    try:
                        # Run experiment
                        results = run_single_experiment(
                            dataset_id, method, k, seed, budget, model_name, device
                        )

                        # Save results
                        save_results(results, output_dir)

                        completed += 1
                        print(
                            f"\n📊 Progress: {completed}/{total} ({100*completed/total:.1f}%)"
                        )

                    except Exception as e:
                        print(
                            f"\n❌ FAILED: {dataset_id} | {method} | k={k} | seed={seed}"
                        )
                        print(f"   Error: {e}")
                        failed.append(
                            {
                                "dataset_id": dataset_id,
                                "method": method,
                                "k": k,
                                "seed": seed,
                                "error": str(e),
                            }
                        )

    # Summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE!")
    print(f"{'='*70}")
    print(f"Completed: {completed}/{total}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed experiments:")
        for f in failed:
            print(
                f"  - {f['dataset_id']} | {f['method']} | k={f['k']} | seed={f['seed']}"
            )

        # Save failed list
        failed_path = output_dir / "failed_experiments.json"
        with open(failed_path, "w") as f:
            json.dump(failed, f, indent=2)
        print(f"\nFailed experiments saved to: {failed_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run protein optimization experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Methods:
  Baselines:
    random              Random mutation selection
    greedy              Greedy hill climbing (not recommended for k>1)
    sa                  Simulated annealing
    
  Original RL:
    bandit              Contextual bandit with Thompson Sampling
    ppo                 PPO v1 (one-hot encoding)
    
  Improved RL (NEW):
    ppo_v2              PPO v2 (ESM-2 embeddings + entropy + learned AA)
    ucb                 UCB bandit (c=2.0, deterministic)
    ucb1                UCB1 bandit (c=sqrt(2), classic variant)
    ucb_tuned           UCB-Tuned (variance-based exploration)

Examples:
  # Run all original methods
  python run_experiments.py --methods random sa bandit ppo
  
  # Compare PPO v1 vs v2
  python run_experiments.py --methods ppo ppo_v2 --datasets SAV1_MOUSE
  
  # Test improved bandits
  python run_experiments.py --methods bandit ucb ucb1 --k_values 1 3
  
  # Quick test on one dataset
  python run_experiments.py --methods ppo_v2 --datasets SAV1_MOUSE --k_values 1 --seeds 42 --budget 100
        """
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["random", "sa", "bandit", "ppo"],
        help="Methods to run (default: random sa bandit ppo)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Specific dataset(s) to run (default: all from balanced_datasets.csv)",
    )
    parser.add_argument(
        "--k_values",
        nargs="+",
        type=int,
        default=[1, 3, 5, 10],
        help="k-values for mutations (default: 1 3 5 10)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456, 789, 1011],
        help="Random seeds (default: 42 123 456 789 1011)",
    )
    parser.add_argument(
        "--budget", 
        type=int, 
        default=300, 
        help="Oracle query budget per experiment (default: 300)"
    )
    parser.add_argument(
        "--model", 
        default="esm2_t33_650M_UR50D", 
        help="ESM-2 model name (default: esm2_t33_650M_UR50D)"
    )
    parser.add_argument(
        "--device", 
        default="auto", 
        help="Device (auto, cuda, mps, cpu) (default: auto)"
    )
    parser.add_argument(
        "--output", 
        default="experiments/results", 
        help="Output directory (default: experiments/results)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (skip completed experiments)",
    )

    args = parser.parse_args()

    # Get datasets
    if args.datasets:
        # Use specified datasets
        dataset_ids = args.datasets
        print(f"Using specified datasets: {dataset_ids}")
    else:
        # Load from balanced_datasets.csv
        datasets_df = load_datasets()
        dataset_ids = datasets_df["DMS_id"].tolist()
        print(f"Loaded {len(dataset_ids)} datasets from balanced_datasets.csv")

    # Run experiments
    run_experiments(
        methods=args.methods,
        datasets=dataset_ids,
        k_values=args.k_values,
        seeds=args.seeds,
        budget=args.budget,
        model_name=args.model,
        device=args.device,
        output_dir=args.output,
        resume=args.resume,
    )
