"""
Test script to compare improved RL methods vs originals

Compares:
1. PPO v1 (original) vs PPO v2 (ESM-2 + entropy)
2. Contextual Bandit (Thompson) vs UCB variants

Run on a small dataset to verify improvements.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.esm_oracle import ESM2Oracle
from src.rl_methods.ppo_optimizer import PPOOptimizer
from src.rl_methods.ppo_optimizer_v2 import PPOOptimizerV2
from src.rl_methods.contextual_bandit import ContextualBandit
from src.rl_methods.contextual_bandit_ucb import (
    ContextualBanditUCB,
    ContextualBanditUCB1,
    ContextualBanditUCBTuned
)
import json


def load_test_sequence():
    """Load a small test sequence"""
    # SAV1 is shortest - good for quick testing
    import pickle
    dataset_path = "data/dms_datasets/SAV1_MOUSE_Tsuboyama_2023_2YSB.pkl"
    
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    
    return data['wt_seq']


def run_comparison(budget=100, k=1, seed=42):
    """
    Run comparison test
    
    Args:
        budget: Small budget for quick testing (100 queries)
        k: Number of mutations
        seed: Random seed
    """
    print("=" * 80)
    print("IMPROVED RL METHODS COMPARISON TEST")
    print("=" * 80)
    print(f"Budget: {budget} queries")
    print(f"k: {k} mutations")
    print(f"Seed: {seed}")
    print()
    
    # Load test sequence
    wt_seq = load_test_sequence()
    print(f"Test sequence: SAV1_MOUSE ({len(wt_seq)} AA)")
    print()
    
    # Initialize oracle
    print("Loading ESM-2 model...")
    oracle = ESM2Oracle(model_name="esm2_t33_650M_UR50D", device="cuda")
    print()
    
    results = {}
    
    # Test PPO v1 (original)
    print("\n" + "=" * 80)
    print("TEST 1: PPO v1 (Original)")
    print("=" * 80)
    ppo_v1 = PPOOptimizer(oracle, k=k, seed=seed)
    results['ppo_v1'] = ppo_v1.optimize(wt_seq, budget=budget)
    
    # Test PPO v2 (improved)
    print("\n" + "=" * 80)
    print("TEST 2: PPO v2 (ESM-2 + Entropy)")
    print("=" * 80)
    ppo_v2 = PPOOptimizerV2(oracle, k=k, seed=seed, entropy_coef=0.01)
    results['ppo_v2'] = ppo_v2.optimize(wt_seq, budget=budget)
    
    # Test Contextual Bandit (Thompson Sampling)
    print("\n" + "=" * 80)
    print("TEST 3: Contextual Bandit (Thompson Sampling)")
    print("=" * 80)
    bandit_thompson = ContextualBandit(oracle, k=k, seed=seed)
    results['bandit_thompson'] = bandit_thompson.optimize(wt_seq, budget=budget)
    
    # Test UCB Bandit
    print("\n" + "=" * 80)
    print("TEST 4: UCB Bandit (c=2.0)")
    print("=" * 80)
    bandit_ucb = ContextualBanditUCB(oracle, k=k, seed=seed, ucb_c=2.0)
    results['bandit_ucb'] = bandit_ucb.optimize(wt_seq, budget=budget)
    
    # Test UCB1 Bandit
    print("\n" + "=" * 80)
    print("TEST 5: UCB1 Bandit (c=sqrt(2))")
    print("=" * 80)
    bandit_ucb1 = ContextualBanditUCB1(oracle, k=k, seed=seed)
    results['bandit_ucb1'] = bandit_ucb1.optimize(wt_seq, budget=budget)
    
    # Test UCB-Tuned
    print("\n" + "=" * 80)
    print("TEST 6: UCB-Tuned (variance-based)")
    print("=" * 80)
    bandit_ucb_tuned = ContextualBanditUCBTuned(oracle, k=k, seed=seed)
    results['bandit_ucb_tuned'] = bandit_ucb_tuned.optimize(wt_seq, budget=budget)
    
    # Print summary
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Method':<25} {'Improvement':<15} {'Best Fitness':<15} {'Improvements'}")
    print("-" * 80)
    
    for method_name, result in results.items():
        # Remove ucb_stats for display
        display_name = method_name.replace('_', ' ').title()
        improvement = result['improvement']
        best_fitness = result['best_fitness']
        improvements_count = result.get('improvements', 0)
        
        print(f"{display_name:<25} {improvement:>+.4f} {' '*7} {best_fitness:>.4f} {' '*7} {improvements_count:>3}")
    
    # Find best method
    best_method = max(results.items(), key=lambda x: x[1]['improvement'])
    print(f"\n🏆 Best Method: {best_method[0].replace('_', ' ').title()}")
    print(f"   Improvement: +{best_method[1]['improvement']:.4f}")
    
    # Save results
    output_file = "improved_rl_comparison.json"
    with open(output_file, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for method, result in results.items():
            json_results[method] = {
                'improvement': float(result['improvement']),
                'best_fitness': float(result['best_fitness']),
                'wt_fitness': float(result['wt_fitness']),
                'improvements': int(result.get('improvements', 0)),
                'queries_used': int(result['queries_used'])
            }
        
        json.dump(json_results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test improved RL methods')
    parser.add_argument('--budget', type=int, default=100, help='Query budget')
    parser.add_argument('--k', type=int, default=1, help='Number of mutations')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    results = run_comparison(budget=args.budget, k=args.k, seed=args.seed)
