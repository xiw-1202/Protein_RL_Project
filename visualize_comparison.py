#!/usr/bin/env python3
"""
Publication-Quality Visualization: Original vs Improved RL Methods
Dataset: SAV1_MOUSE_Tsuboyama_2023_2YSB

Creates two main figures:
1. Comprehensive 4-panel comparison (16"x12", 300 DPI):
   - Panel A: Mean improvement by method (k=1)
   - Panel B: Performance across k-values (1, 3, 5, 10)
   - Panel C: Variance/consistency comparison
   - Panel D: Improvement factors (multipliers vs baselines)

2. Detailed k-value comparison with error bars

Outputs:
- improved_methods_comprehensive_comparison.png
- detailed_k_value_comparison.png

Usage:
    python visualize_comparison.py
"""

import pandas as pd
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_pickle_results(results_dir, pattern="result_*.pkl"):
    """Load results from pickle files"""
    results = []
    for pkl_file in Path(results_dir).glob(pattern):
        with open(pkl_file, 'rb') as f:
            result = pickle.load(f)
            summary = {k: v for k, v in result.items() if k != "history"}
            results.append(summary)
    return results


def load_json_results(results_dir, pattern="summary_*.json"):
    """Load results from JSON files"""
    results = []
    for json_file in Path(results_dir).glob(pattern):
        with open(json_file) as f:
            results.append(json.load(f))
    return results


def create_comprehensive_comparison(df_old, df_new, output_dir="Protein_RL_Results"):
    """Create comprehensive comparison plots"""
    
    sns.set_style("whitegrid")
    sns.set_context("talk")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Prepare data for k=1
    df_old_k1 = df_old[df_old['k'] == 1]
    df_new_k1 = df_new[df_new['k'] == 1]
    
    # Plot 1: Mean Improvement Comparison (k=1)
    ax1 = axes[0, 0]
    
    # Get mean improvements
    old_means = df_old_k1.groupby('method')['improvement'].mean()
    new_means = df_new_k1.groupby('method')['improvement'].mean()
    
    # Prepare data
    methods = []
    improvements = []
    colors = []
    
    # Add baselines
    for method in ['random', 'sa']:
        if method in old_means.index:
            methods.append(method.upper())
            improvements.append(old_means[method])
            colors.append('gray')
    
    # Add original RL
    if 'bandit' in old_means.index:
        methods.append('Bandit\n(Thompson)')
        improvements.append(old_means['bandit'])
        colors.append('steelblue')
    
    if 'ppo' in old_means.index:
        methods.append('PPO v1')
        improvements.append(old_means['ppo'])
        colors.append('steelblue')
    
    # Add improved RL
    method_map = {'ppo_v2': 'PPO v2', 'ucb1': 'UCB1', 'ucb_tuned': 'UCB-Tuned'}
    for method, name in method_map.items():
        if method in new_means.index:
            methods.append(name)
            improvements.append(new_means[method])
            colors.append('green' if 'UCB' in name else 'orange')
    
    bars = ax1.bar(methods, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Mean Fitness Improvement', fontsize=14, fontweight='bold')
    ax1.set_title('Performance Comparison (k=1)', fontsize=16, fontweight='bold')
    
    # Add baseline line
    if 'bandit' in old_means.index:
        ax1.axhline(y=old_means['bandit'], color='steelblue', linestyle='--', 
                   alpha=0.5, label='Thompson Baseline')
        ax1.legend()
    
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Annotate bars
    for bar, val in zip(bars, improvements):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Performance Across K-values
    ax2 = axes[0, 1]
    
    # Prepare data by k
    old_by_k = df_old.groupby(['method', 'k'])['improvement'].mean().unstack(fill_value=0)
    new_by_k = df_new.groupby(['method', 'k'])['improvement'].mean().unstack(fill_value=0)
    
    k_values = sorted(set(df_old['k'].unique()) | set(df_new['k'].unique()))
    
    # Plot lines
    if 'bandit' in old_by_k.index:
        bandit_data = [old_by_k.loc['bandit', k] if k in old_by_k.columns else 0 for k in k_values]
        ax2.plot(k_values, bandit_data, 'o-', linewidth=3, markersize=10, 
                label='Bandit (Thompson) - Old', color='steelblue')
    
    if 'ucb1' in new_by_k.index:
        ucb1_data = [new_by_k.loc['ucb1', k] if k in new_by_k.columns else 0 for k in k_values]
        ax2.plot(k_values, ucb1_data, 's-', linewidth=3, markersize=10, 
                label='UCB1 - New', color='green')
    
    if 'ucb_tuned' in new_by_k.index:
        ucb_tuned_data = [new_by_k.loc['ucb_tuned', k] if k in new_by_k.columns else 0 for k in k_values]
        ax2.plot(k_values, ucb_tuned_data, '^-', linewidth=3, markersize=10, 
                label='UCB-Tuned - New', color='orange')
    
    ax2.set_xlabel('k (simultaneous mutations)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Mean Fitness Improvement', fontsize=14, fontweight='bold')
    ax2.set_title('Performance vs k-value', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(k_values)
    
    # Plot 3: Variance Comparison
    ax3 = axes[1, 0]
    
    # Get standard deviations
    old_stds = df_old_k1.groupby('method')['improvement'].std()
    new_stds = df_new_k1.groupby('method')['improvement'].std()
    
    methods_var = []
    variances = []
    colors_var = []
    
    if 'bandit' in old_stds.index:
        methods_var.append('Bandit\n(Thompson)')
        variances.append(old_stds['bandit'])
        colors_var.append('steelblue')
    
    if 'ppo' in old_stds.index:
        methods_var.append('PPO v1')
        variances.append(old_stds['ppo'])
        colors_var.append('steelblue')
    
    if 'ppo_v2' in new_stds.index:
        methods_var.append('PPO v2')
        variances.append(new_stds['ppo_v2'])
        colors_var.append('orange')
    
    if 'ucb1' in new_stds.index:
        methods_var.append('UCB1')
        variances.append(new_stds['ucb1'])
        colors_var.append('green')
    
    if 'ucb_tuned' in new_stds.index:
        methods_var.append('UCB-Tuned')
        variances.append(new_stds['ucb_tuned'])
        colors_var.append('green')
    
    bars = ax3.bar(methods_var, variances, color=colors_var, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Standard Deviation', fontsize=14, fontweight='bold')
    ax3.set_title('Consistency Comparison (Lower = More Stable)', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, variances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Improvement Factor
    ax4 = axes[1, 1]
    
    comparisons = []
    factors = []
    colors_factor = []
    
    # Calculate improvement factors
    if 'ppo' in old_means.index and 'ppo_v2' in new_means.index:
        comparisons.append('PPO v2\nvs PPO v1')
        factors.append(new_means['ppo_v2'] / old_means['ppo'])
        colors_factor.append('orange')
    
    if 'bandit' in old_means.index:
        if 'ucb1' in new_means.index:
            comparisons.append('UCB1\nvs Thompson')
            factors.append(new_means['ucb1'] / old_means['bandit'])
            colors_factor.append('green')
        
        if 'ucb_tuned' in new_means.index:
            comparisons.append('UCB-Tuned\nvs Thompson')
            factors.append(new_means['ucb_tuned'] / old_means['bandit'])
            colors_factor.append('green')
    
    bars = ax4.bar(comparisons, factors, color=colors_factor, alpha=0.7, 
                   edgecolor='black', linewidth=1.5)
    ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Baseline (1.0x)')
    ax4.set_ylabel('Improvement Factor', fontsize=14, fontweight='bold')
    ax4.set_title('Relative Improvement Over Original Methods', fontsize=16, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, factors):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}x', ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir) / 'improved_methods_comprehensive_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    plt.show()


def create_detailed_k_comparison(df_old, df_new, output_dir="Protein_RL_Results"):
    """Create detailed k-value comparison plot"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Prepare data
    old_by_k = df_old.groupby(['method', 'k'])['improvement'].agg(['mean', 'std']).reset_index()
    new_by_k = df_new.groupby(['method', 'k'])['improvement'].agg(['mean', 'std']).reset_index()
    
    # Plot old methods
    for method in ['bandit', 'ppo']:
        method_data = old_by_k[old_by_k['method'] == method]
        if len(method_data) > 0:
            label = 'Bandit (Thompson)' if method == 'bandit' else 'PPO v1'
            ax.errorbar(method_data['k'], method_data['mean'], 
                       yerr=method_data['std'],
                       marker='o', markersize=10, linewidth=2.5,
                       capsize=5, capthick=2, label=label, 
                       linestyle='--', alpha=0.7)
    
    # Plot new methods
    method_map = {'ppo_v2': 'PPO v2', 'ucb1': 'UCB1', 'ucb_tuned': 'UCB-Tuned'}
    for method, name in method_map.items():
        method_data = new_by_k[new_by_k['method'] == method]
        if len(method_data) > 0:
            ax.errorbar(method_data['k'], method_data['mean'], 
                       yerr=method_data['std'],
                       marker='s', markersize=10, linewidth=2.5,
                       capsize=5, capthick=2, label=name)
    
    ax.set_xlabel('k (number of simultaneous mutations)', fontsize=14)
    ax.set_ylabel('Mean Fitness Improvement ± Std', fontsize=14)
    ax.set_title('SAV1_MOUSE: Detailed Performance Comparison', 
                fontsize=16, fontweight='bold')
    ax.legend(title='Method', fontsize=12, title_fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Set x-ticks
    k_values = sorted(set(df_old['k'].unique()) | set(df_new['k'].unique()))
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / 'detailed_k_value_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Figure saved to: {output_path}")
    
    plt.show()


def main():
    """Main visualization workflow"""
    print("="*70)
    print("CREATING VISUALIZATION: OLD vs IMPROVED RL METHODS")
    print("Dataset: SAV1_MOUSE_Tsuboyama_2023_2YSB")
    print("="*70)
    
    # Load old methods
    print("\nLoading original methods...")
    results_dir_old = Path("Protein_RL_Results/Person4A_SAV1")
    results_old = load_json_results(results_dir_old)
    if not results_old:
        results_old = load_pickle_results(results_dir_old)
    
    if not results_old:
        print("❌ Could not load original methods")
        return
    
    df_old = pd.DataFrame(results_old)
    print(f"✓ Loaded {len(df_old)} original experiments")
    
    # Load improved methods
    print("\nLoading improved methods...")
    results_dir_new = Path("Protein_RL_Results/SAV1_MOUSE_improved_RL")
    results_new = load_pickle_results(results_dir_new)
    
    if not results_new:
        print("❌ Could not load improved methods")
        return
    
    df_new = pd.DataFrame(results_new)
    print(f"✓ Loaded {len(df_new)} improved experiments")
    
    # Create plots
    print("\nCreating comprehensive comparison plot...")
    create_comprehensive_comparison(df_old, df_new)
    
    print("\nCreating detailed k-value comparison plot...")
    create_detailed_k_comparison(df_old, df_new)
    
    print("\n" + "="*70)
    print("✓ Visualization Complete!")
    print("="*70)


if __name__ == "__main__":
    main()
