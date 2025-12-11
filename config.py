"""
Project configuration
"""

# Model settings
MODEL_CONFIG = {
    # For testing/development (fast)
    "small": {
        "name": "esm2_t12_35M_UR50D",
        "params": "35M",
        "speed": "fast",
        "accuracy": "good",
    },
    # For experiments (accurate)
    "large": {
        "name": "esm2_t33_650M_UR50D",
        "params": "650M",
        "speed": "slow (~20x slower)",
        "accuracy": "best",
    },
}

# Experiment settings
EXPERIMENT_CONFIG = {
    "model": "large",  # 'small' or 'large'
    "budget": 500,  # Oracle queries per run
    "k_values": [1, 3, 5, 10],  # Mutation budgets
    "seeds": [42, 123, 456, 789, 1011],  # 5 random seeds
    "device": "auto",  # 'auto', 'mps', 'cuda', or 'cpu'
}

# Dataset settings
DATASET_CONFIG = {
    "path": "data/raw/balanced_datasets.csv",
    "wild_type_dir": "data/raw/wild_types",
}

# Baseline settings
BASELINE_CONFIG = {
    "random": {"enabled": True},
    "greedy": {"enabled": True},
    "simulated_annealing": {"enabled": True, "T_start": 10.0, "T_end": 0.1},
}

# Output settings
OUTPUT_CONFIG = {
    "results_dir": "experiments/results",
    "plots_dir": "experiments/plots",
    "save_history": True,
    "save_plots": True,
}


def get_model_name():
    """Get model name based on config"""
    model_type = EXPERIMENT_CONFIG["model"]
    return MODEL_CONFIG[model_type]["name"]


def print_config():
    """Print current configuration"""
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION")
    print("=" * 70)

    model_type = EXPERIMENT_CONFIG["model"]
    model_info = MODEL_CONFIG[model_type]

    print(f"\nModel: {model_info['name']}")
    print(f"  Parameters: {model_info['params']}")
    print(f"  Speed: {model_info['speed']}")
    print(f"  Accuracy: {model_info['accuracy']}")

    print(f"\nExperiment Settings:")
    print(f"  Query budget: {EXPERIMENT_CONFIG['budget']}")
    print(f"  k-values: {EXPERIMENT_CONFIG['k_values']}")
    print(f"  Random seeds: {len(EXPERIMENT_CONFIG['seeds'])}")
    print(f"  Device: {EXPERIMENT_CONFIG['device']}")

    print(f"\nBaselines:")
    for method, config in BASELINE_CONFIG.items():
        status = "✓" if config["enabled"] else "✗"
        print(f"  {status} {method}")

    print("=" * 70)
