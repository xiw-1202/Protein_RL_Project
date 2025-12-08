"""
Test that all dependencies are installed correctly
"""


def test_torch():
    import torch

    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")


def test_esm():
    import esm

    print(f"✓ ESM installed successfully")


def test_scientific_stack():
    import numpy as np
    import pandas as pd
    import scipy
    import sklearn

    print(f"✓ NumPy version: {np.__version__}")
    print(f"✓ Pandas version: {pd.__version__}")
    print(f"✓ SciPy version: {scipy.__version__}")
    print(f"✓ Scikit-learn version: {sklearn.__version__}")


def test_rl_libraries():
    import gym
    import stable_baselines3

    print(f"✓ Gym version: {gym.__version__}")
    print(f"✓ Stable-Baselines3 version: {stable_baselines3.__version__}")


if __name__ == "__main__":
    print("Testing installation...\n")
    test_torch()
    test_esm()
    test_scientific_stack()
    test_rl_libraries()
    print("\n✓ All dependencies installed successfully!")
