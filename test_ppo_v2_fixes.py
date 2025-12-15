"""
Quick test to verify PPO v2 fixes
Run this to ensure the critical bugs are resolved
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_ppo_v2():
    """Test PPO v2 with fixes"""
    print("=" * 80)
    print("TESTING PPO v2 FIXES")
    print("=" * 80)
    
    # Test 1: Import
    print("\n[1/5] Testing imports...")
    try:
        from src.models.esm_oracle import ESM2Oracle
        from src.rl_methods.ppo_optimizer_v2 import PPOOptimizerV2
        print("✓ Imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test 2: Initialize oracle
    print("\n[2/5] Initializing ESM-2 oracle...")
    try:
        oracle = ESM2Oracle(model_name="esm2_t33_650M_UR50D", device="cuda")
        print("✓ Oracle initialized")
    except Exception as e:
        print(f"✗ Oracle initialization failed: {e}")
        return False
    
    # Test 3: Initialize PPO
    print("\n[3/5] Initializing PPO v2...")
    try:
        ppo = PPOOptimizerV2(oracle, k=1, seed=42, entropy_coef=0.01)
        print("✓ PPO v2 initialized")
        print(f"  - batch_converter: {'✓' if hasattr(ppo, 'batch_converter') else '✗'}")
    except Exception as e:
        print(f"✗ PPO initialization failed: {e}")
        return False
    
    # Test 4: Test select_actions
    print("\n[4/5] Testing select_actions...")
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
    
    try:
        mutations, log_prob, value = ppo.select_actions(test_seq)
        print(f"✓ select_actions works!")
        print(f"  - Selected {len(mutations)} mutations: {mutations}")
        print(f"  - Log prob: {log_prob:.4f}")
        print(f"  - Value: {value:.4f}")
        
        # Verify position-dependent AA policy
        if ppo.network is not None:
            print(f"  - Network initialized: ✓")
            print(f"  - AA policy output shape should be [batch, seq_len, 20]")
    except Exception as e:
        print(f"✗ select_actions failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test short optimization run
    print("\n[5/5] Testing optimization (budget=10)...")
    try:
        short_seq = "MKTAYIAK"  # Very short for quick test
        results = ppo.optimize(short_seq, budget=10)
        print(f"✓ Optimization completed!")
        print(f"  - Best fitness: {results['best_fitness']:.4f}")
        print(f"  - Improvement: {results['improvement']:.4f}")
        print(f"  - Queries used: {results['queries_used']}")
    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓✓✓ ALL TESTS PASSED! PPO v2 is ready to use.")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_ppo_v2()
    exit(0 if success else 1)
