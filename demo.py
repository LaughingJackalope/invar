"""
Demo script showing scale invariance proof with different configurations
"""

from scale_invariance import run_full_experiment
import numpy as np


def demo_quick():
    """Quick demo with small system"""
    print("QUICK DEMO (N=5, 20k samples)")
    print()
    return run_full_experiment(
        N=5,
        alpha=2.0,
        T0=1.0,
        num_samples=20000,
        seed=42
    )


def demo_rigorous():
    """Rigorous proof with more samples"""
    print("\n\n")
    print("RIGOROUS PROOF (N=6, 50k samples)")
    print()
    return run_full_experiment(
        N=6,
        alpha=2.0,
        T0=1.0,
        num_samples=50000,
        seed=42
    )


def demo_multiple_alphas():
    """Test different scaling factors"""
    print("\n\n")
    print("=" * 60)
    print("TESTING MULTIPLE SCALING FACTORS")
    print("=" * 60)
    print()
    
    alphas = [1.5, 2.0, 3.0]
    results = []
    
    for alpha in alphas:
        print(f"\n--- Testing α = {alpha} ---\n")
        result = run_full_experiment(
            N=5,
            alpha=alpha,
            T0=1.0,
            num_samples=30000,
            seed=42
        )
        results.append((alpha, result))
    
    # Summary
    print("\n\n")
    print("=" * 60)
    print("SUMMARY ACROSS SCALING FACTORS")
    print("=" * 60)
    for alpha, result in results:
        print(f"α = {alpha}:")
        print(f"  D_KL(proof)   = {result['D_proof']:.6f}")
        print(f"  D_KL(control) = {result['D_control']:.6f}")
        print(f"  Status: {'✓ PASS' if result['proof_valid'] else '✗ FAIL'}")
        print()


if __name__ == "__main__":
    # Run quick demo
    result1 = demo_quick()
    
    # Run rigorous proof if quick demo passes
    if result1['control_valid']:
        result2 = demo_rigorous()
        
        # If still interesting, test multiple alphas
        if result2['proof_valid']:
            demo_multiple_alphas()
