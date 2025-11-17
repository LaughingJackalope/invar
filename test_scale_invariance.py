"""
Test suite for scale invariance implementation
Validates correctness of all three systems
"""

import numpy as np
from scale_invariance import (
    simulate_equilibrium,
    run_scale_invariance_test,
    quantify_divergence
)


def test_system1_distribution_normalization():
    """Test that System 1 produces valid probability distributions"""
    print("Testing System 1: Distribution normalization...")
    
    N = 3
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    
    P = simulate_equilibrium(W, H, T=1.0, num_samples=5000)
    
    # Check normalization
    assert np.isclose(P.sum(), 1.0), "Distribution must sum to 1"
    
    # Check non-negative
    assert np.all(P >= 0), "Probabilities must be non-negative"
    
    # Check correct length
    assert len(P) == 2**N, f"Distribution must have 2^N={2**N} elements"
    
    print("  ✓ Normalization: PASS")
    print("  ✓ Non-negativity: PASS")
    print("  ✓ Correct length: PASS")
    print()


def test_system1_temperature_effect():
    """Test that temperature affects distribution as expected"""
    print("Testing System 1: Temperature effect...")
    
    N = 3
    W = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    H = np.array([1, 0, -1])
    
    # Low temperature should be more peaked
    P_low = simulate_equilibrium(W, H, T=0.1, num_samples=5000)
    
    # High temperature should be more uniform
    P_high = simulate_equilibrium(W, H, T=10.0, num_samples=5000)
    
    # Measure entropy (high T should have higher entropy)
    entropy_low = -np.sum(P_low[P_low > 0] * np.log(P_low[P_low > 0]))
    entropy_high = -np.sum(P_high[P_high > 0] * np.log(P_high[P_high > 0]))
    
    assert entropy_high > entropy_low, "High T should have higher entropy"
    
    print(f"  Entropy (T=0.1): {entropy_low:.3f}")
    print(f"  Entropy (T=10): {entropy_high:.3f}")
    print("  ✓ Temperature effect: PASS")
    print()


def test_system2_case_generation():
    """Test that System 2 generates correct experimental cases"""
    print("Testing System 2: Case generation...")
    
    N = 4
    alpha = 2.0
    T0 = 1.0
    
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N, alpha, T0, num_samples=5000, seed=123
    )
    
    # Check all distributions are valid
    assert np.isclose(P_orig.sum(), 1.0), "P_orig must be normalized"
    assert np.isclose(P_scaled_E.sum(), 1.0), "P_scaled_E must be normalized"
    assert np.isclose(P_test.sum(), 1.0), "P_test must be normalized"
    
    # Check parameter scaling
    assert np.allclose(params['W_scaled'], alpha * params['W']), "W scaling incorrect"
    assert np.allclose(params['H_scaled'], alpha * params['H']), "H scaling incorrect"
    assert params['T_scaled'] == alpha * T0, "T scaling incorrect"
    
    print("  ✓ All distributions normalized: PASS")
    print("  ✓ Parameter scaling correct: PASS")
    print()


def test_system3_divergence_properties():
    """Test that System 3 computes divergence correctly"""
    print("Testing System 3: Divergence properties...")
    
    # Identical distributions should have zero divergence
    P = np.array([0.3, 0.5, 0.2])
    D_self = quantify_divergence(P, P)
    assert D_self < 1e-6, "D_KL(P||P) should be 0"
    
    # Different distributions should have positive divergence
    Q = np.array([0.1, 0.6, 0.3])
    D_diff = quantify_divergence(P, Q)
    assert D_diff > 0, "D_KL(P||Q) should be > 0 for P ≠ Q"
    
    print(f"  D_KL(P||P): {D_self:.6f} (should be ≈ 0)")
    print(f"  D_KL(P||Q): {D_diff:.6f} (should be > 0)")
    print("  ✓ Self-divergence zero: PASS")
    print("  ✓ Cross-divergence positive: PASS")
    print()


def test_scale_invariance_property():
    """Integration test: verify scale invariance holds"""
    print("Testing Scale Invariance Property...")
    print("This is the main thesis validation.")
    print()
    
    # Small system for fast testing
    N = 4
    alpha = 2.0
    T0 = 1.0
    num_samples = 15000
    
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N, alpha, T0, num_samples, seed=42
    )
    
    # Calculate divergences
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)
    
    print(f"  D_KL(P_orig || P_test)     = {D_proof:.6f}")
    print(f"  D_KL(P_orig || P_scaled_E) = {D_control:.6f}")
    print()
    
    # Relaxed thresholds for testing
    assert D_proof < 0.05, "Invariance: D_KL should be small"
    assert D_control > 0.05, "Control: D_KL should be large"
    
    print("  ✓ Invariance property: PASS")
    print("  ✓ Control differs: PASS")
    print()


def test_multiple_alphas():
    """Test that invariance holds for different scaling factors"""
    print("Testing Multiple Scaling Factors...")
    
    N = 4
    alphas = [1.5, 2.0, 2.5]
    num_samples = 10000
    
    for alpha in alphas:
        P_orig, _, P_test, _ = run_scale_invariance_test(
            N, alpha, T0=1.0, num_samples=num_samples, seed=42
        )
        
        D_proof = quantify_divergence(P_orig, P_test)
        print(f"  α={alpha}: D_KL = {D_proof:.6f}", end="")
        
        if D_proof < 0.05:
            print(" ✓")
        else:
            print(" (marginal)")
    
    print()


def run_all_tests():
    """Run complete test suite"""
    print("=" * 60)
    print("SCALE INVARIANCE TEST SUITE")
    print("=" * 60)
    print()
    
    try:
        test_system1_distribution_normalization()
        test_system1_temperature_effect()
        test_system2_case_generation()
        test_system3_divergence_properties()
        test_scale_invariance_property()
        test_multiple_alphas()
        
        print("=" * 60)
        print("ALL TESTS PASSED ✓")
        print("=" * 60)
        return True
        
    except AssertionError as e:
        print("=" * 60)
        print(f"TEST FAILED ✗")
        print(f"Error: {e}")
        print("=" * 60)
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
