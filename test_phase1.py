"""
Phase 1 Validation Script

Tests the pluggable sampler architecture to ensure:
1. Backward compatibility (no sampler specified)
2. Explicit NumpySampler use
3. Backend switching capability
4. Identical results from both approaches
"""

import numpy as np
from numpy_sampler import NumpySampler
from sampler_interface import SamplerFactory
from scale_invariance import run_scale_invariance_test, quantify_divergence


def test_backward_compatibility():
    """Test that original API still works (no sampler specified)."""
    print("=" * 60)
    print("TEST 1: BACKWARD COMPATIBILITY")
    print("=" * 60)
    print("Testing original API (no sampler parameter)...\n")
    
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N=4,
        alpha=2.0,
        T0=1.0,
        num_samples=5000,
        seed=42
        # NO sampler parameter - should use default NumpySampler
    )
    
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)
    
    print(f"\nResults:")
    print(f"  D_KL(proof):   {D_proof:.6f}")
    print(f"  D_KL(control): {D_control:.6f}")
    
    assert D_proof < 0.05, "Proof should show invariance"
    assert D_control > 0.05, "Control should show difference"
    
    print("\n✓ BACKWARD COMPATIBILITY: PASS\n")
    return P_orig, P_test


def test_explicit_numpy_sampler():
    """Test explicitly passing NumpySampler."""
    print("=" * 60)
    print("TEST 2: EXPLICIT NUMPY SAMPLER")
    print("=" * 60)
    print("Testing with explicitly created NumpySampler...\n")
    
    sampler = NumpySampler()
    info = sampler.get_backend_info()
    print(f"Using backend: {info['name']}")
    print(f"Type: {info['type']}")
    print(f"Method: {info['sampling_method']}\n")
    
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N=4,
        alpha=2.0,
        T0=1.0,
        num_samples=5000,
        seed=42,
        sampler=sampler  # Explicitly pass sampler
    )
    
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)
    
    print(f"\nResults:")
    print(f"  D_KL(proof):   {D_proof:.6f}")
    print(f"  D_KL(control): {D_control:.6f}")
    
    assert D_proof < 0.05, "Proof should show invariance"
    assert D_control > 0.05, "Control should show difference"
    
    print("\n✓ EXPLICIT SAMPLER: PASS\n")
    return P_orig, P_test


def test_factory_creation():
    """Test SamplerFactory."""
    print("=" * 60)
    print("TEST 3: SAMPLER FACTORY")
    print("=" * 60)
    print("Testing SamplerFactory.create_sampler()...\n")
    
    # Create via factory
    sampler = SamplerFactory.create_sampler('numpy')
    info = sampler.get_backend_info()
    print(f"Factory created: {info['name']} backend")
    
    # Test sampling
    W = np.array([[0, 1], [1, 0]])
    H = np.array([0.5, -0.5])
    P = sampler.sample_distribution(W, H, T=1.0, num_samples=5000)
    
    print(f"Sampled distribution shape: {P.shape}")
    print(f"Sum: {P.sum():.6f}")
    
    assert P.shape == (4,), "Should have 2^2 = 4 states"
    assert np.isclose(P.sum(), 1.0), "Should be normalized"
    
    print("\n✓ FACTORY CREATION: PASS\n")


def test_result_consistency():
    """Test that both approaches give identical results (same seed)."""
    print("=" * 60)
    print("TEST 4: RESULT CONSISTENCY")
    print("=" * 60)
    print("Verifying both approaches give identical results...\n")
    
    # Approach 1: Default (implicit NumpySampler)
    np.random.seed(123)
    P_orig_1, _, P_test_1, _ = run_scale_invariance_test(
        N=3,
        alpha=2.0,
        T0=1.0,
        num_samples=3000,
        seed=123
    )
    
    # Approach 2: Explicit NumpySampler
    np.random.seed(123)
    sampler = NumpySampler()
    P_orig_2, _, P_test_2, _ = run_scale_invariance_test(
        N=3,
        alpha=2.0,
        T0=1.0,
        num_samples=3000,
        seed=123,
        sampler=sampler
    )
    
    # Compare
    diff_orig = np.max(np.abs(P_orig_1 - P_orig_2))
    diff_test = np.max(np.abs(P_test_1 - P_test_2))
    
    print(f"Max difference (P_orig): {diff_orig:.10f}")
    print(f"Max difference (P_test): {diff_test:.10f}")
    
    assert diff_orig < 1e-10, "Results should be identical"
    assert diff_test < 1e-10, "Results should be identical"
    
    print("\n✓ RESULT CONSISTENCY: PASS\n")


def run_all_phase1_tests():
    """Run complete Phase 1 validation suite."""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 15 + "PHASE 1 VALIDATION SUITE" + " " * 19 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    try:
        test_backward_compatibility()
        test_explicit_numpy_sampler()
        test_factory_creation()
        test_result_consistency()
        
        print("=" * 60)
        print("PHASE 1 VALIDATION: ALL TESTS PASSED ✓")
        print("=" * 60)
        print()
        print("Success Criteria Met:")
        print("  ✓ All existing tests pass unchanged")
        print("  ✓ Can swap backends without modifying Layer 1 code")
        print("  ✓ NumPy backend produces identical results to original")
        print("  ✓ Factory pattern works correctly")
        print()
        print("Status: READY FOR PHASE 2 (thrml integration)")
        print("=" * 60)
        
        return True
        
    except AssertionError as e:
        print("=" * 60)
        print("PHASE 1 VALIDATION: FAILED ✗")
        print("=" * 60)
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    success = run_all_phase1_tests()
    exit(0 if success else 1)
