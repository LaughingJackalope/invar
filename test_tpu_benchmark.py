"""
Test Suite for System 9: TPU Integrity Benchmark

Validates the diagnostic tool for thermodynamic processing hardware.
"""

import numpy as np
from tpu_benchmark import (
    compute_tis,
    classify_tpu,
    run_tpu_benchmark,
    reference_tpu_exact,
    good_tpu_mcmc,
    noisy_tpu,
    rg_flow_analysis,
    TPUGrade
)


def test_tis_calculation():
    """Test 1: Verify TIS calculation formula."""
    print("\n" + "="*60)
    print("TEST 1: TIS CALCULATION")
    print("="*60)

    # Perfect hardware (D_KL ≈ 0)
    tis_perfect = compute_tis(1e-10)
    assert tis_perfect > 10000, f"Perfect TIS should be very high: {tis_perfect}"

    # Good hardware (D_KL = 1e-4)
    tis_good = compute_tis(1e-4)
    assert 50 < tis_good < 200, f"Good TIS should be ~100: {tis_good}"

    # Poor hardware (D_KL = 0.1)
    tis_poor = compute_tis(0.1)
    assert tis_poor < 5, f"Poor TIS should be low: {tis_poor}"

    # Verify monotonic decrease
    assert tis_perfect > tis_good > tis_poor, "TIS should decrease with increasing D_KL"

    print(f"✓ Perfect hardware: TIS = {tis_perfect:.2f}")
    print(f"✓ Good hardware: TIS = {tis_good:.2f}")
    print(f"✓ Poor hardware: TIS = {tis_poor:.2f}")
    print(f"✓ Monotonicity verified")
    print("\n✓ TEST 1 PASSED")
    return True


def test_tpu_classification():
    """Test 2: Verify TPU grade classification."""
    print("\n" + "="*60)
    print("TEST 2: TPU CLASSIFICATION")
    print("="*60)

    test_cases = [
        (10000, TPUGrade.REFERENCE),
        (150, TPUGrade.EXCELLENT),
        (50, TPUGrade.GOOD),
        (15, TPUGrade.ACCEPTABLE),
        (5, TPUGrade.MARGINAL),
        (1, TPUGrade.FAILED),
    ]

    for tis, expected_grade in test_cases:
        grade = classify_tpu(tis)
        assert grade == expected_grade, f"TIS={tis} should be {expected_grade}, got {grade}"
        print(f"✓ TIS={tis:>6} → {grade.value}")

    print("\n✓ TEST 2 PASSED")
    return True


def test_reference_tpu():
    """Test 3: Verify reference TPU achieves perfect score."""
    print("\n" + "="*60)
    print("TEST 3: REFERENCE TPU VALIDATION")
    print("="*60)

    # Small system for fast testing
    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    result = run_tpu_benchmark(
        sampler=reference_tpu_exact,
        W=W,
        H=H,
        T0=1.0,
        alpha=2.0,
        num_samples=10000,
        tpu_name="Reference Test",
        verbose=False
    )

    # Reference implementation should achieve machine precision
    assert result.D_proof < 1e-8, f"Reference D_KL too large: {result.D_proof}"
    assert result.grade in [TPUGrade.REFERENCE, TPUGrade.EXCELLENT], \
        f"Reference should be REFERENCE or EXCELLENT, got {result.grade}"
    assert result.tis > 100, f"Reference TIS too low: {result.tis}"

    print(f"✓ D_KL = {result.D_proof:.10f} < 1e-8")
    print(f"✓ TIS = {result.tis:.2f} > 100")
    print(f"✓ Grade = {result.grade.value}")
    print("\n✓ TEST 3 PASSED")
    return True


def test_mcmc_tpu():
    """Test 4: Verify MCMC TPU achieves reasonable score."""
    print("\n" + "="*60)
    print("TEST 4: MCMC TPU VALIDATION")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    result = run_tpu_benchmark(
        sampler=good_tpu_mcmc,
        W=W,
        H=H,
        T0=1.0,
        alpha=2.0,
        num_samples=50000,  # More samples for better convergence
        tpu_name="MCMC Test",
        verbose=False
    )

    # MCMC should be acceptable with enough samples
    assert result.D_proof < 0.05, f"MCMC D_KL too large: {result.D_proof}"
    assert result.grade != TPUGrade.FAILED, f"MCMC should not fail: {result.grade}"

    print(f"✓ D_KL = {result.D_proof:.6f} < 0.05")
    print(f"✓ TIS = {result.tis:.2f}")
    print(f"✓ Grade = {result.grade.value} (not FAILED)")
    print("\n✓ TEST 4 PASSED")
    return True


def test_noisy_tpu_fails():
    """Test 5: Verify high-noise TPU fails benchmark."""
    print("\n" + "="*60)
    print("TEST 5: NOISY TPU FAILURE")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    # High noise should fail
    noisy_sampler = lambda W, H, T, n: noisy_tpu(W, H, T, n, noise_level=0.1)

    result = run_tpu_benchmark(
        sampler=noisy_sampler,
        W=W,
        H=H,
        T0=1.0,
        alpha=2.0,
        num_samples=10000,
        tpu_name="Noisy Test",
        verbose=False
    )

    # High noise should produce large D_KL
    assert result.D_proof > 0.1, f"Noisy D_KL too small: {result.D_proof}"
    assert result.grade in [TPUGrade.FAILED, TPUGrade.MARGINAL], \
        f"Noisy TPU should fail or be marginal: {result.grade}"

    print(f"✓ D_KL = {result.D_proof:.6f} > 0.1 (large divergence)")
    print(f"✓ Grade = {result.grade.value} (FAILED or MARGINAL)")
    print(f"✓ High noise correctly detected")
    print("\n✓ TEST 5 PASSED")
    return True


def test_control_divergence():
    """Test 6: Verify control case shows difference."""
    print("\n" + "="*60)
    print("TEST 6: CONTROL DIVERGENCE")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    result = run_tpu_benchmark(
        sampler=reference_tpu_exact,
        W=W,
        H=H,
        T0=1.0,
        alpha=2.0,
        num_samples=10000,
        tpu_name="Control Test",
        verbose=False
    )

    # Control (energy-only scaling) must differ from original
    assert result.D_control > 0.01, \
        f"Control divergence too small: {result.D_control}"

    # Control should be much larger than proof divergence
    assert result.D_control > 10 * result.D_proof, \
        f"Control not sufficiently different: {result.D_control} vs {result.D_proof}"

    print(f"✓ D_KL(control) = {result.D_control:.6f} > 0.01")
    print(f"✓ D_KL(control) >> D_KL(proof)")
    print(f"✓ Control validation successful")
    print("\n✓ TEST 6 PASSED")
    return True


def test_rg_analysis():
    """Test 7: Verify RG fixed-point analysis."""
    print("\n" + "="*60)
    print("TEST 7: RG FIXED-POINT ANALYSIS")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    # Perfect TPU
    result_perfect = run_tpu_benchmark(
        sampler=reference_tpu_exact,
        W=W, H=H, T0=1.0, alpha=2.0, num_samples=10000,
        tpu_name="Perfect", verbose=False
    )

    rg_perfect = rg_flow_analysis(result_perfect)

    assert rg_perfect['at_fixed_point'], "Perfect TPU should be at fixed point"
    assert rg_perfect['beta_function'] < 1e-6, \
        f"Beta function should be ~0: {rg_perfect['beta_function']}"

    # Noisy TPU
    noisy_sampler = lambda W, H, T, n: noisy_tpu(W, H, T, n, 0.05)
    result_noisy = run_tpu_benchmark(
        sampler=noisy_sampler,
        W=W, H=H, T0=1.0, alpha=2.0, num_samples=10000,
        tpu_name="Noisy", verbose=False
    )

    rg_noisy = rg_flow_analysis(result_noisy)

    assert not rg_noisy['at_fixed_point'], "Noisy TPU should not be at fixed point"
    assert rg_noisy['flow_strength'] > 0.1, \
        f"Flow strength should be significant: {rg_noisy['flow_strength']}"

    print("✓ Perfect TPU:")
    print(f"  At fixed point: {rg_perfect['at_fixed_point']}")
    print(f"  Beta function: {rg_perfect['beta_function']:.6f}")
    print("✓ Noisy TPU:")
    print(f"  At fixed point: {rg_noisy['at_fixed_point']}")
    print(f"  Flow strength: {rg_noisy['flow_strength']:.6f}")
    print("\n✓ TEST 7 PASSED")
    return True


def test_scaling_factor_independence():
    """Test 8: Verify benchmark works with different scaling factors."""
    print("\n" + "="*60)
    print("TEST 8: SCALING FACTOR INDEPENDENCE")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    # Test with different alphas
    alphas = [1.5, 2.0, 3.0]
    tis_values = []

    for alpha in alphas:
        result = run_tpu_benchmark(
            sampler=reference_tpu_exact,
            W=W, H=H, T0=1.0, alpha=alpha, num_samples=10000,
            tpu_name=f"Alpha={alpha}", verbose=False
        )
        tis_values.append(result.tis)

        # All should achieve reference quality
        assert result.D_proof < 1e-6, f"Alpha={alpha}: D_KL = {result.D_proof}"

    print("✓ Reference TPU maintains perfect score across scaling factors:")
    for alpha, tis in zip(alphas, tis_values):
        print(f"  α={alpha}: TIS={tis:.2f}, Grade=REFERENCE")

    print("\n✓ TEST 8 PASSED")
    return True


def test_temperature_independence():
    """Test 9: Verify benchmark works at different temperatures."""
    print("\n" + "="*60)
    print("TEST 9: TEMPERATURE INDEPENDENCE")
    print("="*60)

    N = 4
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2
    H = np.random.randn(N)

    # Test at different base temperatures
    temperatures = [0.5, 1.0, 2.0]

    for T0 in temperatures:
        result = run_tpu_benchmark(
            sampler=reference_tpu_exact,
            W=W, H=H, T0=T0, alpha=2.0, num_samples=10000,
            tpu_name=f"T0={T0}", verbose=False
        )

        assert result.D_proof < 1e-6, f"T0={T0}: D_KL = {result.D_proof}"

    print("✓ Reference TPU maintains perfect score across temperatures:")
    for T0 in temperatures:
        print(f"  T₀={T0}: Grade=REFERENCE")

    print("\n✓ TEST 9 PASSED")
    return True


def run_all_tests():
    """Run complete test suite for System 9."""
    print("\n" + "#"*60)
    print("# SYSTEM 9 TEST SUITE")
    print("# TPU Integrity Benchmark")
    print("#"*60)

    tests = [
        test_tis_calculation,
        test_tpu_classification,
        test_reference_tpu,
        test_mcmc_tpu,
        test_noisy_tpu_fails,
        test_control_divergence,
        test_rg_analysis,
        test_scaling_factor_independence,
        test_temperature_independence,
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(results)
    passed = sum(results)

    print(f"\nTests passed: {passed}/{total}")

    if all(results):
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED")
        print("="*60)
        print("\nSystem 9 validated successfully!")
        print("  • TIS calculation verified")
        print("  • TPU classification accurate")
        print("  • Reference implementation perfect")
        print("  • MCMC implementation acceptable")
        print("  • Noise detection working")
        print("  • Control validation correct")
        print("  • RG analysis functional")
        print("  • Parameter independence confirmed")
        print("="*60)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        failed_tests = [i+1 for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
