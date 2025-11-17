"""
Test Suite for System 8: Materials Science Scale Invariance

Tests the generalized Gibbs Free Energy framework applied to:
1. Bulk metallurgy (sword forging)
2. Nanoscale semiconductor deposition
3. Scale invariance validation
4. Thermodynamic consistency
"""

import numpy as np
from materials_invariance import (
    MaterialsSystem,
    gibbs_free_energy,
    boltzmann_probability,
    compute_equilibrium_distribution,
    run_materials_invariance_test,
    create_sword_system,
    create_semiconductor_system,
    R_GAS
)


def test_gibbs_energy_calculation():
    """Test 1: Verify Gibbs free energy calculation."""
    print("\n" + "="*60)
    print("TEST 1: GIBBS FREE ENERGY CALCULATION")
    print("="*60)

    # Simple binary system
    system = MaterialsSystem(
        phases=['A', 'B'],
        G_pure=np.array([0.0, 1000.0]),
        L_matrix=np.array([[0.0, 500.0], [500.0, 0.0]]),
        regime='test'
    )

    # Test pure A: c = [1, 0]
    G_pure_A = gibbs_free_energy(np.array([1.0, 0.0]), system, T=1000.0)
    expected_A = 0.0  # Pure A has G=0 (no mixing, no excess)
    assert np.abs(G_pure_A - expected_A) < 1.0, f"Pure A: {G_pure_A} vs {expected_A}"

    # Test pure B: c = [0, 1]
    G_pure_B = gibbs_free_energy(np.array([0.0, 1.0]), system, T=1000.0)
    expected_B = 1000.0  # Pure B has G=1000
    assert np.abs(G_pure_B - expected_B) < 1.0, f"Pure B: {G_pure_B} vs {expected_B}"

    # Test 50-50 mixture
    G_mix = gibbs_free_energy(np.array([0.5, 0.5]), system, T=1000.0)
    # Expected: 0.5*0 + 0.5*1000 + RT*[0.5*ln(0.5) + 0.5*ln(0.5)] + 0.5*0.5*500
    G_ref = 500.0
    G_ideal = R_GAS * 1000.0 * (0.5 * np.log(0.5) + 0.5 * np.log(0.5))
    G_excess = 0.5 * 0.5 * 500.0
    expected_mix = G_ref + G_ideal + G_excess

    assert np.abs(G_mix - expected_mix) < 1.0, f"50-50 mix: {G_mix} vs {expected_mix}"

    print("✓ Pure component energies correct")
    print("✓ Mixing entropy calculated correctly")
    print("✓ Excess energy term validated")
    print("\n✓ TEST 1 PASSED")
    return True


def test_boltzmann_weights():
    """Test 2: Verify Boltzmann probability weights."""
    print("\n" + "="*60)
    print("TEST 2: BOLTZMANN PROBABILITY WEIGHTS")
    print("="*60)

    system = MaterialsSystem(
        phases=['A', 'B'],
        G_pure=np.array([0.0, 1000.0]),
        L_matrix=np.array([[0.0, 0.0], [0.0, 0.0]]),  # No interactions
        regime='test'
    )

    T = 1000.0

    # Boltzmann weight should be higher for lower energy states
    w_A = boltzmann_probability(np.array([1.0, 0.0]), system, T)
    w_B = boltzmann_probability(np.array([0.0, 1.0]), system, T)

    # A has lower energy (G=0) so should have higher probability
    assert w_A > w_B, f"Lower energy state should have higher weight: {w_A} vs {w_B}"

    # Check ratio matches Boltzmann factor
    # w_A/w_B = exp(-G_A/RT) / exp(-G_B/RT) = exp((G_B-G_A)/RT)
    expected_ratio = np.exp((1000.0 - 0.0) / (R_GAS * T))
    actual_ratio = w_A / w_B

    assert np.abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, \
        f"Ratio mismatch: {actual_ratio} vs {expected_ratio}"

    print("✓ Lower energy states have higher weights")
    print("✓ Boltzmann ratios correct")
    print("\n✓ TEST 2 PASSED")
    return True


def test_distribution_normalization():
    """Test 3: Verify probability distributions are normalized."""
    print("\n" + "="*60)
    print("TEST 3: PROBABILITY DISTRIBUTION NORMALIZATION")
    print("="*60)

    system = MaterialsSystem(
        phases=['A', 'B'],
        G_pure=np.array([0.0, 2000.0]),
        L_matrix=np.array([[0.0, 1000.0], [1000.0, 0.0]]),
        regime='test'
    )

    comps, probs = compute_equilibrium_distribution(system, T=800.0, n_grid=20)

    # Check normalization
    total_prob = np.sum(probs)
    assert np.abs(total_prob - 1.0) < 1e-6, f"Not normalized: {total_prob}"

    # Check all probabilities are non-negative
    assert np.all(probs >= 0), "Negative probabilities found"

    # Check we have the right number of compositions
    assert len(comps) == len(probs), "Composition-probability length mismatch"

    print(f"✓ Distribution normalized to {total_prob:.10f}")
    print(f"✓ All {len(probs)} probabilities non-negative")
    print("\n✓ TEST 3 PASSED")
    return True


def test_sword_system_validity():
    """Test 4: Validate sword forging system properties."""
    print("\n" + "="*60)
    print("TEST 4: SWORD SYSTEM VALIDATION")
    print("="*60)

    system = create_sword_system()

    # Check system properties
    assert len(system.phases) == 3, "Should have 3 phases"
    assert system.regime == 'bulk', "Should be bulk regime"
    assert system.phases == ['Austenite', 'Martensite', 'Pearlite'], "Phase names"

    # Check interaction matrix symmetry
    assert np.allclose(system.L_matrix, system.L_matrix.T), "L_matrix not symmetric"

    # Test at quenching temperature
    T_quench = 1000.0
    comps, probs = compute_equilibrium_distribution(system, T_quench, n_grid=20)

    # Should get valid distribution
    assert np.abs(np.sum(probs) - 1.0) < 1e-6, "Distribution not normalized"

    # Find dominant phase
    max_prob_idx = np.argmax(probs)
    dominant_comp = comps[max_prob_idx]

    print(f"✓ System has 3 phases: {', '.join(system.phases)}")
    print(f"✓ Regime: {system.regime}")
    print(f"✓ Dominant composition at T={T_quench}K: {dominant_comp}")
    print("\n✓ TEST 4 PASSED")
    return True


def test_semiconductor_system_validity():
    """Test 5: Validate semiconductor deposition system properties."""
    print("\n" + "="*60)
    print("TEST 5: SEMICONDUCTOR SYSTEM VALIDATION")
    print("="*60)

    system = create_semiconductor_system()

    # Check system properties
    assert len(system.phases) == 3, "Should have 3 species"
    assert system.regime == 'atomic', "Should be atomic regime"
    assert system.phases == ['Si', 'Ge', 'Vacancy'], "Species names"

    # Check that Si is reference (G=0)
    assert system.G_pure[0] == 0.0, "Si should be reference"

    # Vacancy should be high energy (unfavorable)
    assert system.G_pure[2] > system.G_pure[1], "Vacancy should cost more than dopant"

    # Test at deposition temperature
    T_dep = 800.0
    comps, probs = compute_equilibrium_distribution(system, T_dep, n_grid=20)

    # Should strongly favor Si-rich compositions (low vacancy)
    # Find average vacancy concentration
    avg_vacancy = np.sum(comps[:, 2] * probs)

    print(f"✓ System has 3 species: {', '.join(system.phases)}")
    print(f"✓ Regime: {system.regime}")
    print(f"✓ Average vacancy fraction at T={T_dep}K: {avg_vacancy:.4f}")
    print("✓ Vacancies properly penalized (high energy)")
    print("\n✓ TEST 5 PASSED")
    return True


def test_scale_invariance_sword():
    """Test 6: Verify scale invariance for sword system."""
    print("\n" + "="*60)
    print("TEST 6: SCALE INVARIANCE - SWORD SYSTEM")
    print("="*60)

    system = create_sword_system()
    results = run_materials_invariance_test(
        system=system,
        T0=1000.0,
        alpha=2.0,
        n_grid=25
    )

    assert results['proof_valid'], "Scale invariance proof failed"
    assert results['control_valid'], "Control validation failed"

    print(f"\n✓ D_KL(proof) = {results['D_proof']:.6f} < 1e-8")
    print(f"✓ D_KL(control) = {results['D_control']:.6f} > 0.01")
    print("\n✓ TEST 6 PASSED")
    return True


def test_scale_invariance_semiconductor():
    """Test 7: Verify scale invariance for semiconductor system."""
    print("\n" + "="*60)
    print("TEST 7: SCALE INVARIANCE - SEMICONDUCTOR SYSTEM")
    print("="*60)

    system = create_semiconductor_system()
    results = run_materials_invariance_test(
        system=system,
        T0=800.0,
        alpha=2.0,
        n_grid=25
    )

    assert results['proof_valid'], "Scale invariance proof failed"
    assert results['control_valid'], "Control validation failed"

    print(f"\n✓ D_KL(proof) = {results['D_proof']:.6f} < 1e-8")
    print(f"✓ D_KL(control) = {results['D_control']:.6f} > 0.01")
    print("\n✓ TEST 7 PASSED")
    return True


def test_temperature_dependence():
    """Test 8: Verify proper temperature dependence."""
    print("\n" + "="*60)
    print("TEST 8: TEMPERATURE DEPENDENCE")
    print("="*60)

    system = MaterialsSystem(
        phases=['A', 'B'],
        G_pure=np.array([0.0, 5000.0]),
        L_matrix=np.array([[0.0, 0.0], [0.0, 0.0]]),
        regime='test'
    )

    # At low T, should strongly favor A (lower energy)
    # At high T, entropy dominates, more mixing

    T_low = 100.0
    T_high = 2000.0

    _, probs_low = compute_equilibrium_distribution(system, T_low, n_grid=20)
    _, probs_high = compute_equilibrium_distribution(system, T_high, n_grid=20)

    # At low T, distribution should be more peaked (low entropy)
    # Measure via entropy of the distribution itself
    def dist_entropy(p):
        p_safe = p[p > 1e-12]
        return -np.sum(p_safe * np.log(p_safe))

    S_low = dist_entropy(probs_low)
    S_high = dist_entropy(probs_high)

    assert S_high > S_low, f"Higher T should have higher entropy: {S_high} vs {S_low}"

    print(f"✓ Distribution entropy at T={T_low}K: {S_low:.4f}")
    print(f"✓ Distribution entropy at T={T_high}K: {S_high:.4f}")
    print("✓ Higher temperature increases accessible states")
    print("\n✓ TEST 8 PASSED")
    return True


def run_all_tests():
    """Run complete test suite for System 8."""
    print("\n" + "#"*60)
    print("# SYSTEM 8 TEST SUITE")
    print("# Materials Science Scale Invariance")
    print("#"*60)

    tests = [
        test_gibbs_energy_calculation,
        test_boltzmann_weights,
        test_distribution_normalization,
        test_sword_system_validity,
        test_semiconductor_system_validity,
        test_scale_invariance_sword,
        test_scale_invariance_semiconductor,
        test_temperature_dependence
    ]

    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH ERROR: {e}")
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
        print("\nSystem 8 validated successfully!")
        print("  • Gibbs free energy calculations correct")
        print("  • Boltzmann statistics verified")
        print("  • Sword system (bulk metallurgy) validated")
        print("  • Semiconductor system (nanoscale) validated")
        print("  • Scale invariance confirmed for both regimes")
        print("="*60)
    else:
        print("\n⚠️  SOME TESTS FAILED")
        failed_tests = [i+1 for i, r in enumerate(results) if not r]
        print(f"Failed tests: {failed_tests}")

    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)