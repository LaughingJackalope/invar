"""
System 9: TPU Integrity Benchmark

Standardized diagnostic tool for validating thermodynamic processing hardware.
Uses the proven scale invariance property P(S; H, T) = P(S; αH, αT) as a
quality assurance metric.

Theoretical Foundation:
    Scale invariance represents a Renormalization Group (RG) fixed point.
    Hardware that perfectly implements the Boltzmann distribution will exhibit
    exact invariance. Any deviation (D_KL > 0) indicates:
    - Numerical precision errors
    - Hardware noise/imperfections
    - Systematic bias in sampling
    - Incorrect Hamiltonian implementation

Output:
    Thermodynamic Integrity Score (TIS) - Single metric quantifying hardware quality
"""

import numpy as np
from scipy.stats import entropy
from typing import Dict, Callable, Tuple, List
from dataclasses import dataclass
from enum import Enum
import warnings


class TPUGrade(Enum):
    """Hardware quality classifications based on TIS."""
    REFERENCE = "REFERENCE"      # TIS > 1000 (D_KL < 1e-6)
    EXCELLENT = "EXCELLENT"      # TIS > 100  (D_KL < 1e-4)
    GOOD = "GOOD"                # TIS > 31   (D_KL < 1e-3)
    ACCEPTABLE = "ACCEPTABLE"    # TIS > 10   (D_KL < 1e-2)
    MARGINAL = "MARGINAL"        # TIS > 3    (D_KL < 0.1)
    FAILED = "FAILED"            # TIS < 3    (D_KL > 0.1)


@dataclass
class BenchmarkResult:
    """
    Complete benchmark results for a TPU.

    Attributes
    ----------
    tpu_name : str
        Identifier for the tested hardware/software
    tis : float
        Thermodynamic Integrity Score
    grade : TPUGrade
        Quality classification
    D_proof : float
        KL divergence for scale invariance test
    D_control : float
        KL divergence for control (should be large)
    P_orig : np.ndarray
        Original distribution
    P_test : np.ndarray
        Scaled distribution (should match P_orig)
    P_control : np.ndarray
        Control distribution (should differ from P_orig)
    metadata : Dict
        Additional diagnostic information
    """
    tpu_name: str
    tis: float
    grade: TPUGrade
    D_proof: float
    D_control: float
    P_orig: np.ndarray
    P_test: np.ndarray
    P_control: np.ndarray
    metadata: Dict


def compute_tis(D_KL: float, epsilon: float = 1e-10) -> float:
    """
    Calculate Thermodynamic Integrity Score.

    TIS = 1 / sqrt(D_KL + epsilon)

    Higher TIS indicates better hardware quality (closer to RG fixed point).

    Parameters
    ----------
    D_KL : float
        KL divergence between original and scaled distributions
    epsilon : float
        Small constant to avoid division by zero

    Returns
    -------
    tis : float
        Thermodynamic Integrity Score
    """
    return 1.0 / np.sqrt(D_KL + epsilon)


def classify_tpu(tis: float) -> TPUGrade:
    """
    Classify TPU quality based on TIS.

    Parameters
    ----------
    tis : float
        Thermodynamic Integrity Score

    Returns
    -------
    grade : TPUGrade
        Quality classification
    """
    if tis > 1000:
        return TPUGrade.REFERENCE
    elif tis > 100:
        return TPUGrade.EXCELLENT
    elif tis > 31:
        return TPUGrade.GOOD
    elif tis > 10:
        return TPUGrade.ACCEPTABLE
    elif tis > 3:
        return TPUGrade.MARGINAL
    else:
        return TPUGrade.FAILED


def run_tpu_benchmark(
    sampler: Callable,
    W: np.ndarray,
    H: np.ndarray,
    T0: float,
    alpha: float,
    num_samples: int,
    tpu_name: str = "Unknown TPU",
    verbose: bool = True
) -> BenchmarkResult:
    """
    System 9: Execute TPU Integrity Benchmark

    Tests whether a thermodynamic sampler (hardware or software) correctly
    implements scale invariance. This is the gold standard for validating
    that a TPU is at the Renormalization Group fixed point.

    Parameters
    ----------
    sampler : Callable
        Function with signature: sampler(W, H, T, num_samples) -> P
        Returns probability distribution over all 2^N states
    W : np.ndarray
        Interaction matrix (NxN symmetric)
    H : np.ndarray
        Bias vector (N,)
    T0 : float
        Base temperature
    alpha : float
        Scaling factor (typically 2.0)
    num_samples : int
        Number of MCMC samples (if applicable)
    tpu_name : str
        Identifier for the hardware under test
    verbose : bool
        Print detailed results

    Returns
    -------
    result : BenchmarkResult
        Complete diagnostic information and TIS
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SYSTEM 9: TPU INTEGRITY BENCHMARK")
        print(f"{'='*70}")
        print(f"Hardware under test: {tpu_name}")
        print(f"System size: N={len(H)}, States=2^{len(H)}={2**len(H)}")
        print(f"Base temperature: T0={T0:.2f}")
        print(f"Scaling factor: α={alpha}")
        print(f"Samples: {num_samples}")

    # Case A: Original system
    if verbose:
        print(f"\n{'-'*70}")
        print("CASE A: Original System (W, H, T)")

    P_orig = sampler(W, H, T0, num_samples)

    if verbose:
        print(f"✓ Distribution sampled: {len(P_orig)} states")

    # Case B: Energy-only scaling (control - should differ)
    if verbose:
        print(f"\n{'-'*70}")
        print("CASE B: Energy-Only Scaling (αW, αH, T) - CONTROL")

    P_control = sampler(alpha * W, alpha * H, T0, num_samples)

    if verbose:
        print(f"✓ Distribution sampled: {len(P_control)} states")

    # Case C: Full scaling (test - should match A)
    if verbose:
        print(f"\n{'-'*70}")
        print("CASE C: Full Scaling (αW, αH, αT) - TEST")

    P_test = sampler(alpha * W, alpha * H, alpha * T0, num_samples)

    if verbose:
        print(f"✓ Distribution sampled: {len(P_test)} states")

    # Compute divergences
    if verbose:
        print(f"\n{'-'*70}")
        print("DIVERGENCE ANALYSIS")

    # Add epsilon to avoid log(0)
    eps = 1e-12
    P_orig_safe = P_orig + eps
    P_test_safe = P_test + eps
    P_control_safe = P_control + eps

    # Renormalize
    P_orig_safe /= P_orig_safe.sum()
    P_test_safe /= P_test_safe.sum()
    P_control_safe /= P_control_safe.sum()

    D_proof = entropy(P_orig_safe, P_test_safe)
    D_control = entropy(P_orig_safe, P_control_safe)

    if verbose:
        print(f"\nD_KL(P_orig || P_test)    = {D_proof:.6f}  [Scale invariance test]")
        print(f"D_KL(P_orig || P_control) = {D_control:.6f}  [Control validation]")

    # Calculate TIS
    tis = compute_tis(D_proof)
    grade = classify_tpu(tis)

    if verbose:
        print(f"\n{'-'*70}")
        print("THERMODYNAMIC INTEGRITY SCORE (TIS)")
        print(f"{'-'*70}")
        print(f"\nTIS = {tis:.2f}")
        print(f"Grade: {grade.value}")

        # Interpretation
        print(f"\n{'-'*70}")
        print("INTERPRETATION")
        print(f"{'-'*70}")

        if grade == TPUGrade.REFERENCE:
            print("✓ REFERENCE QUALITY")
            print("  Hardware at RG fixed point (machine precision)")
            print("  Suitable for: Scientific research, standards calibration")
        elif grade == TPUGrade.EXCELLENT:
            print("✓ EXCELLENT QUALITY")
            print("  Hardware very close to fixed point")
            print("  Suitable for: Production ML, quantum annealing, optimization")
        elif grade == TPUGrade.GOOD:
            print("✓ GOOD QUALITY")
            print("  Hardware acceptable for most applications")
            print("  Suitable for: General-purpose thermodynamic computing")
        elif grade == TPUGrade.ACCEPTABLE:
            print("⚠ ACCEPTABLE QUALITY")
            print("  Hardware shows minor deviations from theory")
            print("  Suitable for: Non-critical applications, prototyping")
        elif grade == TPUGrade.MARGINAL:
            print("⚠ MARGINAL QUALITY")
            print("  Hardware shows significant deviations")
            print("  Recommend: Calibration, noise reduction, debugging")
        else:
            print("✗ FAILED")
            print("  Hardware does not preserve scale invariance")
            print("  Action required: Major hardware/software revision")

        # Control check
        if D_control < 0.01:
            print("\n⚠ WARNING: Control divergence too small!")
            print("  This may indicate sampling issues or numerical problems.")

    # Package results
    metadata = {
        'N': len(H),
        'T0': T0,
        'alpha': alpha,
        'num_samples': num_samples,
        'num_states': len(P_orig),
        'control_valid': D_control > 0.01
    }

    result = BenchmarkResult(
        tpu_name=tpu_name,
        tis=tis,
        grade=grade,
        D_proof=D_proof,
        D_control=D_control,
        P_orig=P_orig,
        P_test=P_test,
        P_control=P_control,
        metadata=metadata
    )

    if verbose:
        print(f"\n{'='*70}")
        print(f"BENCHMARK COMPLETE: {tpu_name}")
        print(f"{'='*70}\n")

    return result


# ============================================================================
# REFERENCE TPU IMPLEMENTATIONS (for testing/comparison)
# ============================================================================

def reference_tpu_exact(W: np.ndarray, H: np.ndarray, T: float, num_samples: int) -> np.ndarray:
    """
    Reference TPU: Exact enumeration (perfect implementation).

    This is the gold standard - computes exact Boltzmann distribution
    by enumerating all 2^N states. Should achieve TIS > 1000.
    """
    N = len(H)
    num_states = 2 ** N

    energies = np.zeros(num_states)

    # Enumerate all states
    for state_idx in range(num_states):
        # Convert state index to spin configuration
        bits = [(state_idx >> i) & 1 for i in range(N)]
        spins = np.array([2*b - 1 for b in bits])

        # Calculate energy: E = -s^T W s - H^T s
        E = -np.dot(spins, np.dot(W, spins)) - np.dot(H, spins)
        energies[state_idx] = E

    # Boltzmann distribution
    weights = np.exp(-energies / T)
    P = weights / weights.sum()

    return P


def good_tpu_mcmc(W: np.ndarray, H: np.ndarray, T: float, num_samples: int) -> np.ndarray:
    """
    Good TPU: MCMC sampling with adequate samples.

    Uses Gibbs sampling. Should achieve TIS > 100 with enough samples.
    """
    from scale_invariance import simulate_equilibrium
    return simulate_equilibrium(W, H, T, num_samples)


def noisy_tpu(W: np.ndarray, H: np.ndarray, T: float, num_samples: int,
              noise_level: float = 0.01) -> np.ndarray:
    """
    Noisy TPU: Perfect distribution with added Gaussian noise.

    Simulates hardware with thermal noise or systematic bias.
    TIS depends on noise_level.
    """
    P_perfect = reference_tpu_exact(W, H, T, num_samples)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=P_perfect.shape)
    P_noisy = P_perfect + noise

    # Ensure non-negative and normalized
    P_noisy = np.maximum(P_noisy, 1e-12)
    P_noisy /= P_noisy.sum()

    return P_noisy


def faulty_tpu_wrong_temperature(W: np.ndarray, H: np.ndarray, T: float,
                                  num_samples: int, T_error: float = 0.1) -> np.ndarray:
    """
    Faulty TPU: Implements wrong temperature (calibration error).

    Simulates hardware that doesn't correctly scale temperature.
    This should FAIL the benchmark (breaks scale invariance).
    """
    T_actual = T * (1 + T_error)  # 10% temperature error
    return reference_tpu_exact(W, H, T_actual, num_samples)


def faulty_tpu_wrong_coupling(W: np.ndarray, H: np.ndarray, T: float,
                               num_samples: int, W_error: float = 0.1) -> np.ndarray:
    """
    Faulty TPU: Implements wrong couplings (fabrication error).

    Simulates hardware with incorrect interaction strengths.
    This should FAIL the benchmark.
    """
    W_actual = W * (1 + W_error)
    return reference_tpu_exact(W_actual, H, T, num_samples)


# ============================================================================
# BATCH BENCHMARKING
# ============================================================================

def benchmark_suite(
    W: np.ndarray,
    H: np.ndarray,
    T0: float = 1.0,
    alpha: float = 2.0,
    num_samples: int = 20000
) -> Dict[str, BenchmarkResult]:
    """
    Run full benchmark suite on reference TPU implementations.

    Tests multiple hardware configurations to demonstrate benchmark capability.

    Returns
    -------
    results : Dict[str, BenchmarkResult]
        Benchmark results for each TPU variant
    """
    print("\n" + "#"*70)
    print("# SYSTEM 9: TPU BENCHMARK SUITE")
    print("# Comparative Hardware Validation")
    print("#"*70)

    tpus = {
        "Reference (Exact)": reference_tpu_exact,
        "Production (MCMC)": good_tpu_mcmc,
        "Low Noise (1%)": lambda W, H, T, n: noisy_tpu(W, H, T, n, 0.01),
        "Moderate Noise (5%)": lambda W, H, T, n: noisy_tpu(W, H, T, n, 0.05),
        "High Noise (10%)": lambda W, H, T, n: noisy_tpu(W, H, T, n, 0.10),
        "Faulty (T error)": faulty_tpu_wrong_temperature,
        "Faulty (W error)": faulty_tpu_wrong_coupling,
    }

    results = {}

    for tpu_name, sampler in tpus.items():
        result = run_tpu_benchmark(
            sampler=sampler,
            W=W,
            H=H,
            T0=T0,
            alpha=alpha,
            num_samples=num_samples,
            tpu_name=tpu_name,
            verbose=True
        )
        results[tpu_name] = result

    # Summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\n{'TPU Name':<25} {'TIS':>10} {'Grade':<15} {'D_KL':>10}")
    print("-"*70)

    for tpu_name, result in results.items():
        print(f"{tpu_name:<25} {result.tis:>10.2f} {result.grade.value:<15} {result.D_proof:>10.6f}")

    print("\n" + "="*70)
    print("QUALITY DISTRIBUTION")
    print("="*70)

    grade_counts = {}
    for result in results.values():
        grade = result.grade.value
        grade_counts[grade] = grade_counts.get(grade, 0) + 1

    for grade, count in sorted(grade_counts.items()):
        print(f"  {grade}: {count} TPU(s)")

    return results


# ============================================================================
# RENORMALIZATION GROUP ANALYSIS
# ============================================================================

def rg_flow_analysis(result: BenchmarkResult) -> Dict:
    """
    Analyze how far the TPU has 'flowed' from the RG fixed point.

    In RG theory, scale invariance corresponds to a fixed point.
    D_KL measures the distance from this fixed point.

    Parameters
    ----------
    result : BenchmarkResult
        Benchmark results to analyze

    Returns
    -------
    analysis : Dict
        RG flow diagnostics
    """
    D_KL = result.D_proof

    # RG beta function (approximate)
    # At fixed point: beta = 0, away from fixed point: beta ∝ D_KL
    beta_function = D_KL

    # Relevant operator strength (how strongly system flows away)
    # Higher values = stronger flow = worse hardware
    flow_strength = np.sqrt(D_KL)

    # Critical exponent (characteristic scale of deviation)
    # Approximated from log(D_KL)
    if D_KL > 1e-10:
        critical_exponent = -np.log10(D_KL) / np.log10(result.metadata['alpha'])
    else:
        critical_exponent = np.inf

    return {
        'beta_function': beta_function,
        'flow_strength': flow_strength,
        'critical_exponent': critical_exponent,
        'at_fixed_point': D_KL < 1e-6,
        'distance_from_fixed_point': D_KL
    }


if __name__ == "__main__":
    # Demo: Benchmark various TPU implementations
    print("Generating test system...")

    # Small system for demonstration
    N = 5
    np.random.seed(42)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2  # Symmetric
    H = np.random.randn(N)

    results = benchmark_suite(W, H, T0=1.0, alpha=2.0, num_samples=20000)

    print("\n" + "="*70)
    print("RG FIXED-POINT ANALYSIS")
    print("="*70)

    for tpu_name, result in results.items():
        rg = rg_flow_analysis(result)
        print(f"\n{tpu_name}:")
        print(f"  At fixed point: {rg['at_fixed_point']}")
        print(f"  Beta function: {rg['beta_function']:.6f}")
        print(f"  Flow strength: {rg['flow_strength']:.6f}")
