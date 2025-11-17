"""
System 7: Statistical Noise Floor Calculator

Establishes the minimum expected KL divergence due to sampling error alone.
This provides the critical threshold for hardware validation.

Theoretical Foundation:
----------------------
Even sampling from IDENTICAL distributions produces non-zero KL divergence
due to finite sample size. We must quantify this "noise floor" to determine
if hardware results are "invariant enough."

Method:
1. Sample from same system twice with different seeds: P_A1, P_A2
2. Compute D_KL(P_A1 || P_A2)
3. Repeat 100+ times to get distribution of D_KL_noise
4. Use mean + 2σ as conservative threshold for hardware validation
"""

import numpy as np
from typing import Dict, List, Tuple
from scale_invariance import simulate_equilibrium, quantify_divergence


def compute_noise_floor(
    W: np.ndarray,
    H: np.ndarray,
    T: float,
    num_samples: int,
    num_trials: int = 100,
    base_seed: int = 42
) -> Dict:
    """
    Compute statistical noise floor for given system parameters.
    
    Runs multiple trials where the SAME system is sampled twice with
    different random seeds, measuring the KL divergence between these
    "identical" distributions.
    
    Parameters
    ----------
    W : np.ndarray
        Interaction matrix
    H : np.ndarray
        Bias vector
    T : float
        Temperature
    num_samples : int
        Samples per distribution
    num_trials : int
        Number of trial pairs
    base_seed : int
        Base random seed
        
    Returns
    -------
    results : Dict
        Contains noise floor statistics
    """
    divergences = []
    
    print(f"Computing noise floor over {num_trials} trials...")
    print(f"Each trial: {num_samples} samples per distribution")
    print()
    
    for trial in range(num_trials):
        # Sample SAME system twice with different seeds
        seed1 = base_seed + trial * 2
        seed2 = base_seed + trial * 2 + 1
        
        np.random.seed(seed1)
        P1 = simulate_equilibrium(W, H, T, num_samples)
        
        np.random.seed(seed2)
        P2 = simulate_equilibrium(W, H, T, num_samples)
        
        # Compute KL divergence
        D_KL = quantify_divergence(P1, P2)
        divergences.append(D_KL)
        
        if (trial + 1) % 20 == 0:
            print(f"  Progress: {trial + 1}/{num_trials} trials")
    
    divergences = np.array(divergences)
    
    # Statistics
    mean = np.mean(divergences)
    std = np.std(divergences)
    median = np.median(divergences)
    q95 = np.percentile(divergences, 95)
    max_val = np.max(divergences)
    
    # Conservative threshold: mean + 2σ (covers ~95% of noise)
    threshold_conservative = mean + 2 * std
    
    return {
        'divergences': divergences,
        'mean': mean,
        'std': std,
        'median': median,
        'q95': q95,
        'max': max_val,
        'threshold_conservative': threshold_conservative,
        'num_trials': num_trials,
        'num_samples': num_samples
    }


def run_noise_floor_analysis(
    N: int,
    num_samples: int = 10000,
    num_trials: int = 100,
    seed: int = 42
) -> Dict:
    """
    System 7: Complete noise floor analysis.
    
    Establishes the statistical significance threshold for scale invariance tests.
    
    Parameters
    ----------
    N : int
        System size
    num_samples : int
        Samples per distribution
    num_trials : int
        Number of trial pairs
    seed : int
        Base random seed
        
    Returns
    -------
    results : Dict
        Contains noise floor analysis and recommended thresholds
    """
    np.random.seed(seed)
    
    # Generate random test system
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    T = 1.0
    
    print("=" * 60)
    print("SYSTEM 7: STATISTICAL NOISE FLOOR ANALYSIS")
    print("=" * 60)
    print(f"Configuration: N={N}, samples={num_samples}, trials={num_trials}")
    print(f"Test: Measure D_KL between identical distributions")
    print("=" * 60)
    print()
    
    # Compute noise floor
    results = compute_noise_floor(W, H, T, num_samples, num_trials, seed)
    
    print()
    print("=" * 60)
    print("NOISE FLOOR STATISTICS")
    print("=" * 60)
    print(f"\n📊 KL DIVERGENCE DISTRIBUTION (Identical Systems):")
    print(f"  Mean:       {results['mean']:.6f}")
    print(f"  Std Dev:    {results['std']:.6f}")
    print(f"  Median:     {results['median']:.6f}")
    print(f"  95th %ile:  {results['q95']:.6f}")
    print(f"  Maximum:    {results['max']:.6f}")
    print()
    print(f"🎯 RECOMMENDED THRESHOLDS:")
    print(f"  Conservative (μ + 2σ): {results['threshold_conservative']:.6f}")
    print(f"  Aggressive (95th %):   {results['q95']:.6f}")
    print()
    
    # Interpretation
    print("=" * 60)
    print("INTERPRETATION FOR HARDWARE VALIDATION")
    print("=" * 60)
    print()
    print("For scale invariance test to be statistically significant:")
    print(f"  D_KL(P_orig || P_test) < {results['threshold_conservative']:.6f}  [Conservative]")
    print(f"  D_KL(P_orig || P_test) < {results['q95']:.6f}  [Aggressive]")
    print()
    print("Any hardware result ABOVE this threshold indicates:")
    print("  ⚠️  Either scale invariance is violated")
    print("  ⚠️  Or hardware has systematic bias")
    print()
    print("Current System 1-3 result:")
    print(f"  D_KL ≈ 0.007364 (from demo)")
    
    if 0.007364 < results['threshold_conservative']:
        print(f"  ✓ BELOW noise floor ({results['threshold_conservative']:.6f})")
        print("  ✓ Scale invariance is STATISTICALLY SIGNIFICANT")
    else:
        print(f"  ⚠️  ABOVE noise floor ({results['threshold_conservative']:.6f})")
        print("  ⚠️  May be within sampling error—increase num_samples")
    
    print()
    print("=" * 60)
    
    # Store for reference
    results['params'] = {
        'W': W,
        'H': H,
        'T': T,
        'N': N
    }
    
    return results


def run_multi_scale_noise_floor(
    N: int,
    sample_sizes: List[int],
    num_trials: int = 50,
    seed: int = 42
) -> Dict:
    """
    Analyze how noise floor changes with sample size.
    
    Critical for determining required sample sizes for hardware validation.
    """
    print("=" * 60)
    print("MULTI-SCALE NOISE FLOOR ANALYSIS")
    print("=" * 60)
    print(f"System size: N={N}")
    print(f"Testing sample sizes: {sample_sizes}")
    print("=" * 60)
    print()
    
    np.random.seed(seed)
    
    # Fixed test system
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    T = 1.0
    
    results_by_size = {}
    
    for num_samples in sample_sizes:
        print(f"\n--- Testing {num_samples} samples ---")
        results = compute_noise_floor(W, H, T, num_samples, num_trials, seed)
        results_by_size[num_samples] = results
        print(f"  Mean D_KL: {results['mean']:.6f}")
        print(f"  Threshold: {results['threshold_conservative']:.6f}")
    
    # Summary
    print()
    print("=" * 60)
    print("SUMMARY: NOISE FLOOR vs SAMPLE SIZE")
    print("=" * 60)
    print()
    print("| Samples | Mean D_KL | Threshold (μ+2σ) | Status |")
    print("|---------|-----------|------------------|--------|")
    
    for num_samples in sample_sizes:
        r = results_by_size[num_samples]
        status = "✓ Good" if r['threshold_conservative'] < 0.01 else "⚠️ High"
        print(f"| {num_samples:7d} | {r['mean']:9.6f} | {r['threshold_conservative']:16.6f} | {status} |")
    
    print()
    print("Recommendation:")
    # Find smallest sample size with threshold < 0.01
    for num_samples in sample_sizes:
        if results_by_size[num_samples]['threshold_conservative'] < 0.01:
            print(f"  Use ≥ {num_samples} samples for rigorous validation")
            break
    else:
        print(f"  Use > {sample_sizes[-1]} samples (all tested sizes have high noise)")
    
    print()
    print("=" * 60)
    
    return results_by_size


if __name__ == "__main__":
    # Run noise floor analysis
    print("=== SINGLE-SCALE ANALYSIS ===\n")
    results = run_noise_floor_analysis(
        N=5,
        num_samples=20000,
        num_trials=100,
        seed=42
    )
    
    print("\n\n")
    print("=== MULTI-SCALE ANALYSIS ===\n")
    multi_results = run_multi_scale_noise_floor(
        N=5,
        sample_sizes=[5000, 10000, 20000, 50000],
        num_trials=50,
        seed=42
    )
