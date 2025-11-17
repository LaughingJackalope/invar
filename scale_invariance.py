"""
Scale Invariance Testing Framework for Boltzmann Machines
Implements three systems for rigorous proof of energy-temperature scaling.

Phase 1 Update: Now supports pluggable sampling backends via sampler_interface.
Maintains backward compatibility with original API.
"""

import numpy as np
from scipy.stats import entropy
from typing import Tuple, Dict, Optional
import warnings


def simulate_equilibrium(
    W: np.ndarray,
    H: np.ndarray,
    T: float,
    num_samples: int = 10000
) -> np.ndarray:
    """
    System 1: Core Simulation Runner
    
    Simulates a Boltzmann machine to equilibrium and returns probability distribution.
    
    Parameters
    ----------
    W : np.ndarray
        Interaction matrix (NxN symmetric)
    H : np.ndarray
        Bias vector (N,)
    T : float
        Temperature parameter
    num_samples : int
        Number of samples to generate for distribution estimation
        
    Returns
    -------
    P : np.ndarray
        Probability distribution over all 2^N possible states
    """
    N = len(H)
    
    # Initialize with random state
    state = np.random.choice([-1, 1], size=N)
    
    # Burn-in period (allow system to reach equilibrium)
    burn_in = num_samples // 4
    
    # Store samples
    samples = []
    
    # Gibbs sampling
    for step in range(burn_in + num_samples):
        # Select random neuron to update
        i = np.random.randint(N)
        
        # Calculate local field
        h_i = H[i] + np.dot(W[i], state)
        
        # Compute activation probability
        p_activate = 1.0 / (1.0 + np.exp(-2 * h_i / T))
        
        # Update state
        state[i] = 1 if np.random.rand() < p_activate else -1
        
        # Collect samples after burn-in
        if step >= burn_in:
            samples.append(state.copy())
    
    # Convert samples to state indices and count frequencies
    samples = np.array(samples)
    
    # Map states to indices: convert {-1,1}^N to {0,1,...,2^N-1}
    # Binary representation: -1 -> 0, 1 -> 1
    binary_samples = ((samples + 1) // 2).astype(int)
    
    # Convert to decimal indices
    powers = 2 ** np.arange(N)[::-1]
    indices = binary_samples @ powers
    
    # Count frequencies
    counts = np.bincount(indices, minlength=2**N)
    
    # Normalize to probability distribution
    P = counts / num_samples
    
    return P


def run_scale_invariance_test(
    N: int,
    alpha: float,
    T0: float = 1.0,
    num_samples: int = 10000,
    seed: int = None,
    sampler = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    System 2: Experimental Setup
    
    Runs three experimental cases to test scale invariance hypothesis.
    
    Parameters
    ----------
    N : int
        System size (number of neurons)
    alpha : float
        Scaling factor (e.g., 2.0 for doubling)
    T0 : float
        Base temperature
    num_samples : int
        Samples per simulation
    seed : int, optional
        Random seed for reproducibility
    sampler : BoltzmannSampler, optional
        Sampling backend (default: NumpySampler)
        Can be NumpySampler or ThrmlSampler
        
    Returns
    -------
    P_orig : np.ndarray
        Case A: Original system probability distribution
    P_scaled_E : np.ndarray
        Case B: Scaled energy only (control)
    P_test : np.ndarray
        Case C: Scaled energy with scaled temperature (invariance test)
    params : Dict
        Dictionary containing all experimental parameters
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Use default sampler if none provided
    if sampler is None:
        from numpy_sampler import NumpySampler
        sampler = NumpySampler()
    
    # Generate random base parameters
    # W: symmetric interaction matrix
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)  # No self-connections
    
    # H: bias vector
    H = np.random.randn(N)
    
    # Case A: Original system (Baseline)
    print(f"Running Case A: Original (W, H, T={T0})...")
    P_orig = sampler.sample_distribution(W, H, T0, num_samples)
    
    # Case B: Scaled energy only (Control - should break distribution)
    W_scaled = alpha * W
    H_scaled = alpha * H
    print(f"Running Case B: Scaled Energy (α·W, α·H, T={T0})...")
    P_scaled_E = sampler.sample_distribution(W_scaled, H_scaled, T0, num_samples)
    
    # Case C: Scaled energy with scaled temperature (Invariance Test)
    T_scaled = alpha * T0
    print(f"Running Case C: Invariant Test (α·W, α·H, T={T_scaled})...")
    P_test = sampler.sample_distribution(W_scaled, H_scaled, T_scaled, num_samples)
    
    # Package parameters for reference
    params = {
        'N': N,
        'alpha': alpha,
        'T0': T0,
        'T_scaled': T_scaled,
        'W': W,
        'H': H,
        'W_scaled': W_scaled,
        'H_scaled': H_scaled,
        'num_samples': num_samples,
        'seed': seed
    }
    
    return P_orig, P_scaled_E, P_test, params


def quantify_divergence(P1: np.ndarray, P2: np.ndarray) -> float:
    """
    System 3: Proof Metric Calculator
    
    Calculates Kullback-Leibler divergence to quantify distribution difference.
    
    Parameters
    ----------
    P1 : np.ndarray
        Reference probability distribution
    P2 : np.ndarray
        Comparison probability distribution
        
    Returns
    -------
    D_KL : float
        KL divergence D_KL(P1 || P2)
        
    Notes
    -----
    Adds small epsilon to handle zero probabilities gracefully.
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    P1_safe = P1 + epsilon
    P2_safe = P2 + epsilon
    
    # Renormalize
    P1_safe = P1_safe / P1_safe.sum()
    P2_safe = P2_safe / P2_safe.sum()
    
    # Calculate KL divergence
    D_KL = entropy(P1_safe, P2_safe)
    
    return D_KL


def run_full_experiment(
    N: int = 6,
    alpha: float = 2.0,
    T0: float = 1.0,
    num_samples: int = 10000,
    seed: int = 42,
    sampler = None
) -> Dict:
    """
    Complete experimental pipeline with proof validation.
    
    Parameters
    ----------
    sampler : BoltzmannSampler, optional
        Sampling backend (default: NumpySampler)
    
    Returns dictionary with all results and proof metrics.
    """
    print("=" * 60)
    print("SCALE INVARIANCE PROOF EXPERIMENT")
    print("=" * 60)
    print(f"Configuration: N={N}, α={alpha}, T₀={T0}")
    print(f"Samples per case: {num_samples}")
    print("=" * 60)
    print()
    
    # Run three experimental cases
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N, alpha, T0, num_samples, seed, sampler=sampler
    )
    
    print()
    print("=" * 60)
    print("COMPUTING PROOF METRICS")
    print("=" * 60)
    
    # Calculate divergences
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)
    
    print(f"\n📊 RESULTS:")
    print(f"  D_KL(P_orig || P_test)      = {D_proof:.6f}  [MUST BE ≈ 0]")
    print(f"  D_KL(P_orig || P_scaled_E)  = {D_control:.6f}  [MUST BE >> 0]")
    print()
    
    # Proof validation
    threshold_proof = 0.01  # Tight threshold for invariance
    threshold_control = 0.1  # Control must show significant difference
    
    proof_valid = D_proof < threshold_proof
    control_valid = D_control > threshold_control
    
    print("=" * 60)
    print("PROOF VALIDATION")
    print("=" * 60)
    print(f"✓ Invariance holds (D_KL < {threshold_proof}): {'PASS' if proof_valid else 'FAIL'}")
    print(f"✓ Control differs (D_KL > {threshold_control}): {'PASS' if control_valid else 'FAIL'}")
    print()
    
    if proof_valid and control_valid:
        print("🎉 PROOF SUCCESSFUL: Scale invariance property confirmed!")
    else:
        print("⚠️  PROOF INCONCLUSIVE: Adjust parameters or increase samples")
    
    print("=" * 60)
    
    return {
        'P_orig': P_orig,
        'P_scaled_E': P_scaled_E,
        'P_test': P_test,
        'D_proof': D_proof,
        'D_control': D_control,
        'proof_valid': proof_valid,
        'control_valid': control_valid,
        'params': params
    }


if __name__ == "__main__":
    # Run the experiment
    results = run_full_experiment(
        N=6,           # Small system for tractability
        alpha=2.0,     # Double the energy scale
        T0=1.0,        # Base temperature
        num_samples=10000,
        seed=42
    )
