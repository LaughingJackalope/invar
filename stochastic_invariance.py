"""
System 6: Stochastic Dynamic Invariance Proof

Proves that MCMC transition probabilities are scale-invariant under 
simultaneous E→αE, T→αT scaling.

This fills the gap between System 4 (deterministic theory) and real hardware 
(stochastic dynamics).

Theoretical Foundation:
----------------------
Metropolis-Hastings acceptance probability:
A(S → S') = min(1, exp(-ΔE/T))

Under scaling E→αE, T→αT:
A'(S → S') = min(1, exp(-αΔE/(αT))) = min(1, exp(-ΔE/T)) = A(S → S')

Result: Transition probabilities are EXACTLY invariant.
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class TransitionStats:
    """Statistics for a single transition between states."""
    state_from: np.ndarray
    state_to: np.ndarray
    delta_E: float
    acceptance_prob: float
    accepted_count: int
    total_attempts: int


def compute_energy(s: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """Compute Boltzmann energy E(s) = -s^T W s - H^T s"""
    return -s @ W @ s - H @ s


def metropolis_acceptance_probability(delta_E: float, T: float) -> float:
    """
    Compute Metropolis-Hastings acceptance probability.
    
    A = min(1, exp(-ΔE/T))
    
    If ΔE < 0 (energy decreases), always accept (A=1)
    If ΔE > 0 (energy increases), accept with prob exp(-ΔE/T)
    """
    if delta_E <= 0:
        return 1.0
    else:
        return np.exp(-delta_E / T)


def glauber_flip_probability(local_field: float, T: float) -> float:
    """
    Glauber dynamics: probability of flipping to +1.
    
    P(s_i = +1) = 1/(1 + exp(-2h_i/T))
    
    This is the "heat bath" or "Gibbs sampler" update rule.
    """
    return 1.0 / (1.0 + np.exp(-2.0 * local_field / T))


def sample_transition_statistics(
    W: np.ndarray,
    H: np.ndarray,
    T: float,
    num_samples: int = 10000,
    seed: int = None
) -> Dict[str, TransitionStats]:
    """
    Sample MCMC transitions and compute empirical acceptance rates.
    
    Returns dictionary of transition statistics for different state pairs.
    """
    if seed is not None:
        np.random.seed(seed)
    
    N = len(H)
    transitions = {}
    
    # Initialize random state
    state = np.random.choice([-1, 1], size=N)
    
    for step in range(num_samples):
        # Random single-spin flip proposal (Metropolis)
        i = np.random.randint(N)
        proposed_state = state.copy()
        proposed_state[i] *= -1
        
        # Compute energy change
        E_current = compute_energy(state, W, H)
        E_proposed = compute_energy(proposed_state, W, H)
        delta_E = E_proposed - E_current
        
        # Acceptance probability
        accept_prob = metropolis_acceptance_probability(delta_E, T)
        
        # Track this transition
        key = f"flip_{i}"
        if key not in transitions:
            transitions[key] = {
                'delta_E_sum': 0.0,
                'accept_prob_sum': 0.0,
                'accepted': 0,
                'attempted': 0
            }
        
        transitions[key]['delta_E_sum'] += delta_E
        transitions[key]['accept_prob_sum'] += accept_prob
        transitions[key]['attempted'] += 1
        
        # Accept or reject
        if np.random.rand() < accept_prob:
            state = proposed_state
            transitions[key]['accepted'] += 1
    
    # Compute averages
    stats = {}
    for key, data in transitions.items():
        if data['attempted'] > 0:
            stats[key] = {
                'avg_delta_E': data['delta_E_sum'] / data['attempted'],
                'avg_accept_prob': data['accept_prob_sum'] / data['attempted'],
                'empirical_accept_rate': data['accepted'] / data['attempted'],
                'num_attempts': data['attempted']
            }
    
    return stats


def run_stochastic_invariance_test(
    N: int,
    alpha: float,
    T0: float = 1.0,
    num_samples: int = 10000,
    seed: int = None
) -> Dict:
    """
    System 6: Test stochastic dynamics invariance.
    
    Proves that MCMC transition probabilities are invariant under scaling.
    
    Compares:
    - Case A: Original system (W, H, T₀)
    - Case B: Scaled system (α·W, α·H, α·T₀)
    
    Claim: Acceptance probabilities should be IDENTICAL.
    
    Parameters
    ----------
    N : int
        System size
    alpha : float
        Scaling factor
    T0 : float
        Base temperature
    num_samples : int
        Number of MCMC steps to sample
    seed : int
        Random seed
        
    Returns
    -------
    results : Dict
        Contains transition statistics and proof validation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    
    print("=" * 60)
    print("SYSTEM 6: STOCHASTIC DYNAMIC INVARIANCE TEST")
    print("=" * 60)
    print(f"Configuration: N={N}, α={alpha}, T₀={T0}")
    print(f"MCMC samples: {num_samples}")
    print("=" * 60)
    print()
    
    # Case A: Original system
    print("Case A: Sampling transitions for (W, H, T₀)...")
    stats_orig = sample_transition_statistics(W, H, T0, num_samples, seed=seed)
    
    # Case B: Scaled system
    print("Case B: Sampling transitions for (α·W, α·H, α·T₀)...")
    W_scaled = alpha * W
    H_scaled = alpha * H
    T_scaled = alpha * T0
    stats_scaled = sample_transition_statistics(
        W_scaled, H_scaled, T_scaled, num_samples, seed=seed
    )
    
    print()
    print("=" * 60)
    print("COMPUTING PROOF METRICS")
    print("=" * 60)
    
    # Compare acceptance probabilities across all transitions
    common_keys = set(stats_orig.keys()) & set(stats_scaled.keys())
    
    if not common_keys:
        print("\n⚠️  No common transitions found (increase num_samples)")
        return {}
    
    max_diff_accept = 0.0
    max_diff_empirical = 0.0
    avg_diff_accept = 0.0
    avg_diff_empirical = 0.0
    
    print(f"\n🔀 TRANSITION PROBABILITY COMPARISON:")
    print(f"   Analyzing {len(common_keys)} common transitions\n")
    
    for key in sorted(list(common_keys))[:5]:  # Show first 5
        orig = stats_orig[key]
        scaled = stats_scaled[key]
        
        diff_accept = abs(orig['avg_accept_prob'] - scaled['avg_accept_prob'])
        diff_empirical = abs(orig['empirical_accept_rate'] - scaled['empirical_accept_rate'])
        
        max_diff_accept = max(max_diff_accept, diff_accept)
        max_diff_empirical = max(max_diff_empirical, diff_empirical)
        avg_diff_accept += diff_accept
        avg_diff_empirical += diff_empirical
        
        print(f"  {key}:")
        print(f"    A_theory (orig):   {orig['avg_accept_prob']:.6f}")
        print(f"    A_theory (scaled): {scaled['avg_accept_prob']:.6f}")
        print(f"    Difference:        {diff_accept:.6f}")
        print(f"    Empirical (orig):  {orig['empirical_accept_rate']:.6f}")
        print(f"    Empirical (scale): {scaled['empirical_accept_rate']:.6f}")
        print()
    
    avg_diff_accept /= len(common_keys)
    avg_diff_empirical /= len(common_keys)
    
    print(f"📊 AGGREGATE METRICS:")
    print(f"  Max Δ(A_theory):      {max_diff_accept:.10f}")
    print(f"  Avg Δ(A_theory):      {avg_diff_accept:.10f}")
    print(f"  Max Δ(A_empirical):   {max_diff_empirical:.6f}")
    print(f"  Avg Δ(A_empirical):   {avg_diff_empirical:.6f}")
    print()
    
    # Proof validation
    threshold_theory = 1e-8  # Theoretical should be exact
    threshold_empirical = 0.05  # Empirical has sampling noise
    
    theory_valid = max_diff_accept < threshold_theory
    empirical_valid = avg_diff_empirical < threshold_empirical
    
    print("=" * 60)
    print("PROOF VALIDATION")
    print("=" * 60)
    print(f"✓ Theoretical invariance (Δ < {threshold_theory}): {'PASS' if theory_valid else 'FAIL'}")
    print(f"✓ Empirical consistency (Δ < {threshold_empirical}): {'PASS' if empirical_valid else 'FAIL'}")
    print()
    
    if theory_valid and empirical_valid:
        print("🎉 PROOF SUCCESSFUL: Stochastic dynamics are scale-invariant!")
        print("   Hardware MCMC will preserve transition probabilities exactly.")
    else:
        if not theory_valid:
            print("⚠️  NUMERICAL ISSUE: Check floating point precision")
        if not empirical_valid:
            print("⚠️  SAMPLING ISSUE: Increase num_samples or check implementation")
    
    print("=" * 60)
    
    return {
        'stats_orig': stats_orig,
        'stats_scaled': stats_scaled,
        'max_diff_accept': max_diff_accept,
        'avg_diff_accept': avg_diff_accept,
        'max_diff_empirical': max_diff_empirical,
        'avg_diff_empirical': avg_diff_empirical,
        'theory_valid': theory_valid,
        'empirical_valid': empirical_valid,
        'num_transitions': len(common_keys),
        'params': {
            'W': W,
            'H': H,
            'W_scaled': W_scaled,
            'H_scaled': H_scaled,
            'T0': T0,
            'T_scaled': T_scaled,
            'alpha': alpha
        }
    }


if __name__ == "__main__":
    # Run stochastic invariance test
    results = run_stochastic_invariance_test(
        N=4,
        alpha=2.0,
        T0=1.0,
        num_samples=20000,
        seed=42
    )
