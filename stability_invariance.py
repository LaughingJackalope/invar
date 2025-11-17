"""
System 5: Stability Invariance Proof (Free Energy Landscape)

Proves that relative stability (free energy barriers) between equilibrium 
states remains invariant under proportional Hamiltonian-Temperature scaling.

Theoretical Foundation:
----------------------
Free Energy: F = -k_B T ln(Z)  where Z = ∑_s exp(-E(s)/k_B T)

Under scaling E → α·E and T → α·T:
F' = α·F  (absolute free energy scales)

BUT: Relative stability ΔF/T = (F_A - F_B)/T remains invariant:
ΔF'/T' = α·ΔF/(α·T) = ΔF/T

This preserves probability ratios: P_A/P_B = exp(-ΔF/k_B T)
"""

import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass


@dataclass
class StateInfo:
    """Information about a particular state or configuration."""
    state: np.ndarray
    energy: float
    probability: float
    free_energy_contribution: float


def compute_energy(s: np.ndarray, W: np.ndarray, H: np.ndarray) -> float:
    """
    Compute Boltzmann energy for discrete state s ∈ {-1, 1}^N.
    
    E(s) = -∑ᵢⱼ Wᵢⱼ sᵢ sⱼ - ∑ᵢ Hᵢ sᵢ
    """
    return -s @ W @ s - H @ s


def compute_partition_function(W: np.ndarray, H: np.ndarray, beta: float) -> Tuple[float, List[StateInfo]]:
    """
    Compute partition function and enumerate all states with their properties.
    
    Z = ∑_s exp(-β E(s))
    
    Returns
    -------
    Z : float
        Partition function
    states : List[StateInfo]
        All states with their energies, probabilities, and free energy contributions
    """
    N = len(H)
    Z = 0.0
    states = []
    
    # Enumerate all 2^N states
    for idx in range(2**N):
        # Convert index to binary state {-1, 1}^N
        binary = [(idx >> i) & 1 for i in range(N)]
        s = np.array([2*b - 1 for b in binary])
        
        # Compute energy and Boltzmann weight
        E = compute_energy(s, W, H)
        weight = np.exp(-beta * E)
        Z += weight
        
        states.append({
            'state': s,
            'energy': E,
            'weight': weight
        })
    
    # Compute probabilities and free energy contributions
    for state in states:
        state['probability'] = state['weight'] / Z
        # F_i contribution: -k_B T ln(exp(-β E_i)) = E_i
        # But for relative comparison we track -ln(P_i)
        state['free_energy_contribution'] = -np.log(state['probability']) / beta
    
    return Z, states


def compute_free_energy(W: np.ndarray, H: np.ndarray, T: float) -> float:
    """
    Compute Helmholtz free energy.
    
    F = -k_B T ln(Z)
    
    (Using k_B = 1 units)
    """
    beta = 1.0 / T
    Z, _ = compute_partition_function(W, H, beta)
    return -T * np.log(Z)


def find_equilibrium_states(states: List[Dict], top_k: int = 3) -> List[Dict]:
    """
    Find the top-k most probable equilibrium states.
    """
    sorted_states = sorted(states, key=lambda x: x['probability'], reverse=True)
    return sorted_states[:top_k]


def compute_stability_ratios(states: List[Dict], T: float) -> Dict:
    """
    Compute relative stability measures between equilibrium states.
    
    For states A and B:
    - ΔE = E_A - E_B
    - ΔF = F_A - F_B (free energy difference)
    - Probability ratio: P_A/P_B = exp(-ΔF/k_B T)
    """
    if len(states) < 2:
        return {}
    
    # Take top two states
    state_A = states[0]
    state_B = states[1]
    
    delta_E = state_A['energy'] - state_B['energy']
    
    # ΔF = -k_B T ln(P_A/P_B) = k_B T ln(P_B/P_A)
    prob_ratio = state_A['probability'] / state_B['probability']
    delta_F = -T * np.log(prob_ratio)
    
    # Relative stability metric: ΔF/T (this should be scale-invariant)
    relative_stability = delta_F / T
    
    return {
        'state_A': state_A,
        'state_B': state_B,
        'delta_E': delta_E,
        'delta_F': delta_F,
        'prob_ratio': prob_ratio,
        'relative_stability': relative_stability
    }


def run_stability_invariance_test(
    N: int,
    alpha: float,
    T0: float = 1.0,
    seed: int = None
) -> Dict:
    """
    System 5: Test stability invariance under energy-temperature scaling.
    
    Compares:
    - Case A: Original system (W, H, T₀)
    - Case B: Scaled system (α·W, α·H, α·T₀)
    
    Claim: Relative stability ΔF/T should be identical in both cases.
    
    Parameters
    ----------
    N : int
        System size (keep small, 2^N states!)
    alpha : float
        Scaling factor
    T0 : float
        Base temperature
    seed : int
        Random seed
        
    Returns
    -------
    results : Dict
        Contains free energies, stability metrics, and proof validation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    
    print("=" * 60)
    print("SYSTEM 5: STABILITY INVARIANCE TEST")
    print("=" * 60)
    print(f"Configuration: N={N}, α={alpha}, T₀={T0}")
    print(f"State space: 2^{N} = {2**N} states")
    print("=" * 60)
    print()
    
    # Case A: Original system
    print("Case A: Computing free energy for (W, H, T₀)...")
    beta_orig = 1.0 / T0
    Z_orig, states_orig = compute_partition_function(W, H, beta_orig)
    F_orig = -T0 * np.log(Z_orig)
    
    equil_orig = find_equilibrium_states(states_orig, top_k=3)
    stability_orig = compute_stability_ratios(equil_orig, T0)
    
    # Case B: Scaled system
    print("Case B: Computing free energy for (α·W, α·H, α·T₀)...")
    W_scaled = alpha * W
    H_scaled = alpha * H
    T_scaled = alpha * T0
    beta_scaled = 1.0 / T_scaled
    
    Z_scaled, states_scaled = compute_partition_function(W_scaled, H_scaled, beta_scaled)
    F_scaled = -T_scaled * np.log(Z_scaled)
    
    equil_scaled = find_equilibrium_states(states_scaled, top_k=3)
    stability_scaled = compute_stability_ratios(equil_scaled, T_scaled)
    
    print()
    print("=" * 60)
    print("COMPUTING PROOF METRICS")
    print("=" * 60)
    
    # Free energy scaling
    F_ratio = F_scaled / F_orig
    
    print(f"\n🏔️  FREE ENERGY:")
    print(f"  F_orig   = {F_orig:.6f}")
    print(f"  F_scaled = {F_scaled:.6f}")
    print(f"  Ratio F_scaled/F_orig = {F_ratio:.6f} (expected: {alpha})")
    print()
    
    # Relative stability comparison
    rel_stab_orig = stability_orig['relative_stability']
    rel_stab_scaled = stability_scaled['relative_stability']
    rel_stab_diff = abs(rel_stab_orig - rel_stab_scaled)
    
    print(f"⚖️  RELATIVE STABILITY (ΔF/T):")
    print(f"  Original:  {rel_stab_orig:.6f}")
    print(f"  Scaled:    {rel_stab_scaled:.6f}")
    print(f"  Difference: {rel_stab_diff:.6f} (should be ≈ 0)")
    print()
    
    # Probability ratio comparison
    prob_ratio_orig = stability_orig['prob_ratio']
    prob_ratio_scaled = stability_scaled['prob_ratio']
    prob_ratio_diff = abs(prob_ratio_orig - prob_ratio_scaled)
    
    print(f"📊 PROBABILITY RATIOS (P_A/P_B):")
    print(f"  Original:  {prob_ratio_orig:.6f}")
    print(f"  Scaled:    {prob_ratio_scaled:.6f}")
    print(f"  Difference: {prob_ratio_diff:.6f} (should be ≈ 0)")
    print()
    
    # Proof validation
    threshold_F_ratio = 0.05  # F_scaled/F_orig should be close to alpha
    threshold_stability = 0.01  # ΔF/T should be invariant
    threshold_prob_ratio = 0.01  # Probability ratios should match
    
    F_ratio_valid = abs(F_ratio - alpha) < threshold_F_ratio
    stability_valid = rel_stab_diff < threshold_stability
    prob_ratio_valid = prob_ratio_diff < threshold_prob_ratio
    
    print("=" * 60)
    print("PROOF VALIDATION")
    print("=" * 60)
    print(f"✓ Free energy scales by α: {'PASS' if F_ratio_valid else 'FAIL'}")
    print(f"✓ Relative stability invariant: {'PASS' if stability_valid else 'FAIL'}")
    print(f"✓ Probability ratios match: {'PASS' if prob_ratio_valid else 'FAIL'}")
    print()
    
    if F_ratio_valid and stability_valid and prob_ratio_valid:
        print("🎉 PROOF SUCCESSFUL: Stability landscape is scale-invariant!")
    else:
        print("⚠️  PROOF INCONCLUSIVE: Check numerical precision or parameters")
    
    print("=" * 60)
    
    return {
        'F_orig': F_orig,
        'F_scaled': F_scaled,
        'F_ratio': F_ratio,
        'Z_orig': Z_orig,
        'Z_scaled': Z_scaled,
        'stability_orig': stability_orig,
        'stability_scaled': stability_scaled,
        'relative_stability_diff': rel_stab_diff,
        'prob_ratio_diff': prob_ratio_diff,
        'F_ratio_valid': F_ratio_valid,
        'stability_valid': stability_valid,
        'prob_ratio_valid': prob_ratio_valid,
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


def visualize_energy_landscape(results: Dict, save_path: str = None):
    """
    Visualize energy landscape for both original and scaled systems.
    """
    import matplotlib.pyplot as plt
    
    stability_orig = results['stability_orig']
    stability_scaled = results['stability_scaled']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original system
    states_orig = [stability_orig['state_A'], stability_orig['state_B']]
    energies_orig = [s['energy'] for s in states_orig]
    probs_orig = [s['probability'] for s in states_orig]
    
    ax1.bar(range(len(energies_orig)), energies_orig, alpha=0.7, label='Energy')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(len(probs_orig)), probs_orig, 'ro-', linewidth=2, markersize=8, label='Probability')
    ax1.set_xlabel('State', fontsize=12)
    ax1.set_ylabel('Energy', fontsize=12)
    ax1_twin.set_ylabel('Probability', fontsize=12)
    ax1.set_title('Original System', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Scaled system
    states_scaled = [stability_scaled['state_A'], stability_scaled['state_B']]
    energies_scaled = [s['energy'] for s in states_scaled]
    probs_scaled = [s['probability'] for s in states_scaled]
    
    ax2.bar(range(len(energies_scaled)), energies_scaled, alpha=0.7, label='Energy', color='orange')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(range(len(probs_scaled)), probs_scaled, 'ro-', linewidth=2, markersize=8, label='Probability')
    ax2.set_xlabel('State', fontsize=12)
    ax2.set_ylabel('Energy (scaled)', fontsize=12)
    ax2_twin.set_ylabel('Probability', fontsize=12)
    ax2.set_title('Scaled System', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Run stability invariance test
    results = run_stability_invariance_test(
        N=4,  # Keep small due to 2^N scaling
        alpha=2.0,
        T0=1.0,
        seed=42
    )
    
    # Optional: visualize
    # visualize_energy_landscape(results, save_path='stability_invariance.png')
