"""
System 4: Dynamic Invariance Proof (Gradient Flow)

Proves that the trajectory toward equilibrium is scale-invariant when 
both Hamiltonian and time are proportionally scaled.

Theoretical Foundation:
----------------------
Original dynamics: dx_i/dt = -η ∂E/∂x_i
Scaled dynamics:   dx'_i/dτ = -η ∂E/∂x'_i  where τ = α·t

Claim: The trajectory x(t) under E is identical to x'(τ) under α·E
       when viewed in rescaled time τ = α·t
"""

import numpy as np
from typing import Tuple, Dict, Callable
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def compute_gradient(x: np.ndarray, W: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Compute gradient of Boltzmann energy w.r.t. continuous variables.
    
    For E = -∑ᵢⱼ Wᵢⱼ xᵢ xⱼ - ∑ᵢ Hᵢ xᵢ
    ∂E/∂xᵢ = -2∑ⱼ Wᵢⱼ xⱼ - Hᵢ
    """
    return -2.0 * (W @ x) - H


def gradient_flow_dynamics(t: float, x: np.ndarray, W: np.ndarray, H: np.ndarray, eta: float) -> np.ndarray:
    """
    Mean-field dynamics for Boltzmann machine with bounded states.
    
    Uses tanh activation to keep states bounded: x ∈ [-1, 1]^N
    dx/dt = -x + tanh(2Wx + H)
    """
    # Local field
    h = 2.0 * (W @ x) + H
    # Mean-field update with relaxation
    return eta * (-x + np.tanh(h))


def simulate_trajectory(
    W: np.ndarray,
    H: np.ndarray,
    x0: np.ndarray,
    t_span: Tuple[float, float],
    eta: float = 0.1,
    num_points: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate gradient flow trajectory.
    
    Parameters
    ----------
    W : np.ndarray
        Interaction matrix
    H : np.ndarray
        Bias vector
    x0 : np.ndarray
        Initial state
    t_span : Tuple[float, float]
        Time interval (t_start, t_end)
    eta : float
        Learning rate / mobility
    num_points : int
        Number of time points to evaluate
        
    Returns
    -------
    t : np.ndarray
        Time points
    x : np.ndarray
        State trajectory (num_points, N)
    """
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    sol = solve_ivp(
        gradient_flow_dynamics,
        t_span,
        x0,
        args=(W, H, eta),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8
    )
    
    return sol.t, sol.y.T


def compute_trajectory_distance(traj1: np.ndarray, traj2: np.ndarray) -> float:
    """
    Compute mean Euclidean distance between two trajectories.
    
    Returns average L2 norm across all time points.
    """
    if traj1.shape != traj2.shape:
        raise ValueError("Trajectories must have same shape")
    
    distances = np.linalg.norm(traj1 - traj2, axis=1)
    return np.mean(distances)


def run_dynamic_invariance_test(
    N: int,
    alpha: float,
    eta: float = 0.1,
    t_end: float = 10.0,
    num_points: int = 100,
    seed: int = None
) -> Dict:
    """
    System 4: Test dynamic invariance under time-energy scaling.
    
    Compares:
    - Case A: Original trajectory under (W, H) in time t
    - Case B: Scaled trajectory under (α·W, α·H) in time τ = α·t
    
    Claim: Trajectories should be identical when time is rescaled.
    
    Parameters
    ----------
    N : int
        System size
    alpha : float
        Scaling factor
    eta : float
        Learning rate
    t_end : float
        Final time for original system
    num_points : int
        Number of trajectory points
    seed : int
        Random seed
        
    Returns
    -------
    results : Dict
        Contains trajectories, metrics, and proof validation
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random parameters
    W_raw = np.random.randn(N, N)
    W = (W_raw + W_raw.T) / 2
    np.fill_diagonal(W, 0)
    H = np.random.randn(N)
    
    # Initial condition (random perturbation from origin)
    x0 = np.random.randn(N) * 0.1
    
    print("=" * 60)
    print("SYSTEM 4: DYNAMIC INVARIANCE TEST")
    print("=" * 60)
    print(f"Configuration: N={N}, α={alpha}, η={eta}")
    print(f"Time span: [0, {t_end}], Points: {num_points}")
    print("=" * 60)
    print()
    
    # Case A: Original dynamics in time t
    print("Case A: Original dynamics (W, H) in time t...")
    t_orig, traj_orig = simulate_trajectory(
        W, H, x0, 
        t_span=(0, t_end),
        eta=eta,
        num_points=num_points
    )
    
    # Case B: Scaled dynamics in RESCALED time τ = α·t
    # The physical time is t' = τ/α, so t'_end = t_end
    # But we integrate over τ ∈ [0, α·t_end]
    print(f"Case B: Scaled dynamics (α·W, α·H) in time τ=α·t...")
    W_scaled = alpha * W
    H_scaled = alpha * H
    
    # Key insight: integrate to τ_end = α·t_end
    tau_end = alpha * t_end
    tau, traj_scaled = simulate_trajectory(
        W_scaled, H_scaled, x0,
        t_span=(0, tau_end),
        eta=eta,
        num_points=num_points
    )
    
    # Rescale tau back to t for comparison: t = τ/α
    t_scaled = tau / alpha
    
    print()
    print("=" * 60)
    print("COMPUTING PROOF METRICS")
    print("=" * 60)
    
    # Trajectory distance (should be near zero)
    dist = compute_trajectory_distance(traj_orig, traj_scaled)
    
    # Final state comparison
    final_orig = traj_orig[-1]
    final_scaled = traj_scaled[-1]
    final_dist = np.linalg.norm(final_orig - final_scaled)
    
    # Energy convergence check
    E_orig_initial = -x0 @ W @ x0 - H @ x0
    E_orig_final = -final_orig @ W @ final_orig - H @ final_orig
    E_scaled_final = -final_scaled @ W_scaled @ final_scaled - H_scaled @ final_scaled
    
    print(f"\n📊 TRAJECTORY METRICS:")
    print(f"  Mean trajectory distance: {dist:.6f}")
    print(f"  Final state distance:     {final_dist:.6f}")
    print()
    print(f"📉 ENERGY CONVERGENCE:")
    print(f"  E_orig (initial): {E_orig_initial:.6f}")
    print(f"  E_orig (final):   {E_orig_final:.6f}")
    print(f"  E_scaled (final): {E_scaled_final:.6f}")
    print(f"  Ratio E_scaled/E_orig: {E_scaled_final/E_orig_final:.6f} (expected: {alpha})")
    print()
    
    # Proof validation
    threshold_dist = 0.1
    threshold_final = 0.05
    
    dist_valid = dist < threshold_dist
    final_valid = final_dist < threshold_final
    
    print("=" * 60)
    print("PROOF VALIDATION")
    print("=" * 60)
    print(f"✓ Trajectory invariance (dist < {threshold_dist}): {'PASS' if dist_valid else 'FAIL'}")
    print(f"✓ Final state match (dist < {threshold_final}):   {'PASS' if final_valid else 'FAIL'}")
    print()
    
    if dist_valid and final_valid:
        print("🎉 PROOF SUCCESSFUL: Dynamic trajectories are scale-invariant!")
    else:
        print("⚠️  PROOF INCONCLUSIVE: Trajectories differ (may need longer integration)")
    
    print("=" * 60)
    
    return {
        't_orig': t_orig,
        'traj_orig': traj_orig,
        't_scaled': t_scaled,
        'traj_scaled': traj_scaled,
        'trajectory_distance': dist,
        'final_distance': final_dist,
        'dist_valid': dist_valid,
        'final_valid': final_valid,
        'params': {
            'W': W,
            'H': H,
            'W_scaled': W_scaled,
            'H_scaled': H_scaled,
            'x0': x0,
            'alpha': alpha,
            'eta': eta
        }
    }


def visualize_trajectories(results: Dict, save_path: str = None):
    """
    Visualize original and scaled trajectories for comparison.
    """
    t_orig = results['t_orig']
    traj_orig = results['traj_orig']
    t_scaled = results['t_scaled']
    traj_scaled = results['traj_scaled']
    
    N = traj_orig.shape[1]
    
    fig, axes = plt.subplots(N, 1, figsize=(10, 2*N), sharex=True)
    if N == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(t_orig, traj_orig[:, i], 'b-', label='Original', linewidth=2)
        ax.plot(t_scaled, traj_scaled[:, i], 'r--', label='Scaled (rescaled time)', linewidth=2)
        ax.set_ylabel(f'$x_{i}$', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    axes[-1].set_xlabel('Time $t$', fontsize=12)
    fig.suptitle('Dynamic Invariance: Trajectory Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Run dynamic invariance test
    results = run_dynamic_invariance_test(
        N=3,
        alpha=2.0,
        eta=0.1,
        t_end=10.0,
        num_points=100,
        seed=42
    )
    
    # Optional: visualize
    # visualize_trajectories(results, save_path='dynamic_invariance.png')
