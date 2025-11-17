"""
System 8: Materials Science Scale Invariance

Demonstrates applicability of Systems 1-7 to real materials science:
- Bulk phase transformations (sword forging: austenite → martensite)
- Atomic-scale deposition (semiconductor CVD/PVD)

Key insight: The scale invariance P(S; H, T) = P(S; αH, αT) applies equally
to macroscopic metallurgy and nanoscale epitaxy when H is interpreted as
Gibbs Free Energy rather than Ising energy.

Theoretical Foundation:
    P(S) ∝ exp(-G/kT) where G is Gibbs Free Energy
    G = Σ c_j G_j° + RT Σ c_j ln(c_j) + Σ L_ij c_i c_j

    - c_j: Composition (phase fraction or atomic concentration)
    - G_j°: Pure component free energy
    - L_ij: Interaction parameters (chemical/strain energy)
"""

import numpy as np
from scipy.stats import entropy
from typing import Dict, Tuple, List
from dataclasses import dataclass


# Physical constants
R_GAS = 8.314  # J/(mol·K) - Universal gas constant
K_BOLTZMANN = 1.380649e-23  # J/K - Boltzmann constant


@dataclass
class MaterialsSystem:
    """
    Defines a materials thermodynamic system.

    Attributes
    ----------
    phases : List[str]
        Names of phases or species (e.g., ['Austenite', 'Martensite'])
    G_pure : np.ndarray
        Pure component Gibbs energies (J/mol), shape (n_phases,)
    L_matrix : np.ndarray
        Interaction parameters (J/mol), shape (n_phases, n_phases)
    regime : str
        'bulk' for macroscopic or 'atomic' for nanoscale
    """
    phases: List[str]
    G_pure: np.ndarray
    L_matrix: np.ndarray
    regime: str

    def __post_init__(self):
        """Validate inputs."""
        assert len(self.phases) == len(self.G_pure), "Phase count mismatch"
        assert self.L_matrix.shape == (len(self.phases), len(self.phases)), "L_matrix dimension mismatch"
        # Ensure symmetric interactions
        assert np.allclose(self.L_matrix, self.L_matrix.T), "L_matrix must be symmetric"


def gibbs_free_energy(
    composition: np.ndarray,
    system: MaterialsSystem,
    T: float
) -> float:
    """
    Calculate Gibbs Free Energy for a given composition.

    G = Σ c_j G_j° + RT Σ c_j ln(c_j) + Σ L_ij c_i c_j

    Parameters
    ----------
    composition : np.ndarray
        Phase fractions or atomic concentrations, shape (n_phases,)
        Must sum to 1.0
    system : MaterialsSystem
        Material system definition
    T : float
        Temperature (K)

    Returns
    -------
    G : float
        Gibbs Free Energy (J/mol)
    """
    c = composition

    # Term 1: Pure component energies
    G_ref = np.dot(c, system.G_pure)

    # Term 2: Ideal mixing entropy (configurational)
    # Handle c=0 to avoid log(0)
    c_safe = np.where(c > 1e-12, c, 1e-12)
    G_ideal = R_GAS * T * np.sum(c * np.log(c_safe))

    # Term 3: Excess free energy (non-ideal interactions)
    G_excess = 0.0
    for i in range(len(c)):
        for j in range(i+1, len(c)):
            G_excess += system.L_matrix[i, j] * c[i] * c[j]

    return G_ref + G_ideal + G_excess


def boltzmann_probability(
    composition: np.ndarray,
    system: MaterialsSystem,
    T: float
) -> float:
    """
    Calculate Boltzmann probability for a composition state.

    P(S) ∝ exp(-G/kT) but we use molar units so:
    P(S) ∝ exp(-G/RT)

    Parameters
    ----------
    composition : np.ndarray
        Phase fractions
    system : MaterialsSystem
        Material system
    T : float
        Temperature (K)

    Returns
    -------
    prob : float
        Unnormalized Boltzmann weight
    """
    G = gibbs_free_energy(composition, system, T)
    return np.exp(-G / (R_GAS * T))


def enumerate_compositions(
    n_phases: int,
    n_grid: int = 10
) -> np.ndarray:
    """
    Generate discrete composition grid.

    For simplicity, samples compositions on a regular grid that sum to 1.

    Parameters
    ----------
    n_phases : int
        Number of phases
    n_grid : int
        Grid resolution (points per dimension)

    Returns
    -------
    compositions : np.ndarray
        Array of compositions, shape (n_samples, n_phases)
    """
    if n_phases == 2:
        # Binary system: c1 + c2 = 1
        c1_vals = np.linspace(0, 1, n_grid)
        compositions = np.column_stack([c1_vals, 1 - c1_vals])
    elif n_phases == 3:
        # Ternary: sample simplex
        compositions = []
        for i in range(n_grid):
            for j in range(n_grid - i):
                c1 = i / (n_grid - 1)
                c2 = j / (n_grid - 1)
                c3 = 1 - c1 - c2
                if c3 >= 0:
                    compositions.append([c1, c2, c3])
        compositions = np.array(compositions)
    else:
        raise NotImplementedError(f"Composition enumeration for {n_phases} phases not implemented")

    return compositions


def compute_equilibrium_distribution(
    system: MaterialsSystem,
    T: float,
    n_grid: int = 50
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute equilibrium probability distribution over compositions.

    Uses exact enumeration on a discrete grid (analogous to System 5).

    Parameters
    ----------
    system : MaterialsSystem
        Material system
    T : float
        Temperature (K)
    n_grid : int
        Composition grid resolution

    Returns
    -------
    compositions : np.ndarray
        Sampled compositions, shape (n_samples, n_phases)
    probabilities : np.ndarray
        Normalized probabilities, shape (n_samples,)
    """
    compositions = enumerate_compositions(len(system.phases), n_grid)

    # Calculate Boltzmann weights
    weights = np.array([
        boltzmann_probability(c, system, T)
        for c in compositions
    ])

    # Normalize to probability distribution
    Z = np.sum(weights)
    probabilities = weights / Z

    return compositions, probabilities


def run_materials_invariance_test(
    system: MaterialsSystem,
    T0: float,
    alpha: float,
    n_grid: int = 50
) -> Dict:
    """
    System 8: Materials Science Scale Invariance Test

    Tests that P(S; G, T) = P(S; αG, αT) for materials systems.

    Three-case protocol:
    - Case A: Original system (G_pure, L_matrix, T0)
    - Case B: Energy-only scaled (α·G_pure, α·L_matrix, T0) - control
    - Case C: Full scaling (α·G_pure, α·L_matrix, α·T0) - test

    Parameters
    ----------
    system : MaterialsSystem
        Original material system
    T0 : float
        Base temperature (K)
    alpha : float
        Scaling factor
    n_grid : int
        Composition grid resolution

    Returns
    -------
    results : Dict
        Contains distributions, divergences, and validation results
    """
    print(f"\n{'='*70}")
    print(f"SYSTEM 8: MATERIALS SCIENCE SCALE INVARIANCE TEST")
    print(f"{'='*70}")
    print(f"Regime: {system.regime.upper()}")
    print(f"Phases: {', '.join(system.phases)}")
    print(f"Temperature T0: {T0:.1f} K")
    print(f"Scaling factor α: {alpha}")
    print(f"Grid resolution: {n_grid}")

    # Case A: Original system
    print(f"\n{'-'*70}")
    print("CASE A: Original System (G, T)")
    comps_A, P_orig = compute_equilibrium_distribution(system, T0, n_grid)
    print(f"✓ Computed {len(P_orig)} composition states")

    # Case B: Energy-only scaling (control)
    print(f"\n{'-'*70}")
    print("CASE B: Energy-Only Scaling (α·G, T) - CONTROL")
    system_B = MaterialsSystem(
        phases=system.phases,
        G_pure=alpha * system.G_pure,
        L_matrix=alpha * system.L_matrix,
        regime=system.regime
    )
    comps_B, P_scaled_E = compute_equilibrium_distribution(system_B, T0, n_grid)
    print(f"✓ Computed {len(P_scaled_E)} composition states")

    # Case C: Full scaling (test)
    print(f"\n{'-'*70}")
    print("CASE C: Full Scaling (α·G, α·T) - TEST")
    system_C = MaterialsSystem(
        phases=system.phases,
        G_pure=alpha * system.G_pure,
        L_matrix=alpha * system.L_matrix,
        regime=system.regime
    )
    comps_C, P_test = compute_equilibrium_distribution(system_C, alpha * T0, n_grid)
    print(f"✓ Computed {len(P_test)} composition states")

    # Compute divergences
    print(f"\n{'-'*70}")
    print("DIVERGENCE ANALYSIS")

    # Add small epsilon to avoid log(0)
    eps = 1e-12
    P_orig_safe = P_orig + eps
    P_test_safe = P_test + eps
    P_scaled_E_safe = P_scaled_E + eps

    # Renormalize after adding epsilon
    P_orig_safe /= P_orig_safe.sum()
    P_test_safe /= P_test_safe.sum()
    P_scaled_E_safe /= P_scaled_E_safe.sum()

    D_proof = entropy(P_orig_safe, P_test_safe)
    D_control = entropy(P_orig_safe, P_scaled_E_safe)

    print(f"\nD_KL(P_orig || P_test)      = {D_proof:.6f}  [MUST BE ≈ 0]")
    print(f"D_KL(P_orig || P_scaled_E)  = {D_control:.6f}  [MUST BE >> 0]")

    # Validation
    print(f"\n{'-'*70}")
    print("VALIDATION")

    # For exact enumeration, we expect machine precision
    threshold_proof = 1e-8
    threshold_control = 0.01

    proof_valid = D_proof < threshold_proof
    control_valid = D_control > threshold_control

    print(f"\nProof criterion:   D_KL < {threshold_proof:.1e}  {'✓ PASS' if proof_valid else '✗ FAIL'}")
    print(f"Control criterion: D_KL > {threshold_control:.2f}     {'✓ PASS' if control_valid else '✗ FAIL'}")

    if proof_valid and control_valid:
        print(f"\n{'='*70}")
        print("🎉 SYSTEM 8 VALIDATION SUCCESSFUL")
        print("Scale invariance confirmed for materials thermodynamics!")
        print(f"{'='*70}")
    else:
        print(f"\n{'='*70}")
        print("⚠️  VALIDATION FAILED")
        print(f"{'='*70}")

    return {
        'compositions': comps_A,
        'P_orig': P_orig,
        'P_scaled_E': P_scaled_E,
        'P_test': P_test,
        'D_proof': D_proof,
        'D_control': D_control,
        'proof_valid': proof_valid,
        'control_valid': control_valid,
        'system': system,
        'T0': T0,
        'alpha': alpha
    }


# ============================================================================
# MATERIALS CASE STUDIES
# ============================================================================

def create_sword_system() -> MaterialsSystem:
    """
    Case A: Sword Forging (Bulk Phase Transformation)

    Models differential hardening in Japanese katana:
    - Phases: Austenite (γ-Fe), Martensite (hard), Pearlite (tough)
    - Macroscopic scale (cm to m)
    - Controlled by temperature gradient during quenching

    Returns
    -------
    system : MaterialsSystem
        Sword steel thermodynamic system
    """
    phases = ['Austenite', 'Martensite', 'Pearlite']

    # Approximate Gibbs energies (J/mol) - relative values
    # Martensite is metastable (higher energy)
    # Scaled up for stronger thermodynamic contrast
    G_pure = np.array([
        0.0,      # Austenite (reference)
        5000.0,   # Martensite (metastable, high energy)
        -3000.0   # Pearlite (stable at lower T)
    ])

    # Interaction parameters - phase boundaries
    # Positive L_ij discourages mixing (sharp interfaces)
    # Larger values create stronger phase separation
    L_matrix = np.array([
        [0.0,     12000.0,  8000.0],  # Austenite-X
        [12000.0, 0.0,      10000.0],  # Martensite-X
        [8000.0,  10000.0,  0.0]       # Pearlite-X
    ])

    return MaterialsSystem(
        phases=phases,
        G_pure=G_pure,
        L_matrix=L_matrix,
        regime='bulk'
    )


def create_semiconductor_system() -> MaterialsSystem:
    """
    Case B: Semiconductor Deposition (Atomic-Scale Growth)

    Models CVD/PVD epitaxial layer growth:
    - Species: Si (substrate), Ge (dopant), Vacancy (defect)
    - Nanoscale (atomic layer precision)
    - Controlled by vapor pressure and substrate temperature

    Returns
    -------
    system : MaterialsSystem
        Semiconductor growth thermodynamic system
    """
    phases = ['Si', 'Ge', 'Vacancy']

    # Gibbs energies (J/mol) - surface formation energies
    G_pure = np.array([
        0.0,      # Si (reference, substrate)
        3000.0,   # Ge (dopant, higher energy)
        8000.0    # Vacancy (defect, high cost)
    ])

    # Bonding energies - negative L_ij encourages mixing
    # We want Si-Ge solid solution, not vacancies
    L_matrix = np.array([
        [0.0,    -1000.0,  5000.0],  # Si: bonds well with Ge, not vacancies
        [-1000.0, 0.0,     6000.0],  # Ge: bonds well with Si
        [5000.0,  6000.0,  0.0]      # Vacancy: unfavorable
    ])

    return MaterialsSystem(
        phases=phases,
        G_pure=G_pure,
        L_matrix=L_matrix,
        regime='atomic'
    )


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def run_full_materials_demonstration():
    """
    Complete System 8 demonstration with both case studies.
    """
    print("\n" + "="*70)
    print("SYSTEM 8: MATERIALS SCIENCE SCALE INVARIANCE")
    print("Unified Thermodynamic Framework for Macroscopic & Nanoscale Processes")
    print("="*70)

    # Case A: Sword forging
    print("\n\n" + "#"*70)
    print("# CASE STUDY A: SWORD FORGING (BULK METALLURGY)")
    print("#"*70)

    sword_system = create_sword_system()
    sword_results = run_materials_invariance_test(
        system=sword_system,
        T0=1000.0,  # Kelvin (quenching temperature)
        alpha=2.0,
        n_grid=30
    )

    # Case B: Semiconductor
    print("\n\n" + "#"*70)
    print("# CASE STUDY B: SEMICONDUCTOR DEPOSITION (NANOSCALE)")
    print("#"*70)

    semi_system = create_semiconductor_system()
    semi_results = run_materials_invariance_test(
        system=semi_system,
        T0=800.0,   # Kelvin (CVD temperature)
        alpha=2.0,
        n_grid=30
    )

    # Summary
    print("\n\n" + "="*70)
    print("FINAL SUMMARY: SYSTEM 8 VALIDATION")
    print("="*70)

    all_pass = (sword_results['proof_valid'] and sword_results['control_valid'] and
                semi_results['proof_valid'] and semi_results['control_valid'])

    print(f"\nSword System:        {'✓ PASS' if sword_results['proof_valid'] else '✗ FAIL'}")
    print(f"Semiconductor System: {'✓ PASS' if semi_results['proof_valid'] else '✗ FAIL'}")

    if all_pass:
        print("\n" + "="*70)
        print("🎉 COMPLETE SUCCESS: SCALE INVARIANCE VALIDATED")
        print("="*70)
        print("\nThe framework successfully bridges:")
        print("  • Macroscopic metallurgy (sword forging)")
        print("  • Nanoscale semiconductor fabrication")
        print("\nBoth regimes obey: P(S; G, T) = P(S; α·G, α·T)")
        print("="*70)

    return {
        'sword': sword_results,
        'semiconductor': semi_results,
        'all_pass': all_pass
    }


if __name__ == "__main__":
    results = run_full_materials_demonstration()