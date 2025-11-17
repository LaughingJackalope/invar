import os
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from materials_invariance import (
    create_semiconductor_system,
    compute_equilibrium_distribution,
    MaterialsSystem,  # type: ignore
)


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
PURPLE = "#6a3d9a"
GRAY = "#7f7f7f"


def _expected_species_fractions(
    compositions: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """
    Compute expected species fractions given a discrete distribution
    over a composition simplex.

    Parameters
    ----------
    compositions : np.ndarray
        Array of shape (M, K) with species fractions per state.
    probabilities : np.ndarray
        Array of shape (M,) with probabilities summing to 1.

    Returns
    -------
    np.ndarray
        Expected fractions of shape (K,).
    """
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / max(probs.sum(), 1e-16)
    comps = np.asarray(compositions, dtype=float)
    return (probs[:, None] * comps).sum(axis=0)


def generate_vapor_deposition_animation(
    output_path: str = "viz/publication/vapor_deposition.gif",
    T_start: float = 900.0,
    T_end: float = 800.0,
    steps: int = 60,
    n_grid: int = 60,
    figsize: Tuple[float, float] = (11.5, 6.5),
    dpi: int = 120,
    seed: Optional[int] = 42,
    ge_drive_max: float = 2500.0,
) -> str:
    """
    Create an animated GIF simulating vapor deposition (CVD/PVD) during
    semiconductor crystal growth (System 8).

    What it shows (two-panel animation):
    - Left: Process schedule with Temperature T(t) and Ge vapor drive Δμ_Ge(t)
      curves, each with a moving marker.
    - Right: Evolving surface composition fractions (Si, Ge, Vacancy) as a
      stacked bar that updates over frames.

    Modeling approach:
    - Start from create_semiconductor_system() baseline energies.
    - Vapor supply is modeled as a chemical potential drive Δμ_Ge(t) that
      lowers the effective formation energy of Ge over time, encouraging
      incorporation. Vacancies remain energetically unfavorable.
    - At each frame i, define a modified MaterialsSystem with
      G_pure_mod = [G_Si, G_Ge - Δμ_Ge(i), G_Vac] and solve for equilibrium
      at T(i). Expected species fractions are derived and visualized.

    Parameters
    ----------
    output_path : str
        Where to save the GIF.
    T_start, T_end : float
        Start/end temperatures (K) during the deposition segment.
    steps : int
        Number of frames/steps in the animation.
    n_grid : int
        Composition grid resolution for equilibrium computation.
    figsize : Tuple[float, float]
        Figure size in inches.
    dpi : int
        Output resolution.
    seed : Optional[int]
        Random seed (for determinism if future changes introduce randomness).
    ge_drive_max : float
        Maximum chemical potential drive applied to Ge (J/mol).

    Returns
    -------
    str
        Path to the generated GIF.
    """

    if seed is not None:
        np.random.seed(seed)

    # Base semiconductor system
    base: MaterialsSystem = create_semiconductor_system()
    phase_names: List[str] = list(base.phases)  # type: ignore[attr-defined]

    # Process schedule: typically temperature decreases slightly as layer forms,
    # while Ge vapor drive increases to achieve target composition.
    t_vals = np.linspace(0.0, 1.0, steps)
    # Smooth monotonic temperature ramp (slight cool):
    T_vals = T_start + (T_end - T_start) * (t_vals ** 1.2)
    # Ge drive profile (S-shaped ramp-up):
    mu_profile = (1 - np.exp(-5 * t_vals)) / (1 - np.exp(-5))
    dmu_ge_vals = ge_drive_max * mu_profile

    # Precompute composition fractions per frame
    fractions = []
    for T, dmu in zip(T_vals, dmu_ge_vals):
        # Modify energies: lower Ge energy by Δμ (vapor chemical potential)
        G_mod = np.array(base.G_pure, dtype=float)
        # Order is ['Si', 'Ge', 'Vacancy'] as defined in materials_invariance
        G_mod[1] = G_mod[1] - float(dmu)

        system_mod = MaterialsSystem(
            phases=base.phases,
            G_pure=G_mod,
            L_matrix=base.L_matrix,
            regime=base.regime,
        )

        comps, probs = compute_equilibrium_distribution(system_mod, float(T), n_grid)
        comps_arr = np.asarray(comps)
        exp_frac = _expected_species_fractions(comps_arr, probs)
        exp_frac = np.clip(exp_frac, 0.0, 1.0)
        s = exp_frac.sum()
        if s > 0:
            exp_frac = exp_frac / s
        fractions.append(exp_frac)

    fractions_arr = np.vstack(fractions)  # (steps, K)

    # Prepare output dir
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Figure and axes
    fig, (ax_proc, ax_comp) = plt.subplots(1, 2, figsize=figsize)

    # Left: Process schedule T(t) and Δμ_Ge(t)
    ax_proc.set_title("Semiconductor Vapor Deposition Schedule", fontsize=14, fontweight="bold")
    ax_proc.set_xlabel("Normalized time")
    ax_proc.set_ylabel("Temperature (K)")
    ax_proc.grid(alpha=0.3)
    line_T, = ax_proc.plot(t_vals, T_vals, color=BLUE, lw=2, label="T(t)")
    marker_T, = ax_proc.plot([t_vals[0]], [T_vals[0]], "o", color=RED, ms=7)

    # Twin axis for Δμ_Ge
    ax_mu = ax_proc.twinx()
    ax_mu.set_ylabel(r"Δμ$_{Ge}$ (J/mol)")
    line_mu, = ax_mu.plot(t_vals, dmu_ge_vals, color=PURPLE, lw=2, linestyle="--", label=r"Δμ$_{Ge}$(t)")
    marker_mu, = ax_mu.plot([t_vals[0]], [dmu_ge_vals[0]], "s", color=PURPLE, ms=6)

    # Build a combined legend
    lines_1, labels_1 = ax_proc.get_legend_handles_labels()
    lines_2, labels_2 = ax_mu.get_legend_handles_labels()
    ax_proc.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper left", fontsize=9)

    # Right: stacked bar for composition fractions
    ax_comp.set_title("Surface Composition During Vapor Deposition", fontsize=14, fontweight="bold")
    ax_comp.set_xlim(-0.5, 0.5)
    ax_comp.set_ylim(0.0, 1.02)
    ax_comp.set_ylabel("Fraction")
    ax_comp.set_xticks([])
    ax_comp.grid(axis="y", alpha=0.25)

    colors = {
        "Si": BLUE,
        "Ge": GREEN,
        "Vacancy": GRAY,
    }
    facecolors = [colors.get(name, ORANGE) for name in phase_names]

    # Initialize stacked bars at frame 0
    bottoms = 0.0
    bars = []
    for k, name in enumerate(phase_names):
        val = float(fractions_arr[0, k])
        b = ax_comp.bar(0, val, bottom=bottoms, width=0.6, color=facecolors[k],
                        edgecolor="black", label=name, alpha=0.9)
        bars.append(b[0])
        bottoms += val

    ax_comp.legend(loc="upper right", fontsize=10)
    info = ax_comp.text(
        0.02, 0.98, "", transform=ax_comp.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=10
    )

    def init():
        marker_T.set_data([t_vals[0]], [T_vals[0]])
        marker_mu.set_data([t_vals[0]], [dmu_ge_vals[0]])
        bottoms_local = 0.0
        for k in range(len(phase_names)):
            val0 = float(fractions_arr[0, k])
            bars[k].set_height(val0)
            bars[k].set_y(bottoms_local)
            bottoms_local += val0
        info.set_text("")
        return line_T, line_mu, marker_T, marker_mu, *bars, info

    def animate(i: int):
        # Update schedule markers
        marker_T.set_data([t_vals[i]], [T_vals[i]])
        marker_mu.set_data([t_vals[i]], [dmu_ge_vals[i]])

        # Update composition bars
        bottoms_local = 0.0
        for k in range(len(phase_names)):
            val = float(fractions_arr[i, k])
            bars[k].set_height(val)
            bars[k].set_y(bottoms_local)
            bottoms_local += val

        # Info box
        frac_lines = [f"{name}: {fractions_arr[i, idx]:.2f}" for idx, name in enumerate(phase_names)]
        info.set_text(
            (
                f"Deposition frame {i+1}/{steps}\n"
                f"T = {T_vals[i]:.1f} K  |  Δμ_Ge = {dmu_ge_vals[i]:.0f} J/mol\n"
            ) + "\n".join(frac_lines)
        )
        return line_T, line_mu, marker_T, marker_mu, *bars, info

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=steps,
        interval=90,
        blit=True,
        repeat=False,
    )

    writer = animation.PillowWriter(fps=12)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved vapor deposition animation: {output_path}")
    return output_path


if __name__ == "__main__":
    # Default invocation to produce the vapor deposition animation
    generate_vapor_deposition_animation()
