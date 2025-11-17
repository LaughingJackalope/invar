import os
from typing import Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from materials_invariance import (
    create_sword_system,
    compute_equilibrium_distribution,
    MaterialsSystem,  # type: ignore
)


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
PURPLE = "#6a3d9a"
DARK_RED = "#8B0000"


def _expected_phase_fractions(
    compositions: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    """
    Compute expected phase fractions given a discrete distribution
    over composition simplex.

    Parameters
    ----------
    compositions : np.ndarray
        Array of shape (M, K) with phase fractions per state.
    probabilities : np.ndarray
        Array of shape (M,) with probabilities summing to 1.

    Returns
    -------
    np.ndarray
        Expected phase fractions of shape (K,).
    """
    # Ensure shapes
    probs = np.asarray(probabilities, dtype=float)
    probs = probs / max(probs.sum(), 1e-16)
    comps = np.asarray(compositions, dtype=float)
    return (probs[:, None] * comps).sum(axis=0)


def generate_quenching_animation(
    output_path: str = "viz/publication/quenching.gif",
    T_start: float = 1100.0,
    T_end: float = 300.0,
    steps: int = 60,
    n_grid: int = 60,
    figsize: Tuple[float, float] = (11.0, 6.5),
    dpi: int = 120,
    seed: Optional[int] = 42,
) -> str:
    """
    Create an animated GIF simulating the quenching process for sword steel.

    What it shows (two-panel animation):
    - Left: Cooling schedule T(t) during quench with a moving marker.
    - Right: Evolving phase fractions (Austenite, Martensite, Pearlite) as a
      stacked bar that updates over time.

    The animation uses the thermodynamic model from materials_invariance:
    equilibrium distributions are recomputed at each temperature, and the
    expected phase fractions are derived from those distributions.

    Parameters
    ----------
    output_path : str
        Where to save the GIF.
    T_start : float
        Starting temperature (K) before quench.
    T_end : float
        Final temperature (K) after quench.
    steps : int
        Number of frames/temperature steps in the animation.
    n_grid : int
        Composition grid resolution for equilibrium computation.
    figsize : Tuple[float, float]
        Figure size in inches.
    dpi : int
        Output resolution.
    seed : Optional[int]
        Random seed (not strictly needed here but kept for reproducibility
        if the underlying model ever uses stochastic elements).

    Returns
    -------
    str
        Path to the generated GIF.
    """

    if seed is not None:
        np.random.seed(seed)

    # Prepare system (sword forging case)
    system: MaterialsSystem = create_sword_system()
    phase_names: List[str] = list(system.phases)  # type: ignore[attr-defined]

    # Cooling schedule: fast non-linear quench (exponential-like)
    # Parameterize time t in [0,1], temperature T(t) transitions quickly early on
    t_vals = np.linspace(0.0, 1.0, steps)
    cool_profile = (1 - np.exp(-6 * t_vals)) / (1 - np.exp(-6))  # smooth S-curve
    T_vals = T_start + (T_end - T_start) * cool_profile

    # Precompute expected phase fractions at each temperature (deterministic)
    fractions = []  # list of arrays (K,)
    for T in T_vals:
        comps, probs = compute_equilibrium_distribution(system, float(T), n_grid)
        comps_arr = np.asarray(comps)
        exp_frac = _expected_phase_fractions(comps_arr, probs)
        # Numerical safety and normalization
        exp_frac = np.clip(exp_frac, 0.0, 1.0)
        s = exp_frac.sum()
        if s > 0:
            exp_frac = exp_frac / s
        fractions.append(exp_frac)

    fractions_arr = np.vstack(fractions)  # (steps, K)

    # Output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Figure and axes
    fig, (ax_T, ax_phase) = plt.subplots(1, 2, figsize=figsize)

    # Left: temperature vs time
    ax_T.set_title("Quenching Schedule", fontsize=14, fontweight="bold")
    ax_T.set_xlabel("Normalized time")
    ax_T.set_ylabel("Temperature (K)")
    ax_T.grid(alpha=0.3)
    line_T, = ax_T.plot(t_vals, T_vals, color=BLUE, lw=2, label="T(t)")
    marker_T, = ax_T.plot([t_vals[0]], [T_vals[0]], "o", color=RED, ms=8)
    ax_T.legend(loc="best", fontsize=10)

    # Right: stacked bar of phase fractions
    ax_phase.set_title("Phase Transformation During Quench", fontsize=14, fontweight="bold")
    ax_phase.set_xlim(-0.5, 0.5)
    ax_phase.set_ylim(0.0, 1.02)
    ax_phase.set_ylabel("Phase fraction")
    ax_phase.set_xticks([])
    ax_phase.grid(axis="y", alpha=0.25)

    colors = {
        "Austenite": PURPLE,
        "Martensite": DARK_RED,
        "Pearlite": ORANGE,
    }
    # Ensure consistent color order matching phase_names
    facecolors = [colors.get(name, GREEN) for name in phase_names]

    # Initialize stacked bars
    bottoms = 0.0
    bars = []
    for k, name in enumerate(phase_names):
        val = float(fractions_arr[0, k])
        b = ax_phase.bar(0, val, bottom=bottoms, width=0.6, color=facecolors[k],
                         edgecolor="black", label=name, alpha=0.85)
        bars.append(b[0])
        bottoms += val

    # Legend and info box
    ax_phase.legend(loc="upper right", fontsize=10)
    info = ax_phase.text(
        0.02, 0.98, "", transform=ax_phase.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=10
    )

    def init():
        marker_T.set_data([t_vals[0]], [T_vals[0]])
        # Reset stacked bars
        bottoms_local = 0.0
        for k in range(len(phase_names)):
            val0 = float(fractions_arr[0, k])
            bars[k].set_height(val0)
            bars[k].set_y(bottoms_local)
            bottoms_local += val0
        info.set_text("")
        return line_T, marker_T, *bars, info

    def animate(i: int):
        # Update temperature marker
        marker_T.set_data([t_vals[i]], [T_vals[i]])

        # Update stacked bar heights for frame i
        bottoms_local = 0.0
        for k in range(len(phase_names)):
            val = float(fractions_arr[i, k])
            bars[k].set_height(val)
            bars[k].set_y(bottoms_local)
            bottoms_local += val

        # Build info text with current fractions
        frac_lines = [f"{name}: {fractions_arr[i, idx]:.2f}" for idx, name in enumerate(phase_names)]
        info.set_text(
            "Quench frame {}/{}\nT = {:.1f} K\n".format(i + 1, steps, T_vals[i]) + "\n".join(frac_lines)
        )
        return line_T, marker_T, *bars, info

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

    print(f"Saved quenching animation: {output_path}")
    return output_path


if __name__ == "__main__":
    # Default invocation to produce the quenching animation
    generate_quenching_animation()
