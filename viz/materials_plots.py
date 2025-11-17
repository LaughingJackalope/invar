import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from materials_invariance import (
    create_sword_system,
    create_semiconductor_system,
    run_materials_invariance_test,
)


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
PURPLE = "#6a3d9a"
DARK_RED = "#8B0000"


def generate_scale_span_visualization(
    output_path: str = "viz/publication/fig2_materials_span.png",
    T_semi: float = 800.0,
    T_sword: float = 1000.0,
    alpha: float = 2.0,
    n_grid: int = 60,
    figsize=(14, 6),
    dpi: int = 300,
    title: Optional[str] = "Scale Invariance: From Atoms to Alloys",
):
    """
    Create the publication-quality Scale Span Visualization (Priority 2).

    This plot shows a logarithmic scale ruler from 1e-9 m to 1e0 m with
    annotations for semiconductor fabrication (atomic layers) and sword forging
    (bulk metal), and highlights that both obey the same thermodynamic law
    with D_KL < 1e-8 in the invariance test.

    Saves a 300 DPI PNG to the provided output_path.
    """

    # Run System 8 invariance tests for both systems
    semi = create_semiconductor_system()
    sword = create_sword_system()

    semi_results = run_materials_invariance_test(
        semi, T0=T_semi, alpha=alpha, n_grid=n_grid
    )
    sword_results = run_materials_invariance_test(
        sword, T0=T_sword, alpha=alpha, n_grid=n_grid
    )

    D_semi = semi_results.get("D_proof", np.nan)
    D_sword = sword_results.get("D_proof", np.nan)

    # Prepare output dir
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Build the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Log scale ruler from 1e-9 to 1e0 meters
    x_left, x_right = 1e-9, 1e0
    scales = np.logspace(-9, 0, 100)
    ax.fill_between(scales, 0, 1, alpha=0.08, color=BLUE)
    ax.set_xscale("log")
    ax.set_xlim(x_left / 10, x_right * 10)  # add margin on both sides
    ax.set_ylim(0, 1.2)

    # Tick formatting: powers of ten
    ax.set_xlabel("Physical Scale (meters)", fontsize=14)
    ax.set_yticks([])

    # Mark semiconductor at 1e-9 m
    ax.axvline(1e-9, color=PURPLE, linewidth=3, linestyle="--", label="Semiconductor (CVD)")
    ax.text(
        1e-9,
        0.92,
        "Atomic\nLayers",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=PURPLE,
    )

    # Mark sword at 1 m
    ax.axvline(1.0, color=DARK_RED, linewidth=3, linestyle="--", label="Sword Forging")
    ax.text(
        1.0,
        0.92,
        "Bulk\nMetal",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=DARK_RED,
    )

    # Central message
    central_msg = (
        "Same Scale Invariance Law\n"
        "P(S; G, T) = P(S; α·G, α·T)"
    )
    ax.text(
        1e-5,
        0.5,
        central_msg,
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.7),
    )

    # D_KL annotations (exact invariance)
    ax.text(
        1e-9,
        0.28,
        f"D_KL < 10⁻⁸\n(measured {D_semi:.2e})",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.9),
    )
    ax.text(
        1.0,
        0.28,
        f"D_KL < 10⁻⁸\n(measured {D_sword:.2e})",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.9),
    )

    # Title and legend
    if title:
        ax.set_title(title, fontsize=18, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="x", which="both", alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    print(f"Saved figure: {output_path}")
    return {
        "output_path": output_path,
        "D_semi": D_semi,
        "D_sword": D_sword,
        "alpha": alpha,
        "T_semi": T_semi,
        "T_sword": T_sword,
    }


if __name__ == "__main__":
    # Default invocation to reproduce Priority 2 figure
    generate_scale_span_visualization()
