import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from scale_invariance import run_scale_invariance_test, quantify_divergence


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"


def generate_scale_invariance_animation(
    output_path: str = "viz/publication/scale_invariance_proof.gif",
    N: int = 5,
    alpha: float = 2.0,
    T0: float = 1.0,
    num_samples: int = 50000,
    seed: Optional[int] = 42,
    frames: int = 60,
    dpi: int = 120,
    figsize: Tuple[float, float] = (12, 6),
    sampler=None,
) -> str:
    """
    Create an animated GIF demonstrating the Scale Invariance Proof (Priority 1).

    Shows three distributions:
    - Case A: Original (W, H, T)
    - Case B: Control (αW, αH, T)
    - Case C: Test (αW, αH, αT)

    The animation reveals bars progressively and overlays live D_KL annotations,
    highlighting that Case A and Case C match while Case B differs.

    Parameters
    ----------
    sampler : Optional[BoltzmannSampler or callable]
        If provided, this backend will be used by the underlying
        run_scale_invariance_test to generate distributions. This allows
        plugging in hardware-accelerated samplers (e.g., ThrmlSampler)
        via the common sampler_interface. Defaults to None (uses NumPy
        reference sampler).
    """

    # Compute once (deterministic data for animation)
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N=N, alpha=alpha, T0=T0, num_samples=num_samples, seed=seed, sampler=sampler
    )

    # Safety: ensure same length
    L = int(min(len(P_orig), len(P_scaled_E), len(P_test)))
    P_orig = np.asarray(P_orig[:L])
    P_scaled_E = np.asarray(P_scaled_E[:L])
    P_test = np.asarray(P_test[:L])

    # Normalize explicitly (should already be normalized)
    def nz(x):
        x = np.maximum(x, 1e-15)
        return x / x.sum()

    P_orig = nz(P_orig)
    P_scaled_E = nz(P_scaled_E)
    P_test = nz(P_test)

    # Precompute divergences
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)

    # Prepare output dir
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    indices = np.arange(L)
    width = 0.8

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=figsize)

    # Left: three-panel bars (A, B, C)
    ax_left.set_title("Scale Invariance: Cases A, B (control), C (test)", fontsize=14, fontweight="bold")
    ax_left.set_xlabel("State Index")
    ax_left.set_ylabel("Probability")
    ax_left.grid(alpha=0.2)

    # We will draw stacked panels using offsets in x with gaps between groups
    # For simplicity, split the index range into 3 equal chunks for visual grouping
    # but plot same distributions scaled by an interpolation factor.
    bars_A = ax_left.bar(indices, np.zeros_like(P_orig), color=BLUE, alpha=0.6, label="Case A: Original")
    bars_B = ax_left.bar(indices, np.zeros_like(P_scaled_E), color=ORANGE, alpha=0.6, label="Case B: Control")
    bars_C = ax_left.bar(indices, np.zeros_like(P_test), color=GREEN, alpha=0.6, label="Case C: Test")
    ax_left.legend(fontsize=10, loc="upper right")

    # Right: overlay plot (bars + lines)
    ax_right.set_title("Perfect Overlap: Original vs Test", fontsize=14, fontweight="bold")
    ax_right.set_xlabel("State Index")
    ax_right.set_ylabel("Probability")
    ax_right.grid(alpha=0.2)

    bar_ref = ax_right.bar(indices, np.zeros_like(P_orig), color=BLUE, alpha=0.5, label="Original (A)")
    line_test, = ax_right.plot([], [], "o-", color=GREEN, linewidth=2, markersize=4, label="Test (C)")
    line_ctrl, = ax_right.plot([], [], "s--", color=RED, linewidth=1, markersize=3, alpha=0.6, label="Control (B)")
    ax_right.legend(fontsize=10, loc="upper right")

    # Text boxes for D_KL
    txt_left = ax_left.text(
        0.02, 0.98, "", transform=ax_left.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10
    )
    txt_right = ax_right.text(
        0.02, 0.98, "", transform=ax_right.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=10
    )

    # Set consistent y-limits with a small headroom
    ymax = float(max(P_orig.max(), P_scaled_E.max(), P_test.max()) * 1.15)
    ax_left.set_ylim(0, ymax)
    ax_right.set_ylim(0, ymax)

    def init():
        # zero heights
        for b in list(bars_A) + list(bars_B) + list(bars_C) + list(bar_ref):
            b.set_height(0.0)
        line_test.set_data([], [])
        line_ctrl.set_data([], [])
        txt_left.set_text("")
        txt_right.set_text("")
        return (
            (*bars_A, *bars_B, *bars_C, *bar_ref, line_test, line_ctrl, txt_left, txt_right)
        )

    def animate(frame: int):
        # Interpolation factor from 0→1
        t = (frame + 1) / frames

        # Update left bars
        for i, b in enumerate(bars_A):
            b.set_height(P_orig[i] * t)
        for i, b in enumerate(bars_B):
            b.set_height(P_scaled_E[i] * t)
        for i, b in enumerate(bars_C):
            b.set_height(P_test[i] * t)

        # Update right bars/lines
        for i, b in enumerate(bar_ref):
            b.set_height(P_orig[i] * t)
        x = indices
        line_test.set_data(x, P_test * t)
        line_ctrl.set_data(x, P_scaled_E * t)

        # Update texts
        txt_left.set_text(
            f"Frame {frame+1}/{frames}\n"
            f"D_KL(A||C) = {D_proof:.6f} (invariance)\n"
            f"D_KL(A||B) = {D_control:.6f} (control)"
        )
        txt_right.set_text("Original vs Test should overlap (green on blue)")

        return (
            (*bars_A, *bars_B, *bars_C, *bar_ref, line_test, line_ctrl, txt_left, txt_right)
        )

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=frames,
        interval=80,
        blit=True,
        repeat=False,
    )

    writer = animation.PillowWriter(fps=12)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved Scale Invariance animation: {output_path}")
    return output_path


if __name__ == "__main__":
    generate_scale_invariance_animation()
