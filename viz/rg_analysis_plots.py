import os
from typing import Optional, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from tpu_benchmark import (
    noisy_tpu,
    reference_tpu_exact,
    run_tpu_benchmark,
    rg_flow_analysis,
)


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
PURPLE = "#6a3d9a"
DARK_RED = "#8B0000"


def generate_rg_flow_animation(
    output_path: str = "viz/publication/rg_flow.gif",
    N: int = 5,
    frames: int = 60,
    max_noise: float = 0.20,
    num_samples: int = 6000,
    alpha: float = 2.0,
    T0: float = 1.0,
    seed: Optional[int] = 42,
    dpi: int = 100,
    figsize: Tuple[float, float] = (7.5, 6.0),
) -> str:
    """
    Create an animated GIF showing RG flow away from the fixed point
    as TPU noise increases (System 9).

    The plot uses D_KL on the x-axis and the beta function on the y-axis
    (beta ≈ D_KL). A perfect TPU sits at the fixed point (0,0). As noise
    increases, the hardware "flows" away from the fixed point along the
    diagonal y=x.

    Parameters
    ----------
    output_path : str
        Where to save the GIF.
    N : int
        System size for the synthetic test problem.
    frames : int
        Number of animation frames.
    max_noise : float
        Maximum noise level for the noisy TPU at the last frame.
    num_samples : int
        Samples per frame for the benchmark (trade-off speed vs. stability).
    alpha, T0 : float
        Scale invariance experiment parameters.
    seed : Optional[int]
        Random seed for reproducibility.
    dpi : int
        Output resolution of the GIF.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    str
        The path to the generated GIF.
    """

    # Build synthetic system
    if seed is not None:
        np.random.seed(seed)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2  # symmetric
    H = np.random.randn(N)

    # Precompute RG points for all frames (deterministic animation)
    noise_levels: List[float] = [i * max_noise / (frames - 1) for i in range(frames)]
    D_vals: List[float] = []
    beta_vals: List[float] = []

    # Include the perfect TPU as the first point
    perfect = run_tpu_benchmark(
        sampler=reference_tpu_exact,
        W=W,
        H=H,
        T0=T0,
        alpha=alpha,
        num_samples=num_samples,
        tpu_name="Perfect",
        verbose=False,
    )
    rg0 = rg_flow_analysis(perfect)
    D_vals.append(rg0["distance_from_fixed_point"])  # ~0
    beta_vals.append(rg0["beta_function"])  # ~0

    for nl in noise_levels[1:]:
        sampler = lambda w, h, t, n, _nl=nl: noisy_tpu(w, h, t, n, _nl)
        result = run_tpu_benchmark(
            sampler=sampler,
            W=W,
            H=H,
            T0=T0,
            alpha=alpha,
            num_samples=num_samples,
            tpu_name=f"Noise={nl:.2f}",
            verbose=False,
        )
        rg = rg_flow_analysis(result)
        D_vals.append(rg["distance_from_fixed_point"])
        beta_vals.append(rg["beta_function"])

    # Prepare output dir
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Set plot limits with margin
    max_D = max(D_vals) if D_vals else 1e-2
    pad = max_D * 0.15 + 1e-8
    x_max = max_D + pad
    y_max = max(beta_vals) + pad

    fig, ax = plt.subplots(figsize=figsize)

    # Static elements
    ax.set_title(
        "RG Fixed Point and Flow (System 9)", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("D_KL (distance from fixed point)", fontsize=12)
    ax.set_ylabel("Beta function (≈ D_KL)", fontsize=12)
    ax.set_xlim(0, max(x_max, 1e-6))
    ax.set_ylim(0, max(y_max, 1e-6))
    ax.grid(True, alpha=0.3)

    # Diagonal y=x reference line
    ref_x = np.linspace(0, ax.get_xlim()[1], 200)
    ax.plot(ref_x, ref_x, linestyle="--", color=BLUE, alpha=0.6, label="β ≈ D_KL")

    # Fixed point
    ax.scatter([0], [0], s=60, c=GREEN, label="Fixed Point (Perfect TPU)")

    # Animated elements
    path_line, = ax.plot([], [], "-o", color=RED, alpha=0.8, lw=2, ms=5,
                         label="Noisy TPU trajectory")
    point_scatter = ax.scatter([], [], s=80, c=PURPLE, zorder=3)

    # Legend
    ax.legend(loc="upper left", fontsize=10)

    # Annotation box for current noise and values
    text_box = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    xs: List[float] = []
    ys: List[float] = []

    def init():
        path_line.set_data([], [])
        point_scatter.set_offsets(np.c_[[], []])
        text_box.set_text("")
        return path_line, point_scatter, text_box

    def animate(i: int):
        # Use precomputed values
        x = D_vals[i]
        y = beta_vals[i]
        xs.append(x)
        ys.append(y)

        path_line.set_data(xs, ys)
        point_scatter.set_offsets(np.c_[[x], [y]])

        # Noise display (approx back-computation of noise level from index)
        nl = noise_levels[i] if i < len(noise_levels) else noise_levels[-1]
        text_box.set_text(
            f"Frame {i+1}/{frames}\nNoise ≈ {nl:.3f}\nD_KL = {x:.6f}\nβ = {y:.6f}"
        )
        return path_line, point_scatter, text_box

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(D_vals),
        interval=80,
        blit=True,
        repeat=False,
    )

    # Save using PillowWriter (GIF)
    writer = animation.PillowWriter(fps=12)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved RG flow animation: {output_path}")
    return output_path


if __name__ == "__main__":
    # Default invocation to produce the RG flow animation
    generate_rg_flow_animation()
