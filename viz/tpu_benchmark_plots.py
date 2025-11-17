import os
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from tpu_benchmark import benchmark_suite, TPUGrade


BLUE = "#1f77b4"
GREEN = "#2ca02c"
ORANGE = "#ff7f0e"
RED = "#d62728"
YELLOW = "#ffdd57"
DARK_GREEN = "#006400"


def generate_tis_ladder_animation(
    output_path: str = "viz/publication/tis_quality_ladder.gif",
    N: int = 5,
    T0: float = 1.0,
    alpha: float = 2.0,
    num_samples: int = 20000,
    seed: Optional[int] = 42,
    dpi: int = 120,
    figsize: Tuple[float, float] = (9.0, 7.0),
    fps: int = 12,
) -> str:
    """
    Create an animated GIF showing the Thermodynamic Integrity Score (TIS)
    quality ladder and reveal benchmarked TPUs one by one.

    - Background: horizontal ladder bands for grades (REFERENCE → FAILED)
    - X-axis: TIS (log scale)
    - Points: TPU results plotted at their TIS value on corresponding grade band
    - Animation: points and labels appear sequentially with a subtle grow/pulse

    Returns the path to the generated GIF.
    """

    # Build synthetic system for benchmarking
    if seed is not None:
        np.random.seed(seed)
    W = np.random.randn(N, N)
    W = (W + W.T) / 2  # symmetric
    H = np.random.randn(N)

    # Run suite to obtain a small set of diverse TPUs
    results = benchmark_suite(W, H, T0=T0, alpha=alpha, num_samples=num_samples)

    # Sort results by descending TIS for a clean reveal order
    items = sorted(results.items(), key=lambda kv: kv[1].tis, reverse=True)

    # Prepare output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Ladder definition (from high to low quality)
    grades = [
        TPUGrade.REFERENCE,
        TPUGrade.EXCELLENT,
        TPUGrade.GOOD,
        TPUGrade.ACCEPTABLE,
        TPUGrade.MARGINAL,
        TPUGrade.FAILED,
    ]
    # Color bands to be readable and consistent
    band_colors = {
        TPUGrade.REFERENCE: "#0a7f2e",
        TPUGrade.EXCELLENT: "#2ca02c",
        TPUGrade.GOOD: "#a6d96a",
        TPUGrade.ACCEPTABLE: "#ffdd57",
        TPUGrade.MARGINAL: "#ff9933",
        TPUGrade.FAILED: "#d62728",
    }

    # TIS thresholds to annotate (approximate boundaries from classify_tpu)
    thresholds = {
        TPUGrade.REFERENCE: 1000,
        TPUGrade.EXCELLENT: 100,
        TPUGrade.GOOD: 31,
        TPUGrade.ACCEPTABLE: 10,
        TPUGrade.MARGINAL: 3,
        TPUGrade.FAILED: 0,
    }

    # Compute y positions for bands (top to bottom)
    y_positions = {grade: (len(grades) - i - 0.5) for i, grade in enumerate(grades)}

    # Determine x-axis limits from observed TIS, expand to comfortable range
    tis_vals = [res.tis for _, res in items]
    x_min = min(tis_vals + [2.5])  # ensure lower bound > 0
    x_max = max(tis_vals + [1500])
    # Pad the range
    x_min = max(1.5, x_min * 0.8)
    x_max = x_max * 1.2

    # Figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Thermodynamic Integrity Score (TIS) Quality Ladder", fontsize=16, fontweight="bold")
    ax.set_xlabel("TIS (higher is better)", fontsize=12)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_yticks([y_positions[g] for g in grades])
    ax.set_yticklabels([g.value for g in grades])
    ax.grid(axis="x", which="both", alpha=0.25)

    # Draw background bands
    for g in grades:
        y = y_positions[g]
        ax.fill_between([x_min, x_max], y - 0.4, y + 0.4, color=band_colors[g], alpha=0.18, zorder=0)
        # Add threshold text at left
        thr = thresholds[g]
        label = f"TIS > {thr}" if g != TPUGrade.FAILED else "TIS < 3"
        ax.text(x_min * 1.02, y, label, va="center", ha="left", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6))

    # Prepare animated artists
    scatters = []
    labels = []

    # Precreate scatter objects (invisible at start)
    for _name, res in items:
        y = y_positions[res.grade]
        sc = ax.scatter([res.tis], [y], s=0, c=BLUE if res.grade not in (TPUGrade.MARGINAL, TPUGrade.FAILED) else RED,
                         edgecolor="black", linewidths=0.5, zorder=3)
        scatters.append(sc)
        txt = ax.text(res.tis, y + 0.42, "", ha="center", va="bottom", fontsize=9)
        labels.append(txt)

    # Annotation box for live info
    info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.85), fontsize=10)

    def init():
        for sc, txt in zip(scatters, labels):
            sc.set_sizes([0])
            txt.set_text("")
        info.set_text("")
        return [*scatters, *labels, info]

    # Each item gets a couple of frames for a grow/pulse effect
    frames_per_item = 6
    total_frames = max(1, frames_per_item * len(items))

    def animate(frame: int):
        idx = min(frame // frames_per_item, len(items) - 1)
        # Reveal all up to idx
        for j in range(idx + 1):
            name, res = items[j]
            sc = scatters[j]
            # Pulse size depending on where we are within the segment
            if j == idx:
                t = (frame % frames_per_item + 1) / frames_per_item
            else:
                t = 1.0
            size = 40 + 80 * t  # grow from 40 to 120
            sc.set_offsets([[res.tis, y_positions[res.grade]]])
            sc.set_sizes([size])
            labels[j].set_position((res.tis, y_positions[res.grade] + 0.42))
            labels[j].set_text(f"{name}\nTIS={res.tis:.1f}  D_KL={res.D_proof:.4g}")

        name, res = items[idx]
        info.set_text(
            f"Showing {idx+1}/{len(items)}\n"
            f"TPU: {name}\n"
            f"Grade: {res.grade.value}\n"
            f"TIS: {res.tis:.2f}  D_KL: {res.D_proof:.6f}"
        )
        return [*scatters, *labels, info]

    anim = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=total_frames,
        interval=int(1000 / fps),
        blit=True,
        repeat=False,
    )

    writer = animation.PillowWriter(fps=fps)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)

    print(f"Saved TIS ladder animation: {output_path}")
    return output_path


if __name__ == "__main__":
    # Default invocation to produce the TIS quality ladder animation
    generate_tis_ladder_animation()
