"""
Unhinged but beautiful animations powered by the sampler_interface.

These animations are intentionally artistic rather than purely expository.
They use any Boltzmann-compatible sampler (see sampler_interface.BoltzmannSampler)
to generate probability distributions which are then mapped to visual spaces
using bitwise space-filling curves and color fields.

Quick start (saves three GIFs into viz/viz/publication):

    python -m viz.unhinged_sampler_animations

"""

from typing import Optional, Tuple, Callable
import os
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.transforms import Affine2D

from sampler_interface import BoltzmannSampler, create_default_sampler
from scale_invariance import quantify_divergence


# ---------- Utilities ----------

def _ensure_dir(path: str):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)


def _next_power_of_two(n: int) -> int:
    return 1 << (n - 1).bit_length()


def _index_to_morton_xy(idx: int, nbits: int) -> Tuple[int, int]:
    """
    Map a linear state index to 2D grid using Morton (Z-order) by interleaving bits.

    We treat the lowest 2*nbits of idx as interleaved bits for x and y.
    This creates a space-filling curve layout that looks organic when animated.
    """
    x = 0
    y = 0
    for i in range(nbits):
        x |= ((idx >> (2 * i)) & 1) << i
        y |= ((idx >> (2 * i + 1)) & 1) << i
    return x, y


def _distribution_to_grid(P: np.ndarray) -> np.ndarray:
    """
    Project a probability vector onto a square grid via Morton mapping.

    If len(P) is not a perfect square of a power of two, zero-pad to next 2^(2k).
    """
    L = int(len(P))
    # Find smallest k with 2^(2k) >= L
    k = 0
    while (1 << (2 * k)) < L:
        k += 1
    side = 1 << k
    grid = np.zeros((side, side), dtype=float)
    for i, p in enumerate(P):
        x, y = _index_to_morton_xy(i, k)
        grid[y, x] = p  # row = y for imshow orientation
    # Normalize for safety
    s = grid.sum()
    if s > 0:
        grid = grid / s
    return grid


def _nz_norm(P: np.ndarray) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.maximum(P, 1e-18)
    return P / P.sum()


def _generate_W_H(N: int, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(N, N))
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)
    H = rng.normal(size=(N,))
    return W, H


# ---------- Animations ----------

def animate_probability_carpet(
    output_path: str = "viz/viz/publication/unhinged_probability_carpet.gif",
    N: int = 8,
    frames: int = 120,
    T_base: float = 1.0,
    sampler: Optional[BoltzmannSampler] = None,
    seeds: Optional[Tuple[int, int]] = (123, 456),
    dpi: int = 120,
    figsize: Tuple[float, float] = (6.5, 6.5),
) -> str:
    """
    Psychedelic probability carpet.

    Over time, the bias vector H(t) undulates smoothly; the sampler converts
    each (W, H(t), T(t)) into a distribution P which is splatted onto a square
    using Morton mapping and colored with a cyclic colormap.
    """
    sampler = sampler or create_default_sampler()
    W, H0 = _generate_W_H(N, seed=seeds[0] if seeds else None)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Unhinged Probability Carpet", fontsize=14, fontweight="bold")
    ax.axis("off")

    # Initial frame
    P0 = _nz_norm(sampler.sample_distribution(W, H0, T_base, num_samples=8000))
    grid0 = _distribution_to_grid(P0)
    im = ax.imshow(grid0, cmap="turbo", interpolation="bilinear", vmin=0, vmax=grid0.max()*1.05)

    txt = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.65), fontsize=9
    )

    def animate(i: int):
        t = i / max(frames - 1, 1)
        # Smoothly morph H and T with low-frequency sines
        phase = 2 * math.pi * t
        Ht = H0 * (0.6 + 0.4 * math.sin(phase))
        # Add a drifting direction
        drift = np.sin(phase * 0.5 + np.arange(N) * 0.9)
        Ht = Ht + 0.5 * drift
        Tt = T_base * (0.6 + 0.8 * (0.5 + 0.5 * math.sin(phase * 0.7 + 1.3)))

        P = _nz_norm(sampler.sample_distribution(W, Ht, Tt, num_samples=6000))
        G = _distribution_to_grid(P)
        im.set_data(G)
        im.set_clim(vmin=0.0, vmax=max(1e-12, G.max() * 1.05))
        txt.set_text(f"Frame {i+1}/{frames}\nT = {Tt:.3f}")
        return [im, txt]

    _ensure_dir(output_path)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=60, blit=True)
    writer = animation.PillowWriter(fps=15)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def animate_graycode_trails(
    output_path: str = "viz/viz/publication/unhinged_graycode_trails.gif",
    N: int = 9,
    frames: int = 200,
    sampler: Optional[BoltzmannSampler] = None,
    seed: Optional[int] = 7,
    steps_per_frame: int = 200,
    dpi: int = 120,
    figsize: Tuple[float, float] = (7.5, 6.0),
) -> str:
    """
    Wander along the state space using a probability-guided Gray-code walk.

    We draw glowing trails over a 2D Morton grid where step choices are biased
    by the current probability mass. The result looks like aurora threads.
    """
    rng = np.random.default_rng(seed)
    sampler = sampler or create_default_sampler()
    W, H = _generate_W_H(N, seed=seed)

    P = _nz_norm(sampler.sample_distribution(W, H, T=1.0, num_samples=12000))
    G = _distribution_to_grid(P)
    side = G.shape[0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Gray-Code Probability Trails", fontsize=14, fontweight="bold")
    ax.axis("off")
    base = ax.imshow(G, cmap="magma", interpolation="bilinear", alpha=0.5)
    trail = np.zeros_like(G)
    glow = ax.imshow(trail, cmap="plasma", interpolation="bilinear", alpha=0.8, vmin=0, vmax=1)

    # Start near a high-probability index
    idx = int(np.argmax(P))

    def neighbors_gray(i: int, nbits: int) -> np.ndarray:
        # Flip one bit at a time (Hamming distance 1)
        return np.array([i ^ (1 << b) for b in range(nbits)])

    def animate(f: int):
        nonlocal idx, trail, P
        nbits = int(math.ceil(math.log2(len(P))))
        for _ in range(steps_per_frame):
            nbrs = neighbors_gray(idx, nbits)
            probs = P[nbrs]
            probs = probs / probs.sum()
            idx = int(rng.choice(nbrs, p=probs))
            x, y = _index_to_morton_xy(idx, int(math.log2(side)))
            trail[y, x] = min(1.0, trail[y, x] + 0.06)

        # Temporal decay for glow effect
        trail *= 0.96
        glow.set_data(trail)
        glow.set_clim(vmin=0, vmax=max(1e-6, trail.max()))
        # Slowly remix P to keep motion interesting
        if f % 20 == 0:
            H[:] = H * 0.9 + rng.normal(scale=0.3, size=H.shape)
            P = _nz_norm(sampler.sample_distribution(W, H, T=1.0, num_samples=8000))
        return [glow, base]

    _ensure_dir(output_path)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=50, blit=True)
    writer = animation.PillowWriter(fps=20)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def animate_scale_waves(
    output_path: str = "viz/viz/publication/unhinged_scale_waves.gif",
    N: int = 7,
    frames: int = 140,
    T0: float = 1.0,
    sampler: Optional[BoltzmannSampler] = None,
    alpha_range: Tuple[float, float] = (0.5, 2.5),
    dpi: int = 120,
    figsize: Tuple[float, float] = (7.5, 4.8),
) -> str:
    """
    Hypnotic scale waves: sweep alpha(t) and render two synchronized views.

    Top: P(W,H,T0) as a reference waterfall plot.
    Bottom: P(αW, αH, αT0) — should visually lock to the reference as alpha varies,
    producing moiré-like waves when overlaid in color.
    """
    sampler = sampler or create_default_sampler()
    W, H = _generate_W_H(N, seed=999)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    for ax in (ax1, ax2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax1.set_title("Reference P(W,H,T0)", fontsize=12, fontweight="bold")
    ax2.set_title("Scaled P(αW,αH,αT0)", fontsize=12, fontweight="bold")

    Pref = _nz_norm(sampler.sample_distribution(W, H, T0, num_samples=12000))
    Gref = _distribution_to_grid(Pref)
    im1 = ax1.imshow(Gref, cmap="cividis", interpolation="bilinear")

    im2 = ax2.imshow(Gref, cmap="viridis", interpolation="bilinear")
    txt = ax2.text(0.01, 0.99, "", transform=ax2.transAxes, va="top",
                   bbox=dict(boxstyle="round", facecolor="white", alpha=0.7), fontsize=9)

    def animate(i: int):
        # Oscillate alpha in range
        a, b = alpha_range
        t = i / max(frames - 1, 1)
        alpha = a + (b - a) * 0.5 * (1 + math.sin(2 * math.pi * t))
        Psc = _nz_norm(sampler.sample_distribution(alpha * W, alpha * H, alpha * T0, num_samples=12000))
        Gsc = _distribution_to_grid(Psc)
        im2.set_data(Gsc)
        im2.set_clim(vmin=0, vmax=max(1e-6, Gsc.max()))
        txt.set_text(f"α = {alpha:.3f}")
        return [im2, txt]

    _ensure_dir(output_path)
    anim = animation.FuncAnimation(fig, animate, frames=frames, interval=60, blit=True)
    writer = animation.PillowWriter(fps=15)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def animate_scale_islands(
    output_path: str = "viz/viz/publication/unhinged_scale_islands.gif",
    N: int = 7,
    frames: int = 180,
    T0: float = 1.0,
    sampler: Optional[BoltzmannSampler] = None,
    alpha_range: Tuple[float, float] = (0.1, 10.0),
    eps: float = 7e-3,
    dpi: int = 120,
    figsize: Tuple[float, float] = (8.5, 6.2),
) -> str:
    """
    Scale Islands: sweep α across decades while the probability distribution holds.

    - Top-left: Reference grid P(W,H,T0)
    - Top-right: Scaled grid P(αW, αH, αT0)
    - Bottom: D_KL timeline vs frame with threshold ε; segments below ε are
      highlighted as "ISLANDS" to show regions where distributions are effectively
      identical despite large α changes.

    Parameters
    ----------
    alpha_range : (float, float)
        Start and end α for the sweep; allow wide range to emphasize stability.
    eps : float
        Island threshold on D_KL(reference || scaled).
    """
    sampler = sampler or create_default_sampler()
    W, H = _generate_W_H(N, seed=1234)

    # Reference distribution and grid
    Pref = _nz_norm(sampler.sample_distribution(W, H, T0, num_samples=14000))
    Gref = _distribution_to_grid(Pref)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.65])
    ax_ref = fig.add_subplot(gs[0, 0])
    ax_scl = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, :])

    for ax in (ax_ref, ax_scl):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    ax_ref.set_title("Reference P(W,H,T0)", fontsize=12, fontweight="bold")
    ax_scl.set_title("Scaled P(αW,αH,αT0)", fontsize=12, fontweight="bold")

    im_ref = ax_ref.imshow(Gref, cmap="cividis", interpolation="bilinear")
    im_scl = ax_scl.imshow(Gref, cmap="viridis", interpolation="bilinear")
    badge = ax_scl.text(
        0.02, 0.02, "", transform=ax_scl.transAxes, va="bottom", ha="left",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.85)
    )
    label = ax_scl.text(
        0.98, 0.02, "", transform=ax_scl.transAxes, va="bottom", ha="right",
        fontsize=10, bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # Timeline setup
    ax_tl.set_title("D_KL(reference || scaled) across α sweep — stable islands where D_KL < ε",
                    fontsize=12, fontweight="bold")
    ax_tl.set_xlabel("Frame / Time → α sweep")
    ax_tl.set_ylabel("D_KL")
    ax_tl.grid(alpha=0.25)
    line, = ax_tl.plot([], [], color="#1f77b4", lw=2)
    thr = ax_tl.axhline(eps, color="#ff7f0e", ls="--", lw=1.5, label=f"ε = {eps}")
    fill = None
    ax_tl.legend(loc="upper right", fontsize=9)

    # Buffers for timeline
    d_vals = np.zeros(frames, dtype=float)
    a_vals = np.zeros(frames, dtype=float)

    # Precompute α schedule: smooth log-sweep to cover orders of magnitude
    a0, a1 = alpha_range
    # Use sinusoidal in log-space for pleasant pacing
    log_a0, log_a1 = math.log(a0), math.log(a1)
    def alpha_at(j: int) -> float:
        t = j / max(frames - 1, 1)
        # ease in-out using cosine
        te = 0.5 - 0.5 * math.cos(math.pi * t)
        return math.exp(log_a0 + (log_a1 - log_a0) * te)

    # Set y-limit conservative; will autoscale upward as needed
    ax_tl.set_ylim(0, max(5*eps, 0.02))
    ax_tl.set_xlim(0, frames - 1)

    def init():
        nonlocal fill
        line.set_data([], [])
        if fill is not None:
            for coll in fill.collections:
                coll.remove()
        fill = None
        badge.set_text("")
        label.set_text("")
        return [im_ref, im_scl, line, thr, badge, label]

    def animate(i: int):
        nonlocal fill
        alpha = alpha_at(i)
        a_vals[i] = alpha
        Psc = _nz_norm(sampler.sample_distribution(alpha * W, alpha * H, alpha * T0, num_samples=12000))
        Gsc = _distribution_to_grid(Psc)
        im_scl.set_data(Gsc)
        im_scl.set_clim(vmin=0, vmax=max(1e-6, Gsc.max()))

        dkl = float(quantify_divergence(Pref, Psc))
        d_vals[i] = dkl
        xs = np.arange(i + 1)
        line.set_data(xs, d_vals[: i + 1])

        # Adjust Y if needed
        ymax = max(ax_tl.get_ylim()[1], d_vals[: i + 1].max() * 1.2, 5*eps)
        ax_tl.set_ylim(0, ymax)

        # Highlight island segments where below eps
        if fill is not None:
            for coll in fill.collections:
                coll.remove()
        below = d_vals[: i + 1] < eps
        if below.any():
            # Construct areas where below is True
            x_fill = xs[below]
            y_fill = d_vals[: i + 1][below]
            fill = ax_tl.fill_between(x_fill, 0, y_fill, color="lightgreen", alpha=0.5, step="mid")
        else:
            fill = None

        # Badge when inside island
        if dkl < eps:
            badge.set_text("ISLAND: invariance holds")
        else:
            badge.set_text("")
        label.set_text(f"α = {alpha:.3f}\nD_KL = {dkl:.5f}")

        return [im_scl, line, badge, label]

    _ensure_dir(output_path)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=60, blit=False)
    writer = animation.PillowWriter(fps=15)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


def animate_oblique_isomorphism(
    output_path: str = "viz/viz/publication/unhinged_oblique_isomorphism.gif",
    N: int = 7,
    frames: int = 160,
    T0: float = 1.0,
    sampler: Optional[BoltzmannSampler] = None,
    alpha_range: Tuple[float, float] = (0.25, 4.0),
    dpi: int = 120,
    figsize: Tuple[float, float] = (10.0, 5.6),
) -> str:
    """
    Extremely oblique visualization of scale invariance as an isomorphism.

    Concept: Render probability textures at a bizarre, oblique (sheared/rotated)
    angle and overlay pairs to show that the scale-transformed law preserves the
    *same* texture under a matching transformation, while a control breaks it.

    Left panel (Law):
        Overlay Pref = P(W,H,T0) and Ptest = P(αW,αH,αT0) using an α-dependent
        oblique transform. They should visually coincide (blue+green → cyan).

    Right panel (Control):
        Overlay Pref with Pctrl = P(αW,αH,T0) under the *same* transform. They
        visibly differ (orange contour vs blue base).

    Bottom annotations include D_KL values per frame.
    """
    sampler = sampler or create_default_sampler()
    W, H = _generate_W_H(N, seed=2024)

    # Reference distribution/texture
    Pref = _nz_norm(sampler.sample_distribution(W, H, T0, num_samples=14000))
    Gref = _distribution_to_grid(Pref)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.25])
    ax_law = fig.add_subplot(gs[0, 0])
    ax_ctrl = fig.add_subplot(gs[0, 1])
    ax_txt = fig.add_subplot(gs[1, :])

    for ax in (ax_law, ax_ctrl):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.grid(False)

    ax_law.set_title("Oblique Isomorphism — Scale Law (αW,αH,αT)", fontsize=12, fontweight="bold")
    ax_ctrl.set_title("Oblique Control — Broken Scaling (αW,αH,T)", fontsize=12, fontweight="bold")

    # Placeholders for artists we will update
    # Use extent in Axes coords (0..1) to simplify transforms in Axes frame
    im_ref_law = ax_law.imshow(Gref, extent=(0, 1, 0, 1), cmap="Blues", alpha=0.75,
                               interpolation="bilinear", zorder=1)
    im_tst_law = ax_law.imshow(Gref, extent=(0, 1, 0, 1), cmap="Greens", alpha=0.55,
                               interpolation="bilinear", zorder=2)

    im_ref_ctrl = ax_ctrl.imshow(Gref, extent=(0, 1, 0, 1), cmap="Blues", alpha=0.75,
                                 interpolation="bilinear", zorder=1)
    im_tst_ctrl = ax_ctrl.imshow(Gref, extent=(0, 1, 0, 1), cmap="Oranges", alpha=0.55,
                                 interpolation="bilinear", zorder=2)

    # Text area for divergence readouts
    ax_txt.axis('off')
    box = ax_txt.text(
        0.02, 0.5, "", transform=ax_txt.transAxes, va="center", ha="left",
        fontsize=11, bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    # Prepare α log-sweep and a weird camera path
    a0, a1 = alpha_range
    log_a0, log_a1 = math.log(a0), math.log(a1)

    def alpha_at(j: int) -> float:
        t = j / max(frames - 1, 1)
        te = 0.5 - 0.5 * math.cos(math.pi * t)  # smooth in-out
        return math.exp(log_a0 + (log_a1 - log_a0) * te)

    def oblique_transform(j: int) -> Affine2D:
        # Time parameter
        t = j / max(frames - 1, 1)
        # Wild but smooth path in transform space
        rot = 25.0 * math.sin(2 * math.pi * t)
        sky = 0.8 * math.cos(2 * math.pi * t * 0.5)
        skx = 0.5 * math.sin(2 * math.pi * t * 0.33 + 1.0)
        scy = 0.6 + 0.25 * math.sin(2 * math.pi * t * 0.77 + 0.5)
        scx = 1.0
        # Centered transform
        T = Affine2D()
        T.translate(-0.5, -0.5)
        T.skew_deg(skx=skx * 45.0, sky=sky * 45.0)
        T.rotate_deg(rot)
        T.scale(scx, scy)
        T.translate(0.5, 0.5 - 0.15)  # drop slightly for perspective feel
        return T

    def init():
        # Initial transform
        T = oblique_transform(0)
        im_ref_law.set_transform(T + ax_law.transAxes)
        im_tst_law.set_transform(T + ax_law.transAxes)
        im_ref_ctrl.set_transform(T + ax_ctrl.transAxes)
        im_tst_ctrl.set_transform(T + ax_ctrl.transAxes)
        box.set_text("")
        return [im_ref_law, im_tst_law, im_ref_ctrl, im_tst_ctrl, box]

    def animate(j: int):
        # Compute α and transformed distributions
        alpha = alpha_at(j)
        Ptest = _nz_norm(sampler.sample_distribution(alpha * W, alpha * H, alpha * T0, num_samples=12000))
        Pctrl = _nz_norm(sampler.sample_distribution(alpha * W, alpha * H, T0, num_samples=12000))

        Gtest = _distribution_to_grid(Ptest)
        Gctrl = _distribution_to_grid(Pctrl)

        # Update textures
        im_tst_law.set_data(Gtest)
        im_tst_law.set_clim(vmin=0, vmax=max(1e-6, Gtest.max()))
        im_tst_ctrl.set_data(Gctrl)
        im_tst_ctrl.set_clim(vmin=0, vmax=max(1e-6, Gctrl.max()))

        # Update camera transform for both panels
        T = oblique_transform(j)
        im_ref_law.set_transform(T + ax_law.transAxes)
        im_tst_law.set_transform(T + ax_law.transAxes)
        im_ref_ctrl.set_transform(T + ax_ctrl.transAxes)
        im_tst_ctrl.set_transform(T + ax_ctrl.transAxes)

        # Divergences
        d_proof = float(quantify_divergence(Pref, Ptest))
        d_ctrl = float(quantify_divergence(Pref, Pctrl))
        box.set_text(
            f"α = {alpha:.3f}\n"
            f"Law panel: D_KL(Pref || Pα,α,α) = {d_proof:.6f} (invariance)\n"
            f"Control:   D_KL(Pref || Pα,α,T0) = {d_ctrl:.6f} (broken)"
        )

        return [im_tst_law, im_tst_ctrl, im_ref_law, im_ref_ctrl, box]

    _ensure_dir(output_path)
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames, interval=60, blit=False)
    writer = animation.PillowWriter(fps=15)
    anim.save(output_path, writer=writer, dpi=dpi)
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    paths = []
    try:
        paths.append(animate_probability_carpet())
    except Exception as e:
        print(f"Failed to generate probability carpet: {e}")
    try:
        paths.append(animate_graycode_trails())
    except Exception as e:
        print(f"Failed to generate graycode trails: {e}")
    try:
        paths.append(animate_scale_waves())
    except Exception as e:
        print(f"Failed to generate scale waves: {e}")
    try:
        paths.append(animate_scale_islands())
    except Exception as e:
        print(f"Failed to generate scale islands: {e}")
    try:
        paths.append(animate_oblique_isomorphism())
    except Exception as e:
        print(f"Failed to generate oblique isomorphism: {e}")
    for p in paths:
        print("Saved:", p)
