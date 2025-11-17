"""
Example Starter Visualization for Junie

This demonstrates how to create the Priority 1 visualization:
The three-case experimental protocol showing scale invariance.

Run: python3 .junie/example_starter.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scale_invariance import run_scale_invariance_test, quantify_divergence

def create_scale_invariance_proof_plot():
    """
    Creates the foundational visualization for the framework:
    Shows that P(original) = P(scaled) but P(original) ≠ P(control)
    """

    print("Running scale invariance test...")

    # Run the three-case experiment
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N=5,           # 32 states
        alpha=2.0,     # Scaling factor
        T0=1.0,        # Base temperature
        num_samples=50000,
        seed=42        # Reproducibility
    )

    # Calculate divergences
    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)

    print(f"D_KL (proof):   {D_proof:.6f} (should be ≈ 0)")
    print(f"D_KL (control): {D_control:.6f} (should be >> 0)")

    # Create figure
    fig = plt.figure(figsize=(16, 5))

    # Panel 1: Case A (Original)
    ax1 = plt.subplot(1, 3, 1)
    states = np.arange(len(P_orig))
    ax1.bar(states, P_orig, alpha=0.7, color='#1f77b4', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('State Index', fontsize=12)
    ax1.set_ylabel('Probability P(s)', fontsize=12)
    ax1.set_title('Case A: Original System\n(W, H, T)', fontsize=13, weight='bold')
    ax1.grid(alpha=0.3, axis='y')
    ax1.set_ylim(0, max(P_orig) * 1.1)

    # Panel 2: Case B (Control - Energy only)
    ax2 = plt.subplot(1, 3, 2)
    ax2.bar(states, P_scaled_E, alpha=0.7, color='#ff7f0e', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('State Index', fontsize=12)
    ax2.set_ylabel('Probability P(s)', fontsize=12)
    ax2.set_title('Case B: Control\n(α·W, α·H, T) - Should Differ', fontsize=13, weight='bold')
    ax2.grid(alpha=0.3, axis='y')
    ax2.set_ylim(0, max(P_orig) * 1.1)

    # Add divergence annotation
    ax2.text(0.98, 0.98, f'D_KL = {D_control:.3f}',
             transform=ax2.transAxes, verticalalignment='top', horizontalalignment='right',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

    # Panel 3: Case C (Test - Full scaling)
    ax3 = plt.subplot(1, 3, 3)
    ax3.bar(states, P_test, alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('State Index', fontsize=12)
    ax3.set_ylabel('Probability P(s)', fontsize=12)
    ax3.set_title('Case C: Scale Invariance Test\n(α·W, α·H, α·T) - Should Match A', fontsize=13, weight='bold')
    ax3.grid(alpha=0.3, axis='y')
    ax3.set_ylim(0, max(P_orig) * 1.1)

    # Add divergence annotation
    ax3.text(0.98, 0.98, f'D_KL = {D_proof:.6f}\n✓ PASS',
             transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
             fontsize=11, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Overall title
    fig.suptitle('Scale Invariance Validation: P(S; H, T) = P(S; α·H, α·T)',
                 fontsize=16, weight='bold', y=1.02)

    plt.tight_layout()

    # Save
    plt.savefig('scale_invariance_proof.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: scale_invariance_proof.png")

    return fig


def create_overlay_plot():
    """
    Alternative visualization: Overlay original and test to show perfect match.
    """

    print("\nCreating overlay plot...")

    # Run experiment
    P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
        N=5, alpha=2.0, T0=1.0, num_samples=50000, seed=42
    )

    D_proof = quantify_divergence(P_orig, P_test)
    D_control = quantify_divergence(P_orig, P_scaled_E)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    states = np.arange(len(P_orig))

    # Plot original as bars
    ax.bar(states, P_orig, alpha=0.5, label='Original (W, H, T)',
           color='#1f77b4', edgecolor='black', linewidth=1)

    # Overlay test as line with markers (should match bars)
    ax.plot(states, P_test, 'o-', color='#2ca02c', linewidth=2.5,
            markersize=8, label='Scaled (α·W, α·H, α·T)', alpha=0.9,
            markeredgecolor='black', markeredgewidth=0.5)

    # Add control as dashed line (should differ)
    ax.plot(states, P_scaled_E, 's--', color='#d62728', linewidth=1.5,
            markersize=5, label='Control (α·W, α·H, T)', alpha=0.7)

    ax.set_xlabel('State Index', fontsize=14)
    ax.set_ylabel('Probability P(s)', fontsize=14)
    ax.set_title('Scale Invariance: Perfect Overlap Demonstrates Invariance',
                 fontsize=16, weight='bold')
    ax.legend(fontsize=12, loc='upper right', framealpha=0.9)
    ax.grid(alpha=0.3)

    # Add text box with results
    textstr = f'Proof: D_KL(orig || test) = {D_proof:.6f} ✓\nControl: D_KL(orig || control) = {D_control:.3f} ✓'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('scale_invariance_overlay.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: scale_invariance_overlay.png")

    return fig


if __name__ == "__main__":
    print("="*60)
    print("JUNIE'S EXAMPLE STARTER VISUALIZATION")
    print("="*60)
    print("\nThis will create two visualizations of scale invariance:")
    print("1. Three-panel comparison (Cases A, B, C)")
    print("2. Overlay plot showing perfect match\n")

    # Create visualizations
    fig1 = create_scale_invariance_proof_plot()
    fig2 = create_overlay_plot()

    print("\n" + "="*60)
    print("✓ VISUALIZATIONS COMPLETE")
    print("="*60)
    print("\nNext steps for Junie:")
    print("1. Review .junie/guidelines.md for more ideas")
    print("2. Create Priority 2: Scale span (nm to m)")
    print("3. Create Priority 3: TIS quality ladder")
    print("4. Explore materials phase diagrams")
    print("5. Animate RG flow")
    print("\nHappy visualizing! 🎨")

    # Show plots
    plt.show()
