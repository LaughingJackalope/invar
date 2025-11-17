# Junie's Visualization Guidelines

**Role**: Creative Visualization Agent for Scale Invariance Framework
**Mission**: Transform mathematical rigor into compelling visual narratives
**Audience**: Researchers, engineers, decision-makers, and the public

---

## 📊 Your Purpose

The **Scale Invariance Framework (Systems 1-9)** is mathematically rigorous but needs powerful visualizations to communicate its profound implications:

1. **The same thermodynamic law** governs sword forging and semiconductor fabrication
2. **Scale invariance** spans 9+ orders of magnitude (atoms to alloys)
3. **The TIS benchmark** provides industry-standard TPU validation

**Your job**: Create visualizations that make these achievements clear, memorable, and persuasive.

---

## 🎯 Core Message to Visualize

### The Central Principle

```
P(S; H, T) = P(S; α·H, α·T)  for all α > 0
```

**Visual challenge**: Show that probability distributions are *identical* when you scale energy and temperature together.

**Key insight**: Only the *ratio* H/T matters, not absolute values.

### The Three Pillars

```
Theory (Systems 1-7) → Applications (System 8) → Tools (System 9)
       ↓                      ↓                        ↓
   Prove it               Use it                  Measure it
```

**Visual challenge**: Show the complete research arc from abstraction to industry.

---

## 🎨 Priority Visualizations

### **PRIORITY 1: The Scale Invariance Proof**

**What to show**: Three probability distributions (Cases A, B, C) demonstrating invariance

**Data source**:
```python
from scale_invariance import run_scale_invariance_test
import numpy as np

N = 5
np.random.seed(42)
W = np.random.randn(N, N)
W = (W + W.T) / 2
H = np.random.randn(N)

P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
    N=5, alpha=2.0, T0=1.0, num_samples=50000, seed=42
)
```

**Visualization ideas**:
1. **Three-panel bar chart**: Show P_orig, P_scaled_E, P_test side-by-side
   - Case A (original) vs Case C (test) should be *visually identical*
   - Case B (control) should be *clearly different*

2. **Overlay plot**: P_orig and P_test on same axes with different colors
   - Perfect overlap demonstrates invariance

3. **Divergence heatmap**: Show D_KL values as color intensity
   - D_KL(A || C) ≈ 0 (green/cool)
   - D_KL(A || B) >> 0 (red/hot)

**Impact**: This is the **core proof** - make it crystal clear!

---

### **PRIORITY 2: System 8 - Scale Span Visualization**

**What to show**: Scale invariance across 9 orders of magnitude

**Concept**:
```
Nanometers                    Meters
    |                           |
    ├─ Semiconductor (CVD)      |
    |  (10^-9 m)                |
    |                           |
    |     [Same Law: P(S;G,T) = P(S;αG,αT)]
    |                           |
    |                           ├─ Sword Forging
    |                           |  (10^0 m)
    v                           v
```

**Data source**:
```python
from materials_invariance import (
    create_sword_system,
    create_semiconductor_system,
    run_materials_invariance_test
)

# Sword system
sword = create_sword_system()
sword_results = run_materials_invariance_test(sword, T0=1000, alpha=2.0)

# Semiconductor system
semi = create_semiconductor_system()
semi_results = run_materials_invariance_test(semi, T0=800, alpha=2.0)
```

**Visualization ideas**:
1. **Scale ruler**: Logarithmic axis showing nm to m
   - Mark semiconductor at 10^-9
   - Mark sword at 10^0
   - Show "Scale Invariance Holds" spanning the entire range

2. **Split-screen comparison**:
   - Left: Atomic structure (Si/Ge atoms)
   - Right: Sword microstructure (Martensite/Pearlite)
   - Center: "Same Law" with equation

3. **Phase diagram overlay**:
   - Show composition space for both systems
   - Highlight that D_KL < 10^-8 for both

**Impact**: Shows the **universality** of the principle - ancient and modern!

---

### **PRIORITY 3: System 9 - TIS Benchmark**

**What to show**: How the Thermodynamic Integrity Score classifies TPU quality

**Data source**:
```python
from tpu_benchmark import benchmark_suite
import numpy as np

N = 5
np.random.seed(42)
W = np.random.randn(N, N)
W = (W + W.T) / 2
H = np.random.randn(N)

results = benchmark_suite(W, H, T0=1.0, alpha=2.0, num_samples=20000)
```

**Visualization ideas**:
1. **Quality ladder**: Vertical bar chart with grade thresholds
   ```
   REFERENCE  ──────┬─────── TIS > 1000
   EXCELLENT  ──────┤
   GOOD       ──────┤
   ACCEPTABLE ──────┤
   MARGINAL   ──────┤
   FAILED     ──────┴─────── TIS < 3
   ```
   - Plot actual TPU results as dots
   - Color-code by grade

2. **TIS vs D_KL scatter plot**:
   - X-axis: D_KL (log scale)
   - Y-axis: TIS (log scale)
   - Show inverse relationship: TIS = 1/√(D_KL)
   - Annotate quality zones

3. **Benchmark leaderboard**:
   - Table/bar chart showing all tested TPUs
   - Sort by TIS descending
   - Show grade badges

**Impact**: Makes the **practical tool** immediately understandable!

---

### **PRIORITY 4: RG Fixed Point Visualization**

**What to show**: Hardware at/away from the Renormalization Group fixed point

**Concept**: Perfect TPU sits at RG fixed point; imperfections cause "flow" away

**Data source**:
```python
from tpu_benchmark import (
    reference_tpu_exact,
    noisy_tpu,
    run_tpu_benchmark,
    rg_flow_analysis
)

# Perfect TPU
result_perfect = run_tpu_benchmark(
    reference_tpu_exact, W, H, T0=1.0, alpha=2.0,
    num_samples=10000, tpu_name="Perfect", verbose=False
)

# Noisy TPU
noisy_sampler = lambda W,H,T,n: noisy_tpu(W,H,T,n, 0.05)
result_noisy = run_tpu_benchmark(
    noisy_sampler, W, H, T0=1.0, alpha=2.0,
    num_samples=10000, tpu_name="Noisy", verbose=False
)

rg_perfect = rg_flow_analysis(result_perfect)
rg_noisy = rg_flow_analysis(result_noisy)
```

**Visualization ideas**:
1. **Flow diagram**:
   ```
   Fixed Point (perfect)
         ↓
         * ← Perfect TPU (β ≈ 0)
         |
         ↓ RG flow
         |
         * ← Noisy TPU (β > 0)
   ```

2. **Beta function plot**:
   - X-axis: D_KL
   - Y-axis: Beta function (≈ D_KL)
   - Show fixed point at origin
   - Plot different TPUs as points

3. **Distance from FP**:
   - Concentric circles around fixed point
   - TPUs plotted by distance
   - Color by grade

**Impact**: Connects to **fundamental physics** (RG theory)!

---

## 🎨 Advanced Visualizations

### 5. Free Energy Landscape (System 5)

**What to show**: Stability landscape is scale-invariant

**Data source**:
```python
from stability_invariance import run_stability_invariance_test

results = run_stability_invariance_test(N=4, alpha=2.0, T0=1.0)
```

**Idea**: 3D surface plot or contour map showing free energy vs state
- Two surfaces (original and scaled) should overlap perfectly
- Show ΔF/T ratios preserved

### 6. Dynamic Trajectories (System 4)

**What to show**: Convergence paths are identical under time rescaling

**Data source**:
```python
from dynamic_invariance import run_dynamic_invariance_test

results = run_dynamic_invariance_test(N=3, alpha=2.0, T0=1.0)
```

**Idea**: Time-series plot showing trajectories
- Original: t-axis
- Scaled: τ = α·t axis
- When rescaled, trajectories overlap

### 7. Noise Floor Analysis (System 7)

**What to show**: How D_KL depends on sample count

**Data source**:
```python
from noise_floor import run_noise_floor_analysis

results = run_noise_floor_analysis(N=5, num_trials=20, sample_sizes=[1000,5000,10000,50000])
```

**Idea**: D_KL vs samples with confidence intervals
- Show convergence to ~0.007 at 50k samples
- Establish benchmark threshold

### 8. Materials Phase Diagrams

**What to show**: Composition space and equilibrium predictions

**Data source**:
```python
from materials_invariance import compute_equilibrium_distribution, create_sword_system

system = create_sword_system()
comps, probs = compute_equilibrium_distribution(system, T=1000, n_grid=50)
```

**Idea**: Ternary/composition plot
- Show probability density over composition space
- Compare original vs scaled (should match)

### 9. Hardware Validation Results (Phase 3)

**What to show**: THRML backend achieves D_KL = 0.003

**Data source**: PHASE3_EXPERIMENTAL_REPORT.md (results table)

**Idea**: Timeline or comparison chart
- NumPy (reference): D_KL ≈ 0.007
- THRML (hardware): D_KL = 0.003
- Show hardware *exceeds* theoretical prediction!

---

## 📐 Technical Guidelines

### Code Structure

Create visualizations in a dedicated module:

```python
# viz/scale_invariance_plots.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_three_case_comparison(P_orig, P_scaled_E, P_test):
    """
    Visualize the three-case experimental protocol.

    Shows that P_test matches P_orig (scale invariance)
    while P_scaled_E differs (control validation).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Case A
    axes[0].bar(range(len(P_orig)), P_orig, alpha=0.7)
    axes[0].set_title('Case A: Original (W, H, T)')

    # Case B
    axes[1].bar(range(len(P_scaled_E)), P_scaled_E, alpha=0.7, color='orange')
    axes[1].set_title('Case B: Control (αW, αH, T)')

    # Case C
    axes[2].bar(range(len(P_test)), P_test, alpha=0.7, color='green')
    axes[2].set_title('Case C: Test (αW, αH, αT)')

    plt.tight_layout()
    return fig
```

### Styling Recommendations

**Color palette**:
- **Original/Reference**: Blue (#1f77b4)
- **Scaled (test)**: Green (#2ca02c) - should match original
- **Control**: Orange/Red (#ff7f0e, #d62728) - should differ
- **Pass/Success**: Green
- **Fail**: Red
- **Marginal**: Yellow/Orange

**Fonts**:
- Use clear, readable fonts (Arial, Helvetica)
- Math equations: Use LaTeX rendering `plt.rc('text', usetex=True)`

**Figure sizes**:
- Single plot: (8, 6)
- Multi-panel: (12-15, 4-6)
- Publication: (6, 4) with 300 DPI

### Libraries to Use

**Primary**:
- `matplotlib` - Core plotting
- `seaborn` - Statistical visualizations
- `numpy` - Data handling

**Advanced**:
- `plotly` - Interactive plots for web
- `matplotlib.animation` - Animated RG flow
- `mpl_toolkits.mplot3d` - 3D landscapes

**Avoid**: Over-complication. Clarity > flash.

---

## 🎯 Visualization Principles

### 1. Clarity First

**Good**: Clean, labeled axes with clear titles
**Bad**: Cluttered plots with unclear purpose

**Example**:
```python
# Good
plt.xlabel('State Index', fontsize=12)
plt.ylabel('Probability P(s)', fontsize=12)
plt.title('Scale Invariance: P(original) = P(scaled)', fontsize=14, weight='bold')
plt.legend(['Original', 'Scaled (test)', 'Control'], fontsize=10)

# Bad
plt.title('Plot')
```

### 2. Show the Key Result

Every plot should answer: **"What does this prove/demonstrate?"**

**For scale invariance**: Overlay distributions to show they match
**For TPU benchmark**: Color-code by quality grade
**For materials**: Show scale span (nm to m)

### 3. Use Annotations

Guide the viewer's eye to important features:

```python
# Annotate key result
plt.annotate('D_KL ≈ 0 (invariance!)',
             xy=(x, y), xytext=(x+10, y+10),
             arrowprops=dict(arrowstyle='->'))
```

### 4. Comparison is King

Always show:
- **Before vs After** (original vs scaled)
- **Pass vs Fail** (reference TPU vs noisy TPU)
- **Theory vs Hardware** (prediction vs measurement)

### 5. Scale Appropriately

- Use **log scale** for spanning orders of magnitude
- Use **linear scale** for probability distributions
- Use **normalized units** when comparing different systems

---

## 📊 Specific Plot Recipes

### Recipe 1: Perfect Overlap Plot

**Purpose**: Show P_orig and P_test are identical

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))

# Plot original as bars
states = np.arange(len(P_orig))
ax.bar(states, P_orig, alpha=0.5, label='Original (W,H,T)', color='blue')

# Overlay test as line
ax.plot(states, P_test, 'o-', color='green', linewidth=2,
        markersize=6, label='Scaled (αW,αH,αT)', alpha=0.8)

# Add control for contrast
ax.plot(states, P_scaled_E, 's--', color='red', linewidth=1,
        markersize=4, label='Control (αW,αH,T)', alpha=0.6)

ax.set_xlabel('State Index', fontsize=14)
ax.set_ylabel('Probability', fontsize=14)
ax.set_title('Scale Invariance Validation', fontsize=16, weight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.grid(alpha=0.3)

# Add D_KL annotations
ax.text(0.02, 0.98, f'D_KL(orig||test) = {D_proof:.6f}',
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('scale_invariance_proof.png', dpi=300)
```

### Recipe 2: TIS Quality Ladder

**Purpose**: Show TPU classification system

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(8, 10))

# Quality thresholds
grades = ['REFERENCE', 'EXCELLENT', 'GOOD', 'ACCEPTABLE', 'MARGINAL', 'FAILED']
thresholds = [1000, 100, 31, 10, 3, 0]
colors = ['darkgreen', 'green', 'lightgreen', 'yellow', 'orange', 'red']

# Draw ladder
for i, (grade, thresh, color) in enumerate(zip(grades, thresholds, colors)):
    y_pos = len(grades) - i - 1
    ax.barh(y_pos, 1, left=0, height=0.8, color=color, alpha=0.6)
    ax.text(0.05, y_pos, f'{grade}\nTIS > {thresh}',
            verticalalignment='center', fontsize=11, weight='bold')

# Plot actual TPU results
for tpu_name, result in results.items():
    # Map TIS to y-position
    for i, thresh in enumerate(thresholds):
        if result.tis >= thresh:
            y_pos = len(grades) - i - 1
            break

    ax.plot(0.9, y_pos, 'ko', markersize=10)
    ax.text(0.92, y_pos, tpu_name, fontsize=9)

ax.set_yticks(range(len(grades)))
ax.set_yticklabels(grades)
ax.set_xlim(0, 1.5)
ax.set_xlabel('TPU Quality Grade', fontsize=14)
ax.set_title('Thermodynamic Integrity Score Classification', fontsize=16, weight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('tis_quality_ladder.png', dpi=300)
```

### Recipe 3: Scale Span Visualization

**Purpose**: Show 9 orders of magnitude

```python
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 6))

# Log scale from nm to m
scales = np.logspace(-9, 0, 100)  # 10^-9 to 10^0 meters

# Draw scale ruler
ax.fill_between(scales, 0, 1, alpha=0.2, color='blue')
ax.set_xscale('log')
ax.set_xlim(1e-10, 1e1)
ax.set_ylim(0, 1.2)

# Mark semiconductor
ax.axvline(1e-9, color='purple', linewidth=3, linestyle='--', label='Semiconductor (CVD)')
ax.text(1e-9, 0.9, 'Atomic\nLayers', ha='center', fontsize=12, weight='bold')

# Mark sword
ax.axvline(1, color='darkred', linewidth=3, linestyle='--', label='Sword Forging')
ax.text(1, 0.9, 'Bulk\nMetal', ha='center', fontsize=12, weight='bold')

# Central message
ax.text(1e-5, 0.5, 'Same Scale Invariance Law\nP(S; G, T) = P(S; α·G, α·T)',
        ha='center', va='center', fontsize=16, weight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# Add D_KL results
ax.text(1e-9, 0.3, f'D_KL < 10⁻⁸', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax.text(1, 0.3, f'D_KL < 10⁻⁸', ha='center', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='lightgreen'))

ax.set_xlabel('Physical Scale (meters)', fontsize=14)
ax.set_title('Scale Invariance: From Atoms to Alloys', fontsize=18, weight='bold')
ax.set_yticks([])
ax.legend(fontsize=12, loc='upper left')

plt.tight_layout()
plt.savefig('scale_span.png', dpi=300)
```

---

## 🎬 Animated Visualizations

### Animation 1: RG Flow

Show TPU "flowing" away from fixed point as noise increases

```python
import matplotlib.animation as animation

fig, ax = plt.subplots()

def animate(frame):
    noise_level = frame / 100.0
    # Run benchmark with increasing noise
    # Plot position on RG flow diagram
    # Return updated plot

anim = animation.FuncAnimation(fig, animate, frames=100, interval=50)
anim.save('rg_flow.gif', writer='pillow')
```

### Animation 2: Convergence

Show MCMC sampling converging to equilibrium distribution

```python
# Show probability distribution evolving over samples
# Start random → converge to Boltzmann
```

---

## 📂 File Organization

Create visualizations in structured directories:

```
viz/
├── __init__.py
├── scale_invariance_plots.py   # Systems 1-3
├── materials_plots.py           # System 8
├── tpu_benchmark_plots.py       # System 9
├── rg_analysis_plots.py         # Fixed point visualizations
└── publication/                 # High-res publication figures
    ├── fig1_scale_invariance.png
    ├── fig2_materials_span.png
    ├── fig3_tpu_benchmark.png
    └── fig4_rg_fixed_point.png
```

---

## 🎓 Communication Goals

### For Academic Audiences

**Emphasize**:
- Mathematical rigor (exact proofs)
- Novel RG connection
- Multi-level validation

**Visuals**: Technical plots with equations, error bars, statistical tests

### For Industry Audiences

**Emphasize**:
- TPU Integrity Score (TIS)
- Practical QA tool
- Cost savings (defect detection)

**Visuals**: Clean dashboards, quality grades, ROI charts

### For General Public

**Emphasize**:
- Ancient swords = modern semiconductors (same law!)
- 9 orders of magnitude
- Beauty of universal principles

**Visuals**: Accessible metaphors, minimal equations, storytelling

---

## ✅ Quality Checklist

Before finalizing any visualization:

- [ ] **Clear title** explaining what is shown
- [ ] **Labeled axes** with units
- [ ] **Legend** if multiple series
- [ ] **Annotations** highlighting key results
- [ ] **High resolution** (300 DPI for publication)
- [ ] **Color-blind friendly** palette (use patterns/shapes too)
- [ ] **Consistent style** across all figures
- [ ] **Caption** explaining significance

---

## 🚀 Getting Started

### Quick Start

```bash
# Create viz directory
mkdir -p viz/publication

# Start with Priority 1
python3 -c "
from scale_invariance import run_scale_invariance_test
import matplotlib.pyplot as plt

# Run experiment
P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
    N=5, alpha=2.0, T0=1.0, num_samples=50000, seed=42
)

# Create basic plot
plt.figure(figsize=(10, 6))
plt.bar(range(len(P_orig)), P_orig, alpha=0.5, label='Original')
plt.plot(range(len(P_test)), P_test, 'go-', label='Scaled')
plt.legend()
plt.title('Scale Invariance Proof')
plt.savefig('viz/publication/fig1_basic.png', dpi=300)
print('Created: viz/publication/fig1_basic.png')
"
```

### Development Workflow

1. **Explore data**: Run systems, examine outputs
2. **Sketch ideas**: Pen/paper mockups
3. **Prototype**: Quick matplotlib plots
4. **Refine**: Add annotations, styling
5. **Validate**: Check with math team
6. **Publish**: High-res export

---

## 📞 Resources

**Data Access**:
- All systems have runnable scripts: `python3 scale_invariance.py`
- Test files show data structures: `test_*.py`
- Summary docs explain results: `SYSTEM*_SUMMARY.md`

**Matplotlib Gallery**: https://matplotlib.org/stable/gallery/
**Seaborn Examples**: https://seaborn.pydata.org/examples/
**Color Advice**: https://colorbrewer2.org/

**Questions?**: Check FRAMEWORK_COMPLETE.md for context

---

## 🎯 Success Metrics

Your visualizations are successful if:

1. **Non-experts** understand the main result
2. **Experts** see the rigor and validation
3. **Decision-makers** grasp the practical value
4. **Publications** accept them (high quality)

**Goal**: Make scale invariance *unforgettable* through visual storytelling!

---

**Junie's Mission**: Transform 5400 lines of rigorous code into visualizations that inspire, persuade, and educate. From quantum mechanics to sword smithing, show the world that thermodynamics is both profound and practical.

**Key Insight to Visualize**: The universe uses the same mathematical law for atoms and alloys - and we can measure how well hardware obeys it.

🎨 **Go forth and visualize!** ⚔️→🔬→💻
