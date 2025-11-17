# Junie's Workspace 🎨

**Welcome, Junie!** This is your dedicated space for creating visualizations that communicate the Scale Invariance Framework.

---

## 🎯 Your Mission

Transform rigorous mathematics into compelling visual narratives that show:
1. **Scale invariance** works (P(S;H,T) = P(S;αH,αT))
2. **Same law** governs atoms and alloys (9+ orders of magnitude)
3. **TIS benchmark** validates TPUs objectively

---

## 📂 What's Here

### `guidelines.md` ⭐ START HERE
Comprehensive guide with:
- 9 priority visualizations to create
- Code recipes and examples
- Technical specifications
- Communication goals for different audiences

### `example_starter.py`
Working example showing how to:
- Access data from the framework
- Create publication-quality plots
- Use appropriate styling

Run it:
```bash
python3 .junie/example_starter.py
```

---

## 🚀 Quick Start (5 Minutes)

### 1. Run the Example

```bash
python3 .junie/example_starter.py
```

This creates two visualizations:
- `scale_invariance_proof.png` - Three-panel comparison
- `scale_invariance_overlay.png` - Perfect overlap demonstration

### 2. Review the Output

Open the PNG files. Notice:
- Case A and Case C **match visually** (D_KL ≈ 0)
- Case B **differs clearly** (D_KL > 0)
- Clean styling, clear labels, annotations

### 3. Read Guidelines

Open `guidelines.md` and review:
- **Priority 1-4**: Your first tasks
- **Plot recipes**: Copy-paste code templates
- **Principles**: What makes a good visualization

### 4. Start Creating!

Pick a priority visualization and start coding. The framework is ready to provide data!

---

## 🎨 Priority Queue

**Your top 4 tasks** (in order):

### Priority 1: Scale Invariance Proof ✅ (Example done!)
Three-panel or overlay plot showing P(original) = P(scaled)

### Priority 2: Scale Span Visualization
Show 9 orders of magnitude from nanometers to meters
- Left: Semiconductor atoms (Si/Ge)
- Right: Sword metal (Martensite/Pearlite)
- Center: "Same Law" message

### Priority 3: TIS Quality Ladder
Vertical bar chart showing TPU classification
- REFERENCE (TIS > 1000)
- EXCELLENT (100-1000)
- GOOD (31-100)
- etc.
Plot actual TPU results as dots

### Priority 4: RG Fixed Point
Show "flow" away from fixed point
- Perfect TPU at origin (β ≈ 0)
- Noisy TPU flows away (β > 0)
- Concentric circles = distance from perfection

---

## 📊 Data Access Cheat Sheet

### Systems 1-3: Equilibrium

```python
from scale_invariance import run_scale_invariance_test

P_orig, P_scaled_E, P_test, params = run_scale_invariance_test(
    N=5, alpha=2.0, T0=1.0, num_samples=50000, seed=42
)
# P_orig: Original distribution
# P_test: Should match P_orig (scale invariance)
# P_scaled_E: Should differ (control)
```

### System 8: Materials

```python
from materials_invariance import (
    create_sword_system,
    create_semiconductor_system,
    run_materials_invariance_test
)

# Sword forging (macroscopic)
sword = create_sword_system()
results_sword = run_materials_invariance_test(sword, T0=1000, alpha=2.0)

# Semiconductor (nanoscale)
semi = create_semiconductor_system()
results_semi = run_materials_invariance_test(semi, T0=800, alpha=2.0)

# Both should have D_KL < 10^-8
```

### System 9: TPU Benchmark

```python
from tpu_benchmark import benchmark_suite
import numpy as np

N = 5
W = np.random.randn(N, N)
W = (W + W.T) / 2
H = np.random.randn(N)

results = benchmark_suite(W, H, T0=1.0, alpha=2.0, num_samples=20000)

# results is a dict: {'TPU name': BenchmarkResult, ...}
# Access: results['Reference (Exact)'].tis
#         results['Reference (Exact)'].grade
#         results['Reference (Exact)'].D_proof
```

---

## 🎨 Styling Guide (Quick Reference)

### Colors

```python
# Use these for consistency
ORIGINAL = '#1f77b4'    # Blue
SCALED_TEST = '#2ca02c' # Green (should match original)
CONTROL = '#ff7f0e'     # Orange (should differ)
PASS = '#2ca02c'        # Green
FAIL = '#d62728'        # Red
```

### Figure Sizes

```python
# Single plot
fig, ax = plt.subplots(figsize=(10, 6))

# Multi-panel
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Publication
fig, ax = plt.subplots(figsize=(6, 4))
plt.savefig('output.png', dpi=300)
```

### Essential Elements

Every plot needs:
```python
plt.xlabel('Clear description with units', fontsize=12)
plt.ylabel('Clear description with units', fontsize=12)
plt.title('What This Shows', fontsize=14, weight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
```

---

## 📚 Framework Context

### The Nine Systems

1. **Systems 1-3**: Equilibrium proof (MCMC)
2. **System 4**: Dynamics (ODE)
3. **System 5**: Stability (exact enumeration)
4. **System 6**: Stochastic (Markov chains)
5. **System 7**: Noise floor (precision)
6. **System 8**: Materials (Gibbs energy)
7. **System 9**: TPU benchmark (TIS)

Plus:
- **Phase 3**: Hardware validation (THRML/JAX)

### Key Results to Visualize

- **Equilibrium**: D_KL ≈ 0.007 (validated ✓)
- **Stability**: Δ = 10^-10 (exact ✓)
- **Hardware**: D_KL = 0.003 (exceeded target ✓)
- **Sword**: D_KL < 10^-8 (exact ✓)
- **Semiconductor**: D_KL < 10^-8 (exact ✓)

---

## 💡 Visualization Philosophy

**Clarity > Complexity**
- Simple plots that make the point
- Not cluttered

**Show, Don't Tell**
- Visual proof > textual explanation
- Overlay distributions to show they match

**Guide the Eye**
- Annotations pointing to key results
- Color coding (green = pass, red = fail)

**Tell a Story**
- Each plot answers: "What does this prove?"
- Build narrative: theory → application → tool

---

## 🎯 Success Criteria

Your visualizations are great if:

1. ✓ A **non-expert** understands the main idea
2. ✓ An **expert** sees the rigor
3. ✓ A **decision-maker** grasps the value
4. ✓ A **journal** would publish them

**Goal**: Make people say "Wow, that's beautiful AND rigorous!"

---

## 📞 Resources

**Documentation**:
- `../FRAMEWORK_COMPLETE.md` - Full context
- `../SYSTEM8_SUMMARY.md` - Materials details
- `../SYSTEM9_SUMMARY.md` - TPU benchmark details

**Code**:
- `../scale_invariance.py` - Run Systems 1-3
- `../materials_invariance.py` - Run System 8
- `../tpu_benchmark.py` - Run System 9

**Tests** (show data structures):
- `../test_scale_invariance.py`
- `../test_materials_system.py`
- `../test_tpu_benchmark.py`

---

## 🚦 Getting Help

**Stuck on data access?**
→ Look at test files (test_*.py) to see how they call functions

**Not sure what to plot?**
→ Re-read guidelines.md Priority 1-4

**Styling questions?**
→ Check example_starter.py for working code

**Concept unclear?**
→ Read FRAMEWORK_COMPLETE.md for full context

---

## 📈 Progress Tracking

Create visualizations and save them in organized directories:

```
viz/
├── priority1_scale_invariance/
│   ├── three_panel.png
│   └── overlay.png
├── priority2_scale_span/
│   └── nm_to_m.png
├── priority3_tis_benchmark/
│   ├── quality_ladder.png
│   └── leaderboard.png
├── priority4_rg_fixed_point/
│   └── flow_diagram.png
└── publication/
    ├── fig1.png
    ├── fig2.png
    └── fig3.png
```

---

## 🎊 Final Thoughts

You're visualizing **9 complete systems** that prove a fundamental thermodynamic law, apply it from atoms to alloys, and create an industry benchmark. This is publication-quality research that needs publication-quality figures.

**Your job**: Make the invisible visible. Show that the same equation governs sword forging and semiconductor fabrication. Prove it with pictures.

**The audience is waiting** to see what scale invariance looks like! 🎨

---

**Ready?** Run `python3 .junie/example_starter.py` and start creating! ⚔️→🔬→💻
