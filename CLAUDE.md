# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a computational physics research project proving **scale invariance in Boltzmann-distributed multi-agent systems**. The framework demonstrates that equilibrium probability distributions, stability landscapes, and dynamic trajectories remain invariant under simultaneous scaling of energy parameters (W, H) and temperature (T) by factor α.

**Mathematical claim**: `P(s | W, H, T) = P(s | α·W, α·H, α·T)` for all states s and scaling factors α > 0

**Status**: Theory complete and validated. Phase 3 hardware validation achieved PRIMARY SUCCESS with D_KL = 0.003224 (<<0.007 predicted bound).

## Running the Code

### Basic Commands
```bash
# Run equilibrium proof (Systems 1-3)
python3 scale_invariance.py

# Run stability proof (System 5)
python3 stability_invariance.py

# Run dynamic proof (System 4)
python3 dynamic_invariance.py

# Run materials science proof (System 8)
python3 materials_invariance.py

# Run TPU integrity benchmark (System 9)
python3 tpu_benchmark.py

# Run comprehensive demo
python3 demo.py
```

### Testing
```bash
# Run equilibrium tests (6 tests)
python3 test_scale_invariance.py

# Run stability tests (5 tests)
python3 test_advanced_systems.py

# Run materials science tests (8 tests)
python3 test_materials_system.py

# Run TPU benchmark tests (9 tests)
python3 test_tpu_benchmark.py

# Run backend abstraction tests
python3 test_phase1.py

# Run THRML integration tests (requires JAX)
python3 test_phase2.py

# Run hardware validation (requires JAX + thrml library)
python3 phase3_final_validation.py
```

### Dependencies
- **Core**: `numpy`, `scipy`
- **Hardware acceleration (optional)**: `jax`, `jaxlib`, `thrml`

## Architecture

### Three-Layer Design

**Layer 1 (Mathematical Core)**: Proof systems 1-9
- `scale_invariance.py` - Systems 1-3: Equilibrium distribution invariance
- `dynamic_invariance.py` - System 4: Trajectory invariance (analytical)
- `stability_invariance.py` - System 5: Free energy landscape invariance
- `stochastic_invariance.py` - System 6: Markov chain invariance
- `noise_floor.py` - System 7: Statistical precision analysis
- `materials_invariance.py` - System 8: Materials science applications (Gibbs free energy)
- `tpu_benchmark.py` - System 9: TPU integrity benchmark (hardware QA tool)

**Layer 2 (Interface)**: Backend abstraction
- `sampler_interface.py` - Abstract base class defining sampling contract
- `SamplerFactory` - Factory for creating backend instances

**Layer 3 (Backends)**: Pluggable sampling implementations
- `numpy_sampler.py` - Pure Python/NumPy reference implementation (CPU)
- `thrml_sampler.py` - Hardware-accelerated JAX backend (GPU)

### Proof Hierarchy

1. **Equilibrium** (Systems 1-3): What states the system reaches
   - Method: MCMC Gibbs sampling
   - Metric: KL divergence D_KL(P_orig || P_test)
   - Status: ✓ VALIDATED (D_KL ≈ 0.007)

2. **Dynamics** (System 4): How the system evolves
   - Method: ODE integration with mean-field approximation
   - Metric: Trajectory distance in rescaled time
   - Status: ⚠️ THEORETICAL (nonlinear effects approximate)

3. **Stability** (System 5): Why certain states are preferred
   - Method: Exact enumeration of all 2^N states
   - Metric: Free energy differences ΔF/T
   - Status: ✓ VALIDATED EXACT (Δ = 10^-10)

4. **Stochastic** (System 6): Underlying sampling mechanisms
   - Method: Metropolis-Hastings acceptance ratio analysis
   - Status: ✓ VALIDATED EXACT

5. **Statistical** (System 7): Precision and sample requirements
   - Method: Empirical noise floor determination
   - Status: ✓ VALIDATED (0.007 @ 50k samples)

6. **Hardware** (Phase 3): Real-world validation
   - Method: THRML backend on JAX/GPU
   - Status: ✓ PRIMARY SUCCESS (D_KL = 0.003224)

7. **Materials Science** (System 8): Practical applications
   - Method: Gibbs free energy framework with composition variables
   - Applications: Sword forging (bulk metallurgy), semiconductor deposition (nanoscale)
   - Status: ✓ VALIDATED (both regimes, D_KL < 10^-8)

8. **TPU Benchmark** (System 9): Hardware quality assurance
   - Method: Thermodynamic Integrity Score (TIS) via scale invariance testing
   - Applications: TPU validation, manufacturing QA, fault detection
   - Output: Single quality metric + RG fixed-point analysis
   - Status: ✓ PRODUCTION READY (9/9 tests passing)

### Key Design Patterns

**Energy Function**: E(s) = -s^T W s - H^T s
- W: NxN symmetric interaction matrix
- H: N-dimensional bias vector
- s: State vector in {-1, +1}^N

**State Indexing**: States indexed as integers 0 to 2^N-1
- Binary representation: state i corresponds to binary digits
- Conversion handled internally by each backend

**Sampling Contract** (sampler_interface.py):
```python
def sample_distribution(W, H, T, num_samples) -> P
    # Returns: Probability distribution over all 2^N states
    # Format: NumPy array, length 2^N, normalized to sum=1
```

**Three-Case Experimental Protocol**:
- Case A (Baseline): Parameters (W, H, T₀)
- Case B (Control): Parameters (α·W, α·H, T₀) - should differ
- Case C (Test): Parameters (α·W, α·H, α·T₀) - should match A

**System 8 - Materials Framework** (materials_invariance.py):
Extends to real materials via Gibbs free energy:
- **Generalized Hamiltonian**: G = Σ c_j G_j° + RT Σ c_j ln(c_j) + Σ L_ij c_i c_j
- **State Variables**: Phase fractions or atomic concentrations (c_j), not spins
- **Interaction Matrix**: L_ij represents chemical/strain energy, not magnetic coupling
- **Two Regimes**:
  - Bulk (sword): Phases = {Austenite, Martensite, Pearlite}, macroscopic scale
  - Atomic (semiconductor): Species = {Si, Ge, Vacancy}, nanoscale epitaxy
- **Same Invariance**: P(S; G, T) = P(S; α·G, α·T) holds for both regimes

**System 9 - TPU Benchmark** (tpu_benchmark.py):
Diagnostic tool for validating thermodynamic hardware:
- **Thermodynamic Integrity Score**: TIS = 1/√(D_KL), single quality metric
- **Quality Grades**: REFERENCE (TIS>1000), EXCELLENT (>100), GOOD (>31), ACCEPTABLE (>10), MARGINAL (>3), FAILED (<3)
- **RG Analysis**: Measures distance from Renormalization Group fixed point
- **Applications**: Manufacturing QA, hardware validation, fault detection
- **Reference Implementations**: Exact enumeration, MCMC, noisy variants for testing

## Important Implementation Details

### Gibbs Sampling (numpy_sampler.py)
- Burn-in period: 25% of total samples
- Single-spin flip updates using local fields
- State representation: {-1, +1} spins
- Distribution estimation via frequency counting

### THRML Backend (thrml_sampler.py)
- Uses two-color block Gibbs sampling
- Converts W→edges, H→biases, T→β=1/T
- Requires JAX for hardware acceleration
- State conversion: Binary indexing to match NumPy backend

### Validation Metrics
- **Primary**: D_KL(P_orig || P_test) < 0.007 (tight bound from System 7)
- **Control**: D_KL(P_orig || P_scaled_E) > 0.05 (must show difference)
- **Exact**: Machine precision (~10^-10) for analytical systems

### Recommended Parameters
- Quick test: N=4-5, samples=10k-20k
- Rigorous proof: N=5-6, samples=50k
- Hardware validation: N=6, samples=50k
- Note: State space grows as 2^N, but MCMC makes larger N tractable

## Development Workflow

### Adding New Backends
1. Subclass `BoltzmannSampler` from `sampler_interface.py`
2. Implement `sample_distribution(W, H, T, num_samples)` method
3. Register with `SamplerFactory` in sampler_interface.py
4. Add backend-specific tests in new test file
5. Validate against numpy reference using test_phase1.py pattern

### Modifying Core Systems
- Systems 1-3: Modify `scale_invariance.py`, test with `test_scale_invariance.py`
- System 4: Modify `dynamic_invariance.py`, test with `test_advanced_systems.py`
- System 5: Modify `stability_invariance.py`, test with `test_advanced_systems.py`
- System 8: Modify `materials_invariance.py`, test with `test_materials_system.py`
- System 9: Modify `tpu_benchmark.py`, test with `test_tpu_benchmark.py`
- Always maintain backward compatibility with existing API

### Running Hardware Experiments
1. Install JAX: `pip install jax jaxlib`
2. Install THRML library (if available)
3. Run validation: `python3 phase3_final_validation.py`
4. Compare results with theoretical predictions in PHASE3_EXPERIMENTAL_REPORT.md

## Documentation Structure

Start with README.md for usage, then consult:

- **EXECUTIVE_SUMMARY.md** - High-level overview, deliverables, impact
- **THEORETICAL_FRAMEWORK.md** - Complete mathematical proofs, significance
- **IMPLEMENTATION_SUMMARY.md** - System specifications, deployment details
- **INDEX.md** - Complete navigation map of all files
- **PHASE2_READINESS.md** - Backend abstraction completion summary
- **PHASE3_EXPERIMENTAL_REPORT.md** - Hardware validation results
- **PHASE3_COMPLETION.md** - Final project status

## Theoretical Context

This framework bridges:
- **Statistical mechanics**: Boltzmann distributions, partition functions
- **Multi-agent systems**: Collective behavior, equilibrium selection
- **Thermodynamic computing**: Physical implementation of logical operations
- **Machine learning**: Boltzmann machines, energy-based models
- **Materials science**: Phase transformations, epitaxial growth, metallurgy (System 8)

### Key Insight
Boltzmann distributions exhibit three-fold invariance under simultaneous energy-temperature scaling:
1. Structural (equilibrium behavior)
2. Temporal (dynamic evolution with time rescaling)
3. Energetic (stability landscape)

This means multi-agent systems can be analyzed at any convenient scale without loss of generality, enabling hardware-software abstraction in thermodynamic computing.

**System 8 Extension**: The same invariance principle applies when energy E is generalized to Gibbs free energy G, enabling direct application to real materials processes. The framework successfully models both macroscopic metallurgy (sword forging with phase transformations) and nanoscale semiconductor fabrication (CVD/PVD with atomic precision), demonstrating that scale invariance is a universal thermodynamic property spanning 9+ orders of magnitude in physical scale.

**System 9 Application**: The proven scale invariance becomes a practical diagnostic tool - the **Thermodynamic Integrity Score (TIS)**. By testing whether a TPU (thermodynamic processing unit) preserves P(S; H, T) = P(S; α·H, α·T), we quantify its proximity to the Renormalization Group fixed point. This provides manufacturers with a single, rigorous quality metric for hardware validation, analogous to how clock speed benchmarks validate CPUs. The framework transitions from pure theory (Systems 1-7) through applications (System 8) to practical engineering tools (System 9).

## Known Limitations

- State space grows exponentially (2^N) - practical limit around N=10 for exact enumeration
- MCMC sampling requires sufficient samples for convergence (use System 7 noise floor analysis)
- System 4 dynamics only exact for linear systems (nonlinear activation functions introduce approximation)
- KL divergence sensitive to low-probability states (requires adequate sampling)
- Hardware validation requires JAX-compatible GPU and thrml library
