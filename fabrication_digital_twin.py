"""
Semiconductor Fabrication Digital Twin

Models key semiconductor fabrication processes as a sequence of Energy-Based Models (EBMs)
using the DTM framework. Each process step (lithography, etching, deposition) is represented
as an EBM that transforms the wafer state.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sampler_interface import BoltzmannSampler, SamplerFactory

@dataclass
class ProcessStep:
    """Represents a single fabrication process step as an EBM."""
    name: str
    temperature: float  # Effective temperature (process variations)
    coupling_strength: float  # Controls interaction between features
    bias_strength: float  # Controls material properties
    duration: float  # Process time (arbitrary units)

class FabricationDigitalTwin:
    """Digital twin for semiconductor fabrication process.

    This implementation maps each fabrication step to a local EBM and uses a
    Boltzmann sampler backend to model stochasticity (process variation).

    Notes
    -----
    The sampler interface returns a probability distribution over all 2^N
    states (N = size^2). This grows exponentially; for practical runs, use
    small sizes (e.g., size ≤ 3 → N ≤ 9 → 512 states). For larger systems,
    connect to approximate samplers that return samples instead of full P.
    """

    def __init__(self, size: int = 3, backend: str = "auto"):
        """
        Initialize the digital twin with a wafer of given size.

        Args:
            size: Size of the wafer grid (size x size)
            backend: Sampler backend ('numpy', 'thrml', or 'auto')
        """
        self.size = size
        # Use backend-agnostic factory with graceful fallback
        self.sampler: BoltzmannSampler = SamplerFactory.create_sampler(backend)
        self.process_steps: List[ProcessStep] = []
        self.wafer_state = np.ones((size, size), dtype=np.int8)  # 1 = material, -1 = etched
        # Optional target mask for pattern fidelity (1 for desired material, -1 for etched)
        self.target_mask: Optional[np.ndarray] = None
        # Per-run stats cache
        self._last_run_stats: List[Dict[str, Any]] = []

    def add_process_step(self, step: ProcessStep) -> None:
        """Add a process step to the fabrication sequence."""
        self.process_steps.append(step)

    def set_target_mask(self, mask: np.ndarray) -> None:
        """Set a desired pattern mask to compute fidelity/yield metrics.

        mask should be a (size, size) array with values in {-1, 1}.
        """
        if mask.shape != (self.size, self.size):
            raise ValueError(f"mask must be shape {(self.size, self.size)}, got {mask.shape}")
        unique = np.unique(mask)
        if not set(unique.tolist()).issubset({-1, 1}):
            raise ValueError("mask values must be in {-1, 1}")
        self.target_mask = mask.astype(np.int8)
    
    def _create_ising_model(self, step: ProcessStep) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create Ising model parameters for a given process step.
        
        Returns:
            W: Interaction matrix (N x N)
            H: Bias vector (N,)
        """
        N = self.size * self.size
        W = np.zeros((N, N))
        H = np.zeros(N)
        
        # Convert 2D coordinates to 1D indices
        def idx(i, j):
            return i * self.size + j
        
        # Nearest-neighbor interactions (4-connected grid)
        for i in range(self.size):
            for j in range(self.size):
                current = idx(i, j)
                
                # Bias based on current state and process parameters
                H[current] = step.bias_strength * self.wafer_state[i, j]
                
                # Add interactions with neighbors
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.size and 0 <= nj < self.size:
                        neighbor = idx(ni, nj)
                        W[current, neighbor] = step.coupling_strength
        
        return W, H
    
    def run_simulation(self, num_samples: int = 1000, visualize: bool = True) -> np.ndarray:
        """
        Run the complete fabrication process simulation.
        
        Args:
            num_samples: Number of MCMC samples per step
            visualize: If True, show wafer state after each step
            
        Returns:
            Final wafer state after all process steps
        """
        self._last_run_stats = []
        for step_idx, step in enumerate(self.process_steps):
            print(f"\n--- Step {step_idx + 1}: {step.name} ---")
            print(f"Temperature: {step.temperature:.2f}, Duration: {step.duration}")
            
            # Create Ising model for this step
            W, H = self._create_ising_model(step)
            
            # Sample from the Boltzmann distribution
            P = self.sampler.sample_distribution(
                W, H, T=step.temperature, num_samples=num_samples
            )
            # Entropy (process noise) for this step
            eps = 1e-12
            entropy = float(-(P * np.log(P + eps)).sum())
            
            # Update wafer state based on most probable configuration
            most_probable_state_idx = np.argmax(P)
            new_state = np.array(
                [int(b) for b in f"{most_probable_state_idx:0{self.size*self.size}b}"]
            ).reshape((self.size, self.size))
            new_state = 2 * new_state - 1  # Convert from {0,1} to {-1,1}
            
            # Apply state update (with some inertia)
            self.wafer_state = np.where(
                np.random.random(self.wafer_state.shape) < 0.8,  # 80% update probability
                new_state,
                self.wafer_state
            )
            # Compute expected metrics under P (yield-related)
            metrics = self._compute_expected_metrics(P)

            # Record stats
            self._last_run_stats.append({
                "step": step_idx + 1,
                "name": step.name,
                "temperature": step.temperature,
                "duration": step.duration,
                "entropy": entropy,
                **metrics,
            })

            # Visualize
            if visualize:
                self.visualize(step.name)
        
        return self.wafer_state
    
    def visualize(self, title: str = "") -> None:
        """Visualize the current wafer state."""
        plt.figure(figsize=(8, 8))
        plt.imshow(self.wafer_state, cmap='gray', vmin=-1, vmax=1)
        plt.colorbar(label='Material (1) / Etched (-1)')
        plt.title(title or "Wafer State")
        plt.axis('off')
        plt.show()

    # ------------------- Yield/metrics utilities -------------------
    def _index_to_spin_grid(self, idx: int) -> np.ndarray:
        """Convert a state index to a (size,size) grid with values in {-1,1}."""
        bits = [int(b) for b in f"{idx:0{self.size*self.size}b}"]
        grid01 = np.array(bits, dtype=np.int8).reshape((self.size, self.size))
        return 2 * grid01 - 1

    def _pattern_fidelity(self, grid: np.ndarray) -> float:
        """Fraction of cells matching the target mask (if provided), otherwise 1.0."""
        if self.target_mask is None:
            return 1.0
        return float((grid == self.target_mask).mean())

    def _uniformity(self, grid: np.ndarray) -> float:
        """Simple material uniformity metric: 1 - variance of spins scaled to [0,1]."""
        # Spins in {-1,1}: variance in [0,1]; map to [0,1] where 1 means uniform
        var = float(grid.astype(np.float32).var())
        return 1.0 - min(max(var, 0.0), 1.0)

    def _compute_expected_metrics(self, P: np.ndarray) -> Dict[str, float]:
        """Compute expected yield-related metrics under distribution P."""
        Nstates = P.shape[0]
        exp_fidelity = 0.0
        exp_uniformity = 0.0
        # For small systems, sum over all states
        for s_idx in range(Nstates):
            p = float(P[s_idx])
            if p <= 0.0:
                continue
            grid = self._index_to_spin_grid(s_idx)
            exp_fidelity += p * self._pattern_fidelity(grid)
            exp_uniformity += p * self._uniformity(grid)
        return {
            "expected_fidelity": exp_fidelity,
            "expected_uniformity": exp_uniformity,
        }

    def run_yield_prediction(self, num_samples: int = 2000) -> Dict[str, Any]:
        """
        Execute the process without visualization and return probabilistic yield stats.

        Returns
        -------
        dict with keys:
            - steps: list of per-step dicts (name, temperature, entropy, expected_* metrics)
            - final_state: np.ndarray final wafer grid
            - backend: info about sampler backend
        """
        self.run_simulation(num_samples=num_samples, visualize=False)
        return {
            "steps": self._last_run_stats,
            "final_state": self.wafer_state.copy(),
            "backend": self.sampler.get_backend_info(),
        }

def create_standard_process_flow() -> List[ProcessStep]:
    """Create a standard semiconductor fabrication process flow."""
    return [
        ProcessStep(
            name="Initial Deposition",
            temperature=0.5,
            coupling_strength=1.0,
            bias_strength=1.0,
            duration=1.0
        ),
        ProcessStep(
            name="Lithography Patterning",
            temperature=0.2,
            coupling_strength=2.0,
            bias_strength=-0.5,
            duration=0.5
        ),
        ProcessStep(
            name="Etching",
            temperature=0.3,
            coupling_strength=1.5,
            bias_strength=-1.0,
            duration=0.8
        ),
        ProcessStep(
            name="Final Annealing",
            temperature=0.1,
            coupling_strength=0.5,
            bias_strength=0.2,
            duration=1.2
        )
    ]

if __name__ == "__main__":
    # Create and run the digital twin (keep state space tractable)
    digital_twin = FabricationDigitalTwin(size=3, backend="auto")  # N=9 → 512 states

    # Example: target mask (checkerboard) for fidelity metric
    mask = np.indices((3, 3)).sum(axis=0) % 2
    mask = 2 * mask.astype(np.int8) - 1
    digital_twin.set_target_mask(mask)

    # Add standard process steps
    process_flow = create_standard_process_flow()
    for step in process_flow:
        digital_twin.add_process_step(step)

    # Run yield prediction (no plots)
    results = digital_twin.run_yield_prediction(num_samples=5000)

    # Save final state and print brief report
    np.save("wafer_final_state.npy", results["final_state"])
    print("\nSimulation complete. Final state saved to wafer_final_state.npy")
    print("Backend:", results["backend"]) 
    for s in results["steps"]:
        print(
            f"Step {s['step']}: {s['name']} | T={s['temperature']:.2f} | "
            f"Entropy={s['entropy']:.3f} | Fidelity={s['expected_fidelity']:.3f} | "
            f"Uniformity={s['expected_uniformity']:.3f}"
        )
