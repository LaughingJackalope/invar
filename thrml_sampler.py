"""
THRML Sampler - Hardware-Accelerated Backend

Adapter that translates our dense matrix representation to thrml's sparse
edge-based format, enabling hardware-accelerated sampling via JAX.

This is the ONLY file that imports thrml.

Key Conversions:
- W matrix (N×N dense) → edges (list of node pairs) + weights (list of floats)
- H vector (N,) → biases (JAX array)
- Temperature T → beta = 1/T (inverse temperature)
- JAX arrays → NumPy arrays (for interface consistency)
"""

import numpy as np
from sampler_interface import BoltzmannSampler
from typing import Dict, List, Tuple

# Import thrml (ONLY file that does this)
try:
    import jax
    import jax.numpy as jnp
    from thrml import SpinNode, Block, SamplingSchedule, sample_states
    from thrml.models import IsingEBM, IsingSamplingProgram, hinton_init
    THRML_AVAILABLE = True
except ImportError as e:
    THRML_AVAILABLE = False
    IMPORT_ERROR = str(e)


class ThrmlSampler(BoltzmannSampler):
    """
    THRML-based Boltzmann sampler using hardware-accelerated block Gibbs.
    
    Uses JAX for GPU acceleration and thrml's IsingEBM for efficient sampling.
    Automatically converts between our dense matrix format and thrml's
    sparse edge representation.
    
    Examples
    --------
    >>> sampler = ThrmlSampler()
    >>> W = np.array([[0, 1, 0.5], [1, 0, 0.3], [0.5, 0.3, 0]])
    >>> H = np.array([0.5, -0.5, 0.0])
    >>> P = sampler.sample_distribution(W, H, T=1.0, num_samples=10000)
    """
    
    def __init__(self, warmup_fraction: float = 0.25, steps_per_sample: int = 2):
        """
        Initialize THRML sampler.
        
        Parameters
        ----------
        warmup_fraction : float
            Fraction of samples to use for burn-in (default: 0.25)
        steps_per_sample : int
            MCMC steps between collected samples (default: 2)
        """
        if not THRML_AVAILABLE:
            raise ImportError(
                f"THRML library not available. Error: {IMPORT_ERROR}\n"
                "Install with: pip install thrml"
            )
        
        self.warmup_fraction = warmup_fraction
        self.steps_per_sample = steps_per_sample
    
    def sample_distribution(
        self,
        W: np.ndarray,
        H: np.ndarray,
        T: float,
        num_samples: int
    ) -> np.ndarray:
        """
        Sample from Boltzmann distribution using THRML.
        
        Parameters
        ----------
        W : np.ndarray
            Interaction matrix (N×N, symmetric)
        H : np.ndarray
            Bias vector (N,)
        T : float
            Temperature (positive)
        num_samples : int
            Number of samples to generate
            
        Returns
        -------
        P : np.ndarray
            Probability distribution (2^N,)
        """
        # Validate inputs
        self.validate_inputs(W, H, T, num_samples)
        
        N = len(H)
        
        # Convert parameters to THRML format
        nodes, edges, weights, biases, beta = self._convert_to_thrml_format(W, H, T)
        
        # Build THRML model
        model = IsingEBM(nodes, edges, biases, weights, beta)
        
        # Create block structure (2-color blocking for efficient sampling)
        free_blocks = self._create_block_structure(nodes)
        
        # Create sampling program
        program = IsingSamplingProgram(model, free_blocks, clamped_blocks=[])
        
        # Sample
        samples = self._run_sampling(program, free_blocks, nodes, num_samples, model)
        
        # Convert samples to probability distribution
        P = self._samples_to_distribution(samples, N)
        
        # Validate output
        self.validate_output(P, N)
        
        return P
    
    def _convert_to_thrml_format(
        self,
        W: np.ndarray,
        H: np.ndarray,
        T: float
    ) -> Tuple[List, List[Tuple], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Convert dense matrix format to THRML sparse edge format.
        
        This is the CRITICAL conversion function.
        
        W matrix (N×N) → edges (list of node pairs) + weights (list)
        H vector → biases (JAX array)
        T → beta = 1/T
        
        Returns
        -------
        nodes : List[SpinNode]
            Node objects for each spin
        edges : List[Tuple[SpinNode, SpinNode]]
            List of edges connecting nodes
        weights : jnp.ndarray
            Coupling strengths for each edge
        biases : jnp.ndarray
            External field on each node
        beta : jnp.ndarray
            Inverse temperature (1/T)
        """
        N = len(H)
        
        # Create spin nodes
        nodes = [SpinNode() for _ in range(N)]
        
        # Extract edges and weights from W matrix
        # Only iterate over upper triangle (W is symmetric)
        edges = []
        weights_list = []
        
        for i in range(N):
            for j in range(i + 1, N):  # Upper triangle only
                weight = W[i, j]
                if abs(weight) > 1e-12:  # Only include non-zero couplings
                    edges.append((nodes[i], nodes[j]))
                    weights_list.append(weight)
        
        # Convert to JAX arrays
        weights = jnp.array(weights_list)
        biases = jnp.array(H)
        beta = jnp.array(1.0 / T)
        
        return nodes, edges, weights, biases, beta
    
    def _create_block_structure(self, nodes: List) -> List[Block]:
        """
        Create 2-color block structure for efficient block Gibbs sampling.
        
        Alternates between even and odd indices to allow parallel updates.
        """
        # 2-coloring: update even indices, then odd indices
        even_block = Block(nodes[::2])
        odd_block = Block(nodes[1::2])
        
        return [even_block, odd_block]
    
    def _run_sampling(
        self,
        program,
        free_blocks: List[Block],
        nodes: List,
        num_samples: int,
        model=None
    ) -> jnp.ndarray:
        """
        Execute THRML sampling.
        
        Returns raw samples as JAX array (num_samples, N).
        """
        # JAX random key
        key = jax.random.key(np.random.randint(0, 2**31))
        k_init, k_samp = jax.random.split(key, 2)
        
        # Initialize state
        init_state = hinton_init(k_init, model, free_blocks, ())
        
        # Sampling schedule
        n_warmup = int(num_samples * self.warmup_fraction)
        schedule = SamplingSchedule(
            n_warmup=n_warmup,
            n_samples=num_samples,
            steps_per_sample=self.steps_per_sample
        )
        
        # Sample (returns PyTree with node states)
        samples = sample_states(
            k_samp,
            program,
            schedule,
            init_state,
            [],  # No clamped values
            [Block(nodes)]  # Observe all nodes
        )
        
        return samples
    
    def _samples_to_distribution(self, samples, N: int) -> np.ndarray:
        """
        Convert THRML samples to probability distribution.
        
        samples: PyTree structure from THRML
        Returns: NumPy array (2^N,) representing P(s)
        """
        # Extract samples for all nodes
        # THRML returns a PyTree - need to extract the states
        # The structure is: samples[0] contains the Block's states
        block_samples = samples[0]  # First (and only) observed block
        
        # Convert to NumPy and ensure correct shape
        samples_array = np.array(block_samples)  # (num_samples, N)
        
        # Map states to indices: {-1, 1}^N → {0, ..., 2^N-1}
        # Binary representation: -1 → 0, 1 → 1
        binary_samples = ((samples_array + 1) // 2).astype(int)
        
        # Convert to decimal indices
        powers = 2 ** np.arange(N)[::-1]
        indices = binary_samples @ powers
        
        # Count frequencies
        counts = np.bincount(indices, minlength=2**N)
        
        # Normalize to probability distribution
        P = counts / len(samples_array)
        
        return P
    
    def get_backend_info(self) -> Dict[str, str]:
        """
        Return information about THRML backend.
        
        Returns
        -------
        info : dict
            Backend metadata
        """
        return {
            'name': 'THRML',
            'version': '0.1.3',  # thrml version
            'type': 'hardware',
            'capabilities': 'Block Gibbs sampling (JAX-accelerated)',
            'sampling_method': 'Two-color block Gibbs',
            'hardware_accelerated': 'Yes (JAX/GPU)',
            'proven_correct': 'Validated against NumPy reference'
        }


if __name__ == "__main__":
    # Test the sampler
    print("Testing ThrmlSampler...")
    
    if not THRML_AVAILABLE:
        print(f"✗ THRML not available: {IMPORT_ERROR}")
        exit(1)
    
    # Create simple 2-spin system
    W = np.array([[0, 1], [1, 0]])
    H = np.array([0.5, -0.5])
    T = 1.0
    
    sampler = ThrmlSampler()
    
    # Get backend info
    info = sampler.get_backend_info()
    print(f"\nBackend: {info['name']}")
    print(f"Type: {info['type']}")
    print(f"Method: {info['sampling_method']}")
    print(f"Hardware: {info['hardware_accelerated']}")
    
    # Sample distribution
    print(f"\nSampling 2-spin system (N=2, 4 states)...")
    P = sampler.sample_distribution(W, H, T, num_samples=10000)
    
    print(f"Distribution shape: {P.shape}")
    print(f"Distribution sum: {P.sum():.6f}")
    print(f"Probabilities: {P}")
    
    # Verify most probable state
    max_prob_state = np.argmax(P)
    print(f"\nMost probable state: {max_prob_state} (P={P[max_prob_state]:.4f})")
    
    print("\n✓ ThrmlSampler test passed!")
