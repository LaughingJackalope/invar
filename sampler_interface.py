"""
Sampler Interface - Abstract Base Class for Boltzmann Samplers

Defines the contract between Layer 1 (mathematical core) and Layer 3 (backends).
This abstraction allows pluggable sampling implementations while maintaining
mathematical correctness.

Architecture:
    Layer 1 (Math) → Interface → Layer 3 (Samplers)
    
Implementations:
    - NumpySampler: Pure Python reference (numpy_sampler.py)
    - ThrmlSampler: Hardware-accelerated JAX backend (thrml_sampler.py)
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Tuple


class BoltzmannSampler(ABC):
    """
    Abstract base class for sampling from Boltzmann distributions.
    
    All samplers must implement this interface to be compatible with
    the scale invariance framework (Systems 1-7).
    
    Contract:
        - Input: Ising model parameters (W, H, T) + sample count
        - Output: Probability distribution P(s) over all 2^N states
        - Format: NumPy array of length 2^N, normalized to sum=1
        - States: Indexed as binary numbers (0 to 2^N-1)
        - Spin representation: {-1, +1}
    """
    
    @abstractmethod
    def sample_distribution(
        self,
        W: np.ndarray,
        H: np.ndarray,
        T: float,
        num_samples: int
    ) -> np.ndarray:
        """
        Sample from Boltzmann distribution P(s) ∝ exp(-E(s)/T).
        
        Energy function: E(s) = -s^T W s - H^T s
        
        Parameters
        ----------
        W : np.ndarray
            Interaction matrix (N×N, symmetric)
            W[i,j] represents coupling between spins i and j
            Must satisfy: W[i,j] == W[j,i], W[i,i] == 0
        H : np.ndarray
            Bias vector (N,)
            H[i] represents external field on spin i
        T : float
            Temperature (positive)
            Controls exploration vs exploitation
            T → 0: deterministic (lowest energy)
            T → ∞: uniform distribution
        num_samples : int
            Number of MCMC samples to generate
            More samples → better approximation
            Typical: 10k-50k for N=5-6
            
        Returns
        -------
        P : np.ndarray
            Probability distribution (2^N,)
            P[i] = probability of state i
            States indexed as binary: 000...0 = 0, 111...1 = 2^N-1
            Binary mapping: -1 → 0, +1 → 1
            Must satisfy: P.sum() ≈ 1.0, all(P >= 0)
            
        Notes
        -----
        Implementation must:
        1. Sample states from equilibrium distribution
        2. Convert states to binary indices
        3. Count state frequencies
        4. Normalize to probability distribution
        5. Return as NumPy array (NOT JAX or other)
        
        Examples
        --------
        >>> sampler = NumpySampler()  # or ThrmlSampler()
        >>> W = np.array([[0, 1], [1, 0]])
        >>> H = np.array([0.5, -0.5])
        >>> T = 1.0
        >>> P = sampler.sample_distribution(W, H, T, num_samples=10000)
        >>> print(P.shape)
        (4,)  # 2^2 states
        >>> print(np.isclose(P.sum(), 1.0))
        True
        """
        pass
    
    @abstractmethod
    def get_backend_info(self) -> Dict[str, str]:
        """
        Return information about the sampling backend.
        
        Returns
        -------
        info : dict
            Dictionary with backend metadata:
            - 'name': Backend name (e.g., 'NumPy', 'THRML')
            - 'version': Version string
            - 'type': 'reference' or 'hardware'
            - 'capabilities': Human-readable description
            
        Examples
        --------
        >>> sampler = NumpySampler()
        >>> info = sampler.get_backend_info()
        >>> print(info['name'])
        'NumPy'
        """
        pass
    
    def validate_inputs(
        self,
        W: np.ndarray,
        H: np.ndarray,
        T: float,
        num_samples: int
    ) -> None:
        """
        Validate input parameters (optional helper).
        
        Raises
        ------
        ValueError
            If inputs don't satisfy requirements
        """
        N = len(H)
        
        # Check W shape and symmetry
        if W.shape != (N, N):
            raise ValueError(f"W must be {N}×{N}, got {W.shape}")
        
        if not np.allclose(W, W.T):
            raise ValueError("W must be symmetric")
        
        if not np.allclose(np.diag(W), 0):
            raise ValueError("W diagonal must be zero (no self-interactions)")
        
        # Check H shape
        if H.shape != (N,):
            raise ValueError(f"H must be length {N}, got {H.shape}")
        
        # Check temperature
        if T <= 0:
            raise ValueError(f"Temperature must be positive, got {T}")
        
        # Check sample count
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        
        if num_samples < 100:
            print(f"Warning: num_samples={num_samples} is very small (recommend ≥1000)")
    
    def validate_output(self, P: np.ndarray, N: int) -> None:
        """
        Validate output distribution (optional helper).
        
        Raises
        ------
        ValueError
            If output doesn't satisfy requirements
        """
        expected_size = 2**N
        
        if P.shape != (expected_size,):
            raise ValueError(f"P must be length {expected_size}, got {P.shape}")
        
        if not np.all(P >= 0):
            raise ValueError("All probabilities must be non-negative")
        
        if not np.isclose(P.sum(), 1.0, atol=1e-6):
            raise ValueError(f"Distribution must sum to 1, got {P.sum()}")


class SamplerFactory:
    """
    Factory for creating appropriate sampler instances.
    
    Handles graceful fallback if hardware backends unavailable.
    """
    
    @staticmethod
    def create_sampler(backend: str = 'numpy') -> BoltzmannSampler:
        """
        Create a sampler instance.
        
        Parameters
        ----------
        backend : str
            Backend type: 'numpy', 'thrml', or 'auto'
            'auto' tries thrml first, falls back to numpy
            
        Returns
        -------
        sampler : BoltzmannSampler
            Sampler instance
            
        Examples
        --------
        >>> sampler = SamplerFactory.create_sampler('numpy')
        >>> sampler = SamplerFactory.create_sampler('thrml')
        >>> sampler = SamplerFactory.create_sampler('auto')
        """
        if backend == 'numpy':
            from numpy_sampler import NumpySampler
            return NumpySampler()
        
        elif backend == 'thrml':
            try:
                from thrml_sampler import ThrmlSampler
                return ThrmlSampler()
            except ImportError as e:
                raise ImportError(
                    "THRML backend requested but not available. "
                    "Install with: pip install thrml"
                ) from e
        
        elif backend == 'auto':
            try:
                from thrml_sampler import ThrmlSampler
                print("Using THRML backend (hardware-accelerated)")
                return ThrmlSampler()
            except ImportError:
                from numpy_sampler import NumpySampler
                print("THRML not available, using NumPy backend (reference)")
                return NumpySampler()
        
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'numpy', 'thrml', or 'auto'")


# Convenience function for Layer 1
def create_default_sampler() -> BoltzmannSampler:
    """
    Create default sampler for Layer 1 code.
    
    Returns NumpySampler (always available, no dependencies).
    Layer 1 code can override by passing explicit sampler parameter.
    
    Returns
    -------
    sampler : BoltzmannSampler
        Default NumpySampler instance
    """
    from numpy_sampler import NumpySampler
    return NumpySampler()
