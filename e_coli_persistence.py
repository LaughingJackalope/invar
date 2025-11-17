"""
Digital Twin of E. coli Persistence
-----------------------------------

This module implements a minimal, self-contained two-state thermodynamic
model for bacterial persistence switching under environmental stress.

Core features:
- Two-state Boltzmann policy for Normal (S1) vs Persister (S2)
- Free Energy Landscape (1D) along a reaction coordinate X
- Stress-to-energy mapping for antibiotics to shift ΔG and ΔG‡
- Lightweight environment string parser and end-to-end runner

Notes:
- Energies are expressed in J/mol by default. Gas constant R uses SI units.
- Temperatures are in Kelvin internally; simple parser accepts °C.
- The FEL rendering helper uses cubic interpolation between key points.

This module avoids any dependencies beyond the Python standard library,
NumPy, and Matplotlib (for optional plotting). It is designed to integrate
cleanly with the existing repository without altering tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Union, Iterable
import re
import numpy as np


# Universal gas constant (J/mol/K)
R_GAS = 8.31446261815324


def compute_persister_probability(G1: float, G2: float, T: float, R: float = R_GAS) -> float:
    """Return P(Persister) given free energies of Normal (G1) and Persister (G2).

    P(Persister) = exp(-G2/RT) / [exp(-G1/RT) + exp(-G2/RT)]

    Args:
        G1: Gibbs free energy for Normal state (J/mol)
        G2: Gibbs free energy for Persister state (J/mol)
        T: Temperature in Kelvin
        R: Gas constant (default SI J/mol/K)
    """
    # Use numerically stable logistic formulation using ΔG = G2 - G1
    dG = G2 - G1
    # ratio = exp(-ΔG/RT)
    z = -dG / (R * T)
    # P2 = 1 / (1 + exp(+ΔG/RT))
    # clip z to avoid overflow in exp
    z = np.clip(z, -700, 700)
    return 1.0 / (1.0 + np.exp(-z))


def persister_ratio_from_deltaG(deltaG: float, T: float, R: float = R_GAS) -> float:
    """Return P(Persister)/P(Normal) = exp(-ΔG/RT)."""
    z = -deltaG / (R * T)
    z = np.clip(z, -700, 700)
    return float(np.exp(z))


@dataclass
class FELParams:
    """Free Energy Landscape key points along reaction coordinate X.

    The landscape is represented by three anchor points:
    - x1, G1: Normal minimum
    - xb, Gb: Barrier (transition state)
    - x2, G2: Persister minimum
    """

    x1: float = 0.0
    x2: float = 1.0
    xb: float = 0.5
    G1: float = 0.0          # J/mol
    G2: float = 5_000.0      # J/mol (persister typically less favorable)
    Gb: float = 25_000.0     # J/mol barrier height


def _cubic_interpolate(xs: np.ndarray, xk: Iterable[float], yk: Iterable[float]) -> np.ndarray:
    """Simple C2 cubic spline through three key points using piecewise cubics.

    We construct two cubics on [x1, xb] and [xb, x2] with zero slope at minima
    (x1, x2) and zero slope at barrier peak (xb) for a smooth iconic shape.
    This is not a physical derivation, just a visually faithful FEL for plots.
    """
    x1, xb, x2 = xk
    y1, yb, y2 = yk

    def cubic_segment(xa, ya, da, xb_, yb_, db):
        # Solve for a cubic y = a t^3 + b t^2 + c t + d, t in [0,1]
        # with constraints y(0)=ya, y'(0)=da, y(1)=yb, y'(1)=db
        # Return coefficients for mapping t=(x-xa)/(xb-xa)
        L = xb_ - xa
        if L == 0:
            return (0.0, 0.0, 0.0, ya)
        A = np.array(
            [
                [1, 1, 1, 1],      # for y(1)
                [3, 2, 1, 0],      # for y'(1) scaled by L
                [0, 0, 1, 0],      # for y'(0) scaled by L
                [0, 0, 0, 1],      # for y(0)
            ],
            dtype=float,
        )
        b = np.array([yb, db * L, da * L, ya], dtype=float)
        # Solve for coefficients in basis [t^3, t^2, t, 1]
        a3, a2, a1, a0 = np.linalg.solve(A, b)
        return a3, a2, a1, a0

    # Zero slope at the three anchor points by construction
    aL = cubic_segment(x1, y1, 0.0, xb, yb, 0.0)
    aR = cubic_segment(xb, yb, 0.0, x2, y2, 0.0)

    ys = np.empty_like(xs, dtype=float)
    # Left segment
    left_mask = xs <= xb
    if np.any(left_mask):
        t = (xs[left_mask] - x1) / (xb - x1 + 1e-12)
        a3, a2, a1, a0 = aL
        ys[left_mask] = ((a3 * t + a2) * t + a1) * t + a0
    # Right segment
    right_mask = xs > xb
    if np.any(right_mask):
        t = (xs[right_mask] - xb) / (x2 - xb + 1e-12)
        a3, a2, a1, a0 = aR
        ys[right_mask] = ((a3 * t + a2) * t + a1) * t + a0
    return ys


def double_well_free_energy(X: np.ndarray, params: FELParams, T: float) -> np.ndarray:
    """Evaluate a smooth double-well Free Energy Landscape over points X.

    The values match exactly the provided G1, Gb, G2 at x1, xb, x2 and are
    smoothly interpolated elsewhere for visualization and qualitative trends.
    """
    xk = (params.x1, params.xb, params.x2)
    yk = (params.G1, params.Gb, params.G2)
    return _cubic_interpolate(np.asarray(X, dtype=float), xk, yk)


@dataclass
class StressMapping:
    """Coefficients for mapping environmental stress to energy shifts.

    Shifts are reported in J/mol. The mapping uses a soft-log relationship
    with antibiotic concentration in μM to model saturation-like effects.
    """

    k_deltaG_kJ_per_mol: float = -2.0  # kJ/mol per log10(1 + conc_μM)
    k_barrier_kJ_per_mol: float = 1.5  # kJ/mol per log10(1 + conc_μM)
    temp_sensitivity_kJ_per_mol_per_K: float = 0.0  # optional

    def shifts(self, antibiotic_uM: float, temp_K: float, base_temp_K: float = 310.15) -> Tuple[float, float]:
        c = max(0.0, float(antibiotic_uM))
        s = np.log10(1.0 + c)
        dG_kJ = self.k_deltaG_kJ_per_mol * s + self.temp_sensitivity_kJ_per_mol_per_K * (temp_K - base_temp_K)
        dGb_kJ = self.k_barrier_kJ_per_mol * s
        return dG_kJ * 1e3, dGb_kJ * 1e3  # to J/mol


ENV_PATTERN = re.compile(
    r"(?i)\b(?P<drug>[A-Za-z0-9_\- ]+)\b\s*,?\s*"
    r"(?P<conc>\d+(?:\.\d+)?)\s*(?:uM|µM|μM)\s*,?\s*"
    r"(?P<temp>\-?\d+(?:\.\d+)?)\s*(?:C|°C)"
)


def parse_environment(env: Union[str, Dict]) -> Dict:
    """Parse environment inputs into a normalized dict.

    Accepts either a string like "Antibiotic A, 20uM, 37C" or a dict with keys:
    {antibiotic: str, concentration_uM: float, temperature_C: float}
    """
    if isinstance(env, dict):
        drug = str(env.get("antibiotic", "Antibiotic")).strip()
        conc = float(env.get("concentration_uM", 0.0))
        tempC = float(env.get("temperature_C", 37.0))
        return {"antibiotic": drug, "concentration_uM": conc, "temperature_C": tempC}
    if not isinstance(env, str):
        raise TypeError("env must be a str or dict")
    m = ENV_PATTERN.search(env)
    if not m:
        # Fallback: try to find just numbers for conc and temp
        numbers = [float(x) for x in re.findall(r"[-+]?[0-9]*\.?[0-9]+", env)]
        conc = numbers[0] if numbers else 0.0
        tempC = numbers[1] if len(numbers) > 1 else 37.0
        drug = "Antibiotic"
    else:
        drug = m.group("drug").strip()
        conc = float(m.group("conc"))
        tempC = float(m.group("temp"))
    return {"antibiotic": drug, "concentration_uM": conc, "temperature_C": tempC}


@dataclass
class BaseModel:
    """Baseline energies and geometry for the two-state model (J/mol)."""

    # Minima positions for reaction coordinate X
    x1: float = 0.2
    x2: float = 0.8
    xb: float = 0.5

    # Energies (J/mol)
    G1: float = 0.0
    G2: float = 7_000.0
    Gb: float = 30_000.0

    # Reference temperature (K)
    T_ref_K: float = 310.15  # 37°C

    def fel_params(self) -> FELParams:
        return FELParams(x1=self.x1, x2=self.x2, xb=self.xb, G1=self.G1, G2=self.G2, Gb=self.Gb)


def apply_stress(base: BaseModel, mapping: StressMapping, conc_uM: float, temp_C: float) -> FELParams:
    """Apply stress-induced shifts to ΔG and ΔG‡ and return new FELParams."""
    T_K = temp_C + 273.15
    dG, dGb = mapping.shifts(conc_uM, T_K, base.T_ref_K)
    # Shift G2 relative to G1 by dG and barrier by dGb; keep G1 as reference 0
    G1 = base.G1
    G2 = base.G2 + dG
    Gb = max(base.Gb + dGb, max(G1, G2) + 1_000.0)  # ensure barrier remains above minima
    return FELParams(x1=base.x1, x2=base.x2, xb=base.xb, G1=G1, G2=G2, Gb=Gb)


def run_persistence_prediction(
    env: Union[str, Dict],
    base: BaseModel | None = None,
    mapping: StressMapping | None = None,
) -> Dict:
    """End-to-end prediction for P(Persister) with FEL and stress mapping.

    Returns a dict with fields:
    - inputs: parsed environment
    - temperature_K
    - delta_G, delta_G_dagger (J/mol)
    - P_persister
    - fel_params: the FELParams used
    """
    base = base or BaseModel()
    mapping = mapping or StressMapping()
    info = parse_environment(env)
    T_K = info["temperature_C"] + 273.15
    params = apply_stress(base, mapping, info["concentration_uM"], info["temperature_C"])

    dG = params.G2 - params.G1
    dG_dagger = params.Gb - min(params.G1, params.G2)
    P2 = compute_persister_probability(params.G1, params.G2, T_K)

    return {
        "inputs": info,
        "temperature_K": T_K,
        "delta_G": dG,
        "delta_G_dagger": dG_dagger,
        "P_persister": P2,
        "fel_params": params,
    }


def sweep_concentration(
    concentrations_uM: Iterable[float], env_template: Union[str, Dict], base: BaseModel | None = None, mapping: StressMapping | None = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate P(Persister) across a set of antibiotic concentrations (μM)."""
    base = base or BaseModel()
    mapping = mapping or StressMapping()
    if isinstance(env_template, str):
        # try to replace the first number with concentration; otherwise fall back
        def subst(s: str, value: float) -> str:
            return re.sub(r"(\d+(?:\.\d+)?)(?=\s*(?:uM|µM|μM))", f"{value}", s, count=1)

        envs = [subst(env_template, c) for c in concentrations_uM]
    else:
        d = dict(env_template)
        envs = []
        for c in concentrations_uM:
            d["concentration_uM"] = float(c)
            envs.append(dict(d))

    Ps = []
    for e in envs:
        res = run_persistence_prediction(e, base=base, mapping=mapping)
        Ps.append(res["P_persister"])
    return np.array(list(concentrations_uM), dtype=float), np.array(Ps, dtype=float)


# Optional plotting helpers kept lightweight to avoid hard dependencies at import
def _maybe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception as _:
        return None


def plot_fel(params: FELParams, T_K: float, n: int = 200):
    """Plot the free energy landscape with annotated states and barrier."""
    plt = _maybe_import_matplotlib()
    if plt is None:
        raise RuntimeError("matplotlib is required for plotting")
    X = np.linspace(min(params.x1, params.x2) - 0.2, max(params.x1, params.x2) + 0.2, n)
    G = double_well_free_energy(X, params, T_K)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(X, G / 1000.0, color="#1f77b4", lw=2)
    ax.plot([params.x1, params.xb, params.x2], [params.G1/1000.0, params.Gb/1000.0, params.G2/1000.0],
            "o", color="#2ca02c")
    ax.set_xlabel("Reaction coordinate X")
    ax.set_ylabel("Free energy (kJ/mol)")
    ax.set_title("E. coli Persistence Free Energy Landscape")
    ax.annotate("Normal (S1)", (params.x1, params.G1/1000.0), xytext=(params.x1-0.25, params.G1/1000.0+5),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("Barrier ΔG‡", (params.xb, params.Gb/1000.0), xytext=(params.xb+0.1, params.Gb/1000.0+5),
                arrowprops=dict(arrowstyle="->"))
    ax.annotate("Persister (S2)", (params.x2, params.G2/1000.0), xytext=(params.x2+0.05, params.G2/1000.0+5),
                arrowprops=dict(arrowstyle="->"))
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig, ax


def demo_quickstart():
    """Quick demonstration returning result dict and a FEL plot figure."""
    env = "Antibiotic A, 20uM, 37C"
    res = run_persistence_prediction(env)
    fig, ax = plot_fel(res["fel_params"], res["temperature_K"])
    return res, fig
