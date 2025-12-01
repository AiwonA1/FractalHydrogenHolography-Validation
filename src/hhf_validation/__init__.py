"""
HHF Validation Package
======================

Empirical validation suite for the Hydrogen-Holographic Fractal (HHF) framework.

This package provides automated validation pipelines for testing HHF predictions
across atomic, molecular, macro, and biological scales using real-world data.

Key Theoretical Results
-----------------------
The HHF framework predicts fractal holographic scaling from the Planck scale
to macroscopic systems, characterized by:

- **HHF Radius:** R_HHF = h / (m_p × c × α) ≈ 1.81 × 10⁻¹³ m
- **Scaling Ratio:** Λᴴᴴ = R_HHF / L_P ≈ 1.12 × 10²²

Modules
-------
core : Shared fractal analysis utilities (PFD, HFD)
validations : Validation experiment modules
utils : Physical constants (CODATA 2018) and utility functions

Quick Start
-----------
>>> from hhf_validation.validations import run_biological_validation
>>> result = run_biological_validation()

>>> from hhf_validation.core import petrosian_fd, higuchi_fd
>>> pfd = petrosian_fd(signal)
>>> hfd = higuchi_fd(signal)

>>> from hhf_validation.utils import R_HHF, LAMBDA_HH
>>> print(f"HHF Radius: {R_HHF:.2e} m")

Validations
-----------
1. **Biological Proxy:** Fractal analysis of environmental time series (PFD, HFD)
2. **Isotopologue Scaling:** Mass-invariant HHF scaling across H₂, D₂, T₂
3. **Molecular/Photonic:** HHF constants and Rydberg spectral consistency
4. **PEFF Seismic/EEG:** Cross-domain fractal signatures

References
----------
- CODATA 2018 fundamental constants
- NIST spectroscopic data for hydrogen isotopologues
"""

__version__ = "1.0.0"
__author__ = "FractiAI Research Team"

from .core import petrosian_fd, higuchi_fd, compute_fractal_metrics
from .utils import (
    L_P, h, c, m_p, m_e, alpha,
    R_HHF, LAMBDA_HH,
    ISOTOPOLOGUE_DATA
)

__all__ = [
    # Version info
    '__version__',
    '__author__',
    # Core functions
    'petrosian_fd',
    'higuchi_fd',
    'compute_fractal_metrics',
    # Constants
    'L_P', 'h', 'c', 'm_p', 'm_e', 'alpha',
    'R_HHF', 'LAMBDA_HH',
    'ISOTOPOLOGUE_DATA',
]

