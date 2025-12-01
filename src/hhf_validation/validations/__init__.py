"""
HHF Validation Modules
======================
Empirical validation pipelines for the Hydrogen-Holographic Fractal framework.

This module provides four independent validation experiments:

Biological Proxy (biological_proxy)
-----------------------------------
Analyzes fractal properties of environmental time series (e.g., temperature data).
Computes Petrosian Fractal Dimension (PFD) and Higuchi Fractal Dimension (HFD).

>>> from hhf_validation.validations import run_biological_validation
>>> result = run_biological_validation(output_dir="outputs/biological")
>>> print(f"PFD: {result.pfd:.4f}, HFD: {result.hfd:.4f}")

Isotopologue Scaling (isotopologue)
-----------------------------------
Validates mass-invariant HHF scaling across hydrogen isotopes (H₂, D₂, T₂).
Computes normalized scaling ratio Λᴴᴴ which should be ~1.0 for all isotopes.

>>> from hhf_validation.validations import run_isotopologue_validation
>>> result = run_isotopologue_validation()
>>> print(f"Max deviation: {result.max_deviation:.4f}, Passed: {result.passed}")

Molecular/Photonic (molecular_photonic)
---------------------------------------
Validates HHF theoretical constants against CODATA and spectroscopic data.
Computes HHF radius (R_HHF ≈ 1.81e-13 m) and scaling ratio (Λᴴᴴ ≈ 1.12e22).

>>> from hhf_validation.validations import run_molecular_photonic_validation
>>> result = run_molecular_photonic_validation()
>>> print(f"Spectral validated: {result.spectral.validated}")

PEFF Seismic/EEG (peff_seismic_eeg)
-----------------------------------
Tests Paradise Energy Fractal Force hypothesis using seismic and EEG data.
Demonstrates cross-domain fractal signatures (PFD ≈ 1.02 for both).

>>> from hhf_validation.validations import run_peff_validation
>>> result = run_peff_validation(include_eeg=False)  # Seismic only
>>> print(f"Seismic PFD: {result.seismic.pfd:.4f}")
"""

from .biological_proxy import run_validation as run_biological_validation
from .isotopologue import run_validation as run_isotopologue_validation
from .molecular_photonic import run_validation as run_molecular_photonic_validation
from .peff_seismic_eeg import run_validation as run_peff_validation

__all__ = [
    'run_biological_validation',
    'run_isotopologue_validation',
    'run_molecular_photonic_validation',
    'run_peff_validation'
]

