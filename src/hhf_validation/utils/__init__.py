"""
HHF Validation Utilities
========================
Physical constants, plotting utilities, and helper functions for HHF validation.
"""

from .constants import (
    # Fundamental constants
    L_P, h, hbar, c, m_p, m_e, alpha, e, k_B, N_A, u, R_INF,
    PLANCK_CONSTANT, HBAR, SPEED_OF_LIGHT, PROTON_MASS, ELECTRON_MASS,
    FINE_STRUCTURE, ELEMENTARY_CHARGE, BOLTZMANN, AVOGADRO,
    ATOMIC_MASS_UNIT, RYDBERG_INF,
    # HHF constants
    R_HHF, HHF_RADIUS, LAMBDA_HH, HHF_SCALING_RATIO, S_H, V_H,
    compute_hhf_radius, compute_hhf_scaling_ratio,
    # Isotopologue data
    ISOTOPOLOGUE_DATA, compute_reduced_mass, compute_effective_frequency
)

__all__ = [
    'L_P', 'h', 'hbar', 'c', 'm_p', 'm_e', 'alpha', 'e', 'k_B', 'N_A', 'u', 'R_INF',
    'PLANCK_CONSTANT', 'HBAR', 'SPEED_OF_LIGHT', 'PROTON_MASS', 'ELECTRON_MASS',
    'FINE_STRUCTURE', 'ELEMENTARY_CHARGE', 'BOLTZMANN', 'AVOGADRO',
    'ATOMIC_MASS_UNIT', 'RYDBERG_INF',
    'R_HHF', 'HHF_RADIUS', 'LAMBDA_HH', 'HHF_SCALING_RATIO', 'S_H', 'V_H',
    'compute_hhf_radius', 'compute_hhf_scaling_ratio',
    'ISOTOPOLOGUE_DATA', 'compute_reduced_mass', 'compute_effective_frequency'
]

