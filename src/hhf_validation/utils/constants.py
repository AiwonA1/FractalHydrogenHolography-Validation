"""
Physical and HHF Constants Module
=================================
CODATA 2018 fundamental constants and HHF-derived theoretical values.

These constants provide the empirical foundation for HHF validation experiments.
"""

import numpy as np

# =============================================================================
# CODATA 2018 / SI Exact Fundamental Constants
# =============================================================================

# Planck length (m)
L_P = 1.616255e-35

# Planck constant (J·s) - Exact in SI
h = 6.62607015e-34
PLANCK_CONSTANT = h

# Reduced Planck constant (J·s)
hbar = h / (2 * np.pi)
HBAR = hbar

# Speed of light (m/s) - Exact in SI
c = 2.99792458e8
SPEED_OF_LIGHT = c

# Proton mass (kg)
m_p = 1.67262192369e-27
PROTON_MASS = m_p

# Electron mass (kg)
m_e = 9.1093837015e-31
ELECTRON_MASS = m_e

# Fine-structure constant (dimensionless)
alpha = 7.2973525693e-3
FINE_STRUCTURE = alpha

# Elementary charge (C) - Exact in SI
e = 1.602176634e-19
ELEMENTARY_CHARGE = e

# Boltzmann constant (J/K) - Exact in SI
k_B = 1.380649e-23
BOLTZMANN = k_B

# Avogadro constant (mol^-1) - Exact in SI
N_A = 6.02214076e23
AVOGADRO = N_A

# Atomic mass unit (kg)
u = 1.66053906660e-27
ATOMIC_MASS_UNIT = u

# Rydberg constant for infinite mass (m^-1)
R_INF = 10973731.568160
RYDBERG_INF = R_INF

# =============================================================================
# HHF Theoretical Constants
# =============================================================================

def compute_hhf_radius() -> float:
    """
    Compute the HHF radius from fundamental constants.
    
    R_HHF = h / (m_p * c * α) ≈ 1.81 × 10^-13 m
    
    This represents the characteristic holographic scale linking the proton
    Compton wavelength (h / m_p c) with the fine-structure constant α.
    The result is approximately 6× the proton charge radius.
    
    Returns
    -------
    float
        HHF radius in meters.
        
    Notes
    -----
    The HHF radius relates quantum and relativistic scales through:
    R_HHF = λ_proton / α
    
    where λ_proton = h / (m_p * c) is the proton Compton wavelength.
    """
    # Proton Compton wavelength: λ_p = h / (m_p * c)
    # HHF radius: R_HHF = λ_p / α = h / (m_p * c * α)
    lambda_proton = h / (m_p * c)
    return lambda_proton / alpha


def compute_hhf_scaling_ratio() -> float:
    """
    Compute the HHF scaling ratio: R_HHF / L_P
    
    Λ^HH = R_HHF / L_P ≈ 1.12 × 10^22
    
    This dimensionless ratio represents the holographic scaling factor
    between the Planck scale and the hydrogen holographic scale.
    
    Returns
    -------
    float
        Dimensionless scaling ratio.
    """
    R_HHF = compute_hhf_radius()
    return R_HHF / L_P


# Pre-computed HHF values
R_HHF = compute_hhf_radius()
HHF_RADIUS = R_HHF

LAMBDA_HH = compute_hhf_scaling_ratio()
HHF_SCALING_RATIO = LAMBDA_HH

# Area and volume scaling
S_H = LAMBDA_HH ** 2  # Area scaling
V_H = LAMBDA_HH ** 3  # Volume scaling

# =============================================================================
# Isotopologue Spectroscopic Constants (NIST)
# =============================================================================

# Vibrational constants for hydrogen isotopologues (cm^-1)
# omega_e: harmonic frequency, omega_exe: anharmonicity constant
ISOTOPOLOGUE_DATA = {
    'H2': {
        'omega_e': 4401.21,      # cm^-1
        'omega_exe': 121.33,    # cm^-1
        'mass_1': 1.00782503207,  # u (protium)
        'mass_2': 1.00782503207,  # u
    },
    'D2': {
        'omega_e': 3115.50,
        'omega_exe': 60.28,
        'mass_1': 2.01410177785,  # u (deuterium)
        'mass_2': 2.01410177785,
    },
    'T2': {
        'omega_e': 2546.47,
        'omega_exe': 41.735,
        'mass_1': 3.0160492777,   # u (tritium)
        'mass_2': 3.0160492777,
    },
    'HD': {
        'omega_e': 3813.15,
        'omega_exe': 91.65,
        'mass_1': 1.00782503207,
        'mass_2': 2.01410177785,
    }
}


def compute_reduced_mass(mass_1: float, mass_2: float) -> float:
    """
    Compute reduced mass for a diatomic molecule.
    
    Parameters
    ----------
    mass_1, mass_2 : float
        Atomic masses in atomic mass units (u).
        
    Returns
    -------
    float
        Reduced mass in atomic mass units.
    """
    return (mass_1 * mass_2) / (mass_1 + mass_2)


def compute_effective_frequency(omega_e: float, omega_exe: float) -> float:
    """
    Compute anharmonic-corrected effective vibrational frequency.
    
    omega_eff = omega_e - 2 * omega_exe
    
    Parameters
    ----------
    omega_e : float
        Harmonic frequency (cm^-1).
    omega_exe : float
        Anharmonicity constant (cm^-1).
        
    Returns
    -------
    float
        Effective frequency (cm^-1).
    """
    return omega_e - 2 * omega_exe

