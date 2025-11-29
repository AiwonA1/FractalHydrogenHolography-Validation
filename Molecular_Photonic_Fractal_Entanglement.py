import numpy as np
from scipy.constants import physical_constants

# --- 1. Setup and Official CODATA 2018 Constants ---
# Using high-precision CODATA 2018 values, derived from official sources (NIST/CODATA).
# Some constants (c, h) are exact in the SI redefinition, which CODATA 2018 adopted.

print("--- 1. Defining Official CODATA 2018 Constants ---")

# Define constants directly from CODATA 2018 / SI Exact Values
L_P = 1.616255e-35      # Planck length (m)
h = 6.62607015e-34      # Planck constant (J·s) - Exact in SI
c = 2.99792458e+8       # Speed of light (m/s) - Exact in SI
m_p = 1.67262192369e-27 # Proton mass (kg)
m_e = 9.1093837015e-31  # Electron mass (kg)
alpha = 7.2973525693e-3 # Fine-structure constant (dimensionless)

print(f"Planck length (L_P): {L_P:.4e} m")
print(f"Proton mass (m_p): {m_p:.12e} kg\n")

# --- 2. Spectral Validation (Rydberg Constant R_H) ---
# Validates the input constants against the foundational constant derived from Hydrogen spectra.

print("--- 2. Spectral Validation via Hydrogen Rydberg Constant (R_H) ---")

# Calculate Rydberg constant for infinite mass (R_infinity)
# Formula: R_infinity = (m_e * c * alpha^2) / (2 * h)
R_inf = (m_e * c * alpha**2) / (2 * h)
print(f"Rydberg (Infinite Mass) R_inf: {R_inf:.10f} m^-1")

# Calculate reduced mass correction for Hydrogen atom
# Factor: m_p / (m_e + m_p)
reduced_mass_factor = m_p / (m_e + m_p)

# Calculate Rydberg constant for Hydrogen atom (R_H)
R_H_calc = R_inf * reduced_mass_factor
