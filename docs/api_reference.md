# API Reference

**HHF Validation Package Public API**

---

## Quick Import Reference

```python
# Core fractal functions
from hhf_validation.core import petrosian_fd, higuchi_fd, compute_fractal_metrics

# Validation runners
from hhf_validation.validations import (
    run_biological_validation,
    run_isotopologue_validation,
    run_molecular_photonic_validation,
    run_peff_validation
)

# Physical constants
from hhf_validation.utils import (
    h, c, m_p, m_e, alpha, L_P,  # CODATA 2018
    R_HHF, LAMBDA_HH, S_H, V_H   # HHF derived
)
```

---

## hhf_validation.core.fractal

Fractal dimension computation for time-series analysis.

### petrosian_fd

```python
def petrosian_fd(signal: ArrayLike) -> float
```

Compute Petrosian Fractal Dimension (PFD) of a 1D signal.

**Parameters:**
- `signal` (array-like): 1D input time series

**Returns:**
- `float`: PFD value (typically 1.0–1.03), or `np.nan` if signal too short

**Example:**
```python
import numpy as np
from hhf_validation.core import petrosian_fd

# White noise
signal = np.random.randn(1000)
pfd = petrosian_fd(signal)
print(f"PFD: {pfd:.4f}")  # ~1.03

# Sinusoidal (more regular)
t = np.linspace(0, 10, 1000)
signal = np.sin(2 * np.pi * t)
pfd = petrosian_fd(signal)
print(f"PFD: {pfd:.4f}")  # ~1.0
```

---

### higuchi_fd

```python
def higuchi_fd(signal: ArrayLike, k_max: int = 10) -> float
```

Compute Higuchi Fractal Dimension (HFD) of a 1D signal.

**Parameters:**
- `signal` (array-like): 1D input time series
- `k_max` (int, optional): Maximum interval scale (default: 10)

**Returns:**
- `float`: HFD value (1.0–2.0), or `np.nan` if insufficient data

**Example:**
```python
from hhf_validation.core import higuchi_fd

signal = np.random.randn(1000)
hfd = higuchi_fd(signal, k_max=10)
print(f"HFD: {hfd:.4f}")  # ~1.5 for noise
```

---

### compute_fractal_metrics

```python
def compute_fractal_metrics(signal: ArrayLike, k_max: int = 10) -> dict
```

Compute both PFD and HFD for a signal.

**Parameters:**
- `signal` (array-like): 1D input time series
- `k_max` (int, optional): Maximum interval scale for HFD

**Returns:**
- `dict`: `{'n_points': int, 'pfd': float, 'hfd': float}`

---

## hhf_validation.validations

Validation experiment runners.

### run_biological_validation

```python
def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> BiologicalValidationResult
```

Analyze fractal properties of environmental time series.

**Parameters:**
- `output_dir` (Path, optional): Output directory (default: `outputs/biological`)
- `show_plots` (bool): Display plots interactively

**Returns:**
- `BiologicalValidationResult`:
  - `.dataset_name` (str): Name of analyzed dataset
  - `.n_points` (int): Number of data points
  - `.pfd` (float): Petrosian Fractal Dimension
  - `.hfd` (float): Higuchi Fractal Dimension

**Example:**
```python
from hhf_validation.validations import run_biological_validation

result = run_biological_validation(output_dir="outputs/biological")
print(f"Dataset: {result.dataset_name}")
print(f"PFD: {result.pfd:.4f}")
print(f"HFD: {result.hfd:.4f}")
```

---

### run_isotopologue_validation

```python
def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
    threshold: float = 0.1
) -> IsotopologueValidationResult
```

Validate HHF scaling invariance across hydrogen isotopologues.

**Parameters:**
- `output_dir` (Path, optional): Output directory (default: `outputs/isotopologue`)
- `show_plots` (bool): Display plots interactively
- `threshold` (float): Maximum allowed deviation from unity (default: 0.1)

**Returns:**
- `IsotopologueValidationResult`:
  - `.isotopologues` (dict): Per-isotope results
  - `.max_deviation` (float): Maximum |Λᴴᴴ - 1|
  - `.threshold` (float): Pass/fail threshold
  - `.passed` (bool): Whether validation passed

**Example:**
```python
from hhf_validation.validations import run_isotopologue_validation

result = run_isotopologue_validation()
print(f"Passed: {result.passed}")
print(f"Max deviation: {result.max_deviation:.4f}")

for iso, data in result.isotopologues.items():
    print(f"{iso}: Λᴴᴴ = {data.lambda_hh:.4f}")
```

---

### run_molecular_photonic_validation

```python
def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> MolecularPhotonicResult
```

Validate HHF theoretical constants against spectroscopic data.

**Parameters:**
- `output_dir` (Path, optional): Output directory (default: `outputs/molecular_photonic`)
- `show_plots` (bool): Display plots interactively

**Returns:**
- `MolecularPhotonicResult`:
  - `.theoretical` (TheoreticalResult):
    - `.R_HHF` (float): HHF radius (m)
    - `.Lambda_HH` (float): Scaling ratio
    - `.S_H` (float): Area scaling
    - `.V_H` (float): Volume scaling
  - `.spectral` (SpectralResult):
    - `.R_H_calculated` (float): Calculated Rydberg constant
    - `.R_H_official` (float): CODATA value
    - `.relative_error` (float): Relative error
    - `.validated` (bool): Whether error < threshold

**Example:**
```python
from hhf_validation.validations import run_molecular_photonic_validation

result = run_molecular_photonic_validation()
print(f"R_HHF = {result.theoretical.R_HHF:.4e} m")
print(f"Λᴴᴴ = {result.theoretical.Lambda_HH:.4e}")
print(f"Spectral validated: {result.spectral.validated}")
```

---

### run_peff_validation

```python
def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
    include_seismic: bool = True,
    include_eeg: bool = True,
    seismic_start: str = "2025-01-01",
    seismic_end: str = "2025-01-31"
) -> PEFFValidationResult
```

Test PEFF hypothesis with seismic and EEG data.

**Parameters:**
- `output_dir` (Path, optional): Output directory (default: `outputs/peff`)
- `show_plots` (bool): Display plots interactively
- `include_seismic` (bool): Include seismic analysis (requires network)
- `include_eeg` (bool): Include EEG analysis (requires MNE + network)
- `seismic_start` (str): Seismic query start date (YYYY-MM-DD)
- `seismic_end` (str): Seismic query end date (YYYY-MM-DD)

**Returns:**
- `PEFFValidationResult`:
  - `.seismic` (SeismicResult, optional):
    - `.n_points` (int): Number of events
    - `.pfd` (float): Seismic PFD
    - `.date_range` (str): Query date range
  - `.eeg` (EEGResult, optional):
    - `.mean_pfd` (float): Mean PFD across channels
    - `.min_pfd` (float): Minimum channel PFD
    - `.max_pfd` (float): Maximum channel PFD
    - `.n_channels` (int): Number of channels
    - `.channels` (dict): Per-channel results

**Example:**
```python
from hhf_validation.validations import run_peff_validation

# Seismic only (no MNE required)
result = run_peff_validation(include_eeg=False)
print(f"Seismic PFD: {result.seismic.pfd:.4f}")
print(f"Events: {result.seismic.n_points}")

# Full validation
result = run_peff_validation()
if result.eeg:
    print(f"EEG mean PFD: {result.eeg.mean_pfd:.4f}")
```

---

## hhf_validation.utils.constants

CODATA 2018 physical constants and HHF-derived values.

### CODATA 2018 Constants

| Constant | Symbol | Value | Units |
|----------|--------|-------|-------|
| Planck constant | `h` | 6.62607015e-34 | J·s |
| Reduced Planck | `hbar` | 1.054571817e-34 | J·s |
| Speed of light | `c` | 2.99792458e8 | m/s |
| Proton mass | `m_p` | 1.67262192e-27 | kg |
| Electron mass | `m_e` | 9.10938370e-31 | kg |
| Fine-structure | `alpha` | 7.2973526e-3 | — |
| Planck length | `L_P` | 1.616255e-35 | m |
| Elementary charge | `e` | 1.602176634e-19 | C |
| Boltzmann | `k_B` | 1.380649e-23 | J/K |
| Avogadro | `N_A` | 6.02214076e23 | mol⁻¹ |
| Atomic mass unit | `u` | 1.66053907e-27 | kg |
| Rydberg (∞) | `R_INF` | 10973731.568 | m⁻¹ |

### HHF Derived Constants

| Constant | Symbol | Value | Formula |
|----------|--------|-------|---------|
| HHF Radius | `R_HHF` | 1.81e-13 m | h / (m_p × c × α) |
| Scaling Ratio | `LAMBDA_HH` | 1.12e22 | R_HHF / L_P |
| Area Scaling | `S_H` | 1.26e44 | (Λᴴᴴ)² |
| Volume Scaling | `V_H` | 1.41e66 | (Λᴴᴴ)³ |

### Helper Functions

```python
def compute_hhf_radius() -> float
    """Compute R_HHF from fundamental constants."""

def compute_hhf_scaling_ratio() -> float
    """Compute Λᴴᴴ = R_HHF / L_P."""

def compute_reduced_mass(mass_1: float, mass_2: float) -> float
    """Compute reduced mass for diatomic molecule (in u)."""

def compute_effective_frequency(omega_e: float, omega_exe: float) -> float
    """Compute anharmonic-corrected frequency: ω_e - 2·ω_exe."""
```

### Isotopologue Data

```python
ISOTOPOLOGUE_DATA = {
    'H2': {'omega_e': 4401.21, 'omega_exe': 121.33, ...},
    'D2': {'omega_e': 3115.50, 'omega_exe': 60.28, ...},
    'T2': {'omega_e': 2546.47, 'omega_exe': 41.735, ...},
    'HD': {'omega_e': 3813.15, 'omega_exe': 91.65, ...}
}
```

---

## hhf_validation.utils.plotting

Publication-quality visualization utilities.

### configure_scientific_style

```python
def configure_scientific_style() -> None
```

Configure matplotlib for scientific publication style:
- Serif fonts
- Grid lines
- High DPI (300 for saved figures)
- Consistent color palette

### COLORS

Color palette dictionary:

```python
COLORS = {
    'primary': '#1f77b4',    # Blue
    'secondary': '#ff7f0e',  # Orange
    'accent': '#d62728',     # Red
    'success': '#2ca02c',    # Green
    'info': '#9467bd',       # Purple
    'gray': '#7f7f7f'
}
```

### Helper Functions

```python
def add_panel_labels(axes, fontweight='bold', fontsize=12)
    """Add (a), (b), (c) labels to subplot axes."""

def add_statistics_annotation(ax, text, loc='upper right')
    """Add boxed statistics annotation to plot."""
```

---

## Result Dataclasses

All validation functions return typed dataclass results with `.to_dict()` and `.to_json()` methods:

```python
# Example: Serialize results
result = run_isotopologue_validation()
json_str = result.to_json()
dict_obj = result.to_dict()
```

