# HHF Validation Suite

Empirical validation framework for the Hydrogen-Holographic Fractal Whole Brain AI (HHF-WB) hypothesis.

## Overview

This package provides automated validation pipelines testing HHF predictions across four independent domains:

| Validation | Domain | Key Metric |
|------------|--------|------------|
| Biological Proxy | Environmental time series | PFD: 1.024, HFD: 0.871 |
| Isotopologue Scaling | Hydrogen isotopes (H₂, D₂, T₂) | Λᴴᴴ deviation: <2.4% |
| Molecular/Photonic | HHF constants vs CODATA | Spectral consistency: PASS |
| PEFF Seismic/EEG | Geophysical + neural | Cross-domain PFD: ~1.02 |

## Installation

### Using uv (Recommended)

```bash
# Create virtual environment and install all dependencies
uv venv --python 3.12
uv pip install -e ".[all]"

# Run commands via uv
uv run pytest tests/ -v
uv run python examples/run_all.py
```

### Using pip

```bash
# Install from source
pip install -e .

# With EEG analysis support
pip install -e ".[eeg]"

# With development tools
pip install -e ".[dev]"
```

## Quick Start

```python
from hhf_validation.validations import (
    run_biological_validation,
    run_isotopologue_validation,
    run_molecular_photonic_validation,
    run_peff_validation
)

# Run individual validations
result = run_biological_validation(output_dir="outputs/biological")
result = run_isotopologue_validation(output_dir="outputs/isotopologue")
result = run_molecular_photonic_validation(output_dir="outputs/molecular_photonic")
result = run_peff_validation(output_dir="outputs/peff")
```

Or use the command-line examples:

```bash
# Run all validations
python examples/run_all.py

# Run individual validations
python examples/run_biological.py
python examples/run_isotopologue.py
python examples/run_molecular_photonic.py
python examples/run_peff.py --seismic-only  # Skip EEG if MNE not installed
```

## Package Structure

```
fhh/
├── src/hhf_validation/
│   ├── core/               # Fractal dimension algorithms (PFD, HFD)
│   ├── validations/        # Validation experiment modules
│   └── utils/              # Physical constants (CODATA 2018)
├── tests/                  # Unit tests
├── examples/               # Runnable scripts
├── outputs/                # Generated results (gitignored)
└── docs/reports/           # Validation white papers
```

## Validations

### 1. Biological Proxy

Fractal analysis of Daily Minimum Temperatures (Melbourne, 1981-1990).

- **Prediction:** Macroscale environmental proxies reflect HHF fractal dynamics
- **Result:** PFD = 1.024, HFD = 0.871

### 2. Isotopologue Scaling Invariance

Mass-normalized scaling across hydrogen isotopes.

- **Prediction:** Λᴴᴴ remains constant when corrected for reduced mass
- **Result:** All isotopes within 2.4% of unity (PASS)

### 3. Molecular/Photonic Entanglement

HHF theoretical constants validated against spectroscopy.

- **Prediction:** Calculated Rydberg matches CODATA to high precision
- **Result:** Relative error < 10⁻⁶ (VALIDATED)

### 4. PEFF Seismic/EEG

Cross-domain fractal signatures in geophysical and neural systems.

- **Prediction:** Similar fractal complexity across scales
- **Result:** Seismic PFD ≈ 1.026, EEG mean PFD ≈ 1.020

## Core Functions

```python
from hhf_validation.core import petrosian_fd, higuchi_fd

# Petrosian Fractal Dimension
pfd = petrosian_fd(signal)

# Higuchi Fractal Dimension
hfd = higuchi_fd(signal, k_max=10)
```

## Physical Constants

All CODATA 2018 values are available:

```python
from hhf_validation.utils import L_P, h, c, m_p, m_e, alpha, R_HHF, LAMBDA_HH
```

## Testing

```bash
# Using uv
uv run pytest tests/ -v

# Or with activated venv
pytest tests/ -v
```

## Requirements

- Python ≥ 3.9
- NumPy, Pandas, Matplotlib, SciPy, Requests
- MNE (optional, for EEG analysis)

## License

MIT

## Citation

If using this validation framework in research, please cite the HHF-WB AI framework paper.
