# Output Files Reference

**Structure and Schema of Generated Outputs**

---

## Directory Structure

```
outputs/
├── biological/
│   ├── logs/
│   │   ├── fractal_analysis_log.json
│   │   └── fractal_analysis_summary.csv
│   └── plots/
│       ├── daily_min_temperatures.png
│       ├── temperature_distribution.png
│       └── autocorrelation_analysis.png
│
├── isotopologue/
│   ├── logs/
│   │   ├── isotopologue_scaling_log.json
│   │   └── isotopologue_scaling_summary.csv
│   └── plots/
│       ├── isotopologue_scaling.png
│       ├── force_constants.png
│       └── mass_frequency_regression.png
│
├── molecular_photonic/
│   ├── logs/
│   │   ├── molecular_photonic_log.json
│   │   └── molecular_photonic_summary.csv
│   └── plots/
│       ├── molecular_photonic_scaling.png
│       └── scaling_hierarchy.png
│
└── peff/
    ├── logs/
    │   ├── peff_summary.json
    │   └── peff_summary.csv
    └── plots/
        ├── seismic_fd.png
        ├── seismic_histogram.png
        ├── eeg_distribution.png
        ├── eeg_topography.png
        └── peff_comparison.png
```

---

## Biological Validation Outputs

### fractal_analysis_log.json

Complete analysis results in JSON format.

```json
{
  "Daily_Minimum_Temperatures_Melbourne": {
    "dataset_name": "Daily_Minimum_Temperatures_Melbourne",
    "n_points": 3650,
    "pfd": 1.024332464786391,
    "hfd": 0.8710348624219364
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `dataset_name` | string | Name of the analyzed dataset |
| `n_points` | int | Number of data points analyzed |
| `pfd` | float | Petrosian Fractal Dimension |
| `hfd` | float | Higuchi Fractal Dimension |

### fractal_analysis_summary.csv

Tabular summary for spreadsheet import.

```csv
dataset_name,n_points,pfd,hfd
Daily_Minimum_Temperatures_Melbourne,3650,1.024332,0.871035
```

### daily_min_temperatures.png

Time series visualization with fractal metrics annotation.

**Specifications:**
- Resolution: 300 DPI
- Size: 7" × 2.8" (single column width)
- Style: Scientific with grid lines
- Annotations: n, PFD, HFD values

---

## Isotopologue Validation Outputs

### isotopologue_scaling_log.json

Complete isotopologue scaling analysis.

```json
{
  "isotopologues": {
    "H2": {
      "isotope": "H2",
      "omega_eff": 4158.55,
      "mu": 0.503912516035,
      "k_eff": 513.4394570808789,
      "r_eff": 0.04413217988729939,
      "lambda_hh": 1.0
    },
    "D2": {
      "isotope": "D2",
      "omega_eff": 2994.94,
      "mu": 1.007050888925,
      "k_eff": 532.203724683022,
      "r_eff": 0.043347199557762915,
      "lambda_hh": 0.9822129717693284
    },
    "T2": {
      "isotope": "T2",
      "omega_eff": 2463.0,
      "mu": 1.5080246388499998,
      "k_eff": 538.998431058503,
      "r_eff": 0.0430731119148585,
      "lambda_hh": 0.9760023643711814
    }
  },
  "max_deviation": 0.023997635628818625,
  "threshold": 0.1,
  "status": "PASS"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `isotopologues` | object | Per-isotope analysis results |
| `isotopologues.*.omega_eff` | float | Effective frequency (cm⁻¹) |
| `isotopologues.*.mu` | float | Reduced mass (u) |
| `isotopologues.*.k_eff` | float | Force constant (N/m) |
| `isotopologues.*.r_eff` | float | Fractal radius proxy |
| `isotopologues.*.lambda_hh` | float | Normalized scaling ratio |
| `max_deviation` | float | Maximum |Λᴴᴴ - 1| |
| `threshold` | float | Pass/fail threshold |
| `status` | string | "PASS" or "FAIL" |

### isotopologue_scaling_summary.csv

```csv
isotope,omega_eff_cm-1,mu,k_eff,r_eff,Lambda_HH
H2,4158.55,0.503913,513.439,0.044132,1.000000
D2,2994.94,1.007051,532.204,0.043347,0.982213
T2,2463.00,1.508025,538.998,0.043073,0.976002
```

### isotopologue_scaling.png

Dual-panel visualization.

**Panel (a):** Λᴴᴴ bar chart with unity reference and threshold band
**Panel (b):** ω_eff vs reduced mass scatter plot

**Specifications:**
- Resolution: 300 DPI
- Size: 7" × 3" (dual panel)
- Status badge: PASS/FAIL indicator

---

## Molecular/Photonic Validation Outputs

### molecular_photonic_log.json

Theoretical and spectral validation results.

```json
{
  "theoretical": {
    "R_HHF": 1.8108071973231944e-13,
    "R_ratio": 1.1203722168365725e+22,
    "S_H": 1.2552339042592958e+44,
    "V_H": 1.4063291919634134e+66,
    "Lambda_HH": 1.1203722168365725e+22
  },
  "spectral": {
    "R_inf_calculated": 10973731.568138644,
    "R_H_calculated": 10967758.340259008,
    "R_H_official": 10967758.34,
    "relative_error": 2.3615423548907396e-11,
    "validated": true
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `theoretical.R_HHF` | float | HHF radius (m) |
| `theoretical.Lambda_HH` | float | Scaling ratio |
| `theoretical.S_H` | float | Area scaling factor |
| `theoretical.V_H` | float | Volume scaling factor |
| `spectral.R_H_calculated` | float | Computed Rydberg constant (m⁻¹) |
| `spectral.R_H_official` | float | CODATA reference value |
| `spectral.relative_error` | float | Relative error |
| `spectral.validated` | bool | Whether error < threshold |

### molecular_photonic_summary.csv

```csv
quantity,value,units
R_HHF,1.8108e-13,m
Lambda_HH,1.1204e+22,dimensionless
S_H,1.2552e+44,dimensionless
V_H,1.4063e+66,dimensionless
Rydberg_error,2.36e-11,relative
```

### molecular_photonic_scaling.png

Dual-panel visualization.

**Panel (a):** HHF scaling exponents bar chart
**Panel (b):** Rydberg constant comparison

**Specifications:**
- Resolution: 300 DPI
- Size: 7" × 3" (dual panel)
- Scientific notation for large values

---

## PEFF Validation Outputs

### peff_summary.json

Complete seismic and EEG analysis.

```json
{
  "seismic": {
    "n_points": 7928,
    "pfd": 1.025855841443894,
    "date_range": "2025-01-01 to 2025-01-31",
    "min_magnitude": 1.0
  },
  "eeg": {
    "mean_pfd": 1.0215329979370593,
    "min_pfd": 1.018713265072227,
    "max_pfd": 1.027571478628715,
    "n_channels": 64,
    "channels": {
      "Fc5.": {
        "channel": "Fc5.",
        "pfd": 1.0221021356163598,
        "n_samples": 6100
      }
      // ... 63 more channels
    }
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `seismic.n_points` | int | Number of seismic events |
| `seismic.pfd` | float | Seismic magnitude PFD |
| `seismic.date_range` | string | Query date range |
| `eeg.mean_pfd` | float | Mean PFD across channels |
| `eeg.min_pfd` | float | Minimum channel PFD |
| `eeg.max_pfd` | float | Maximum channel PFD |
| `eeg.n_channels` | int | Number of EEG channels |
| `eeg.channels.*.pfd` | float | Per-channel PFD |
| `eeg.channels.*.n_samples` | int | Samples per channel |

### peff_summary.csv

Flat table of all PFD values.

```csv
dataset,PFD
seismic,1.025856
Fc5.,1.022102
Fc3.,1.020663
...
```

### seismic_fd.png

Seismic magnitude time series.

**Specifications:**
- Resolution: 300 DPI
- Size: 7" × 2.8"
- Shows: Individual events + binned average
- Annotations: n, PFD

### peff_comparison.png

Cross-domain PFD comparison bar chart.

**Specifications:**
- Resolution: 300 DPI
- Size: 4.5" × 3.5"
- Shows: Seismic vs EEG mean PFD
- Error bars: EEG min/max range

---

## Plot Specifications

All plots follow scientific publication standards:

| Property | Value |
|----------|-------|
| DPI (saved) | 300 |
| DPI (display) | 150 |
| Font family | DejaVu Sans / Serif |
| Title weight | Bold |
| Grid | Dashed, 0.5pt |
| Edge color | Light gray |
| Background | White |
| Color palette | Colorblind-friendly |

### Standard Sizes

| Type | Dimensions | Use Case |
|------|------------|----------|
| Single panel | 7" × 2.8" | Time series |
| Dual panel | 7" × 3" | Comparison plots |
| Square | 4.5" × 3.5" | Bar charts |

---

## Regenerating Outputs

To regenerate all outputs:

```bash
# All validations
python examples/run_all.py

# Individual validations
python examples/run_biological.py
python examples/run_isotopologue.py
python examples/run_molecular_photonic.py
python examples/run_peff.py
```

**Note:** PEFF validation requires network access (USGS API, PhysioNet).

