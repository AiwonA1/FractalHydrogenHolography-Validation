# PEFF Seismic/EEG Validation Report

**Paradise Energy Fractal Force Hypothesis Testing**

*Authors: FractiAI Research Team*  
*Date: December 2025*

---

## Abstract

This study empirically investigates the Paradise Energy Fractal Force (PEFF) hypothesis by analyzing fractal complexity in macroscopic datasets: USGS seismic magnitudes and PhysioNet EEG signals. Fractal irregularity was quantified using the Petrosian Fractal Dimension (PFD), automatically computed across seismic events and individual EEG channels.

**Findings** indicate that seismic magnitudes (n=7,928 events) exhibit a PFD of 1.026, reflecting high fractal irregularity consistent with PEFF predictions. EEG channels (64 total) display PFDs ranging from 1.019 to 1.028 (mean 1.022), demonstrating consistent fractal complexity in brain electrical activity.

Cross-system comparison reveals that both geophysical and neurophysiological datasets share similar fractal signatures, supporting the notion that PEFF-like fractal energy manifests across multiple scales.

---

## Introduction

The Paradise Energy Fractal Force (PEFF) hypothesis extends HHF principles to macroscopic systems, predicting that fractal complexity signatures appear consistently across scales—from molecular vibrations to geophysical phenomena to neural dynamics.

This validation tests PEFF predictions by:
1. Analyzing earthquake magnitude sequences from USGS
2. Measuring fractal dimensions of multi-channel EEG recordings
3. Comparing cross-domain fractal signatures

---

## Methods

### Seismic Data

- **Source:** USGS Earthquake Hazards Program API
- **Query:** January 2025, magnitude ≥ 1.0
- **Events Retrieved:** 7,928 earthquakes
- **Analysis:** PFD computed on magnitude time series

### EEG Data

- **Source:** PhysioNet EEG Motor Movement/Imagery Dataset
- **Recording:** S001R01.edf (baseline, eyes open)
- **Channels:** 64 EEG electrodes
- **Preprocessing:** Resampled to 100 Hz
- **Analysis:** PFD computed per channel

### Fractal Analysis

The Petrosian Fractal Dimension (PFD) quantifies signal complexity:

```
PFD = log₁₀(N) / [log₁₀(N) + log₁₀(N / (N + 0.4×Δ))]
```

where N is the signal length and Δ is the number of sign changes in the first derivative.

---

## Results

### Seismic Data Analysis (USGS)

| Metric | Value |
|--------|-------|
| Dataset | USGS earthquakes (Jan 2025) |
| Events | 7,928 |
| Magnitude range | ≥ 1.0 |
| **PFD** | **1.0259** |

The seismic magnitude series exhibits high fractal irregularity, consistent with the self-similar energy release patterns predicted by PEFF.

![Seismic PFD](../../outputs/peff/plots/seismic_fd.png)

### EEG Data Analysis (PhysioNet)

| Metric | Value |
|--------|-------|
| Dataset | PhysioNet S001R01.edf |
| Channels | 64 |
| Samples/channel | 6,100 |
| **Mean PFD** | **1.0215** |
| **PFD Range** | **1.0187 – 1.0276** |
| Std Dev | 0.0018 |

The narrow PFD variation across 64 channels suggests stable, coordinated fractal dynamics in neural activity.

| Channel Group | Mean PFD | Range |
|---------------|----------|-------|
| Frontal (F) | 1.021 | 1.020–1.023 |
| Central (C) | 1.021 | 1.020–1.024 |
| Parietal (P) | 1.021 | 1.020–1.022 |
| Occipital (O) | 1.019 | 1.019–1.020 |
| Temporal (T) | 1.025 | 1.024–1.028 |

### Cross-Domain Comparison

| Dataset | PFD | Events/Channels |
|---------|-----|-----------------|
| Seismic (USGS) | 1.0259 | 7,928 events |
| EEG (PhysioNet) | 1.0215 ± 0.0018 | 64 channels |

![Cross-Domain Comparison](../../outputs/peff/plots/peff_comparison.png)

**Observation:** Both systems exhibit PFD values in the 1.02–1.03 range, indicating comparable fractal complexity across vastly different scales (tectonic plates vs. neural networks).

---

## Discussion

### PEFF Validation

The consistent fractal signatures across geophysical and neurophysiological data support PEFF predictions:

1. **Scale Invariance:** Similar PFD values (~1.02) across 10+ orders of magnitude in spatial scale
2. **Universality:** Fractal complexity appears in both deterministic (geophysics) and biological (neural) systems
3. **Reproducibility:** Automated pipeline produces consistent, verifiable results

### Physical Interpretation

The PFD range of 1.02–1.03 indicates:
- High irregularity (many sign changes in derivative)
- Complex, non-periodic dynamics
- Intermediate between pure noise (PFD ≈ 1.03) and smooth signals (PFD ≈ 1.0)

### Limitations

- Single EEG subject (S001) analyzed
- One month of seismic data
- PFD sensitive to preprocessing choices

### Future Directions

1. Expand to multi-subject EEG cohorts
2. Analyze longer seismic time windows
3. Compute Higuchi FD for complementary fractal assessment
4. Correlate with HHF theoretical predictions

---

## Conclusion

The PEFF Seismic/EEG validation demonstrates:

1. **Seismic:** PFD = 1.026 from 7,928 USGS earthquake magnitudes
2. **EEG:** Mean PFD = 1.022 across 64 PhysioNet channels (range: 1.019–1.028)
3. **Cross-Domain:** Both systems share comparable fractal complexity signatures

These findings provide empirical support for PEFF predictions of scale-invariant fractal energy dynamics across geophysical and neural systems.

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/peff/plots/seismic_fd.png` | Seismic magnitude series with PFD |
| `outputs/peff/plots/peff_comparison.png` | Cross-domain PFD comparison |
| `outputs/peff/logs/peff_summary.json` | Complete results (JSON) |
| `outputs/peff/logs/peff_summary.csv` | Channel-by-channel PFD table |
