# Isotopologue Scaling Invariance Report

**HHF Framework Validation Across Hydrogen Isotopes**

*Authors: FractiAI Research Team*  
*Date: December 2025*

---

## Abstract

We present an automated, empirical validation of the Hydrogen-Holographic Fractal (HHF) scaling hypothesis across hydrogen isotopologues (H₂, D₂, T₂). The central prediction of the HHF framework is that the fractal scaling ratio, Λᴴᴴ, remains effectively constant across isotopic systems when derived from force-constant-based radii.

Using spectroscopic vibrational constants from NIST and published literature, we computed effective vibrational frequencies (ω_eff), force constants (k_eff), and fractal radius proxies (r_eff). Our analysis yielded Λᴴᴴ values within ±2.4% of unity, passing the pre-defined consistency threshold of 0.1.

These results demonstrate mass-invariant fractal scaling in hydrogen isotopologues and provide strong empirical support for HHF internal symmetry.

---

## Introduction

Fractal scaling invariance is a core prediction of the Hydrogen-Holographic Fractal (HHF) framework, which posits that atomic, molecular, and isotopic systems exhibit self-similar dynamics under a universal fractal scaling law. Hydrogen isotopologues provide an ideal testing ground due to well-characterized vibrational spectroscopy data (H₂, D₂, T₂) and their varying reduced masses.

---

## Methods

### Data Sources

Vibrational constants (ω_e, ω_exe) for H₂, D₂, and T₂ were obtained from NIST Chemistry WebBook and standard spectroscopic references:

| Isotope | ω_e (cm⁻¹) | ω_exe (cm⁻¹) | Mass (u) |
|---------|------------|--------------|----------|
| H₂ | 4401.21 | 121.33 | 1.00783 |
| D₂ | 3115.50 | 60.28 | 2.01410 |
| T₂ | 2546.47 | 41.735 | 3.01605 |

### Calculation Procedure

1. **Reduced mass:**
   
   μ = (m₁ × m₂) / (m₁ + m₂)

2. **Effective vibrational frequency (anharmonic correction):**
   
   ω_eff = ω_e − 2·ω_exe

3. **Effective force constant:**
   
   k_eff = (2π × c × ω_eff)² × μ
   
   where c = 2.998 × 10⁸ m/s and μ is converted to kg.

4. **Fractal radius proxy:**
   
   r_eff = 1 / √(k_eff)
   
   This provides a mass-normalized geometric proxy since k_eff already incorporates reduced mass.

5. **HHF scaling ratio:**
   
   Λᴴᴴ = r_eff / r_eff,H₂

### Validation Criterion

**Threshold for consistency:** max |Λᴴᴴ − 1| < 0.1

---

## Results

| Isotope | ω_eff (cm⁻¹) | μ (u)   | k_eff (N/m) | r_eff | Λᴴᴴ   |
|---------|--------------|---------|-------------|-------|-------|
| H₂      | 4158.55      | 0.5039  | 513.44      | 0.04413 | 1.0000 |
| D₂      | 2994.94      | 1.0071  | 532.20      | 0.04335 | 0.9822 |
| T₂      | 2463.00      | 1.5080  | 539.00      | 0.04307 | 0.9760 |

- **Maximum deviation from unity:** 0.024 (2.4%)
- **Consistency threshold:** 0.1 (10%)
- **Result:** **PASS**

### Visualization

![Isotopologue Scaling](../../outputs/isotopologue/plots/isotopologue_scaling.png)

The dual-panel visualization shows:
- **(a)** Λᴴᴴ scaling ratios with unity reference line and ±0.1 threshold band
- **(b)** Effective frequency vs reduced mass relationship

All isotopologues cluster tightly within the acceptance threshold, confirming mass-invariant scaling.

---

## Discussion

### Validation of HHF Prediction

The Λᴴᴴ ratios remain within 2.4% of unity, confirming that effective fractal scaling holds across isotopologues. The force-constant-based approach naturally incorporates mass effects through k_eff = (2πcω)²μ, yielding intrinsically mass-normalized r_eff values.

### Physical Interpretation

The near-constant Λᴴᴴ across isotopes suggests that the underlying holographic fractal structure is invariant under isotopic substitution. The small deviations (< 2.5%) likely arise from:
- Higher-order anharmonic corrections not captured by ω_exe alone
- Born-Oppenheimer approximation limitations
- Relativistic mass corrections for heavier isotopes

### Implications

Mass-invariant fractal scaling indicates that HHF principles are fundamental to:
- Hydrogen bonding dynamics
- Molecular vibrational coherence
- Multi-scale photonic interactions in hydrogen-rich systems

### Next Steps

1. Extend validation to heteronuclear isotopologues (HD, HT, DT)
2. Compare with high-precision spectroscopic measurements
3. Integrate with molecular photonic entanglement studies

---

## Conclusion

The automated HHF isotopologue scaling experiment demonstrates successful mass-invariant fractal scaling across H₂, D₂, and T₂, passing the defined consistency threshold with maximum deviation of 2.4%. This validation supports the internal symmetry of the HHF framework and provides a foundation for cross-domain empirical studies.

---

## Output Files

- `outputs/isotopologue/logs/isotopologue_scaling_log.json` — Complete results
- `outputs/isotopologue/logs/isotopologue_scaling_summary.csv` — Tabular summary
- `outputs/isotopologue/plots/isotopologue_scaling.png` — Publication-quality visualization
