# Documentation Directory

## Contents

| Document | Description |
|----------|-------------|
| [theory.md](theory.md) | HHF theoretical foundation and derivations |
| [data_sources.md](data_sources.md) | External data provenance and access methods |
| [api_reference.md](api_reference.md) | Public API documentation |
| [outputs.md](outputs.md) | Output file structure and schemas |
| `reports/` | Validation white papers and technical reports |

---

## Theory and Background

- **[theory.md](theory.md)** — Mathematical basis of the HHF framework
  - HHF radius derivation: R_HHF = h / (m_p × c × α)
  - Scaling hierarchy: Planck → HHF → molecular
  - Fractal dimension analysis (PFD, HFD)
  - Physical interpretation of Λᴴᴴ ≈ 1.12 × 10²²

---

## Data Sources

- **[data_sources.md](data_sources.md)** — Real data provenance
  - USGS Earthquake Catalog (seismic)
  - PhysioNet EEG Motor Movement Dataset
  - Melbourne Daily Temperatures (biological proxy)
  - NIST Chemistry WebBook (spectroscopic constants)

---

## API Reference

- **[api_reference.md](api_reference.md)** — Public API documentation
  - Core fractal functions (`petrosian_fd`, `higuchi_fd`)
  - Validation runners (`run_*_validation`)
  - Physical constants (CODATA 2018 + HHF derived)
  - Result dataclasses and serialization

---

## Output Files

- **[outputs.md](outputs.md)** — Generated output documentation
  - Directory structure (`outputs/{validation}/plots|logs`)
  - JSON log schemas
  - CSV summary formats
  - Plot specifications (300 DPI, scientific style)

---

## Validation Reports

| Report | Description |
|--------|-------------|
| [biological_report.md](reports/biological_report.md) | Biological proxy validation using environmental time series |
| [isotopologue_report.md](reports/isotopologue_report.md) | Mass-invariant scaling validation across H₂, D₂, T₂ |
| [molecular_photonic_report.md](reports/molecular_photonic_report.md) | HHF constants and spectral consistency validation |
| [peff_report.md](reports/peff_report.md) | PEFF hypothesis validation with seismic/EEG data |

---

## Report Structure

All reports follow a standard academic structure:

1. **Abstract** — Summary of methods and findings
2. **Introduction** — Background and motivation
3. **Methods** — Data sources, calculations, validation criteria
4. **Results** — Quantitative findings with tables and figures
5. **Discussion** — Interpretation and implications
6. **Conclusion** — Summary and next steps
7. **Output Files** — Links to generated data

---

## Adding New Documentation

### New Theory Documents

Add to `docs/` with clear section headings:
- Use LaTeX-style math notation where applicable
- Include derivation steps
- Reference CODATA values

### New Validation Reports

Add to `docs/reports/` following the standard structure:
1. Update corresponding validation module
2. Run validation to generate outputs
3. Write report referencing actual output values
4. Update this index

### Formatting Guidelines

- Use Markdown with consistent heading levels
- Include relative links to output files
- Add tables for quantitative data
- Reference code paths using backticks
