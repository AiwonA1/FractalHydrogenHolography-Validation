# Outputs Directory

This directory contains generated outputs from HHF validation experiments.

## Structure

```
outputs/
├── biological/
│   ├── plots/      # Visualization outputs
│   └── logs/       # JSON and CSV logs
├── isotopologue/
│   ├── plots/
│   └── logs/
├── molecular_photonic/
│   ├── plots/
│   └── logs/
└── peff/
    ├── plots/
    └── logs/
```

## Generating Outputs

Run the example scripts to populate these directories:

```bash
# Run all validations
python examples/run_all.py

# Or run individual validations
python examples/run_biological.py
python examples/run_isotopologue.py
python examples/run_molecular_photonic.py
python examples/run_peff.py
```

## Note

This directory is excluded from version control (via `.gitignore`).
Regenerate outputs by running the validation scripts.

