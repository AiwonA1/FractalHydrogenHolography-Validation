# HHF Validation Package — Agent Guidelines

## Repository Structure

```
fhh/
├── src/hhf_validation/      # Main package source
│   ├── core/                # Shared fractal analysis utilities
│   ├── validations/         # Validation experiment modules
│   └── utils/               # Physical constants and utilities
├── tests/                   # Unit and integration tests
├── examples/                # Runnable validation scripts
├── outputs/                 # Generated outputs (gitignored)
└── docs/                    # Documentation and reports
```

## Key Entry Points

Using `uv` (recommended):
```bash
uv venv --python 3.12
uv pip install -e ".[all]"
uv run python examples/run_all.py
uv run pytest tests/
```

Alternative (pip):
- **Run all validations:** `python examples/run_all.py`
- **Run tests:** `pytest tests/`
- **Install package:** `pip install -e .`

## Development Guidelines

1. **TDD Approach:** Write tests before implementing new features
2. **Modular Design:** Keep validation modules independent and reusable
3. **Real Data Only:** No mock data in production validation code
4. **Logging:** Use Python logging module for informative output
5. **Type Hints:** Include type annotations for public functions

## Adding New Validations

1. Create module in `src/hhf_validation/validations/`
2. Implement `run_validation()` function with standard interface
3. Add export to `validations/__init__.py`
4. Create example runner in `examples/`
5. Add tests in `tests/`
6. Document in `docs/reports/`

## Physical Constants

All CODATA 2018 constants are centralized in `src/hhf_validation/utils/constants.py`.
Do not hardcode physical constants elsewhere.

