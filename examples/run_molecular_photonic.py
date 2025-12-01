#!/usr/bin/env python3
"""
Run Molecular/Photonic Fractal Entanglement Validation
=======================================================
Validate HHF theoretical constants and spectral consistency.

Usage:
    python run_molecular_photonic.py [--output-dir PATH] [--show-plots]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhf_validation.validations.molecular_photonic import run_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Run molecular/photonic HHF validation"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs/molecular_photonic'),
        help='Output directory'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("HHF Molecular/Photonic Fractal Entanglement Validation")
    print("=" * 70)
    
    result = run_validation(
        output_dir=args.output_dir,
        show_plots=args.show_plots
    )
    
    return 0 if result.spectral.validated else 1


if __name__ == '__main__':
    sys.exit(main())

