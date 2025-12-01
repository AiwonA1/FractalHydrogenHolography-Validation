#!/usr/bin/env python3
"""
Run Biological Proxy Validation
===============================
Analyze fractal properties of biological/environmental time series.

Dataset: Daily Minimum Temperatures — Melbourne (1981-1990)

Usage:
    python run_biological.py [--output-dir PATH] [--show-plots]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhf_validation.validations.biological_proxy import run_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Run biological proxy HHF validation"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs/biological'),
        help='Output directory'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HHF Biological Proxy Validation")
    print("Dataset: Daily Minimum Temperatures (Melbourne)")
    print("=" * 60)
    
    result = run_validation(
        output_dir=args.output_dir,
        show_plots=args.show_plots
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

