#!/usr/bin/env python3
"""
Run Isotopologue Scaling Validation
====================================
Validate mass-invariant HHF scaling across hydrogen isotopes (H₂, D₂, T₂).

Usage:
    python run_isotopologue.py [--output-dir PATH] [--show-plots] [--threshold FLOAT]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhf_validation.validations.isotopologue import run_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Run isotopologue scaling HHF validation"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs/isotopologue'),
        help='Output directory'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Maximum deviation threshold for PASS (default: 0.1)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("HHF Isotopologue Scaling Invariance Validation")
    print("Isotopes: H₂, D₂, T₂")
    print("=" * 60)
    
    result = run_validation(
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        threshold=args.threshold
    )
    
    return 0 if result.passed else 1


if __name__ == '__main__':
    sys.exit(main())

