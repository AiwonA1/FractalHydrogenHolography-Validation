#!/usr/bin/env python3
"""
Run PEFF Seismic/EEG Validation
================================
Validate PEFF hypothesis through fractal analysis of seismic and EEG data.

Requires network access to download data from:
- USGS earthquake API
- PhysioNet EEG database

Usage:
    python run_peff.py [--output-dir PATH] [--show-plots] [--seismic-only] [--eeg-only]
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhf_validation.validations.peff_seismic_eeg import run_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def main():
    parser = argparse.ArgumentParser(
        description="Run PEFF seismic/EEG HHF validation"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs/peff'),
        help='Output directory'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    parser.add_argument(
        '--seismic-only',
        action='store_true',
        help='Only run seismic analysis (skip EEG)'
    )
    parser.add_argument(
        '--eeg-only',
        action='store_true',
        help='Only run EEG analysis (skip seismic)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default='2025-01-01',
        help='Seismic data start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default='2025-01-31',
        help='Seismic data end date (YYYY-MM-DD)'
    )
    
    args = parser.parse_args()
    
    include_seismic = not args.eeg_only
    include_eeg = not args.seismic_only
    
    print("=" * 60)
    print("PEFF Seismic/EEG Fractal Validation")
    print("=" * 60)
    
    if not include_seismic and not include_eeg:
        print("Error: Cannot skip both seismic and EEG analysis")
        return 1
    
    result = run_validation(
        output_dir=args.output_dir,
        show_plots=args.show_plots,
        include_seismic=include_seismic,
        include_eeg=include_eeg,
        seismic_start=args.start_date,
        seismic_end=args.end_date
    )
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

