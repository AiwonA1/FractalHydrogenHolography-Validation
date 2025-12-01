#!/usr/bin/env python3
"""
Run All HHF Validations
=======================
Execute all four HHF validation pipelines and generate consolidated outputs.

Usage:
    python run_all.py [--output-dir PATH] [--show-plots]
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hhf_validation import __version__
from hhf_validation.validations import (
    run_biological_validation,
    run_isotopologue_validation,
    run_molecular_photonic_validation,
    run_peff_validation
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run all HHF validation experiments"
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path('outputs'),
        help='Base output directory (default: outputs)'
    )
    parser.add_argument(
        '--show-plots',
        action='store_true',
        help='Display plots interactively'
    )
    parser.add_argument(
        '--skip-peff',
        action='store_true',
        help='Skip PEFF validation (requires network access)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose (DEBUG) logging'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress INFO logging (show only warnings/errors)'
    )
    
    args = parser.parse_args()
    output_base = args.output_dir
    
    # Configure log level based on verbosity flags
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    print("=" * 70)
    print(f"HHF-WB AI Framework — Comprehensive Empirical Validation (v{__version__})")
    print("=" * 70)
    
    results = {}
    timings = {}
    start_total = time.perf_counter()
    
    # 1. Biological Proxy Validation
    print("\n[1/4] Biological Proxy Validation...")
    t0 = time.perf_counter()
    try:
        results['biological'] = run_biological_validation(
            output_dir=output_base / 'biological',
            show_plots=args.show_plots
        )
        duration = time.perf_counter() - t0
        timings['biological'] = duration
        print(f"      ✓ Complete ({duration:.1f}s)")
    except Exception as e:
        duration = time.perf_counter() - t0
        timings['biological'] = duration
        logger.error(f"Biological validation failed: {e}")
        print(f"      ✗ Failed: {e} ({duration:.1f}s)")
    
    # 2. Isotopologue Scaling Validation
    print("\n[2/4] Isotopologue Scaling Validation...")
    t0 = time.perf_counter()
    try:
        results['isotopologue'] = run_isotopologue_validation(
            output_dir=output_base / 'isotopologue',
            show_plots=args.show_plots
        )
        duration = time.perf_counter() - t0
        timings['isotopologue'] = duration
        print(f"      ✓ Complete ({duration:.1f}s)")
    except Exception as e:
        duration = time.perf_counter() - t0
        timings['isotopologue'] = duration
        logger.error(f"Isotopologue validation failed: {e}")
        print(f"      ✗ Failed: {e} ({duration:.1f}s)")
    
    # 3. Molecular/Photonic Validation
    print("\n[3/4] Molecular/Photonic Validation...")
    t0 = time.perf_counter()
    try:
        results['molecular_photonic'] = run_molecular_photonic_validation(
            output_dir=output_base / 'molecular_photonic',
            show_plots=args.show_plots
        )
        duration = time.perf_counter() - t0
        timings['molecular_photonic'] = duration
        print(f"      ✓ Complete ({duration:.1f}s)")
    except Exception as e:
        duration = time.perf_counter() - t0
        timings['molecular_photonic'] = duration
        logger.error(f"Molecular/photonic validation failed: {e}")
        print(f"      ✗ Failed: {e} ({duration:.1f}s)")
    
    # 4. PEFF Seismic/EEG Validation
    if not args.skip_peff:
        print("\n[4/4] PEFF Seismic/EEG Validation...")
        t0 = time.perf_counter()
        try:
            results['peff'] = run_peff_validation(
                output_dir=output_base / 'peff',
                show_plots=args.show_plots
            )
            duration = time.perf_counter() - t0
            timings['peff'] = duration
            print(f"      ✓ Complete ({duration:.1f}s)")
        except Exception as e:
            duration = time.perf_counter() - t0
            timings['peff'] = duration
            logger.error(f"PEFF validation failed: {e}")
            print(f"      ✗ Failed: {e} ({duration:.1f}s)")
    else:
        print("\n[4/4] PEFF Validation: Skipped")
    
    total_duration = time.perf_counter() - start_total
    
    # Summary
    print("\n" + "=" * 70)
    print("Validation Summary")
    print("=" * 70)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        duration_str = f" ({timings.get(name, 0):.1f}s)" if name in timings else ""
        print(f"  {name}: {status}{duration_str}")
    
    print(f"\nAll outputs saved to: {output_base.absolute()}")
    print(f"Total execution time: {total_duration:.1f}s")
    print("=" * 70)
    
    return 0 if all(results.values()) else 1


if __name__ == '__main__':
    sys.exit(main())

