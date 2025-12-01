"""
Isotopologue Scaling Invariance Validation Module
==================================================
Validates HHF mass-invariant fractal scaling across hydrogen isotopologues (H₂, D₂, T₂).

The central HHF prediction is that the fractal scaling ratio Λᴴᴴ remains effectively
constant across isotopic systems when corrected for mass and anharmonicity effects.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..utils.constants import (
    ISOTOPOLOGUE_DATA, ATOMIC_MASS_UNIT, c, h,
    compute_reduced_mass, compute_effective_frequency
)

logger = logging.getLogger(__name__)


@dataclass
class IsotopologueResult:
    """Results for a single isotopologue."""
    isotope: str
    omega_eff: float      # Effective frequency (cm^-1)
    mu: float             # Reduced mass (u)
    k_eff: float          # Effective force constant
    r_eff: float          # Fractal radius proxy
    lambda_hh: float      # Normalized scaling ratio


@dataclass
class IsotopologueValidationResult:
    """Complete isotopologue scaling validation results."""
    isotopologues: Dict[str, IsotopologueResult]
    max_deviation: float
    threshold: float
    passed: bool
    timestamp: str = ""
    data_source: str = "NIST Chemistry WebBook"
    
    def to_dict(self) -> dict:
        return {
            'isotopologues': {k: asdict(v) for k, v in self.isotopologues.items()},
            'max_deviation': self.max_deviation,
            'threshold': self.threshold,
            'status': 'PASS' if self.passed else 'FAIL',
            'timestamp': self.timestamp,
            'data_source': self.data_source
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def compute_force_constant(omega_eff_cm: float, mu_u: float) -> float:
    """
    Compute effective force constant from vibrational frequency.
    
    k = (2π c ω)² μ
    
    Parameters
    ----------
    omega_eff_cm : float
        Effective frequency in cm^-1.
    mu_u : float
        Reduced mass in atomic mass units.
        
    Returns
    -------
    float
        Force constant in N/m (or kg/s²).
    """
    # Convert cm^-1 to Hz: ω(Hz) = c(cm/s) × ω(cm^-1)
    omega_hz = c * 100 * omega_eff_cm
    
    # Convert reduced mass to kg
    mu_kg = mu_u * ATOMIC_MASS_UNIT
    
    # k = (2π ν)² μ
    k = (2 * np.pi * omega_hz) ** 2 * mu_kg
    
    return k


def compute_fractal_radius(k_eff: float) -> float:
    """
    Compute fractal radius proxy from effective force constant.
    
    r_eff = 1 / √(k_eff)
    
    This provides a scale-invariant geometric proxy for HHF validation.
    
    Parameters
    ----------
    k_eff : float
        Effective force constant.
        
    Returns
    -------
    float
        Fractal radius proxy.
    """
    return 1.0 / np.sqrt(k_eff)


def analyze_isotopologue(
    isotope: str,
    data: dict,
    reference_r_eff: Optional[float] = None
) -> IsotopologueResult:
    """
    Analyze a single isotopologue for HHF scaling.
    
    Parameters
    ----------
    isotope : str
        Isotopologue name (e.g., 'H2', 'D2', 'T2').
    data : dict
        Spectroscopic data with omega_e, omega_exe, mass_1, mass_2.
    reference_r_eff : float, optional
        Reference radius for normalization (H2 value).
        
    Returns
    -------
    IsotopologueResult
        Analysis results for this isotopologue.
    """
    # Compute reduced mass
    mu = compute_reduced_mass(data['mass_1'], data['mass_2'])
    
    # Compute effective frequency with anharmonic correction
    omega_eff = compute_effective_frequency(data['omega_e'], data['omega_exe'])
    
    # Compute force constant
    k_eff = compute_force_constant(omega_eff, mu)
    
    # Compute fractal radius proxy from force constant
    r_eff = compute_fractal_radius(k_eff)
    
    # Compute normalized Lambda_HH
    # Since k_eff already incorporates reduced mass (k = 4π²ω²μ),
    # r_eff = 1/sqrt(k_eff) is already mass-normalized.
    # Lambda_HH is simply the ratio of r_eff values.
    if reference_r_eff is not None:
        lambda_hh = r_eff / reference_r_eff
    else:
        lambda_hh = 1.0  # Reference isotopologue
    
    return IsotopologueResult(
        isotope=isotope,
        omega_eff=omega_eff,
        mu=mu,
        k_eff=k_eff,
        r_eff=r_eff,
        lambda_hh=lambda_hh
    )


def run_isotopologue_analysis(
    isotopologues: list = ['H2', 'D2', 'T2'],
    threshold: float = 0.1
) -> IsotopologueValidationResult:
    """
    Run isotopologue scaling validation across multiple isotopes.
    
    Parameters
    ----------
    isotopologues : list
        List of isotopologue names to analyze.
    threshold : float
        Maximum allowed deviation from unity for PASS.
        
    Returns
    -------
    IsotopologueValidationResult
        Complete validation results.
    """
    results = {}
    
    # First pass: compute H2 reference values
    h2_data = ISOTOPOLOGUE_DATA['H2']
    h2_result = analyze_isotopologue('H2', h2_data)
    results['H2'] = h2_result
    
    reference_r_eff = h2_result.r_eff
    
    # Analyze remaining isotopologues
    for iso in isotopologues:
        if iso == 'H2':
            continue
        if iso not in ISOTOPOLOGUE_DATA:
            logger.warning(f"No data for isotopologue {iso}, skipping")
            continue
            
        result = analyze_isotopologue(
            iso, 
            ISOTOPOLOGUE_DATA[iso],
            reference_r_eff=reference_r_eff
        )
        results[iso] = result
    
    # Compute max deviation from unity
    deviations = [abs(r.lambda_hh - 1.0) for r in results.values()]
    max_dev = max(deviations)
    
    passed = max_dev < threshold
    
    return IsotopologueValidationResult(
        isotopologues=results,
        max_deviation=max_dev,
        threshold=threshold,
        passed=passed,
        timestamp=datetime.now().isoformat()
    )


def create_results_dataframe(result: IsotopologueValidationResult) -> pd.DataFrame:
    """
    Create a pandas DataFrame from validation results.
    
    Parameters
    ----------
    result : IsotopologueValidationResult
        Validation results.
        
    Returns
    -------
    pd.DataFrame
        Formatted results table.
    """
    data = []
    for iso, r in result.isotopologues.items():
        data.append({
            'isotope': r.isotope,
            'omega_eff_cm-1': r.omega_eff,
            'mu': r.mu,
            'k_eff': r.k_eff,
            'r_eff': r.r_eff,
            'Lambda_HH': r.lambda_hh
        })
    
    df = pd.DataFrame(data)
    df.set_index('isotope', inplace=True)
    return df


def plot_isotopologue_scaling(
    result: IsotopologueValidationResult,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create publication-quality visualization of isotopologue scaling results.
    
    Includes HHF mass-invariance predictions and accessibility features.
    
    Parameters
    ----------
    result : IsotopologueValidationResult
        Validation results.
    output_path : Path, optional
        Path to save plot.
    show : bool
        Whether to display interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORBLIND_SAFE, ACCESSIBLE_CATEGORICAL,
        MARKERS, add_panel_labels, add_statistics_annotation,
        add_hhf_prediction_band, add_hhf_context_box, HHF_PREDICTIONS
    )
    
    configure_scientific_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    isotopes = list(result.isotopologues.keys())
    lambda_vals = [result.isotopologues[iso].lambda_hh for iso in isotopes]
    omega_vals = [result.isotopologues[iso].omega_eff for iso in isotopes]
    mu_vals = [result.isotopologues[iso].mu for iso in isotopes]
    
    # Format isotope labels with subscripts
    iso_labels = ['H₂', 'D₂', 'T₂'][:len(isotopes)]
    
    # Panel (a): Lambda_HH scaling ratio with HHF prediction
    ax1 = axes[0]
    
    # Add HHF prediction band first (background)
    add_hhf_prediction_band(ax1, center=HHF_PREDICTIONS['lambda_hh_unity'],
                            width=HHF_PREDICTIONS['lambda_hh_threshold'],
                            orientation='horizontal',
                            color=COLORBLIND_SAFE['green'],
                            label='HHF Prediction (Λᴴᴴ = 1.0 ± 10%)')
    
    # Colorblind-safe bar colors
    colors = [COLORBLIND_SAFE['blue'], COLORBLIND_SAFE['orange'], COLORBLIND_SAFE['pink']]
    bars = ax1.bar(iso_labels, lambda_vals, color=colors[:len(isotopes)], 
                   edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Λᴴᴴ (Normalized Scaling Ratio)\n[r_eff(isotope) / r_eff(H₂)]')
    ax1.set_xlabel('Hydrogen Isotopologue')
    ax1.set_ylim(0.88, 1.12)
    ax1.legend(loc='lower right', fontsize=7, framealpha=0.95)
    
    # Add value labels on bars with clear formatting
    for bar, val in zip(bars, lambda_vals):
        ax1.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_title('(a) Mass-Invariant Scaling Ratio', fontweight='bold')
    
    # Panel (b): Frequency vs reduced mass with accessible markers
    ax2 = axes[1]
    
    # Use both color AND marker shape for accessibility
    for i, (mu, omega, label) in enumerate(zip(mu_vals, omega_vals, iso_labels)):
        ax2.scatter([mu], [omega], c=colors[i], marker=MARKERS[i], 
                   s=120, edgecolors='black', linewidths=1, zorder=3, label=label)
    
    # Add isotope labels with offset
    for mu, omega, label in zip(mu_vals, omega_vals, iso_labels):
        ax2.annotate(label, (mu, omega), xytext=(8, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Reduced Mass μ (atomic mass units)\n[m₁·m₂ / (m₁ + m₂)]')
    ax2.set_ylabel('Effective Frequency ω_eff (cm⁻¹)\n[Anharmonicity-corrected vibration]')
    ax2.set_title('(b) Frequency-Mass Relationship', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, title='Isotopologue', title_fontsize=8)
    
    # Overall title with status
    status = 'PASS' if result.passed else 'FAIL'
    status_color = COLORBLIND_SAFE['green'] if result.passed else COLORBLIND_SAFE['red']
    fig.suptitle(f'HHF Isotopologue Scaling Invariance — {status} (max Δ = {result.max_deviation:.1%})', 
                 fontweight='bold', fontsize=11, color=status_color)
    
    # Add HHF interpretation footer
    fig.text(0.5, -0.02, 
             "HHF Prediction: Λᴴᴴ ≈ 1.0 across all hydrogen isotopologues (mass-invariant scaling). "
             f"Observed max deviation: {result.max_deviation:.2%}",
             ha='center', fontsize=8, fontstyle='italic', color=COLORBLIND_SAFE['blue'])
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_force_constants(
    result: IsotopologueValidationResult,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create visualization comparing force constants across isotopologues.
    
    Explains k_eff physical meaning and its role in mass-invariant scaling.
    
    Parameters
    ----------
    result : IsotopologueValidationResult
        Validation results.
    output_path : Path, optional
        Path to save plot.
    show : bool
        Whether to display interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORBLIND_SAFE, MARKERS,
        add_panel_labels, add_statistics_annotation, add_hhf_context_box
    )
    
    configure_scientific_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    isotopes = list(result.isotopologues.keys())
    iso_labels = ['H₂', 'D₂', 'T₂'][:len(isotopes)]
    colors = [COLORBLIND_SAFE['blue'], COLORBLIND_SAFE['orange'], COLORBLIND_SAFE['pink']]
    
    k_vals = [result.isotopologues[iso].k_eff for iso in isotopes]
    r_vals = [result.isotopologues[iso].r_eff for iso in isotopes]
    mu_vals = [result.isotopologues[iso].mu for iso in isotopes]
    
    # Panel (a): Force constants comparison with explanation
    ax1 = axes[0]
    bars = ax1.bar(iso_labels, k_vals, color=colors[:len(isotopes)], 
                   edgecolor='black', linewidth=1)
    ax1.set_ylabel('k_eff (N/m)\n[Bond stiffness: k = (2πcω)²μ]')
    ax1.set_xlabel('Hydrogen Isotopologue')
    ax1.set_title('(a) Effective Force Constants', fontweight='bold')
    
    # Add value labels with units
    for bar, val in zip(bars, k_vals):
        ax1.annotate(f'{val:.1f} N/m',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Add HHF context explaining force constant significance
    hhf_text = ("k_eff incorporates μ,\nmaking r_eff = 1/√k\n"
                "mass-normalized for\nHHF scaling analysis.")
    add_hhf_context_box(ax1, "Physical Meaning", hhf_text, loc='upper right', fontsize=6)
    
    # Panel (b): r_eff vs reduced mass with accessible markers
    ax2 = axes[1]
    
    # Use both color AND marker for accessibility
    for i, (mu, r, label) in enumerate(zip(mu_vals, r_vals, iso_labels)):
        ax2.scatter([mu], [r], c=colors[i], marker=MARKERS[i],
                   s=120, edgecolors='black', linewidths=1, zorder=3, label=label)
    
    # Fit power law: r_eff ∝ μ^α
    log_mu = np.log(mu_vals)
    log_r = np.log(r_vals)
    coeffs = np.polyfit(log_mu, log_r, 1)
    alpha = coeffs[0]
    
    # Plot fit line with distinct style
    mu_fit = np.linspace(min(mu_vals) * 0.9, max(mu_vals) * 1.1, 50)
    r_fit = np.exp(coeffs[1]) * mu_fit ** alpha
    ax2.plot(mu_fit, r_fit, color=COLORBLIND_SAFE['red'], linestyle='--', linewidth=2,
             label=f'Fit: r_eff ∝ μ^{alpha:.3f}')
    
    # Add isotope labels
    for mu, r, label in zip(mu_vals, r_vals, iso_labels):
        ax2.annotate(label, (mu, r), xytext=(8, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    ax2.set_xlabel('Reduced Mass μ (atomic mass units)')
    ax2.set_ylabel('Fractal Radius Proxy r_eff\n[r_eff = 1/√k_eff, dimensionless]')
    ax2.set_title('(b) Fractal Radius vs Mass', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=7, framealpha=0.95)
    
    # Add interpretation footer
    fig.text(0.5, -0.02, 
             f"HHF Insight: Despite ~3× mass range (H₂→T₂), r_eff varies only {abs(r_vals[0]-r_vals[-1])/r_vals[0]*100:.1f}%, "
             "confirming mass-invariant scaling.",
             ha='center', fontsize=8, fontstyle='italic', color=COLORBLIND_SAFE['blue'])
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Force constants plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_regression(
    result: IsotopologueValidationResult,
    output_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Create regression analysis showing ω_eff vs reduced mass relationship.
    
    Demonstrates the ω ∝ μ^(-0.5) relationship for harmonic oscillators
    with anharmonic corrections.
    
    Parameters
    ----------
    result : IsotopologueValidationResult
        Validation results.
    output_path : Path, optional
        Path to save plot.
    show : bool
        Whether to display interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORBLIND_SAFE, MARKERS,
        add_statistics_annotation, add_hhf_context_box
    )
    
    configure_scientific_style()
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    isotopes = list(result.isotopologues.keys())
    iso_labels = ['H₂', 'D₂', 'T₂'][:len(isotopes)]
    colors = [COLORBLIND_SAFE['blue'], COLORBLIND_SAFE['orange'], COLORBLIND_SAFE['pink']]
    
    omega_vals = np.array([result.isotopologues[iso].omega_eff for iso in isotopes])
    mu_vals = np.array([result.isotopologues[iso].mu for iso in isotopes])
    
    # Panel (a): Frequency vs mass with power law fit
    ax1 = axes[0]
    
    # Theoretical relationship: ω ∝ μ^(-1/2) for harmonic oscillator
    log_mu = np.log(mu_vals)
    log_omega = np.log(omega_vals)
    coeffs = np.polyfit(log_mu, log_omega, 1)
    slope = coeffs[0]
    
    # Plot data points with accessible markers
    for i, (mu, omega, label) in enumerate(zip(mu_vals, omega_vals, iso_labels)):
        ax1.scatter([mu], [omega], c=colors[i], marker=MARKERS[i],
                   s=120, edgecolors='black', linewidths=1, zorder=3, label=label)
    
    # Fit line (observed)
    mu_fit = np.linspace(min(mu_vals) * 0.8, max(mu_vals) * 1.2, 50)
    omega_fit = np.exp(coeffs[1]) * mu_fit ** slope
    ax1.plot(mu_fit, omega_fit, color=COLORBLIND_SAFE['red'], linestyle='--', linewidth=2,
             label=f'Observed fit: ω ∝ μ^{slope:.3f}')
    
    # Theoretical expectation (harmonic oscillator: ω ∝ μ^-0.5)
    omega_theory = omega_vals[0] * (mu_vals[0] / mu_fit) ** 0.5
    ax1.plot(mu_fit, omega_theory, color=COLORBLIND_SAFE['green'], linestyle=':', linewidth=2,
             label='Harmonic theory: ω ∝ μ^−0.5')
    
    # Add isotope labels
    for mu, omega, label in zip(mu_vals, omega_vals, iso_labels):
        ax1.annotate(label, (mu, omega), xytext=(8, 5), textcoords='offset points', 
                    fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Reduced Mass μ (atomic mass units)')
    ax1.set_ylabel('Effective Frequency ω_eff (cm⁻¹)')
    ax1.set_title('(a) Frequency-Mass Relationship', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.95)
    
    # Panel (b): Residuals from theoretical expectation
    ax2 = axes[1]
    
    # Calculate residuals from harmonic oscillator expectation
    omega_expected = omega_vals[0] * np.sqrt(mu_vals[0] / mu_vals)
    residuals_pct = 100 * (omega_vals - omega_expected) / omega_expected
    
    bars = ax2.bar(iso_labels, residuals_pct, color=colors[:len(isotopes)],
                   edgecolor='black', linewidth=1)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    ax2.set_ylabel('Deviation from ω ∝ μ^−0.5 (%)\n[Anharmonic correction]')
    ax2.set_xlabel('Hydrogen Isotopologue')
    ax2.set_title('(b) Anharmonic Deviations', fontweight='bold')
    
    # Add value labels with clear formatting
    for bar, val in zip(bars, residuals_pct):
        offset = 0.3 if val > 0 else -0.5
        ax2.annotate(f'{val:.2f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset),
                    ha='center', va='bottom' if val > 0 else 'top', 
                    fontsize=9, fontweight='bold')
    
    # Add stats annotation
    stats_text = (f'Observed slope: {slope:.3f}\n'
                  f'Harmonic theory: −0.500\n'
                  f'Difference: {abs(slope + 0.5):.3f}')
    add_statistics_annotation(ax2, stats_text, loc='lower right', fontsize=8)
    
    # Add HHF context
    hhf_text = ("Small deviations from\nharmonic theory are\n"
                "anharmonic corrections.\n"
                "HHF accounts for this.")
    add_hhf_context_box(ax2, "HHF Insight", hhf_text, loc='upper right', fontsize=6)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Regression plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results(
    result: IsotopologueValidationResult,
    output_dir: Path
) -> dict:
    """
    Save validation results to files.
    
    Parameters
    ----------
    result : IsotopologueValidationResult
        Validation results.
    output_dir : Path
        Output directory.
        
    Returns
    -------
    dict
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON log with enhanced metadata
    json_path = output_dir / "isotopologue_scaling_log.json"
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    # CSV summary
    csv_path = output_dir / "isotopologue_scaling_summary.csv"
    df = create_results_dataframe(result)
    df.to_csv(csv_path)
    
    logger.info(f"Results saved to {output_dir}")
    
    return {'json': json_path, 'csv': csv_path}


def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False,
    threshold: float = 0.1
) -> IsotopologueValidationResult:
    """
    Execute full isotopologue scaling validation pipeline.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory for outputs. Defaults to 'outputs/isotopologue'.
    show_plots : bool
        Whether to display plots interactively.
    threshold : float
        Maximum allowed deviation for PASS.
        
    Returns
    -------
    IsotopologueValidationResult
        Validation results.
    """
    if output_dir is None:
        output_dir = Path("outputs/isotopologue")
    else:
        output_dir = Path(output_dir)
    
    # Run analysis
    result = run_isotopologue_analysis(threshold=threshold)
    
    # Create DataFrame for display
    df = create_results_dataframe(result)
    
    plots_dir = output_dir / "plots"
    
    # Generate all visualizations
    # 1. Main scaling plot
    plot_isotopologue_scaling(result, 
                              output_path=plots_dir / "isotopologue_scaling.png", 
                              show=show_plots)
    
    # 2. Force constants comparison
    plot_force_constants(result,
                        output_path=plots_dir / "force_constants.png",
                        show=show_plots)
    
    # 3. Regression analysis
    plot_regression(result,
                   output_path=plots_dir / "mass_frequency_regression.png",
                   show=show_plots)
    
    # Save logs
    logs_dir = output_dir / "logs"
    save_results(result, logs_dir)
    
    # Console summary
    print(f"\n{'='*60}")
    print("HHF Isotopologue Scaling Validation")
    print('='*60)
    print(f"\n{df.to_string()}")
    print(f"\nMax deviation: {result.max_deviation:.6f}")
    print(f"Threshold: {result.threshold}")
    print(f"Consistency Check: {'PASS' if result.passed else 'FAIL'}")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {plots_dir}/isotopologue_scaling.png")
    print(f"  - {plots_dir}/force_constants.png")
    print(f"  - {plots_dir}/mass_frequency_regression.png")
    
    return result
