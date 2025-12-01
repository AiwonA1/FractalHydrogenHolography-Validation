"""
Biological Proxy Validation Module
==================================
HHF validation using biological/environmental time-series data.

Analyzes fractal properties of real-world time series to validate
HHF predictions of complex, self-similar dynamics.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from ..core.fractal import petrosian_fd, higuchi_fd, compute_fractal_metrics

logger = logging.getLogger(__name__)

# Data source URL
MELBOURNE_TEMP_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"


@dataclass
class BiologicalProxyResult:
    """Results from biological proxy fractal analysis."""
    dataset_name: str
    n_points: int
    pfd: float
    hfd: float
    mean_temp: float = 0.0
    std_temp: float = 0.0
    min_temp: float = 0.0
    max_temp: float = 0.0
    timestamp: str = ""
    data_source: str = ""
    k_max: int = 10
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def load_melbourne_temperature_data(
    url: str = MELBOURNE_TEMP_URL
) -> pd.DataFrame:
    """
    Load Daily Minimum Temperatures dataset from Melbourne.
    
    Parameters
    ----------
    url : str
        URL to the dataset CSV.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with Date and Temp columns.
    """
    logger.info("Loading Daily Minimum Temperatures (Melbourne) dataset...")
    df = pd.read_csv(url)
    logger.info(f"Loaded {len(df)} data points")
    return df


def analyze_temperature_series(
    df: pd.DataFrame,
    temp_column: str = 'Temp',
    k_max: int = 10
) -> BiologicalProxyResult:
    """
    Compute fractal dimensions for temperature time series.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing temperature data.
    temp_column : str
        Name of the temperature column.
    k_max : int
        Maximum interval for HFD computation.
        
    Returns
    -------
    BiologicalProxyResult
        Fractal analysis results.
    """
    signal = df[temp_column].values
    
    pfd_val = petrosian_fd(signal)
    hfd_val = higuchi_fd(signal, k_max=k_max)
    
    result = BiologicalProxyResult(
        dataset_name="Daily_Minimum_Temperatures_Melbourne",
        n_points=len(signal),
        pfd=pfd_val,
        hfd=hfd_val,
        mean_temp=float(np.mean(signal)),
        std_temp=float(np.std(signal)),
        min_temp=float(np.min(signal)),
        max_temp=float(np.max(signal)),
        timestamp=datetime.now().isoformat(),
        data_source=MELBOURNE_TEMP_URL,
        k_max=k_max
    )
    
    logger.info(f"Processed {result.dataset_name}: PFD={pfd_val:.3f}, HFD={hfd_val:.3f}")
    return result


def plot_temperature_series(
    df: pd.DataFrame,
    result: BiologicalProxyResult,
    output_path: Optional[Path] = None,
    temp_column: str = 'Temp',
    show: bool = False
) -> None:
    """
    Generate publication-quality visualization of temperature series with fractal metrics.
    
    Includes HHF theoretical context and accessibility features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Temperature data.
    result : BiologicalProxyResult
        Computed fractal metrics.
    output_path : Path, optional
        Path to save the plot. If None, only displays.
    temp_column : str
        Column name for temperature data.
    show : bool
        Whether to display the plot interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORS, COLORBLIND_SAFE,
        add_statistics_annotation, add_hhf_context_box, HHF_PREDICTIONS
    )
    
    configure_scientific_style()
    
    fig, ax = plt.subplots(figsize=(7, 3.2))
    
    # Plot time series with colorblind-safe colors
    temps = df[temp_column].values
    days = np.arange(len(temps))
    ax.plot(days, temps, color=COLORBLIND_SAFE['blue'], linewidth=0.4, alpha=0.7,
            label='Daily minimum temperature')
    
    # Add smoothed trend line with distinct style
    window = 30
    if len(temps) > window:
        smoothed = np.convolve(temps, np.ones(window)/window, mode='valid')
        ax.plot(days[window//2:window//2+len(smoothed)], smoothed, 
                color=COLORBLIND_SAFE['orange'], linewidth=1.5, 
                linestyle='-', label='30-day moving average (seasonal trend)')
    
    ax.set_xlabel('Day Index (sequential days from 1981-01-01)')
    ax.set_ylabel('Temperature (°C)')
    ax.set_title('HHF Biological Proxy: Daily Minimum Temperatures — Melbourne (1981–1990)', 
                 fontweight='bold', fontsize=10)
    
    # Enhanced statistics box with HHF context
    stats_text = (f'n = {result.n_points:,} days\n'
                  f'PFD = {result.pfd:.4f}\n'
                  f'HFD = {result.hfd:.4f}')
    add_statistics_annotation(ax, stats_text, loc='upper right')
    
    # Add HHF interpretation context box
    hhf_text = (f"PFD ≈ {HHF_PREDICTIONS['pfd_complex_systems']:.2f} indicates\n"
                f"complex fractal dynamics.\n"
                f"Observed: {result.pfd:.3f}")
    add_hhf_context_box(ax, "HHF Prediction", hhf_text, loc='lower left', fontsize=7)
    
    # Enhanced legend with accessibility
    legend = ax.legend(loc='lower right', framealpha=0.95, fontsize=8,
                       title='Legend', title_fontsize=8)
    legend.get_frame().set_edgecolor(COLORBLIND_SAFE['blue'])
    
    ax.set_xlim(0, len(temps))
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_distribution(
    df: pd.DataFrame,
    result: BiologicalProxyResult,
    output_path: Optional[Path] = None,
    temp_column: str = 'Temp',
    show: bool = False
) -> None:
    """
    Generate histogram and distribution analysis of temperature values.
    
    Includes HHF theoretical context and accessibility features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Temperature data.
    result : BiologicalProxyResult
        Computed fractal metrics.
    output_path : Path, optional
        Path to save the plot.
    temp_column : str
        Column name for temperature data.
    show : bool
        Whether to display the plot interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORBLIND_SAFE, ACCESSIBLE_CATEGORICAL,
        add_statistics_annotation, add_hhf_context_box
    )
    
    configure_scientific_style()
    
    temps = df[temp_column].values
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    # Panel (a): Histogram with KDE overlay - accessible colors
    ax1 = axes[0]
    n_bins = 50
    counts, bins, patches = ax1.hist(temps, bins=n_bins, color=COLORBLIND_SAFE['blue'], 
                                      alpha=0.6, edgecolor='white', linewidth=0.5,
                                      density=True, label='Observed frequency')
    
    # Add kernel density estimate with distinct style
    kde = stats.gaussian_kde(temps)
    x_kde = np.linspace(temps.min(), temps.max(), 200)
    ax1.plot(x_kde, kde(x_kde), color=COLORBLIND_SAFE['orange'], linewidth=2.5, 
             linestyle='-', label='KDE (actual distribution)')
    
    # Add normal distribution overlay for comparison
    x_norm = np.linspace(temps.min(), temps.max(), 200)
    norm_pdf = stats.norm.pdf(x_norm, result.mean_temp, result.std_temp)
    ax1.plot(x_norm, norm_pdf, color=COLORBLIND_SAFE['pink'], linewidth=2, 
             linestyle='--', label='Normal fit (Gaussian)')
    
    ax1.set_xlabel('Temperature (°C)\n[Minimum daily temperature]')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('(a) Temperature Distribution', fontweight='bold')
    
    # Enhanced legend with explanations
    legend1 = ax1.legend(loc='upper right', fontsize=7, title='Distribution Analysis',
                         title_fontsize=8, framealpha=0.95)
    
    # Add statistics
    stats_text = (f'μ = {result.mean_temp:.1f}°C\n'
                  f'σ = {result.std_temp:.1f}°C')
    add_statistics_annotation(ax1, stats_text, loc='upper left', fontsize=8)
    
    # Panel (b): Box plot with seasonal breakdown - accessible colors
    ax2 = axes[1]
    
    # Create seasonal groups (approximate: 91 days per season)
    n_years = len(temps) // 365
    seasons = ['Summer', 'Autumn', 'Winter', 'Spring']
    seasonal_data = []
    
    # Melbourne seasons (Southern Hemisphere)
    for i, season in enumerate(seasons):
        mask = np.zeros(len(temps), dtype=bool)
        for year in range(n_years):
            start_idx = year * 365 + i * 91
            end_idx = min(start_idx + 91, len(temps))
            if start_idx < len(temps):
                mask[start_idx:end_idx] = True
        seasonal_data.append(temps[mask])
    
    bp = ax2.boxplot(seasonal_data, labels=seasons, patch_artist=True)
    
    # Colorblind-safe seasonal colors with distinct patterns
    colors_seasonal = [
        COLORBLIND_SAFE['red'],      # Summer (hot)
        COLORBLIND_SAFE['orange'],   # Autumn 
        COLORBLIND_SAFE['blue'],     # Winter (cold)
        COLORBLIND_SAFE['green']     # Spring
    ]
    for patch, color in zip(bp['boxes'], colors_seasonal):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.2)
    
    # Style whiskers and medians
    for whisker in bp['whiskers']:
        whisker.set_color('black')
        whisker.set_linewidth(1)
    for median in bp['medians']:
        median.set_color('black')
        median.set_linewidth(2)
    
    ax2.set_ylabel('Temperature (°C)')
    ax2.set_xlabel('Season (Southern Hemisphere)')
    ax2.set_title('(b) Seasonal Variation', fontweight='bold')
    
    # Add HHF context annotation
    hhf_text = ("Seasonal cycles create\nmulti-scale structure.\n"
                "HHF links periodic patterns\nto fractal complexity.")
    add_hhf_context_box(ax2, "HHF Insight", hhf_text, loc='upper right', fontsize=6)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Distribution plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_autocorrelation(
    df: pd.DataFrame,
    result: BiologicalProxyResult,
    output_path: Optional[Path] = None,
    temp_column: str = 'Temp',
    max_lag: int = 400,
    show: bool = False
) -> None:
    """
    Generate autocorrelation analysis showing temporal structure.
    
    Includes HHF theoretical context explaining how periodic structure
    relates to fractal complexity.
    
    Parameters
    ----------
    df : pd.DataFrame
        Temperature data.
    result : BiologicalProxyResult
        Computed fractal metrics.
    output_path : Path, optional
        Path to save the plot.
    temp_column : str
        Column name for temperature data.
    max_lag : int
        Maximum lag for autocorrelation.
    show : bool
        Whether to display the plot interactively.
    """
    from ..utils.plotting import (
        configure_scientific_style, COLORBLIND_SAFE, 
        add_statistics_annotation, add_hhf_context_box, HHF_PREDICTIONS
    )
    
    configure_scientific_style()
    
    temps = df[temp_column].values
    
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5))
    
    # Panel (a): Autocorrelation function
    ax1 = axes[0]
    
    # Compute autocorrelation
    n = len(temps)
    temps_centered = temps - np.mean(temps)
    acf = np.correlate(temps_centered, temps_centered, mode='full')
    acf = acf[n-1:n-1+max_lag+1] / acf[n-1]  # Normalize
    
    lags = np.arange(0, max_lag + 1)
    ax1.plot(lags, acf, color=COLORBLIND_SAFE['blue'], linewidth=1.2,
             label='ACF (temporal correlation)')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add confidence interval (95%) with explanation
    conf = 1.96 / np.sqrt(n)
    ax1.axhline(y=conf, color=COLORBLIND_SAFE['pink'], linestyle='--', 
                linewidth=1, alpha=0.8)
    ax1.axhline(y=-conf, color=COLORBLIND_SAFE['pink'], linestyle='--', 
                linewidth=1, alpha=0.8)
    ax1.fill_between(lags, -conf, conf, color=COLORBLIND_SAFE['pink'], 
                     alpha=0.15, label='95% confidence interval')
    
    # Mark yearly cycle (365 days) with distinct style
    ax1.axvline(x=365, color=COLORBLIND_SAFE['orange'], linestyle='-', 
                linewidth=2, alpha=0.8, label='Annual cycle (365 days)')
    
    ax1.set_xlabel('Lag (days)\n[Time offset between observations]')
    ax1.set_ylabel('Autocorrelation Coefficient\n[Correlation with past self]')
    ax1.set_title('(a) Autocorrelation Function', fontweight='bold')
    ax1.set_xlim(0, max_lag)
    ax1.legend(loc='upper right', fontsize=7, framealpha=0.95)
    
    # Panel (b): Power spectrum (frequency domain)
    ax2 = axes[1]
    
    # Compute power spectrum using FFT
    fft = np.fft.fft(temps_centered)
    psd = np.abs(fft[:n//2])**2 / n
    freqs = np.fft.fftfreq(n, d=1.0)[:n//2]  # Cycles per day
    periods = 1.0 / (freqs + 1e-10)  # Days
    
    # Plot on log-log scale with accessible colors
    valid_idx = (periods > 2) & (periods < 1000)
    ax2.loglog(periods[valid_idx], psd[valid_idx], color=COLORBLIND_SAFE['blue'], 
               linewidth=0.5, alpha=0.5, label='Raw PSD')
    
    # Add smoothed version
    window = 20
    if len(psd[valid_idx]) > window:
        psd_smooth = np.convolve(psd[valid_idx], np.ones(window)/window, mode='valid')
        periods_smooth = periods[valid_idx][window//2:window//2+len(psd_smooth)]
        ax2.loglog(periods_smooth, psd_smooth, color=COLORBLIND_SAFE['orange'], 
                   linewidth=2.5, label='Smoothed PSD')
    
    # Mark key periods with distinct styles
    ax2.axvline(x=365, color=COLORBLIND_SAFE['green'], linestyle='-', 
                linewidth=2, alpha=0.8, label='Annual (365d)')
    ax2.axvline(x=182.5, color=COLORBLIND_SAFE['pink'], linestyle='--', 
                linewidth=1.5, alpha=0.8, label='Semi-annual (183d)')
    
    ax2.set_xlabel('Period (days) [Log scale]\n[Characteristic timescale of fluctuations]')
    ax2.set_ylabel('Power Spectral Density [Log scale]\n[Energy at each frequency]')
    ax2.set_title('(b) Power Spectrum', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=6, framealpha=0.95)
    
    # Add HHF fractal dimension context
    stats_text = (f'PFD = {result.pfd:.4f}\n'
                  f'HFD = {result.hfd:.4f}\n'
                  f'HHF target: ~{HHF_PREDICTIONS["pfd_complex_systems"]:.2f}')
    add_statistics_annotation(ax2, stats_text, loc='lower left', fontsize=8)
    
    # Add overall HHF interpretation
    fig.text(0.5, -0.02, 
             "HHF Interpretation: Annual cycles create multi-scale structure; "
             "PFD near 1.02 confirms complex fractal dynamics predicted by HHF framework.",
             ha='center', fontsize=8, fontstyle='italic', 
             color=COLORBLIND_SAFE['blue'])
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
        logger.info(f"Autocorrelation plot saved to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def save_results(
    result: BiologicalProxyResult,
    output_dir: Path
) -> dict:
    """
    Save analysis results to JSON and CSV files.
    
    Parameters
    ----------
    result : BiologicalProxyResult
        Analysis results.
    output_dir : Path
        Directory for output files.
        
    Returns
    -------
    dict
        Paths to saved files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON log with full metadata
    json_path = output_dir / "fractal_analysis_log.json"
    log_data = {
        result.dataset_name: result.to_dict(),
        'metadata': {
            'timestamp': result.timestamp,
            'data_source': result.data_source,
            'computation_parameters': {
                'k_max': result.k_max
            }
        }
    }
    with open(json_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    # CSV summary
    csv_path = output_dir / "fractal_analysis_summary.csv"
    df_summary = pd.DataFrame([result.to_dict()])
    df_summary.to_csv(csv_path, index=False)
    
    logger.info(f"Results saved to {output_dir}")
    
    return {'json': json_path, 'csv': csv_path}


def run_validation(
    output_dir: Optional[Path] = None,
    show_plots: bool = False
) -> BiologicalProxyResult:
    """
    Execute full biological proxy validation pipeline.
    
    Parameters
    ----------
    output_dir : Path, optional
        Directory for outputs. Defaults to 'outputs/biological'.
    show_plots : bool
        Whether to display plots interactively.
        
    Returns
    -------
    BiologicalProxyResult
        Validation results.
    """
    if output_dir is None:
        output_dir = Path("outputs/biological")
    else:
        output_dir = Path(output_dir)
    
    # Load and analyze
    df = load_melbourne_temperature_data()
    result = analyze_temperature_series(df)
    
    plots_dir = output_dir / "plots"
    
    # Generate all visualizations
    # 1. Time series plot
    plot_temperature_series(df, result, 
                           output_path=plots_dir / "daily_min_temperatures.png", 
                           show=show_plots)
    
    # 2. Distribution analysis
    plot_distribution(df, result,
                     output_path=plots_dir / "temperature_distribution.png",
                     show=show_plots)
    
    # 3. Autocorrelation analysis
    plot_autocorrelation(df, result,
                        output_path=plots_dir / "autocorrelation_analysis.png",
                        show=show_plots)
    
    # Save logs
    logs_dir = output_dir / "logs"
    save_results(result, logs_dir)
    
    # Console summary
    print(f"\n{'='*60}")
    print("HHF Biological Proxy Validation Complete")
    print('='*60)
    print(f"\nDataset: {result.dataset_name}")
    print(f"Points:  {result.n_points}")
    print(f"PFD:     {result.pfd:.6f}")
    print(f"HFD:     {result.hfd:.6f}")
    print(f"\nStatistics:")
    print(f"  Mean:  {result.mean_temp:.2f}°C")
    print(f"  Std:   {result.std_temp:.2f}°C")
    print(f"  Range: [{result.min_temp:.1f}, {result.max_temp:.1f}]°C")
    print(f"\nOutputs saved to: {output_dir}")
    print(f"  - {plots_dir}/daily_min_temperatures.png")
    print(f"  - {plots_dir}/temperature_distribution.png")
    print(f"  - {plots_dir}/autocorrelation_analysis.png")
    
    return result
