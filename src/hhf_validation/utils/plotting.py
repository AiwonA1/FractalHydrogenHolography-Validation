"""
Scientific Visualization Configuration
======================================
Publication-quality plotting utilities for HHF validation outputs.

Provides consistent, journal-ready figure styling across all validation modules.
Includes accessibility features (colorblind-safe palettes) and HHF-specific
annotation helpers for connecting visualizations to theoretical predictions.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from typing import Tuple, Optional, List, Union
import numpy as np


# =============================================================================
# Color Palettes
# =============================================================================

# Primary palette for categorical data (original)
COLORS = {
    'primary': '#2C3E50',      # Dark blue-gray
    'secondary': '#7F8C8D',    # Medium gray
    'accent': '#E74C3C',       # Muted red for highlights
    'success': '#27AE60',      # Muted green
    'warning': '#F39C12',      # Orange
    'info': '#3498DB',         # Blue
    'gray': '#95A5A6',         # Light gray
    'dark_gray': '#34495E',    # Dark gray
}

# Colorblind-safe palette (Wong 2011, Nature Methods)
# Distinguishable by people with all forms of color vision deficiency
COLORBLIND_SAFE = {
    'blue': '#0072B2',         # Strong blue
    'orange': '#E69F00',       # Orange/amber
    'green': '#009E73',        # Bluish green
    'pink': '#CC79A7',         # Reddish purple
    'yellow': '#F0E442',       # Yellow
    'cyan': '#56B4E9',         # Sky blue
    'red': '#D55E00',          # Vermillion
    'black': '#000000',        # Black
}

# Accessible categorical palette (colorblind-safe order)
ACCESSIBLE_CATEGORICAL = [
    COLORBLIND_SAFE['blue'],
    COLORBLIND_SAFE['orange'],
    COLORBLIND_SAFE['green'],
    COLORBLIND_SAFE['pink'],
    COLORBLIND_SAFE['cyan'],
    COLORBLIND_SAFE['red'],
]

# Sequential palette for ordered data
SEQUENTIAL = ['#2C3E50', '#5D6D7E', '#85929E', '#ABB2B9', '#D5D8DC']

# Categorical palette for multiple series (original)
CATEGORICAL = ['#2C3E50', '#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']

# Marker styles for redundant encoding (accessibility)
MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

# Line styles for redundant encoding
LINESTYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 2))]


# =============================================================================
# HHF Theoretical Constants for Annotations
# =============================================================================

# HHF prediction values for annotation reference
HHF_PREDICTIONS = {
    'pfd_complex_systems': 1.02,      # Expected PFD for complex natural systems
    'pfd_range': (1.01, 1.03),        # Acceptable PFD range
    'lambda_hh_unity': 1.0,           # Mass-invariant scaling target
    'lambda_hh_threshold': 0.1,       # Acceptable deviation from unity
    'scaling_ratio': 1.12e22,         # Λᴴᴴ = R_HHF / L_P
}


# =============================================================================
# Figure Configuration
# =============================================================================

def configure_scientific_style():
    """
    Configure matplotlib for publication-quality figures.
    
    Sets font families, sizes, line widths, and other parameters
    suitable for journal submission.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    
    mpl.rcParams.update({
        # Font configuration - use DejaVu for better unicode support
        'font.family': 'serif',
        'font.serif': ['DejaVu Serif', 'Times New Roman', 'serif'],
        'mathtext.fontset': 'dejavuserif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        
        # Figure configuration
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Line and marker configuration
        'lines.linewidth': 1.0,
        'lines.markersize': 5,
        'axes.linewidth': 0.8,
        
        # Grid configuration
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Legend configuration
        'legend.framealpha': 0.9,
        'legend.edgecolor': '0.8',
        
        # Tick configuration
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
    })


def create_figure(
    width: str = 'single',
    aspect: float = 0.618,
    nrows: int = 1,
    ncols: int = 1
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Create a figure with publication-ready dimensions.
    
    Parameters
    ----------
    width : str
        'single' for single-column (3.5"), 'double' for double-column (7").
    aspect : float
        Height/width ratio (default: golden ratio inverse).
    nrows, ncols : int
        Subplot grid dimensions.
        
    Returns
    -------
    fig : Figure
        Matplotlib figure object.
    axes : ndarray
        Array of axes objects.
    """
    configure_scientific_style()
    
    widths = {
        'single': 3.5,
        'double': 7.0,
        'wide': 5.5,
    }
    
    fig_width = widths.get(width, 7.0)
    fig_height = fig_width * aspect * (nrows / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))
    
    return fig, np.atleast_1d(axes)


def add_panel_labels(
    axes,
    labels: Optional[list] = None,
    loc: str = 'upper left',
    fontweight: str = 'bold'
):
    """
    Add panel labels (a, b, c, ...) to subplots.
    
    Parameters
    ----------
    axes : array-like
        Axes objects to label.
    labels : list, optional
        Custom labels. Defaults to (a), (b), (c), ...
    loc : str
        Label location ('upper left', 'upper right', etc.).
    fontweight : str
        Font weight for labels.
    """
    if labels is None:
        labels = [f'({chr(97 + i)})' for i in range(len(axes.flat))]
    
    for ax, label in zip(axes.flat, labels):
        if loc == 'upper left':
            ax.text(0.02, 0.98, label, transform=ax.transAxes,
                   fontweight=fontweight, va='top', ha='left')
        elif loc == 'upper right':
            ax.text(0.98, 0.98, label, transform=ax.transAxes,
                   fontweight=fontweight, va='top', ha='right')


def format_axis_scientific(ax, axis: str = 'y', scilimits: tuple = (-2, 3)):
    """
    Format axis with scientific notation for large/small values.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    axis : str
        'x', 'y', or 'both'.
    scilimits : tuple
        Range of exponents to use scientific notation.
    """
    if axis in ('y', 'both'):
        ax.ticklabel_format(style='scientific', axis='y', scilimits=scilimits)
    if axis in ('x', 'both'):
        ax.ticklabel_format(style='scientific', axis='x', scilimits=scilimits)


def add_statistics_annotation(
    ax,
    text: str,
    loc: str = 'upper right',
    fontsize: int = 9
):
    """
    Add a statistics annotation box to the plot.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    text : str
        Statistics text (can include newlines).
    loc : str
        Box location.
    fontsize : int
        Font size for text.
    """
    props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='0.7')
    
    if loc == 'upper right':
        x, y, va, ha = 0.97, 0.97, 'top', 'right'
    elif loc == 'upper left':
        x, y, va, ha = 0.03, 0.97, 'top', 'left'
    elif loc == 'lower right':
        x, y, va, ha = 0.97, 0.03, 'bottom', 'right'
    else:  # lower left
        x, y, va, ha = 0.03, 0.03, 'bottom', 'left'
    
    ax.text(x, y, text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, horizontalalignment=ha, bbox=props)


def save_figure(fig, path, formats: list = ['png', 'pdf']):
    """
    Save figure in multiple formats.
    
    Parameters
    ----------
    fig : Figure
        Matplotlib figure.
    path : Path
        Output path (without extension).
    formats : list
        List of formats to save ('png', 'pdf', 'svg').
    """
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(path.with_suffix(f'.{fmt}'), format=fmt)


# =============================================================================
# HHF-Specific Annotation Helpers
# =============================================================================

def add_hhf_prediction_band(
    ax,
    center: float,
    width: float,
    orientation: str = 'horizontal',
    color: str = None,
    alpha: float = 0.15,
    label: str = 'HHF Predicted Range',
    show_center_line: bool = True
):
    """
    Add a shaded band showing HHF theoretical prediction range.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    center : float
        Center value of the prediction.
    width : float
        Half-width of the prediction band (center ± width).
    orientation : str
        'horizontal' for y-axis band, 'vertical' for x-axis band.
    color : str, optional
        Band color. Defaults to colorblind-safe green.
    alpha : float
        Band transparency.
    label : str
        Legend label for the band.
    show_center_line : bool
        Whether to draw a dashed line at center.
    """
    if color is None:
        color = COLORBLIND_SAFE['green']
    
    if orientation == 'horizontal':
        ax.axhspan(center - width, center + width, 
                   alpha=alpha, color=color, label=label, zorder=0)
        if show_center_line:
            ax.axhline(y=center, color=color, linestyle='--', 
                      linewidth=1.5, alpha=0.8, zorder=1)
    else:
        ax.axvspan(center - width, center + width,
                   alpha=alpha, color=color, label=label, zorder=0)
        if show_center_line:
            ax.axvline(x=center, color=color, linestyle='--',
                      linewidth=1.5, alpha=0.8, zorder=1)


def add_hhf_context_box(
    ax,
    title: str,
    text: str,
    loc: str = 'lower left',
    fontsize: int = 8,
    title_fontsize: int = 9,
    width: float = 0.35
):
    """
    Add an interpretive context box explaining HHF significance.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    title : str
        Bold title for the box.
    text : str
        Explanatory text (can include newlines).
    loc : str
        Box location: 'upper left', 'upper right', 'lower left', 'lower right'.
    fontsize : int
        Font size for body text.
    title_fontsize : int
        Font size for title.
    width : float
        Box width as fraction of axes (0-1).
    """
    props = dict(boxstyle='round,pad=0.5', facecolor='#FFFEF0', 
                 alpha=0.95, edgecolor=COLORBLIND_SAFE['blue'], linewidth=1.2)
    
    positions = {
        'upper right': (0.97, 0.97, 'top', 'right'),
        'upper left': (0.03, 0.97, 'top', 'left'),
        'lower right': (0.97, 0.03, 'bottom', 'right'),
        'lower left': (0.03, 0.03, 'bottom', 'left'),
    }
    
    x, y, va, ha = positions.get(loc, positions['lower left'])
    
    full_text = f"$\\bf{{{title}}}$\n{text}"
    
    ax.text(x, y, full_text, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment=va, horizontalalignment=ha, bbox=props,
            linespacing=1.4)


def add_interpretation_legend(
    ax,
    interpretations: List[Tuple[str, str, str]],
    loc: str = 'upper left',
    fontsize: int = 8
):
    """
    Add an interpretation legend explaining what visual elements mean.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    interpretations : list of tuples
        Each tuple: (color_or_marker, label, meaning).
        Example: [('#0072B2', 'Blue points', 'Observed data')]
    loc : str
        Legend location.
    fontsize : int
        Font size.
    """
    handles = []
    labels = []
    
    for item in interpretations:
        if len(item) == 3:
            color, label, meaning = item
            # Create a patch for color-based items
            patch = mpatches.Patch(color=color, label=f"{label}: {meaning}")
            handles.append(patch)
            labels.append(f"{label}: {meaning}")
        elif len(item) == 2:
            handle, label = item
            handles.append(handle)
            labels.append(label)
    
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, 
              framealpha=0.95, edgecolor='0.7')


def add_axis_explanation(
    ax,
    x_explanation: Optional[str] = None,
    y_explanation: Optional[str] = None,
    fontsize: int = 8
):
    """
    Add explanatory text below axis labels.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    x_explanation : str, optional
        Explanation for x-axis meaning.
    y_explanation : str, optional
        Explanation for y-axis meaning.
    fontsize : int
        Font size for explanations.
    """
    if x_explanation:
        ax.annotate(x_explanation, xy=(0.5, -0.18), xycoords='axes fraction',
                   ha='center', va='top', fontsize=fontsize, fontstyle='italic',
                   color=COLORS['dark_gray'])
    
    if y_explanation:
        ax.annotate(y_explanation, xy=(-0.18, 0.5), xycoords='axes fraction',
                   ha='right', va='center', fontsize=fontsize, fontstyle='italic',
                   color=COLORS['dark_gray'], rotation=90)


def create_accessible_scatter(
    ax,
    x_data: List[np.ndarray],
    y_data: List[np.ndarray],
    labels: List[str],
    use_colorblind_safe: bool = True
):
    """
    Create scatter plot with both color AND marker differentiation for accessibility.
    
    Parameters
    ----------
    ax : Axes
        Matplotlib axes object.
    x_data : list of arrays
        X coordinates for each series.
    y_data : list of arrays
        Y coordinates for each series.
    labels : list of str
        Labels for each series.
    use_colorblind_safe : bool
        Use colorblind-safe palette.
        
    Returns
    -------
    list
        List of scatter plot handles.
    """
    colors = ACCESSIBLE_CATEGORICAL if use_colorblind_safe else CATEGORICAL
    handles = []
    
    for i, (x, y, label) in enumerate(zip(x_data, y_data, labels)):
        color = colors[i % len(colors)]
        marker = MARKERS[i % len(MARKERS)]
        
        scatter = ax.scatter(x, y, c=color, marker=marker, s=60,
                           edgecolors='white', linewidths=0.5,
                           label=label, zorder=3)
        handles.append(scatter)
    
    ax.legend(loc='best', fontsize=8)
    return handles


def format_hhf_title(base_title: str, validation_type: str) -> str:
    """
    Create consistent HHF validation plot title.
    
    Parameters
    ----------
    base_title : str
        Main title text.
    validation_type : str
        Type of validation (e.g., 'Biological', 'Isotopologue').
        
    Returns
    -------
    str
        Formatted title with HHF branding.
    """
    return f"HHF {validation_type} Validation: {base_title}"

