"""
Fractal Dimension Computation Module
=====================================
Provides Petrosian Fractal Dimension (PFD) and Higuchi Fractal Dimension (HFD)
for time-series analysis within the HHF validation framework.

These metrics quantify signal complexity and self-similarity, key properties
predicted by the Hydrogen-Holographic Fractal (HHF) framework.
"""

import numpy as np
from typing import Union, Optional
from numpy.typing import ArrayLike


def petrosian_fd(signal: ArrayLike) -> float:
    """
    Compute Petrosian Fractal Dimension (PFD) of a 1D signal.
    
    PFD captures overall irregularity in a time series by analyzing
    zero-crossings in the first derivative.
    
    Parameters
    ----------
    signal : array-like
        1D input signal (time series).
        
    Returns
    -------
    float
        Petrosian Fractal Dimension value.
        Returns np.nan if signal too short or no sign changes detected.
        
    References
    ----------
    Petrosian A. (1995). Kolmogorov complexity of finite sequences and 
    recognition of different preictal EEG patterns.
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    
    if N < 3:
        return np.nan
    
    # First derivative
    diff = np.diff(sig)
    
    # Count sign changes in derivative (zero crossings)
    sign_changes = np.sum(diff[:-1] * diff[1:] < 0)
    
    if sign_changes <= 0:
        return np.nan
    
    # Petrosian formula
    return np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * sign_changes)))


def higuchi_fd(signal: ArrayLike, k_max: int = 10) -> float:
    """
    Compute Higuchi Fractal Dimension (HFD) of a 1D signal.
    
    HFD is sensitive to local fluctuations, providing a complementary
    measure of complexity to PFD. It estimates fractal dimension by
    analyzing curve length at multiple scales.
    
    Parameters
    ----------
    signal : array-like
        1D input signal (time series).
    k_max : int, optional
        Maximum interval scale for analysis (default: 10).
        Higher values provide more scale information but require longer signals.
        
    Returns
    -------
    float
        Higuchi Fractal Dimension value.
        Returns np.nan if insufficient data for computation.
        
    References
    ----------
    Higuchi T. (1988). Approach to an irregular time series on the basis 
    of the fractal theory. Physica D: Nonlinear Phenomena.
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    
    L = []
    for k in range(1, k_max + 1):
        Lk = []
        for m in range(k):
            # Subsample indices starting at offset m with step k
            idxs = np.arange(m, N, k)
            x = sig[idxs]
            
            if len(x) < 2:
                continue
            
            # Normalized path length for this subsample
            lk = np.sum(np.abs(np.diff(x))) * (N - 1) / ((len(x) - 1) * k)
            Lk.append(lk)
        
        if Lk:
            L.append(np.mean(Lk))
    
    if len(L) < 2:
        return np.nan
    
    L_arr = np.array(L)
    
    # Handle zero or negative values (constant signal edge case)
    # These produce undefined log values, so return NaN
    if np.any(L_arr <= 0):
        return np.nan
    
    # Linear regression in log-log space: log(L) vs log(1/k)
    lnL = np.log(L_arr)
    lnk = np.log(1.0 / np.arange(1, len(L) + 1))
    
    # Least squares fit: lnL = fd * lnk + intercept
    A = np.vstack([lnk, np.ones(len(lnk))]).T
    fd, _ = np.linalg.lstsq(A, lnL, rcond=None)[0]
    
    return float(fd)


def compute_fractal_metrics(
    signal: ArrayLike,
    k_max: int = 10
) -> dict:
    """
    Compute both PFD and HFD for a signal.
    
    Parameters
    ----------
    signal : array-like
        1D input signal (time series).
    k_max : int, optional
        Maximum interval scale for HFD (default: 10).
        
    Returns
    -------
    dict
        Dictionary with keys 'pfd', 'hfd', and 'n_points'.
    """
    sig = np.asarray(signal, dtype=float)
    
    return {
        'n_points': len(sig),
        'pfd': petrosian_fd(sig),
        'hfd': higuchi_fd(sig, k_max=k_max)
    }

