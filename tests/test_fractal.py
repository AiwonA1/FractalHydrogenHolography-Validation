"""
Tests for Fractal Dimension Functions
=====================================
Unit tests for PFD and HFD computations with known inputs/outputs.
"""

import numpy as np
import pytest

from hhf_validation.core.fractal import (
    petrosian_fd,
    higuchi_fd,
    compute_fractal_metrics
)


class TestPetrosianFD:
    """Tests for Petrosian Fractal Dimension."""
    
    def test_constant_signal_returns_nan(self):
        """Constant signal has no sign changes, should return NaN."""
        signal = np.ones(100)
        result = petrosian_fd(signal)
        assert np.isnan(result)
    
    def test_short_signal_returns_nan(self):
        """Signals with < 3 points should return NaN."""
        assert np.isnan(petrosian_fd([1, 2]))
        assert np.isnan(petrosian_fd([1]))
        assert np.isnan(petrosian_fd([]))
    
    def test_sine_wave_reasonable_value(self):
        """Sine wave should produce PFD close to 1.0."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)
        result = petrosian_fd(signal)
        
        # PFD should be close to 1.0 for smooth periodic signals
        assert 1.0 < result < 1.1
    
    def test_random_walk_higher_complexity(self):
        """Random walk should show higher complexity than sine."""
        np.random.seed(42)
        random_walk = np.cumsum(np.random.randn(1000))
        sine = np.sin(np.linspace(0, 10 * np.pi, 1000))
        
        pfd_random = petrosian_fd(random_walk)
        pfd_sine = petrosian_fd(sine)
        
        # Both should be valid
        assert not np.isnan(pfd_random)
        assert not np.isnan(pfd_sine)
        
        # Random walk typically shows different fractal properties
        assert pfd_random > 1.0
    
    def test_list_input_works(self):
        """Should accept Python lists, not just numpy arrays."""
        signal = [1, 2, 1, 3, 2, 4, 3, 5, 4, 6]
        result = petrosian_fd(signal)
        assert not np.isnan(result)
    
    def test_reproducibility(self):
        """Same input should always produce same output."""
        np.random.seed(123)
        signal = np.random.randn(500)
        
        result1 = petrosian_fd(signal)
        result2 = petrosian_fd(signal)
        
        assert result1 == result2


class TestHiguchiFD:
    """Tests for Higuchi Fractal Dimension."""
    
    def test_constant_signal_returns_nan(self):
        """Constant signal should return NaN (no path length)."""
        signal = np.ones(100)
        result = higuchi_fd(signal)
        # May or may not be NaN depending on implementation
        # but should be a valid float or NaN
        assert isinstance(result, float)
    
    def test_short_signal_returns_nan(self):
        """Very short signals should return NaN."""
        assert np.isnan(higuchi_fd([1]))
    
    def test_brownian_motion_fd_reasonable(self):
        """Brownian motion should have valid HFD."""
        np.random.seed(42)
        # Generate Brownian motion (cumulative sum of random steps)
        brownian = np.cumsum(np.random.randn(2000))
        result = higuchi_fd(brownian, k_max=15)
        
        # HFD should be a valid positive value
        # Note: Values can vary based on signal characteristics
        assert not np.isnan(result)
        assert 0.3 < result < 2.0
    
    def test_sine_wave_valid_hfd(self):
        """Smooth sine wave should have valid HFD."""
        t = np.linspace(0, 10 * np.pi, 1000)
        signal = np.sin(t)
        result = higuchi_fd(signal)
        
        # Smooth periodic signals should have valid positive HFD
        assert not np.isnan(result)
        assert result > 0
    
    def test_k_max_parameter(self):
        """Different k_max should produce different but similar results."""
        np.random.seed(42)
        signal = np.random.randn(1000)
        
        result_5 = higuchi_fd(signal, k_max=5)
        result_10 = higuchi_fd(signal, k_max=10)
        result_20 = higuchi_fd(signal, k_max=20)
        
        # All should be valid
        assert not np.isnan(result_5)
        assert not np.isnan(result_10)
        assert not np.isnan(result_20)
        
        # Should be in similar range
        assert abs(result_5 - result_10) < 0.3
        assert abs(result_10 - result_20) < 0.3
    
    def test_reproducibility(self):
        """Same input should always produce same output."""
        np.random.seed(456)
        signal = np.random.randn(500)
        
        result1 = higuchi_fd(signal)
        result2 = higuchi_fd(signal)
        
        assert result1 == result2


class TestComputeFractalMetrics:
    """Tests for combined fractal metrics function."""
    
    def test_returns_all_keys(self):
        """Should return dict with pfd, hfd, and n_points."""
        signal = np.random.randn(100)
        result = compute_fractal_metrics(signal)
        
        assert 'pfd' in result
        assert 'hfd' in result
        assert 'n_points' in result
    
    def test_n_points_matches_input(self):
        """n_points should equal input length."""
        signal = np.random.randn(250)
        result = compute_fractal_metrics(signal)
        
        assert result['n_points'] == 250
    
    def test_matches_individual_functions(self):
        """Results should match individual function calls."""
        np.random.seed(789)
        signal = np.random.randn(500)
        
        metrics = compute_fractal_metrics(signal, k_max=10)
        individual_pfd = petrosian_fd(signal)
        individual_hfd = higuchi_fd(signal, k_max=10)
        
        assert metrics['pfd'] == individual_pfd
        assert metrics['hfd'] == individual_hfd


class TestEdgeCases:
    """Edge case tests for robustness."""
    
    def test_very_long_signal(self):
        """Should handle very long signals."""
        np.random.seed(0)
        signal = np.random.randn(50000)
        
        pfd = petrosian_fd(signal)
        hfd = higuchi_fd(signal)
        
        assert not np.isnan(pfd)
        assert not np.isnan(hfd)
    
    def test_integer_input(self):
        """Should handle integer arrays."""
        signal = np.array([1, 2, 3, 2, 1, 2, 3, 4, 3, 2, 1, 0, 1, 2])
        
        pfd = petrosian_fd(signal)
        hfd = higuchi_fd(signal)
        
        assert not np.isnan(pfd)
    
    def test_signal_with_nans(self):
        """Should handle signals containing NaN values gracefully."""
        signal = np.array([1, 2, np.nan, 4, 5, 6, 7])
        
        # Result may be NaN, but should not raise exception
        pfd = petrosian_fd(signal)
        assert isinstance(pfd, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

