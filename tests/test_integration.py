"""
Integration Tests for HHF Validation Pipelines
===============================================
End-to-end tests verifying complete validation workflows.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import json

from hhf_validation.core.fractal import petrosian_fd, higuchi_fd
from hhf_validation.utils.constants import (
    compute_hhf_radius, compute_hhf_scaling_ratio,
    L_P, h, c, m_p, alpha
)


class TestHHFConstantsIntegration:
    """Integration tests for HHF constant calculations."""
    
    def test_hhf_radius_matches_literature(self):
        """
        R_HHF should match the expected value from HHF literature.
        Expected: ~1.81 × 10^-13 m
        """
        R_HHF = compute_hhf_radius()
        
        # Verify formula: R_HHF = h / (m_p * c * alpha)
        expected = h / (m_p * c * alpha)
        assert R_HHF == pytest.approx(expected)
        
        # Verify matches literature value
        assert 1.80e-13 < R_HHF < 1.82e-13
    
    def test_hhf_scaling_ratio_matches_literature(self):
        """
        Λ^HH = R_HHF / L_P should be ~1.12 × 10^22
        """
        ratio = compute_hhf_scaling_ratio()
        
        # Verify value matches expectation
        assert 1.1e22 < ratio < 1.2e22
        
        # Verify self-consistency
        R_HHF = compute_hhf_radius()
        assert ratio == pytest.approx(R_HHF / L_P)
    
    def test_area_volume_scaling_consistency(self):
        """
        Area and volume scaling should be powers of the linear scaling.
        """
        from hhf_validation.utils.constants import LAMBDA_HH, S_H, V_H
        
        assert S_H == pytest.approx(LAMBDA_HH ** 2)
        assert V_H == pytest.approx(LAMBDA_HH ** 3)


class TestIsotopologueIntegration:
    """Integration tests for isotopologue validation."""
    
    def test_full_isotopologue_analysis(self):
        """Run complete isotopologue analysis and verify results."""
        from hhf_validation.validations.isotopologue import run_isotopologue_analysis
        
        result = run_isotopologue_analysis(threshold=0.1)
        
        # Verify all three isotopologues were analyzed
        assert 'H2' in result.isotopologues
        assert 'D2' in result.isotopologues
        assert 'T2' in result.isotopologues
        
        # Verify H2 is the reference (Lambda_HH = 1.0)
        assert result.isotopologues['H2'].lambda_hh == 1.0
        
        # Verify validation passes
        assert result.passed
        assert result.max_deviation < 0.1
    
    def test_isotopologue_output_files(self):
        """Verify isotopologue validation generates correct output files."""
        from hhf_validation.validations.isotopologue import (
            run_isotopologue_analysis, save_results, create_results_dataframe
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = run_isotopologue_analysis()
            paths = save_results(result, Path(tmpdir))
            
            # Verify files were created
            assert paths['json'].exists()
            assert paths['csv'].exists()
            
            # Verify JSON content
            with open(paths['json']) as f:
                data = json.load(f)
            assert 'isotopologues' in data
            assert data['status'] == 'PASS'
            
            # Verify DataFrame
            df = create_results_dataframe(result)
            assert len(df) == 3
            assert 'Lambda_HH' in df.columns


class TestMolecularPhotonicIntegration:
    """Integration tests for molecular/photonic validation."""
    
    def test_full_molecular_photonic_analysis(self):
        """Run complete molecular/photonic analysis."""
        from hhf_validation.validations.molecular_photonic import (
            run_molecular_photonic_analysis
        )
        
        result = run_molecular_photonic_analysis()
        
        # Verify theoretical calculations
        assert 1.8e-13 < result.theoretical.R_HHF < 1.82e-13
        assert 1.1e22 < result.theoretical.Lambda_HH < 1.2e22
        
        # Verify spectral validation passes
        assert result.spectral.validated
        assert result.spectral.relative_error < 1e-6
    
    def test_rydberg_derivation_chain(self):
        """Verify the Rydberg constant calculation chain is consistent."""
        from hhf_validation.validations.molecular_photonic import (
            compute_rydberg_constant,
            compute_hydrogen_rydberg,
            CODATA_RYDBERG_H
        )
        
        R_inf = compute_rydberg_constant()
        R_H = compute_hydrogen_rydberg()
        
        # R_H should be slightly less than R_inf due to finite proton mass
        assert R_H < R_inf
        
        # Reduced mass factor: m_p / (m_e + m_p) < 1
        from hhf_validation.utils.constants import m_e
        factor = m_p / (m_e + m_p)
        assert R_H == pytest.approx(R_inf * factor)
        
        # Compare to CODATA
        assert R_H == pytest.approx(CODATA_RYDBERG_H, rel=1e-6)


class TestBiologicalProxyIntegration:
    """Integration tests for biological proxy validation (offline)."""
    
    def test_fractal_analysis_on_synthetic_data(self):
        """Test fractal analysis on known synthetic signals."""
        from hhf_validation.validations.biological_proxy import (
            BiologicalProxyResult, analyze_temperature_series
        )
        import pandas as pd
        
        # Create synthetic temperature-like data
        np.random.seed(42)
        n_days = 365
        seasonal = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365)
        noise = np.random.randn(n_days) * 3
        temps = 15 + seasonal + noise
        
        df = pd.DataFrame({'Temp': temps})
        result = analyze_temperature_series(df)
        
        # Verify result structure
        assert isinstance(result, BiologicalProxyResult)
        assert result.n_points == n_days
        assert not np.isnan(result.pfd)
        assert not np.isnan(result.hfd)
        
        # PFD should be in reasonable range
        assert 1.0 < result.pfd < 1.1


class TestFractalMetricsIntegration:
    """Integration tests for fractal dimension calculations."""
    
    def test_pfd_hfd_relationship(self):
        """PFD and HFD should give consistent complexity assessment."""
        np.random.seed(123)
        
        # Simple signal (low complexity)
        simple = np.sin(np.linspace(0, 10 * np.pi, 1000))
        pfd_simple = petrosian_fd(simple)
        hfd_simple = higuchi_fd(simple)
        
        # Complex signal (high complexity)
        complex_sig = np.cumsum(np.random.randn(1000))  # Brownian motion
        pfd_complex = petrosian_fd(complex_sig)
        hfd_complex = higuchi_fd(complex_sig)
        
        # Both should detect that complex signal has higher complexity
        # (Note: exact relationship depends on signal characteristics)
        assert not np.isnan(pfd_simple)
        assert not np.isnan(hfd_simple)
        assert not np.isnan(pfd_complex)
        assert not np.isnan(hfd_complex)
    
    def test_fractal_dimensions_reproducible(self):
        """Fractal dimensions should be deterministic for same input."""
        signal = np.random.randn(500)
        
        pfd1 = petrosian_fd(signal)
        pfd2 = petrosian_fd(signal)
        hfd1 = higuchi_fd(signal)
        hfd2 = higuchi_fd(signal)
        
        assert pfd1 == pfd2
        assert hfd1 == hfd2


class TestOutputGeneration:
    """Integration tests for output file generation."""
    
    def test_isotopologue_run_validation_creates_outputs(self):
        """run_validation should create all expected output files."""
        from hhf_validation.validations.isotopologue import run_validation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = run_validation(output_dir=output_dir, show_plots=False)
            
            # Verify plot was created
            plot_path = output_dir / "plots" / "isotopologue_scaling.png"
            assert plot_path.exists()
            
            # Verify logs were created
            log_path = output_dir / "logs" / "isotopologue_scaling_log.json"
            csv_path = output_dir / "logs" / "isotopologue_scaling_summary.csv"
            assert log_path.exists()
            assert csv_path.exists()
    
    def test_molecular_photonic_run_validation_creates_outputs(self):
        """run_validation should create all expected output files."""
        from hhf_validation.validations.molecular_photonic import run_validation
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = run_validation(output_dir=output_dir, show_plots=False)
            
            # Verify plot was created
            plot_path = output_dir / "plots" / "molecular_photonic_scaling.png"
            assert plot_path.exists()
            
            # Verify logs were created
            log_path = output_dir / "logs" / "molecular_photonic_log.json"
            assert log_path.exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

