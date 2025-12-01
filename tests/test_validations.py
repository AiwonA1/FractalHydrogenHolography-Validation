"""
Tests for Validation Modules
============================
Unit tests for validation module calculations and consistency checks.
"""

import numpy as np
import pytest

from hhf_validation.utils.constants import (
    L_P, h, c, m_p, m_e, alpha,
    ISOTOPOLOGUE_DATA,
    compute_reduced_mass,
    compute_effective_frequency,
    compute_hhf_radius,
    compute_hhf_scaling_ratio
)


class TestPhysicalConstants:
    """Tests for physical constant values."""
    
    def test_planck_length_order_of_magnitude(self):
        """Planck length should be ~10^-35 m."""
        assert 1e-36 < L_P < 1e-34
    
    def test_speed_of_light_exact(self):
        """Speed of light is exact in SI: 299792458 m/s."""
        assert c == 299792458
    
    def test_proton_heavier_than_electron(self):
        """Proton mass > electron mass by ~1836x."""
        ratio = m_p / m_e
        assert 1835 < ratio < 1837
    
    def test_fine_structure_constant_value(self):
        """Fine structure constant ≈ 1/137."""
        inverse = 1 / alpha
        assert 137.0 < inverse < 137.1


class TestIsotopologueData:
    """Tests for isotopologue spectroscopic data."""
    
    def test_all_isotopes_present(self):
        """Should have data for H2, D2, T2, HD."""
        assert 'H2' in ISOTOPOLOGUE_DATA
        assert 'D2' in ISOTOPOLOGUE_DATA
        assert 'T2' in ISOTOPOLOGUE_DATA
        assert 'HD' in ISOTOPOLOGUE_DATA
    
    def test_h2_frequency_highest(self):
        """H2 should have highest vibrational frequency (lightest)."""
        h2_omega = ISOTOPOLOGUE_DATA['H2']['omega_e']
        d2_omega = ISOTOPOLOGUE_DATA['D2']['omega_e']
        t2_omega = ISOTOPOLOGUE_DATA['T2']['omega_e']
        
        assert h2_omega > d2_omega > t2_omega
    
    def test_mass_ordering(self):
        """Mass should increase: H2 < HD < D2 < T2."""
        h2_mass = ISOTOPOLOGUE_DATA['H2']['mass_1']
        d2_mass = ISOTOPOLOGUE_DATA['D2']['mass_1']
        t2_mass = ISOTOPOLOGUE_DATA['T2']['mass_1']
        
        assert h2_mass < d2_mass < t2_mass


class TestReducedMass:
    """Tests for reduced mass calculation."""
    
    def test_equal_masses(self):
        """For equal masses, reduced mass = m/2."""
        result = compute_reduced_mass(2.0, 2.0)
        assert result == pytest.approx(1.0)
    
    def test_one_infinite_mass(self):
        """For m2 >> m1, reduced mass ≈ m1."""
        result = compute_reduced_mass(1.0, 1e10)
        assert result == pytest.approx(1.0, rel=1e-6)
    
    def test_h2_reduced_mass(self):
        """H2 reduced mass should be ~0.5 u."""
        data = ISOTOPOLOGUE_DATA['H2']
        mu = compute_reduced_mass(data['mass_1'], data['mass_2'])
        assert 0.5 < mu < 0.51


class TestEffectiveFrequency:
    """Tests for anharmonic-corrected frequency."""
    
    def test_formula_correctness(self):
        """omega_eff = omega_e - 2*omega_exe."""
        omega_e = 4000
        omega_exe = 100
        result = compute_effective_frequency(omega_e, omega_exe)
        assert result == 3800
    
    def test_h2_effective_frequency(self):
        """H2 effective frequency should be ~4158 cm^-1."""
        data = ISOTOPOLOGUE_DATA['H2']
        omega_eff = compute_effective_frequency(data['omega_e'], data['omega_exe'])
        assert 4100 < omega_eff < 4200


class TestHHFConstants:
    """Tests for HHF theoretical constants."""
    
    def test_hhf_radius_order_of_magnitude(self):
        """HHF radius should be ~10^-13 m (proton scale)."""
        R_HHF = compute_hhf_radius()
        assert 1e-14 < R_HHF < 1e-12
    
    def test_hhf_scaling_ratio_order_of_magnitude(self):
        """R_HHF / L_P should be ~10^22."""
        ratio = compute_hhf_scaling_ratio()
        assert 1e21 < ratio < 1e23
    
    def test_scaling_ratio_self_consistency(self):
        """Scaling ratio should equal R_HHF / L_P."""
        R_HHF = compute_hhf_radius()
        ratio = compute_hhf_scaling_ratio()
        
        assert ratio == pytest.approx(R_HHF / L_P)


class TestIsotopologueScaling:
    """Tests for isotopologue scaling validation logic."""
    
    def test_import_isotopologue_module(self):
        """Should be able to import isotopologue validation."""
        from hhf_validation.validations import isotopologue
        assert hasattr(isotopologue, 'run_isotopologue_analysis')
    
    def test_lambda_hh_h2_equals_one(self):
        """Lambda_HH for H2 (reference) should be exactly 1.0."""
        from hhf_validation.validations.isotopologue import run_isotopologue_analysis
        
        result = run_isotopologue_analysis()
        assert result.isotopologues['H2'].lambda_hh == 1.0
    
    def test_all_lambda_hh_near_unity(self):
        """All Lambda_HH values should be within threshold of 1.0."""
        from hhf_validation.validations.isotopologue import run_isotopologue_analysis
        
        result = run_isotopologue_analysis(threshold=0.1)
        
        for iso, data in result.isotopologues.items():
            assert abs(data.lambda_hh - 1.0) < 0.1, f"{iso}: Lambda_HH = {data.lambda_hh}"


class TestMolecularPhotonicValidation:
    """Tests for molecular/photonic validation."""
    
    def test_rydberg_constant_accuracy(self):
        """Calculated Rydberg should match CODATA within tolerance."""
        from hhf_validation.validations.molecular_photonic import (
            compute_hydrogen_rydberg,
            CODATA_RYDBERG_H
        )
        
        R_H = compute_hydrogen_rydberg()
        relative_error = abs(R_H - CODATA_RYDBERG_H) / CODATA_RYDBERG_H
        
        # Should match within 1 ppm
        assert relative_error < 1e-6
    
    def test_spectral_validation_passes(self):
        """Spectral consistency check should pass."""
        from hhf_validation.validations.molecular_photonic import (
            validate_spectral_consistency
        )
        
        result = validate_spectral_consistency()
        assert result.validated


class TestBiologicalProxyValidation:
    """Tests for biological proxy validation structures."""
    
    def test_result_dataclass_serialization(self):
        """BiologicalProxyResult should serialize to dict/JSON."""
        from hhf_validation.validations.biological_proxy import BiologicalProxyResult
        
        result = BiologicalProxyResult(
            dataset_name="test",
            n_points=100,
            pfd=1.02,
            hfd=0.85
        )
        
        d = result.to_dict()
        assert d['dataset_name'] == 'test'
        assert d['n_points'] == 100
        assert d['pfd'] == 1.02
        assert d['hfd'] == 0.85
        
        j = result.to_json()
        assert 'test' in j
        assert '1.02' in j


class TestPEFFValidation:
    """Tests for PEFF validation structures."""
    
    def test_seismic_result_structure(self):
        """SeismicResult should have expected fields."""
        from hhf_validation.validations.peff_seismic_eeg import SeismicResult
        
        result = SeismicResult(n_points=1000, pfd=1.025)
        assert result.n_points == 1000
        assert result.pfd == 1.025
    
    def test_peff_result_to_dict(self):
        """PEFFValidationResult should serialize properly."""
        from hhf_validation.validations.peff_seismic_eeg import (
            PEFFValidationResult,
            SeismicResult
        )
        
        result = PEFFValidationResult(
            seismic=SeismicResult(n_points=500, pfd=1.03)
        )
        
        d = result.to_dict()
        assert 'seismic' in d
        assert d['seismic']['n_points'] == 500


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

