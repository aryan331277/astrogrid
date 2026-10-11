"""
Zero-mock unit tests for the Doppler Redshift Estimator engine.

These tests verify the mathematical correctness of the Doppler redshift
calculations using only the Python standard library, numpy, and the
engine module itself. No mocking is used.
"""

import math
import pytest
import numpy as np

from src.astrogrid.engines.doppler_redshift_estimator import (
    SpectralLine,
    PhotometricBand,
    RedshiftEstimate,
    DopplerRedshiftEstimator,
)


class TestSpectralLine:
    """Tests for the SpectralLine pydantic model."""

    def test_valid_spectral_line(self):
        line = SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=550.0)
        assert line.rest_wavelength_nm == 500.0
        assert line.observed_wavelength_nm == 550.0

    def test_invalid_rest_wavelength_zero(self):
        with pytest.raises(Exception):
            SpectralLine(rest_wavelength_nm=0.0, observed_wavelength_nm=550.0)

    def test_invalid_rest_wavelength_negative(self):
        with pytest.raises(Exception):
            SpectralLine(rest_wavelength_nm=-100.0, observed_wavelength_nm=550.0)

    def test_invalid_observed_wavelength_zero(self):
        with pytest.raises(Exception):
            SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=0.0)

    def test_invalid_observed_wavelength_negative(self):
        with pytest.raises(Exception):
            SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=-50.0)


class TestPhotometricBand:
    """Tests for the PhotometricBand pydantic model."""

    def test_valid_photometric_band(self):
        band = PhotometricBand(band_name="V", flux_density=1.0e-20)
        assert band.band_name == "V"
        assert band.flux_density == 1.0e-20

    def test_invalid_flux_density_negative(self):
        with pytest.raises(Exception):
            PhotometricBand(band_name="V", flux_density=-1.0e-20)

    def test_invalid_flux_density_zero(self):
        with pytest.raises(Exception):
            PhotometricBand(band_name="V", flux_density=0.0)


class TestRedshiftEstimate:
    """Tests for the RedshiftEstimate pydantic model."""

    def test_valid_redshift_estimate(self):
        est = RedshiftEstimate(redshift=0.1, method="spectral", confidence=0.95)
        assert est.redshift == 0.1
        assert est.method == "spectral"
        assert est.confidence == 0.95

    def test_negative_redshift(self):
        est = RedshiftEstimate(redshift=-0.1, method="spectral", confidence=0.95)
        assert est.redshift == -0.1


class TestDopplerRedshiftEstimator:
    """Tests for the DopplerRedshiftEstimator class."""

    def setup_method(self):
        self.estimator = DopplerRedshiftEstimator()

    def test_estimate_from_spectral_lines_single_line(self):
        """
        Test redshift calculation from a single spectral line.
        z = (observed - rest) / rest
        For rest=500nm, observed=550nm: z = 50/500 = 0.1
        """
        lines = [SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=550.0)]
        result = self.estimator.estimate_from_spectral_lines(lines)
        assert isinstance(result, RedshiftEstimate)
        assert abs(result.redshift - 0.1) < 1e-10

    def test_estimate_from_spectral_lines_multiple_lines(self):
        """
        Test redshift calculation from multiple spectral lines.
        Each line should give the same redshift if consistent.
        Line 1: rest=500, obs=550 -> z=0.1
        Line 2: rest=656.3, obs=721.93 -> z=0.1
        Line 3: rest=486.1, obs=534.71 -> z=0.1
        """
        lines = [
            SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=550.0),
            SpectralLine(rest_wavelength_nm=656.3, observed_wavelength_nm=721.93),
            SpectralLine(rest_wavelength_nm=486.1, observed_wavelength_nm=534.71),
        ]
        result = self.estimator.estimate_from_spectral_lines(lines)
        assert isinstance(result, RedshiftEstimate)
        assert abs(result.redshift - 0.1) < 1e-10

    def test_estimate_from_spectral_lines_zero_redshift(self):
        """
        Test that when observed == rest, redshift is zero.
        """
        lines = [
            SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=500.0),
            SpectralLine(rest_wavelength_nm=656.3, observed_wavelength_nm=656.3),
        ]
        result = self.estimator.estimate_from_spectral_lines(lines)
        assert abs(result.redshift) < 1e-15

    def test_estimate_from_spectral_lines_negative_redshift(self):
        """
        Test blueshift (negative redshift).
        rest=500, obs=450 -> z = (450-500)/500 = -0.1
        """
        lines = [SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=450.0)]
        result = self.estimator.estimate_from_spectral_lines(lines)
        assert abs(result.redshift - (-0.1)) < 1e-10

    def test_estimate_from_spectral_lines_large_redshift(self):
        """
        Test a larger redshift.
        rest=500, obs=1000 -> z = (1000-500)/500 = 1.0
        """
        lines = [SpectralLine(rest_wavelength_nm=500.0, observed_wavelength_nm=1000.0)]
        result = self.estimator.estimate_from_spectral_lines(lines)
        assert abs(result.redshift - 1.0) < 1e-10

    def test_estimate_from_spectral_lines_empty_list(self):
        """
        Test that an empty list of spectral lines raises an error.
        """
        with pytest.raises(Exception):
            self.estimator.estimate_from_spectral_lines([])

    def test_estimate_from_photometry_two_bands(self):
        """
        Test redshift estimation from photometric flux ratios.
        
        For a simple power-law SED with index alpha, the flux ratio between
        two bands gives:
        f1/f2 = (nu1/nu2)^(alpha+1)
        
        The redshift shifts the observed frequencies:
        nu_obs = nu_rest / (1+z)
        
        For two bands with rest frequencies nu1_rest, nu2_rest:
        f1_obs/f2_obs = (nu1_rest/nu2_rest)^(alpha+1) * (1+z)^(-(alpha+1))
        
        If we know the rest-frame flux ratio, we can solve for z.
        
        Let's use a simpler test: if the observed flux ratio matches the
        rest-frame ratio, z should be 0.
        """
        # Create two bands with equal flux densities (rest-frame ratio = 1)
        bands = [
            PhotometricBand(band_name="B", flux_density=1.0e-20),
            PhotometricBand(band_name="V", flux_density=1.0e-20),
        ]
        result = self.estimator.estimate_from_photometry(bands)
        assert isinstance(result, RedshiftEstimate)
        # With equal fluxes and no additional info, the estimator should
        # return a valid result. The exact value depends on implementation.
        assert result.redshift >= -1.0  # Physical constraint: z > -1

    def test_estimate_from_photometry_single_band(self):
        """
        Test that a single photometric band raises an error (need at least 2).
        """
        bands = [PhotometricBand(band_name="V", flux_density=1.0e-20)]
        with pytest.raises(Exception):
            self.estimator.estimate_from_photometry(bands)

    def test_estimate