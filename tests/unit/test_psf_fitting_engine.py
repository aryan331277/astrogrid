"""Zero-mock unit tests for psf_fitting_engine module.

Tests mathematical invariants, numerical correctness, and structural
properties of the PSF fitting engine without any mocking.
"""

import numpy as np
import pytest

from src.astrogrid.engines.psf_fitting_engine import (
    PSFModelType,
    PSFFitResult,
    PSFFittingEngine,
)


class TestPSFModelType:
    """Tests for the PSFModelType enum."""

    def test_enum_values_exist(self):
        """Verify all expected PSF model types are defined."""
        assert hasattr(PSFModelType, "GAUSSIAN")
        assert hasattr(PSFModelType, "MOG")
        assert hasattr(PSFModelType, "KRON")

    def test_enum_is_string_based(self):
        """Verify PSFModelType inherits from str."""
        assert issubclass(PSFModelType, str)

    def test_enum_values_are_strings(self):
        """Verify each enum member has a string value."""
        for member in PSFModelType:
            assert isinstance(member.value, str)
            assert len(member.value) > 0


class TestPSFFitResult:
    """Tests for the PSFFitResult pydantic model."""

    def test_basic_construction(self):
        """Verify PSFFitResult can be constructed with valid fields."""
        result = PSFFitResult(
            model_type=PSFModelType.GAUSSIAN,
            flux=100.0,
            fwhm=2.5,
            chi2_reduced=1.05,
            n_iterations=42,
            converged=True,
        )
        assert result.model_type == PSFModelType.GAUSSIAN
        assert result.flux == 100.0
        assert result.fwhm == 2.5
        assert result.chi2_reduced == 1.05
        assert result.n_iterations == 42
        assert result.converged is True

    def test_flux_zero_value(self):
        """Verify that zero flux is accepted."""
        result = PSFFitResult(
            model_type=PSFModelType.GAUSSIAN,
            flux=0.0,
            fwhm=1.0,
            chi2_reduced=1.0,
            n_iterations=1,
            converged=True,
        )
        assert result.flux == 0.0


class TestGaussianPSF:
    """Tests for the _gaussian_psf method."""

    def setup_method(self):
        """Create a PSFFittingEngine instance for each test."""
        self.engine = PSFFittingEngine()

    def test_gaussian_psf_shape(self):
        """Verify the output shape matches the input image shape."""
        image_shape = (32, 32)
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=(16.0, 16.0),
            fwhm=4.0,
            flux=1.0,
        )
        assert psf.shape == image_shape

    def test_gaussian_psf_non_negative(self):
        """Verify all PSF values are non-negative."""
        image_shape = (64, 64)
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=(32.0, 32.0),
            fwhm=5.0,
            flux=100.0,
        )
        assert np.all(psf >= 0.0)

    def test_gaussian_psf_peak_at_center(self):
        """Verify the maximum value is at or near the center."""
        image_shape = (51, 51)
        center = (25.0, 25.0)
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=3.0,
            flux=1.0,
        )
        max_idx = np.unravel_index(np.argmax(psf), psf.shape)
        assert max_idx[0] == int(center[0])
        assert max_idx[1] == int(center[1])

    def test_gaussian_psf_flux_conservation(self):
        """Verify the total flux approximates the input flux."""
        image_shape = (256, 256)
        center = (128.0, 128.0)
        fwhm = 8.0
        flux = 1000.0
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=fwhm,
            flux=flux,
        )
        total_flux = np.sum(psf)
        assert abs(total_flux - flux) / flux < 0.05

    def test_gaussian_psf_symmetry(self):
        """Verify the PSF is symmetric about the center."""
        image_shape = (65, 65)
        center = (32.0, 32.0)
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=4.0,
            flux=1.0,
        )
        cy, cx = int(center[0]), int(center[1])
        for i in range(0, 10):
            for j in range(0, 10):
                mirror_i = 2 * cy - i
                mirror_j = 2 * cx - j
                if 0 <= mirror_i < image_shape[0] and 0 <= mirror_j < image_shape[1]:
                    assert np.isclose(psf[i, j], psf[mirror_i, mirror_j], rtol=1e-10)

    def test_gaussian_psf_zero_flux(self):
        """Verify a zero-flux PSF is all zeros."""
        image_shape = (32, 32)
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=(16.0, 16.0),
            fwhm=3.0,
            flux=0.0,
        )
        assert np.all(psf == 0.0)

    def test_gaussian_psf_fwhm_relationship(self):
        """Verify the FWHM is consistent with the Gaussian sigma."""
        image_shape = (129, 129)
        center = (64.0, 64.0)
        fwhm = 6.0
        psf = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=fwhm,
            flux=1.0,
        )
        profile = psf[64, :]
        peak = np.max(profile)
        half_max = peak / 2.0
        left_crossings = np.where(profile[:64] >= half_max)[0]
        right_crossings = np.where(profile[65:] >= half_max)[0]
        if len(left_crossings) > 0 and len(right_crossings) > 0:
            left_edge = left_crossings[0]
            right_edge = 65 + right_crossings[-1]
            measured_fwhm = right_edge - left_edge
            assert abs(measured_fwhm - fwhm) / fwhm < 0.15

    def test_gaussian_psf_larger_fwhm_spreads_more(self):
        """Verify that a larger FWHM produces a broader PSF."""
        image_shape = (129, 129)
        center = (64.0, 64.0)
        flux = 1.0

        psf_narrow = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=3.0,
            flux=flux,
        )
        psf_wide = self.engine._gaussian_psf(
            image_shape=image_shape,
            center=center,
            fwhm=10.0,
            flux=flux,
        )

        # The wide PSF should have lower peak but broader extent
        assert np.max(psf_wide) < np.max(psf_narrow)

        # Measure the second moment (variance) along one axis
        y, x = np.mgrid[0:image_shape[0], 0:image_shape[1]]
        total = np.sum(psf_narrow)
        if total > 0:
            var_narrow = np.sum((x - 64.0) ** 2 * psf_narrow) / total
            total_wide = np.sum(psf_wide)