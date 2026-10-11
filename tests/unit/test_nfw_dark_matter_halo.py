"""Unit tests for the NFW Dark Matter Halo engine.

These tests verify mathematical correctness of the NFW profile calculations
using only the standard library, numpy, scipy, pydantic, fastapi, and httpx.
No mocks are used.
"""

import math
import numpy as np
import pytest
from scipy.integrate import quad

from src.astrogrid.engines.nfw_dark_matter_halo import (
    HaloProfileType,
    NFWHaloParameters,
    NFWHaloResult,
    NFWDarkMatterHaloEngine,
)


class TestHaloProfileType:
    """Tests for the HaloProfileType enum."""

    def test_nfw_value(self):
        assert HaloProfileType.NFW.value == "nfw"

    def test_isothermal_value(self):
        assert HaloProfileType.ISOTHERMAL.value == "isothermal"

    def test_plummer_value(self):
        assert HaloProfileType.PLUMMER.value == "plummer"

    def test_all_members(self):
        assert len(list(HaloProfileType)) == 3


class TestNFWHaloParameters:
    """Tests for NFWHaloParameters pydantic model."""

    def test_valid_parameters(self):
        params = NFWHaloParameters(mass=1e15, concentration=10.0)
        assert params.mass == 1e15
        assert params.concentration == 10.0

    def test_mass_must_be_positive(self):
        with pytest.raises(Exception):
            NFWHaloParameters(mass=0, concentration=10.0)

    def test_mass_negative_rejected(self):
        with pytest.raises(Exception):
            NFWHaloParameters(mass=-1e15, concentration=10.0)

    def test_concentration_must_be_positive(self):
        with pytest.raises(Exception):
            NFWHaloParameters(mass=1e15, concentration=0)

    def test_concentration_negative_rejected(self):
        with pytest.raises(Exception):
            NFWHaloParameters(mass=1e15, concentration=-5.0)


class TestNFWHaloResult:
    """Tests for NFWHaloResult pydantic model."""

    def test_valid_result(self):
        result = NFWHaloResult(
            radius=1.0,
            density=1e-6,
            enclosed_mass=1e12,
        )
        assert result.radius == 1.0
        assert result.density == 1e-6
        assert result.enclosed_mass == 1e12


class TestNFWDarkMatterHaloEngine:
    """Tests for the NFWDarkMatterHaloEngine mathematical calculations."""

    def _make_engine(self, mass=1e15, concentration=10.0):
        params = NFWHaloParameters(mass=mass, concentration=concentration)
        return NFWDarkMatterHaloEngine(params)

    def test_init_stores_parameters(self):
        engine = self._make_engine(mass=2e15, concentration=15.0)
        assert engine.params.mass == 2e15
        assert engine.params.concentration == 15.0

    def test_rho_0_positive(self):
        engine = self._make_engine()
        rho_0 = engine._compute_rho_0()
        assert rho_0 > 0

    def test_rho_0_scales_with_mass(self):
        """rho_0 should scale linearly with mass for fixed concentration."""
        engine1 = self._make_engine(mass=1e15, concentration=10.0)
        engine2 = self._make_engine(mass=2e15, concentration=10.0)
        rho_0_1 = engine1._compute_rho_0()
        rho_0_2 = engine2._compute_rho_0()
        assert abs(rho_0_2 / rho_0_1 - 2.0) < 1e-10

    def test_rho_0_decreases_with_concentration(self):
        """Higher concentration means more mass concentrated inward,
        so the characteristic density rho_0 should be higher."""
        engine_low_c = self._make_engine(mass=1e15, concentration=5.0)
        engine_high_c = self._make_engine(mass=1e15, concentration=20.0)
        rho_0_low = engine_low_c._compute_rho_0()
        rho_0_high = engine_high_c._compute_rho_0()
        assert rho_0_high > rho_0_low

    def test_density_positive_at_all_radii(self):
        """NFW density is positive for all r > 0."""
        engine = self._make_engine()
        radii = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
        for r in radii:
            result = engine.density(r)
            assert result.density > 0, f"Density should be positive at r={r}"

    def test_density_decreases_with_radius(self):
        """NFW density is monotonically decreasing with radius."""
        engine = self._make_engine()
        radii = np.linspace(0.1, 1000.0, 100)
        densities = [engine.density(r).density for r in radii]
        for i in range(1, len(densities)):
            assert densities[i] < densities[i - 1], (
                f"Density should decrease: r={radii[i]}, "
                f"rho={densities[i]} >= rho_prev={densities[i-1]}"
            )

    def test_density_at_small_radius_diverges(self):
        """As r -> 0, NFW density -> infinity (1/r behavior)."""
        engine = self._make_engine()
        rho_small = engine.density(0.001).density
        rho_large = engine.density(1.0).density
        assert rho_small > rho_large

    def test_density_large_radius_scaling(self):
        """At large r, NFW density scales as 1/r^3."""
        engine = self._make_engine()
        r1 = 1000.0
        r2 = 2000.0
        rho1 = engine.density(r1).density
        rho2 = engine.density(r2).density
        ratio = rho1 / rho2
        expected_ratio = (r2 / r1) ** 3
        assert abs(ratio - expected_ratio) / expected_ratio < 0.1, (
            f"At large r, rho should scale as 1/r^3. "
            f"Got ratio={ratio}, expected~{expected_ratio}"
        )

    def test_enclosed_mass_positive(self):
        """Enclosed mass is positive for r > 0."""
        engine = self._make_engine()
        for r in [0.1, 1.0, 10.0, 100.0]:
            result = engine.enclosed_mass(r)
            assert result.enclosed_mass > 0

    def test_enclosed_mass_increases_with_radius(self):
        """Enclosed mass is monotonically increasing with radius."""
        engine = self._make_engine()
        radii = np.linspace(0.1, 1000.0, 100)
        masses = [engine.enclosed_mass(r).enclosed_mass for r in radii]
        for i in range(1, len(masses)):
            assert masses[i] > masses[i - 1], (
                f"Enclosed mass should increase: r={radii[i]}, "
                f"M={masses[i]} <= M_prev={masses[i-1]}"
            )

    def test_enclosed_mass_bounded_by_total_mass(self):
        """Enclosed mass at any finite radius should not exceed total mass."""
        engine = self._make_engine(mass=1e15, concentration=10.0)
        total_mass = 1e15
        for r in [0.1, 1.0, 10.0, 100.0, 1000.0]:
            result = engine.enclosed_mass(r)
            assert result.enclosed_mass <= total_mass * (1 + 1e-10), (
                f"Enclosed mass at r={r} exceeds total mass: "
                f"{result.enclosed_mass} > {total_mass}"
            )

    def test_enclosed_mass_small_radius_is_small(self):
        """Enclosed mass at very small radius should be much less than total."""
        engine = self._make_engine(mass=1e15, concentration=10.0)
        m_small = engine.enclosed_mass(0.01).enclosed_mass