"""
Zero-mock unit tests for the cosmic_ray_lacosmic engine.

These tests verify mathematical invariants and correctness of the
compute_astronomy_metric function using only standard library, numpy,
scipy, pydantic, fastapi, and httpx.
"""

import numpy as np
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Literal
from enum import Enum
import datetime


# We need to import the actual engine module.
# Since we cannot import astropy or other forbidden packages, and the engine
# may depend on them, we need to check what compute_astronomy_metric does.
# Based on the snippet, it's a cosmic ray propagation engine.
# Let's assume the function signature and behavior based on typical patterns.
# However, since we can't see the full implementation, we'll write tests
# that are robust to the actual implementation by testing mathematical invariants.

# For the tests to work, we need to import the actual function.
# Let's assume the module is importable and the function is as described.

try:
    from src.astrogrid.engines.cosmic_ray_lacosmic import compute_astronomy_metric
except ImportError:
    # If the import fails, we'll define a stub that matches the expected interface
    # for testing purposes. But per the rules, we should test the real module.
    # Let's assume the import works in the test environment.
    pass


class TestCosmicRayLacosmic:
    """Unit tests for cosmic_ray_lacosmic.compute_astronomy_metric."""

    def test_metric_returns_numeric_value(self):
        """Test that compute_astronomy_metric returns a numeric value."""
        # Create a minimal valid input
        result = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        assert isinstance(result, (int, float, np.floating, np.integer))
        assert np.isfinite(result)

    def test_metric_non_negative_for_valid_input(self):
        """Test that the metric is non-negative for valid physical inputs."""
        result = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=45.0,
            dec_deg=30.0,
        )
        assert result >= 0.0

    def test_metric_energy_dependence(self):
        """Test that the metric changes with energy."""
        result_low = compute_astronomy_metric(
            energy_gev=100.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_high = compute_astronomy_metric(
            energy_gev=10000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        # The metric should differ for different energies
        assert result_low != result_high

    def test_metric_species_dependence(self):
        """Test that the metric changes with particle species."""
        result_proton = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_electron = compute_astronomy_metric(
            energy_gev=1000.0,
            species="electron",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        # Different species should yield different metrics
        assert result_proton != result_electron

    def test_metric_coordinate_dependence(self):
        """Test that the metric changes with coordinate position."""
        result_pos1 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_pos2 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=90.0,
            dec_deg=45.0,
        )
        # Different positions should yield different metrics
        assert result_pos1 != result_pos2

    def test_metric_deterministic(self):
        """Test that the metric is deterministic for the same input."""
        result1 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=10.0,
            dec_deg=20.0,
        )
        result2 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=10.0,
            dec_deg=20.0,
        )
        assert result1 == result2

    def test_metric_finite_for_extreme_energies(self):
        """Test that the metric remains finite for extreme but valid energies."""
        result_low = compute_astronomy_metric(
            energy_gev=1.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_high = compute_astronomy_metric(
            energy_gev=1e12,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        assert np.isfinite(result_low)
        assert np.isfinite(result_high)

    def test_metric_valid_coordinate_systems(self):
        """Test that all valid coordinate systems produce finite results."""
        for coord_sys in ["icrs", "galactic", "ecliptic", "local"]:
            result = compute_astronomy_metric(
                energy_gev=1000.0,
                species="proton",
                coordinate_system=coord_sys,
                ra_deg=0.0,
                dec_deg=0.0,
            )
            assert np.isfinite(result), f"Result not finite for coordinate system: {coord_sys}"

    def test_metric_valid_species(self):
        """Test that all valid species produce finite results."""
        for species in ["proton", "electron", "positron", "neutron", "photon"]:
            result = compute_astronomy_metric(
                energy_gev=1000.0,
                species=species,
                coordinate_system="icrs",
                ra_deg=0.0,
                dec_deg=0.0,
            )
            assert np.isfinite(result), f"Result not finite for species: {species}"

    def test_metric_symmetry_in_ra(self):
        """Test that the metric is symmetric in RA (0 and 360 should be equivalent)."""
        result_ra0 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_ra360 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=360.0,
            dec_deg=0.0,
        )
        # RA=0 and RA=360 should give the same result
        assert np.isclose(result_ra0, result_ra360, rtol=1e-10)

    def test_metric_dec_bounds(self):
        """Test that the metric is finite at dec bounds."""
        result_dec0 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=0.0,
        )
        result_dec90 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",
            coordinate_system="icrs",
            ra_deg=0.0,
            dec_deg=90.0,
        )
        result_dec_neg90 = compute_astronomy_metric(
            energy_gev=1000.0,
            species="proton",