"""
Zero-mock unit tests for wcs_tangent_solver module.

Tests the mathematical correctness of the tangent-plane (gnomonic) projection
conversions between celestial coordinates (RA, Dec) and tangent-plane coordinates (x, y).
"""

import math
import pytest
import numpy as np
from pydantic import ValidationError

from src.astrogrid.engines.wcs_tangent_solver import (
    WcsTangentSolverInput,
    WcsTangentSolverOutput,
    _to_cartesian,
    _from_cartesian,
    solve_tangent_point,
    solve_inverse_tangent_point,
)


class TestToCartesian:
    """Tests for the _to_cartesian conversion function."""

    def test_north_pole(self):
        """RA=0, Dec=90 should map to (0, 0, 1)."""
        x, y, z = _to_cartesian(0.0, 90.0)
        assert math.isclose(x, 0.0, abs_tol=1e-10)
        assert math.isclose(y, 0.0, abs_tol=1e-10)
        assert math.isclose(z, 1.0, abs_tol=1e-10)

    def test_south_pole(self):
        """RA=0, Dec=-90 should map to (0, 0, -1)."""
        x, y, z = _to_cartesian(0.0, -90.0)
        assert math.isclose(x, 0.0, abs_tol=1e-10)
        assert math.isclose(y, 0.0, abs_tol=1e-10)
        assert math.isclose(z, -1.0, abs_tol=1e-10)

    def test_equator_ra0(self):
        """RA=0, Dec=0 should map to (1, 0, 0)."""
        x, y, z = _to_cartesian(0.0, 0.0)
        assert math.isclose(x, 1.0, abs_tol=1e-10)
        assert math.isclose(y, 0.0, abs_tol=1e-10)
        assert math.isclose(z, 0.0, abs_tol=1e-10)

    def test_equator_ra90(self):
        """RA=90, Dec=0 should map to (0, 1, 0)."""
        x, y, z = _to_cartesian(90.0, 0.0)
        assert math.isclose(x, 0.0, abs_tol=1e-10)
        assert math.isclose(y, 1.0, abs_tol=1e-10)
        assert math.isclose(z, 0.0, abs_tol=1e-10)

    def test_unit_vector_norm(self):
        """The resulting vector should always have unit norm."""
        for ra in [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]:
            for dec in [-90, -60, -30, 0, 30, 60, 90]:
                x, y, z = _to_cartesian(float(ra), float(dec))
                norm = math.sqrt(x**2 + y**2 + z**2)
                assert math.isclose(norm, 1.0, abs_tol=1e-10), (
                    f"Norm {norm} != 1 for ra={ra}, dec={dec}"
                )

    def test_known_angle(self):
        """RA=45, Dec=30 should produce correct Cartesian coordinates."""
        ra_rad = math.radians(45.0)
        dec_rad = math.radians(30.0)
        x, y, z = _to_cartesian(45.0, 30.0)
        expected_x = math.cos(dec_rad) * math.cos(ra_rad)
        expected_y = math.cos(dec_rad) * math.sin(ra_rad)
        expected_z = math.sin(dec_rad)
        assert math.isclose(x, expected_x, abs_tol=1e-10)
        assert math.isclose(y, expected_y, abs_tol=1e-10)
        assert math.isclose(z, expected_z, abs_tol=1e-10)


class TestFromCartesian:
    """Tests for the _from_cartesian conversion function."""

    def test_north_pole(self):
        """(0, 0, 1) should map to RA=0, Dec=90."""
        ra, dec = _from_cartesian(0.0, 0.0, 1.0)
        assert math.isclose(dec, 90.0, abs_tol=1e-10)

    def test_south_pole(self):
        """(0, 0, -1) should map to RA=0, Dec=-90."""
        ra, dec = _from_cartesian(0.0, 0.0, -1.0)
        assert math.isclose(dec, -90.0, abs_tol=1e-10)

    def test_equator_ra0(self):
        """(1, 0, 0) should map to RA=0, Dec=0."""
        ra, dec = _from_cartesian(1.0, 0.0, 0.0)
        assert math.isclose(ra, 0.0, abs_tol=1e-10)
        assert math.isclose(dec, 0.0, abs_tol=1e-10)

    def test_equator_ra90(self):
        """(0, 1, 0) should map to RA=90, Dec=0."""
        ra, dec = _from_cartesian(0.0, 1.0, 0.0)
        assert math.isclose(ra, 90.0, abs_tol=1e-10)
        assert math.isclose(dec, 0.0, abs_tol=1e-10)

    def test_roundtrip_identity(self):
        """Converting to Cartesian and back should recover original coordinates."""
        test_cases = [
            (0.0, 0.0),
            (45.0, 30.0),
            (120.0, -45.0),
            (270.0, 60.0),
            (330.0, -75.0),
            (180.0, 0.0),
            (90.0, 90.0),
            (270.0, -90.0),
        ]
        for ra, dec in test_cases:
            x, y, z = _to_cartesian(ra, dec)
            ra2, dec2 = _from_cartesian(x, y, z)
            # Handle RA wrap-around
            ra_diff = abs(ra - ra2)
            if ra_diff > 180:
                ra_diff = 360 - ra_diff
            assert math.isclose(ra_diff, 0.0, abs_tol=1e-8), (
                f"RA mismatch: {ra} vs {ra2}"
            )
            assert math.isclose(dec, dec2, abs_tol=1e-8), (
                f"Dec mismatch: {dec} vs {dec2}"
            )

    def test_ra_range(self):
        """RA should be in [0, 360) range."""
        for ra in [0, 45, 90, 135, 180, 225, 270, 315]:
            for dec in [-80, -40, 0, 40, 80]:
                x, y, z = _to_cartesian(float(ra), float(dec))
                ra_out, dec_out = _from_cartesian(x, y, z)
                assert 0.0 <= ra_out < 360.0, (
                    f"RA {ra_out} out of range for input ra={ra}, dec={dec}"
                )

    def test_dec_range(self):
        """Dec should be in [-90, 90] range."""
        for ra in [0, 45, 90, 135, 180, 225, 270, 315]:
            for dec in [-89, -45, 0, 45, 89]:
                x, y, z = _to_cartesian(float(ra), float(dec))