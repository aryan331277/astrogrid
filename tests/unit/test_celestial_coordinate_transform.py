"""Zero-mock unit tests for celestial_coordinate_transform module.

Tests mathematical invariants and correctness of coordinate transformations
between ICRF, GCRS, and ITRF coordinate systems.
"""

import numpy as np
import pytest
from datetime import datetime, timezone

from src.astrogrid.engines.celestial_coordinate_transform import (
    CoordinateSystem,
    CelestialCoordinate,
    TransformResult,
    _julian_day,
    _julian_century,
    _precession_matrix,
    _nutation_angles,
    _earth_rotation_angle,
    _icrf_to_gcrs,
    _gcrs_to_itrf,
    transform_coordinates,
)


class TestJulianDay:
    """Tests for Julian Day calculation."""

    def test_known_julian_day_2000_jan1_12h(self):
        """J2000.0 epoch: 2000-01-01 12:00:00 UTC should give JD = 2451545.0."""
        dt = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        jd = _julian_day(dt)
        assert abs(jd - 2451545.0) < 1e-6

    def test_known_julian_day_2000_jan1_0h(self):
        """2000-01-01 00:00:00 UTC should give JD = 2451544.5."""
        dt = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        jd = _julian_day(dt)
        assert abs(jd - 2451544.5) < 1e-6

    def test_julian_day_monotonically_increasing(self):
        """Julian Day should increase with time."""
        dt1 = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2000, 1, 2, 0, 0, 0, tzinfo=timezone.utc)
        jd1 = _julian_day(dt1)
        jd2 = _julian_day(dt2)
        assert jd2 > jd1
        assert abs(jd2 - jd1 - 1.0) < 1e-6

    def test_julian_day_half_day_offset(self):
        """12 hours apart should give exactly 0.5 JD difference."""
        dt1 = datetime(2000, 6, 15, 0, 0, 0, tzinfo=timezone.utc)
        dt2 = datetime(2000, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        jd1 = _julian_day(dt1)
        jd2 = _julian_day(dt2)
        assert abs(jd2 - jd1 - 0.5) < 1e-6


class TestJulianCentury:
    """Tests for Julian Century calculation."""

    def test_j2000_epoch_gives_zero(self):
        """At J2000.0 (JD=2451545.0), T should be 0.0."""
        t = _julian_century(2451545.0)
        assert abs(t) < 1e-10

    def test_one_century_later(self):
        """36525 days after J2000 should give T = 1.0."""
        jd = 2451545.0 + 36525.0
        t = _julian_century(jd)
        assert abs(t - 1.0) < 1e-6

    def test_half_century(self):
        """18262.5 days after J2000 should give T = 0.5."""
        jd = 2451545.0 + 18262.5
        t = _julian_century(jd)
        assert abs(t - 0.5) < 1e-6

    def test_negative_t_before_j2000(self):
        """Before J2000, T should be negative."""
        jd = 2451545.0 - 3652.5
        t = _julian_century(jd)
        assert t < 0
        assert abs(t - (-0.1)) < 1e-6


class TestPrecessionMatrix:
    """Tests for precession matrix properties."""

    def test_identity_at_j2000(self):
        """Precession matrix at T=0 should be identity."""
        t = 0.0
        m = _precession_matrix(t)
        expected = np.eye(3)
        np.testing.assert_allclose(m, expected, atol=1e-10)

    def test_orthogonality(self):
        """Precession matrix should be orthogonal (rotation matrix)."""
        t = 0.5
        m = _precession_matrix(t)
        product = m @ m.T
        np.testing.assert_allclose(product, np.eye(3), atol=1e-8)

    def test_determinant_is_one(self):
        """Precession matrix should have determinant 1 (proper rotation)."""
        t = 0.3
        m = _precession_matrix(t)
        det = np.linalg.det(m)
        assert abs(det - 1.0) < 1e-8

    def test_preserves_vector_norm(self):
        """Precession should preserve vector length."""
        t = 0.2
        m = _precession_matrix(t)
        v = np.array([1.0, 2.0, 3.0])
        v_rotated = m @ v
        assert abs(np.linalg.norm(v_rotated) - np.linalg.norm(v)) < 1e-10

    def test_non_identity_for_nonzero_t(self):
        """Precession matrix should differ from identity for T != 0."""
        t = 0.1
        m = _precession_matrix(t)
        assert not np.allclose(m, np.eye(3), atol=1e-10)


class TestNutationAngles:
    """Tests for nutation angle calculations."""

    def test_returns_three_angles(self):
        """Nutation should return three angles."""
        t = 0.0
        result = _nutation_angles(t)
        assert len(result) == 3

    def test_angles_are_finite(self):
        """All nutation angles should be finite."""
        t = 0.5
        result = _nutation_angles(t)
        for angle in result:
            assert np.isfinite(angle)

    def test_angles_small_magnitude(self):
        """Nutation angles should be small (arcseconds to arcminutes range)."""
        t = 0.0
        result = _nutation_angles(t)
        for angle in result:
            assert abs(angle) < 1e-3

    def test_deterministic(self):
        """Same input should give same output."""
        t = 0.25
        r1 = _nutation_angles(t)
        r2 = _nutation_angles(t)
        for a, b in zip(r1, r2):
            assert abs(a - b) < 1e-15


class TestEarthRotationAngle:
    """Tests for Earth Rotation Angle (ERA) calculation."""

    def test_era_in_valid_range(self):
        """ERA should be in [0, 2*pi) range."""
        jd = 2451545.0
        era = _earth_rotation_angle(jd)
        assert 0 <= era < 2 * np.pi

    def test_era_increases_with_time(self):
        """ERA should increase with time."""
        jd1 = 2451545.0
        jd2 = jd1 + 0.5
        era1 = _earth_rotation_angle(jd1)
        era2 = _earth_rotation_angle(jd2)
        diff = (era2 - era1) % (2 * np.pi)
        assert abs(diff - np.pi) < 0.1

    def test_era_deterministic(self):
        """Same JD should give same ERA."""
        jd = 2451545.0
        era1 = _earth_rotation_angle(jd)
        era2 = _earth_rotation_angle(jd)
        assert abs(era1 - era2) < 1