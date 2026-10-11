"""Zero-mock unit tests for radial_velocity_solver module."""

import numpy as np
import pytest
from datetime import datetime
from typing import Optional

from src.astrogrid.engines.radial_velocity_solver import (
    CoordinateSystem,
    RadialVelocityInput,
    RadialVelocityResult,
    RadialVelocitySolver,
)


class TestCoordinateSystem:
    """Tests for the CoordinateSystem enum."""

    def test_icrs_value(self):
        assert CoordinateSystem.ICRS.value == "ICRS"

    def test_gcrs_value(self):
        assert CoordinateSystem.GCRS.value == "GCRS"

    def test_bcrs_value(self):
        assert CoordinateSystem.BCRS.value == "BCRS"

    def test_all_members(self):
        assert len(list(CoordinateSystem)) == 3


class TestRadialVelocityInput:
    """Tests for RadialVelocityInput pydantic model."""

    def test_valid_input_creation(self):
        """Test creating a valid RadialVelocityInput."""
        inp = RadialVelocityInput(
            ra_deg=45.0,
            dec_deg=30.0,
            velocity_x=100.0,
            velocity_y=200.0,
            velocity_z=300.0,
            coordinate_system=CoordinateSystem.ICRS,
            observation_time=datetime(2024, 1, 1, 12, 0, 0),
        )
        assert inp.ra_deg == 45.0
        assert inp.dec_deg == 30.0
        assert inp.velocity_x == 100.0
        assert inp.velocity_y == 200.0
        assert inp.velocity_z == 300.0
        assert inp.coordinate_system == CoordinateSystem.ICRS
        assert inp.observation_time == datetime(2024, 1, 1, 12, 0, 0)

    def test_validate_velocity_vector_zero(self):
        """Test that a zero velocity vector is valid."""
        inp = RadialVelocityInput(
            ra_deg=0.0,
            dec_deg=0.0,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            coordinate_system=CoordinateSystem.ICRS,
            observation_time=datetime(2024, 1, 1, 12, 0, 0),
        )
        # Should not raise
        inp.validate_velocity_vector()

    def test_validate_velocity_vector_nonzero(self):
        """Test that a non-zero velocity vector is valid."""
        inp = RadialVelocityInput(
            ra_deg=0.0,
            dec_deg=0.0,
            velocity_x=100.0,
            velocity_y=200.0,
            velocity_z=300.0,
            coordinate_system=CoordinateSystem.ICRS,
            observation_time=datetime(2024, 1, 1, 12, 0, 0),
        )
        # Should not raise
        inp.validate_velocity_vector()

    def test_validate_velocity_vector_negative_components(self):
        """Test that negative velocity components are valid."""
        inp = RadialVelocityInput(
            ra_deg=0.0,
            dec_deg=0.0,
            velocity_x=-100.0,
            velocity_y=-200.0,
            velocity_z=-300.0,
            coordinate_system=CoordinateSystem.ICRS,
            observation_time=datetime(2024, 1, 1, 12, 0, 0),
        )
        # Should not raise
        inp.validate_velocity_vector()


class TestRadialVelocityResult:
    """Tests for RadialVelocityResult pydantic model."""

    def test_valid_result_creation(self):
        """Test creating a valid RadialVelocityResult."""
        result = RadialVelocityResult(
            radial_velocity=150.5,
            barycentric_correction=10.2,
            total_velocity=160.7,
            coordinate_system=CoordinateSystem.ICRS,
        )
        assert result.radial_velocity == 150.5
        assert result.barycentric_correction == 10.2
        assert result.total_velocity == 160.7
        assert result.coordinate_system == CoordinateSystem.ICRS


class TestRadialVelocitySolver:
    """Tests for RadialVelocitySolver class."""

    def setup_method(self):
        """Set up solver instance for each test."""
        self.solver = RadialVelocitySolver()

    def test_deg_to_rad_zero(self):
        """Test degree to radian conversion for 0 degrees."""
        assert self.solver._deg_to_rad(0.0) == pytest.approx(0.0, abs=1e-15)

    def test_deg_to_rad_90(self):
        """Test degree to radian conversion for 90 degrees."""
        assert self.solver._deg_to_rad(90.0) == pytest.approx(np.pi / 2, abs=1e-15)

    def test_deg_to_rad_180(self):
        """Test degree to radian conversion for 180 degrees."""
        assert self.solver._deg_to_rad(180.0) == pytest.approx(np.pi, abs=1e-15)

    def test_deg_to_rad_360(self):
        """Test degree to radian conversion for 360 degrees."""
        assert self.solver._deg_to_rad(360.0) == pytest.approx(2 * np.pi, abs=1e-15)

    def test_deg_to_rad_45(self):
        """Test degree to radian conversion for 45 degrees."""
        assert self.solver._deg_to_rad(45.0) == pytest.approx(np.pi / 4, abs=1e-15)

    def test_deg_to_rad_negative(self):
        """Test degree to radian conversion for negative degrees."""
        assert self.solver._deg_to_rad(-90.0) == pytest.approx(-np.pi / 2, abs=1e-15)

    def test_unit_vector_from_ra_dec_north_pole(self):
        """Test unit vector for RA=0, Dec=90 (North Celestial Pole)."""
        vec = self.solver._unit_vector_from_ra_dec(0.0, 90.0)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec_south_pole(self):
        """Test unit vector for RA=0, Dec=-90 (South Celestial Pole)."""
        vec = self.solver._unit_vector_from_ra_dec(0.0, -90.0)
        expected = np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec_equator_ra0(self):
        """Test unit vector for RA=0, Dec=0 (on celestial equator)."""
        vec = self.solver._unit_vector_from_ra_dec(0.0, 0.0)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec_equator_ra90(self):
        """Test unit vector for RA=90, Dec=0."""
        vec = self.solver._unit_vector_from_ra_dec(90.0, 0.0)
        expected = np.array([0.0, 1.0, 0.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec_equator_ra180(self):
        """Test unit vector for RA=180, Dec=0."""
        vec = self.solver._unit_vector_from_ra_dec(180.0, 0.0)
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec_equator_ra270(self):
        """Test unit vector for RA=270, Dec=0."""
        vec = self.solver._unit_vector_from_ra_dec(270.0, 0.0)
        expected = np.array([0.0, -1.0, 0.0])
        np.testing.assert_allclose(vec, expected, atol=1e-15)

    def test_unit_vector_from_ra_dec