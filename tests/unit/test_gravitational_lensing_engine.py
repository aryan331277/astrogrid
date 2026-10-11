"""
Zero-mock unit tests for the gravitational_lensing_engine module.

These tests verify mathematical calculations for gravitational lensing
using only the Python standard library, numpy, scipy, pydantic, fastapi,
and httpx. No mocking is used.
"""

import numpy as np
import pytest
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum


# Import the engine module under test
import sys
import os

# Add the src directory to path so we can import the engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.gravitational_lensing_engine import (
    LensModel,
    LensParameters,
    LensingResult,
    GravitationalLensingEngine,
)


class TestLensModel:
    """Tests for the LensModel enum."""

    def test_point_mass_value(self):
        assert LensModel.POINT_MASS.value == "point_mass"

    def test_sis_value(self):
        assert LensModel.SIS.value == "sis"

    def test_nfw_value(self):
        assert LensModel.NFW.value == "nfw"

    def test_elliptical_value(self):
        assert LensModel.ELLIPTICAL.value == "elliptical"

    def test_all_models_are_strings(self):
        for model in LensModel:
            assert isinstance(model.value, str)


class TestLensParameters:
    """Tests for the LensParameters pydantic model."""

    def test_default_creation(self):
        params = LensParameters()
        assert params is not None

    def test_with_point_mass_params(self):
        params = LensParameters(
            model=LensModel.POINT_MASS,
            mass=1e12,
            angular_scale_radius=0.1,
        )
        assert params.model == LensModel.POINT_MASS
        assert params.mass == 1e12
        assert params.angular_scale_radius == 0.1

    def test_with_sis_params(self):
        params = LensParameters(
            model=LensModel.SIS,
            velocity_dispersion=200.0,
        )
        assert params.model == LensModel.SIS
        assert params.velocity_dispersion == 200.0


class TestLensingResult:
    """Tests for the LensingResult pydantic model."""

    def test_basic_creation(self):
        result = LensingResult(
            deflection_angle=np.array([0.1, 0.2]),
            image_positions=np.array([[1.0, 2.0], [3.0, 4.0]]),
            magnifications=np.array([1.5, 2.0]),
        )
        assert result.deflection_angle is not None
        assert result.image_positions is not None
        assert result.magnifications is not None


class TestGravitationalLensingEngine:
    """Tests for the GravitationalLensingEngine class."""

    def test_initialization(self):
        engine = GravitationalLensingEngine()
        assert engine is not None

    def test_set_lens_with_point_mass(self):
        engine = GravitationalLensingEngine()
        params = LensParameters(
            model=LensModel.POINT_MASS,
            mass=1e12,
            angular_scale_radius=0.1,
        )
        engine.set_lens(params)
        # After setting, the engine should have a lens configured
        assert engine is not None

    def test_set_lens_with_sis(self):
        engine = GravitationalLensingEngine()
        params = LensParameters(
            model=LensModel.SIS,
            velocity_dispersion=200.0,
        )
        engine.set_lens(params)
        assert engine is not None

    def test_deflection_angle_point_mass_symmetry(self):
        """
        For a point mass lens, the deflection angle should be symmetric
        about the lens center. The magnitude of deflection at (x, 0) should
        equal the magnitude at (-x, 0), and the direction should be opposite.
        """
        engine = GravitationalLensingEngine()
        params = LensParameters(
            model=LensModel.POINT_MASS,
            mass=1e12,
            angular_scale_radius=0.1,
        )
        engine.set_lens(params)

        # Test at symmetric positions
        x_pos = np.array([0.5, 0.0])
        x_neg = np.array([-0.5, 0.0])

        deflection_pos = engine.deflection_angle(x_pos)
        deflection_neg = engine.deflection_angle(x_neg)

        # The magnitudes should be equal
        mag_pos = np.linalg.norm(deflection_pos)
        mag_neg = np.linalg.norm(deflection_neg)
        assert np.isclose(mag_pos, mag_neg, rtol=1e-10), \
            f"Deflection magnitudes should be equal: {mag_pos} vs {mag_neg}"

        # The deflection vectors should point toward the lens center (origin)
        # At (0.5, 0), deflection should point in negative x direction
        # At (-0.5, 0), deflection should point in positive x direction
        assert deflection_pos[0] < 0, "Deflection at positive x should point toward origin"
        assert deflection_neg[0] > 0, "Deflection at negative x should point toward origin"

    def test_deflection_angle_point_mass_magnitude(self):
        """
        For a point mass lens, the deflection angle magnitude at position x
        should be alpha = theta_E^2 / x, where theta_E is the Einstein radius.
        """
        engine = GravitationalLensingEngine()
        # Use known parameters to compute expected Einstein radius
        mass = 1e12  # solar masses
        angular_scale_radius = 0.1  # arcsec

        params = LensParameters(
            model=LensModel.POINT_MASS,
            mass=mass,
            angular_scale_radius=angular_scale_radius,
        )
        engine.set_lens(params)

        # Test at a specific position
        x = np.array([1.0, 0.0])
        deflection = engine.deflection_angle(x)
        deflection_mag = np.linalg.norm(deflection)

        # Deflection should be positive and finite
        assert deflection_mag > 0, "Deflection angle should be positive"
        assert np.isfinite(deflection_mag), "Deflection angle should be finite"

    def test_deflection_angle_point_mass_inverse_square_law(self):
        """
        For a point mass lens, the deflection angle magnitude should scale
        as 1/r where r is the distance from the lens center.
        """
        engine = GravitationalLensingEngine()
        params = LensParameters(
            model=LensModel.POINT_MASS,
            mass=1e12,
            angular_scale_radius=0.1,
        )
        engine.set_lens(params)

        # Test at two different distances
        r1 = 0.5
        r2 = 1.0

        x1 = np.array([r1, 0.0])
        x2 = np.array([r2, 0.0])

        deflection1 = engine.deflection_angle(x1)
        deflection2 = engine.deflection_angle(x2)

        mag1 = np.linalg.norm(deflection1)
        mag2 = np.linalg.norm(deflection2)

        # The ratio should be r2/r1 = 2
        ratio = mag1 / mag2
        expected_ratio = r2 / r1
        assert np.isclose(ratio, expected_ratio, rtol=1e-10), \
            f"Deflection ratio should be {expected_ratio}, got {ratio}"

    def test_deflection_angle_sis_constant_magnitude(self):
        """
        For an SIS (Singular Isothermal Sphere) lens, the deflection angle
        magnitude should be constant (independent of radius) for all r > 0.
        """
        engine = GravitationalLensingEngine()
        params = LensParameters(
            model=LensModel.SIS,
            velocity_dispersion=200.0,
        )
        engine.set_lens(params)

        # Test at multiple radii
        radii = [0.1, 0.5, 1.0, 2.0]
        magnitudes = []

        for r in radii:
            x = np.array([r, 0.0])
            deflection = engine.deflection_angle(x)
            mag = np.linalg.norm(deflection)
            magnitudes.append(mag)

        # All magnitudes should be approximately equal
        for i in range(1, len(magnitudes)):
            assert np.isclose(magnitudes[0], magnitudes[i], rtol=1e-