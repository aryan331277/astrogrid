"""
Zero-mock unit tests for the HealPix Spherical Pixelator Engine.

These tests verify the mathematical correctness of the HEALPix pixelization
implementation using only the Python standard library, numpy, and pydantic.
No mocking is used; all tests are deterministic and rely on known mathematical
properties of the HEALPix scheme.
"""

import math
import sys
import os

import numpy as np
import pytest
from pydantic import ValidationError

# Add the src directory to the path so we can import the engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.healpix_spherical_pixelator import (
    HealPixInputSchema,
    HealPixOutputSchema,
    _theta_to_ring,
    _phi_to_ring_index,
    theta_phi_to_healpix_ring,
    theta_phi_to_healpix_nested,
    healpix_pixelator,
)


class TestHealPixInputSchema:
    """Tests for the HealPixInputSchema validation logic."""

    def test_valid_nside_power_of_two(self):
        """NSIDE must be a power of two."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=128,
            ordering="ring"
        )
        assert schema.nside == 128

    def test_invalid_nside_not_power_of_two(self):
        """NSIDE that is not a power of two should raise ValidationError."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=1.0,
                nside=100,
                ordering="ring"
            )

    def test_nside_one_is_valid(self):
        """NSIDE=1 is a valid power of two (2^0)."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=1,
            ordering="ring"
        )
        assert schema.nside == 1

    def test_nside_two_is_valid(self):
        """NSIDE=2 is a valid power of two (2^1)."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=2,
            ordering="ring"
        )
        assert schema.nside == 2

    def test_nside_zero_invalid(self):
        """NSIDE=0 is not a power of two and should be invalid."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=1.0,
                nside=0,
                ordering="ring"
            )

    def test_nside_negative_invalid(self):
        """Negative NSIDE should be invalid."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=1.0,
                nside=-128,
                ordering="ring"
            )

    def test_theta_bounds(self):
        """Theta should be in [0, pi]."""
        schema = HealPixInputSchema(
            theta=0.0,
            phi=0.0,
            nside=128,
            ordering="ring"
        )
        assert schema.theta == 0.0

        schema = HealPixInputSchema(
            theta=math.pi,
            phi=0.0,
            nside=128,
            ordering="ring"
        )
        assert schema.theta == math.pi

    def test_theta_out_of_bounds(self):
        """Theta outside [0, pi] should raise ValidationError."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=-0.1,
                phi=0.0,
                nside=128,
                ordering="ring"
            )

        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=math.pi + 0.1,
                phi=0.0,
                nside=128,
                ordering="ring"
            )

    def test_phi_bounds(self):
        """Phi should be in [0, 2*pi)."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=0.0,
            nside=128,
            ordering="ring"
        )
        assert schema.phi == 0.0

        schema = HealPixInputSchema(
            theta=0.5,
            phi=2 * math.pi - 1e-10,
            nside=128,
            ordering="ring"
        )
        assert schema.phi < 2 * math.pi

    def test_phi_out_of_bounds(self):
        """Phi outside [0, 2*pi) should raise ValidationError."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=-0.1,
                nside=128,
                ordering="ring"
            )

        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=2 * math.pi,
                nside=128,
                ordering="ring"
            )

    def test_ordering_valid_values(self):
        """Ordering must be 'ring' or 'nested'."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=128,
            ordering="ring"
        )
        assert schema.ordering == "ring"

        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=128,
            ordering="nested"
        )
        assert schema.ordering == "nested"

    def test_ordering_invalid_value(self):
        """Invalid ordering should raise ValidationError."""
        with pytest.raises(ValidationError):
            HealPixInputSchema(
                theta=0.5,
                phi=1.0,
                nside=128,
                ordering="invalid"
            )

    def test_normalize_phi_method(self):
        """Test the normalize_phi method if it exists."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=128,
            ordering="ring"
        )
        # normalize_phi should handle wrapping
        if hasattr(schema, 'normalize_phi'):
            result = schema.normalize_phi(3 * math.pi)
            assert 0 <= result < 2 * math.pi
            result = schema.normalize_phi(-math.pi)
            assert 0 <= result < 2 * math.pi

    def test_validate_nside_power_of_two_method(self):
        """Test the validate_nside_power_of_two method if it exists."""
        schema = HealPixInputSchema(
            theta=0.5,
            phi=1.0,
            nside=128,
            ordering="ring"
        )
        if hasattr(schema, 'validate_nside_power_of_two'):
            assert schema.validate_nside_power_of_two(128) is True
            assert schema.validate_nside_power_of_two(100) is False
            assert schema.validate_nside_power_of_two(1) is True
            assert schema.validate_nside_power_of_two(2) is True
            assert schema.validate_nside_power_of_two(4) is True


class TestThetaToRing:
    """Tests for the _theta_to_ring helper function."""

    def test_theta_zero_maps_to_ring_zero(self):
        """Theta=0 (north pole) should map to ring 0."""
        ring = _theta_to_ring(0.0, 128)
        assert ring == 0

    def test_theta_pi_maps_to_last_ring(self):
        """Theta=pi (south pole) should map to the last ring."""
        nside = 128
        ring = _theta_to_ring(math.pi, nside)
        # Total number of rings is 4 * nside
        assert ring == 4 * nside - 1

    def test_theta_pi_over_two_maps_to_equator_ring(self):
        """Theta=pi/2 (equator) should map to the equatorial ring."""
        nside = 128
        ring = _theta_to_ring(math.pi / 2, nside)
        # The equator is at ring 2 * nside
        assert ring == 2 * nside

    def test_ring_is_integer(self):
        """Ring index should be an integer."""
        ring = _theta_to_ring(0.7, 64)
        assert isinstance(ring, int)

    def test_ring_monotonically_increasing_with_theta(self):
        """As