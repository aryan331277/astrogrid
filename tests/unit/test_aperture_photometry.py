"""Zero-mock unit tests for aperture_photometry module.

Tests mathematical calculations for aperture photometry including
mask creation, flux computation, background correction, and
aperture corrections.
"""

import numpy as np
import pytest
from pydantic import ValidationError

from src.astrogrid.engines.aperture_photometry import (
    ApertureType,
    ApertureConfig,
    ApertureResult,
    _create_circular_mask,
    _create_elliptical_mask,
    _create_rectangular_mask,
    _create_annulus_mask,
    compute_aperture_photometry,
)


class TestCreateCircularMask:
    """Tests for _create_circular_mask function."""

    def test_circular_mask_basic_shape(self):
        """Mask should have the same shape as the input image."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 10)
        assert mask.shape == image_shape

    def test_circular_mask_center_pixel_included(self):
        """The center pixel should be included in the mask."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 10)
        assert mask[50, 50] == True

    def test_circular_mask_outside_excluded(self):
        """Pixels far outside the radius should be excluded."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 10)
        # Pixel at (0, 0) is distance sqrt(50^2 + 50^2) ≈ 70.7 from center
        assert mask[0, 0] == False

    def test_circular_mask_boundary(self):
        """Pixels exactly at the radius boundary should be included."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 10)
        # Pixel at (60, 50) is exactly 10 pixels from center
        assert mask[60, 50] == True
        # Pixel at (50, 60) is exactly 10 pixels from center
        assert mask[50, 60] == True

    def test_circular_mask_just_outside_boundary(self):
        """Pixels just outside the radius should be excluded."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 10)
        # Pixel at (61, 50) is 11 pixels from center
        assert mask[61, 50] == False
        # Pixel at (50, 61) is 11 pixels from center
        assert mask[50, 61] == False

    def test_circular_mask_pixel_count(self):
        """Verify approximate pixel count for a circular mask."""
        image_shape = (100, 100)
        radius = 10
        mask = _create_circular_mask(image_shape, 50, 50, radius)
        pixel_count = np.sum(mask)
        # Area of circle = pi * r^2 ≈ 314.16
        # Pixel count should be close to this (within reasonable tolerance)
        expected_area = np.pi * radius ** 2
        assert abs(pixel_count - expected_area) < 20

    def test_circular_mask_small_radius(self):
        """Test with a very small radius (1 pixel)."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 1)
        # Should include center and possibly adjacent pixels
        assert mask[50, 50] == True
        pixel_count = np.sum(mask)
        assert pixel_count >= 1
        assert pixel_count <= 5  # At most a small cross pattern

    def test_circular_mask_off_center(self):
        """Test with center not at image center."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 20, 30, 5)
        assert mask[20, 30] == True
        assert mask[0, 0] == False

    def test_circular_mask_dtype(self):
        """Mask should be boolean dtype."""
        image_shape = (50, 50)
        mask = _create_circular_mask(image_shape, 25, 25, 5)
        assert mask.dtype == bool

    def test_circular_mask_radius_zero(self):
        """Radius of zero should include only the center pixel."""
        image_shape = (100, 100)
        mask = _create_circular_mask(image_shape, 50, 50, 0)
        assert mask[50, 50] == True
        assert mask[51, 50] == False
        assert mask[50, 51] == False


class TestCreateEllipticalMask:
    """Tests for _create_elliptical_mask function."""

    def test_elliptical_mask_basic_shape(self):
        """Mask should have the same shape as the input image."""
        image_shape = (100, 100)
        mask = _create_elliptical_mask(image_shape, 50, 50, 15, 10, 0.0)
        assert mask.shape == image_shape

    def test_elliptical_mask_center_included(self):
        """The center pixel should be included."""
        image_shape = (100, 100)
        mask = _create_elliptical_mask(image_shape, 50, 50, 15, 10, 0.0)
        assert mask[50, 50] == True

    def test_elliptical_mask_major_axis(self):
        """Pixels along the major axis at the boundary should be included."""
        image_shape = (100, 100)
        # Position angle 0 means major axis is along x-axis
        mask = _create_elliptical_mask(image_shape, 50, 50, 15, 10, 0.0)
        # Pixel at (65, 50) is 15 pixels along x-axis (major axis)
        assert mask[65, 50] == True
        # Pixel at (66, 50) is 16 pixels, outside
        assert mask[66, 50] == False

    def test_elliptical_mask_minor_axis(self):
        """Pixels along the minor axis at the boundary should be included."""
        image_shape = (100, 100)
        # Position angle 0 means minor axis is along y-axis
        mask = _create_elliptical_mask(image_shape, 50, 50, 15, 10, 0.0)
        # Pixel at (50, 60) is 10 pixels along y-axis (minor axis)
        assert mask[50, 60] == True
        # Pixel at (50, 61) is 11 pixels, outside
        assert mask[50, 61] == False

    def test_elliptical_mask_rotated(self):
        """Test with a rotated ellipse (position angle = 90 degrees)."""
        image_shape = (100, 100)
        # Position angle 90 degrees (pi/2) means major axis along y-axis
        mask = _create_elliptical_mask(image_shape, 50, 50, 15, 10, np.pi / 2)
        # Pixel at (50, 65) is 15 pixels along y-axis (now major axis)
        assert mask[50, 65] == True
        # Pixel at (50, 66) is 16 pixels, outside
        assert mask[50, 66] == False

    def test_elliptical_mask_circular_special_case(self):
        """When major == minor, should behave like circular mask."""
        image_shape = (100, 100)
        radius = 10
        circ_mask = _create_circular_mask(image_shape, 50, 50, radius)
        ellip_mask = _create_elliptical_mask(image_shape, 50, 50, radius, radius, 0.0)