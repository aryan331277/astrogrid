"""Zero-mock unit tests for fits_header_wcs_parser module.

Tests mathematical calculations for WCS parsing and coordinate transformations
using only standard library, numpy, scipy, pydantic, fastapi, and httpx.
"""

import math
import pytest
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from src.astrogrid.engines.fits_header_wcs_parser import (
    WCSAxisType,
    WCSAxis,
    WCSInfo,
    _parse_axis_type,
    parse_wcs_header,
    pixel_to_world,
    world_to_pixel,
)


class TestParseAxisType:
    """Tests for the _parse_axis_type function."""

    def test_parse_ra_type(self):
        """Test parsing RA axis type."""
        result = _parse_axis_type("RA---A")
        assert result == WCSAxisType.RA

    def test_parse_dec_type(self):
        """Test parsing DEC axis type."""
        result = _parse_axis_type("DEC--A")
        assert result == WCSAxisType.DEC

    def test_parse_freq_type(self):
        """Test parsing FREQ axis type."""
        result = _parse_axis_type("FREQ--")
        assert result == WCSAxisType.FREQ

    def test_parse_time_type(self):
        """Test parsing TIME axis type."""
        result = _parse_axis_type("TIME--")
        assert result == WCSAxisType.TIME

    def test_parse_velo_type(self):
        """Test parsing VELO axis type."""
        result = _parse_axis_type("VELO--")
        assert result == WCSAxisType.VELO

    def test_parse_stokes_type(self):
        """Test parsing STOKES axis type."""
        result = _parse_axis_type("STOKES")
        assert result == WCSAxisType.STOKES

    def test_parse_unknown_type(self):
        """Test parsing unknown axis type returns None or raises."""
        with pytest.raises((ValueError, KeyError)):
            _parse_axis_type("UNKNOWN")

    def test_parse_empty_string(self):
        """Test parsing empty string raises exception."""
        with pytest.raises((ValueError, KeyError, IndexError)):
            _parse_axis_type("")


class TestWCSAxis:
    """Tests for WCSAxis data model."""

    def test_wcs_axis_creation(self):
        """Test creating a WCSAxis instance."""
        axis = WCSAxis(
            axis_type=WCSAxisType.RA,
            reference_value=180.0,
            reference_pixel=1.0,
            scale=0.001,
            rotation=0.0,
        )
        assert axis.axis_type == WCSAxisType.RA
        assert axis.reference_value == 180.0
        assert axis.reference_pixel == 1.0
        assert axis.scale == 0.001
        assert axis.rotation == 0.0

    def test_wcs_axis_default_rotation(self):
        """Test WCSAxis default rotation is 0."""
        axis = WCSAxis(
            axis_type=WCSAxisType.DEC,
            reference_value=45.0,
            reference_pixel=100.0,
            scale=-0.002,
        )
        assert axis.rotation == 0.0


class TestWCSInfo:
    """Tests for WCSInfo data model."""

    def test_wcs_info_creation(self):
        """Test creating a WCSInfo instance."""
        axis1 = WCSAxis(
            axis_type=WCSAxisType.RA,
            reference_value=180.0,
            reference_pixel=1.0,
            scale=0.001,
        )
        axis2 = WCSAxis(
            axis_type=WCSAxisType.DEC,
            reference_value=45.0,
            reference_pixel=1.0,
            scale=-0.002,
        )
        wcs = WCSInfo(axes=[axis1, axis2])
        assert len(wcs.axes) == 2
        assert wcs.axes[0].axis_type == WCSAxisType.RA
        assert wcs.axes[1].axis_type == WCSAxisType.DEC


class TestParseWCSHeader:
    """Tests for parse_wcs_header function."""

    def test_parse_simple_2d_header(self):
        """Test parsing a simple 2D WCS header."""
        header = {
            "CTYPE1": "RA---A",
            "CRVAL1": 180.0,
            "CRPIX1": 1.0,
            "CDELT1": 0.001,
            "CTYPE2": "DEC--A",
            "CRVAL2": 45.0,
            "CRPIX2": 1.0,
            "CDELT2": -0.002,
        }
        wcs = parse_wcs_header(header)
        assert len(wcs.axes) == 2
        assert wcs.axes[0].axis_type == WCSAxisType.RA
        assert wcs.axes[0].reference_value == 180.0
        assert wcs.axes[0].reference_pixel == 1.0
        assert wcs.axes[0].scale == 0.001
        assert wcs.axes[1].axis_type == WCSAxisType.DEC
        assert wcs.axes[1].reference_value == 45.0
        assert wcs.axes[1].reference_pixel == 1.0
        assert wcs.axes[1].scale == -0.002

    def test_parse_header_with_rotation(self):
        """Test parsing header with rotation matrix elements."""
        header = {
            "CTYPE1": "RA---A",
            "CRVAL1": 180.0,
            "CRPIX1": 1.0,
            "CDELT1": 0.001,
            "CTYPE2": "DEC--A",
            "CRVAL2": 45.0,
            "CRPIX2": 1.0,
            "CDELT2": -0.002,
            "CD1_1": 0.001,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": -0.002,
        }
        wcs = parse_wcs_header(header)
        assert len(wcs.axes) == 2

    def test_parse_empty_header(self):
        """Test parsing empty header raises exception."""
        with pytest.raises((ValueError, KeyError)):
            parse_wcs_header({})

    def test_parse_header_missing_ctype(self):
        """Test parsing header missing CTYPE raises exception."""
        header = {
            "CRVAL1": 180.0,
            "CRPIX1": 1.0,
            "CDELT1": 0.001,
        }
        with pytest.raises((ValueError, KeyError)):
            parse_wcs_header(header)

    def test_parse_1d_header(self):
        """Test parsing a 1D WCS header."""
        header = {
            "CTYPE1": "FREQ--",
            "CRVAL1": 1.4e9,
            "CRPIX1": 1.0,
            "CDELT1": 1000.0,
        }
        wcs = parse_wcs_header(header)
        assert len(wcs.axes) == 1
        assert wcs.axes[0].axis_type == WCSAxisType.FREQ
        assert wcs.axes[0].reference_value == 1.4e9
        assert wcs.axes[0].scale == 1000.0


class TestPixelToWorld:
    """Tests for pixel_to_world transformation."""

    def _create_simple_wcs(self):
        """Create a simple 2D WCS for testing."""
        axis1 = WCSAxis(
            axis_type=WCSAxisType.RA,
            reference_value=180.0,
            reference_pixel=1.0,
            scale=0.001,
            rotation=0.0,
        )
        axis2 = WCSAxis(
            axis_type=WCSAxisType.DEC,
            reference_value=45.0,
            reference_pixel=1.0,
            scale=-0.002,
            rotation=0.0,
        )
        return WCSInfo(axes=[axis1, axis2])

    def test_pixel_to_world_at_reference(self):
        """Test pixel_to_world at reference pixel returns reference value."""
        wcs = self._create_simple_wcs()
        world = pixel_to_world(wcs, (1.0, 1.0))
        assert len(world) == 2
        assert abs(world[0] - 180.0) < 1e-10
        assert abs(world[1] - 45.0) < 1e-10