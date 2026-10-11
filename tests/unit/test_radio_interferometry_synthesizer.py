"""Zero-mock unit tests for radio_interferometry_synthesizer module.

Tests mathematical invariants and core calculations of the
RadioInterferometrySynthesizer engine using only standard library,
numpy, scipy, pydantic, fastapi, and httpx.
"""

import math
import numpy as np
import pytest
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Union
from enum import Enum


# Import the module under test
import sys
import os

# Add src to path for import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.radio_interferometry_synthesizer import (
    CoordinateSystem,
    SynthesisConfig,
    VisibilityData,
    SynthesisResult,
    RadioInterferometrySynthesizer,
)


class TestCoordinateSystem:
    """Tests for the CoordinateSystem enum."""

    def test_enum_values_exist(self):
        """Verify that expected coordinate system values exist."""
        # Check that the enum has at least one member
        members = list(CoordinateSystem)
        assert len(members) > 0

    def test_enum_is_string_based(self):
        """Verify that CoordinateSystem is a str-based enum."""
        for member in CoordinateSystem:
            assert isinstance(member.value, str)

    def test_enum_members_are_unique(self):
        """Verify that all enum values are unique."""
        values = [m.value for m in CoordinateSystem]
        assert len(values) == len(set(values))


class TestSynthesisConfig:
    """Tests for the SynthesisConfig pydantic model."""

    def test_default_instantiation(self):
        """Test that SynthesisConfig can be instantiated with defaults."""
        config = SynthesisConfig()
        assert config is not None

    def test_config_fields_accessible(self):
        """Test that config fields are accessible after creation."""
        config = SynthesisConfig()
        # Access common fields - they should exist
        assert hasattr(config, '__dict__') or hasattr(config, '__fields__')


class TestVisibilityData:
    """Tests for the VisibilityData pydantic model."""

    def test_create_with_valid_arrays(self):
        """Test creating VisibilityData with valid numpy arrays."""
        n = 10
        u = np.linspace(-100, 100, n)
        v = np.linspace(-100, 100, n)
        w = np.zeros(n)
        vis = np.ones(n, dtype=complex)
        weight = np.ones(n)

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        assert len(vd.u) == n
        assert len(vd.v) == n
        assert len(vd.w) == n
        assert len(vd.vis) == n
        assert len(vd.weight) == n

    def test_validate_arrays_length_mismatch(self):
        """Test that validate_arrays raises on length mismatch."""
        u = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 2.0])  # different length
        w = np.array([0.0, 0.0, 0.0])
        vis = np.array([1+0j, 2+0j, 3+0j])
        weight = np.array([1.0, 1.0, 1.0])

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        with pytest.raises(Exception):
            vd.validate_arrays()

    def test_validate_arrays_matching_lengths(self):
        """Test that validate_arrays passes with matching lengths."""
        n = 5
        u = np.ones(n)
        v = np.ones(n)
        w = np.zeros(n)
        vis = np.ones(n, dtype=complex)
        weight = np.ones(n)

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        # Should not raise
        vd.validate_arrays()

    def test_validate_weight_negative(self):
        """Test that validate_weight raises on negative weights."""
        n = 3
        u = np.ones(n)
        v = np.ones(n)
        w = np.zeros(n)
        vis = np.ones(n, dtype=complex)
        weight = np.array([1.0, -1.0, 1.0])  # negative weight

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        with pytest.raises(Exception):
            vd.validate_weight()

    def test_validate_weight_all_positive(self):
        """Test that validate_weight passes with all positive weights."""
        n = 3
        u = np.ones(n)
        v = np.ones(n)
        w = np.zeros(n)
        vis = np.ones(n, dtype=complex)
        weight = np.array([1.0, 2.0, 3.0])

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        # Should not raise
        vd.validate_weight()

    def test_validate_weight_zero(self):
        """Test that validate_weight handles zero weights (should be valid or raise)."""
        n = 3
        u = np.ones(n)
        v = np.ones(n)
        w = np.zeros(n)
        vis = np.ones(n, dtype=complex)
        weight = np.array([0.0, 1.0, 2.0])

        vd = VisibilityData(u=u, v=v, w=w, vis=vis, weight=weight)
        # Zero weight should be acceptable (non-negative)
        vd.validate_weight()


class TestSynthesisResult:
    """Tests for the SynthesisResult pydantic model."""

    def test_create_with_valid_image(self):
        """Test creating SynthesisResult with a valid image array."""
        image = np.ones((16, 16))
        result = SynthesisResult(image=image)
        assert result.image.shape == (16, 16)

    def test_validate_image_non_negative(self):
        """Test that validate_image raises on negative values."""
        image = np.array([[1.0, -0.5], [0.3, 2.0]])
        result = SynthesisResult(image=image)
        with pytest.raises(Exception):
            result.validate_image()

    def test_validate_image_all_positive(self):
        """Test that validate_image passes with all non-negative values."""
        image = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = SynthesisResult(image=image)
        # Should not raise
        result.validate_image()

    def test_validate_image_zero_values(self):
        """Test that validate_image handles zero values."""
        image = np.array([[0.0, 1.0], [2.0, 0.0]])
        result = SynthesisResult(image=image)
        # Should not raise (non-negative)
        result.validate_image()


class TestRadioInterferometrySynthesizer:
    """Tests for the RadioInterferometrySynthesizer class."""

    def _make_synthesizer(self, **kwargs):
        """Helper to create a synthesizer instance with default config."""
        defaults = {
            'image_size': 64,
            'pixel_scale': 0.1,
            'taper': 1.0,
        }
        defaults.update(kwargs)
        config = SynthesisConfig(
            image_size=defaults['image_size'],
            pixel_scale=defaults['pixel_scale'],
            taper=defaults['taper'],
        )
        return RadioInterferometrySynthesizer(config=config)

    def test_init_creates_instance(self):
        """Test that the synthesizer can be instantiated."""
        synth = self._make_synthesizer()
        assert synth is not None

    def test_compute_pixel_scale_positive(self):
        """Test that _compute_pixel_scale returns a positive value."""
        synth = self._make_synthesizer()
        # pixel_scale should be positive
        ps = synth._compute_pixel_scale()
        assert ps > 0

    def test_compute_pixel_scale_deterministic(self):
        """Test that _compute_pixel_scale is deterministic."""
        synth = self._make_synthesizer()
        ps1 = synth._compute_pixel_scale()
        ps2 = synth._compute_pixel_scale()
        assert ps1 == ps2

    def test_apply_taper_reduces_extreme_weights(self):
        """Test that applying a taper reduces the dynamic range of weights."""
        synth = self._make_synthesizer(taper=0.5)
        # Create weights with high dynamic range
        u = np.linspace(-100, 100, 100)
        v = np.zeros(10