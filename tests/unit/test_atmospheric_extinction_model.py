"""
Unit tests for the atmospheric_extinction_model module.

These tests verify the mathematical correctness of the atmospheric extinction
model without using any mocks. All tests use real data structures and
deterministic mathematical assertions.
"""

import math
import pytest
import numpy as np

from src.astrogrid.engines.atmospheric_extinction_model import (
    ExtinctionModelType,
    ExtinctionParameters,
    ExtinctionResult,
    AtmosphericExtinctionEngine,
)


class TestExtinctionModelType:
    """Tests for the ExtinctionModelType enum."""

    def test_enum_members_exist(self):
        """Verify that expected enum members exist."""
        # The enum should have at least one member
        members = list(ExtinctionModelType)
        assert len(members) >= 1

    def test_enum_values_are_unique(self):
        """Verify that enum values are unique."""
        members = list(ExtinctionModelType)
        values = [m.value for m in members]
        assert len(values) == len(set(values))


class TestExtinctionParameters:
    """Tests for the ExtinctionParameters dataclass/pydantic model."""

    def test_create_with_valid_parameters(self):
        """Test creating ExtinctionParameters with valid values."""
        params = ExtinctionParameters(
            extinction_coefficient=0.2,
            model_type=ExtinctionModelType.LIST[0] if hasattr(ExtinctionModelType, 'LIST') else list(ExtinctionModelType)[0],
        )
        assert params.extinction_coefficient == 0.2

    def test_extinction_coefficient_positive(self):
        """Test that extinction coefficient is stored correctly."""
        params = ExtinctionParameters(
            extinction_coefficient=0.15,
            model_type=list(ExtinctionModelType)[0],
        )
        assert params.extinction_coefficient == 0.15


class TestExtinctionResult:
    """Tests for the ExtinctionResult dataclass/pydantic model."""

    def test_create_result(self):
        """Test creating an ExtinctionResult."""
        result = ExtinctionResult(
            airmass=1.0,
            extinction=0.2,
            extinction_coefficient=0.2,
        )
        assert result.airmass == 1.0
        assert result.extinction == 0.2
        assert result.extinction_coefficient == 0.2


class TestComputeAirmass:
    """Tests for the compute_airmass method."""

    @pytest.fixture
    def engine(self):
        """Create an AtmosphericExtinctionEngine instance."""
        params = ExtinctionParameters(
            extinction_coefficient=0.2,
            model_type=list(ExtinctionModelType)[0],
        )
        return AtmosphericExtinctionEngine(params)

    def test_airmass_at_zenith(self, engine):
        """Airmass at zenith (z=0) should be approximately 1.0."""
        airmass = engine.compute_airmass(0.0)
        assert airmass == pytest.approx(1.0, rel=1e-6)

    def test_airmass_increases_with_zenith_angle(self, engine):
        """Airmass should increase as zenith angle increases."""
        airmass_0 = engine.compute_airmass(0.0)
        airmass_30 = engine.compute_airmass(30.0)
        airmass_60 = engine.compute_airmass(60.0)
        assert airmass_0 < airmass_30 < airmass_60

    def test_airmass_at_30_degrees(self, engine):
        """Airmass at 30 degrees should be approximately 1.15 (Kasten & Young)."""
        airmass = engine.compute_airmass(30.0)
        # Kasten & Young: X = 1 / (cos(z) + 0.50572 * (96.07995 - z)^-1.6364)
        z_rad = math.radians(30.0)
        cos_z = math.cos(z_rad)
        # z in degrees for the formula
        z_deg = 30.0
        denominator = cos_z + 0.50572 * (96.07995 - z_deg) ** (-1.6364)
        expected = 1.0 / denominator
        assert airmass == pytest.approx(expected, rel=1e-6)

    def test_airmass_at_60_degrees(self, engine):
        """Airmass at 60 degrees should be approximately 2.0 (Kasten & Young)."""
        airmass = engine.compute_airmass(60.0)
        z_rad = math.radians(60.0)
        cos_z = math.cos(z_rad)
        z_deg = 60.0
        denominator = cos_z + 0.50572 * (96.07995 - z_deg) ** (-1.6364)
        expected = 1.0 / denominator
        assert airmass == pytest.approx(expected, rel=1e-6)

    def test_airmass_positive(self, engine):
        """Airmass should always be positive for valid zenith angles."""
        for z in [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]:
            airmass = engine.compute_airmass(z)
            assert airmass > 0

    def test_airmass_monotonically_increasing(self, engine):
        """Airmass should be monotonically increasing with zenith angle."""
        angles = np.linspace(0, 85, 100)
        airmasses = [engine.compute_airmass(z) for z in angles]
        for i in range(1, len(airmasses)):
            assert airmasses[i] >= airmasses[i - 1]

    def test_airmass_at_85_degrees_large(self, engine):
        """Airmass at 85 degrees should be significantly larger than at zenith."""
        airmass_85 = engine.compute_airmass(85.0)
        airmass_0 = engine.compute_airmass(0.0)
        assert airmass_85 > airmass_0 * 5


class TestComputeExtinctionCoefficient:
    """Tests for the compute_extinction_coefficient method."""

    @pytest.fixture
    def engine(self):
        """Create an AtmosphericExtinctionEngine instance."""
        params = ExtinctionParameters(
            extinction_coefficient=0.2,
            model_type=list(ExtinctionModelType)[0],
        )
        return AtmosphericExtinctionEngine(params)

    def test_coefficient_returns_parameter_value(self, engine):
        """The extinction coefficient should match the parameter value."""
        coeff = engine.compute_extinction_coefficient()
        assert coeff == pytest.approx(0.2, rel=1e-9)

    def test_coefficient_positive(self, engine):
        """The extinction coefficient should be positive."""
        coeff = engine.compute_extinction_coefficient()
        assert coeff > 0


class TestComputeExtinction:
    """Tests for the compute_extinction method."""

    @pytest.fixture
    def engine(self):
        """Create an AtmosphericExtinctionEngine instance."""
        params = ExtinctionParameters(
            extinction_coefficient=0.2,
            model_type=list(ExtinctionModelType)[0],
        )
        return AtmosphericExtinctionEngine(params)

    def test_extinction_at_zenith(self, engine):
        """Extinction at zenith should be k * 1.0 = k."""
        result = engine.compute_extinction(0.0)
        assert result.extinction == pytest.approx(0.2, rel=1e-6)
        assert result.airmass == pytest.approx(1.0, rel=1e-6)

    def test_extinction_increases_with_zenith_angle(self, engine):
        """Extinction should increase as zenith angle increases."""
        ext_0 = engine.compute_extinction(0.0)
        ext_30 = engine.compute_extinction(30.0)
        ext_60 = engine.compute_extinction(60.0)
        assert ext_0.extinction < ext_30.extinction < ext_60.extinction

    def test_extinction_equals_k_times_airmass(self, engine):
        """Extinction should equal k * X(z) for all tested angles."""
        k = 0.2
        for z in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0]:
            result = engine.compute_extinction