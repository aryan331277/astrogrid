"""
Unit tests for the Supernova Arnett Model engine.

These tests verify the mathematical correctness of the Arnett (1982) model
implementation using only the Python standard library, numpy, scipy, pydantic,
fastapi, and httpx. No mocks are used.
"""

import numpy as np
import pytest
from typing import Optional

from src.astrogrid.engines.supernova_arnett_model import (
    OpacityModel,
    ArnettModelParams,
    ArnettModelResult,
    SupernovaArnettModel,
)


class TestOpacityModel:
    """Tests for the OpacityModel enum."""

    def test_grey_value(self):
        assert OpacityModel.GREY.value == "grey"

    def test_is_string_enum(self):
        assert isinstance(OpacityModel.GREY, str)


class TestArnettModelParams:
    """Tests for ArnettModelParams pydantic model."""

    def test_valid_params(self):
        params = ArnettModelParams(
            ejecta_mass=10.0,
            kinetic_energy=1e51,
            nickel_mass=0.1,
            opacity=0.1,
            opacity_model=OpacityModel.GREY,
        )
        assert params.ejecta_mass == 10.0
        assert params.kinetic_energy == 1e51
        assert params.nickel_mass == 0.1
        assert params.opacity == 0.1
        assert params.opacity_model == OpacityModel.GREY

    def test_negative_ejecta_mass_rejected(self):
        with pytest.raises(Exception):
            ArnettModelParams(
                ejecta_mass=-1.0,
                kinetic_energy=1e51,
                nickel_mass=0.1,
                opacity=0.1,
                opacity_model=OpacityModel.GREY,
            )

    def test_negative_kinetic_energy_rejected(self):
        with pytest.raises(Exception):
            ArnettModelParams(
                ejecta_mass=10.0,
                kinetic_energy=-1e51,
                nickel_mass=0.1,
                opacity=0.1,
                opacity_model=OpacityModel.GREY,
            )

    def test_negative_nickel_mass_rejected(self):
        with pytest.raises(Exception):
            ArnettModelParams(
                ejecta_mass=10.0,
                kinetic_energy=1e51,
                nickel_mass=-0.1,
                opacity=0.1,
                opacity_model=OpacityModel.GREY,
            )

    def test_negative_opacity_rejected(self):
        with pytest.raises(Exception):
            ArnettModelParams(
                ejecta_mass=10.0,
                kinetic_energy=1e51,
                nickel_mass=0.1,
                opacity=-0.1,
                opacity_model=OpacityModel.GREY,
            )


class TestArnettModelResult:
    """Tests for ArnettModelResult pydantic model."""

    def test_valid_result(self):
        result = ArnettModelResult(
            times=np.array([1.0, 2.0, 3.0]),
            luminosities=np.array([1e42, 2e42, 3e42]),
            peak_luminosity=2e42,
            peak_time=2.0,
        )
        assert len(result.times) == 3
        assert len(result.luminosities) == 3
        assert result.peak_luminosity == 2e42
        assert result.peak_time == 2.0


class TestSupernovaArnettModel:
    """Tests for the SupernovaArnettModel class."""

    def _make_model(self, **kwargs):
        """Helper to create a model with default or overridden params."""
        defaults = {
            "ejecta_mass": 10.0,
            "kinetic_energy": 1e51,
            "nickel_mass": 0.1,
            "opacity": 0.1,
            "opacity_model": OpacityModel.GREY,
        }
        defaults.update(kwargs)
        params = ArnettModelParams(**defaults)
        return SupernovaArnettModel(params)

    def test_init_stores_params(self):
        model = self._make_model()
        assert model.params.ejecta_mass == 10.0
        assert model.params.kinetic_energy == 1e51

    def test_get_result_returns_result_object(self):
        model = self._make_model()
        result = model.get_result()
        assert isinstance(result, ArnettModelResult)

    def test_result_has_matching_array_lengths(self):
        model = self._make_model()
        result = model.get_result()
        assert len(result.times) == len(result.luminosities)

    def test_luminosities_non_negative(self):
        """Luminosity should be non-negative at all times."""
        model = self._make_model()
        result = model.get_result()
        assert np.all(result.luminosities >= 0.0)

    def test_times_positive(self):
        """Times should be positive."""
        model = self._make_model()
        result = model.get_result()
        assert np.all(result.times > 0.0)

    def test_peak_luminosity_is_max(self):
        """Peak luminosity should equal the maximum of the luminosity array."""
        model = self._make_model()
        result = model.get_result()
        assert np.isclose(result.peak_luminosity, np.max(result.luminosities), rtol=1e-6)

    def test_peak_time_corresponds_to_peak_luminosity(self):
        """Peak time should correspond to the time of maximum luminosity."""
        model = self._make_model()
        result = model.get_result()
        peak_idx = np.argmax(result.luminosities)
        assert np.isclose(result.peak_time, result.times[peak_idx], rtol=1e-6)

    def test_luminosity_decreases_after_peak(self):
        """After the peak, luminosity should generally decrease."""
        model = self._make_model()
        result = model.get_result()
        peak_idx = np.argmax(result.luminosities)
        if peak_idx < len(result.luminosities) - 1:
            post_peak_lums = result.luminosities[peak_idx + 1:]
            assert np.all(post_peak_lums < result.peak_luminosity)

    def test_luminosity_increases_before_peak(self):
        """Before the peak, luminosity should generally increase."""
        model = self._make_model()
        result = model.get_result()
        peak_idx = np.argmax(result.luminosities)
        if peak_idx > 0:
            pre_peak_lums = result.luminosities[:peak_idx]
            assert pre_peak_lums[0] < result.peak_luminosity

    def test_get_luminosity_at_time_returns_scalar(self):
        """get_luminosity_at_time should return a scalar value."""
        model = self._make_model()
        result = model.get_result()
        t = result.times[len(result.times) // 2]
        lum = model.get_luminosity_at_time(t)
        assert np.isscalar(lum) or (isinstance(lum, np.ndarray) and lum.ndim == 0)

    def test_get_luminosity_at_time_non_negative(self):
        """Luminosity at any time should be non-negative."""
        model = self._make_model()
        result = model.get_result()
        for t in result.times:
            lum = model.get_luminosity_at_time(t)
            assert lum >= 0.0

    def test_get_luminosity_at_time_matches_array(self):
        """Luminosity at a given time should match the array value (within tolerance)."""
        model = self._make_model()
        result = model.get_result()
        for idx in [0, len(result.times) // 2, len(result.times) - 1]:
            t = result.times[idx]
            lum = model.get_luminosity_at_time(t)
            assert np.isclose(lum, result.luminosities[idx], rtol=1e-3)

    def test_different_ejecta_mass_affects_peak(self):
        """Different ejecta masses should produce different peak luminosities."""
        model1 = self._make_model(ejecta_mass=5.0)
        model2 = self._make_model(ejecta_mass=20.0)
        result1 = model1.get_result()
        result2 = model2.get_result()
        assert not np.isclose(result1.peak_luminosity, result2.peak_luminosity, rtol=1e-6)

    def test_different_kinetic_energy_affects_peak(self):
        """Different kinetic energies should produce different peak luminosities."""