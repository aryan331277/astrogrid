"""Zero-mock unit tests for stokes_polarimetry_engine.

These tests verify the mathematical correctness of the Stokes polarimetry
engine using only real data structures and deterministic assertions.
"""

import numpy as np
import pytest
import sys
import os

# Ensure the source is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from astrogrid.engines.stokes_polarimetry_engine import (
    compute_astronomy_metric,
    CoordinateSystem,
    PolarizationState,
)


class TestComputeAstronomyMetricBasic:
    """Test basic functionality of compute_astronomy_metric."""

    def test_returns_dict(self):
        """Verify that compute_astronomy_metric returns a dictionary."""
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=[
                {"intensity": 1.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1}
            ],
        )
        assert isinstance(result, dict)

    def test_contains_metric_name(self):
        """Verify the result contains the metric name."""
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=[
                {"intensity": 1.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1}
            ],
        )
        assert "metric_name" in result
        assert result["metric_name"] == "stokes_I"

    def test_contains_value(self):
        """Verify the result contains a numeric value."""
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=[
                {"intensity": 1.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1}
            ],
        )
        assert "value" in result
        assert isinstance(result["value"], (int, float, np.floating))


class TestStokesIComputation:
    """Test Stokes I (total intensity) computation."""

    def test_single_observation_intensity(self):
        """Stokes I for a single observation should equal the intensity."""
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=[
                {"intensity": 5.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1}
            ],
        )
        assert np.isclose(result["value"], 5.0, atol=1e-10)

    def test_multiple_observations_mean_intensity(self):
        """Stokes I for multiple observations should be the mean intensity."""
        obs = [
            {"intensity": 1.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 3.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 5.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1},
        ]
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        expected = np.mean([1.0, 3.0, 5.0])
        assert np.isclose(result["value"], expected, atol=1e-10)

    def test_stokes_i_non_negative_for_positive_inputs(self):
        """Stokes I should be non-negative when all intensities are positive."""
        obs = [
            {"intensity": 2.0, "q": 0.5, "u": 0.3, "v": 0.1, "noise": 0.1},
            {"intensity": 4.0, "q": -0.2, "u": 0.1, "v": -0.1, "noise": 0.1},
        ]
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        assert result["value"] >= 0.0

    def test_stokes_i_zero_intensity(self):
        """Stokes I should be zero when all intensities are zero."""
        obs = [
            {"intensity": 0.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 0.0, "q": 0.0, "u": 0.0, "v": 0.0, "noise": 0.1},
        ]
        result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        assert np.isclose(result["value"], 0.0, atol=1e-10)


class TestStokesQComputation:
    """Test Stokes Q (linear polarization, horizontal-vertical) computation."""

    def test_single_observation_q(self):
        """Stokes Q for a single observation should equal the q value."""
        result = compute_astronomy_metric(
            metric_name="stokes_Q",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=[
                {"intensity": 1.0, "q": 0.5, "u": 0.0, "v": 0.0, "noise": 0.1}
            ],
        )
        assert np.isclose(result["value"], 0.5, atol=1e-10)

    def test_multiple_observations_mean_q(self):
        """Stokes Q for multiple observations should be the mean q."""
        obs = [
            {"intensity": 1.0, "q": 0.2, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 1.0, "q": 0.4, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 1.0, "q": 0.6, "u": 0.0, "v": 0.0, "noise": 0.1},
        ]
        result = compute_astronomy_metric(
            metric_name="stokes_Q",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        expected = np.mean([0.2, 0.4, 0.6])
        assert np.isclose(result["value"], expected, atol=1e-10)

    def test_stokes_q_bounded_by_intensity(self):
        """|Q| should not exceed I for physically valid polarization."""
        obs = [
            {"intensity": 1.0, "q": 0.5, "u": 0.0, "v": 0.0, "noise": 0.1},
            {"intensity": 1.0, "q": -0.3, "u": 0.0, "v": 0.0, "noise": 0.1},
        ]
        result = compute_astronomy_metric(
            metric_name="stokes_Q",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        i_result = compute_astronomy_metric(
            metric_name="stokes_I",
            coordinates={"ra": 10.0, "dec": 20.0},
            observations=obs,
        )
        assert abs(result["value"]) <= i_result["value"] + 1e-10


class Test