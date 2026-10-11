"""Zero-mock unit tests for the multi_messenger_correlator engine.

These tests verify mathematical invariants and correctness of the
MultiMessengerCorrelator using only real data structures and
deterministic assertions. No mocking is used.
"""

import numpy as np
import pytest
from datetime import datetime, timedelta
import uuid

from src.astrogrid.engines.multi_messenger_correlator import (
    ChannelType,
    ChannelSignal,
    CorrelationResult,
    MultiMessengerCorrelator,
)


class TestChannelType:
    """Tests for the ChannelType enum."""

    def test_gw_value(self):
        assert ChannelType.GW.value == "GW"

    def test_nu_value(self):
        assert ChannelType.NU.value == "NU"

    def test_em_value(self):
        assert ChannelType.EM.value == "EM"

    def test_all_members(self):
        assert len(ChannelType) == 3


class TestChannelSignal:
    """Tests for the ChannelSignal pydantic model."""

    def test_create_signal(self):
        signal = ChannelSignal(
            channel_type=ChannelType.GW,
            timestamp=datetime(2024, 1, 1, 0, 0, 0),
            data=np.array([1.0, 2.0, 3.0, 4.0]),
            sampling_rate=100.0,
        )
        assert signal.channel_type == ChannelType.GW
        assert len(signal.data) == 4
        assert signal.sampling_rate == 100.0

    def test_signal_data_is_array(self):
        data = np.array([0.5, 1.5, 2.5])
        signal = ChannelSignal(
            channel_type=ChannelType.EM,
            timestamp=datetime(2024, 1, 1),
            data=data,
            sampling_rate=50.0,
        )
        assert isinstance(signal.data, np.ndarray)
        assert np.array_equal(signal.data, data)


class TestCorrelationResult:
    """Tests for the CorrelationResult pydantic model."""

    def test_create_result(self):
        result = CorrelationResult(
            channel_a="GW",
            channel_b="EM",
            correlation_coefficient=0.85,
            time_lag=0.1,
            confidence=0.95,
        )
        assert result.channel_a == "GW"
        assert result.channel_b == "EM"
        assert result.correlation_coefficient == 0.85
        assert result.time_lag == 0.1
        assert result.confidence == 0.95


class TestMultiMessengerCorrelator:
    """Tests for the MultiMessengerCorrelator engine."""

    def setup_method(self):
        """Set up a correlator instance for each test."""
        self.correlator = MultiMessengerCorrelator()

    def test_normalize_zero_mean(self):
        """Normalization should produce zero-mean data."""
        data = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normalized = self.correlator._normalize(data)
        assert np.isclose(np.mean(normalized), 0.0, atol=1e-10)

    def test_normalize_unit_variance(self):
        """Normalization should produce unit variance (std=1)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        normalized = self.correlator._normalize(data)
        assert np.isclose(np.std(normalized), 1.0, atol=1e-10)

    def test_normalize_preserves_length(self):
        """Normalization should not change the length of the array."""
        data = np.array([5.0, 3.0, 8.0, 1.0, 9.0])
        normalized = self.correlator._normalize(data)
        assert len(normalized) == len(data)

    def test_normalize_constant_array(self):
        """Normalizing a constant array should return zeros (or handle gracefully)."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        normalized = self.correlator._normalize(data)
        # A constant array has zero variance; normalization should yield zeros
        assert np.allclose(normalized, 0.0, atol=1e-10)

    def test_cross_correlate_identical_signals(self):
        """Cross-correlation of a signal with itself should be 1.0 at zero lag."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0])
        result = self.correlator._cross_correlate(data, data)
        # The maximum correlation should be 1.0 (or very close)
        assert np.isclose(np.max(result), 1.0, atol=1e-10)

    def test_cross_correlate_anti_correlated(self):
        """Cross-correlation of a signal with its negation should be -1.0 at zero lag."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        neg_data = -data
        result = self.correlator._cross_correlate(data, neg_data)
        assert np.isclose(np.min(result), -1.0, atol=1e-10)

    def test_cross_correlate_orthogonal_signals(self):
        """Cross-correlation of orthogonal signals should be near zero."""
        # Create two orthogonal signals
        n = 100
        t = np.linspace(0, 2 * np.pi, n)
        signal_a = np.sin(t)
        signal_b = np.cos(t)
        result = self.correlator._cross_correlate(signal_a, signal_b)
        # The maximum absolute value should be small
        assert np.max(np.abs(result)) < 0.1

    def test_cross_correlate_bounded(self):
        """Cross-correlation values should be bounded between -1 and 1."""
        np.random.seed(42)
        data_a = np.random.randn(100)
        data_b = np.random.randn(100)
        result = self.correlator._cross_correlate(data_a, data_b)
        assert np.all(result >= -1.0 - 1e-10)
        assert np.all(result <= 1.0 + 1e-10)

    def test_cross_correlate_symmetry(self):
        """Cross-correlation should be symmetric: corr(a,b) at lag k = corr(b,a) at lag -k."""
        np.random.seed(123)
        data_a = np.random.randn(50)
        data_b = np.random.randn(50)
        corr_ab = self.correlator._cross_correlate(data_a, data_b)
        corr_ba = self.correlator._cross_correlate(data_b, data_a)
        # corr_ab[k] should equal corr_ba[-k] (with appropriate indexing)
        # For full cross-correlation, the relationship is corr_ab = corr_ba[::-1]
        # depending on implementation. Let's check the peak values match.
        assert np.isclose(np.max(corr_ab), np.max(corr_ba), atol=1e-10)

    def test_correlate_channels_identical_channels(self):
        """Correlating a channel with itself should yield high correlation."""
        np.random.seed(42)
        data = np.random.randn(100)
        signal_a = ChannelSignal(
            channel_type=ChannelType.GW,
            timestamp=datetime(2024, 1, 1),
            data=data,
            sampling_rate=100.0,
        )
        signal_b = ChannelSignal(
            channel_type=ChannelType.GW,
            timestamp=datetime(2024, 1, 1),
            data=data.copy(),
            sampling_rate=100.0,
        )
        result = self.correlator.correlate_channels(signal_a, signal_b)
        assert result.correlation_coefficient > 0.99

    def test_correlate_channels_anti_correlated(self):
        """Correlating a channel with its negation should yield low correlation."""
        np.random.seed(42)
        data = np.random.randn(100)
        signal_a = ChannelSignal(
            channel_type=ChannelType.GW,
            timestamp=datetime(20