"""Zero-mock unit tests for the Fast Radio Burst (FRB) detection engine."""

import numpy as np
import pytest
from pydantic import ValidationError

from src.astrogrid.engines.fast_radio_burst_detector import (
    PolarizationMode,
    FrequencyBand,
    FRBDetectionResult,
    FRBDetectorConfig,
    FastRadioBurstDetector,
)


class TestPolarizationMode:
    """Tests for the PolarizationMode enum."""

    def test_rcp_value(self):
        assert PolarizationMode.RCP.value == "RCP"

    def test_lcp_value(self):
        assert PolarizationMode.LCP.value == "LCP"

    def test_both_value(self):
        assert PolarizationMode.BOTH.value == "BOTH"

    def test_all_members(self):
        assert len(list(PolarizationMode)) == 3

    def test_string_comparison(self):
        assert PolarizationMode.RCP == "RCP"
        assert PolarizationMode.LCP == "LCP"
        assert PolarizationMode.BOTH == "BOTH"


class TestFrequencyBand:
    """Tests for the FrequencyBand pydantic model."""

    def test_valid_frequency_band(self):
        band = FrequencyBand(min_freq_ghz=0.1, max_freq_ghz=1.5)
        assert band.min_freq_ghz == 0.1
        assert band.max_freq_ghz == 1.5

    def test_min_freq_must_be_positive(self):
        with pytest.raises(ValidationError):
            FrequencyBand(min_freq_ghz=0.0, max_freq_ghz=1.0)

    def test_negative_min_freq_rejected(self):
        with pytest.raises(ValidationError):
            FrequencyBand(min_freq_ghz=-0.5, max_freq_ghz=1.0)

    def test_validate_max_freq(self):
        band = FrequencyBand(min_freq_ghz=0.1, max_freq_ghz=1.5)
        # validate_max_freq should return the max_freq_ghz or validate it
        result = band.validate_max_freq()
        assert result == 1.5

    def test_min_freq_less_than_max_freq(self):
        band = FrequencyBand(min_freq_ghz=0.1, max_freq_ghz=1.5)
        assert band.min_freq_ghz < band.max_freq_ghz


class TestFRBDetectorConfig:
    """Tests for the FRBDetectorConfig pydantic model."""

    def test_default_config(self):
        config = FRBDetectorConfig()
        assert config is not None

    def test_config_with_custom_values(self):
        config = FRBDetectorConfig(
            snr_threshold=10.0,
            min_duration_us=10.0,
            max_duration_us=1000.0,
        )
        assert config.snr_threshold == 10.0
        assert config.min_duration_us == 10.0
        assert config.max_duration_us == 1000.0


class TestFRBDetectionResult:
    """Tests for the FRBDetectionResult pydantic model."""

    def test_basic_result(self):
        result = FRBDetectionResult(
            snr=15.0,
            duration_us=50.0,
            peak_frequency_ghz=1.2,
            polarization=PolarizationMode.RCP,
        )
        assert result.snr == 15.0
        assert result.duration_us == 50.0
        assert result.peak_frequency_ghz == 1.2
        assert result.polarization == PolarizationMode.RCP


class TestFastRadioBurstDetector:
    """Tests for the FastRadioBurstDetector class."""

    def _make_detector(self, **kwargs):
        """Create a detector with default or custom config."""
        config = FRBDetectorConfig(**kwargs)
        return FastRadioBurstDetector(config)

    def test_init_with_default_config(self):
        detector = self._make_detector()
        assert detector is not None

    def test_init_with_custom_config(self):
        detector = self._make_detector(snr_threshold=20.0)
        assert detector is not None

    def test_estimate_noise_sigma_constant_signal(self):
        """Noise sigma should be near zero for a perfectly constant signal."""
        detector = self._make_detector()
        # A constant signal has no noise
        constant_signal = np.ones(1000) * 5.0
        sigma = detector._estimate_noise_sigma(constant_signal)
        assert sigma >= 0.0
        # For a truly constant signal, sigma should be very small
        assert sigma < 1e-6

    def test_estimate_noise_sigma_with_known_noise(self):
        """Noise sigma should approximate the known standard deviation."""
        detector = self._make_detector()
        rng = np.random.default_rng(42)
        n_samples = 10000
        true_sigma = 2.5
        noisy_signal = rng.normal(loc=0.0, scale=true_sigma, size=n_samples)
        sigma = detector._estimate_noise_sigma(noisy_signal)
        # Should be within 10% of true sigma for large sample
        assert abs(sigma - true_sigma) / true_sigma < 0.1

    def test_estimate_noise_sigma_non_negative(self):
        """Noise sigma must always be non-negative."""
        detector = self._make_detector()
        rng = np.random.default_rng(123)
        for _ in range(5):
            signal = rng.normal(0, 1, 500)
            sigma = detector._estimate_noise_sigma(signal)
            assert sigma >= 0.0

    def test_compute_snr_zero_signal(self):
        """SNR for a zero signal should be zero or handled gracefully."""
        detector = self._make_detector()
        zero_signal = np.zeros(100)
        snr = detector._compute_snr(zero_signal)
        assert snr >= 0.0

    def test_compute_snr_constant_signal(self):
        """SNR for a constant signal should be very high (infinite in ideal case)."""
        detector = self._make_detector()
        constant_signal = np.ones(100) * 10.0
        snr = detector._compute_snr(constant_signal)
        # A constant signal has no noise, so SNR should be very large
        assert snr > 100.0

    def test_compute_snr_with_noise(self):
        """SNR should reflect the ratio of signal amplitude to noise."""
        detector = self._make_detector()
        rng = np.random.default_rng(99)
        n_samples = 10000
        signal_amplitude = 10.0
        noise_sigma = 1.0
        signal = np.full(n_samples, signal_amplitude) + rng.normal(0, noise_sigma, n_samples)
        snr = detector._compute_snr(signal)
        # Expected SNR is roughly signal_amplitude / noise_sigma = 10
        # Allow some tolerance
        assert 5.0 < snr < 20.0

    def test_compute_snr_higher_amplitude_higher_snr(self):
        """Higher signal amplitude should yield higher SNR."""
        detector = self._make_detector()
        rng = np.random.default_rng(55)
        n_samples = 5000
        noise = rng.normal(0, 1.0, n_samples)
        signal_low = np.full(n_samples, 5.0) + noise
        signal_high = np.full(n_samples, 20.0) + noise
        snr_low = detector._compute_snr(signal_low)
        snr_high = detector._compute_snr(signal_high)
        assert snr_high > snr_low

    def test_measure_duration_short_pulse(self):
        """Duration measurement for a short pulse should be small."""
        detector = self._make_detector()
        # Create a signal with a short pulse
        n_samples = 1000
        signal = np.zeros(n_samples)
        # Pulse from index 400 to 410 (10 samples wide)
        signal[400:410] = 1.0
        duration = detector._measure_duration(signal)
        assert duration >= 0.0
        # Duration should be small relative to total signal length
        assert duration < 100.0

    def test_measure_duration_long_pulse(self):
        """Duration measurement for a long pulse should be larger."""
        detector = self._make_detector()
        n_samples = 1000
        signal = np.zeros(n_samples)
        # Pulse from index 200 to 800 (600 samples wide)
        signal[200:800] = 1.0
        duration = detector._measure_duration(signal)