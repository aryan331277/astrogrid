"""Multi-messenger correlator engine for AstroGrid.

Computes cross-correlation metrics between distinct observational channels
(GW, NU, EM) to identify coincident multi-messenger events.
"""

import numpy as np
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class ChannelType(str, Enum):
    GW = "GW"
    NU = "NU"
    EM = "EM"


class ChannelSignal(BaseModel):
    """Represents a time-series signal from a single observational channel."""
    channel_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    channel_type: ChannelType
    timestamps: List[float] = Field(description="Unix timestamps in seconds")
    amplitudes: List[float] = Field(description="Signal amplitudes at each timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class CorrelationResult(BaseModel):
    """Result of cross-correlation between two channels."""
    channel_a: str
    channel_b: str
    peak_lag: float = Field(description="Time lag (seconds) at peak correlation")
    peak_correlation: float = Field(description="Maximum normalized cross-correlation value")
    significance: float = Field(description="Statistical significance in sigma")
    is_coincident: bool = Field(description="Whether the event is considered coincident")
    confidence: float = Field(description="Confidence score between 0 and 1")


class MultiMessengerCorrelator:
    """Engine for computing multi-messenger event correlations.

    Uses normalized cross-correlation in the time domain to identify
    temporal coincidences between different observational channels.
    """

    def __init__(
        self,
        coincidence_window: float = 10.0,
        significance_threshold: float = 3.0,
        min_correlation: float = 0.3,
    ):
        self.coincidence_window = coincidence_window
        self.significance_threshold = significance_threshold
        self.min_correlation = min_correlation

    def _normalize(self, signal: np.ndarray) -> np.ndarray:
        """Normalize signal to zero mean and unit variance."""
        std = np.std(signal)
        if std < 1e-12:
            return np.zeros_like(signal)
        return (signal - np.mean(signal)) / std

    def _cross_correlate(
        self,
        signal_a: np.ndarray,
        signal_b: np.ndarray,
        dt: float,
    ) -> tuple:
        """Compute normalized cross-correlation between two signals.

        Returns:
            Tuple of (correlation_values, lags) where lags are in seconds.
        """
        n = min(len(signal_a), len(signal_b))
        if n < 2:
            return np.array([0.0]), np.array([0.0])

        a = self._normalize(signal_a[:n])
        b = self._normalize(signal_b[:n])

        # Compute cross-correlation using FFT for efficiency
        n_fft = 1 << (2 * n - 1).bit_length()
        fa = np.fft.rfft(a, n=n_fft)
        fb = np.fft.rfft(b, n=n_fft)

        # Cross-correlation via FFT
        cc = np.fft.irfft(fa * np.conj(fb), n=n_fft)
        cc = np.concatenate([cc[-n + 1:], cc[:n]])

        # Normalize
        norm = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
        if norm > 1e-12:
            cc = cc / norm

        lags = np.arange(-n + 1, n) * dt
        return cc, lags

    def correlate_channels(
        self,
        channel_a: ChannelSignal,
        channel_b: ChannelSignal,
    ) -> CorrelationResult:
        """Compute correlation between two channel signals.

        Args:
            channel_a: First channel signal.
            channel_b: Second channel signal.

        Returns:
            CorrelationResult with peak lag, correlation, and significance.
        """
        ts_a = np.array(channel_a.timestamps)
        ts_b = np.array(channel_b.timestamps)
        amp_a = np.array(channel_a.amplitudes)
        amp_b = np.array(channel_b.amplitudes)

        if len(ts_a) < 2 or len(ts_b) < 2:
            return CorrelationResult(
                channel_a=channel_a.channel_id,
                channel_b=channel_b.channel_id,
                peak_lag=0.0,
                peak_correlation=0.0,
                significance=0.0,
                is_coincident=False,
                confidence=0.0,
            )

        # Estimate time step
        dt_a = np.median(np.diff(ts_a)) if len(ts_a) > 1 else 1.0
        dt_b = np.median(np.diff(ts_b)) if len(ts_b) > 1 else 1.0
        dt = min(abs(dt_a), abs(dt_b))
        if dt < 1e-9:
            dt = 1.0

        # Interpolate both signals onto a common time grid
        t_min = max(ts_a[0], ts_b[0])
        t_max = min(ts_a[-1], ts_b[-1])
        if t_max <= t_min:
            return CorrelationResult(
                channel_a=channel_a.channel_id,
                channel_b=channel_b.channel_id,
                peak_lag=0.0,
                peak_correlation=0.0,
                significance=0.0,
                is_coincident=False,
                confidence=0.0,
            )

        common_times = np.arange(t_min, t_max, dt)
        if len(common_times) < 2:
            return CorrelationResult(
                channel_a=channel_a.channel_id,
                channel_b=channel_b.channel_id,
                peak_lag=0.0,
                peak_correlation=0.0,
                significance=0.0,
                is_coincident=False,
                confidence=0.0,
            )

        interp_a = np.interp(common_times, ts_a, amp_a)
        interp_b = np.interp(common_times, ts_b, amp_b)

        cc, lags = self._cross_correlate(interp_a, interp_b, dt)

        peak_idx = np.argmax(np.abs(cc))
        peak_lag = lags[peak_idx]
        peak_corr = cc[peak_idx]

        # Estimate significance using noise floor
        noise_level = np.std(cc)
        if noise_level > 1e-12:
            significance = abs(peak_corr) / noise_level
        else:
            significance = 0.0

        is_coincident = (
            abs(peak_lag) <= self.coincidence_window
            and abs(peak_corr) >= self.min_correlation
            and significance >= self.significance_threshold
        )

        confidence = min(1.0, abs(peak_corr) * (significance / self.significance_threshold))

        return CorrelationResult(
            channel_a=channel_a.channel_id,
            channel_b=channel_b.channel_id,
            peak_lag=float(peak_lag),
            peak_correlation=float(peak_corr),
            significance=float(significance),
            is_coincident=is_coincident,
            confidence=float(confidence),
        )

    def correlate_all_pairs(
        self,
        channels: List[ChannelSignal],
    ) -> List[CorrelationResult]:
        """Compute pairwise correlations for all channel combinations.

        Args:
            channels: List of channel signals.

        Returns:
            List of CorrelationResult for each unique pair.
        """
        results = []
        for i in range(len(channels)):
            for j in range(i + 1, len(channels)):
                result = self.correlate_channels(channels[i], channels[j])
                results.append(result)
        return results

    def find_coincident_events(
        self,
        channels: List[ChannelSignal],
    ) -> List[CorrelationResult]:
        """Identify all coincident multi-messenger events.

        Args:
            channels: List of channel signals.

        Returns:
            List of CorrelationResult where is_coincident is True.
        """
        all_results = self.correlate_all_pairs(channels)
        return [r for r in all_results if r.is_coincident]