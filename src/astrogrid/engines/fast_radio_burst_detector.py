"""Fast Radio Burst (FRB) detection engine for AstroGrid platform."""

import numpy as np
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
import enum
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


class PolarizationMode(str, enum.Enum):
    RCP = "RCP"
    LCP = "LCP"
    BOTH = "BOTH"


class FrequencyBand(BaseModel):
    """Frequency band specification for FRB detection."""
    min_freq_ghz: float = Field(gt=0.0, description="Minimum frequency in GHz")
    max_freq_ghz: float = Field(gt=0.0, description="Maximum frequency in GHz")

    @field_validator("max_freq_ghz")
    @classmethod
    def validate_max_freq(cls, v: float, info) -> float:
        min_freq = info.data.get("min_freq_ghz", 0.0)
        if v <= min_freq:
            raise ValueError("max_freq_ghz must be greater than min_freq_ghz")
        return v


class FRBDetectionResult(BaseModel):
    """Result of a single FRB detection."""
    time_s: float = Field(description="Time of burst peak in seconds")
    frequency_ghz: float = Field(description="Central frequency in GHz")
    duration_ms: float = Field(description="Burst duration in milliseconds")
    fluence_jy_ms: float = Field(description="Fluence in Jy*ms")
    snr: float = Field(description="Signal-to-noise ratio")
    polarization: str = Field(description="Polarization mode")
    dispersion_measure_pc_cm3: float = Field(description="Dispersion measure in pc/cm^3")
    is_candidate: bool = Field(description="Whether this passes candidate threshold")


class FRBDetectorConfig(BaseModel):
    """Configuration for the FRB detector."""
    frequency_band: FrequencyBand
    polarization_mode: PolarizationMode = PolarizationMode.BOTH
    min_snr: float = Field(default=8.0, gt=0.0, description="Minimum SNR threshold")
    min_duration_ms: float = Field(default=1.0, gt=0.0, description="Minimum burst duration in ms")
    max_duration_ms: float = Field(default=1000.0, gt=0.0, description="Maximum burst duration in ms")
    min_fluence_jy_ms: float = Field(default=1.0, gt=0.0, description="Minimum fluence threshold")
    dm_range_pc_cm3: tuple = Field(default=(0.0, 3000.0), description="DM search range")
    dm_step_pc_cm3: float = Field(default=1.0, gt=0.0, description="DM step size")
    time_resolution_ms: float = Field(default=1.0, gt=0.0, description="Time resolution in ms")
    frequency_resolution_mhz: float = Field(default=1.0, gt=0.0, description="Frequency resolution in MHz")


class FastRadioBurstDetector:
    """
    FRB detection engine operating on time-frequency data cubes D(t, f, p).
    
    Implements matched-filter detection with dispersion measure search,
    SNR estimation, and candidate classification.
    """

    def __init__(self, config: FRBDetectorConfig):
        self.config = config
        self._noise_sigma_cache: Optional[float] = None

    def _estimate_noise_sigma(self, data: np.ndarray) -> float:
        """Estimate noise sigma from the data using robust statistics."""
        if self._noise_sigma_cache is not None:
            return self._noise_sigma_cache
        # Use median absolute deviation for robust noise estimation
        median_val = np.median(data)
        mad = np.median(np.abs(data - median_val))
        sigma = 1.4826 * mad  # Convert MAD to sigma for Gaussian
        self._noise_sigma_cache = float(sigma)
        return self._noise_sigma_cache

    def _compute_snr(self, signal: np.ndarray, noise_sigma: float) -> float:
        """Compute SNR for a signal array."""
        if noise_sigma <= 0:
            return 0.0
        peak = np.max(np.abs(signal))
        return float(peak / noise_sigma)

    def _measure_duration(self, signal: np.ndarray, time_resolution_ms: float,
                          threshold_fraction: float = 0.5) -> float:
        """
        Measure burst duration as FWHM of the signal envelope.
        
        Args:
            signal: 1D signal array
            time_resolution_ms: Time resolution in ms
            threshold_fraction: Fraction of peak for FWHM calculation
            
        Returns:
            Duration in milliseconds
        """
        peak = np.max(np.abs(signal))
        if peak <= 0:
            return 0.0
        threshold = peak * threshold_fraction
        above_threshold = np.abs(signal) >= threshold
        if not np.any(above_threshold):
            return 0.0
        indices = np.where(above_threshold)[0]
        duration_samples = indices[-1] - indices[0] + 1
        return float(duration_samples * time_resolution_ms)

    def _compute_fluence(self, signal: np.ndarray, time_resolution_ms: float) -> float:
        """
        Compute fluence as integral of |signal| over time.
        
        Args:
            signal: 1D signal array
            time_resolution_ms: Time resolution in ms
            
        Returns:
            Fluence in Jy*ms
        """
        return float(np.sum(np.abs(signal)) * time_resolution_ms)

    def _search_dm(self, data: np.ndarray, freqs_ghz: np.ndarray,
                   time_resolution_ms: float) -> tuple:
        """
        Search over dispersion measure values to find best alignment.
        
        For a given DM, the arrival time delay between frequencies is:
            delta_t = 4.15e3 * DM * (f_low^-2 - f_high^-2)  [seconds]
        
        Args:
            data: 2D array (time, frequency)
            freqs_ghz: Frequency array in GHz
            time_resolution_ms: Time resolution in ms
            
        Returns:
            Tuple of (best_dm, best_snr, best_time_index)
        """
        best_dm = 0.0
        best_snr = 0.0
        best_time_idx = 0

        dm_min, dm_max = self.config.dm_range_pc_cm3
        dm_step = self.config.dm_step_pc_cm3
        dm_values = np.arange(dm_min, dm_max + dm_step, dm_step)

        f_min = freqs_ghz[0]
        f_max = freqs_ghz[-1]

        for dm in dm_values:
            # Compute time delay for each frequency relative to f_max
            # delta_t(f) = 4.15e3 * DM * (f^-2 - f_max^-2) in seconds
            delays_s = 4.15e3 * dm * (freqs_ghz ** -2 - f_max ** -2)
            delays_ms = delays_s * 1000.0

            # For each frequency channel, shift the time series by the delay
            # and sum coherently
            aligned_sum = np.zeros_like(data[:, 0])
            for i, delay_ms in enumerate(delays_ms):
                shift_samples = int(round(delay_ms / time_resolution_ms))
                if shift_samples >= 0:
                    aligned_sum[shift_samples:] += data[:len(data) - shift_samples, i]
                else:
                    aligned_sum[:shift_samples] += data[-shift_samples:, i]

            snr = self._compute_snr(aligned_sum, self._estimate_noise_sigma(data))
            if snr > best_snr:
                best_snr = snr
                best_dm = float(dm)
                best_time_idx = int(np.argmax(aligned_sum))

        return best_dm, best_snr, best_time_idx

    def detect(self, data_cube: np.ndarray, freqs_ghz: np.ndarray,
               time_axis_s: np.ndarray) -> List[FRBDetectionResult]:
        """
        Run FRB detection on a time-frequency-polarization data cube.
        
        Args:
            data_cube: 3D array with shape (n_time, n_freq, n_pol)
            freqs_ghz: Frequency array in GHz
            time_axis_s: Time axis in seconds
            
        Returns:
            List of FRBDetectionResult objects for detected candidates
        """
        results: List[FRBDetectionResult] = []
        n_time, n_freq, n_pol = data_cube.shape
        time_resolution_ms = self.config.time_resolution_ms

        # Determine polarization channels to process
        if self.config.polarization_mode == PolarizationMode.RCP:
            pol_indices = [0]
        elif self.config.polarization_mode == PolarizationMode.LCP:
            pol_indices = [1] if n_pol > 1 else [0]
        else:  # BOTH
            pol_indices = list(range(n_pol))

        noise_sigma = self._estimate_noise_sigma(data_cube)

        for pol_idx in pol_indices:
            data_2d = data_cube[:, :, pol_idx]

            # Search over DM values
            best_dm, best_snr, best_time_idx = self._search_dm(
                data_2d, freqs_ghz, time_resolution_ms
            )

            if best_snr < self.config.min_snr:
                continue

            # Extract the aligned signal at the best DM for detailed analysis
            f_min = freqs_ghz[0]
            f_max = freqs_ghz[-1]
            delays_s = 4.15e3