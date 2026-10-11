"""
Pydantic data schemas, telemetry records, and coordinate models for the
Box-Less Least Squares (BLS) transit detector module.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
import numpy as np


class BLSInputData(BaseModel):
    """
    Input schema for the Box-Less Least Squares (BLS) transit detector.

    Attributes:
        time: Array of observation times (e.g., BJD).
        flux: Array of observed flux values.
        flux_err: Array of flux uncertainties.
        period_range: Tuple (min_period, max_period) in days.
        period_step: Step size for period grid in days.
        transit_duration: Expected transit duration in days.
        n_transits: Number of transits to consider.
    """

    time: List[float] = Field(..., description="Observation times in BJD")
    flux: List[float] = Field(..., description="Observed flux values")
    flux_err: List[float] = Field(..., description="Flux uncertainties")
    period_range: List[float] = Field(
        default=[0.5, 10.0],
        description="Period search range [min, max] in days"
    )
    period_step: float = Field(
        default=1e-5,
        description="Period grid step size in days"
    )
    transit_duration: float = Field(
        default=0.05,
        description="Expected transit duration in days"
    )
    n_transits: int = Field(
        default=3,
        description="Number of transits to consider"
    )

    @field_validator("time", "flux", "flux_err")
    @classmethod
    def validate_arrays(cls, v: List[float]) -> List[float]:
        if len(v) == 0:
            raise ValueError("Arrays must not be empty")
        return v

    @field_validator("period_range")
    @classmethod
    def validate_period_range(cls, v: List[float]) -> List[float]:
        if len(v) != 2:
            raise ValueError("period_range must have exactly 2 elements")
        if v[0] >= v[1]:
            raise ValueError("period_range[0] must be less than period_range[1]")
        if v[0] <= 0:
            raise ValueError("period_range[0] must be positive")
        return v

    @field_validator("period_step")
    @classmethod
    def validate_period_step(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("period_step must be positive")
        return v

    @field_validator("transit_duration")
    @classmethod
    def validate_transit_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("transit_duration must be positive")
        return v

    @field_validator("n_transits")
    @classmethod
    def validate_n_transits(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_transits must be at least 1")
        return v


class BLSCandidate(BaseModel):
    """
    A single candidate transit signal detected by BLS.

    Attributes:
        period: Best-fit period in days.
        epoch: Transit epoch (mid-transit time) in BJD.
        depth: Transit depth (fractional flux decrease).
        duration: Transit duration in days.
        likelihood: Log-likelihood value for this candidate.
        snr: Signal-to-noise ratio of the detection.
        n_transits: Number of transits in the signal.
    """

    period: float = Field(..., description="Best-fit period in days")
    epoch: float = Field(..., description="Transit epoch in BJD")
    depth: float = Field(..., description="Transit depth (fractional)")
    duration: float = Field(..., description="Transit duration in days")
    likelihood: float = Field(..., description="Log-likelihood value")
    snr: float = Field(..., description="Signal-to-noise ratio")
    n_transits: int = Field(..., description="Number of transits")

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("period must be positive")
        return v

    @field_validator("depth")
    @classmethod
    def validate_depth(cls, v: float) -> float:
        if v < 0 or v >= 1:
            raise ValueError("depth must be in [0, 1)")
        return v

    @field_validator("duration")
    @classmethod
    def validate_duration(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("duration must be positive")
        return v


class BLSResult(BaseModel):
    """
    Complete result from a BLS transit search.

    Attributes:
        candidates: List of detected transit candidates, sorted by SNR.
        n_periods_tested: Number of periods evaluated.
        n_epochs_tested: Number of epochs evaluated.
        computation_time: Wall-clock time in seconds.
        metadata: Additional metadata dictionary.
    """

    candidates: List[BLSCandidate] = Field(
        default_factory=list,
        description="Detected transit candidates sorted by SNR"
    )
    n_periods_tested: int = Field(
        default=0,
        description="Number of periods evaluated"
    )
    n_epochs_tested: int = Field(
        default=0,
        description="Number of epochs evaluated"
    )
    computation_time: float = Field(
        default=0.0,
        description="Computation time in seconds"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class BLSStatus(BaseModel):
    """
    Telemetry status record for the BLS detector.

    Attributes:
        status: Current status string.
        progress: Progress percentage (0-100).
        message: Human-readable status message.
        error: Optional error message if status is 'error'.
    """

    status: str = Field(
        default="idle",
        description="Status: idle, running, completed, error"
    )
    progress: float = Field(
        default=0.0,
        description="Progress percentage (0-100)"
    )
    message: str = Field(
        default="",
        description="Human-readable status message"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if status is 'error'"
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid_statuses = {"idle", "running", "completed", "error"}
        if v not in valid_statuses:
            raise ValueError(f"status must be one of {valid_statuses}")
        return v

    @field_validator("progress")
    @classmethod
    def validate_progress(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("progress must be between 0 and 100")
        return v


class BLSConfiguration(BaseModel):
    """
    Configuration parameters for the BLS detector.

    Attributes:
        max_periods: Maximum number of periods to evaluate.
        max_epochs: Maximum number of epochs to evaluate.
        min_snr: Minimum SNR threshold for candidate detection.
        max_candidates: Maximum number of candidates to return.
        use_box_model: Whether to use box model or BLS.
        box_width: Box width for box model in days.
    """

    max_periods: int = Field(
        default=10000,
        description="Maximum number of periods to evaluate"
    )
    max_epochs: int = Field(
        default=1000,
        description="Maximum number of epochs to evaluate"
    )
    min_snr: float = Field(
        default=5.0,
        description="Minimum SNR threshold for detection"
    )
    max_candidates: int = Field(
        default=10,
        description="Maximum number of candidates to return"
    )
    use_box_model: bool = Field(
        default=False,
        description="Use box model instead of BLS"
    )
    box_width: float = Field(
        default=0.05,
        description="Box width for box model in days"
    )

    @field_validator("max_periods", "max_epochs", "max_candidates")
    @classmethod
    def validate_positive_int(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Value must be at least 1")
        return v

    @field_validator("min_snr")
    @classmethod
    def validate_min_snr(cls, v: float) -> float:
        if v < 0:
            raise ValueError("min_snr must be non-negative")
        return v

    @field_validator("box_width")
    @classmethod
    def validate_box_width(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("box_width must be positive")
        return v